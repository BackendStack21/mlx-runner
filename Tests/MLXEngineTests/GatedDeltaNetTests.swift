import Foundation
import Metal
import Testing
@testable import MLXEngine

/// The Qwen 3.5 Gated DeltaNet kernels on random inputs: the fused decode kernel (`gdn_step`) token by
/// token and the batch path (`gdn_conv` → `gdn_prep` → `gdn_scan` → `gdn_norm`), split into batches
/// at different offsets, must give bit-identical outputs and state, and match a float64 reference of
/// the recurrence. Key/value head ratios 1, 2 and 4 (Qwen 3.5 0.8B/2B use 1, 4B/9B use 2).
@Suite("Gated DeltaNet kernels", .serialized)
struct GatedDeltaNetTests {
    static let D = 128

    struct Shape { let Hk: Int, Hv: Int, T: Int
        var keyDim: Int { Hk * GatedDeltaNetTests.D }
        var valueDim: Int { Hv * GatedDeltaNetTests.D }
        var convDim: Int { 2 * keyDim + valueDim }
        var cols: Int { (convDim + valueDim + 2 * Hv + 3) / 4 * 4 }
    }

    struct Weights { let convW: [Float]; let negA: [Float]; let dtBias: [Float]; let normW: [Float]; let eps: Float = 1e-6 }

    struct RNG { var s: UInt64
        mutating func next() -> Float {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            return Float(s >> 40) / Float(1 << 24) * 2 - 1
        }
    }

    func buffer(_ v: [Float]) -> MTLBuffer {
        let b = try! GPU.shared.buffer(max(16, v.count * 4))
        v.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return b
    }
    func floats(_ b: MTLBuffer, _ n: Int) -> [Float] { Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: n), count: n)) }

    /// float64 reference: outputs [T][Hv · D] from zero state.
    func reference(_ sh: Shape, _ proj: [Float], _ w: Weights) -> [Double] {
        let D = Self.D, ratio = sh.Hv / sh.Hk
        var S = [Double](repeating: 0, count: sh.Hv * D * D)
        var out: [Double] = []
        func x(_ t: Int, _ c: Int) -> Double { t < 0 ? 0 : Double(proj[t * sh.cols + c]) }
        for t in 0..<sh.T {
            var conv = [Double](repeating: 0, count: sh.convDim)
            for c in 0..<sh.convDim {
                let s = Double(w.convW[4 * c]) * x(t - 3, c) + Double(w.convW[4 * c + 1]) * x(t - 2, c)
                    + Double(w.convW[4 * c + 2]) * x(t - 1, c) + Double(w.convW[4 * c + 3]) * x(t, c)
                conv[c] = s / (1 + exp(-s))
            }
            func normed(_ off: Int, _ scale: Double) -> [Double] {
                let v = Array(conv[off..<off + D])
                let inv = 1 / (v.reduce(0) { $0 + $1 * $1 } / Double(D) + 1e-6 / Double(D)).squareRoot()
                return v.map { $0 * inv * scale }
            }
            let base = 2 * sh.keyDim + 2 * sh.valueDim
            for h in 0..<sh.Hv {
                let hk = h / ratio
                let q = normed(hk * D, 1 / Double(D)), k = normed(sh.keyDim + hk * D, 1 / Double(D).squareRoot())
                let v = Array(conv[(2 * sh.keyDim + h * D)..<(2 * sh.keyDim + (h + 1) * D)])
                let a = x(t, base + sh.Hv + h) + Double(w.dtBias[h]), b = x(t, base + h)
                let g = exp(Double(w.negA[h]) * (max(a, 0) + log(1 + exp(-abs(a))))), beta = 1 / (1 + exp(-b))
                var y = [Double](repeating: 0, count: D)
                for r in 0..<D {
                    let row = (h * D + r) * D
                    var kv = 0.0
                    for c in 0..<D { S[row + c] *= g; kv += S[row + c] * k[c] }
                    let delta = (v[r] - kv) * beta
                    for c in 0..<D { S[row + c] += k[c] * delta; y[r] += S[row + c] * q[c] }
                }
                let inv = 1 / (y.reduce(0) { $0 + $1 * $1 } / Double(D) + Double(w.eps)).squareRoot()
                for r in 0..<D {
                    let z = x(t, 2 * sh.keyDim + sh.valueDim + h * D + r)
                    out.append(y[r] * inv * Double(w.normW[r]) * (z / (1 + exp(-z))))
                }
            }
        }
        return out
    }

    /// Runs the kernels: `batches` = token counts per pass (1 = the decode kernel). Returns outputs and state.
    func run(_ sh: Shape, _ proj: [Float], _ w: Weights, batches: [Int], lpr: Int) throws -> (out: [Float], state: [Float]) {
        let gpu = GPU.shared, D = Self.D
        let step = try gpu.pipeline("gdn_step_\(lpr)"), conv = try gpu.pipeline("gdn_conv"), prep = try gpu.pipeline("gdn_prep")
        let scan = try gpu.pipeline("gdn_scan_\(lpr)"), norm = try gpu.pipeline("gdn_norm")
        let convW = buffer(w.convW), negA = buffer(w.negA), dtBias = buffer(w.dtBias), normW = buffer(w.normW)
        let ring = buffer([Float](repeating: .nan, count: 4 * sh.convDim))      // never read before written
        let state = buffer([Float](repeating: .nan, count: sh.Hv * D * D))     // a pass at offset 0 ignores it
        var out: [Float] = []
        var offset = 0
        for n in batches {
            let pb = buffer(Array(proj[(offset * sh.cols)..<((offset + n) * sh.cols)]))
            let ob = buffer([Float](repeating: 0, count: n * sh.valueDim))
            let p = GDNParams(T: Int32(n), offset: Int32(offset), cols: Int32(sh.cols), Hk: Int32(sh.Hk), Hv: Int32(sh.Hv),
                              convDim: Int32(sh.convDim), outStride: Int32(sh.valueDim), eps: w.eps)
            let cb = gpu.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            if n == 1 {
                enc.setComputePipelineState(step)
                for (i, b) in [pb, convW, ring, state, negA, dtBias, normW, ob].enumerated() { enc.set(b, 0, i) }
                enc.params(p, 8)
                enc.dispatchThreadgroups(MTLSize(width: sh.Hv, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: D * lpr, height: 1, depth: 1))
            } else {
                enc.setComputePipelineState(conv)
                enc.set(pb, 0, 0); enc.set(convW, 0, 1); enc.set(ring, 0, 2); enc.params(p, 3)
                enc.dispatchThreads(MTLSize(width: sh.convDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc.setComputePipelineState(prep)
                enc.set(pb, 0, 0); enc.set(negA, 0, 1); enc.set(dtBias, 0, 2); enc.params(p, 3)
                enc.dispatchThreadgroups(MTLSize(width: sh.Hk, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 2 * D, height: 1, depth: 1))
                enc.setComputePipelineState(scan)
                enc.set(pb, 0, 0); enc.set(state, 0, 1); enc.params(p, 2)
                enc.dispatchThreadgroups(MTLSize(width: D / (64 / lpr), height: sh.Hv, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc.setComputePipelineState(norm)
                enc.set(pb, 0, 0); enc.set(normW, 0, 1); enc.set(ob, 0, 2); enc.params(p, 3)
                enc.dispatchThreadgroups(MTLSize(width: sh.Hv, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            #expect(cb.error == nil)
            out += floats(ob, n * sh.valueDim)
            offset += n
        }
        return (out, floats(state, sh.Hv * D * D))
    }

    @Test(arguments: [(2, 4), (1, 4), (2, 2)])
    func decodeBatchAndReferenceAgree(heads: (Int, Int)) throws {
        let sh = Shape(Hk: heads.0, Hv: heads.1, T: 9)
        var rng = RNG(s: 0x9E3779B97F4A7C15 &+ UInt64(sh.Hv * 7 + sh.Hk))
        let proj = (0..<(sh.T * sh.cols)).map { _ in rng.next() * 2 }
        let w = Weights(convW: (0..<(4 * sh.convDim)).map { _ in rng.next() * 0.7 },
                        negA: (0..<sh.Hv).map { _ in -exp(rng.next() * 2) }, dtBias: (0..<sh.Hv).map { _ in rng.next() },
                        normW: (0..<Self.D).map { _ in 1 + 0.2 * rng.next() })
        let ref = reference(sh, proj, w)
        for lpr in [4, 8] {
            let decode = try run(sh, proj, w, batches: Array(repeating: 1, count: sh.T), lpr: lpr)
            // Batches that start mid-ring and end on a single decode step after a batch.
            for split in [[sh.T], [5, 4], [2, 6, 1], [3, 1, 5]] {
                let batch = try run(sh, proj, w, batches: split, lpr: lpr)
                #expect(batch.out == decode.out, "outputs: batches \(split) vs decode, LPR \(lpr), heads \(heads)")
                #expect(batch.state == decode.state, "state: batches \(split) vs decode, LPR \(lpr), heads \(heads)")
            }
            let err = zip(decode.out, ref).map { abs(Double($0) - $1) / max(1, abs($1)) }.max()!
            #expect(err < 1e-4, "decode vs float64 reference: \(err) (LPR \(lpr), heads \(heads))")
        }
    }
}
