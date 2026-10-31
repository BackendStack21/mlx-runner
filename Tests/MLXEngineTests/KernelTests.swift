import Foundation
import Metal
import Testing
@testable import MLXEngine

// GPU kernels are checked against straightforward CPU references.

struct RNG {
    var s: UInt64
    mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
    mutating func float(_ lo: Float = -1, _ hi: Float = 1) -> Float {
        lo + (hi - lo) * Float(next() % 1_000_000) / 1_000_000
    }
}

func makeBuffer<T>(_ a: [T]) -> MTLBuffer {
    a.withUnsafeBytes { GPU.shared.device.makeBuffer(bytes: $0.baseAddress!, length: max(16, $0.count), options: .storageModeShared)! }
}

func readFloats(_ b: MTLBuffer, _ n: Int) -> [Float] {
    Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: n), count: n))
}

func runGPU(_ body: (MTLComputeCommandEncoder) throws -> Void) rethrows {
    let cb = GPU.shared.queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
    try body(enc)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

/// Packs unsigned `bits`-wide values little-endian into a byte stream (MLX layout).
func pack(_ q: [Int], bits: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: (q.count * bits + 7) / 8)
    for (i, v) in q.enumerated() {
        for b in 0..<bits where (v >> b) & 1 == 1 {
            let bit = i * bits + b
            out[bit / 8] |= UInt8(1 << (bit % 8))
        }
    }
    return out
}

func bf16(_ f: Float) -> UInt16 { UInt16(f.bitPattern >> 16) }
func fromBF16(_ u: UInt16) -> Float { Float(bitPattern: UInt32(u) << 16) }

struct QuantFixture {
    let linear: Linear
    let dense: [[Float]]          // dequantized reference weights [O][K]
    let bias: [Float]?

    init(O: Int, K: Int, bits: Int, gs: Int, scaleType: DType = .f16, withBias: Bool = false, seed: UInt64 = 7) throws {
        var rng = RNG(s: seed)
        let groups = K / gs
        var q = [Int](), scales = [Float](), biases = [Float]()
        for _ in 0..<(O * K) { q.append(Int(rng.next() % UInt64(1 << bits))) }
        for _ in 0..<(O * groups) {
            scales.append(rng.float(0.001, 0.05))
            biases.append(rng.float(-0.2, 0.2))
        }
        // Round scale/bias to the storage precision so the reference sees the same values.
        func store(_ v: [Float]) -> (MTLBuffer, [Float]) {
            switch scaleType {
            case .bf16:
                let u = v.map(bf16)
                return (makeBuffer(u), u.map(fromBF16))
            case .f32:
                return (makeBuffer(v), v)
            default:
                let h = v.map { Float16($0) }
                return (makeBuffer(h), h.map { Float($0) })
            }
        }
        let (sBuf, sRef) = store(scales)
        let (bBuf, bRef) = store(biases)
        var packed: [UInt8] = []
        for r in 0..<O { packed += pack(Array(q[(r * K)..<((r + 1) * K)]), bits: bits) }
        let wT = Tensor(buffer: makeBuffer(packed), offset: 0, dtype: .u32, shape: [O, K * bits / 32])
        let sT = Tensor(buffer: sBuf, offset: 0, dtype: scaleType, shape: [O, groups])
        let bT = Tensor(buffer: bBuf, offset: 0, dtype: scaleType, shape: [O, groups])
        var biasT: Tensor?
        var biasRef: [Float]?
        if withBias {
            let raw = (0..<O).map { _ in rng.float(-0.5, 0.5) }
            let (buf, ref) = store(raw)
            biasT = Tensor(buffer: buf, offset: 0, dtype: scaleType, shape: [O])
            biasRef = ref
        }
        linear = try Linear(weight: wT, scales: sT, biases: bT, bias: biasT, inDim: K, name: "test")
        var d = [[Float]]()
        for r in 0..<O {
            d.append((0..<K).map { k in
                let g = r * groups + k / gs
                return sRef[g] * Float(q[r * K + k]) + bRef[g]
            })
        }
        dense = d
        bias = biasRef
    }

    func reference(_ x: [Float], T: Int, K: Int) -> [Float] {
        let O = dense.count
        var y = [Float](repeating: 0, count: T * O)
        for t in 0..<T {
            for o in 0..<O {
                var acc: Double = 0
                for k in 0..<K { acc += Double(dense[o][k]) * Double(x[t * K + k]) }
                y[t * O + o] = Float(acc) + (bias?[o] ?? 0)
            }
        }
        return y
    }
}

func assertClose(_ a: [Float], _ b: [Float], tol: Float = 2e-3, _ label: String) {
    #expect(a.count == b.count)
    var worst: Float = 0
    for (x, y) in zip(a, b) {
        let err = abs(x - y) / max(1, abs(y))
        worst = max(worst, err)
    }
    #expect(worst < tol, "\(label): max relative error \(worst)")
}

@Suite("Quantized and dense matmul kernels")
struct MatmulKernelTests {
    @Test(arguments: [2, 3, 4, 5, 6, 8])
    func quantizedMatchesReference(bits: Int) throws {
        let K = 512, O = 40
        for gs in [32, 64, 128] {
            for scaleType in [DType.f16, .bf16] {
                let fx = try QuantFixture(O: O, K: K, bits: bits, gs: gs, scaleType: scaleType)
                // decode (qmv), small batch (qmm / qmvb4), short-prompt tiles (qgemms 8 / 16), prefill tiles (qgemm)
                for T in [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 33] {
                    var rng = RNG(s: UInt64(T * 31 + bits))
                    let x = (0..<(T * K)).map { _ in rng.float() }
                    let xb = makeBuffer(x)
                    let yb = try GPU.shared.buffer(T * O * 4)
                    runGPU { fx.linear.encode($0, x: xb, xStride: K, y: yb, yStride: O, T: T) }
                    assertClose(readFloats(yb, T * O), fx.reference(x, T: T, K: K),
                                "bits \(bits) gs \(gs) \(scaleType) T \(T)")
                }
            }
        }
    }

    @Test func fusedDecodeWithBiasMatchesSeparate() throws {
        let K = 256
        let parts = try [(64, 1), (16, 2), (16, 3)].map {
            try QuantFixture(O: $0.0, K: K, bits: 4, gs: 64, withBias: true, seed: UInt64($0.1))
        }
        #expect(Linear.fusable(parts.map(\.linear)))
        var rng = RNG(s: 99)
        let x = (0..<K).map { _ in rng.float() }
        let xb = makeBuffer(x)
        let yb = try GPU.shared.buffer(96 * 4)
        runGPU { Linear.encodeFusedDecode($0, parts.map(\.linear), x: xb, y: yb) }
        let ref = parts.flatMap { $0.reference(x, T: 1, K: K) }
        assertClose(readFloats(yb, 96), ref, "fused q/k/v")
    }

    @Test func denseHalfMatchesReference() throws {
        let K = 128, O = 24
        var rng = RNG(s: 5)
        let w = (0..<(O * K)).map { _ in Float16(rng.float()) }
        let lin = try Linear(weight: Tensor(buffer: makeBuffer(w), offset: 0, dtype: .f16, shape: [O, K]),
                             scales: nil, biases: nil, bias: nil, inDim: K, name: "dense")
        for T in [1, 3, 40] {
            let x = (0..<(T * K)).map { _ in rng.float() }
            let yb = try GPU.shared.buffer(T * O * 4)
            runGPU { lin.encode($0, x: makeBuffer(x), xStride: K, y: yb, yStride: O, T: T) }
            var ref = [Float](repeating: 0, count: T * O)
            for t in 0..<T { for o in 0..<O { for k in 0..<K { ref[t * O + o] += Float(w[o * K + k]) * x[t * K + k] } } }
            assertClose(readFloats(yb, T * O), ref, "dense T \(T)")
        }
    }

    @Test func quantizedEmbeddingLookup() throws {
        let K = 128, V = 16
        let fx = try QuantFixture(O: V, K: K, bits: 4, gs: 64)
        let ids: [Int32] = [3, 0, 15, 7]
        let out = try GPU.shared.buffer(ids.count * K * 4)
        try runGPU { try fx.linear.encodeLookup($0, ids: makeBuffer(ids), out: out, T: ids.count) }
        let got = readFloats(out, ids.count * K)
        let ref = ids.flatMap { fx.dense[Int($0)] }
        assertClose(got, ref, tol: 1e-5, "embedding")
    }
}
