import Foundation
import Metal
import Testing
@testable import MLXEngine

/// Phase 0 of docs/sieve-architecture.md, step 1: what it costs to read a small result back between
/// two command buffers, which is what a Pyramid exit test does (commit, wait, read, encode, commit).
/// The work is a model's own MLP matvecs at T = 1 (most of a decode step), split in the middle of
/// the stack. Opt-in; it measures, it checks nothing, and nothing else should use the GPU:
///
///     MLXR_SIEVE_TIMING=mlx-community/Llama-3.2-1B-Instruct-4bit \
///         swift test -c release -Xswiftc -enable-testing --filter SieveTiming
@Suite("Sieve phase 0 timing", .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXR_SIEVE_TIMING"] != nil))
struct SieveTimingTests {
    @Test func readbackBetweenCommandBuffers() throws {
        guard let repo = ProcessInfo.processInfo.environment["MLXR_SIEVE_TIMING"], let dir = installedModel(repo) else {
            print("SIEVE TIMING: model not installed"); return
        }
        let t = try Transformer(directory: dir, contextLength: 1024, sparse: nil)
        let q = GPU.shared.queue
        let H = t.config.hiddenSize
        // Real inputs in the scratch buffers, so the matvecs are not multiplying zeros.
        _ = try t.forward((0..<32).map { Int32(1000 + $0 * 37) }, offset: 0)
        let L = t.layers.count, half = L / 2

        func encode(_ layers: Range<Int>) -> MTLCommandBuffer {
            let cb = q.makeCommandBufferWithUnretainedReferences()!
            let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            for li in layers {
                let l = t.layers[li], I = l.intermediate
                if l.gateUp.count > 1, Linear.fusable(l.gateUp), Linear.canFuse(l.gateUp, T: 1) {
                    Linear.encodeFusedDecode(enc, l.gateUp, x: t.xn, xStride: H, y: t.guBuf, yStride: 2 * I, T: 1)
                } else {
                    var col = 0
                    for p in l.gateUp { p.encode(enc, x: t.xn, xStride: H, y: t.guBuf, yOffset: col * 4, yStride: 2 * I, T: 1); col += p.outDim }
                }
                l.down.encode(enc, x: t.mlpBuf, xStride: I, y: t.tmp, yStride: H, T: 1)
            }
            enc.endEncoding()
            return cb
        }
        func gpu(_ cb: MTLCommandBuffer) -> Double { cb.gpuEndTime - cb.gpuStartTime }
        var sink: Float = 0
        func read() { sink += t.tmp.contents().load(as: Float.self) }

        // Each variant returns (wall seconds, GPU seconds).
        let variants: [(String, () -> (Double, Double))] = [
            ("one command buffer", {
                let s = DispatchTime.now().uptimeNanoseconds
                let cb = encode(0..<L); cb.commit(); cb.waitUntilCompleted(); read()
                return (Double(DispatchTime.now().uptimeNanoseconds - s) / 1e9, gpu(cb))
            }),
            ("two, both committed at once", {
                let s = DispatchTime.now().uptimeNanoseconds
                let a = encode(0..<half), b = encode(half..<L)
                a.commit(); b.commit(); b.waitUntilCompleted(); read()
                return (Double(DispatchTime.now().uptimeNanoseconds - s) / 1e9, gpu(a) + gpu(b))
            }),
            ("two, readback, then encode", {
                let s = DispatchTime.now().uptimeNanoseconds
                let a = encode(0..<half); a.commit(); a.waitUntilCompleted(); read()
                let b = encode(half..<L); b.commit(); b.waitUntilCompleted(); read()
                return (Double(DispatchTime.now().uptimeNanoseconds - s) / 1e9, gpu(a) + gpu(b))
            }),
            ("two, readback, second pre-encoded", {
                let s = DispatchTime.now().uptimeNanoseconds
                let a = encode(0..<half), b = encode(half..<L)
                a.commit(); a.waitUntilCompleted(); read()
                b.commit(); b.waitUntilCompleted(); read()
                return (Double(DispatchTime.now().uptimeNanoseconds - s) / 1e9, gpu(a) + gpu(b))
            }),
        ]
        let rounds = 300, warm = 30
        var wall = [[Double]](repeating: [], count: variants.count), gpuT = wall
        for r in 0..<(rounds + warm) {
            for (i, v) in variants.enumerated() {
                let (w, g) = v.1()
                if r >= warm { wall[i].append(w); gpuT[i].append(g) }
            }
        }
        func med(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
        func p10(_ a: [Double]) -> Double { a.sorted()[a.count / 10] }
        print("SIEVE TIMING \(repo): MLP matvecs of \(L) layers at T = 1, split after layer \(half); \(rounds) alternating rounds")
        for (i, v) in variants.enumerated() {
            print(String(format: "SIEVE TIMING %-36@ wall median %.3f ms (p10 %.3f), GPU median %.3f ms; against one buffer: wall %+.3f ms, GPU %+.3f ms",
                         v.0 as NSString, med(wall[i]) * 1e3, p10(wall[i]) * 1e3, med(gpuT[i]) * 1e3,
                         (med(wall[i]) - med(wall[0])) * 1e3, (med(gpuT[i]) - med(gpuT[0])) * 1e3))
        }
        _ = sink
    }

    /// Stride's cost side: a pass over two tokens against a single-token step, without the head
    /// (`profileSkip`), in consecutive positions as decode runs them. Alternating blocks.
    @Test func twoTokenPassAgainstOneTokenStep() throws {
        guard let repo = ProcessInfo.processInfo.environment["MLXR_SIEVE_TIMING"], let dir = installedModel(repo) else {
            print("SIEVE TIMING: model not installed"); return
        }
        let t = try Transformer(directory: dir, contextLength: 4096, sparse: nil)
        t.profileSkip = ["lmhead"]
        defer { t.profileSkip = [] }
        var pos = 0
        _ = try t.forward((0..<16).map { Int32(1000 + $0 * 37) }, offset: 0); pos = 16
        func block(_ n: Int, tokens: Int) throws -> Double {
            let s = DispatchTime.now().uptimeNanoseconds
            for i in 0..<n {
                _ = try t.forward((0..<tokens).map { Int32(2000 + (pos + i + $0) % 5000) }, offset: pos)
                pos += tokens
            }
            return Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6 / Double(n)
        }
        _ = try block(30, tokens: 1); _ = try block(30, tokens: 2)
        var one: [Double] = [], two: [Double] = []
        for _ in 0..<6 { one.append(try block(60, tokens: 1)); two.append(try block(60, tokens: 2)) }
        one.sort(); two.sort()
        print(String(format: "SIEVE TIMING %@: body only, positions 16 to %d: one token %.3f ms, two tokens %.3f ms per pass (median of 6 blocks of 60): ratio %.2f",
                     repo as NSString, pos, one[3], two[3], two[3] / one[3]))
    }
}
