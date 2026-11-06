import Foundation
import Metal
import Testing
@testable import MLXEngine

/// GPU time of the sparse decode kernels on their own, at the shapes of Qwen3 0.6B (8 KV heads,
/// head size 128, 16 query heads) at 8K and 30K context. Opt-in (MLXR_KERNEL_TIMING=1): it measures,
/// it checks nothing, and nothing else should use the GPU while it runs.
@Suite("Sparse kernel timing", .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXR_KERNEL_TIMING"] != nil))
struct SparseTimingTests {
    /// GPU seconds per repetition of `body` (`reps` repetitions in one serial encoder, best of 3).
    func time(reps: Int, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
        var best = Double.infinity
        for _ in 0..<3 {
            let cb = GPU.shared.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            for _ in 0..<reps { try body(enc) }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(reps))
        }
        return best
    }

    @Test(arguments: [8192, 30000])
    func scoreAndSelect(context: Int) throws {
        let nKV = 8, D = 128, nH = 16, pool = SparseAttention.pool
        guard let plan = SparseAttention().plan(position: context - 1) else { return }
        var sp = plan.params
        sp.nKV = Int32(nKV); sp.D = Int32(D); sp.maxSeq = Int32(context); sp.nH = Int32(nH)
        // MLXR_SPARSE_SHARED=1 times the shared selection, otherwise one per KV head.
        let rows = ProcessInfo.processInfo.environment["MLXR_SPARSE_SHARED"] == "1" ? 1 : nKV
        sp.rows = Int32(rows); sp.listStride = rows == 1 ? 0 : Int32(context)
        let nb = Int(sp.nb)
        var rng = RNG(s: 5)
        // Index rows and a query with a common direction, so block scores share magnitudes (and top
        // key bits) as real ones do.
        let idx = makeBuffer((0..<((context / pool + 1) * nKV * D)).map { i in Float16(0.5 + 0.3 * rng.float() + (i % 7 == 0 ? 0.2 : 0)) })
        let qkv = makeBuffer((0..<(nH * D + 2 * nKV * D)).map { _ in 0.5 + 0.2 * rng.float() })
        let scores = makeBuffer([Float](repeating: 0, count: rows * nb + 64))
        let list = makeBuffer([Int32](repeating: 0, count: rows * context + 64))
        let scorePSO = try GPU.shared.pipeline("sparse_score"), selectPSO = try GPU.shared.pipeline("sparse_select")
        let threads = min(1024, selectPSO.maxTotalThreadsPerThreadgroup) / 32 * 32
        func score(_ enc: MTLComputeCommandEncoder) {
            enc.setComputePipelineState(scorePSO)
            enc.set(qkv, 0, 0); enc.set(idx, 0, 1); enc.set(scores, 0, 2); enc.params(sp, 3)
            enc.dispatchThreadgroups(MTLSize(width: (nb + Transformer.sparseScoreBlocks - 1) / Transformer.sparseScoreBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Transformer.sparseScoreThreads, height: 1, depth: 1))
        }
        func select(_ e: MTLComputeCommandEncoder) {
            e.setComputePipelineState(selectPSO)
            e.set(scores, 0, 0); e.set(list, 0, 1); e.params(sp, 2)
            e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
        let tScore = try time(reps: 200) { score($0) }
        let tSelect = try time(reps: 200) { select($0) }
        let tBoth = try time(reps: 200) { score($0); select($0) }
        let s = readFloats(scores, nb)
        let topBytes = Set(s.map { SparseAttentionTests.key($0) >> 24 }).count
        print(String(format: "TIMING ctx %d nb %d (distinct top bytes %d): score %.1f us, select %.1f us, score+select %.1f us",
                     context, nb, topBytes, tScore * 1e6, tSelect * 1e6, tBoth * 1e6))
    }

    /// One sparse layer step as decode runs it (score, select, attention_sparse, combine), and
    /// dense split attention, over `layers` distinct caches and indexes so nothing stays cached
    /// between layers: GPU µs per layer.
    @Test(arguments: [8192, 30000])
    func layerStep(context: Int) throws {
        let nKV = 8, D = 128, nH = 16, pool = SparseAttention.pool, layers = 4, V = Transformer.attnSplitVirtualSG
        guard let plan = SparseAttention().plan(position: context - 1) else { return }
        var sp = plan.params
        sp.nKV = Int32(nKV); sp.D = Int32(D); sp.maxSeq = Int32(context); sp.nH = Int32(nH)
        // MLXR_SPARSE_SHARED=1 times the shared selection, otherwise one per KV head.
        let rows = ProcessInfo.processInfo.environment["MLXR_SPARSE_SHARED"] == "1" ? 1 : nKV
        sp.rows = Int32(rows); sp.listStride = rows == 1 ? 0 : Int32(context)
        let nb = Int(sp.nb)
        let gpu = GPU.shared
        var rng = RNG(s: 9)
        let rowHalves = [Float16]((0..<(nKV * context * D)).map { i in Float16(0.5 + 0.3 * rng.float() + (i % 7 == 0 ? 0.2 : 0)) })
        let kc = (0..<layers).map { _ in makeBuffer(rowHalves) }, vc = (0..<layers).map { _ in makeBuffer(rowHalves) }
        let idx = (0..<layers).map { _ in makeBuffer([Float16](rowHalves.prefix((context / pool + 1) * nKV * D))) }
        let qkv = makeBuffer((0..<(nH * D + 2 * nKV * D)).map { _ in 0.5 + 0.2 * rng.float() })
        let scores = makeBuffer([Float](repeating: 0, count: rows * nb + 64))
        let list = makeBuffer([Int32](repeating: 0, count: rows * context + 64))
        let part = makeBuffer([Float](repeating: 0, count: nH * V * (D + 2)))
        let out = makeBuffer([Float](repeating: 0, count: nH * D))
        let scorePSO = try gpu.pipeline("sparse_score"), selectPSO = try gpu.pipeline("sparse_select")
        let sparsePSO = try gpu.pipeline("attention_sparse_4_2"), splitPSO = try gpu.pipeline("attention_split_4_2")
        let combinePSO = try gpu.pipeline("attention_combine_4")
        let threads = min(1024, selectPSO.maxTotalThreadsPerThreadgroup) / 32 * 32
        let ap = AttnParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), qStride: Int32(nH * D + 2 * nKV * D), maxSeq: Int32(context),
                            offset: Int32(context - 1), scale: 0.088, outStride: Int32(nH * D))
        let attnGrid = MTLSize(width: nKV, height: V / Transformer.attnSplitSG, depth: 1)
        let attnThreads = MTLSize(width: Transformer.attnSplitSG * 32, height: 1, depth: 1)
        func combine(_ enc: MTLComputeCommandEncoder) {
            enc.setComputePipelineState(combinePSO)
            enc.set(part, 0, 0); enc.set(out, 0, 1); enc.params(ap, 2)
            enc.dispatchThreadgroups(MTLSize(width: nH, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        func sparseStep(_ enc: MTLComputeCommandEncoder, _ l: Int, score: Bool, select: Bool, count: Int) {
            if score {
                enc.setComputePipelineState(scorePSO)
                enc.set(qkv, 0, 0); enc.set(idx[l], 0, 1); enc.set(scores, 0, 2); enc.params(sp, 3)
                enc.dispatchThreadgroups(MTLSize(width: (nb + Transformer.sparseScoreBlocks - 1) / Transformer.sparseScoreBlocks, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Transformer.sparseScoreThreads, height: 1, depth: 1))
            }
            if select {
                enc.setComputePipelineState(selectPSO)
                enc.set(scores, 0, 0); enc.set(list, 0, 1); enc.params(sp, 2)
                enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            }
            enc.setComputePipelineState(sparsePSO)
            enc.set(qkv, 0, 0); enc.set(kc[l], 0, 1); enc.set(vc[l], 0, 2); enc.set(part, 0, 3); enc.params(ap, 4)
            enc.set(list, 0, 5); enc.params(SIMD2<Int32>(Int32(count), sp.listStride), 6)
            enc.dispatchThreadgroups(attnGrid, threadsPerThreadgroup: attnThreads)
            combine(enc)
        }
        var l = 0
        let reps = 40
        // Fill the list once with a real selection.
        runGPU { sparseStep($0, 0, score: true, select: true, count: plan.count) }
        let full = try time(reps: reps) { enc in sparseStep(enc, l, score: true, select: true, count: plan.count); l = (l + 1) % layers }
        let noSel = try time(reps: reps) { enc in sparseStep(enc, l, score: true, select: false, count: plan.count); l = (l + 1) % layers }
        let attnOnly = try time(reps: reps) { enc in sparseStep(enc, l, score: false, select: false, count: plan.count); l = (l + 1) % layers }
        let attnTiny = try time(reps: reps) { enc in sparseStep(enc, l, score: false, select: false, count: 12); l = (l + 1) % layers }
        let dense = try time(reps: reps) { enc in
            enc.setComputePipelineState(splitPSO)
            enc.set(qkv, 0, 0); enc.set(kc[l], 0, 1); enc.set(vc[l], 0, 2); enc.set(part, 0, 3); enc.params(ap, 4)
            enc.dispatchThreadgroups(attnGrid, threadsPerThreadgroup: attnThreads)
            combine(enc)
            l = (l + 1) % layers
        }
        print(String(format: "TIMING layer ctx %d: sparse full %.1f us (score %.1f, select %.1f, attention+combine %.1f of which 12-key floor %.1f); dense split+combine %.1f us",
                     context, full * 1e6, (noSel - attnOnly) * 1e6, (full - noSel) * 1e6, attnOnly * 1e6, attnTiny * 1e6, dense * 1e6))
    }
}
