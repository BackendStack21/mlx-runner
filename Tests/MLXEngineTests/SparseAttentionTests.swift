import Foundation
import Metal
import Testing
@testable import MLXEngine

/// Sparse decode attention (`SparseAttention`): the plan, and each kernel against a CPU reference.
/// The end-to-end check is `ModelTests.sparseDecodeWithFullBudgetIsBitIdentical`.
@Suite("Sparse attention")
struct SparseAttentionTests {
    static let pool = SparseAttention.pool

    // MARK: Plan

    @Test func planPartitionsTheContext() {
        let configs = [SparseAttention(), SparseAttention(topK: 64, window: 16, sink: 4, minContext: 1),
                       SparseAttention(topK: 8, window: 5, sink: 1, minContext: 1), SparseAttention(topK: 1 << 20, window: 4, sink: 0, minContext: 1)]
        for c in configs {
            for L in [1, 2, 7, 64, 65, 300, 5127, 5128, 5129, 8191, 32768] {
                guard let plan = c.plan(position: L - 1) else {
                    // Dense: below the minimum context, or nothing between the sink and the window.
                    let sinkEnd = (c.sink + Self.pool - 1) / Self.pool * Self.pool
                    #expect(L < c.minContext || L - c.window <= sinkEnd || (L - c.window) / Self.pool * Self.pool <= sinkEnd, "L=\(L) \(c)")
                    continue
                }
                let p = plan.params
                let (pool, b0, nb, kept, sinkEnd, winStart) = (Int(p.pool), Int(p.b0), Int(p.nb), Int(p.p), Int(p.sinkEnd), Int(p.winStart))
                #expect(L >= c.minContext)
                #expect(Int(p.L) == L && pool == Self.pool)
                #expect(sinkEnd >= c.sink && sinkEnd % pool == 0 && b0 * pool == sinkEnd)
                #expect(winStart <= L - c.window && winStart % pool == 0 && (b0 + nb) * pool == winStart && nb > 0)
                #expect(kept == min(c.topK / pool, nb) && kept >= 1)
                #expect(plan.count == sinkEnd + kept * pool + (L - winStart) && plan.count <= L)
            }
        }
        #expect(SparseAttention(minContext: 1).plan(position: 100_000) != nil)
        #expect(SparseAttention().plan(position: 100) == nil)   // default minimum context: 2 · (4 + 2048 + 512)
    }

    @Test func configIsClampedToWholeBlocks() {
        let c = SparseAttention(topK: 2050, window: 0, sink: -3, minContext: -1)
        #expect(c.topK == 2048 && c.window == Self.pool && c.sink == 0 && c.minContext == 1)
        #expect(SparseAttention(topK: 1).topK == Self.pool)
    }

    // MARK: Kernels

    @Test func indexIsTheMeanOfEachBlock() throws {
        let nKV = 2, D = 128, maxSeq = 64, b0 = 3, nb = 7
        var rng = RNG(s: 11)
        let kh = (0..<(nKV * maxSeq * D)).map { _ in Float16(rng.float()) }
        let sentinel = Float16(7)
        let idx = makeBuffer([Float16](repeating: sentinel, count: maxSeq / Self.pool * nKV * D))
        try runGPU { enc in
            enc.setComputePipelineState(try GPU.shared.pipeline("sparse_index"))
            enc.set(makeBuffer(kh), 0, 0); enc.set(idx, 0, 1)
            enc.params(SparseParams(pool: Int32(Self.pool), nKV: Int32(nKV), D: Int32(D), maxSeq: Int32(maxSeq), b0: Int32(b0), nb: Int32(nb)), 2)
            enc.dispatchThreadgroups(MTLSize(width: nb, height: nKV, depth: 1), threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
        }
        let out = Array(UnsafeBufferPointer(start: idx.contents().bindMemory(to: Float16.self, capacity: maxSeq / Self.pool * nKV * D),
                                            count: maxSeq / Self.pool * nKV * D))
        for b in 0..<(maxSeq / Self.pool) {
            for h in 0..<nKV {
                for d in 0..<D {
                    let got = out[(b * nKV + h) * D + d]
                    if b < b0 || b >= b0 + nb { #expect(got == sentinel, "block \(b) outside the range was written"); continue }
                    var s: Float = 0
                    for i in 0..<Self.pool { s += Float(kh[(h * maxSeq + b * Self.pool + i) * D + d]) }
                    #expect(abs(Float(got) - s / Float(Self.pool)) <= 1e-3, "block \(b) head \(h) dim \(d)")
                }
            }
        }
    }

    // (32, 32, 96): one lane per KV head (Phi-3.5); (12, 6, 64): 6 KV heads, the generic per-head loop.
    @Test(arguments: [(8, 2, 128), (16, 8, 128), (32, 8, 64), (4, 1, 256), (32, 32, 96), (12, 6, 64)])
    func scoresSumTheHeadsDotProducts(nH: Int, nKV: Int, D: Int) throws {
        let W = nKV * D, G = nH / nKV, b0 = 2, nb = 150   // three threadgroups, the last one partial
        var rng = RNG(s: UInt64(nH * 1000 + D))
        let q = (0..<(nH * D)).map { _ in rng.float() }
        let idx = (0..<((b0 + nb) * W)).map { _ in Float16(rng.float()) }
        for perHead in [false, true] {
        let rows = perHead ? nKV : 1
        let scores = try GPU.shared.buffer((rows * nb + 1) * 4)
        try runGPU { enc in
            enc.setComputePipelineState(try GPU.shared.pipeline("sparse_score"))
            enc.set(makeBuffer(q), 0, 0); enc.set(makeBuffer(idx), 0, 1); enc.set(scores, 0, 2)
            enc.params(SparseParams(pool: Int32(Self.pool), nKV: Int32(nKV), D: Int32(D), b0: Int32(b0), nb: Int32(nb), nH: Int32(nH), rows: Int32(rows)), 3)
            enc.dispatchThreadgroups(MTLSize(width: (nb + Transformer.sparseScoreBlocks - 1) / Transformer.sparseScoreBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Transformer.sparseScoreThreads, height: 1, depth: 1))
        }
        let got = readFloats(scores, rows * nb)
        for n in 0..<nb {
            var perKV = [Float](repeating: 0, count: nKV)
            for h in 0..<nKV {
                for d in 0..<D {
                    var qs: Float = 0
                    for g in 0..<G { qs += q[(h * G + g) * D + d] }
                    perKV[h] += qs * Float(idx[(b0 + n) * W + h * D + d])
                }
            }
            // Per KV head: scores[kvh · nb + n]; shared: scores[n], the sum over the KV heads.
            let expected = perHead ? perKV : [perKV.reduce(0, +)]
            for (r, s) in expected.enumerated() {
                #expect(abs(got[r * nb + n] - s) <= 1e-3 * max(1, abs(s)), "perHead \(perHead) row \(r) block \(n): \(got[r * nb + n]) vs \(s)")
            }
        }
        }
    }

    /// Total order of `sparse_key` (the kernel's), so the reference breaks ties the same way.
    static func key(_ f: Float) -> UInt32 {
        let u = f.bitPattern
        return u & 0x8000_0000 != 0 ? ~u : u | 0x8000_0000
    }

    struct SelectCase: CustomTestStringConvertible, Sendable {
        let N: Int, p: Int, values: [Float]?, threads: Int
        var testDescription: String { "N=\(N) p=\(p) \(values == nil ? "distinct" : "ties") threads=\(threads)" }
    }
    static let selectCases: [SelectCase] = [
        SelectCase(N: 1000, p: 100, values: nil, threads: 1024),
        SelectCase(N: 1000, p: 100, values: nil, threads: 96),            // several blocks per thread, 3 simdgroups
        SelectCase(N: 777, p: 777, values: nil, threads: 1024),            // everything
        SelectCase(N: 5000, p: 512, values: [-1, 0, 0.5, 2], threads: 1024), // ties at the threshold
        SelectCase(N: 5000, p: 512, values: [-1, 0, 0.5, 2], threads: 64),
        SelectCase(N: 33, p: 1, values: nil, threads: 1024),
        SelectCase(N: 4096, p: 3, values: [-3, -2.5], threads: 1024),      // negative scores
        SelectCase(N: 3, p: 2, values: [1], threads: 1024),                // fewer blocks than threads, all tied
    ]

    @Test(arguments: selectCases)
    func selectKeepsTheTopBlocksInPositionOrder(c: SelectCase) throws {
        let pool = Self.pool, sinkEnd = 4, b0 = sinkEnd / pool
        let winStart = (b0 + c.N) * pool, L = winStart + 37
        var rng = RNG(s: UInt64(c.N * 7 + c.p))
        let scores: [Float] = (0..<c.N).map { _ in
            if let v = c.values { return v[Int(rng.next() % UInt64(v.count))] }
            return rng.float(-10, 10)
        }
        // Reference: highest key first, earlier block first among equals; then back in position order.
        let kept = (0..<c.N).sorted { a, b in
            let ka = Self.key(scores[a]), kb = Self.key(scores[b])
            return ka != kb ? ka > kb : a < b
        }.prefix(c.p).sorted()
        var expected = (0..<sinkEnd).map { Int32($0) }
        for n in kept { for i in 0..<pool { expected.append(Int32((b0 + n) * pool + i)) } }
        expected += (winStart..<L).map { Int32($0) }

        let pso = try GPU.shared.pipeline("sparse_select")
        let threads = min(c.threads, pso.maxTotalThreadsPerThreadgroup) / 32 * 32
        let list = makeBuffer([Int32](repeating: -1, count: L + 8))
        runGPU { enc in
            enc.setComputePipelineState(pso)
            enc.set(makeBuffer(scores), 0, 0); enc.set(list, 0, 1)
            enc.params(SparseParams(pool: Int32(pool), b0: Int32(b0), nb: Int32(c.N), p: Int32(c.p),
                                    sinkEnd: Int32(sinkEnd), winStart: Int32(winStart), L: Int32(L)), 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
        let got = Array(UnsafeBufferPointer(start: list.contents().bindMemory(to: Int32.self, capacity: L + 8), count: L + 8))
        #expect(Array(got.prefix(expected.count)) == expected)
        #expect(got[expected.count...].allSatisfy { $0 == -1 }, "wrote past the list")
    }

    /// One selection per row (KV head): row r reads its own scores and writes its own list,
    /// `listStride` entries apart, each equal to selecting that row alone.
    @Test func selectRowsAreIndependent() throws {
        let pool = Self.pool, sinkEnd = 4, b0 = sinkEnd / pool, N = 700, p = 64, rows = 3
        let winStart = (b0 + N) * pool, L = winStart + 21, stride = L + 50
        var rng = RNG(s: 77)
        let scores: [Float] = (0..<(rows * N)).map { _ in rng.float(-10, 10) }
        let pso = try GPU.shared.pipeline("sparse_select")
        let lists = makeBuffer([Int32](repeating: -1, count: rows * stride))
        runGPU { enc in
            enc.setComputePipelineState(pso)
            enc.set(makeBuffer(scores), 0, 0); enc.set(lists, 0, 1)
            enc.params(SparseParams(pool: Int32(pool), b0: Int32(b0), nb: Int32(N), p: Int32(p), sinkEnd: Int32(sinkEnd),
                                    winStart: Int32(winStart), L: Int32(L), listStride: Int32(stride), rows: Int32(rows)), 2)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(1024, pso.maxTotalThreadsPerThreadgroup) / 32 * 32, height: 1, depth: 1))
        }
        let got = Array(UnsafeBufferPointer(start: lists.contents().bindMemory(to: Int32.self, capacity: rows * stride), count: rows * stride))
        let count = sinkEnd + p * pool + (L - winStart)
        for r in 0..<rows {
            let s = Array(scores[(r * N)..<((r + 1) * N)])
            let kept = (0..<N).sorted { a, b in
                let ka = Self.key(s[a]), kb = Self.key(s[b])
                return ka != kb ? ka > kb : a < b
            }.prefix(p).sorted()
            var expected = (0..<sinkEnd).map { Int32($0) }
            for n in kept { for i in 0..<pool { expected.append(Int32((b0 + n) * pool + i)) } }
            expected += (winStart..<L).map { Int32($0) }
            #expect(expected.count == count)
            #expect(Array(got[(r * stride)..<(r * stride + count)]) == expected, "row \(r)")
            #expect(got[(r * stride + count)..<((r + 1) * stride)].allSatisfy { $0 == -1 }, "row \(r) wrote past its list")
        }
    }

    /// attention_sparse over the list 0 ..< L reproduces attention_split bit for bit; over a subset
    /// it equals attention over those positions.
    // (128, 10, 2) and (128, 16, 2): 5 and 8 query heads per KV head (Qwen3 14B / 32B shapes).
    @Test(arguments: [(128, 4, 2), (128, 8, 2), (64, 8, 1), (256, 8, 1), (128, 10, 2), (128, 16, 2)])
    func attentionOverTheList(D: Int, nH: Int, nKV: Int) throws {
        let maxSeq = 128, offset = 100, L = offset + 1, e = D / 32
        let G = nH / nKV, gSub = e <= 4 ? G : min(G, 2)
        let V = Transformer.attnSplitVirtualSG, NSG = Transformer.attnSplitSG
        var rng = RNG(s: UInt64(D + nH))
        let kh = (0..<(nKV * maxSeq * D)).map { _ in Float16(rng.float()) }
        let vh = (0..<(nKV * maxSeq * D)).map { _ in Float16(rng.float()) }
        let kBuf = makeBuffer(kh), vBuf = makeBuffer(vh)
        let q = (0..<(nH * D)).map { _ in rng.float() }
        let scale = 3 / Float(D).squareRoot()
        let p = AttnParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), qStride: Int32(nH * D), maxSeq: Int32(maxSeq),
                           offset: Int32(offset), scale: scale, outStride: Int32(nH * D), T: 1, window: 0, softcap: 0)
        /// One list shared by every KV head, or one per KV head (equal lengths).
        func run(lists: [[Int32]]?) throws -> [Float] {
            let part = try GPU.shared.buffer(nH * V * (D + 2) * 4), out = try GPU.shared.buffer(nH * D * 4)
            try runGPU { enc in
                if let lists {
                    enc.setComputePipelineState(try GPU.shared.pipeline("attention_sparse_\(e)_\(gSub)"))
                    enc.set(makeBuffer(q), 0, 0); enc.set(kBuf, 0, 1); enc.set(vBuf, 0, 2); enc.set(part, 0, 3); enc.params(p, 4)
                    let count = lists[0].count
                    enc.set(makeBuffer(lists.flatMap { $0 }), 0, 5)
                    enc.params(SIMD2<Int32>(Int32(count), lists.count == 1 ? 0 : Int32(count)), 6)
                } else {
                    enc.setComputePipelineState(try GPU.shared.pipeline("attention_split_\(e)_\(gSub)"))
                    enc.set(makeBuffer(q), 0, 0); enc.set(kBuf, 0, 1); enc.set(vBuf, 0, 2); enc.set(part, 0, 3); enc.params(p, 4)
                }
                enc.dispatchThreadgroups(MTLSize(width: nKV, height: V / NSG, depth: G / gSub), threadsPerThreadgroup: MTLSize(width: NSG * 32, height: 1, depth: 1))
                enc.setComputePipelineState(try GPU.shared.pipeline("attention_combine_\(e)"))
                enc.set(part, 0, 0); enc.set(out, 0, 1); enc.params(p, 2)
                enc.dispatchThreadgroups(MTLSize(width: nH, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            return readFloats(out, nH * D)
        }
        func run(sparseList: [Int32]) throws -> [Float] { try run(lists: [sparseList]) }
        /// Attention of every query head over its KV head's positions, on the CPU.
        func reference(_ positions: (Int) -> [Int32]) -> [Float] {
            var ref = [Float](repeating: 0, count: nH * D)
            for h in 0..<nH {
                let kvh = h / (nH / nKV)
                let list = positions(kvh)
                let s: [Float] = list.map { j in
                    var d: Float = 0
                    for x in 0..<D { d += q[h * D + x] * scale * Float(kh[(kvh * maxSeq + Int(j)) * D + x]) }
                    return d
                }
                let m = s.max()!
                let w = s.map { exp($0 - m) }
                let z = w.reduce(0, +)
                for x in 0..<D {
                    var acc: Float = 0
                    for (i, j) in list.enumerated() { acc += w[i] * Float(vh[(kvh * maxSeq + Int(j)) * D + x]) }
                    ref[h * D + x] = acc / z
                }
            }
            return ref
        }
        let split = try run(lists: nil)
        let all = try run(sparseList: (0..<L).map { Int32($0) })
        #expect(all.map(\.bitPattern) == split.map(\.bitPattern), "full list vs attention_split")

        // A subset: the sink, scattered blocks and the window; and entries out of range, which clamp to L - 1.
        var subset: [Int32] = [0, 1, 2, 3, 20, 21, 22, 23, 52, 53, 54, 55]
        subset += (90..<L).map { Int32($0) }
        assertClose(try run(sparseList: subset), reference { _ in subset }, "sparse attention over a subset D=\(D)")
        let clamped = try run(sparseList: subset + [Int32(L + 5), -3])
        var refClamped = subset + [Int32(L - 1), 0]
        refClamped.sort()
        assertClose(clamped, try run(sparseList: refClamped), tol: 1e-5, "out-of-range entries clamp")

        // One list per KV head: each head attends to its own blocks.
        func headList(_ kvh: Int) -> [Int32] {
            var list: [Int32] = [0, 1, 2, 3]
            for b in 0..<3 {
                let start = 8 + 4 * (kvh + 5 * b)
                for i in 0..<4 { list.append(Int32(start + i)) }
            }
            for j in 90..<L { list.append(Int32(j)) }
            return list
        }
        let perHead = try run(lists: (0..<nKV).map(headList))
        assertClose(perHead, reference(headList), "sparse attention over per-head lists D=\(D)")
        if nKV > 1 { #expect(perHead != (try run(sparseList: headList(0))), "the heads' lists were not used") }
    }
}
