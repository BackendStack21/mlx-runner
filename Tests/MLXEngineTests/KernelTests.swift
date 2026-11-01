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

@Suite("GPU token selection")
struct SampleKernelTests {
    /// Runs `sample_token` `draws` times over the same logits and history; returns the drawn tokens
    /// and the final pool size / attempt count.
    func gpuSample(_ logits: [Float], _ params: SamplingParameters, history: [Int32] = [], draws: Int = 1,
                   seed: UInt64 = 1) throws -> (tokens: [Int], pool: Int, attempts: Int) {
        let pso = try GPU.shared.pipeline("sample_token")
        let threads = min(1024, pso.maxTotalThreadsPerThreadgroup) / 32 * 32
        let hist = try GPU.shared.buffer((history.count + 2) * 4)
        history.withUnsafeBytes { hist.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let state = try GPU.shared.buffer(16)
        state.contents().storeBytes(of: seed, as: UInt64.self)
        state.contents().storeBytes(of: Float(8), toByteOffset: 8, as: Float.self)
        let status = try GPU.shared.buffer((history.count + 2) * 4)
        let diag = try GPU.shared.buffer(16)
        var tokens: [Int] = []
        let pos = max(0, history.count - 1)
        for _ in 0..<draws {
            let lb = makeBuffer(logits)   // the penalty edits logits in place: fresh copy per draw
            var sp = SampleParams(params, vocab: logits.count)
            sp.pos = Int32(pos)
            runGPU { enc in
                enc.setComputePipelineState(pso)
                enc.set(lb, 0, 0); enc.set(hist, 0, 1); enc.set(state, 0, 2); enc.set(status, 0, 3)
                enc.params(sp, 4)
                enc.set(diag, 0, 5)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            }
            #expect(status.contents().load(fromByteOffset: (pos + 1) * 4, as: Int32.self) == 0, "status")
            tokens.append(Int(hist.contents().load(fromByteOffset: (pos + 1) * 4, as: Int32.self)))
        }
        let d = diag.contents().bindMemory(to: Int32.self, capacity: 2)
        return (tokens, Int(d[0]), Int(d[1]))
    }

    /// Peaked logits with a long tail and a few exact ties, like a language model's.
    func testLogits(_ V: Int, seed: UInt64 = 3) -> [Float] {
        var rng = RNG(s: seed)
        var l = (0..<V).map { _ in -12 + 6 * rng.float() }
        for i in stride(from: 0, to: V, by: 97) { l[i] += 8 + 4 * rng.float() }
        l[5] = 7.25; l[777] = 7.25; l[V - 1] = 7.25
        return l
    }

    @Test func greedyMatchesCPUArgmaxIncludingTiesAndPenalty() throws {
        var params = SamplingParameters()
        params.temperature = 0
        for V in [1000, 32768, 151936] {
            var l = testLogits(V)
            l[300] = 30; l[9] = 30   // tie at the max: the lowest index wins
            params.repetitionPenalty = 1
            #expect(try gpuSample(l, params).tokens == [9])
            // Penalising token 9 (recent) hands the win to 300.
            params.repetitionPenalty = 1.3
            params.repetitionWindow = 64
            let history: [Int32] = [1, 2, 9, 9, 4]
            let cpu = { () -> Int in
                let p = UnsafeMutablePointer<Float>.allocate(capacity: V); p.initialize(from: l, count: V)
                var s = Sampler(params: params, rng: XorShift(seed: 1))
                return s.sample(p, count: V, recent: history.map { Int($0) }.suffix(64))
            }()
            #expect(cpu == 300)
            #expect(try gpuSample(l, params, history: history).tokens == [cpu])
        }
    }

    @Test func nonFiniteLogitsAreReported() throws {
        var l = testLogits(4096)
        l[7] = .infinity
        let pso = try GPU.shared.pipeline("sample_token")
        let hist = try GPU.shared.buffer(64), state = try GPU.shared.buffer(16), status = try GPU.shared.buffer(64), diag = try GPU.shared.buffer(16)
        var sp = SampleParams(SamplingParameters(), vocab: l.count)
        sp.pos = 0
        let lb = makeBuffer(l)
        runGPU { enc in
            enc.setComputePipelineState(pso)
            enc.set(lb, 0, 0); enc.set(hist, 0, 1); enc.set(state, 0, 2); enc.set(status, 0, 3)
            enc.params(sp, 4)
            enc.set(diag, 0, 5)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        #expect(status.contents().load(fromByteOffset: 4, as: Int32.self) == 1)
        #expect(hist.contents().load(fromByteOffset: 4, as: Int32.self) == 0, "a valid id is left for the queued next step")
    }

    /// The GPU draws follow the CPU sampler's distribution: same support (top-k, min-p and top-p
    /// cuts) and matching frequencies.
    @Test(arguments: [(40, Float(0.95), Float(0)), (0, Float(1), Float(0)), (200, Float(0.8), Float(0.05)), (5, Float(1), Float(0))])
    func sampledDistributionMatchesCPU(topK: Int, topP: Float, minP: Float) throws {
        var params = SamplingParameters()
        params.temperature = 0.9
        params.topK = topK; params.topP = topP; params.minP = minP
        params.repetitionPenalty = 1.1
        let V = 32768
        let l = testLogits(V, seed: 11)
        let history: [Int32] = (0..<40).map { Int32($0 * 97) }   // penalise some of the peaks
        let p = UnsafeMutablePointer<Float>.allocate(capacity: V); p.initialize(from: l, count: V)
        var s = Sampler(params: params, rng: XorShift(seed: 1))
        let distOpt = s.distribution(p, count: V, recent: history.map { Int($0) }.suffix(64))
        let dist = try #require(distOpt)
        let cpu = Dictionary(uniqueKeysWithValues: dist.map { ($0.id, $0.p) })
        let draws = 3000
        let r = try gpuSample(l, params, history: history, draws: draws)
        #expect(r.pool >= dist.count && r.pool <= 1024, "pool \(r.pool) after \(r.attempts) attempts")
        var freq: [Int: Int] = [:]
        for t in r.tokens { freq[t, default: 0] += 1 }
        for (t, n) in freq { #expect(cpu[t] != nil, "token \(t) drawn \(n) times is outside the CPU distribution") }
        for (t, pr) in cpu where pr > 0.01 {
            let f = Float(freq[t] ?? 0) / Float(draws)
            let sigma = (pr * (1 - pr) / Float(draws)).squareRoot()
            #expect(abs(f - pr) < 4.5 * sigma + 0.002, "token \(t): p \(pr) drawn \(f)")
        }
    }

    @Test func hugeTopKIsCappedWithoutExtraPasses() throws {
        var params = SamplingParameters()
        params.temperature = 1; params.topK = 5000; params.topP = 1; params.repetitionPenalty = 1
        let r = try gpuSample(testLogits(40000), params, draws: 5)
        #expect(r.pool >= 1024 && r.pool <= 1024 * 2, "pool \(r.pool)")
        #expect(r.attempts <= 12, "attempts \(r.attempts)")
    }

    @Test func tiesAtTheTopAndTinyVocabularies() throws {
        var params = SamplingParameters()
        params.temperature = 1; params.topK = 3; params.topP = 1; params.repetitionPenalty = 1
        // 2000 logits tied at the max: the top 3 are the three lowest ids, drawn with equal probability.
        let l = [Float](repeating: 1, count: 2000)
        let r = try gpuSample(l, params, draws: 60)
        #expect(Set(r.tokens) == [0, 1, 2], "\(Set(r.tokens))")
        #expect(r.attempts < 40, "attempts \(r.attempts)")
        // Fewer logits than k.
        params.topK = 40
        let small: [Float] = [0.5, 2, -1]
        let r2 = try gpuSample(small, params, draws: 200)
        #expect(Set(r2.tokens).isSubset(of: [0, 1, 2]))
        #expect(r2.tokens.filter { $0 == 1 }.count > 100)
    }
}
