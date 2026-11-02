import Foundation
import Testing
@testable import MLXEngine

/// Speculative verification on synthetic logits: the emitted tokens must be exactly what the
/// target alone would produce (argmax when greedy, the target's distribution when sampling).
@Suite("Speculative decoding")
struct SpeculativeTests {
    /// Runs `verifyDraft` over `rows` (one logits row per draft token plus the bonus row).
    static func verify(_ rows: [[Float]], draft: [Int], params: SamplingParameters, sampler: inout Sampler,
                       stopAfter: Int = .max) -> (emitted: [Int], accepted: Int, running: Bool) {
        var shaper = Sampler(params: params, rng: XorShift(seed: 1))
        var emitted: [Int] = []
        let (accepted, running) = sampler.verifyDraft(draft, dist: { i in
            var row = rows[i]
            return row.withUnsafeMutableBufferPointer { shaper.distribution($0.baseAddress!, count: $0.count, recent: [])! }
        }) { t, _ in
            emitted.append(t)
            return emitted.count < stopAfter
        }
        return (emitted, accepted, running)
    }

    static var greedy: SamplingParameters {
        var p = SamplingParameters()
        p.temperature = 0
        p.repetitionPenalty = 1
        return p
    }

    /// Temperature 1 with every filter off: the processed distribution is exactly softmax(logits).
    static var plain: SamplingParameters {
        var p = SamplingParameters()
        p.temperature = 1; p.topK = 0; p.topP = 1; p.minP = 0; p.repetitionPenalty = 1
        return p
    }

    static func onehot(_ i: Int, _ V: Int = 8) -> [Float] { (0..<V).map { $0 == i ? 4 : Float($0) * 0.1 } }

    @Test func greedyRejectsAtFirstMismatchAndEmitsTheTargetToken() {
        var s = Sampler(params: Self.greedy, rng: XorShift(seed: 1))
        let rows = [Self.onehot(3), Self.onehot(5), Self.onehot(1), Self.onehot(7)]
        let r = Self.verify(rows, draft: [3, 5, 2], params: Self.greedy, sampler: &s)
        #expect(r.emitted == [3, 5, 1])
        #expect(r.accepted == 2)
        #expect(r.running)
        let r0 = Self.verify(rows, draft: [6, 5, 1], params: Self.greedy, sampler: &s)
        #expect(r0.emitted == [3] && r0.accepted == 0)
    }

    @Test func greedyAcceptsAllAndAddsTheBonusToken() {
        var s = Sampler(params: Self.greedy, rng: XorShift(seed: 1))
        let rows = [Self.onehot(3), Self.onehot(5), Self.onehot(1), Self.onehot(7)]
        let r = Self.verify(rows, draft: [3, 5, 1], params: Self.greedy, sampler: &s)
        #expect(r.emitted == [3, 5, 1, 7])
        #expect(r.accepted == 3)
    }

    @Test func stopsWhenEmitSaysSo() {
        var s = Sampler(params: Self.greedy, rng: XorShift(seed: 1))
        let rows = [Self.onehot(3), Self.onehot(5), Self.onehot(1), Self.onehot(7)]
        let r = Self.verify(rows, draft: [3, 5, 1], params: Self.greedy, sampler: &s, stopAfter: 2)
        #expect(r.emitted == [3, 5])
        #expect(!r.running)
    }

    /// Sampling: whatever the draft proposes, the first emitted token is distributed as the target's
    /// processed distribution, a token with p = 0 is never emitted, and a rejected draft is never
    /// emitted as its own correction.
    @Test(arguments: [0, 2, 5])
    func sampledOutputFollowsTheTargetDistribution(draft: Int) {
        let row: [Float] = [1.0, 0.2, 2.0, -0.5, 0.7, -Float.infinity, 1.5, 0.0]   // token 5 has p = 0
        let mx = row.max()!
        let e = row.map { exp($0 - mx) }
        let p = e.map { $0 / e.reduce(0, +) }
        var s = Sampler(params: Self.plain, rng: XorShift(seed: 12345))
        let n = 200_000
        var counts = [Int](repeating: 0, count: row.count)
        for _ in 0..<n {
            let r = Self.verify([row, Self.onehot(0)], draft: [draft], params: Self.plain, sampler: &s)
            counts[r.emitted[0]] += 1
            if r.accepted == 0 { #expect(r.emitted[0] != draft) }
            #expect(r.emitted.count == (r.accepted == 1 ? 2 : 1))
        }
        #expect(counts[5] == 0)
        for i in 0..<row.count {
            #expect(abs(Double(counts[i]) / Double(n) - Double(p[i])) < 0.006, "token \(i): \(Double(counts[i]) / Double(n)) vs \(p[i])")
        }
    }

    /// Two drafted tokens: the pair distribution matches the target's chain (row 1 is conditioned on
    /// the first draft being accepted, which is the only case it is used).
    @Test func sampledSecondTokenIsConditionedOnAcceptance() {
        let row0: [Float] = [0.5, 1.0, 0.0, -1.0]
        let row1: [Float] = [0.0, -0.5, 1.2, 0.3]
        func softmax(_ r: [Float]) -> [Double] { let m = r.max()!; let e = r.map { Double(exp($0 - m)) }; let z = e.reduce(0, +); return e.map { $0 / z } }
        let p0 = softmax(row0), p1 = softmax(row1)
        var s = Sampler(params: Self.plain, rng: XorShift(seed: 777))
        let n = 200_000
        var pair = [Int](repeating: 0, count: 4)   // second token, given the first is the draft (1)
        var firstIsDraft = 0
        for _ in 0..<n {
            let r = Self.verify([row0, row1, row0], draft: [1, 2], params: Self.plain, sampler: &s)
            guard r.emitted[0] == 1 else { continue }
            firstIsDraft += 1
            pair[r.emitted[1]] += 1
        }
        #expect(abs(Double(firstIsDraft) / Double(n) - p0[1]) < 0.006)
        for i in 0..<4 { #expect(abs(Double(pair[i]) / Double(firstIsDraft) - p1[i]) < 0.008) }
    }

    @Test func memoryCheckSumsTargetAndDraft() throws {
        guard let q = installedModel(ModelTests.qwen3) else { return }
        let one = try InferenceEngine.memoryNeed(directory: q, contextLength: 4096).bytes
        try InferenceEngine.checkMemory(directories: [q], contextLength: 4096, budget: UInt64(one) + 1)
        #expect(throws: EngineError.self) {
            try InferenceEngine.checkMemory(directories: [q, q], contextLength: 4096, budget: UInt64(one) + 1)
        }
    }

    @Test func refusesADraftWithAnotherTokenizer() throws {
        guard let q = installedModel(ModelTests.qwen3), let l = installedModel(ModelTests.llama) else { return }
        let tok = try Tokenizer(directory: q)
        try InferenceEngine.checkDraftCompatible(target: q, targetTokenizer: tok, draft: q)
        #expect(throws: EngineError.self) { try InferenceEngine.checkDraftCompatible(target: q, targetTokenizer: tok, draft: l) }
    }

    @Test func autoDraftPicksASmallModelOfTheSameFamily() throws {
        guard let big = installedModel("mlx-community/Qwen3-8B-4bit"), installedModel(ModelTests.qwen3) != nil else { return }
        let d = InferenceEngine.autoDraft(for: big)
        #expect(d?.modelType == "qwen3")
        #expect((d?.sizeBytes ?? .max) * 4 <= (ModelStore.scan().first { $0.directory == big }?.sizeBytes ?? 0))
        #expect(InferenceEngine.autoDraft(for: installedModel(ModelTests.qwen3)!) == nil)
    }

    /// With the model as its own draft, every model draft is accepted (up to near-ties between the
    /// batched verify and single-token kernels) and the output equals plain greedy decoding.
    @Test(.enabled(if: installedModel(ModelTests.qwen3) != nil))
    func selfDraftMatchesPlainGreedy() async throws {
        let dir = installedModel(ModelTests.qwen3)!
        var p = SamplingParameters()
        p.temperature = 0
        p.maxTokens = 200
        p.seed = 1   // the default repetition penalty stays on: its window must include accepted drafts
        let msgs = [ChatMessage(role: .user, content: "Write a short story about a lighthouse keeper.")]
        func run(_ e: InferenceEngine) async throws -> (String, GenerationStats) {
            var text = "", stats = GenerationStats()
            for try await ev in e.chat(msgs, parameters: p, enableThinking: false) {
                if case .text(let t) = ev { text += t }
                if case .finished(let s) = ev { stats = s }
            }
            return (text, stats)
        }
        let engine = InferenceEngine()
        _ = try await engine.load(directory: dir, contextLength: 1024)
        let (plain, plainStats) = try await run(engine)
        let info = try await engine.load(directory: dir, draftDirectory: dir, contextLength: 1024)
        #expect(info.draftName != nil)
        let (spec, s) = try await run(engine)
        #expect(spec == plain)
        #expect(s.generatedTokens == plainStats.generatedTokens)
        let drafted = s.modelDraftTokens ?? 0, accepted = s.acceptedModelDraftTokens ?? 0
        #expect(drafted > 50)
        #expect(Double(accepted) >= 0.95 * Double(drafted), "accepted \(accepted)/\(drafted)")
        // A second turn reuses both caches (the draft's prefix bookkeeping must survive it).
        let (again, _) = try await run(engine)
        #expect(again == plain)
    }
}
