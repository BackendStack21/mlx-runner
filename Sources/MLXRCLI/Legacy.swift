import Foundation
import MLXEngine

// The original `mlxr` command bodies, kept as they were while the commands move onto the parser kit.

/// `--draft <model|auto>`: the draft model for speculative decoding.
func draftDirectory(_ s: String?, target: URL) throws -> URL? {
    guard let s else { return nil }
    guard s == "auto" else { return try resolveModel(s) }
    guard let m = InferenceEngine.autoDraft(for: target) else {
        print("no installed draft model pairs with \(target.lastPathComponent); running without one")
        return nil
    }
    return m.directory
}

func sparseSummary(_ info: LoadedModelInfo) -> String {
    if info.sparseLayers > 0 {
        return "sparse attention on \(info.sparseLayers) layers from \(info.sparseMinContext) tokens (\(info.sparseTopK) long-range keys)"
    }
    return "sparse attention off" + (info.sparseNote.map { ": \($0)" } ?? "")
}

func formatBytes(_ b: Int) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file)
}

/// Long-context settings of `mlxr bench` (MLXR_BENCH_CTX turns the section on).
struct LongBench {
    let contexts: [Int]
    /// Teacher-forced positions of the text after the prompt (MLXR_BENCH_POSITIONS, default 512).
    let positions: Int
    /// Token offsets into the text where prompts start (MLXR_BENCH_TEXT_OFFSETS, default 0).
    let offsets: [Int]
    static let steps = 64
    /// Rows the model needs past the longest prompt: the measured positions plus a needle question.
    var margin: Int { max(Self.steps, positions) + 96 }

    init(env: [String: String]) {
        func ints(_ key: String) -> [Int] {
            (env[key] ?? "").split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }
        contexts = ints("MLXR_BENCH_CTX").filter { $0 > 0 }
        positions = min(max(ints("MLXR_BENCH_POSITIONS").first ?? 512, 0), 8192)
        let o = ints("MLXR_BENCH_TEXT_OFFSETS").filter { $0 >= 0 }
        offsets = o.isEmpty ? [0] : o
    }
}

/// `mlxr bench` with MLXR_BENCH_CTX: decode after long prompts, dense and (for a model loaded with
/// MLXR_SPARSE_ATTN) sparse. For each context:
/// - greedy decode speed, and how many greedy tokens sparse decode shares with dense decode;
/// - teacher-forced on dense decode's tokens: a control (dense forward against the GPU's greedy
///   tokens, must be all) and sparse argmax agreement;
/// - teacher-forced on the text after the prompt (MLXR_BENCH_TEXT only): KL(dense‖sparse), the
///   text's NLL under both, and top-1 agreement by dense decode's top-1 margin. Each position runs
///   sparse first and then dense, so the cache stays dense decode's;
/// - the cost of a dense verification pass of T tokens at that context;
/// - needles (MLXR_BENCH_TEXT only): five passkeys at 10–90% depth, recalled dense and sparse.
/// The prompt is MLXR_BENCH_TEXT from each of MLXR_BENCH_TEXT_OFFSETS, or pseudo-random tokens, on
/// which agreement says little.
func benchLongContext(_ t: Transformer, dir: URL, bench: LongBench) throws {
    var greedy = SamplingParameters()
    greedy.temperature = 0
    greedy.repetitionPenalty = 1
    let sp = SampleParams(greedy, vocab: t.vocabSize)
    let V = t.vocabSize
    let tokenizer = try Tokenizer(directory: dir)
    var text: [Int32] = []
    if let path = ProcessInfo.processInfo.environment["MLXR_BENCH_TEXT"] {
        guard let s = try? String(contentsOfFile: path, encoding: .utf8) else { throw CLIError.failure("cannot read MLXR_BENCH_TEXT \(path)") }
        text = tokenizer.encode(s).map { Int32($0) }.filter { $0 >= 0 && Int($0) < V }
        guard !text.isEmpty else { throw CLIError.failure("MLXR_BENCH_TEXT \(path) has no tokens") }
        let bos = tokenizer.bosToken.flatMap { b in tokenizer.tokenId(b).map { " (prompts start with \(b), id \($0))" } } ?? ""
        print("long-context prompt: \(path), \(text.count) tokens (repeated to length)\(bos)")
    } else {
        print("long-context prompt: pseudo-random tokens (MLXR_BENCH_TEXT=<file.txt> for real text)")
    }
    let original = t.sparse
    defer { t.sparse = original }
    // Hybrid models (recurrent state): every rerun of a position first restores the state there.
    var base: Transformer.RecurrentSnapshot?, last: Transformer.RecurrentSnapshot?
    func mark() { if t.config.isHybrid { last = t.snapshotRecurrent() } }
    func seek(_ pos: Int) throws {
        guard t.config.isHybrid, t.recurrentPosition != pos else { return }
        guard let s = [base, last].compactMap({ $0 }).first(where: { $0.position == pos }) else { throw CLIError.failure("no recurrent snapshot at \(pos)") }
        t.restoreRecurrent(s)
    }
    let sparse = t.sparseAvailable ? original : nil
    let steps = LongBench.steps
    func logits(_ tokens: [Int32], offset: Int) throws -> [Float] {
        Array(UnsafeBufferPointer(start: try t.forward(tokens, offset: offset), count: V))
    }
    func argmax(_ x: [Float]) -> Int {
        var best = 0
        for v in 1..<x.count where x[v] > x[best] { best = v }
        return best
    }
    func logSoftmax(_ x: [Float]) -> [Double] {
        let m = Double(x.max() ?? 0)
        var s = 0.0
        for v in x { s += exp(Double(v) - m) }
        let lse = m + log(s)
        return x.map { Double($0) - lse }
    }
    for ctx in bench.contexts {
        guard ctx + bench.margin < t.maxSeq else { print("decode @\(ctx): skipped (the model's context is \(t.maxSeq))"); continue }
        for (oi, off) in (text.isEmpty ? [0] : bench.offsets).enumerated() {
            var rng: UInt64 = 0x9E3779B97F4A7C15
            let lo = min(1000, V / 2), span = UInt64(max(1, min(V, 30000) - lo))
            func token(_ i: Int) -> Int32 {
                if !text.isEmpty { return text[(off + i) % text.count] }
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
                return Int32(lo + Int(rng % span))
            }
            var prompt = (0..<ctx).map { token($0) }
            // Needles: "The passkey for door k is NNNNN." at 10, 30, 50, 70 and 90% of the prompt.
            var needles: [(door: Int, code: String)] = []
            if !text.isEmpty {
                for (k, depth) in [0.1, 0.3, 0.5, 0.7, 0.9].enumerated() {
                    let code = String(10000 + (k + 1) * 17389 % 89999)
                    let ids = tokenizer.encode(" The passkey for door \(k + 1) is \(code). ").map { Int32($0) }
                    let at = Int(Double(ctx) * depth)
                    guard at + ids.count < ctx else { continue }
                    prompt.replaceSubrange(at..<(at + ids.count), with: ids)
                    needles.append((k + 1, code))
                }
                // Text starts with BOS when the tokenizer has one: Gemma without it gives nonsense
                // (and Gemma 4's tokenizer does not say to add it). It replaces the first text token.
                if let b = tokenizer.bosToken, let id = tokenizer.tokenId(b), id >= 0, id < V { prompt[0] = Int32(id) }
            }
            let at = off > 0 ? " from text token \(off)" : ""
            t.sparse = nil
            t.seedSampler(1)
            let s0 = Date()
            let first = try t.prefill(prompt, offset: 0, sample: sp)
            let prefillTime = Date().timeIntervalSince(s0)
            if t.config.isHybrid { base = t.snapshotRecurrent() }
            print(String(format: "prefill @%d%@: %.1f s (%.0f tok/s)", ctx, at as NSString, prefillTime, Double(ctx) / prefillTime))
            // Each run decodes from the same prompt (positions from ctx on are rewritten, the index with them).
            var dense: [Int] = []
            var modes: [(String, SparseAttention?)] = [("dense", nil)]
            if let sparse, oi == 0 { modes.append(("sparse", sparse)) }
            for (label, mode) in modes {
                t.sparse = mode
                t.seedSampler(1)
                try seek(ctx)
                var tokens: [Int] = []
                var gpu = 0.0
                let s = Date()
                var cur = try t.beginDecodeStep(offset: ctx, sample: sp)
                for i in 0..<steps {
                    let next = i + 1 < steps ? try t.beginDecodeStep(offset: ctx + i + 1, sample: sp) : nil
                    tokens.append(try t.finish(cur))
                    gpu += t.lastGPUTime
                    if let next { cur = next }
                }
                let dt = Date().timeIntervalSince(s)
                if oi == 0 {
                    print(String(format: "%@ @%d: %6.1f tok/s  (wall %.2f ms, gpu %.2f ms per token)", label as NSString, ctx,
                                 Double(steps) / dt, dt / Double(steps) * 1000, gpu / Double(steps) * 1000))
                }
                if mode == nil {
                    dense = tokens
                } else {
                    let same = zip(tokens, dense).prefix { pair in pair.0 == pair.1 }.count
                    print("  sparse greedy: the first \(same) of \(steps) tokens equal dense decode's")
                }
            }
            guard let sparse else { continue }
            // Teacher-forced on dense decode's tokens: sparse first, then dense (which rewrites the
            // position, so the cache stays dense decode's).
            var match = 0, control = 0
            var input = Int32(first)
            try seek(ctx)
            for i in 0..<steps {
                t.sparse = sparse
                mark()
                if argmax(try logits([input], offset: ctx + i)) == dense[i] { match += 1 }
                try seek(ctx + i)
                t.sparse = nil
                if argmax(try logits([input], offset: ctx + i)) == dense[i] { control += 1 }
                input = Int32(dense[i])
            }
            print("  teacher-forced on dense decode's tokens: sparse argmax agrees at \(match) of \(steps) (control, dense forward: \(control) of \(steps))")
            if !text.isEmpty && bench.positions > 0 {
                // Teacher-forced on the text after the prompt.
                let N = bench.positions
                var kl: [Double] = [], nllDense = 0.0, nllSparse = 0.0, agree = 0
                let edges: [Float] = [0.1, 0.5, 1]
                var bucketTotal = [Int](repeating: 0, count: edges.count + 1), bucketAgree = bucketTotal
                try seek(ctx)
                for i in 0..<N {
                    let input = text[(off + ctx + i) % text.count], target = Int(text[(off + ctx + i + 1) % text.count])
                    t.sparse = sparse
                    mark()
                    let ls = try logits([input], offset: ctx + i)
                    try seek(ctx + i)
                    t.sparse = nil
                    let ld = try logits([input], offset: ctx + i)
                    let pd = logSoftmax(ld), ps = logSoftmax(ls)
                    var k = 0.0
                    for v in 0..<V where pd[v] > -40 { k += exp(pd[v]) * (pd[v] - ps[v]) }
                    kl.append(max(0, k))
                    nllDense -= pd[target]
                    nllSparse -= ps[target]
                    let top = argmax(ld)
                    var second = top == 0 ? 1 : 0
                    for v in 0..<V where v != top && ld[v] > ld[second] { second = v }
                    let bucket = edges.firstIndex { ld[top] - ld[second] < $0 } ?? edges.count
                    bucketTotal[bucket] += 1
                    if argmax(ls) == top { agree += 1; bucketAgree[bucket] += 1 }
                }
                kl.sort()
                let meanKL = kl.reduce(0, +) / Double(N)
                let dNLL = (nllSparse - nllDense) / Double(N)
                print(String(format: "  teacher-forced on the text (%d positions): KL(dense‖sparse) mean %.4f, median %.4f, p95 %.4f, max %.3f nats",
                             N, meanKL, kl[N / 2], kl[min(N - 1, N * 95 / 100)], kl[N - 1]))
                print(String(format: "    text NLL: dense %.4f, sparse %.4f nats/token (Δ %+.4f, perplexity ×%.4f)",
                             nllDense / Double(N), nllSparse / Double(N), dNLL, exp(dNLL)))
                let labels = ["<0.1", "0.1–0.5", "0.5–1", "≥1"]
                let byMargin = zip(labels, zip(bucketAgree, bucketTotal)).map { "\($0.0): \($0.1.0)/\($0.1.1)" }.joined(separator: ", ")
                print(String(format: "    top-1 agreement %d/%d (%.1f%%); by dense top-1 margin (nats) %@", agree, N,
                             100 * Double(agree) / Double(N), byMargin as NSString))
            }
            if oi == 0 && !t.config.isHybrid {
                // A dense verification pass of T tokens at this context (speculative decoding).
                t.sparse = nil
                var line: [String] = []
                for T in [1, 2, 5, 8] where T <= Transformer.maxVerifyTokens {
                    var gpu = 0.0
                    for i in 0..<4 {
                        _ = try t.forward((0..<T).map { Int32(600 + $0 + i) }, offset: ctx, allLogits: true)
                        gpu += t.lastGPUTime
                    }
                    line.append(String(format: "T=%d %.2f ms", T, gpu / 4 * 1000))
                }
                print("  dense verify @\(ctx) (gpu per pass): " + line.joined(separator: ", "))
            }
            if !needles.isEmpty {
                // Each question is prefilled dense except its last token, which decodes in the mode
                // measured; then 8 greedy tokens.
                var passed = [0, 0]
                var answers: [String] = []
                for (door, code) in needles {
                    let q = tokenizer.encode("\n\nQuestion: What is the passkey for door \(door)?\nAnswer: The passkey for door \(door) is").map { Int32($0) }
                    for (mi, mode) in [nil, sparse].enumerated() {
                        t.sparse = nil
                        try seek(ctx)
                        _ = try t.forward(Array(q.dropLast()), offset: ctx)
                        t.sparse = mode
                        var out: [Int] = []
                        var input = q.last!
                        for i in 0..<8 {
                            let next = argmax(try logits([input], offset: ctx + q.count - 1 + i))
                            out.append(next)
                            input = Int32(next)
                        }
                        let answer = tokenizer.decode(out)
                        if answer.contains(code) { passed[mi] += 1 }
                        if mi == 1 && !answer.contains(code) { answers.append("door \(door) (\(code)): \(answer.debugDescription)") }
                    }
                }
                print("  needles: dense \(passed[0])/\(needles.count), sparse \(passed[1])/\(needles.count)"
                      + (answers.isEmpty ? "" : "; sparse missed " + answers.joined(separator: "; ")))
            }
        }
    }
}


/// Rate limiter for progress output.
final class LockedDate: @unchecked Sendable {
    private var date = Date.distantPast
    private let lock = NSLock()
    func elapsed(_ s: TimeInterval) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard Date().timeIntervalSince(date) >= s else { return false }
        date = Date()
        return true
    }
}

func legacySearch(_ query: String) async throws {
    let results = try await HubClient.search(query, limit: 25)
    for m in results {
        print(String(format: "%-60@ %10d ↓  %@", m.id as NSString, m.downloads ?? 0, (m.modelTypeTag ?? "") as NSString))
    }

}

func legacyDownload(_ repo: String) async throws {
    try validateRepoID(repo)
    let details = try await HubClient.details(repo)
    if !details.isSupported { print("warning: \(details.modelType ?? "unknown") architecture is not supported by the engine") }
    print("Downloading \(repo) (\(formatBytes(details.downloadSize)))")
    let dl = ModelDownload(details: details, destination: try ModelStore.directory(for: repo))
    let last = LockedDate()
    dl.onProgress = { p, done, total, speed in
        guard last.elapsed(0.25) || p >= 1 else { return }
        print(String(format: "\r  %5.1f%%  %@ / %@  %@/s   ", p * 100, formatBytes(Int(done)), formatBytes(Int(total)), formatBytes(Int(speed))), terminator: "")
    }
    dl.onEvent = { print("\n  \($0)") }
    let dir = try await performDownload(start: { try await dl.start() })
    print("\nSaved to \(dir.path)")

}

func legacyList() {
    for m in ModelStore.scan() {
        print(String(format: "%-60@ %10@  %@%@", m.repo as NSString, formatBytes(m.sizeBytes) as NSString,
                     (m.modelType ?? "?") as NSString, m.isSupported ? "" : " (unsupported)"))
    }

}

func legacyTokenize(_ model: String, _ text: String, idsOnly: Bool = false) throws {
    let tok = try Tokenizer(directory: try resolveModel(model))
    let ids = tok.encode(text)
    print(ids)
    if idsOnly { return }
    print(ids.map { tok.token($0) ?? "?" })
    print("roundtrip:", tok.decode(ids, includeSpecial: true))

}

func legacyBench(_ model: String) throws {
    let dir = try resolveModel(model)
    // Long-context decode (opt-in): MLXR_BENCH_CTX=8192,32768 adds a section at the end.
    let longBench = LongBench(env: ProcessInfo.processInfo.environment)
    let t = try Transformer(directory: dir, contextLength: max(4096, (longBench.contexts.max() ?? 0) + longBench.margin + 64))
    print("weights: \(formatBytes(t.weightBytes))")
    if let sp = t.sparse {
        print(t.sparseAvailable
              ? "sparse decode attention: top-k \(sp.topK), window \(sp.window), sink \(sp.sink), from \(sp.minContext) tokens"
              : "sparse decode attention: off (\(t.sparseOffReason ?? "unavailable"))")
    }
    print("attention: \(t.attentionSummary)")
    let quick = ProcessInfo.processInfo.environment["MLXR_BENCH_DECODE"] != nil
    let noProfile = quick || ProcessInfo.processInfo.environment["MLXR_BENCH_NOPROFILE"] != nil
    if !noProfile { print(try t.profile()) }
    if ProcessInfo.processInfo.environment["MLXR_BENCH_PROFILE_ONLY"] != nil { exit(0) }
    _ = try t.forward([1], offset: 0)
    for n in (quick ? [] : [128, 512, 2048]) {
        let toks = (0..<n).map { Int32($0 % 1000 + 100) }
        let s = Date()
        _ = try t.forward(toks, offset: 0)
        let dt = Date().timeIntervalSince(s)
        print(String(format: "prefill %5d tokens: %7.1f tok/s", n, Double(n) / dt))
    }
    // Token selection cost on real logits (greedy, the app's default sampler, and top-k off).
    // (ids from the regular vocabulary: Mistral's low ids are unused control tokens with zero embeddings)
    _ = try t.forward((0..<32).map { Int32(1000 + $0 * 37) }, offset: 0)
    var g = SamplingParameters(); g.temperature = 0; g.repetitionPenalty = 1
    var wide = SamplingParameters(); wide.topK = 0
    for p in [g, SamplingParameters(), wide] { print(t.profileSampler(p)) }
    // Decode as generation runs it: token selection on the GPU, the next step queued before the
    // current one is read back. "decode" is greedy; "sampled" uses the app's default sampler.
    var greedyParams = SamplingParameters()
    greedyParams.temperature = 0
    greedyParams.repetitionPenalty = 1
    for (label, params) in [("decode", greedyParams), ("sampled", SamplingParameters())] {
        let sp = SampleParams(params, vocab: t.vocabSize)
        for ctx in [16, 1024] {
            var gpu = 0.0
            let steps = 128
            t.seedSampler(1)
            _ = try t.prefill((0..<ctx).map { Int32($0 % 1000 + 100) }, offset: 0, sample: sp)
            let s = Date()
            var cur = try t.beginDecodeStep(offset: ctx, sample: sp)
            for i in 0..<steps {
                let next = i + 1 < steps ? try t.beginDecodeStep(offset: ctx + i + 1, sample: sp) : nil
                _ = try t.finish(cur)
                gpu += t.lastGPUTime
                if let next { cur = next }
            }
            let dt = Date().timeIntervalSince(s)
            print(String(format: "%@ @%4d: %6.1f tok/s  (wall %.2f ms, gpu %.2f ms per token)", label as NSString, ctx, Double(steps) / dt, dt / Double(steps) * 1000, gpu / Double(steps) * 1000))
            let diag = t.takeSampleDiagnostics()
            if ProcessInfo.processInfo.environment["MLXR_BENCH_SAMPLEDIAG"] != nil, diag.attempts > 0 {
                print(String(format: "  sampler: %.2f attempts and pool %.0f per token on average, at most %d attempts", Double(diag.attempts) / Double(steps + 1), Double(diag.pool) / Double(steps + 1), diag.maxAttempts))
            }
        }
    }
    // Speculative verification cost: one forward over T tokens with logits for every row.
    // At a short context and after a 1900-token prompt (the attention part grows with the context).
    // (A recurrent state cannot rewind to rerun a position: no verification on hybrid models.)
    for ctx in t.config.isHybrid ? [] : [64, 1900] {
        if ctx > 64 { _ = try t.forward((0..<ctx).map { Int32(100 + $0 % 5000) }, offset: 0) }
        for T in 1...Transformer.maxVerifyTokens {
            let steps = 32
            let s = Date()
            var gpu = 0.0
            for i in 0..<steps {
                _ = try t.forward((0..<T).map { Int32(600 + $0 + i) }, offset: ctx, allLogits: true)
                gpu += t.lastGPUTime
            }
            let dt = Date().timeIntervalSince(s) / Double(steps)
            let label = ctx > 64 ? "verify @\(ctx)" : "verify"
            print(String(format: "%@ T=%d: %6.2f ms per pass (%5.2f ms per token, gpu %.2f ms)", label as NSString, T, dt * 1000, dt * 1000 / Double(T), gpu / Double(steps) * 1000))
        }
    }
    for T in [5, 8, 12, 15] {
        let s = Date()
        for i in 0..<8 { _ = try t.forward((0..<T).map { Int32(600 + $0 + i) }, offset: t.config.isHybrid ? 0 : 64) }
        print(String(format: "prefill T=%d: %6.2f ms", T, Date().timeIntervalSince(s) / 8 * 1000))
    }
    if !longBench.contexts.isEmpty { try benchLongContext(t, dir: dir, bench: longBench) }

}

struct LegacyRun {
    var model: String
    var prompt: String
    var maxTokens = 256
    var temp: Float = 0
    var ctx = 4096
    var raw = false
    /// nil: the template's default (Qwen 3.5 0.8B and 2B answer directly, Qwen3 thinks).
    var thinking: Bool?
    var showIds = false
    var draft: String?
    var sparse: String?
    var spec = false
    var system: String?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repeatPenalty: Float?
    var repeatLastN: Int?
    var seed: UInt64?
}

func legacyRun(_ o: LegacyRun) async throws {
    let sparseArg = try sparseSetting(o.sparse)
    let engine = InferenceEngine.shared
    let dir = try resolveModel(o.model)
    let info = try await engine.load(directory: dir, draftDirectory: try draftDirectory(o.draft, target: dir), contextLength: o.ctx, sparse: sparseArg)
    print("Loaded \(info.name) [\(info.modelType), \(info.quantization), \(info.layers) layers, \(info.chatFormat)]\(info.draftName.map { " + draft \($0)" } ?? "") in \(String(format: "%.2f", info.loadSeconds))s")
    print(sparseSummary(info))
    var params = SamplingParameters()
    params.temperature = o.temp
    params.maxTokens = o.maxTokens
    params.repetitionPenalty = 1.0
    params.seed = o.seed ?? 42
    if let v = o.topP { params.topP = v }
    if let v = o.topK { params.topK = v }
    if let v = o.minP { params.minP = v }
    if let v = o.repeatPenalty { params.repetitionPenalty = v }
    if let v = o.repeatLastN { params.repetitionWindow = v }
    let system = o.system.map { [ChatMessage(role: .system, content: $0)] } ?? []
    if o.spec { params.speculative = true }
    let prompt = o.prompt
    let stream = o.raw ? engine.complete(prompt, parameters: params)
        : engine.chat(system + [ChatMessage(role: .user, content: prompt)], parameters: params, enableThinking: o.thinking)
    if o.showIds {
        let p = o.raw ? prompt : engine.renderChat(system + [ChatMessage(role: .user, content: prompt)], enableThinking: o.thinking)
        print("prompt ids:", engine.encode(p))
    }
    for try await ev in stream {
        switch ev {
        case .promptProcessed(let s):
            print(String(format: "[prompt: %d tokens, %.0f tok/s, ttft %.3fs]", s.promptTokens, s.promptTokensPerSecond, s.timeToFirstToken))
        case .text(let t): print(t, terminator: "")
        case .finished(let s):
            let model = s.modelDraftTokens.map { String(format: " (draft model %d/%d)", s.acceptedModelDraftTokens ?? 0, $0) } ?? ""
            print(String(format: "\n[generated %d tokens at %.1f tok/s, stop: %@, drafts accepted %d/%d%@]", s.generatedTokens, s.tokensPerSecond, s.stopReason, s.acceptedDraftTokens ?? 0, s.draftTokens ?? 0, model))
            if let c = s.draftCycles, c > 0, let dt = s.draftSeconds, let vt = s.verifySeconds {
                print(String(format: "[draft model: %d cycles, %.2f tokens/cycle, draft %.1f ms + verify %.1f ms per cycle]", c,
                             Double(s.acceptedModelDraftTokens ?? 0) / Double(c) + 1, dt / Double(c) * 1000, vt / Double(c) * 1000))
            }
        }
    }
}

func legacyServe(model: String, port: UInt16, ctx: Int, lan: Bool, apiKey: String?, draft: String?, sparse: String?) async throws {
    let sparseArg = try sparseSetting(sparse)
    // LAN mode always needs a key: use --api-key / MLXR_API_KEY, or generate one.
    var key = apiKey ?? ProcessInfo.processInfo.environment["MLXR_API_KEY"]
    if lan && key == nil { key = OpenAIServer.generateAPIKey() }
    let dir = try resolveModel(model)
    let info = try await InferenceEngine.shared.load(directory: dir, draftDirectory: try draftDirectory(draft, target: dir), contextLength: ctx, sparse: sparseArg)
    print("Loaded \(info.name)\(info.draftName.map { " + draft \($0)" } ?? "") in \(String(format: "%.2f", info.loadSeconds))s")
    print(sparseSummary(info))
    let server = OpenAIServer(port: port, allowLAN: lan, apiKey: key)
    server.onLog = { print($0) }
    try startServer { try server.start() }
    print("OpenAI-compatible server on http://\(lan ? "0.0.0.0" : "127.0.0.1"):\(port)/v1  (Ctrl-C to stop)")
    if let key { print("API key (send as 'Authorization: Bearer <key>'): \(key)") }
    while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }
}
