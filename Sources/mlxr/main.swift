import Foundation
import MLXEngine

// mlxr — command-line companion to MLX Runner.
//   mlxr search <query>
//   mlxr download <org/repo>
//   mlxr list
//   mlxr run <org/repo|path> [-p prompt] [--raw] [--max N] [--temp T] [--no-think] [--ctx N] [--draft <model|auto>] [--sparse <on|off|N>]
//   mlxr tokenize <org/repo|path> <text>
//   mlxr serve <org/repo|path> [--port 1234] [--ctx N] [--lan] [--draft <model|auto>] [--sparse <on|off|N>]

setvbuf(stdout, nil, _IONBF, 0)
var args = Array(CommandLine.arguments.dropFirst())

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]
    args.removeSubrange(i...(i + 1))
    return v
}

func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}

func resolve(_ s: String) -> URL {
    if FileManager.default.fileExists(atPath: s) { return URL(fileURLWithPath: s) }
    if let m = ModelStore.scan().first(where: { $0.repo == s || $0.name == s }) { return m.directory }
    fail("model not found: \(s) (try `mlxr download \(s)`)")
}

/// `--draft <model|auto>`: the draft model for speculative decoding.
func draftDirectory(_ s: String?, target: URL) -> URL? {
    guard let s else { return nil }
    guard s == "auto" else { return resolve(s) }
    guard let m = InferenceEngine.autoDraft(for: target) else {
        print("no installed draft model pairs with \(target.lastPathComponent); running without one")
        return nil
    }
    return m.directory
}

/// `--sparse <on|off|N>`: sparse decode attention for long contexts (N: long-range keys; the window,
/// minimum context and shared selection still come from MLXR_SPARSE_*). Without the
/// option: MLXR_SPARSE_ATTN, which defaults to on.
func sparseOption(_ s: String?) -> SparseAttention? {
    guard let s else { return SparseAttention.fromEnvironment }
    switch s.lowercased() {
    case "on", "1": return .configured(topK: SparseAttention.standard.topK)
    case "off", "0": return nil
    default:
        guard let n = Int(s), n > 0 else { fail("--sparse takes on, off or a number of long-range keys") }
        return .configured(topK: n)
    }
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
        guard let s = try? String(contentsOfFile: path, encoding: .utf8) else { fail("cannot read MLXR_BENCH_TEXT \(path)") }
        text = tokenizer.encode(s).map { Int32($0) }.filter { $0 >= 0 && Int($0) < V }
        guard !text.isEmpty else { fail("MLXR_BENCH_TEXT \(path) has no tokens") }
        let bos = tokenizer.bosToken.flatMap { b in tokenizer.tokenId(b).map { " (prompts start with \(b), id \($0))" } } ?? ""
        print("long-context prompt: \(path), \(text.count) tokens (repeated to length)\(bos)")
    } else {
        print("long-context prompt: pseudo-random tokens (MLXR_BENCH_TEXT=<file.txt> for real text)")
    }
    let original = t.sparse
    defer { t.sparse = original }
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
            print(String(format: "prefill @%d%@: %.1f s (%.0f tok/s)", ctx, at as NSString, prefillTime, Double(ctx) / prefillTime))
            // Each run decodes from the same prompt (positions from ctx on are rewritten, the index with them).
            var dense: [Int] = []
            var modes: [(String, SparseAttention?)] = [("dense", nil)]
            if let sparse, oi == 0 { modes.append(("sparse", sparse)) }
            for (label, mode) in modes {
                t.sparse = mode
                t.seedSampler(1)
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
            for i in 0..<steps {
                t.sparse = sparse
                if argmax(try logits([input], offset: ctx + i)) == dense[i] { match += 1 }
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
                for i in 0..<N {
                    let input = text[(off + ctx + i) % text.count], target = Int(text[(off + ctx + i + 1) % text.count])
                    t.sparse = sparse
                    let ls = try logits([input], offset: ctx + i)
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
            if oi == 0 {
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

guard let cmd = args.first else {
    print("usage: mlxr <search|download|list|run|tokenize|serve> ...")
    exit(0)
}
args.removeFirst()

switch cmd {
case "search":
    let results = try await HubClient.search(args.joined(separator: " "), limit: 25)
    for m in results {
        print(String(format: "%-60@ %10d ↓  %@", m.id as NSString, m.downloads ?? 0, (m.modelTypeTag ?? "") as NSString))
    }

case "download":
    guard let repo = args.first else { fail("usage: mlxr download <org/repo>") }
    guard RepoID.isValid(repo) else { fail("invalid repo id: \(repo) (expected owner/name)") }
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
    do {
        let dir = try await dl.start()
        print("\nSaved to \(dir.path)")
    } catch {
        fail("\ndownload failed: \(error.localizedDescription)")
    }

case "list":
    for m in ModelStore.scan() {
        print(String(format: "%-60@ %10@  %@%@", m.repo as NSString, formatBytes(m.sizeBytes) as NSString,
                     (m.modelType ?? "?") as NSString, m.isSupported ? "" : " (unsupported)"))
    }

case "tokenize":
    guard args.count >= 2 else { fail("usage: mlxr tokenize <model> <text>") }
    let tok = try Tokenizer(directory: resolve(args[0]))
    let ids = tok.encode(args[1...].joined(separator: " "))
    print(ids)
    print(ids.map { tok.token($0) ?? "?" })
    print("roundtrip:", tok.decode(ids, includeSpecial: true))

case "run":
    let prompt = option("-p") ?? "Write a haiku about Apple Silicon."
    let maxTokens = Int(option("--max") ?? "256") ?? 256
    let temp = Float(option("--temp") ?? "0") ?? 0
    let ctx = Int(option("--ctx") ?? "4096") ?? 4096
    let raw = flag("--raw")
    let noThink = flag("--no-think")
    let showIds = flag("--ids")
    let draftArg = option("--draft")
    let sparseArg = sparseOption(option("--sparse"))
    guard let model = args.first else { fail("usage: mlxr run <model> [-p prompt]") }
    let engine = InferenceEngine.shared
    let dir = resolve(model)
    let info = try await engine.load(directory: dir, draftDirectory: draftDirectory(draftArg, target: dir), contextLength: ctx, sparse: sparseArg)
    print("Loaded \(info.name) [\(info.modelType), \(info.quantization), \(info.layers) layers, \(info.chatFormat)]\(info.draftName.map { " + draft \($0)" } ?? "") in \(String(format: "%.2f", info.loadSeconds))s")
    print(sparseSummary(info))
    var params = SamplingParameters()
    params.temperature = temp
    params.maxTokens = maxTokens
    params.repetitionPenalty = 1.0
    params.seed = 42
    if flag("--spec") { params.speculative = true }
    let stream = raw ? engine.complete(prompt, parameters: params)
        : engine.chat([ChatMessage(role: .user, content: prompt)], parameters: params, enableThinking: !noThink)
    if showIds {
        let p = raw ? prompt : engine.renderChat([ChatMessage(role: .user, content: prompt)], enableThinking: !noThink)
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

case "serve":
    let port = UInt16(option("--port") ?? "1234") ?? 1234
    let ctx = Int(option("--ctx") ?? "8192") ?? 8192
    let lan = flag("--lan")
    let draftArg = option("--draft")
    let sparseArg = sparseOption(option("--sparse"))
    guard let model = args.first else { fail("usage: mlxr serve <model> [--port 1234] [--ctx N] [--lan] [--api-key KEY] [--draft <model|auto>] [--sparse <on|off|N>]") }
    // LAN mode always needs a key: use --api-key / MLXR_API_KEY, or generate one.
    var key = option("--api-key") ?? ProcessInfo.processInfo.environment["MLXR_API_KEY"]
    if lan && key == nil { key = OpenAIServer.generateAPIKey() }
    let dir = resolve(model)
    let info = try await InferenceEngine.shared.load(directory: dir, draftDirectory: draftDirectory(draftArg, target: dir), contextLength: ctx, sparse: sparseArg)
    print("Loaded \(info.name)\(info.draftName.map { " + draft \($0)" } ?? "") in \(String(format: "%.2f", info.loadSeconds))s")
    print(sparseSummary(info))
    let server = OpenAIServer(port: port, allowLAN: lan, apiKey: key)
    server.onLog = { print($0) }
    do { try server.start() } catch { fail("cannot start server: \(error.localizedDescription)") }
    print("OpenAI-compatible server on http://\(lan ? "0.0.0.0" : "127.0.0.1"):\(port)/v1  (Ctrl-C to stop)")
    if let key { print("API key (send as 'Authorization: Bearer <key>'): \(key)") }
    while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }

case "bench":
    guard let model = args.first else { fail("usage: mlxr bench <model>") }
    let dir = resolve(model)
    // Long-context decode (opt-in): MLXR_BENCH_CTX=8192,32768 adds a section at the end.
    let longBench = LongBench(env: ProcessInfo.processInfo.environment)
    let t = try Transformer(directory: dir, contextLength: max(4096, (longBench.contexts.max() ?? 0) + longBench.margin + 64))
    print("weights: \(formatBytes(t.weightBytes))")
    if let sp = t.sparse {
        print(t.sparseAvailable
              ? "sparse decode attention: top-k \(sp.topK), window \(sp.window), sink \(sp.sink), from \(sp.minContext) tokens"
              : "sparse decode attention: off (\(t.sparseOffReason ?? "unavailable"))")
    }
    let quick = ProcessInfo.processInfo.environment["MLXR_BENCH_DECODE"] != nil
    let noProfile = quick || ProcessInfo.processInfo.environment["MLXR_BENCH_NOPROFILE"] != nil
    if !noProfile { print(try t.profile()) }
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
        }
    }
    // Speculative verification cost: one forward over T tokens with logits for every row.
    // At a short context and after a 1900-token prompt (the attention part grows with the context).
    for ctx in [64, 1900] {
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
        for i in 0..<8 { _ = try t.forward((0..<T).map { Int32(600 + $0 + i) }, offset: 64) }
        print(String(format: "prefill T=%d: %6.2f ms", T, Date().timeIntervalSince(s) / 8 * 1000))
    }
    if !longBench.contexts.isEmpty { try benchLongContext(t, dir: dir, bench: longBench) }

default:
    fail("unknown command \(cmd)")
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
