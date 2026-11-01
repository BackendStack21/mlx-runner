import Foundation
import MLXEngine

// mlxr — command-line companion to MLX Runner.
//   mlxr search <query>
//   mlxr download <org/repo>
//   mlxr list
//   mlxr run <org/repo|path> [-p prompt] [--raw] [--max N] [--temp T] [--no-think] [--ctx N]
//   mlxr tokenize <org/repo|path> <text>
//   mlxr serve <org/repo|path> [--port 1234] [--ctx N] [--lan]

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

func formatBytes(_ b: Int) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file)
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
    guard let model = args.first else { fail("usage: mlxr run <model> [-p prompt]") }
    let engine = InferenceEngine.shared
    let info = try await engine.load(directory: resolve(model), contextLength: ctx)
    print("Loaded \(info.name) [\(info.modelType), \(info.quantization), \(info.layers) layers, \(info.chatFormat)] in \(String(format: "%.2f", info.loadSeconds))s")
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
            print(String(format: "\n[generated %d tokens at %.1f tok/s, stop: %@, drafts accepted %d/%d]", s.generatedTokens, s.tokensPerSecond, s.stopReason, s.acceptedDraftTokens ?? 0, s.draftTokens ?? 0))
        }
    }

case "serve":
    let port = UInt16(option("--port") ?? "1234") ?? 1234
    let ctx = Int(option("--ctx") ?? "8192") ?? 8192
    let lan = flag("--lan")
    guard let model = args.first else { fail("usage: mlxr serve <model> [--port 1234] [--ctx N] [--lan] [--api-key KEY]") }
    // LAN mode always needs a key: use --api-key / MLXR_API_KEY, or generate one.
    var key = option("--api-key") ?? ProcessInfo.processInfo.environment["MLXR_API_KEY"]
    if lan && key == nil { key = OpenAIServer.generateAPIKey() }
    let info = try await InferenceEngine.shared.load(directory: resolve(model), contextLength: ctx)
    print("Loaded \(info.name) in \(String(format: "%.2f", info.loadSeconds))s")
    let server = OpenAIServer(port: port, allowLAN: lan, apiKey: key)
    server.onLog = { print($0) }
    do { try server.start() } catch { fail("cannot start server: \(error.localizedDescription)") }
    print("OpenAI-compatible server on http://\(lan ? "0.0.0.0" : "127.0.0.1"):\(port)/v1  (Ctrl-C to stop)")
    if let key { print("API key (send as 'Authorization: Bearer <key>'): \(key)") }
    while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }

case "bench":
    guard let model = args.first else { fail("usage: mlxr bench <model>") }
    let dir = resolve(model)
    let t = try Transformer(directory: dir, contextLength: 4096)
    print("weights: \(formatBytes(t.weightBytes))")
    let quick = ProcessInfo.processInfo.environment["MLXR_BENCH_DECODE"] != nil
    if !quick { print(try t.profile()) }
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
    for T in 1...Transformer.maxVerifyTokens {
        let steps = 32
        let s = Date()
        for i in 0..<steps { _ = try t.forward((0..<T).map { Int32(600 + $0 + i) }, offset: 64, allLogits: true) }
        let dt = Date().timeIntervalSince(s) / Double(steps)
        print(String(format: "verify T=%d: %6.2f ms per pass (%5.2f ms per token)", T, dt * 1000, dt * 1000 / Double(T)))
    }
    for T in [5, 8, 12, 15] {
        let s = Date()
        for i in 0..<8 { _ = try t.forward((0..<T).map { Int32(600 + $0 + i) }, offset: 64) }
        print(String(format: "prefill T=%d: %6.2f ms", T, Date().timeIntervalSince(s) / 8 * 1000))
    }

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
