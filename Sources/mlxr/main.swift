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
    let details = try await HubClient.details(repo)
    if !details.isSupported { print("warning: \(details.modelType ?? "unknown") architecture is not supported by the engine") }
    print("Downloading \(repo) (\(formatBytes(details.downloadSize)))")
    let dl = ModelDownload(details: details, destination: ModelStore.directory(for: repo))
    let last = LockedDate()
    dl.onProgress = { p, done, total, speed in
        guard last.elapsed(0.25) || p >= 1 else { return }
        print(String(format: "\r  %5.1f%%  %@ / %@  %@/s   ", p * 100, formatBytes(Int(done)), formatBytes(Int(total)), formatBytes(Int(speed))), terminator: "")
    }
    let dir = try await dl.start()
    print("\nSaved to \(dir.path)")

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
            print(String(format: "\n[generated %d tokens at %.1f tok/s, stop: %@]", s.generatedTokens, s.tokensPerSecond, s.stopReason))
        }
    }

case "serve":
    let port = UInt16(option("--port") ?? "1234") ?? 1234
    let ctx = Int(option("--ctx") ?? "8192") ?? 8192
    let lan = flag("--lan")
    guard let model = args.first else { fail("usage: mlxr serve <model> [--port 1234] [--ctx N] [--lan]") }
    let info = try await InferenceEngine.shared.load(directory: resolve(model), contextLength: ctx)
    print("Loaded \(info.name) in \(String(format: "%.2f", info.loadSeconds))s")
    let server = OpenAIServer(port: port, allowLAN: lan)
    server.onLog = { print($0) }
    try server.start()
    print("OpenAI-compatible server on http://\(lan ? "0.0.0.0" : "127.0.0.1"):\(port)/v1  (Ctrl-C to stop)")
    while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }

case "bench":
    guard let model = args.first else { fail("usage: mlxr bench <model>") }
    let dir = resolve(model)
    let t = try Transformer(directory: dir, contextLength: 4096)
    print("weights: \(formatBytes(t.weightBytes))")
    print(try t.profile())
    _ = try t.forward([1], offset: 0)
    for n in [128, 512, 2048] {
        let toks = (0..<n).map { Int32($0 % 1000 + 100) }
        let s = Date()
        _ = try t.forward(toks, offset: 0)
        let dt = Date().timeIntervalSince(s)
        print(String(format: "prefill %5d tokens: %7.1f tok/s", n, Double(n) / dt))
    }
    for ctx in [16, 1024] {
        var gpu = 0.0
        let steps = 128
        let s = Date()
        for i in 0..<steps { _ = try t.forward([Int32(500 + i)], offset: ctx + i); gpu += t.lastGPUTime }
        let dt = Date().timeIntervalSince(s)
        print(String(format: "decode @%4d: %6.1f tok/s  (wall %.2f ms, gpu %.2f ms per token)", ctx, Double(steps) / dt, dt / Double(steps) * 1000, gpu / Double(steps) * 1000))
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
