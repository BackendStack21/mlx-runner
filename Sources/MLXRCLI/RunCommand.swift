import Foundation
import MLXEngine

/// What `run` reaches outside the process.
struct RunDependencies {
    var resolver: ModelResolver
    var hub: HubServices
    var engine: GenerationEngine
    var interrupts: Interrupts
}

extension ModelResolver {
    /// The resolver over the real model library and presets.
    static var live: ModelResolver { ModelResolver(library: .live, presets: .live) }
}

/// The directory of an installed model, by path, repo id, `hf.co/org/repo`, catalog alias, preset or
/// short name (the SL03 resolver). A model that resolves but is not installed is an error that
/// names `mlxr pull`. Used by serve, tokenize, bench and `--draft`.
func installedModelDirectory(_ name: String, resolver: ModelResolver) throws -> URL {
    let r = try resolver.resolve(name)
    guard let dir = r.directory else {
        throw CLIError.failure("model '\(name)' is not installed (pull it with 'mlxr pull \(name)')")
    }
    return dir
}

extension CLI {
    /// Options Ollama has that the engine cannot honor: clear errors before anything loads.
    static let unsupportedRunFlags: [(flag: String, why: String)] = [
        ("format", "is not supported by the engine"),
        ("presence-penalty", "is not supported by the engine"),
        ("frequency-penalty", "is not supported by the engine"),
        ("truncate", "is not supported: mlxr runs text-generation models only"),
        ("dimensions", "is not supported: mlxr runs text-generation models only"),
        ("voice", "is not supported: mlxr runs text-generation models only"),
        ("ask", "is not supported: mlxr runs text-generation models only"),
        ("choice", "is not supported: mlxr runs text-generation models only"),
        ("level", "is not supported: mlxr runs text-generation models only"),
    ]

    /// The model directory for `run`: installed, or pulled first (progress on stderr) when the name
    /// resolves to a Hugging Face repo whose architecture the engine supports.
    private static func prepareModel(_ name: String, console: Console, deps: RunDependencies) async throws -> URL {
        let r = try deps.resolver.resolve(name)
        if let dir = r.directory { return dir }
        guard let repo = r.repo else { throw CLIError.failure("model '\(name)' not found") }
        let hub = deps.hub
        let details = try await hub.details(repo)
        if !SupportedArchitectures.isSupported(details.modelType) || details.isMoE {
            throw CLIError.failure("\(details.modelType ?? "unknown") architecture is not supported by the engine (\(repo))")
        }
        guard details.isSupported else {
            throw CLIError.failure("\(repo) has no tokenizer.json or safetensors weights")
        }
        console.err("pulling manifest\n")
        let destination = try hub.destination(repo)
        let renderer = PullRenderer(console: console)
        let hooks = DownloadHooks(progress: { renderer.update(done: $0, total: $1, speed: $2) }, event: { _ in })
        let task = Task { try await performDownload(start: { try await hub.download(details, destination, hooks) }) }
        deps.interrupts.enter(.downloading, onCancel: { task.cancel() })
        let dir: URL
        do { dir = try await task.value } catch {
            renderer.finish()
            if deps.interrupts.wasInterrupted || error is CancellationError { throw CLIError.interrupted }
            throw error
        }
        renderer.finish()
        console.err("verifying sha256 digest\nwriting manifest\nsuccess\n")
        return dir
    }

    /// The body of `mlxr run`.
    static func executeRun(_ inv: Invocation, _ console: Console, _ deps: RunDependencies) async throws {
        for (flag, why) in unsupportedRunFlags where inv.has(flag) {
            throw CLIError.failure("--\(flag) \(why)")
        }
        let note = "has no effect: mlxr runs models in-process, without a server"
        if inv.has("keepalive") { console.err("warning: --keepalive \(note)\n") }
        if inv.bool("insecure") { console.err("warning: --insecure has no effect: mlxr always verifies Hugging Face downloads\n") }
        let sparse = try sparseSetting(inv.string("sparse"))
        let words = Array(inv.positionals.dropFirst())
        let stdin = console.stdinIsTTY ? "" : console.readAllStdin()
        let input = try assembleRunInput(stdin: stdin, words: words, promptFlag: inv.string("prompt"),
                                         stdinIsTTY: console.stdinIsTTY, stdoutIsTTY: console.stdoutIsTTY)
        guard case .oneShot(let prompt) = input else {
            throw CLIError.failure("interactive chat is not available yet: give a prompt, e.g. mlxr run \(inv.positionals[0]) \"Hello\"")
        }
        let interrupts = deps.interrupts, engine = deps.engine
        let dir = try await prepareModel(inv.positionals[0], console: console, deps: deps)
        if interrupts.wasInterrupted { throw CLIError.interrupted }

        // Sampling: model defaults (validated), else the app's, then the flags.
        let defaults = GenerationDefaults.load(directory: dir)
        defaults.warnings.forEach { console.err($0 + "\n") }
        var params = SamplingParameters()
        if let v = defaults.temperature { params.temperature = v }
        if let v = defaults.topP { params.topP = v }
        if let v = defaults.topK { params.topK = v }
        if let v = defaults.minP { params.minP = v }
        if let v = defaults.repetitionPenalty { params.repetitionPenalty = v }
        params.maxTokens = inv.int("max-tokens") ?? SamplingLimits.maxTokens
        if let v = inv.double("temp") { params.temperature = Float(v) }
        if let v = inv.double("top-p") { params.topP = Float(v) }
        if let v = inv.int("top-k") { params.topK = v }
        if let v = inv.double("min-p") { params.minP = Float(v) }
        if let v = inv.double("repeat-penalty") { params.repetitionPenalty = Float(v) }
        if let v = inv.int("repeat-last-n") { params.repetitionWindow = v }
        if let v = inv.string("seed") { params.seed = UInt64(v) }
        if inv.bool("spec") { params.speculative = true }

        var draftDir: URL?
        if let d = inv.string("draft") {
            if d == "auto" {
                draftDir = engine.autoDraft(for: dir)?.directory
                if draftDir == nil { console.err("no installed draft model pairs with \(dir.lastPathComponent); running without one\n") }
            } else {
                draftDir = try installedModelDirectory(d, resolver: deps.resolver)
            }
        }

        interrupts.enter(.loading)
        let info = try await engine.load(directory: dir, draftDirectory: draftDir, contextLength: inv.int("ctx") ?? 4096, sparse: sparse)
        if interrupts.wasInterrupted { throw CLIError.interrupted }
        console.err("Loaded \(info.name) [\(info.modelType), \(info.quantization), \(info.layers) layers, \(info.chatFormat)]"
                    + "\(info.draftName.map { " + draft \($0)" } ?? "") in \(String(format: "%.2f", info.loadSeconds))s\n")
        console.err(sparseSummary(info) + "\n")
        if prompt.isEmpty { interrupts.enter(.idle); return }

        let thinking: Bool? = inv.bool("no-think") ? false : inv.has("think") ? inv.string("think") != "false" : nil
        let raw = inv.bool("raw")
        let messages = (inv.string("system").map { [ChatMessage(role: .system, content: $0)] } ?? [])
            + [ChatMessage(role: .user, content: prompt)]
        if inv.bool("ids") {
            console.err("prompt ids: \(engine.encode(raw ? prompt : engine.renderChat(messages, enableThinking: thinking)))\n")
        }
        let token = CancelToken()
        interrupts.enter(.generating, onCancel: { token.cancel() })
        let stream = raw ? engine.complete(prompt, parameters: params, token: token)
            : engine.chat(messages, parameters: params, enableThinking: thinking, token: token)
        let renderer = ResponseRenderer(console: console, hideThinking: inv.bool("hidethinking"), wordWrap: !inv.bool("nowordwrap"))
        var splitter = ReasoningSplitter(enabled: !raw && thinking != false)
        var promptStats: GenerationStats?
        var final: GenerationStats?
        func emit(_ parts: (reasoning: String, content: String)) {
            renderer.reasoning(parts.reasoning)
            renderer.content(parts.content)
        }
        do {
            for try await ev in stream {
                switch ev {
                case .promptProcessed(let s): promptStats = s
                case .text(let t): emit(splitter.feed(t))
                case .finished(let s): final = s
                }
            }
        } catch {
            interrupts.enter(.idle)
            if token.isCancelled || interrupts.wasInterrupted { throw CLIError.interrupted }
            throw error
        }
        interrupts.enter(.idle)
        emit(splitter.flush())
        if token.isCancelled || interrupts.wasInterrupted || final?.stopReason == "cancelled" { throw CLIError.interrupted }
        console.out("\n\n")
        if inv.bool("verbose"), let final {
            console.err(StatsFormat.block(RunStats(loadSeconds: info.loadSeconds, stats: final)))
            StatsFormat.mlxrLines(prompt: promptStats, final: final).forEach { console.err($0 + "\n") }
        }
    }
}
