import Foundation
import MLXEngine

/// What the chat prompt returned.
public enum LineEvent: Equatable, Sendable {
    /// A line of input (without the newline).
    case line(String)
    /// Ctrl-C at the prompt.
    case interrupt
    /// Ctrl-D on an empty line, or the end of piped input.
    case eof
}

/// Where the chat loop reads lines from: libedit on a terminal, `readLine()` otherwise, a script in tests.
public protocol LineSource: AnyObject {
    /// Shows `prompt` (`>>> ` or `... `) and returns the next event.
    func read(prompt: String) -> LineEvent
    /// Ctrl-C while the chat waits for input: makes a `read` in progress return `.interrupt`, from any
    /// thread. Nil when the source cannot be woken: Ctrl-C at its prompt then ends the process (130).
    var interrupter: (@Sendable () -> Void)? { get }
}

extension LineSource {
    public var interrupter: (@Sendable () -> Void)? { nil }
}

/// The `readLine()` fallback used when stdin or stdout is not a terminal: each line is one event,
/// the end of input is `.eof`. Writes nothing to stdout. Ctrl-C at its prompt ends the process.
public final class ConsoleLineSource: LineSource {
    private let console: Console
    public init(console: Console) { self.console = console }
    public func read(prompt: String) -> LineEvent {
        console.readLine().map { .line($0) } ?? .eof
    }
}

/// The chat input history: one line per entry in a file under Application Support/MLX Runner/.
public struct ChatHistory: Sendable {
    /// The history file; nil keeps no history.
    public var file: URL?
    public init(file: URL?) { self.file = file }

    private static let sizeLimit = 4 << 20

    /// `Application Support/MLX Runner/history`.
    public static var live: ChatHistory {
        ChatHistory(file: ModelStore.root.deletingLastPathComponent().appendingPathComponent("history"))
    }

    /// Keeps the file bounded: past the cap, rewrites it with the newest lines (about half the cap).
    private func trimIfOversized(_ file: URL) {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: file.path)[.size] as? NSNumber)?.intValue,
              size > Self.sizeLimit else { return }
        let kept = tail(file, keep: Self.sizeLimit / 2).joined(separator: "\n") + "\n"
        let tmp = file.deletingLastPathComponent().appendingPathComponent(".history-\(UUID().uuidString)")
        guard FileManager.default.createFile(atPath: tmp.path, contents: Data(kept.utf8), attributes: [.posixPermissions: 0o600]) else { return }
        if (try? FileManager.default.replaceItemAt(file, withItemAt: tmp)) == nil { try? FileManager.default.removeItem(at: tmp) }
    }

    /// The newest lines of the file (at most the last `sizeLimit` bytes), oldest first.
    private func tail(_ file: URL, keep: Int) -> [String] {
        guard let handle = try? FileHandle(forReadingFrom: file) else { return [] }
        defer { try? handle.close() }
        guard let end = try? handle.seekToEnd() else { return [] }
        let start = end > UInt64(keep) ? end - UInt64(keep) : 0
        guard (try? handle.seek(toOffset: start)) != nil, let data = try? handle.readToEnd() else { return [] }
        var lines = String(decoding: data, as: UTF8.self).split(separator: "\n").map(String.init)
        if start > 0, !lines.isEmpty { lines.removeFirst() }  // a cut-off first line
        return lines
    }

    /// The saved lines, oldest first (the newest ones when the file is larger than the cap).
    public func lines() -> [String] {
        guard let file else { return [] }
        return tail(file, keep: Self.sizeLimit)
    }

    /// Adds a line (creates the file with mode 0600). Failures are ignored: history is a convenience.
    public func append(_ line: String) {
        guard let file else { return }
        let fm = FileManager.default
        try? fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fm.fileExists(atPath: file.path) {
            fm.createFile(atPath: file.path, contents: nil, attributes: [.posixPermissions: 0o600])
        }
        guard let handle = try? FileHandle(forWritingTo: file) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        let flat = line.replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "\r", with: " ")
        try? handle.write(contentsOf: Data((flat + "\n").utf8))
        try? handle.close()
        trimIfOversized(file)
    }
}

/// The state a chat starts with: the loaded model, the sampling parameters and the display flags.
public struct ChatSettings {
    /// The model name as typed.
    public var model: String
    public var repo: String?
    public var directory: URL
    public var info: LoadedModelInfo
    public var preset: Preset?
    public var parameters = SamplingParameters()
    /// Parameters set by the user (flags, `/set parameter`, a preset), by Ollama key, as typed.
    public var userParameters: [String: String] = [:]
    public var system: String?
    /// nil: the template's default.
    public var thinking: Bool?
    public var hideThinking = false
    public var wordWrap = true
    public var verbose = false
    public var contextLength = 4096
    public var draftDirectory: URL?
    public var sparse: SparseAttention?

    public init(model: String, repo: String?, directory: URL, info: LoadedModelInfo, preset: Preset? = nil) {
        self.model = model; self.repo = repo; self.directory = directory; self.info = info; self.preset = preset
    }
}

/// `/set parameter` keys: Ollama's, with what the engine does with each.
enum ChatParameter {
    static let noEffect: Set<String> = ["num_gpu", "main_gpu", "use_mmap", "num_thread", "num_batch",
                                        "typical_p", "num_keep", "stop", "draft_num_predict"]
    static let unsupported: Set<String> = ["presence_penalty", "frequency_penalty"]

    enum Outcome { case set, ignored }

    /// Validates `value` against the shared limits and applies it; throws the reason.
    static func apply(_ key: String, _ value: String, to p: inout SamplingParameters) throws -> Outcome {
        func real(_ range: ClosedRange<Double>) throws -> Float {
            guard let v = Double(value), v.isFinite, range.contains(v) else {
                throw CLIError.failure("invalid value '\(value)' for \(key) (\(range.lowerBound) to \(range.upperBound))")
            }
            return Float(v)
        }
        func int(_ range: ClosedRange<Int>) throws -> Int {
            guard let v = Int(value), range.contains(v) else {
                throw CLIError.failure("invalid value '\(value)' for \(key) (\(range.lowerBound) to \(range.upperBound))")
            }
            return v
        }
        switch key {
        case "temperature": p.temperature = try real(SamplingLimits.temperature)
        case "top_k": p.topK = try int(SamplingLimits.topK)
        case "min_p": p.minP = try real(SamplingLimits.minP)
        case "top_p":
            guard let v = Double(value), v.isFinite, SamplingLimits.isValidTopP(v) else {
                throw CLIError.failure("invalid value '\(value)' for top_p (above 0, up to 1)")
            }
            p.topP = Float(v)
        case "repeat_penalty":
            guard let v = Double(value), v.isFinite, SamplingLimits.isValidRepeatPenalty(v) else {
                throw CLIError.failure("invalid value '\(value)' for repeat_penalty (above 0, up to 100)")
            }
            p.repetitionPenalty = Float(v)
        case "repeat_last_n": p.repetitionWindow = try int(SamplingLimits.repeatLastN)
        case "num_predict": p.maxTokens = try int(1...SamplingLimits.maxTokens)
        case "seed":
            guard let v = UInt64(value) else { throw CLIError.failure("invalid value '\(value)' for seed") }
            p.seed = v
        case "num_ctx": _ = try int(SamplingLimits.context)
        case _ where noEffect.contains(key): return .ignored
        case _ where unsupported.contains(key): throw CLIError.failure("\(key) is not supported by the engine")
        default: throw CLIError.failure("unknown parameter '\(key)'")
        }
        return .set
    }
}

/// The interactive chat loop (`mlxr run MODEL` on a terminal).
public final class ChatSession {
    private let console: Console
    private let engine: GenerationEngine
    private let resolver: ModelResolver
    private let presets: PresetStore
    private let interrupts: Interrupts
    private let lines: LineSource
    private let history: ChatHistory
    private var settings: ChatSettings
    private var messages: [ChatMessage] = []
    private var recordHistory: Bool
    private var reachedEnd = false
    /// True once the chat ended because the engine could not load a model (already reported).
    public private(set) var failed = false

    public init(console: Console, engine: GenerationEngine, resolver: ModelResolver, presets: PresetStore,
                interrupts: Interrupts, lines: LineSource, history: ChatHistory, settings: ChatSettings) {
        self.console = console; self.engine = engine; self.resolver = resolver; self.presets = presets
        self.interrupts = interrupts; self.lines = lines; self.history = history; self.settings = settings
        let flag = (console.env["OLLAMA_NOHISTORY"] ?? "").lowercased()
        recordHistory = flag.isEmpty || flag == "0" || flag == "false"
    }

    private func say(_ s: String) { console.err(s + "\n") }

    /// Between turns: Ctrl-C wakes the prompt when the line source can show it (Ollama's "Use Ctrl + d
    /// or /bye to exit."), and ends the process otherwise.
    private func awaitInput() {
        if let wake = lines.interrupter { interrupts.enter(.prompt, onCancel: wake) } else { interrupts.enter(.idle) }
    }

    /// Reads and handles input until `/bye`, `/exit` or Ctrl-D; returns the exit code.
    @discardableResult
    public func run() async -> Int32 {
        awaitInput()
        while true {
            switch lines.read(prompt: ">>> ") {
            case .eof: return 0
            case .interrupt: say("\nUse Ctrl + d or /bye to exit.")
            case .line(let raw):
                let line = raw.trimmingCharacters(in: .whitespaces)
                if line.isEmpty { continue }
                if recordHistory { history.append(raw) }
                if line.hasPrefix("\"\"\"") {
                    guard let text = gather(line) else { if reachedEnd { return 0 }; continue }
                    await send(text)
                } else if line.hasPrefix("/") {
                    if let code = await command(line) { return code }
                } else {
                    await send(line)
                }
            }
            if reachedEnd { return 0 }
        }
    }

    // MARK: input

    /// A triple-quoted message: `first` starts with `"""`; lines join with newlines until one ends with
    /// `"""`. Ctrl-C discards it (nil); end of input discards it and ends the chat.
    private func gather(_ first: String) -> String? {
        let q = "\"\"\""
        var rest = String(first.dropFirst(3))
        var parts: [String] = []
        if rest.hasSuffix(q) { return String(rest.dropLast(3)) }
        if !rest.isEmpty { parts.append(rest) }
        while true {
            switch lines.read(prompt: "... ") {
            case .interrupt: return nil
            case .eof: reachedEnd = true; return nil
            case .line(let l):
                rest = l
                if rest.hasSuffix(q) {
                    let before = String(rest.dropLast(3))
                    if !before.isEmpty || !parts.isEmpty { parts.append(before) }
                    return parts.joined(separator: "\n")
                }
                parts.append(rest)
            }
        }
    }

    // MARK: a turn

    private func send(_ text: String) async {
        messages.append(ChatMessage(role: .user, content: text))
        let request = (settings.system.map { [ChatMessage(role: .system, content: $0)] } ?? []) + messages
        let token = CancelToken()
        let userCancel = UserCancel()
        interrupts.enter(.generating, onCancel: { userCancel.mark(); token.cancel() })
        let stream = engine.chat(request, parameters: settings.parameters, enableThinking: settings.thinking, token: token)
        let renderer = ResponseRenderer(console: console, hideThinking: settings.hideThinking, wordWrap: settings.wordWrap)
        var splitter = ReasoningSplitter(enabled: settings.thinking != false)
        var answer = ""
        var promptStats: GenerationStats?
        var final: GenerationStats?
        func emit(_ parts: (reasoning: String, content: String)) {
            renderer.reasoning(parts.reasoning)
            renderer.content(parts.content)
            answer += parts.content
        }
        var failure: Error?
        do {
            for try await ev in stream {
                switch ev {
                case .promptProcessed(let s): promptStats = s
                case .text(let t): emit(splitter.feed(t))
                case .finished(let s): final = s
                }
            }
        } catch { failure = error }
        awaitInput()
        let cancelled = userCancel.isSet || final?.stopReason == "cancelled"
        if failure == nil && !cancelled { emit(splitter.flush()) }
        if cancelled || failure != nil {
            if !cancelled { messages.removeLast() }  // a cancelled turn keeps the user message
            console.out("\n")
            if !cancelled, let failure { console.err("Error: \(CLI.message(for: failure))\n") }
            return
        }
        messages.append(ChatMessage(role: .assistant, content: answer))
        console.out("\n\n")
        if settings.verbose, let final {
            console.err(StatsFormat.block(RunStats(loadSeconds: 0, stats: final)))
            StatsFormat.mlxrLines(prompt: promptStats, final: final).forEach { console.err($0 + "\n") }
        }
    }

    // MARK: slash commands

    private static let help = """
    Available Commands:
      /set            Set session variables
      /show           Show model information
      /load <model>   Load a session or model
      /save <model>   Save your current session
      /clear          Clear session context
      /bye            Exit
      /?, /help       Help for a command
      /? shortcuts    Help for keyboard shortcuts

    Use \"\"\" to begin a multi-line message.


    """

    private static let shortcuts = """
    Available Keyboard Shortcuts:
      Ctrl + a            Move to the beginning of the line (Home)
      Ctrl + e            Move to the end of the line (End)
      Alt + b             Move back (left) one word
      Alt + f             Move forward (right) one word
      Ctrl + k            Delete the sentence after the cursor
      Ctrl + u            Delete the sentence before the cursor
      Ctrl + w            Delete the word before the cursor
      Ctrl + l            Clear the screen
      Ctrl + c            Stop the model from responding
      Ctrl + d            Exit mlxr (/bye)


    """

    private func unknown(_ text: String) { say("Unknown command '\(text)'. Type /? for help") }

    /// Handles one slash command; returns an exit code when the chat ends.
    private func command(_ line: String) async -> Int32? {
        let words = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        let name = words[0]
        let args = Array(words.dropFirst())
        switch name {
        case "/?", "/help":
            console.err(args.first == "shortcuts" ? Self.shortcuts : Self.help)
        case "/bye", "/exit": return 0
        case "/clear":
            messages = []
            say("Cleared session context")
        case "/set": return await set(line, args)
        case "/show": show(args)
        case "/load": return await load(args.joined(separator: " "))
        case "/save": save(args.first)
        default: unknown(name)
        }
        return nil
    }

    private func set(_ line: String, _ args: [String]) async -> Int32? {
        guard let sub = args.first else {
            say("Usage:\n  /set parameter ...\n  /set system <string>\n  /set history|nohistory\n  /set wordwrap|nowordwrap\n  /set format json\n  /set noformat\n  /set verbose|quiet\n  /set think|nothink")
            return nil
        }
        switch sub {
        case "parameter":
            guard args.count >= 3 else { say("Usage:\n  /set parameter <param> <value>"); return nil }
            return await setParameter(args[1], args.dropFirst(2).joined(separator: " "))
        case "system":
            guard let range = line.range(of: "system") else { return nil }
            let rest = line[range.upperBound...].trimmingCharacters(in: .whitespaces)
            guard !rest.isEmpty else { say("Usage:\n  /set system <string>"); return nil }
            if rest.hasPrefix("\"\"\"") {
                guard let text = gather(rest) else { return nil }
                settings.system = text
            } else {
                settings.system = rest
            }
            say("Set system message.")
        case "history": recordHistory = true
        case "nohistory": recordHistory = false
        case "wordwrap": settings.wordWrap = true; say("Set 'wordwrap' mode.")
        case "nowordwrap": settings.wordWrap = false; say("Set 'nowordwrap' mode.")
        case "verbose": settings.verbose = true; say("Set 'verbose' mode.")
        case "quiet": settings.verbose = false; say("Set 'quiet' mode.")
        case "format":
            if args.count == 2, args[1] == "json" {
                say("Set format to 'json' mode.")
                say("warning: format has no effect: not supported by the engine")
            } else {
                say("Invalid or missing format. For 'json' mode use '/set format json'")
            }
        case "noformat": say("Disabled format.")
        case "think":
            settings.thinking = true
            if args.count > 1, ["high", "medium", "low"].contains(args[1]) { say("Set 'think' mode to '\(args[1])'.") }
            else { say("Set 'think' mode.") }
        case "nothink":
            settings.thinking = false
            say("Set 'nothink' mode.")
        default: unknown("/set \(sub)")
        }
        return nil
    }

    private func setParameter(_ key: String, _ value: String) async -> Int32? {
        var p = settings.parameters
        do {
            switch try ChatParameter.apply(key, value, to: &p) {
            case .set:
                if key == "num_ctx", let ctx = Int(value), ctx != settings.contextLength,
                   let code = await reload(contextLength: ctx) { return code }
                settings.parameters = p
                settings.userParameters[key] = value
                say("Set parameter '\(key)' to '\(value)'")
            case .ignored:
                say("warning: parameter '\(key)' has no effect: mlxr does not use it")
            }
        } catch {
            say("Couldn't set parameter: \"\(CLI.message(for: error))\"")
        }
        return nil
    }

    /// Reloads the current model with a new context length; the conversation is kept.
    private func reload(contextLength: Int) async -> Int32? {
        interrupts.enter(.loading)
        defer { awaitInput() }
        do {
            settings.info = try await engine.load(directory: settings.directory, draftDirectory: settings.draftDirectory,
                                                  contextLength: contextLength, sparse: settings.sparse)
            settings.contextLength = contextLength
            return nil
        } catch {
            say("Error: \(CLI.message(for: error))")
            failed = true
            return 1
        }
    }
    private func show(_ args: [String]) {
        guard let sub = args.first else {
            say("Available Commands:\n  /show info         Show details for this model\n  /show license      Show model license\n  /show modelfile    Show Modelfile for this model\n  /show parameters   Show parameters for this model\n  /show system       Show system message\n  /show template     Show prompt template")
            return
        }
        guard ["info", "license", "modelfile", "parameters", "system", "template"].contains(sub) else {
            unknown("/show \(sub)"); return
        }
        let info: ShowInfo
        do { info = try ShowInfo.read(directory: settings.directory, repo: settings.repo, preset: nil) } catch {
            say("Error: \(CLI.message(for: error))"); return
        }
        switch sub {
        case "info": console.err(info.tables())
        case "license": console.out(info.license.map(CLI.endingInNewline) ?? "No license was specified for this model.\n")
        case "template": console.out(info.template.map(CLI.endingInNewline) ?? "No prompt template was specified for this model.\n")
        case "modelfile": console.out(info.modelfile())
        case "system":
            let text = settings.system ?? info.system
            console.out(text.map(CLI.endingInNewline) ?? "No system message was specified for this model.\n")
        default:
            func rows(_ kv: [(key: String, value: String)]) -> String {
                kv.map { $0.key.padding(toLength: 30, withPad: " ", startingAt: 0) + " " + $0.value + "\n" }.joined()
            }
            console.out("Model defined parameters:\n" + rows(info.parameters) + "\nUser defined parameters:\n"
                        + rows(settings.userParameters.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }))
        }
    }

    private func save(_ name: String?) {
        guard let name else { say("Usage:\n  /save <modelname>"); return }
        guard let repo = settings.repo else { say("Couldn't save: the model has no repository id"); return }
        do {
            try presets.save(Preset(name: name, repo: repo, system: settings.system, parameters: settings.userParameters))
            say("Created new model '\(name)'")
        } catch {
            say(CLI.message(for: error))
        }
    }

    private func load(_ name: String) async -> Int32? {
        guard !name.isEmpty else { say("Usage:\n  /load <modelname>"); return nil }
        guard let r = try? resolver.resolve(name), let dir = r.directory else {
            say("Couldn't find model '\(name)'")
            return nil
        }
        say("Loading model '\(name)'")
        interrupts.enter(.loading)
        do {
            let info = try await engine.load(directory: dir, draftDirectory: settings.draftDirectory,
                                             contextLength: settings.contextLength, sparse: settings.sparse)
            awaitInput()
            var next = settings
            next.model = name; next.repo = r.repo; next.directory = dir; next.info = info; next.preset = r.preset
            next.parameters = CLI.samplingDefaults(directory: dir, console: console)
            next.userParameters = [:]
            next.system = r.preset?.system
            settings = next
            messages = []
            for (k, v) in (r.preset?.parameters ?? [:]).sorted(by: { $0.key < $1.key }) {
                var p = settings.parameters
                if (try? ChatParameter.apply(k, v, to: &p)) == .set {
                    settings.parameters = p
                    settings.userParameters[k] = v
                }
            }
            return nil
        } catch {
            awaitInput()
            say("Error: \(CLI.message(for: error))")
            failed = true
            return 1
        }
    }
}

extension ChatParameter.Outcome: Equatable {}

extension CLI {
    /// The model's own sampling defaults (validated generation_config.json) over the engine's.
    static func samplingDefaults(directory: URL, console: Console) -> SamplingParameters {
        let defaults = GenerationDefaults.load(directory: directory)
        defaults.warnings.forEach { console.err($0 + "\n") }
        var params = SamplingParameters()
        if let v = defaults.temperature { params.temperature = v }
        if let v = defaults.topP { params.topP = v }
        if let v = defaults.topK { params.topK = v }
        if let v = defaults.minP { params.minP = v }
        if let v = defaults.repetitionPenalty { params.repetitionPenalty = v }
        params.maxTokens = SamplingLimits.maxTokens
        return params
    }

    /// The chat input for this terminal: libedit when stdin and stdout are terminals, else `readLine()`.
    static func liveLineSource(console: Console, history: ChatHistory) -> LineSource {
        if console.stdinIsTTY && console.stdoutIsTTY {
            let noHistory = ["1", "true"].contains((console.env["OLLAMA_NOHISTORY"] ?? "").lowercased())
            return LibeditLineSource(history: noHistory ? [] : history.lines(), console: console)
        }
        return ConsoleLineSource(console: console)
    }
}
