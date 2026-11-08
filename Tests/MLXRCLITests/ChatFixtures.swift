import Foundation
import MLXEngine
@testable import MLXRCLI

/// A line source that plays back a script and records the prompts it was asked to show.
final class ScriptedLines: LineSource, @unchecked Sendable {
    private let lock = NSLock()
    private var events: [LineEvent]
    private var shown: [String] = []
    private var taken = 0

    init(_ lines: [String]) { events = lines.map { .line($0) } }
    init(events: [LineEvent]) { self.events = events }

    /// The prompts shown, in order.
    var prompts: [String] { lock.withLock { shown } }
    /// How many reads the loop made.
    var asked: Int { lock.withLock { shown.count } }
    /// Events the loop never consumed.
    var remaining: Int { lock.withLock { events.count } }

    func read(prompt: String) -> LineEvent {
        lock.withLock {
            shown.append(prompt)
            return events.isEmpty ? .eof : events.removeFirst()
        }
    }
}

/// The text `/?` and `/help` print on stderr (Ollama's, from the spec).
let ollamaChatHelp = """
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

/// A chat session over a fake engine and two installed fixture models (`qwen3:0.6b`, installed as
/// mlx-community/Qwen3-0.6B-4bit, and mlx-community/Qwen3-8B-4bit), a temporary presets directory and a
/// temporary history file. No GPU, no network, no real library.
final class ChatHarness: @unchecked Sendable {
    let engine: FakeEngine
    let library: LibraryFixture
    let dirSmall: ShowModelDir
    let dirBig: ShowModelDir
    let exits = ExitLog()
    let interrupts: Interrupts
    let historyFile: URL
    let capture = Capture()
    let console: Console
    let model = "qwen3:0.6b"

    init(engine: FakeEngine = FakeEngine(), env: [String: String] = [:], license: String? = ShowModelDir.license) throws {
        self.engine = engine
        dirSmall = try ShowModelDir(license: license)
        dirBig = try ShowModelDir()
        let log = exits
        interrupts = Interrupts(exit: { log.add($0) })
        library = LibraryFixture(models: [
            LocalModel(repo: LibraryFixture.qwen06, directory: dirSmall.directory, sizeBytes: 350_000_000,
                       modelType: "qwen3", bits: 4, source: .library, modified: LibraryFixture.now),
            LocalModel(repo: LibraryFixture.qwen8, directory: dirBig.directory, sizeBytes: 5_200_000_000,
                       modelType: "qwen3", bits: 4, source: .library, modified: LibraryFixture.now),
        ])
        historyFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxr-chat-history-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("history")
        console = capture.console(env: env)
    }

    deinit {
        dirSmall.remove(); dirBig.remove()
        try? FileManager.default.removeItem(at: historyFile.deletingLastPathComponent())
    }

    var stdout: String { capture.stdout }
    var stderr: String { capture.stderr }
    /// Both streams, for messages the spec does not pin to one of them.
    var all: String { capture.stdout + capture.stderr }

    func settings(_ tweak: (inout ChatSettings) -> Void = { _ in }) -> ChatSettings {
        var s = ChatSettings(model: model, repo: LibraryFixture.qwen06, directory: dirSmall.directory, info: FakeEngine.model(directory: dirSmall.directory))
        tweak(&s)
        return s
    }

    func session(_ lines: LineSource, history: ChatHistory? = nil, _ tweak: (inout ChatSettings) -> Void = { _ in }) -> ChatSession {
        ChatSession(console: console, engine: engine, resolver: library.resolver, presets: library.presets,
                    interrupts: interrupts, lines: lines, history: history ?? ChatHistory(file: historyFile), settings: settings(tweak))
    }

    /// Plays `lines` (one event each) through a chat and returns the exit code.
    @discardableResult
    func chat(_ lines: [String], _ tweak: (inout ChatSettings) -> Void = { _ in }) async -> Int32 {
        await session(ScriptedLines(lines), tweak).run()
    }

    /// Runs `mlxr run ...` with both ends on a terminal and the scripted lines as the chat input.
    func run(_ args: [String], lines: ScriptedLines) async -> (code: Int32, out: String, err: String) {
        let cap = Capture()
        let console = cap.console(stdoutTTY: true, stderrTTY: false, stdinTTY: true)
        let table = CLI.commands(services: FakeServices().services, library: library.library, presets: library.presets,
                                 catalog: .standard, now: { LibraryFixture.now }, hub: FakeHub().services,
                                 engine: engine, interrupts: interrupts, lineSource: lines,
                                 chatHistory: ChatHistory(file: historyFile))
        let code = await CLI.run(args, console: console, commands: table)
        return (code, cap.stdout, cap.stderr)
    }
}
