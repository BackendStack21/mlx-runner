import Foundation
@testable import MLXRCLI
import MLXEngine

/// A latch the test opens by hand.
final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    var isOpen: Bool { lock.withLock { opened } }
    func open() { lock.withLock { opened = true } }
}

/// Records the exit codes the interrupt router asked for.
final class ExitLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [Int32] = []
    var codes: [Int32] { lock.withLock { items } }
    func add(_ c: Int32) { lock.withLock { items.append(c) } }
}

/// Counts how often a cancel action ran.
final class CancelCount: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    var count: Int { lock.withLock { n } }
    func hit() { lock.withLock { n += 1 } }
}

/// A scripted engine: records every load and generation, streams the scripted events and honors the
/// cancel token. No GPU, no model.
final class FakeEngine: GenerationEngine, @unchecked Sendable {
    enum Step {
        case promptProcessed(GenerationStats)
        case text(String)
        case finished(GenerationStats)
        /// Blocks until the request's token is cancelled.
        case waitForCancel
        /// Blocks until the test opens the gate (ignores the token).
        case wait(Gate)
        case fail(Error)
    }

    struct Load: @unchecked Sendable {
        let directory: URL, draftDirectory: URL?, contextLength: Int, sparse: SparseAttention?
    }
    struct Chat: @unchecked Sendable {
        let messages: [ChatMessage], parameters: SamplingParameters, enableThinking: Bool?
    }
    struct Completion: @unchecked Sendable {
        let prompt: String, parameters: SamplingParameters
    }

    private let lock = NSLock()
    private var loadLog: [Load] = []
    private var chatLog: [Chat] = []
    private var completionLog: [Completion] = []
    private var streamStarted = false
    var script: [Step]
    var info: LoadedModelInfo
    var loadError: Error?
    var onLoad: (@Sendable () -> Void)?
    var draft: LocalModel?

    var loads: [Load] { lock.withLock { loadLog } }
    var chats: [Chat] { lock.withLock { chatLog } }
    var completions: [Completion] { lock.withLock { completionLog } }
    var started: Bool { lock.withLock { streamStarted } }

    static func stats(prompt: Int = 100, cached: Int = 20, ttft: Double = 0.5, promptRate: Double = 200,
                      generated: Int = 50, rate: Double = 25, stop: String = "stop") -> GenerationStats {
        var s = GenerationStats()
        s.promptTokens = prompt; s.cachedPromptTokens = cached; s.timeToFirstToken = ttft
        s.promptTokensPerSecond = promptRate; s.generatedTokens = generated; s.tokensPerSecond = rate
        s.stopReason = stop
        return s
    }

    static func model(thinksByDefault: Bool = false, loadSeconds: Double = 1.5, directory: URL = URL(fileURLWithPath: "/fixture/model")) -> LoadedModelInfo {
        LoadedModelInfo(name: "Qwen3-0.6B-4bit", directory: directory, modelType: "qwen3", quantization: "4-bit",
                        parameterBytes: 350_000_000, contextLength: 4096, layers: 28, hiddenSize: 1024, chatFormat: "chatml",
                        supportsThinking: true, loadSeconds: loadSeconds, thinksByDefault: thinksByDefault)
    }

    /// The default reply: "Hello", " world" with stats.
    static func reply(_ chunks: [String] = ["Hello", " world"]) -> [Step] {
        [.promptProcessed(stats(generated: 0, rate: 0))] + chunks.map { .text($0) } + [.finished(stats())]
    }

    init(script: [Step] = FakeEngine.reply(), info: LoadedModelInfo = FakeEngine.model()) {
        self.script = script; self.info = info
    }

    func load(directory: URL, draftDirectory: URL?, contextLength: Int, sparse: SparseAttention?) async throws -> LoadedModelInfo {
        lock.withLock { loadLog.append(Load(directory: directory, draftDirectory: draftDirectory, contextLength: contextLength, sparse: sparse)) }
        onLoad?()
        if let e = loadError { throw e }
        return info
    }

    func chat(_ messages: [ChatMessage], parameters: SamplingParameters, enableThinking: Bool?,
              token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        lock.withLock { chatLog.append(Chat(messages: messages, parameters: parameters, enableThinking: enableThinking)) }
        return stream(token)
    }

    func complete(_ prompt: String, parameters: SamplingParameters, token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        lock.withLock { completionLog.append(Completion(prompt: prompt, parameters: parameters)) }
        return stream(token)
    }

    func renderChat(_ messages: [ChatMessage], enableThinking: Bool?) -> String { messages.map(\.content).joined(separator: "\n") }
    func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    func autoDraft(for target: URL) -> LocalModel? { draft }

    private func stream(_ token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        let steps = script
        return AsyncThrowingStream { cont in
            self.lock.withLock { self.streamStarted = true }
            let task = Task {
                for step in steps {
                    switch step {
                    case .promptProcessed(let s): cont.yield(.promptProcessed(s))
                    case .text(let t): cont.yield(.text(t))
                    case .finished(let s): cont.yield(.finished(s))
                    case .fail(let e): cont.finish(throwing: e); return
                    case .waitForCancel:
                        var waited = 0
                        while !token.isCancelled && waited < 1000 { try? await Task.sleep(nanoseconds: 5_000_000); waited += 1 }
                    case .wait(let gate):
                        var waited = 0
                        while !gate.isOpen && waited < 1000 { try? await Task.sleep(nanoseconds: 5_000_000); waited += 1 }
                    }
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    /// Waits (up to 5 s) until a generation has started streaming.
    func waitUntilStreaming() async {
        var waited = 0
        while !started && waited < 1000 { try? await Task.sleep(nanoseconds: 5_000_000); waited += 1 }
    }
}

/// Runs `mlxr run ...` against a fake engine, hub and library. The model `qwen3:0.6b` is installed
/// in a real temporary directory (so generation_config.json can be read) unless `installed` is false.
final class RunHarness: @unchecked Sendable {
    let library: LibraryFixture
    let hub: FakeHub
    let engine: FakeEngine
    let exits = ExitLog()
    let interrupts: Interrupts
    let modelDir: ShowModelDir?
    /// Replaces the hub the commands see (for scripted downloads).
    var hubOverride: HubServices?

    init(installed: Bool = true, generation: String? = #"{"temperature":0.6,"top_k":20,"top_p":0.95}"#,
         engine: FakeEngine = FakeEngine(), hub: FakeHub = FakeHub()) throws {
        self.engine = engine
        self.hub = hub
        let log = exits
        interrupts = Interrupts(exit: { log.add($0) })
        if installed {
            let dir = try ShowModelDir(generation: generation)
            modelDir = dir
            library = LibraryFixture(models: [
                LocalModel(repo: LibraryFixture.qwen06, directory: dir.directory, sizeBytes: 350_000_000,
                           modelType: "qwen3", bits: 4, source: .library, modified: LibraryFixture.now),
            ])
        } else {
            modelDir = nil
            library = LibraryFixture(models: [])
        }
    }

    deinit { modelDir?.remove() }

    func run(_ args: [String], stdin: String = "", stdoutTTY: Bool = false, stderrTTY: Bool = false,
             stdinTTY: Bool = false, width: Int = 80) async -> (code: Int32, out: String, err: String) {
        let cap = Capture()
        let console = cap.console(stdoutTTY: stdoutTTY, stderrTTY: stderrTTY, stdinTTY: stdinTTY, width: width)
        console.readAllStdin = { stdin }
        let table = CLI.commands(services: FakeServices().services, library: library.library, presets: library.presets,
                                 catalog: .standard, now: { LibraryFixture.now }, hub: hubOverride ?? hub.services,
                                 engine: engine, interrupts: interrupts)
        let code = await CLI.run(args, console: console, commands: table)
        return (code, cap.stdout, cap.stderr)
    }
}
