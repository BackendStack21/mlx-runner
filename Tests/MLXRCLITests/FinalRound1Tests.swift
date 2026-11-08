import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

/// The real engine cancels the request's token whenever its stream terminates, a normal finish
/// included (Engine.swift onTermination). This engine does the same around a scripted FakeEngine.
final class TerminatingEngine: GenerationEngine, @unchecked Sendable {
    let inner: FakeEngine
    init(_ inner: FakeEngine = FakeEngine()) { self.inner = inner }

    func load(directory: URL, draftDirectory: URL?, contextLength: Int, sparse: SparseAttention?) async throws -> LoadedModelInfo {
        try await inner.load(directory: directory, draftDirectory: draftDirectory, contextLength: contextLength, sparse: sparse)
    }
    func chat(_ messages: [ChatMessage], parameters: SamplingParameters, enableThinking: Bool?, token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        wrap(inner.chat(messages, parameters: parameters, enableThinking: enableThinking, token: token), token)
    }
    func complete(_ prompt: String, parameters: SamplingParameters, token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        wrap(inner.complete(prompt, parameters: parameters, token: token), token)
    }
    func renderChat(_ messages: [ChatMessage], enableThinking: Bool?) -> String { inner.renderChat(messages, enableThinking: enableThinking) }
    func encode(_ text: String) -> [Int] { inner.encode(text) }
    func autoDraft(for target: URL) -> LocalModel? { inner.autoDraft(for: target) }

    private func wrap(_ source: AsyncThrowingStream<GenerationEvent, Error>, _ token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { cont in
            cont.onTermination = { _ in token.cancel() }
            let task = Task {
                do {
                    for try await ev in source { cont.yield(ev) }
                    token.cancel()  // what onTermination does when the stream ends
                    cont.finish()
                } catch { token.cancel(); cont.finish(throwing: error) }
            }
            let previous = cont.onTermination
            cont.onTermination = { reason in previous?(reason); task.cancel() }
        }
    }
}

@Suite("final round 1: a finished generation is not an interrupt (R-01)")
struct FinishedRunTests {
    let model = "qwen3:0.6b"

    func run(_ args: [String]) async throws -> (code: Int32, out: String, err: String, harness: RunHarness) {
        let h = try RunHarness()
        let cap = Capture()
        let console = cap.console()
        console.readAllStdin = { "" }
        let table = CLI.commands(services: FakeServices().services, library: h.library.library, presets: h.library.presets,
                                 catalog: .standard, now: { LibraryFixture.now }, hub: h.hub.services,
                                 engine: TerminatingEngine(h.engine), interrupts: h.interrupts)
        let code = await CLI.run(args, console: console, commands: table)
        return (code, cap.stdout, cap.stderr, h)
    }

    @Test func aFinishedOneShotRunExitsZeroWithItsTrailingNewlines() async throws {
        let r = try await run(["run", model, "-p", "hi"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "Hello world\n\n")
        #expect(!r.harness.interrupts.wasInterrupted)
        #expect(r.harness.exits.codes.isEmpty)
    }

    @Test func verboseStillPrintsTheStatsAfterAFinishedRun() async throws {
        let r = try await run(["run", model, "-p", "hi", "--verbose"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.err.contains(StatsFormat.block(RunStats(loadSeconds: 1.5, stats: FakeEngine.stats()))), "\(r.err)")
        #expect(r.err.contains("[prompt: 100 tokens, 200 tok/s, ttft 0.500s]"), "\(r.err)")
        #expect(r.err.contains("[generated 50 tokens at 25.0 tok/s, stop: stop, drafts accepted 0/0]"), "\(r.err)")
    }

    @Test func aRawFinishedRunExitsZeroToo() async throws {
        let r = try await run(["run", model, "--raw", "-p", "once"])
        #expect(r.code == 0 && r.out == "Hello world\n\n", "\(r.err)")
    }

    @Test func pipedChatContinuesAfterAFinishedReplyAndKeepsItInHistory() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["a", "b"])
        let session = ChatSession(console: h.console, engine: TerminatingEngine(h.engine), resolver: h.library.resolver,
                                  presets: h.library.presets, interrupts: h.interrupts, lines: lines,
                                  history: ChatHistory(file: h.historyFile), settings: h.settings())
        let code = await session.run()
        #expect(code == 0, "\(h.stderr)")
        #expect(h.engine.chats.count == 2, "the second line is answered")
        #expect(h.engine.chats.last?.messages == [ChatMessage(role: .user, content: "a"),
                                                  ChatMessage(role: .assistant, content: "Hello world"),
                                                  ChatMessage(role: .user, content: "b")])
        #expect(lines.remaining == 0)
    }
}

@Suite("final round 1: error text and docs (R-02, R-03, R-04)")
struct FinalRound1CLITests {
    func stderr(_ args: [String]) async -> (Int32, String) {
        let cap = Capture()
        let code = await CLI.run(args, console: cap.console())
        return (code, cap.stderr)
    }

    @Test func invalidValuesNameTheFlagAsTyped() async {
        let cases: [([String], String)] = [
            (["run", "m", "--max", "abc"], "--max"),
            (["run", "m", "--max-tokens", "abc"], "--max-tokens"),
            (["run", "m", "-n", "abc"], "-n, --max-tokens"),
            (["run", "m", "--temperature", "abc"], "--temperature"),
            (["run", "m", "--temp", "abc"], "--temp"),
        ]
        for (args, shown) in cases {
            let (code, err) = await stderr(args)
            #expect(code == 1 && err.hasPrefix("Error: invalid argument \"abc\" for \"\(shown)\" flag"), "\(args): \(err)")
        }
        let (_, zero) = await stderr(["run", "m", "--max", "0"])
        #expect(zero.hasPrefix("Error: invalid argument \"0\" for \"--max\" flag"), "\(zero)")
    }

    @Test func searchWithoutAQueryIsTheOneArgumentError() async {
        let (code, err) = await stderr(["search"])
        #expect(code == 1 && err == "Error: accepts 1 arg(s), received 0\n", "\(err)")
    }

    @Test func theReproducibleGreedyCommandKeepsTheRepetitionPenaltyAtOne() {
        let greedy = "--temp 0 --repeat-penalty 1 --seed 42 --max 256"
        for f in ["README.md", "AGENTS.md", "CHANGELOG.md"] {
            let text = ReleaseDocsTests.read(f)
            #expect(text.contains(greedy), "\(f) lacks '\(greedy)'")
        }
        for f in ["README.md", "AGENTS.md"] {
            #expect(!ReleaseDocsTests.read(f).contains("--temp 0 --seed 42 --max 256"), "\(f) still gives the old greedy command")
        }
    }
}

/// Runs the real engine through the CLI when a small model is installed (skipped otherwise, like
/// Tests/MLXEngineTests/ModelTests.swift).
func smallInstalledModel() -> String? {
    ["mlx-community/Qwen2.5-0.5B-Instruct-4bit", "mlx-community/Llama-3.2-1B-Instruct-4bit"]
        .first { repo in ModelStore.scan().contains { $0.repo == repo } }
}

@Suite("final round 1: the real engine through the CLI", .serialized)
struct RealEngineRunTests {
    @Test(.enabled(if: smallInstalledModel() != nil))
    func aOneShotRunOnARealModelExitsZero() async {
        let repo = smallInstalledModel()!
        let cap = Capture()
        let console = cap.console()
        console.readAllStdin = { "" }
        let code = await CLI.run(["run", repo, "-p", "Say hi in two words", "--temp", "0", "--seed", "42", "--max", "8", "--verbose"], console: console)
        #expect(code == 0, "\(cap.stderr)")
        #expect(cap.stdout.hasSuffix("\n\n") && cap.stdout.count > 2, "\(cap.stdout)")
        #expect(cap.stderr.contains("eval rate:") && cap.stderr.contains("[generated "), "\(cap.stderr)")
    }
}
