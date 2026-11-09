import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

/// A line source that records the Ctrl-C phase at each read. In its script, `.interrupt` stands for
/// Ctrl-C while the prompt waits: the read fires the router and returns `.interrupt` if that woke it,
/// `.line("not woken")` otherwise. `wakeable: false` is a source that cannot be woken.
final class WakeableLines: LineSource, @unchecked Sendable {
    private let lock = NSLock()
    private var script: [LineEvent]
    private var seen: [InterruptPhase] = []
    private var woken = false
    private let interrupts: Interrupts
    private let wakeable: Bool

    init(_ script: [LineEvent], interrupts: Interrupts, wakeable: Bool = true) {
        self.script = script; self.interrupts = interrupts; self.wakeable = wakeable
    }

    var interrupter: (@Sendable () -> Void)? {
        guard wakeable else { return nil }
        return { [weak self] in self?.wake() }
    }
    /// The router's phase at each read.
    var phases: [InterruptPhase] { lock.withLock { seen } }

    private func wake() { lock.withLock { woken = true } }

    func read(prompt: String) -> LineEvent {
        let next: LineEvent = lock.withLock {
            seen.append(interrupts.phase)
            woken = false
            return script.isEmpty ? .eof : script.removeFirst()
        }
        guard next == .interrupt else { return next }
        interrupts.fire()   // outside the lock: the router calls `interrupter`
        return lock.withLock { woken } ? .interrupt : .line("not woken")
    }
}

@Suite("Ctrl-C at the chat prompt")
struct ChatInterruptTests {
    static let hint = "Use Ctrl + d or /bye to exit."

    @Test func ctrlCAtThePromptShowsTheHintAndTheChatGoesOn() async throws {
        let h = try ChatHarness()
        let lines = WakeableLines([.interrupt, .interrupt, .line("hi")], interrupts: h.interrupts)
        let code = await h.session(lines).run()
        #expect(code == 0)
        #expect(h.exits.codes.isEmpty)
        #expect(h.stderr.components(separatedBy: Self.hint).count - 1 == 2, "\(h.stderr)")
        #expect(h.engine.chats.last?.messages == [ChatMessage(role: .user, content: "hi")])
        #expect(!h.interrupts.wasInterrupted)
    }

    @Test func thePromptIsWakeableBeforeAndAfterEveryTurn() async throws {
        let h = try ChatHarness()
        let lines = WakeableLines([.line("one"), .line("two")], interrupts: h.interrupts)
        await h.session(lines).run()
        #expect(lines.phases == [.prompt, .prompt, .prompt])
    }

    @Test func ctrlCAtAPromptThatCannotBeWokenEndsTheProcess() async throws {
        let h = try ChatHarness()
        let lines = WakeableLines([.interrupt], interrupts: h.interrupts, wakeable: false)
        await h.session(lines).run()
        #expect(lines.phases.first == .idle)
        #expect(h.exits.codes == [130])
    }

    @Test func ctrlCWhileGeneratingStillCancelsTheReplyAndReturnsToThePrompt() async throws {
        let engine = FakeEngine(script: [.text("partial"), .waitForCancel, .finished(FakeEngine.stats(stop: "cancelled"))])
        let h = try ChatHarness(engine: engine)
        let lines = WakeableLines([.line("first")], interrupts: h.interrupts)
        let interrupts = h.interrupts
        let canceller = Task { await engine.waitUntilStreaming(); interrupts.fire() }
        let code = await h.session(lines).run()
        await canceller.value
        #expect(code == 0)
        #expect(h.exits.codes.isEmpty)
        #expect(lines.phases == [.prompt, .prompt])
    }
}
