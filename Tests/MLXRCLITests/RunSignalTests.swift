import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("run and Ctrl-C (AC-17, REQ-9)")
struct RunSignalTests {
    let model = "qwen3:0.6b"

    static let cancelled = FakeEngine.stats(generated: 3, rate: 10, stop: "cancelled")

    @Test func ctrlCDuringGenerationStopsItAndExits130() async throws {
        let engine = FakeEngine(script: [.text("partial"), .waitForCancel, .finished(Self.cancelled)])
        let h = try RunHarness(engine: engine)
        let task = Task { await h.run(["run", model, "-p", "hi"]) }
        await engine.waitUntilStreaming()
        h.interrupts.fire()
        let r = await task.value
        #expect(r.code == 130, "\(r.err)")
        #expect(r.out.hasPrefix("partial"))
        #expect(!r.err.contains("Error:"), "\(r.err)")
        #expect(h.exits.codes.isEmpty, "a graceful cancel does not _exit")
    }

    @Test func aSecondCtrlCWhileGeneratingForcesTheExit() async throws {
        let gate = Gate()
        let engine = FakeEngine(script: [.text("x"), .wait(gate), .finished(Self.cancelled)])
        let h = try RunHarness(engine: engine)
        let task = Task { await h.run(["run", model, "-p", "hi"]) }
        await engine.waitUntilStreaming()
        h.interrupts.fire()
        h.interrupts.fire()
        #expect(h.exits.codes == [130])
        gate.open()
        let r = await task.value
        #expect(r.code == 130)
    }

    @Test func ctrlCWhileLoadingExits130BeforeGenerating() async throws {
        let h = try RunHarness()
        let interrupts = h.interrupts
        h.engine.onLoad = { interrupts.fire() }
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(h.exits.codes == [130])
        #expect(r.code == 130)
        #expect(h.engine.chats.isEmpty)
        #expect(!r.err.contains("Error:"), "\(r.err)")
    }

    @Test func ctrlCDuringAPullCancelsItAndExits130() async throws {
        let h = try RunHarness(installed: false)
        let interrupts = h.interrupts
        let fake = h.hub.services
        h.hubOverride = HubServices(details: fake.details, destination: fake.destination,
                                    download: { _, _, _ in interrupts.fire(); throw CancellationError() })
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 130, "\(r.err)")
        #expect(!r.err.contains("Error:"), "\(r.err)")
        #expect(h.exits.codes.isEmpty, "the command unwinds by itself and keeps the partial files")
        #expect(h.engine.loads.isEmpty)
        #expect(h.interrupts.wasInterrupted)
    }

    @Test func aCleanRunLeavesNoInterrupt() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 0 && !h.interrupts.wasInterrupted)
    }
}
