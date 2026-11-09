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

@Suite("Ctrl-C in the other commands")
struct CommandSignalTests {
    let model = "qwen3:0.6b"

    @Test func ctrlCDuringPullCancelsItAndExits130() async throws {
        let h = try RunHarness(installed: false)
        let interrupts = h.interrupts
        let fake = h.hub.services
        h.hubOverride = HubServices(details: fake.details, destination: fake.destination,
                                    download: { _, _, _ in interrupts.fire(); throw CancellationError() })
        let r = await h.run(["pull", model])
        #expect(r.code == 130, "\(r.err)")
        #expect(!r.err.contains("Error:"), "\(r.err)")
        #expect(h.exits.codes.isEmpty, "the command unwinds by itself and keeps the partial files")
        #expect(h.interrupts.wasInterrupted)
        #expect(r.out.isEmpty, "no saved path for an unfinished pull")
    }

    /// Ctrl-C during `rm a b`: the delete in progress finishes (a half-deleted model directory can still
    /// look installed), then the command stops before the next one.
    @Test func ctrlCDuringRmFinishesTheCurrentDeleteThenStops() async throws {
        let f = LibraryFixture.standard()
        let log = ExitLog()
        let interrupts = Interrupts(exit: { log.add($0) })
        let base = f.library
        let library = ModelLibrary(scan: base.scan, delete: { m in
            interrupts.fire()
            #expect(log.codes.isEmpty, "the delete is not cut short")
            try base.delete(m)
        })
        let table = CLI.commands(services: FakeServices().services, library: library, presets: f.presets, catalog: .standard,
                                 now: { LibraryFixture.now }, interrupts: interrupts)
        let cap = Capture()
        let code = await CLI.run(["rm", LibraryFixture.qwen8, LibraryFixture.qwen06], console: cap.console(), commands: table)
        #expect(log.codes == [130], "\(cap.stderr)")
        #expect(code == 130)
        #expect(f.deleted == [LibraryFixture.qwen8], "the second model stays")
        #expect(cap.stdout == "deleted '\(LibraryFixture.qwen8)'\n")
    }
}
