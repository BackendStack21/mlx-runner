import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("chat review round 1 (AC-18)")
struct ChatRoundTwoTests {
    // MARK: R-01 the user turn stays after a cancelled reply

    @Test func aCancelledReplyKeepsTheUserTurnInHistory() async throws {
        let engine = FakeEngine(script: [.promptProcessed(FakeEngine.stats(generated: 0, rate: 0)), .text("partial"),
                                         .waitForCancel, .finished(FakeEngine.stats(stop: "cancelled"))])
        let h = try ChatHarness(engine: engine)
        let session = h.session(ScriptedLines(["first", "second"]))
        let interrupts = h.interrupts
        let canceller = Task { await engine.waitUntilStreaming(); interrupts.fire() }
        await session.run()
        await canceller.value
        #expect(engine.chats.last?.messages == [ChatMessage(role: .user, content: "first"),
                                                ChatMessage(role: .user, content: "second")])
    }

    // MARK: R-02 parameters

    @Test func numCtxReloadsTheModelWithTheNewContextBeforeTheNextTurn() async throws {
        let h = try ChatHarness()
        await h.chat(["a", "/set parameter num_ctx 8192", "b"])
        #expect(h.engine.loads.count == 1, "one reload")
        #expect(h.engine.loads.first?.contextLength == 8192)
        #expect(h.engine.loads.first?.directory == h.dirSmall.directory)
        #expect(h.engine.chats.last?.messages == [ChatMessage(role: .user, content: "a"),
                                                  ChatMessage(role: .assistant, content: "Hello world"),
                                                  ChatMessage(role: .user, content: "b")], "the conversation survives the reload")
    }

    @Test func numCtxOutOfRangeIsRejectedWithoutReloading() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter num_ctx 100", "/set parameter num_ctx 2000000", "/set parameter num_ctx abc", "hi"])
        #expect(h.all.components(separatedBy: "Couldn't set parameter:").count - 1 == 3, "\(h.all)")
        #expect(h.engine.loads.isEmpty)
    }

    @Test func numCtxEqualToTheCurrentContextDoesNotReload() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter num_ctx 4096", "hi"]) { $0.contextLength = 4096 }
        #expect(h.engine.loads.isEmpty)
        #expect(h.engine.chats.count == 1)
    }

    @Test func parametersTheEngineCannotHonorWarnInsteadOfClaimingSuccess() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter typical_p 0.5", "/set parameter stop END", "/set parameter num_keep 4",
                      "/set parameter draft_num_predict 3"])
        #expect(h.stderr.components(separatedBy: "no effect").count - 1 == 4, "\(h.stderr)")
        #expect(!h.all.contains("Set parameter 'typical_p'") && !h.all.contains("Set parameter 'stop'"))
    }

    @Test func seedStillAppliesAndIsSavedWithAPreset() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter seed 11", "hi", "/save s"])
        #expect(h.engine.chats.first?.parameters.seed == 11)
        #expect(h.library.presets.load("s")?.parameters["seed"] == "11")
    }

    // MARK: R-03 Ctrl-C at the libedit prompt writes nothing to stdout

    @Test func ctrlCAtTheEditorPromptEchoesOnStderrOnly() {
        let cap = Capture()
        let console = cap.console()
        #expect(LibeditLineSource.event(line: nil, interrupted: true, console: console) == .interrupt)
        #expect(cap.stdout.isEmpty, "\(cap.stdout)")
        #expect(cap.stderr == "\n")
    }

    @Test func theEditorMapsLinesAndEndOfInput() {
        let cap = Capture()
        let console = cap.console()
        #expect(LibeditLineSource.event(line: "hello", interrupted: false, console: console) == .line("hello"))
        #expect(LibeditLineSource.event(line: nil, interrupted: false, console: console) == .eof)
        #expect(cap.stdout.isEmpty && cap.stderr.isEmpty)
    }

    // MARK: R-04 the history file stays bounded

    private func bigHistory() throws -> (ChatHistory, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-histcap-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("history")
        var text = ""
        text.reserveCapacity(5 << 20)
        for i in 0..<250_000 { text += "history line number \(i)\n" }
        try Data(text.utf8).write(to: file)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        return (ChatHistory(file: file), dir)
    }

    @Test func anOversizedHistoryStillYieldsItsNewestLines() throws {
        let (history, dir) = try bigHistory(); defer { try? FileManager.default.removeItem(at: dir) }
        let lines = history.lines()
        #expect(!lines.isEmpty)
        #expect(lines.last == "history line number 249999", "\(lines.last ?? "nil")")
        #expect(!lines.contains("history line number 0"))
    }

    @Test func appendingToAnOversizedHistoryTrimsItToTheNewestLines() throws {
        let (history, dir) = try bigHistory(); defer { try? FileManager.default.removeItem(at: dir) }
        history.append("brand new")
        let file = try #require(history.file)
        let size = (try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? NSNumber)?.intValue ?? Int.max
        #expect(size <= 4 << 20, "\(size)")
        let lines = history.lines()
        #expect(lines.last == "brand new")
        #expect(lines.contains("history line number 249999"))
        #expect(!lines.contains("history line number 0"))
        let mode = (try FileManager.default.attributesOfItem(atPath: file.path)[.posixPermissions] as? NSNumber)?.intValue
        #expect(mode == 0o600)
    }
}
