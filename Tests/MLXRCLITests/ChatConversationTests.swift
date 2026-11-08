import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("chat conversation (AC-18)")
struct ChatConversationTests {
    // MARK: turns

    @Test func aLineIsOneTurnAndTheReplyGoesToStdout() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["hi"])
        let code = await h.session(lines).run()
        #expect(code == 0)
        #expect(h.engine.chats.count == 1)
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "hi")])
        #expect(h.stdout == "Hello world\n\n", "\(h.stdout)")
        #expect(lines.prompts == [">>> ", ">>> "], "prompt, then the prompt that meets end of input")
    }

    @Test func historyAccumulatesAcrossTurnsWithTheAnswerOnly() async throws {
        let h = try ChatHarness()
        await h.chat(["first", "second"])
        #expect(h.engine.chats.count == 2)
        #expect(h.engine.chats.last?.messages == [
            ChatMessage(role: .user, content: "first"),
            ChatMessage(role: .assistant, content: "Hello world"),
            ChatMessage(role: .user, content: "second"),
        ])
        #expect(h.stdout == "Hello world\n\nHello world\n\n")
    }

    @Test func reasoningIsShownButNotKeptInHistory() async throws {
        let engine = FakeEngine(script: FakeEngine.reply(["<think>plan it</think>", "answer"]))
        let h = try ChatHarness(engine: engine)
        await h.chat(["q1", "q2"])
        #expect(h.stdout.contains("Thinking..."), "\(h.stdout)")
        #expect(h.stdout.contains("plan it") && h.stdout.contains("...done thinking."))
        #expect(h.stdout.contains("answer"))
        #expect(engine.chats.last?.messages == [
            ChatMessage(role: .user, content: "q1"),
            ChatMessage(role: .assistant, content: "answer"),
            ChatMessage(role: .user, content: "q2"),
        ])
    }

    @Test func hideThinkingDropsTheReasoningDisplay() async throws {
        let engine = FakeEngine(script: FakeEngine.reply(["<think>plan it</think>", "answer"]))
        let h = try ChatHarness(engine: engine)
        await h.chat(["q"]) { $0.hideThinking = true }
        #expect(!h.stdout.contains("Thinking...") && !h.stdout.contains("plan it"), "\(h.stdout)")
        #expect(h.stdout.contains("answer"))
    }

    @Test func blankLinesAreNotTurns() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["", "   ", "hi"])
        await h.session(lines).run()
        #expect(h.engine.chats.count == 1)
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "hi")])
    }

    @Test func theSessionSettingsReachTheEngine() async throws {
        let h = try ChatHarness()
        await h.chat(["hi"]) {
            $0.system = "be brief"
            $0.thinking = false
            $0.parameters.temperature = 0.25
            $0.parameters.seed = 9
        }
        let call = h.engine.chats.first
        #expect(call?.messages == [ChatMessage(role: .system, content: "be brief"), ChatMessage(role: .user, content: "hi")])
        #expect(call?.enableThinking == false)
        #expect(call?.parameters.temperature == 0.25 && call?.parameters.seed == 9)
    }

    @Test func aFailingGenerationReportsTheErrorAndTheChatContinues() async throws {
        struct Boom: LocalizedError { var errorDescription: String? { "out of memory" } }
        let engine = FakeEngine(script: [.fail(Boom())])
        let h = try ChatHarness(engine: engine)
        let lines = ScriptedLines(["one", "two"])
        let code = await h.session(lines).run()
        #expect(code == 0)
        #expect(engine.chats.count == 2, "the second line is still read and sent")
        #expect(h.stderr.contains("out of memory"), "\(h.stderr)")
        #expect(engine.chats.last?.messages.filter { $0.role == .assistant }.isEmpty == true, "no reply, none stored")
    }

    // MARK: multiline

    @Test func tripleQuotesJoinLinesWithNewlines() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["\"\"\"", "line one", "line two\"\"\""])
        await h.session(lines).run()
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "line one\nline two")])
        #expect(lines.prompts.prefix(3) == [">>> ", "... ", "... "], "\(lines.prompts)")
    }

    @Test func textAfterTheOpeningQuotesIsPartOfTheMessage() async throws {
        let h = try ChatHarness()
        await h.chat(["\"\"\"first", "second", "third\"\"\""])
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "first\nsecond\nthird")])
    }

    @Test func openAndCloseOnOneLineIsASingleLineMessage() async throws {
        let h = try ChatHarness()
        await h.chat(["\"\"\"just one line\"\"\""])
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "just one line")])
    }

    @Test func multilineMessageInsideAChatKeepsBlankLines() async throws {
        let h = try ChatHarness()
        await h.chat(["\"\"\"", "a", "", "b\"\"\""])
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "a\n\nb")])
    }

    @Test func slashLinesInsideAMultilineMessageAreText() async throws {
        let h = try ChatHarness()
        await h.chat(["\"\"\"", "/bye", "done\"\"\"", "/bye"])
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "/bye\ndone")])
    }

    // MARK: Ctrl-C and Ctrl-D

    @Test func ctrlCAtThePromptPrintsTheHintAndKeepsTheChat() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(events: [.interrupt, .line("hi")])
        let code = await h.session(lines).run()
        #expect(code == 0)
        #expect(h.stderr.contains("Use Ctrl + d or /bye to exit."), "\(h.stderr)")
        #expect(!h.stdout.contains("Use Ctrl + d"), "hints are not payload")
        #expect(h.engine.chats.count == 1, "the chat went on after Ctrl-C")
        #expect(lines.asked >= 2)
    }

    @Test func ctrlCInsideAMultilineMessageDiscardsIt() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(events: [.line("\"\"\""), .line("abandoned"), .interrupt, .line("hi")])
        await h.session(lines).run()
        #expect(h.engine.chats.count == 1)
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "hi")])
        #expect(lines.prompts == [">>> ", "... ", "... ", ">>> ", ">>> "], "\(lines.prompts)")
    }

    @Test func ctrlDOnAnEmptyLineExitsWithZero() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(events: [.line("hi"), .eof, .line("never read")])
        let code = await h.session(lines).run()
        #expect(code == 0)
        #expect(h.engine.chats.count == 1)
        #expect(lines.remaining == 1, "nothing is read after end of input")
    }

    @Test func byeAndExitLeaveWithZeroAndReadNothingMore() async throws {
        for word in ["/bye", "/exit"] {
            let h = try ChatHarness()
            let lines = ScriptedLines(["hi", word, "never"])
            let code = await h.session(lines).run()
            #expect(code == 0, "\(word)")
            #expect(h.engine.chats.count == 1, "\(word)")
            #expect(lines.remaining == 1, "\(word)")
        }
    }

    @Test func ctrlCMidReplyCancelsItAndDropsItFromHistory() async throws {
        let engine = FakeEngine(script: [.promptProcessed(FakeEngine.stats(generated: 0, rate: 0)), .text("partial"),
                                         .waitForCancel, .finished(FakeEngine.stats(stop: "cancelled"))])
        let h = try ChatHarness(engine: engine)
        let lines = ScriptedLines(["first", "second"])
        let session = h.session(lines)
        let interrupts = h.interrupts
        let canceller = Task {
            await engine.waitUntilStreaming()
            interrupts.fire()
        }
        let code = await session.run()
        await canceller.value
        #expect(code == 0, "the chat goes back to the prompt and ends normally")
        #expect(engine.chats.count == 2, "the second line was read after the cancel")
        #expect(h.exits.codes.isEmpty, "one Ctrl-C only cancels, it does not leave the process")
        let second = engine.chats.last?.messages ?? []
        #expect(!second.contains { $0.role == .assistant }, "the cancelled reply is not in the history: \(second)")
        #expect(second.last == ChatMessage(role: .user, content: "second"))
    }

    @Test func aSecondCtrlCWhileTheReplyIsStillStoppingLeavesTheProcess() async throws {
        let gate = Gate()
        let engine = FakeEngine(script: [.text("stuck"), .wait(gate)])
        let h = try ChatHarness(engine: engine)
        let session = h.session(ScriptedLines(["hi"]))
        let interrupts = h.interrupts
        let firer = Task {
            await engine.waitUntilStreaming()
            interrupts.fire()
            interrupts.fire()
            gate.open()
        }
        _ = await session.run()
        await firer.value
        #expect(h.exits.codes == [130])
    }

    @Test func ctrlCAtThePromptAfterAReplyDoesNotLeaveTheProcess() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(events: [.line("hi"), .interrupt, .interrupt])
        await h.session(lines).run()
        #expect(h.exits.codes.isEmpty)
        #expect(h.stderr.components(separatedBy: "Use Ctrl + d or /bye to exit.").count == 3, "the hint once per Ctrl-C: \(h.stderr)")
    }
}
