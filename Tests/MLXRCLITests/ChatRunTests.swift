import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("mlxr run into chat (AC-18, AC-14)")
struct ChatRunTests {
    let model = "qwen3:0.6b"

    @Test func runWithoutAPromptOnTerminalsStartsTheChat() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["hi", "/bye"])
        let r = await h.run(["run", model], lines: lines)
        #expect(r.code == 0, "\(r.err)")
        #expect(h.engine.loads.count == 1)
        #expect(h.engine.loads.first?.directory == h.dirSmall.directory)
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "hi")])
        #expect(r.out.contains("Hello world"), "\(r.out)")
        #expect(r.err.contains("Qwen3-0.6B-4bit"), "the load banner is on stderr: \(r.err)")
        #expect(!r.err.contains("not available yet"))
    }

    @Test func chatFlagsReachTheSession() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["hi"])
        let r = await h.run(["run", model, "--system", "be brief", "--temp", "0.3", "--seed", "5", "--no-think", "--ctx", "2048"], lines: lines)
        #expect(r.code == 0, "\(r.err)")
        #expect(h.engine.loads.first?.contextLength == 2048)
        let call = h.engine.chats.first
        #expect(call?.messages.first == ChatMessage(role: .system, content: "be brief"))
        #expect(call?.parameters.temperature == 0.3 && call?.parameters.seed == 5)
        #expect(call?.enableThinking == false)
    }

    @Test func verboseFlagPrintsStatsAfterEachChatReply() async throws {
        let h = try ChatHarness()
        let r = await h.run(["run", model, "--verbose"], lines: ScriptedLines(["a", "b"]))
        #expect(r.err.components(separatedBy: "eval rate:").count - 1 == 4, "two blocks of two rates: \(r.err)")
    }

    @Test func modelDefaultsFromGenerationConfigApplyToTheChat() async throws {
        let h = try ChatHarness()
        _ = await h.run(["run", model], lines: ScriptedLines(["hi"]))
        let p = h.engine.chats.first?.parameters
        #expect(p?.temperature == 0.6 && p?.topK == 20 && p?.topP == 0.95)
    }

    @Test func aPromptStillMeansOneShotEvenOnTerminals() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["never read"])
        let r = await h.run(["run", model, "Say", "hi"], lines: lines)
        #expect(r.code == 0, "\(r.err)")
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "Say hi")])
        #expect(lines.asked == 0)
    }

    @Test func ctrlDEndsTheChatWithExitZero() async throws {
        let h = try ChatHarness()
        let r = await h.run(["run", model], lines: ScriptedLines(events: [.eof]))
        #expect(r.code == 0, "\(r.err)")
        #expect(h.engine.loads.count == 1, "the model was loaded before the prompt")
        #expect(h.engine.chats.isEmpty)
    }
}
