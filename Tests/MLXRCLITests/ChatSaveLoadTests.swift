import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("chat /load and /save (AC-18)")
struct ChatSaveLoadTests {
    // MARK: /save

    @Test func saveCreatesAPresetWithTheModelSystemAndParameters() async throws {
        let h = try ChatHarness()
        await h.chat(["/set system be brief", "/set parameter temperature 0.2", "/set parameter seed 7", "/save mine"])
        #expect(h.all.contains("Created new model 'mine'"), "\(h.all)")
        let preset = h.library.presets.load("mine")
        #expect(preset?.repo == LibraryFixture.qwen06)
        #expect(preset?.system == "be brief")
        #expect(preset?.parameters["temperature"] == "0.2" && preset?.parameters["seed"] == "7")
    }

    @Test func aSavedPresetResolvesLikeAnyModelName() async throws {
        let h = try ChatHarness()
        await h.chat(["/save mine"])
        let r = try h.library.resolver.resolve("mine")
        #expect(r.source == .preset && r.repo == LibraryFixture.qwen06 && r.directory == h.dirSmall.directory)
    }

    @Test func savingWithoutANameShowsUsage() async throws {
        let h = try ChatHarness()
        await h.chat(["/save"])
        #expect(h.all.contains("Usage:\n  /save <modelname>"), "\(h.all)")
        #expect(h.library.presets.list().isEmpty)
    }

    @Test func savingWithAnUnsafeNameIsRefusedAndTheChatContinues() async throws {
        let h = try ChatHarness()
        let lines = ScriptedLines(["/save ../evil", "/save bad/name", "hi"])
        await h.session(lines).run()
        #expect(h.all.contains("invalid model name"), "\(h.all)")
        #expect(h.library.presets.list().isEmpty)
        #expect(h.engine.chats.count == 1)
    }

    // MARK: /load

    @Test func loadSwitchesTheModelAndStartsAFreshConversation() async throws {
        let h = try ChatHarness()
        await h.chat(["hi", "/load \(LibraryFixture.qwen8)", "again"])
        #expect(h.all.contains("Loading model '\(LibraryFixture.qwen8)'"), "\(h.all)")
        #expect(h.engine.loads.count == 1)
        #expect(h.engine.loads.first?.directory == h.dirBig.directory)
        #expect(h.engine.chats.last?.messages == [ChatMessage(role: .user, content: "again")], "the old conversation is gone")
    }

    @Test func loadKeepsTheSessionsContextLength() async throws {
        let h = try ChatHarness()
        await h.chat(["/load \(LibraryFixture.qwen8)"]) { $0.contextLength = 8192 }
        #expect(h.engine.loads.first?.contextLength == 8192)
    }

    @Test func loadOfAMissingModelSaysSoAndLoadsNothing() async throws {
        let h = try ChatHarness()
        await h.chat(["/load nope", "hi"])
        #expect(h.all.contains("Couldn't find model 'nope'"), "\(h.all)")
        #expect(h.engine.loads.isEmpty)
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "hi")], "the chat is unchanged")
    }

    @Test func loadNeverPullsAModelThatIsNotInstalled() async throws {
        let h = try ChatHarness()
        await h.chat(["/load qwen3:4b"])
        #expect(h.all.contains("Couldn't find model 'qwen3:4b'"), "\(h.all)")
        #expect(h.engine.loads.isEmpty)
    }

    @Test func loadWithoutANameShowsUsage() async throws {
        let h = try ChatHarness()
        await h.chat(["/load"])
        #expect(h.all.contains("Usage:\n  /load <modelname>"), "\(h.all)")
        #expect(h.engine.loads.isEmpty)
    }

    @Test func loadOfAPresetAppliesItsSystemMessageAndParameters() async throws {
        let h = try ChatHarness()
        try h.library.presets.save(Preset(name: "terse", repo: LibraryFixture.qwen8, system: "be terse",
                                          parameters: ["temperature": "0.3"]))
        await h.chat(["/load terse", "hi"])
        #expect(h.engine.loads.first?.directory == h.dirBig.directory)
        let call = h.engine.chats.first
        #expect(call?.messages == [ChatMessage(role: .system, content: "be terse"), ChatMessage(role: .user, content: "hi")])
        #expect(call?.parameters.temperature == 0.3)
    }

    @Test func aFailedLoadIsReportedAndTheChatEndsWithoutAModelCrash() async throws {
        struct Boom: LocalizedError { var errorDescription: String? { "bad weights" } }
        let h = try ChatHarness()
        h.engine.loadError = Boom()
        let lines = ScriptedLines(["/load \(LibraryFixture.qwen8)", "/bye"])
        let code = await h.session(lines).run()
        #expect(h.stderr.contains("bad weights"), "\(h.stderr)")
        #expect(code == 1, "a failed load ends the chat with exit 1")
        #expect(lines.remaining == 1, "the next line (/bye) is never read: the chat ended at the failed load")
        #expect(h.engine.chats.isEmpty)
        #expect(h.engine.loads.count == 1)
    }
}
