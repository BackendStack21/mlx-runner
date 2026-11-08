import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("chat slash commands (AC-18)")
struct ChatSlashCommandTests {
    // MARK: help, clear, unknown

    @Test func helpPrintsOllamasTextOnStderr() async throws {
        for cmd in ["/?", "/help"] {
            let h = try ChatHarness()
            await h.chat([cmd])
            #expect(h.stderr.contains(ollamaChatHelp), "\(cmd): \(h.stderr)")
            #expect(h.stdout.isEmpty, "\(cmd)")
            #expect(h.engine.chats.isEmpty, "\(cmd) is not a turn")
        }
    }

    @Test func unknownSlashCommandNamesItAndPointsToHelp() async throws {
        let h = try ChatHarness()
        await h.chat(["/bogus", "/usr/bin/env"])
        #expect(h.all.contains("Unknown command '/bogus'. Type /? for help"), "\(h.all)")
        #expect(h.all.contains("Unknown command '/usr/bin/env'. Type /? for help"))
        #expect(h.engine.chats.isEmpty, "an unknown command is not sent to the model")
    }

    @Test func clearForgetsTheConversationButKeepsTheSystemMessage() async throws {
        let h = try ChatHarness()
        await h.chat(["/set system be brief", "hi", "/clear", "again"])
        #expect(h.all.contains("Cleared session context"), "\(h.all)")
        #expect(h.engine.chats.last?.messages == [
            ChatMessage(role: .system, content: "be brief"),
            ChatMessage(role: .user, content: "again"),
        ])
    }

    // MARK: /set parameter

    @Test func setParameterAcceptsOllamaKeysAndChangesTheNextRequest() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter temperature 0.2", "/set parameter top_k 7", "/set parameter top_p 0.5",
                      "/set parameter min_p 0.1", "/set parameter repeat_penalty 1.2", "/set parameter repeat_last_n 32",
                      "/set parameter seed 42", "/set parameter num_predict 99", "hi"])
        for (k, v) in [("temperature", "0.2"), ("top_k", "7"), ("top_p", "0.5"), ("min_p", "0.1"),
                       ("repeat_penalty", "1.2"), ("repeat_last_n", "32"), ("seed", "42"), ("num_predict", "99")] {
            #expect(h.all.contains("Set parameter '\(k)' to '\(v)'"), "\(k): \(h.all)")
        }
        let p = h.engine.chats.first?.parameters
        #expect(p?.temperature == 0.2 && p?.topK == 7 && p?.topP == 0.5 && p?.minP == 0.1)
        #expect(p?.repetitionPenalty == 1.2 && p?.repetitionWindow == 32 && p?.seed == 42 && p?.maxTokens == 99)
    }

    @Test func setParameterNumCtxIsAcceptedAndOtherInertKeysWarn() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter num_ctx 8192", "/set parameter typical_p 0.5", "/set parameter num_keep 4",
                      "/set parameter stop END", "/set parameter draft_num_predict 3"])
        #expect(h.all.contains("Set parameter 'num_ctx' to '8192'"), "\(h.all)")
        for k in ["typical_p", "num_keep", "stop", "draft_num_predict"] {
            #expect(h.stderr.contains(k) && h.stderr.contains("no effect"), "\(k): \(h.stderr)")
            #expect(!h.all.contains("Set parameter '\(k)'"), "\(k) must not claim success")
        }
        #expect(!h.all.contains("Couldn't set parameter"))
    }

    @Test func setParameterWithAnUnknownKeyExplainsIt() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter bogus 1", "hi"])
        #expect(h.all.contains("Couldn't set parameter: \"unknown parameter 'bogus'\""), "\(h.all)")
        #expect(h.engine.chats.first?.parameters == SamplingParameters(), "nothing changed")
    }

    @Test func parametersTheEngineCannotHonorAreErrors() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter presence_penalty 1", "/set parameter frequency_penalty 1"])
        let errors = h.all.components(separatedBy: "Couldn't set parameter:").count - 1
        #expect(errors == 2, "\(h.all)")
        #expect(h.all.contains("presence_penalty") && h.all.contains("frequency_penalty"))
        #expect(!h.all.contains("Set parameter 'presence_penalty'"))
    }

    @Test func gpuParametersAreWarnedAboutAndIgnored() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter num_gpu 1", "/set parameter main_gpu 0", "/set parameter use_mmap true",
                      "/set parameter num_thread 4", "/set parameter num_batch 512", "hi"])
        for k in ["num_gpu", "main_gpu", "use_mmap", "num_thread", "num_batch"] {
            #expect(h.stderr.contains(k), "a warning names \(k): \(h.stderr)")
        }
        #expect(!h.all.contains("Couldn't set parameter"))
        #expect(h.engine.chats.first?.parameters == SamplingParameters(), "no-op keys change nothing")
    }

    @Test func badValuesAreRejectedWithTheSharedLimits() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter temperature 99", "/set parameter temperature abc", "/set parameter top_k 100001",
                      "/set parameter top_p 0", "/set parameter repeat_penalty 101", "/set parameter num_predict 0",
                      "/set parameter seed -1", "hi"])
        #expect(h.all.components(separatedBy: "Couldn't set parameter:").count - 1 == 7, "\(h.all)")
        #expect(h.engine.chats.first?.parameters == SamplingParameters(), "rejected values change nothing")
    }

    // MARK: /set system and modes

    @Test func setSystemSetsTheSystemMessage() async throws {
        let h = try ChatHarness()
        await h.chat(["/set system You are terse.", "hi"])
        #expect(h.all.contains("Set system message."), "\(h.all)")
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .system, content: "You are terse."),
                                                   ChatMessage(role: .user, content: "hi")])
    }

    @Test func setSystemTakesATripleQuotedMultilineText() async throws {
        let h = try ChatHarness()
        await h.chat(["/set system \"\"\"", "be brief", "and kind\"\"\"", "hi"])
        #expect(h.all.contains("Set system message."))
        #expect(h.engine.chats.first?.messages.first == ChatMessage(role: .system, content: "be brief\nand kind"))
    }

    @Test func setThinkAndNothinkChangeTheThinkingRequest() async throws {
        let h = try ChatHarness()
        await h.chat(["/set think", "a", "/set nothink", "b", "/set think high", "c"])
        #expect(h.all.contains("Set 'think' mode.\n"), "\(h.all)")
        #expect(h.all.contains("Set 'nothink' mode.\n"))
        #expect(h.all.contains("Set 'think' mode to 'high'."))
        #expect(h.engine.chats.map(\.enableThinking) == [true, false, true])
    }

    @Test func setVerboseShowsStatsAfterEachReplyAndQuietStopsThem() async throws {
        let h = try ChatHarness()
        await h.chat(["/set verbose", "a", "/set quiet", "b"])
        #expect(h.all.contains("Set 'verbose' mode.\n") && h.all.contains("Set 'quiet' mode.\n"), "\(h.all)")
        #expect(h.stderr.components(separatedBy: "eval rate:").count - 1 == 2, "one block (prompt and generation rates) for the verbose turn only: \(h.stderr)")
        #expect(!h.stdout.contains("eval rate"), "stats are not payload")
    }

    @Test func wordWrapAndFormatAndHistoryModes() async throws {
        let h = try ChatHarness()
        await h.chat(["/set wordwrap", "/set nowordwrap", "/set history", "/set nohistory",
                      "/set format json", "/set format xml", "/set format", "/set noformat"])
        #expect(h.all.contains("Set 'wordwrap' mode.\n") && h.all.contains("Set 'nowordwrap' mode.\n"), "\(h.all)")
        #expect(h.all.contains("Set format to 'json' mode.\n"))
        #expect(h.all.components(separatedBy: "Invalid or missing format. For 'json' mode use '/set format json'").count - 1 == 2)
        #expect(h.all.contains("Disabled format.\n"))
    }

    @Test func historyAndNohistoryAreSilent() async throws {
        let h = try ChatHarness()
        await h.chat(["/set history", "/set nohistory", "hi"])
        #expect(h.engine.chats.count == 1)
        #expect(h.all == "Hello world\n\n", "\(h.all)")
    }

    @Test func otherSetSubcommandsAreUnknown() async throws {
        let h = try ChatHarness()
        await h.chat(["/set bogus", "/set"])
        #expect(h.all.contains("Unknown command '/set bogus'. Type /? for help"), "\(h.all)")
        #expect(h.engine.chats.isEmpty)
    }

    // MARK: /show

    @Test func showInfoPrintsTheShowTablesOnStderr() async throws {
        let h = try ChatHarness()
        await h.chat(["/show info"])
        #expect(h.stderr.contains("  Model\n    architecture        qwen3\n"), "\(h.stderr)")
        #expect(h.stderr.contains("  Capabilities\n") && h.stderr.contains("  Parameters\n"))
        #expect(h.stdout.isEmpty)
    }

    @Test func showLicenseAndTemplatePrintTheRawText() async throws {
        let h = try ChatHarness()
        await h.chat(["/show license", "/show template"])
        #expect(h.all.contains("Apache License\nVersion 2.0, January 2004\n"), "the whole license, not the excerpt: \(h.all)")
        #expect(h.all.contains("TERMS AND CONDITIONS"))
        #expect(h.all.contains("<|im_start|>{{ role }}"), "\(h.all)")
    }

    @Test func showLicenseWithoutOneSaysSo() async throws {
        let h = try ChatHarness(license: nil)
        await h.chat(["/show license"])
        #expect(h.all.contains("No license was specified for this model.\n"), "\(h.all)")
    }

    @Test func showSystemPrefersTheSessionsThenTheModelsThenSaysNone() async throws {
        let h = try ChatHarness()
        await h.chat(["/show system", "/set system from the session", "/show system"])
        #expect(h.all.contains("No system message was specified for this model.\n"), "\(h.all)")
        #expect(h.all.contains("from the session\n"))
    }

    @Test func showParametersListsModelDefinedThenUserDefinedRows() async throws {
        let h = try ChatHarness()
        await h.chat(["/set parameter temperature 0.2", "/show parameters"])
        let text = h.all
        let model = text.range(of: "Model defined parameters:")
        let user = text.range(of: "User defined parameters:")
        #expect(model != nil && user != nil, "\(text)")
        if let model, let user { #expect(model.lowerBound < user.lowerBound) }
        #expect(text.contains("top_k".padding(toLength: 30, withPad: " ", startingAt: 0) + " 20"), "rows are %-30s %v: \(text)")
        #expect(text.contains("temperature".padding(toLength: 30, withPad: " ", startingAt: 0) + " 0.2"))
    }

    @Test func showModelfilePrintsTheModelfile() async throws {
        let h = try ChatHarness()
        await h.chat(["/show modelfile"])
        #expect(h.all.contains("FROM \(LibraryFixture.qwen06)"), "\(h.all)")
    }

    @Test func showWithAnUnknownSectionIsAnUnknownCommand() async throws {
        let h = try ChatHarness()
        await h.chat(["/show bogus"])
        #expect(h.all.contains("Unknown command '/show bogus'. Type /? for help"), "\(h.all)")
    }
}
