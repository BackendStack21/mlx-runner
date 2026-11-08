import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("run one-shot (AC-14, AC-15)")
struct RunCommandTests {
    let model = "qwen3:0.6b"

    // MARK: input and output

    @Test func promptWordsRunOneGenerationAndStdoutCarriesOnlyTheReply() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "Say", "hi"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "Hello world\n\n")
        #expect(h.engine.loads.count == 1)
        #expect(h.engine.loads.first?.directory == h.modelDir?.directory)
        #expect(h.engine.chats.count == 1)
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "Say hi")])
        #expect(h.engine.completions.isEmpty)
        #expect(h.hub.detailsCalls.isEmpty && h.hub.downloads.isEmpty)
    }

    @Test func theLoadBannerGoesToStderr() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.err.contains("Qwen3-0.6B-4bit"), "\(r.err)")
        #expect(!r.out.contains("Qwen3-0.6B-4bit"))
    }

    @Test func promptFlagIsAnotherWayToGiveThePrompt() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "Say hi"])
        #expect(r.code == 0 && r.out == "Hello world\n\n", "\(r.err)")
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "Say hi")])
    }

    @Test func pipedInputGoesBeforeThePromptWords() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "summarize", "this"], stdin: "A long text")
        #expect(r.code == 0, "\(r.err)")
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "A long text summarize this")])
    }

    @Test func pipedInputAloneIsThePrompt() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model], stdin: "only stdin")
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "only stdin")])
    }

    @Test func anEmptyPromptOnlyLoadsTheModel() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model])
        #expect(r.code == 0 && r.out.isEmpty, "\(r.err)")
        #expect(h.engine.loads.count == 1)
        #expect(h.engine.chats.isEmpty && h.engine.completions.isEmpty)
    }

    @Test func wordsAndPromptFlagTogetherFailBeforeLoading() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "hello", "-p", "x"])
        #expect(r.code == 1 && r.err.hasPrefix("Error: ") && r.err.contains("-p"), "\(r.err)")
        #expect(r.out.isEmpty)
        #expect(h.engine.loads.isEmpty)
    }

    @Test func rawSendsThePromptWithoutTheChatTemplate() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "--raw", "-p", "once upon"])
        #expect(r.code == 0 && r.out == "Hello world\n\n", "\(r.err)")
        #expect(h.engine.completions.first?.prompt == "once upon")
        #expect(h.engine.chats.isEmpty)
    }

    @Test func systemPromptComesFirst() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model, "--system", "Be brief", "-p", "hi"])
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .system, content: "Be brief"), ChatMessage(role: .user, content: "hi")])
    }

    @Test func aFailingStreamIsAnErrorLine() async throws {
        let boom = NSError(domain: "t", code: 1, userInfo: [NSLocalizedDescriptionKey: "boom"])
        let h = try RunHarness(engine: FakeEngine(script: [.text("par"), .fail(boom)]))
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 1 && r.err.contains("Error: boom"), "\(r.err)")
    }

    @Test func aFailingLoadIsAnErrorLine() async throws {
        let h = try RunHarness()
        h.engine.loadError = NSError(domain: "t", code: 1, userInfo: [NSLocalizedDescriptionKey: "no weights"])
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 1 && r.err.contains("Error: no weights"), "\(r.err)")
        #expect(h.engine.chats.isEmpty)
    }

    // MARK: sampling

    @Test func samplingComesFromGenerationConfigThenTheAppDefaults() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model, "-p", "hi"])
        let p = try #require(h.engine.chats.first?.parameters)
        let d = SamplingParameters()
        #expect(p.temperature == 0.6 && p.topK == 20 && p.topP == 0.95)
        #expect(p.minP == d.minP && p.repetitionPenalty == d.repetitionPenalty && p.repetitionWindow == d.repetitionWindow)
        #expect(p.maxTokens == SamplingLimits.maxTokens, "no --max: generate to EOS or the context")
        #expect(p.seed == nil && p.speculative == nil)
    }

    @Test func withoutGenerationConfigTheAppDefaultsApply() async throws {
        let h = try RunHarness(generation: nil)
        _ = await h.run(["run", model, "-p", "hi"])
        let p = try #require(h.engine.chats.first?.parameters)
        let d = SamplingParameters()
        #expect(p.temperature == d.temperature && p.topP == d.topP && p.topK == d.topK && p.minP == d.minP)
        #expect(p.repetitionPenalty == d.repetitionPenalty && p.repetitionWindow == d.repetitionWindow)
    }

    @Test func badGenerationConfigValuesAreDroppedWithAWarning() async throws {
        let h = try RunHarness(generation: #"{"temperature":99,"top_k":20}"#)
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 0 && r.out == "Hello world\n\n", "\(r.err)")
        #expect(r.err.contains("generation_config.json") && r.err.contains("temperature"), "\(r.err)")
        let p = try #require(h.engine.chats.first?.parameters)
        #expect(p.temperature == SamplingParameters().temperature && p.topK == 20)
    }

    @Test func flagsOverrideEveryDefault() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi", "--temp", "0", "--seed", "42", "--max", "256", "--top-k", "5", "--top-p", "0.5",
                             "--min-p", "0.1", "--repeat-penalty", "1.2", "--repeat-last-n", "16", "--spec"])
        #expect(r.code == 0, "\(r.err)")
        let p = try #require(h.engine.chats.first?.parameters)
        #expect(p.temperature == 0 && p.seed == 42 && p.maxTokens == 256 && p.topK == 5 && p.topP == 0.5)
        #expect(p.minP == 0.1 && p.repetitionPenalty == 1.2 && p.repetitionWindow == 16 && p.speculative == true)
    }

    @Test func maxTokensShortFlag() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model, "-p", "hi", "-n", "7"])
        #expect(h.engine.chats.first?.parameters.maxTokens == 7)
    }

    // MARK: load options

    @Test func contextDefaultsTo4096AndFollowsCtx() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model, "-p", "hi"])
        _ = await h.run(["run", model, "-p", "hi", "--ctx", "2048"])
        #expect(h.engine.loads.map(\.contextLength) == [4096, 2048])
    }

    @Test func sparseOptionReachesTheLoad() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model, "-p", "hi", "--sparse", "off"])
        _ = await h.run(["run", model, "-p", "hi", "--sparse", "512"])
        _ = await h.run(["run", model, "-p", "hi"])
        let sparse = h.engine.loads.map(\.sparse)
        guard sparse.count == 3 else { Issue.record("expected 3 loads, got \(sparse.count)"); return }
        #expect(sparse[0] == nil)
        #expect(sparse[1] == SparseAttention.configured(topK: 512))
        #expect(sparse[2] == SparseAttention.fromEnvironment)
    }

    @Test func aBadSparseValueFailsBeforeLoading() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi", "--sparse", "bogus"])
        #expect(r.code == 1 && r.err.contains("--sparse takes on, off or a number"), "\(r.err)")
        #expect(h.engine.loads.isEmpty)
    }

    @Test func draftModelsResolveThroughTheResolver() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model, "-p", "hi", "--draft", "qwen3:0.6b"])
        #expect(h.engine.loads.first?.draftDirectory == h.modelDir?.directory)
        let missing = await h.run(["run", model, "-p", "hi", "--draft", "qwen3:4b"])
        #expect(missing.code == 1 && missing.err.contains("mlxr pull"), "\(missing.err)")
        #expect(h.engine.loads.count == 1)
    }

    @Test func draftAutoUsesThePairedModelOrSaysThereIsNone() async throws {
        let h = try RunHarness()
        h.engine.draft = LibraryFixture.model(LibraryFixture.qwen06, bytes: 1, ageSeconds: 1)
        _ = await h.run(["run", model, "-p", "hi", "--draft", "auto"])
        #expect(h.engine.loads.last?.draftDirectory == URL(fileURLWithPath: "/fixture/mlx-community/Qwen3-0.6B-4bit"))
        h.engine.draft = nil
        let r = await h.run(["run", model, "-p", "hi", "--draft", "auto"])
        #expect(r.code == 0 && h.engine.loads.last?.draftDirectory == nil)
        #expect(r.err.contains("no installed draft model") && !r.out.contains("no installed draft model"), "\(r.err)")
    }

    // MARK: thinking

    @Test func thinkingFlagsReachTheEngine() async throws {
        let h = try RunHarness()
        let cases: [([String], Bool?)] = [
            (["-p", "hi"], nil), (["-p", "hi", "--think"], true), (["-p", "hi", "--think=true"], true),
            (["-p", "hi", "--think=high"], true), (["-p", "hi", "--think=low"], true),
            (["-p", "hi", "--think=false"], false), (["-p", "hi", "--no-think"], false),
        ]
        for (args, expected) in cases {
            h.engine.script = FakeEngine.reply()
            _ = await h.run(["run", model] + args)
            #expect(h.engine.chats.last?.enableThinking == expected, "\(args)")
        }
    }

    @Test func thinkWithoutEqualsLeavesTheWordAsPromptText() async throws {
        let h = try RunHarness()
        _ = await h.run(["run", model, "--think", "say", "hello"])
        #expect(h.engine.chats.first?.enableThinking == true)
        #expect(h.engine.chats.first?.messages == [ChatMessage(role: .user, content: "say hello")])
    }

    static let thinkingReply: [FakeEngine.Step] = [
        .promptProcessed(FakeEngine.stats(generated: 0, rate: 0)), .text("<think>\nplan"), .text("\n</think>\n\nAnswer"),
        .finished(FakeEngine.stats()),
    ]

    @Test func reasoningIsShownBetweenThinkingMarkers() async throws {
        let h = try RunHarness(engine: FakeEngine(script: Self.thinkingReply, info: FakeEngine.model(thinksByDefault: true)))
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "Thinking...\nplan\n...done thinking.\n\nAnswer\n\n")
    }

    @Test func hideThinkingKeepsOnlyTheAnswer() async throws {
        let h = try RunHarness(engine: FakeEngine(script: Self.thinkingReply, info: FakeEngine.model(thinksByDefault: true)))
        let r = await h.run(["run", model, "-p", "hi", "--hidethinking"])
        #expect(r.code == 0 && r.out == "Answer\n\n", "\(r.err) \(r.out)")
    }

    // MARK: word wrap

    @Test func wordWrapOnATerminalUnlessNoWordWrap() async throws {
        let text = "hello wonderful world"
        let h = try RunHarness(engine: FakeEngine(script: FakeEngine.reply([text])))
        let wrapped = await h.run(["run", model, "-p", "hi"], stdoutTTY: true, width: 20)
        #expect(wrapped.out == "hello wonderful\u{1B}[9D\u{1B}[K\nwonderful world\n\n", "\(wrapped.err)")
        let plain = await h.run(["run", model, "-p", "hi", "--nowordwrap"], stdoutTTY: true, width: 20)
        #expect(plain.out == text + "\n\n", "\(plain.err)")
        let piped = await h.run(["run", model, "-p", "hi"], stdoutTTY: false, width: 20)
        #expect(piped.out == text + "\n\n")
    }

    // MARK: verbose

    @Test func verbosePrintsOllamasBlockThenMlxrLinesOnStderr() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi", "--verbose"])
        #expect(r.code == 0 && r.out == "Hello world\n\n", "\(r.err)")
        let block = StatsFormat.block(RunStats(loadSeconds: 1.5, stats: FakeEngine.stats()))
        let lines = StatsFormat.mlxrLines(prompt: FakeEngine.stats(generated: 0, rate: 0), final: FakeEngine.stats())
        guard lines.count == 2 else { Issue.record("expected 2 mlxr lines, got \(lines.count)"); return }
        let a = try #require(r.err.range(of: block), "\(r.err)")
        let b = try #require(r.err.range(of: lines[0]))
        let c = try #require(r.err.range(of: lines[1]))
        #expect(a.upperBound <= b.lowerBound && b.upperBound <= c.lowerBound)
    }

    @Test func withoutVerboseNoStatsAreShown() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 0 && r.out == "Hello world\n\n" && !r.err.contains("total duration") && !r.err.contains("[generated") && !r.err.contains("[prompt:"), "\(r.err)")
    }

    // MARK: accepted and rejected flags

    @Test func theRunTableHasOllamasFlags() throws {
        let run = try #require(CLI.commands.first { $0.name == "run" })
        let names = Set(run.options.map(\.long))
        for flag in ["verbose", "think", "hidethinking", "nowordwrap", "keepalive", "insecure", "format", "prompt", "system", "temp", "top-p",
                     "top-k", "min-p", "repeat-penalty", "repeat-last-n", "seed", "max-tokens", "ctx", "raw", "draft", "sparse", "spec", "ids", "no-think"] {
            #expect(names.contains(flag), "run has no --\(flag)")
        }
    }

    @Test func keepaliveAndInsecureWarnAndAreIgnored() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi", "--keepalive", "5m", "--insecure"])
        #expect(r.code == 0 && r.out == "Hello world\n\n", "\(r.err)")
        #expect(r.err.contains("warning: --keepalive has no effect"), "\(r.err)")
        #expect(r.err.contains("warning: --insecure has no effect"), "\(r.err)")
    }

    @Test func unsupportedFlagsAreClearErrorsBeforeLoading() async throws {
        let cases: [(flag: String, args: [String])] = [
            ("--format", ["--format", "json"]), ("--presence-penalty", ["--presence-penalty", "1"]),
            ("--frequency-penalty", ["--frequency-penalty", "1"]), ("--truncate", ["--truncate"]),
            ("--dimensions", ["--dimensions", "3"]), ("--voice", ["--voice", "v"]), ("--ask", ["--ask", "q"]),
        ]
        for c in cases {
            let h = try RunHarness()
            let r = await h.run(["run", model, "-p", "hi"] + c.args)
            #expect(r.code == 1, "\(c.flag): \(r.err)")
            #expect(r.err.hasPrefix("Error: ") && r.err.contains(c.flag) && !r.err.contains("unknown flag"), "\(c.flag): \(r.err)")
            #expect(r.out.isEmpty && h.engine.loads.isEmpty, "\(c.flag)")
        }
        let h = try RunHarness()
        let f = await h.run(["run", model, "-p", "hi", "--format", "json"])
        #expect(f.err.contains("not supported"), "\(f.err)")
        let p = await h.run(["run", model, "-p", "hi", "--presence-penalty", "1"])
        #expect(p.err.contains("not supported"), "\(p.err)")
    }

    @Test func invalidNumbersFailBeforeLoading() async throws {
        let h = try RunHarness()
        for bad in [["--max", "0"], ["--temp", "nan"], ["--ctx", "255"], ["--top-k", "100001"]] {
            let r = await h.run(["run", model, "-p", "hi"] + bad)
            #expect(r.code == 1 && r.err.hasPrefix("Error: "), "\(bad)")
        }
        #expect(h.engine.loads.isEmpty)
    }
}
