import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("show (AC-11)")
struct ShowCommandTests {
    static let repo = "mlx-community/Qwen3-8B-4bit"

    /// A harness whose only installed model is `dir`.
    func harness(_ dir: ShowModelDir) -> CLIHarness {
        CLIHarness(models: [LocalModel(repo: Self.repo, directory: dir.directory, sizeBytes: 5_000_000_000, modelType: "qwen3",
                                       bits: 4, source: .library, modified: LibraryFixture.now)])
    }

    static let defaultInfo = """
          Model
            architecture        qwen3
            context length      40960
            embedding length    4096
            quantization        4-bit

          Capabilities
            completion
            tools
            thinking

          Parameters
            temperature    0.6
            top_k          20
            top_p          0.95

          License
            Apache License
            Version 2.0, January 2004
            ...


        """

    @Test func printsTheSectionsInOllamasOrderWithIndentAndBlankLines() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let r = await harness(dir).run(["show", Self.repo])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == Self.defaultInfo, "\(r.out)")
        #expect(r.err.isEmpty)
    }

    @Test func aPresetAddsItsSystemPromptAndOverridesParameters() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let h = harness(dir)
        try h.library.presets.save(Preset(name: "mine", repo: Self.repo, system: "Be brief.\nAnswer in one line.\nNo emoji.",
                                          parameters: ["temperature": "0.2"]))
        let r = await h.run(["show", "mine"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out.contains("  Parameters\n    temperature    0.2\n    top_k          20\n    top_p          0.95\n\n"), "\(r.out)")
        #expect(r.out.contains("  System\n    Be brief.\n    Answer in one line.\n    ...\n\n  License\n"), "\(r.out)")
        let two = Preset(name: "two", repo: Self.repo, system: "One.\nTwo.")
        try h.library.presets.save(two)
        let r2 = await h.run(["show", "two"])
        #expect(r2.out.contains("  System\n    One.\n    Two.\n\n"), "two lines show no ellipsis: \(r2.out)")
    }

    @Test func aTemplateWithoutThinkingOrToolsOnlyCompletes() async throws {
        let dir = try ShowModelDir(tokenizerConfig: #"{"chat_template":"<|im_start|>{{ content }}<|im_end|>"}"#, generation: nil, license: nil)
        defer { dir.remove() }
        let r = await harness(dir).run(["show", Self.repo])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "  Model\n    architecture        qwen3\n    context length      40960\n    embedding length    4096\n    quantization        4-bit\n\n  Capabilities\n    completion\n\n", "\(r.out)")
    }

    @Test func eachSingleSectionFlagPrintsOnlyItsContent() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let h = harness(dir)
        try h.library.presets.save(Preset(name: "mine", repo: Self.repo, system: "Be brief.", parameters: ["temperature": "0.2"]))
        let license = await h.run(["show", Self.repo, "--license"])
        #expect(license.out == ShowModelDir.license && license.err.isEmpty, "\(license.out)")
        let template = await h.run(["show", Self.repo, "--template"])
        #expect(template.out == ShowModelDir.template + "\n", "\(template.out)")
        let system = await h.run(["show", "mine", "--system"])
        #expect(system.out == "Be brief.\n", "\(system.out)")
        let params = await h.run(["show", "mine", "--parameters"])
        func row(_ k: String, _ v: String) -> String { k.padding(toLength: 30, withPad: " ", startingAt: 0) + " " + v + "\n" }
        #expect(params.out == row("temperature", "0.2") + row("top_k", "20") + row("top_p", "0.95"), "\(params.out)")
        let modelfile = await h.run(["show", "mine", "--modelfile"])
        #expect(modelfile.code == 0, "\(modelfile.err)")
        #expect(modelfile.out.contains("FROM \(Self.repo)\n"), "\(modelfile.out)")
        #expect(modelfile.out.contains("TEMPLATE \"\"\"\(ShowModelDir.template)\"\"\""), "\(modelfile.out)")
        #expect(modelfile.out.contains("SYSTEM \"\"\"Be brief.\"\"\""), "\(modelfile.out)")
        #expect(modelfile.out.contains("PARAMETER temperature 0.2\n"), "\(modelfile.out)")
        for r in [license, template, system, params, modelfile] { #expect(!r.out.contains("  Model\n")) }
    }

    @Test func missingPartsUseOllamasMessages() async throws {
        let dir = try ShowModelDir(tokenizerConfig: "{}", generation: nil, license: nil); defer { dir.remove() }
        let h = harness(dir)
        let license = await h.run(["show", Self.repo, "--license"])
        #expect(license.out == "No license was specified for this model.\n", "\(license.out)")
        let template = await h.run(["show", Self.repo, "--template"])
        #expect(template.out == "No prompt template was specified for this model.\n", "\(template.out)")
        let system = await h.run(["show", Self.repo, "--system"])
        #expect(system.out == "No system message was specified for this model.\n", "\(system.out)")
        #expect([license, template, system].allSatisfy { $0.code == 0 && $0.err.isEmpty })
    }

    @Test func twoSectionFlagsAreAnErrorAndPrintNothing() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let r = await harness(dir).run(["show", Self.repo, "--license", "--template"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: only one of") && r.err.contains("--license") && r.err.contains("--template"), "\(r.err)")
        let verbose = await harness(dir).run(["show", Self.repo, "--system", "-v"])
        #expect(verbose.code == 0, "-v is not a section flag: \(verbose.err)")
    }

    @Test func readsTheModelWithoutOpeningItsWeights() async throws {
        let dir = try ShowModelDir(weightsReadable: false); defer { dir.remove() }
        let weights = dir.directory.appendingPathComponent("model.safetensors")
        #expect(!FileManager.default.isReadableFile(atPath: weights.path), "the fixture weights cannot be read")
        let r = await harness(dir).run(["show", Self.repo])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out.hasPrefix("  Model\n"))
    }

    @Test func aBadGenerationConfigValueIsDroppedWithAWarning() async throws {
        let dir = try ShowModelDir(generation: #"{"temperature":99,"top_k":20}"#); defer { dir.remove() }
        let r = await harness(dir).run(["show", Self.repo])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out.contains("  Parameters\n    top_k    20\n\n"), "\(r.out)")
        #expect(!r.out.contains("temperature"))
        #expect(r.err.contains("generation_config.json") && r.err.contains("temperature"), "\(r.err)")
    }

    @Test func aPathShowsTheModelInThatDirectory() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let r = await CLIHarness().run(["show", dir.directory.path])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == Self.defaultInfo, "\(r.out)")
    }

    @Test func jsonHasTheArchitectureAndCapabilities() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let r = await harness(dir).run(["show", Self.repo, "--json"])
        #expect(r.code == 0 && r.err.isEmpty, "\(r.err)")
        let obj = try #require(JSONSerialization.jsonObject(with: Data(r.out.utf8)) as? [String: Any])
        #expect(obj["capabilities"] as? [String] == ["completion", "tools", "thinking"])
        #expect((obj["details"] as? [String: Any])?["architecture"] as? String == "qwen3")
    }

    @Test func notInstalledBrokenAndUnparsableModelsAreErrors() async throws {
        let h = CLIHarness()
        let missing = await h.run(["show", "qwen3:4b"])
        #expect(missing.code == 1 && missing.out.isEmpty)
        #expect(missing.err.hasPrefix("Error: model 'qwen3:4b' not found"), "\(missing.err)")
        #expect(h.hub.detailsCalls.isEmpty, "show never asks the network")
        let noConfig = try ShowModelDir(config: nil); defer { noConfig.remove() }
        let a = await harness(noConfig).run(["show", Self.repo])
        #expect(a.code == 1 && a.out.isEmpty && a.err.hasPrefix("Error: ") && a.err.contains("config.json"), "\(a.err)")
        let garbage = try ShowModelDir(config: "{not json"); defer { garbage.remove() }
        let b = await harness(garbage).run(["show", Self.repo])
        #expect(b.code == 1 && b.out.isEmpty && b.err.hasPrefix("Error: ") && b.err.contains("config.json"), "\(b.err)")
    }
}
