import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("pull is the command, download its alias (REQ-16); show strips terminal controls")
struct PullNamingAndSanitizingTests {
    static let repo = "mlx-community/Qwen3-8B-4bit"

    @Test func pullIsTheCommandNameAndDownloadItsAlias() throws {
        #expect(try CLI.parse(["pull", "a/b"]).command.name == "pull")
        #expect(try CLI.parse(["download", "a/b"]).command.name == "pull")
        #expect(CLI.commands.first { $0.name == "pull" }?.aliases == ["download"])
        #expect(!CLI.commands.contains { $0.name == "download" })
    }

    @Test func helpAndUsageNamePull() {
        let root = CLI.rootHelp(commands: CLI.commands)
        #expect(root.contains("\n  pull "), "\(root)")
        #expect(!root.contains("\n  download "), "\(root)")
        let pull = CLI.commands.first { $0.name == "pull" }
        let help = pull.map(CLI.commandHelp) ?? ""
        #expect(help.contains("Usage:\n  mlxr pull MODEL [flags]"), "\(help)")
        #expect(help.contains("Aliases:\n  pull, download\n"), "\(help)")
    }

    @Test func downloadStillPullsAModel() async {
        let h = CLIHarness()
        let r = await h.run(["download", "qwen3:8b"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == h.hub.destination(Self.repo).path + "\n")
        let wrongCount = await h.run(["download"])
        #expect(wrongCount.err == "Error: accepts 1 arg(s), received 0\n")
    }

    // MARK: terminal control characters from model files

    static let hostile = "A\u{1B}[31mRED\u{1B}[0m\u{07}\u{08}B\r\n\tC\u{7F}\u{00}D\n"

    func harness(_ dir: ShowModelDir) -> CLIHarness {
        CLIHarness(models: [LocalModel(repo: Self.repo, directory: dir.directory, sizeBytes: 1, modelType: "qwen3",
                                       bits: 4, source: .library, modified: LibraryFixture.now)])
    }

    func noControls(_ s: String, _ what: String) {
        let bad = s.unicodeScalars.filter { ($0.value < 0x20 && $0 != "\n" && $0 != "\t") || $0.value == 0x7F }
        #expect(bad.isEmpty, "\(what) still has control characters: \(bad.map(\.value))")
    }

    @Test func licenseIsStrippedOfControlCharactersButKeepsNewlinesAndTabs() async throws {
        let dir = try ShowModelDir(license: Self.hostile); defer { dir.remove() }
        let h = harness(dir)
        let raw = await h.run(["show", Self.repo, "--license"])
        #expect(raw.code == 0, "\(raw.err)")
        noControls(raw.out, "--license")
        #expect(raw.out == "A[31mRED[0mB\n\tCD\n", "\(raw.out.debugDescription)")
        let info = await h.run(["show", Self.repo])
        noControls(info.out, "info tables")
        #expect(info.out.contains("    A[31mRED[0mB\n    \tCD\n"), "\(info.out.debugDescription)")
        let json = await h.run(["show", Self.repo, "--json"])
        noControls(json.out, "--json")
    }

    @Test func templateIsStripped() async throws {
        let cfg = String(decoding: try JSONSerialization.data(withJSONObject: ["chat_template": "<|im_start|>\u{1B}]0;pwned\u{07}{{ content }}\r\n"]), as: UTF8.self)
        let dir = try ShowModelDir(tokenizerConfig: cfg); defer { dir.remove() }
        let h = harness(dir)
        let t = await h.run(["show", Self.repo, "--template"])
        noControls(t.out, "--template")
        #expect(t.out == "<|im_start|>]0;pwned{{ content }}\n", "\(t.out.debugDescription)")
        let m = await h.run(["show", Self.repo, "--modelfile"])
        noControls(m.out, "--modelfile")
    }

    @Test func systemTextIsStripped() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let h = harness(dir)
        try h.library.presets.save(Preset(name: "evil", repo: Self.repo, system: "Be\u{1B}[2J nice\u{07}", parameters: ["top_k\u{1B}": "5\u{1B}"]))
        for flag in ["--system", "--modelfile", "--parameters", "--json"] {
            let r = await h.run(["show", "evil", flag])
            noControls(r.out, flag)
        }
        let info = await h.run(["show", "evil"])
        noControls(info.out, "info tables")
        #expect(info.out.contains("  System\n    Be[2J nice\n"), "\(info.out.debugDescription)")
    }

    // MARK: show -v

    @Test func verboseAddsAMetadataSectionOfTheScalarConfigKeys() async throws {
        let dir = try ShowModelDir(); defer { dir.remove() }
        let h = harness(dir)
        let plain = await h.run(["show", Self.repo])
        #expect(!plain.out.contains("Metadata"))
        let r = await h.run(["show", Self.repo, "-v"])
        #expect(r.code == 0, "\(r.err)")
        func row(_ k: String, _ v: String) -> String { "    " + k.padding(toLength: 27, withPad: " ", startingAt: 0) + v + "\n" }
        let metadata = "  Metadata\n" + row("hidden_size", "4096") + row("max_position_embeddings", "40960") + row("model_type", "qwen3") + "\n"
        #expect(r.out.contains(metadata), "\(r.out)")
        let parameters = try #require(r.out.range(of: "  Parameters\n")), license = try #require(r.out.range(of: "  License\n"))
        let meta = try #require(r.out.range(of: "  Metadata\n"))
        #expect(parameters.lowerBound < meta.lowerBound && meta.lowerBound < license.lowerBound, "Metadata sits between Parameters and License")
        #expect(r.out.replacingOccurrences(of: metadata, with: "") == plain.out)
    }
}
