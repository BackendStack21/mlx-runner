import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("pull and download (AC-13, AC-5)")
struct PullCommandTests {
    static let qwen8 = "mlx-community/Qwen3-8B-4bit"

    @Test func pullPrintsOllamaProgressOnStderrAndOnlyThePathOnStdout() async {
        let h = CLIHarness()
        let r = await h.run(["pull", "qwen3:8b"])
        let dest = h.hub.destination(Self.qwen8)
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == dest.path + "\n")
        #expect(r.err.hasPrefix("pulling manifest\n"), "\(r.err)")
        #expect(r.err.hasSuffix("verifying sha256 digest\nwriting manifest\nsuccess\n"), "\(r.err)")
        #expect(r.err.contains(PullProgress.line(label: "pulling model:", done: 1_000_000_000, total: 1_000_000_000, speed: 50_000_000)))
        #expect(h.hub.detailsCalls == [Self.qwen8])
        #expect(h.hub.downloads.map(\.repo) == [Self.qwen8])
        #expect(h.hub.downloads.first?.destination == dest)
    }

    @Test func downloadIsTheSameCommandAsPull() async {
        let h = CLIHarness()
        let r = await h.run(["download", "hf.co/acme/Custom-4bit"])
        #expect(r.code == 0, "\(r.err)")
        #expect(h.hub.downloads.map(\.repo) == ["acme/Custom-4bit"])
        #expect(r.out == h.hub.destination("acme/Custom-4bit").path + "\n")
    }

    @Test func carriageReturnRedrawsOnlyWhenStderrIsATerminal() async {
        let plain = await CLIHarness().run(["pull", "qwen3:8b"])
        #expect(!plain.err.contains("\r") && !plain.out.contains("\r"))
        let tty = await CLIHarness().run(["pull", "qwen3:8b"], stderrTTY: true)
        #expect(tty.code == 0, "\(tty.err)")
        #expect(tty.err.contains("\r"))
        #expect(tty.out.hasSuffix("\n") && !tty.out.contains("%") && !tty.out.contains("pulling"), "\(tty.out)")
    }

    @Test func aCompleteInstalledModelIsSkipped() async {
        let installed = LibraryFixture.model(Self.qwen8, bytes: 5_000_000_000, ageSeconds: 10)
        let h = CLIHarness(models: [installed])
        let r = await h.run(["pull", Self.qwen8.lowercased()])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == installed.directory.path + "\n")
        #expect(r.err.hasSuffix("success\n"), "\(r.err)")
        #expect(h.hub.detailsCalls.isEmpty && h.hub.downloads.isEmpty, "no network for a model that is already complete")
    }

    @Test func aNameThatDoesNotResolveNeverReachesTheHub() async {
        let h = CLIHarness()
        let r = await h.run(["pull", "qwen33"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: model 'qwen33' not found"), "\(r.err)")
        #expect(h.hub.detailsCalls.isEmpty)
        let bad = await h.run(["pull", "hf.co/../x"])
        #expect(bad.err == "Error: invalid repo id: ../x (expected owner/name)\n", "\(bad.err)")
        let deep = await h.run(["pull", "a/b/c"])
        #expect(deep.code == 1 && deep.err.hasPrefix("Error: invalid repo id"), "\(deep.err)")
        #expect(h.hub.detailsCalls.isEmpty && h.hub.downloads.isEmpty)
    }

    @Test func huggingFace401And404MapToTheHint() async {
        for status in [401, 404] {
            let err = NSError(domain: "HuggingFace", code: status, userInfo: [NSLocalizedDescriptionKey: "HTTP \(status)"])
            let onDetails = FakeHub(); onDetails.detailsError = err
            let a = await CLIHarness(hub: onDetails).run(["pull", "acme/Private-4bit"])
            #expect(a.code == 1 && a.out.isEmpty)
            #expect(a.err.hasSuffix("Error: not found, or private/gated (set HF_TOKEN)\n"), "\(a.err)")
            let onDownload = FakeHub(); onDownload.downloadError = err
            let b = await CLIHarness(hub: onDownload).run(["pull", "acme/Private-4bit"])
            #expect(b.code == 1 && b.out.isEmpty)
            #expect(b.err.hasSuffix("Error: not found, or private/gated (set HF_TOKEN)\n"), "\(b.err)")
        }
    }

    @Test func otherDownloadFailuresSayDownloadFailedAndPrintNoPath() async {
        struct Disk: LocalizedError { var errorDescription: String? { "disk full" } }
        let hub = FakeHub(); hub.downloadError = Disk()
        let r = await CLIHarness(hub: hub).run(["pull", "qwen3:8b"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasSuffix("Error: download failed: disk full\n"), "\(r.err)")
        #expect(!r.err.contains("success"))
    }

    @Test func anUnsupportedArchitectureWarnsOnStderrAndStillDownloads() async {
        let hub = FakeHub(); hub.modelType = "mamba"
        let r = await CLIHarness(hub: hub).run(["pull", "acme/Odd-4bit"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.err.contains("warning:") && r.err.contains("mamba"), "\(r.err)")
        #expect(r.out == hub.destination("acme/Odd-4bit").path + "\n")
    }

    @Test func insecureIsAcceptedWithAWarning() async {
        let r = await CLIHarness().run(["pull", "--insecure", "qwen3:8b"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.err.contains("--insecure"), "\(r.err)")
    }

    @Test func argumentCountIsCheckedBeforeAnyRequest() async {
        let h = CLIHarness()
        let r = await h.run(["pull"])
        #expect(r.err == "Error: accepts 1 arg(s), received 0\n")
        #expect(h.hub.detailsCalls.isEmpty)
    }
}
