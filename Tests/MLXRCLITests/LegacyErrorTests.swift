import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("Legacy command errors reach CLI.run")
struct LegacyErrorTests {
    func message(_ body: () throws -> Void) -> String? {
        do { try body(); return nil } catch { return CLI.message(for: error) }
    }

    @Test func modelNotFound() {
        let m = message { _ = try resolveModel("definitely-not-installed/model-xyz") }
        #expect(m == "model not found: definitely-not-installed/model-xyz (try `mlxr download definitely-not-installed/model-xyz`)")
    }

    @Test func existingPathResolves() throws {
        let dir = FileManager.default.temporaryDirectory
        #expect(try resolveModel(dir.path).path == dir.path)
    }

    @Test func sparseValues() throws {
        #expect(try sparseSetting("off") == nil)
        #expect(try sparseSetting("0") == nil)
        #expect(try sparseSetting("on") != nil)
        #expect(try sparseSetting("12") != nil)
        for bad in ["bogus", "-3", ""] {
            #expect(message { _ = try sparseSetting(bad) } == "--sparse takes on, off or a number of long-range keys", "\(bad)")
        }
    }

    @Test func invalidRepoID() throws {
        try validateRepoID("mlx-community/Qwen3-4B-4bit")
        for bad in ["../x", "a/b/c", "noslash"] {
            #expect(message { try validateRepoID(bad) } == "invalid repo id: \(bad) (expected owner/name)", "\(bad)")
        }
    }

    @Test func serverStartFailure() {
        struct Busy: LocalizedError { var errorDescription: String? { "address in use" } }
        #expect(message { try startServer { throw Busy() } } == "cannot start server: address in use")
        #expect(message { try startServer {} } == nil)
    }

    func downloadCommand(_ error: Error) -> Command {
        Command(name: "dl", summary: "x", run: { _, _ in
            _ = try await performDownload(start: { throw error })
        })
    }

    @Test func downloadHuggingFaceNotFoundGivesTheHint() async {
        for status in [401, 404] {
            let err = NSError(domain: "HuggingFace", code: status, userInfo: [NSLocalizedDescriptionKey: "HTTP \(status)"])
            let cap = Capture()
            #expect(await CLI.run(["dl"], console: cap.console(), commands: [downloadCommand(err)]) == 1)
            #expect(cap.stderr == "Error: not found, or private/gated (set HF_TOKEN)\n")
            #expect(cap.stdout.isEmpty)
        }
    }

    @Test func otherDownloadFailuresSayDownloadFailed() async {
        struct Disk: LocalizedError { var errorDescription: String? { "disk full" } }
        let cap = Capture()
        #expect(await CLI.run(["dl"], console: cap.console(), commands: [downloadCommand(Disk())]) == 1)
        #expect(cap.stderr == "Error: download failed: disk full\n")
        let other = NSError(domain: "HuggingFace", code: 500, userInfo: [NSLocalizedDescriptionKey: "HTTP 500"])
        let cap2 = Capture()
        #expect(await CLI.run(["dl"], console: cap2.console(), commands: [downloadCommand(other)]) == 1)
        #expect(cap2.stderr == "Error: download failed: HTTP 500\n")
    }

    @Test func downloadSuccessReturnsTheDirectory() async throws {
        let url = URL(fileURLWithPath: "/tmp/x")
        #expect(try await performDownload(start: { url }) == url)
    }

    @Test func shortFlagErrorsNameBothForms() async {
        let cap = Capture()
        #expect(await CLI.run(["run", "m", "-n", "abc"], console: cap.console()) == 1)
        #expect(cap.stderr.hasPrefix("Error: invalid argument \"abc\" for \"-n, --max-tokens\" flag"), "\(cap.stderr)")
        let cap2 = Capture()
        #expect(await CLI.run(["run", "m", "--max", "abc"], console: cap2.console()) == 1)
        #expect(cap2.stderr.hasPrefix("Error: invalid argument \"abc\" for \"--max-tokens\" flag"), "\(cap2.stderr)")
    }
}
