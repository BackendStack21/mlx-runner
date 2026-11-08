import Foundation
import Testing
@testable import MLXRCLI

@Suite("Errors and exit codes")
struct ErrorTests {
    @Test func unknownCommand() async {
        let cap = Capture()
        let code = await CLI.run(["frobnicate"], console: cap.console())
        #expect(code == 1)
        #expect(cap.stderr == "Error: unknown command \"frobnicate\" for \"mlxr\"\n")
        #expect(cap.stdout.isEmpty)
    }

    @Test func unknownCommandSuggestsTheClosest() async {
        let cap = Capture()
        let code = await CLI.run(["lst"], console: cap.console())
        #expect(code == 1)
        #expect(cap.stderr.hasPrefix("Error: unknown command \"lst\" for \"mlxr\"\n\nDid you mean this?\n\tlist\n"), "\(cap.stderr)")
        #expect(cap.stdout.isEmpty)
    }

    @Test func unknownFlag() async {
        let cap = Capture()
        let code = await CLI.run(["list", "--x"], console: cap.console())
        #expect(code == 1)
        #expect(cap.stderr == "Error: unknown flag: --x\n")
        #expect(cap.stdout.isEmpty)
    }

    @Test func wrongArgumentCountOnRealTable() async {
        let cap = Capture()
        let code = await CLI.run(["show"], console: cap.console())
        #expect(code == 1)
        #expect(cap.stderr == "Error: accepts 1 arg(s), received 0\n")
        #expect(cap.stdout.isEmpty)
    }

    @Test func throwingCommandPrintsErrorAndExits1() async {
        let cmd = Command(name: "boom", summary: "x", run: { _, _ in throw CLIError.failure("it broke") })
        let cap = Capture()
        let code = await CLI.run(["boom"], console: cap.console(), commands: [cmd])
        #expect(code == 1)
        #expect(cap.stderr == "Error: it broke\n")
        #expect(cap.stdout.isEmpty)
    }

    @Test func foreignErrorsUseTheirDescription() async {
        struct Oops: LocalizedError { var errorDescription: String? { "disk on fire" } }
        let cmd = Command(name: "boom", summary: "x", run: { _, _ in throw Oops() })
        let cap = Capture()
        let code = await CLI.run(["boom"], console: cap.console(), commands: [cmd])
        #expect(code == 1)
        #expect(cap.stderr == "Error: disk on fire\n")
    }

    @Test func huggingFace401And404MapToOneHint() async {
        let hint = "not found, or private/gated (set HF_TOKEN)"
        for status in [401, 404] {
            let err = NSError(domain: "HuggingFace", code: status,
                              userInfo: [NSLocalizedDescriptionKey: "Hugging Face returned HTTP \(status)"])
            #expect(CLI.message(for: err) == hint)
            let cmd = Command(name: "boom", summary: "x", run: { _, _ in throw err })
            let cap = Capture()
            #expect(await CLI.run(["boom"], console: cap.console(), commands: [cmd]) == 1)
            #expect(cap.stderr == "Error: \(hint)\n")
        }
    }

    @Test func otherHuggingFaceErrorsKeepTheirMessage() {
        let err = NSError(domain: "HuggingFace", code: 500,
                          userInfo: [NSLocalizedDescriptionKey: "Hugging Face returned HTTP 500"])
        #expect(CLI.message(for: err) == "Hugging Face returned HTTP 500")
        #expect(CLI.message(for: CLIError.failure("plain")) == "plain")
    }

    @Test func errorsNeverPrintUsage() async {
        let cap = Capture()
        _ = await CLI.run(["list", "--x"], console: cap.console())
        #expect(!cap.stderr.contains("Usage:"))
        #expect(!cap.stderr.contains("Fatal error"))
    }
}
