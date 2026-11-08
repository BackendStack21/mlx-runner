import Foundation
import Testing
@testable import MLXRCLI

@Suite("ps, stop, serve without a model and commands mlxr does not have (AC-19)")
struct UnsupportedCommandTests {
    func run(_ args: [String]) async -> (code: Int32, out: String, err: String) {
        await LibraryFixture.standard().run(args)
    }

    @Test func psPrintsOnlyTheHeader() async {
        for args in [["ps"], ["ps", "qwen"]] {
            let r = await run(args)
            #expect(r.code == 0, "\(r.err)")
            #expect(r.out == "NAME    ID    SIZE    PROCESSOR    CONTEXT    RUNNER    UNTIL \n")
            #expect(r.err.isEmpty)
        }
    }

    @Test func stopSaysNothingIsRunning() async {
        let r = await run(["stop", "qwen3:8b"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: couldn't find model \"qwen3:8b\" to stop"), "\(r.err)")
        #expect(r.err.contains("mlxr serve"), "explains there is no background server: \(r.err)")
        #expect(r.err.hasSuffix("\n"))
        #expect(await run(["stop"]).err == "Error: accepts 1 arg(s), received 0\n")
    }

    @Test func serveWithoutAModelPointsAtTheNextPhase() async {
        for args in [["serve"], ["start"], ["serve", "--port", "8080"]] {
            let r = await run(args)
            #expect(r.code == 1 && r.out.isEmpty, "\(args)")
            #expect(r.err.hasPrefix("Error: "), "\(args): \(r.err)")
            #expect(!r.err.contains("accepts 1 arg"), "\(args): \(r.err)")
            #expect(r.err.contains("mlxr serve MODEL"), "\(args): \(r.err)")
        }
    }

    @Test func commandsOllamaHasAndMlxrDoesNot() async {
        let pull = "Error: 'create' is not supported by mlxr; use 'mlxr pull <hf-repo>'\n"
        let cases: [([String], String)] = [
            (["create", "m"], pull),
            (["create", "m", "-f", "Modelfile"], pull),
            (["push", "m"], "Error: 'push' is not supported by mlxr\n"),
            (["signin"], "Error: 'signin' is not supported by mlxr\n"),
            (["login"], "Error: 'signin' is not supported by mlxr\n"),
            (["signout"], "Error: 'signout' is not supported by mlxr\n"),
            (["logout"], "Error: 'signout' is not supported by mlxr\n"),
            (["launch"], "Error: 'launch' is not supported by mlxr\n"),
            (["launch", "claude", "-m", "x", "--yes"], "Error: 'launch' is not supported by mlxr\n"),
        ]
        for (args, want) in cases {
            let r = await run(args)
            #expect(r.code == 1 && r.out.isEmpty, "\(args)")
            #expect(r.err == want, "\(args): \(r.err)")
        }
    }
}
