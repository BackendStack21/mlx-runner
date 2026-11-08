import Foundation
import Testing
@testable import MLXRCLI

@Suite("Command table")
struct CommandTableTests {
    @Test func runOptionsParseWithAliasesAndShorts() throws {
        let inv = try CLI.parse(["run", "m", "hello", "world", "--max", "10", "-n", "20", "--temperature", "0.5",
                                 "--ctx", "512", "--raw", "--no-think", "--ids", "--spec", "--draft", "auto",
                                 "--sparse", "off"])
        #expect(inv.command.name == "run")
        #expect(inv.positionals == ["m", "hello", "world"])
        #expect(inv.int("max-tokens") == 20)
        #expect(inv.double("temp") == 0.5)
        #expect(inv.int("ctx") == 512)
        #expect(inv.bool("raw") && inv.bool("no-think") && inv.bool("ids") && inv.bool("spec"))
        #expect(inv.string("draft") == "auto")
        #expect(inv.string("sparse") == "off")
        #expect(!inv.has("think"))
    }

    @Test func serveAndStartAlias() throws {
        let inv = try CLI.parse(["start", "m", "--port", "8080", "--lan", "--api-key", "k", "--ctx", "1024"])
        #expect(inv.command.name == "serve")
        #expect(inv.int("port") == 8080)
        #expect(inv.bool("lan"))
        #expect(inv.string("api-key") == "k")
        #expect(inv.positionals == ["m"])
    }

    @Test func optionsMayPrecedeTheModel() throws {
        let inv = try CLI.parse(["serve", "--api-key", "K", "m"])
        #expect(inv.positionals == ["m"])
        #expect(inv.string("api-key") == "K")
        let spec = try CLI.parse(["run", "--spec", "m"])
        #expect(spec.positionals == ["m"])
        #expect(spec.bool("spec"))
    }

    @Test func searchTokenizeBenchAcceptTheirArguments() throws {
        #expect(try CLI.parse(["search", "qwen", "4bit"]).positionals == ["qwen", "4bit"])
        #expect(try CLI.parse(["tokenize", "m", "some", "text"]).positionals == ["m", "some", "text"])
        #expect(try CLI.parse(["bench", "m"]).command.name == "bench")
        #expect(try CLI.parse(["download", "a/b"]).positionals == ["a/b"])
        #expect(try CLI.parse(["list"]).command.name == "list")
    }

    @Test func showIsRegisteredAndFailsCleanlyForNow() async {
        let cap = Capture()
        #expect(await CLI.run(["show", "m"], console: cap.console()) == 1)
        #expect(cap.stderr.hasPrefix("Error: "))
    }

    @Test func rejectsOutOfRangeValuesBeforeRunning() async {
        for argv in [["run", "m", "--max", "0"], ["run", "m", "--max", "32769"], ["run", "m", "--temp", "nan"],
                     ["run", "m", "--temp", "2.1"], ["run", "m", "--ctx", "255"], ["serve", "m", "--port", "0"],
                     ["serve", "m", "--port", "70000"]] {
            let cap = Capture()
            #expect(await CLI.run(argv, console: cap.console()) == 1, "\(argv)")
            #expect(cap.stderr.hasPrefix("Error: invalid argument"), "\(argv): \(cap.stderr)")
        }
    }

    @Test func everyCommandHasHelp() async {
        for c in CLI.commands {
            let cap = Capture()
            #expect(await CLI.run([c.name, "--help"], console: cap.console()) == 0)
            #expect(cap.stdout.contains("Usage:\n  mlxr \(c.name)"))
        }
        let cap = Capture()
        #expect(await CLI.run(["serve", "-h"], console: cap.console()) == 0)
        #expect(cap.stdout.contains("Environment Variables:") && cap.stdout.contains("MLXR_API_KEY"))
        #expect(cap.stdout.contains("--port int"))
    }
}
