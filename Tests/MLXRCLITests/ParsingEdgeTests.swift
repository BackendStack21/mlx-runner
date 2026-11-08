import Foundation
import Testing
@testable import MLXRCLI

@Suite("Parsing edge cases")
struct ParsingEdgeTests {
    let table: [Command] = [
        Command(name: "t", summary: "t", args: [Arg("A"), Arg("B", optional: true)], options: [
            Opt(long: "name", short: "p", value: .text),
            Opt(long: "yes", short: "y"),
            Opt(long: "seed", value: .uint64),
            Opt(long: "sparse", value: .choiceOrInt(["on", "off"])),
            Opt(long: "tag", value: .text, repeatable: true),
            Opt(long: "hidden", hidden: true),
        ]),
        Command(name: "none", summary: "n"),
        Command(name: "secret", summary: "s", hidden: true),
    ]

    func code(_ argv: [String]) async -> (Int32, String, String) {
        let cap = Capture()
        let c = await CLI.run(argv, console: cap.console(), commands: table)
        return (c, cap.stdout, cap.stderr)
    }

    @Test func shortFormsAndBooleans() throws {
        let a = try CLI.parse(["t", "x", "-p", "v", "-y"], commands: table)
        #expect(a.string("name") == "v" && a.bool("yes"))
        let b = try CLI.parse(["t", "x", "-pv"], commands: table)
        #expect(b.string("name") == "v")
        let c = try CLI.parse(["t", "x", "-p=v"], commands: table)
        #expect(c.string("name") == "v")
        let d = try CLI.parse(["t", "x", "--yes=0"], commands: table)
        #expect(!d.bool("yes"))
        #expect(try CLI.parse(["t", "x", "-"], commands: table).positionals == ["x", "-"])
    }

    @Test func repeatableCollectsAllValues() throws {
        let a = try CLI.parse(["t", "x", "--tag", "a", "--tag=b"], commands: table)
        #expect(a.strings("tag") == ["a", "b"])
    }

    @Test func kindValidation() async {
        #expect(try! CLI.parse(["t", "x", "--sparse", "on"], commands: table).string("sparse") == "on")
        #expect(try! CLI.parse(["t", "x", "--sparse", "64"], commands: table).string("sparse") == "64")
        #expect(try! CLI.parse(["t", "x", "--seed", "42"], commands: table).string("seed") == "42")
        for argv in [["t", "x", "--sparse", "0"], ["t", "x", "--sparse", "zz"], ["t", "x", "--seed", "-1"],
                     ["t", "x", "--yes=maybe"]] {
            let r = await code(argv)
            #expect(r.0 == 1 && r.2.hasPrefix("Error: invalid argument"), "\(argv): \(r.2)")
        }
    }

    @Test func flagErrors() async {
        #expect(await code(["t", "x", "-q"]).2 == "Error: unknown shorthand flag: 'q' in -q\n")
        #expect(await code(["t", "x", "-p"]).2 == "Error: flag needs an argument: 'p' in -p\n")
        #expect(await code(["t", "x", "--name"]).2 == "Error: flag needs an argument: --name\n")
        #expect(await code(["--bogus"]).2 == "Error: unknown flag: --bogus\n")
        #expect(await code(["t", "x", "y", "z"]).2 == "Error: accepts between 1 and 2 arg(s), received 3\n")
        #expect(await code(["none", "extra"]).2 == "Error: accepts 0 arg(s), received 1\n")
        #expect(await code(["help", "nope"]).2 == "Error: unknown help topic \"nope\"\n")
    }

    @Test func parseWithoutCommandThrows() {
        #expect(throws: CLIError.self) { try CLI.parse([], commands: table) }
    }

    @Test func hiddenThingsStayOutOfHelpAndSuggestions() async {
        let root = await code([])
        #expect(!root.1.contains("secret"))
        let help = await code(["t", "-h"])
        #expect(!help.1.contains("--hidden"))
        #expect(help.1.contains("-p, --name string"))
        #expect(help.1.contains("Usage:\n  mlxr t A [B] [flags]"))
        let miss = await code(["secrex"])
        #expect(!miss.2.contains("Did you mean"))
    }

    @Test func interruptedExitsWith130AndNoMessage() async {
        let cmd = Command(name: "i", summary: "i", run: { _, _ in throw CLIError.interrupted })
        let cap = Capture()
        #expect(await CLI.run(["i"], console: cap.console(), commands: [cmd]) == 130)
        #expect(cap.stderr.isEmpty)
        #expect(CLI.message(for: CLIError.interrupted) == "interrupted")
    }

    @Test func editDistance() {
        #expect(CLI.editDistance("", "abc") == 3)
        #expect(CLI.editDistance("abc", "") == 3)
        #expect(CLI.editDistance("lst", "list") == 1)
        #expect(CLI.editDistance("kitten", "sitting") == 3)
    }
}
