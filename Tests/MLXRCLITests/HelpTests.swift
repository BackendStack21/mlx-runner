import Testing
@testable import MLXRCLI
import MLXEngine

@Suite("Help and version")
struct HelpTests {
    func output(_ argv: [String], commands: [Command] = CLI.commands) async -> (code: Int32, out: String, err: String) {
        let cap = Capture()
        let code = await CLI.run(argv, console: cap.console(), commands: commands)
        return (code, cap.stdout, cap.stderr)
    }

    @Test func bareAndDashHelpAreIdenticalRootHelp() async {
        let bare = await output([])
        let flag = await output(["--help"])
        let short = await output(["-h"])
        #expect(bare.code == 0 && flag.code == 0 && short.code == 0)
        #expect(bare.out == flag.out)
        #expect(bare.out == short.out)
        #expect(bare.err.isEmpty)
        for needle in ["Usage:", "Available Commands:", "Flags:",
                       "Use \"mlxr [command] --help\" for more information about a command."] {
            #expect(bare.out.contains(needle), "\(needle)")
        }
        for cmd in ["run", "serve", "pull", "list", "show"] {
            #expect(bare.out.contains("\n  \(cmd) "), "\(cmd)")
        }
    }

    @Test func developerCommandsSitUnderTheirOwnHeading() async {
        let out = await output([]).out
        let dev = out.range(of: "Developer")
        #expect(dev != nil)
        for cmd in ["bench", "tokenize"] {
            let at = out.range(of: "\n  \(cmd) ")
            #expect(at != nil, "\(cmd)")
            if let dev, let at { #expect(dev.lowerBound < at.lowerBound, "\(cmd) below Developer") }
        }
        if let dev, let run = out.range(of: "\n  run ") { #expect(run.lowerBound < dev.lowerBound) }
    }

    @Test func commandHelpListsFlagsAndEnvironment() async {
        let r = await output(["run", "-h"])
        #expect(r.code == 0)
        #expect(r.out.contains("Usage:"))
        #expect(r.out.contains("Flags:"))
        #expect(r.out.contains("Environment Variables:"))
        #expect(r.out.contains("--temp"))
        #expect(r.err.isEmpty)
    }

    @Test func helpCommandEqualsDashH() async {
        let viaHelp = await output(["help", "run"])
        let viaFlag = await output(["run", "-h"])
        #expect(viaHelp.code == 0)
        #expect(!viaHelp.out.isEmpty)
        #expect(viaHelp.out == viaFlag.out)
        let helpOnly = await output(["help"])
        let root = await output([])
        #expect(helpOnly.out == root.out)
    }

    @Test func helpFlagNeverRunsTheCommand() async {
        let log = RunLog()
        let r = await output(["run", "-h"], commands: miniTable(log))
        #expect(r.code == 0)
        #expect(log.ran.isEmpty)
        #expect(r.out.contains("Usage:"))
        let search = await output(["search", "-h"])
        #expect(search.out.contains("Usage:"))
        #expect(search.code == 0)
    }

    @Test func helpShowsAliasesAndExamplesWhenDeclared() async {
        let cmd = Command(name: "list", aliases: ["ls"], summary: "List models", examples: ["  mlxr list"])
        let r = await output(["list", "--help"], commands: [cmd])
        #expect(r.out.contains("Aliases:\n  list, ls"))
        #expect(r.out.contains("Examples:\n  mlxr list"))
    }

    @Test func versionInAllThreeForms() async {
        let expected = "mlxr version is \(MLXRunnerVersion.current)\n"
        for argv in [["-v"], ["--version"], ["version"]] {
            let r = await output(argv)
            #expect(r.code == 0, "\(argv)")
            #expect(r.out == expected, "\(argv)")
            #expect(r.err.isEmpty, "\(argv)")
        }
    }
}
