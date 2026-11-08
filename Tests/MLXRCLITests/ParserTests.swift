import Testing
@testable import MLXRCLI

@Suite("Parser grammar")
struct ParserTests {
    @Test func equalsFormDoubleDashAndMixedPositionals() throws {
        let inv = try CLI.parse(["run", "m", "--temp=0.5", "-p", "hi", "--", "-x"], commands: miniTable())
        #expect(inv.command.name == "run")
        #expect(inv.double("temp") == 0.5)
        #expect(inv.string("prompt") == "hi")
        #expect(inv.positionals == ["m", "-x"])
    }

    @Test func optionsMixFreelyWithPositionals() throws {
        let inv = try CLI.parse(["run", "--temp", "1", "m", "word", "--verbose"], commands: miniTable())
        #expect(inv.positionals == ["m", "word"])
        #expect(inv.double("temp") == 1)
        #expect(inv.bool("verbose"))
    }

    @Test func thinkAloneIsTrueAndTakesValueOnlyViaEquals() throws {
        let alone = try CLI.parse(["run", "m", "--think"], commands: miniTable())
        #expect(alone.string("think") == "true")
        #expect(alone.bool("think"))

        let spaced = try CLI.parse(["run", "--think", "high", "m"], commands: miniTable())
        #expect(spaced.string("think") == "true")
        #expect(spaced.positionals == ["high", "m"])

        let equals = try CLI.parse(["run", "m", "--think=high"], commands: miniTable())
        #expect(equals.string("think") == "high")
        #expect(equals.positionals == ["m"])
    }

    @Test func lastValueWins() throws {
        let inv = try CLI.parse(["run", "m", "--temp", "0.1", "--temp", "0.7"], commands: miniTable())
        #expect(inv.double("temp") == 0.7)
    }

    @Test func boolAcceptsEqualsTrueFalse() throws {
        let off = try CLI.parse(["run", "m", "--verbose=false"], commands: miniTable())
        #expect(!off.bool("verbose"))
        let on = try CLI.parse(["run", "m", "--verbose=true"], commands: miniTable())
        #expect(on.bool("verbose"))
        #expect(!(try CLI.parse(["run", "m"], commands: miniTable())).bool("verbose"))
    }

    @Test func aliasesResolveToTheirCommand() throws {
        #expect(try CLI.parse(["ls"], commands: miniTable()).command.name == "list")
        #expect(try CLI.parse(["pull", "a/b"], commands: miniTable()).command.name == "download")
        #expect(try CLI.parse(["download", "a/b"], commands: miniTable()).command.name == "download")
    }

    @Test func realTableResolvesLsAndPull() throws {
        #expect(try CLI.parse(["ls"]).command.name == "list")
        #expect(try CLI.parse(["pull", "a/b"]).command.name == "pull")
    }

    @Test func invalidValuesNameTheFlag() async {
        for (argv, flag) in [(["run", "m", "--temp", "abc"], "--temp"), (["run", "m", "--count", "11"], "--count"),
                             (["run", "m", "--count", "0"], "--count"), (["run", "m", "--think=maybe"], "--think")] {
            let cap = Capture()
            let code = await CLI.run(argv, console: cap.console(), commands: miniTable())
            #expect(code == 1, "\(argv)")
            #expect(cap.stderr.hasPrefix("Error: invalid argument \"\(argv.last!.split(separator: "=").last!)\" for \"\(flag)\" flag"),
                    "\(argv): \(cap.stderr)")
            #expect(cap.stdout.isEmpty)
        }
    }

    @Test func missingValueIsAnError() async {
        let cap = Capture()
        let code = await CLI.run(["run", "m", "--temp"], console: cap.console(), commands: miniTable())
        #expect(code == 1)
        #expect(cap.stderr == "Error: flag needs an argument: --temp\n")
    }

    @Test func acceptsExactlyTheDeclaredArgumentCounts() async {
        let log = RunLog()
        let cap = Capture()
        #expect(await CLI.run(["download"], console: cap.console(), commands: miniTable(log)) == 1)
        #expect(cap.stderr == "Error: accepts 1 arg(s), received 0\n")
        let cap2 = Capture()
        #expect(await CLI.run(["download", "a/b", "c/d"], console: cap2.console(), commands: miniTable(log)) == 1)
        #expect(cap2.stderr == "Error: accepts 1 arg(s), received 2\n")
        let cap3 = Capture()
        #expect(await CLI.run(["run"], console: cap3.console(), commands: miniTable(log)) == 1)
        #expect(cap3.stderr == "Error: requires at least 1 arg(s), only received 0\n")
        #expect(log.ran.isEmpty)
    }

    @Test func aValidCommandRuns() async {
        let log = RunLog()
        let cap = Capture()
        #expect(await CLI.run(["ls"], console: cap.console(), commands: miniTable(log)) == 0)
        #expect(log.ran == ["list"])
        #expect(cap.stderr.isEmpty)
    }
}
