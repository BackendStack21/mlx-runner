import Foundation
import Testing
@testable import MLXRCLI

/// A table with a hidden command, a hidden flag, a choice flag and hostile summary text.
private func hostileTable() -> [Command] {
    [
        Command(name: "chat", aliases: ["talk"], summary: "It's a \"quoted\" $HOME `tick` \\ summary; (parens) [brackets]",
                args: [Arg("MODEL", kind: .model)],
                options: [
                    Opt(long: "mode", short: "m", value: .choice(["alpha", "beta-two"]), help: "pick: 'one' or \"two\" $x"),
                    Opt(long: "secret-flag", hidden: true, help: "hidden"),
                    Opt(long: "plain", help: "a flag"),
                ]),
        Command(name: "ghost-command", summary: "hidden", hidden: true),
    ]
}

private func runShell(_ shell: String, flag: String, script: String) throws -> (status: Int32, stderr: String) {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-completion-\(UUID().uuidString).\(shell)")
    try script.write(to: file, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: file) }
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/bin/\(shell)")
    p.arguments = [flag, file.path]
    let err = Pipe()
    p.standardError = err
    p.standardOutput = Pipe()
    try p.run()
    p.waitUntilExit()
    return (p.terminationStatus, String(decoding: err.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
}

@Suite("Shell completions (AC-20)")
struct CompletionTests {
    /// Every visible command name and alias, and every visible long and short flag, of a table.
    func expectedWords(_ table: [Command]) -> [String] {
        var words: [String] = []
        for c in table where !c.hidden {
            words += [c.name] + c.aliases
            for o in c.options where !o.hidden {
                words.append("--" + o.long)
                words += o.aliases.map { "--" + $0 }
                if let s = o.short { words.append("-\(s)") }
            }
        }
        return words
    }

    @Test(arguments: ["zsh", "bash"])
    func scriptContainsEveryCommandAndFlagOfTheRealTable(shell: String) async throws {
        let table = LibraryFixture.standard().table()
        let script = try CLI.completionScript(shell: shell, commands: table)
        #expect(!script.isEmpty)
        for w in Set(expectedWords(table)) {
            #expect(script.contains(w), "\(shell) script lacks \(w)")
        }
        #expect(script.contains("completion"), "the completion command completes itself")
        #expect(table.contains { $0.name == "completion" }, "completion is in the command table")
    }

    @Test(arguments: ["zsh", "bash"])
    func scriptIsGeneratedFromTheGivenTableNotAFixedList(shell: String) throws {
        let script = try CLI.completionScript(shell: shell, commands: hostileTable())
        for w in ["chat", "talk", "--mode", "-m", "--plain", "alpha", "beta-two"] {
            #expect(script.contains(w), "\(shell) script lacks \(w)")
        }
        for gone in ["ghost-command", "secret-flag", "download", "--temp"] {
            #expect(!script.contains(gone), "\(shell) script has \(gone)")
        }
    }

    @Test(arguments: ["zsh", "bash"])
    func modelNamesComeFromMlxrList(shell: String) throws {
        let script = try CLI.completionScript(shell: shell, commands: hostileTable())
        #expect(script.contains("mlxr list"), "\(shell): model arguments complete from `mlxr list`")
    }

    @Test(arguments: ["zsh", "bash"])
    func scriptIsAcceptedByTheShellSyntaxCheck(shell: String) throws {
        for table in [hostileTable(), LibraryFixture.standard().table(), miniTable()] {
            let script = try CLI.completionScript(shell: shell, commands: table)
            #expect(!script.isEmpty)
            let r = try runShell(shell, flag: "-n", script: script)
            #expect(r.status == 0, "\(shell) -n rejected the script: \(r.stderr)")
        }
    }

    @Test func zshScriptIsACompdefFileAndBashRegistersWithComplete() throws {
        let zsh = try CLI.completionScript(shell: "zsh", commands: miniTable())
        #expect(zsh.hasPrefix("#compdef mlxr"), "zsh: \(zsh.prefix(40))")
        let bash = try CLI.completionScript(shell: "bash", commands: miniTable())
        #expect(bash.contains("complete ") && bash.contains(" mlxr"), "bash registers a completion for mlxr")
    }

    @Test func unknownShellThrowsNamingTheSupportedOnes() {
        for shell in ["fish", "powershell", "", "ZSH"] {
            let msg = failureMessage { try CLI.completionScript(shell: shell, commands: miniTable()) }
            #expect(msg != nil, "\(shell.debugDescription) must throw")
            #expect(msg?.contains("zsh") == true && msg?.contains("bash") == true, "\(msg ?? "nil")")
        }
    }

    @Test(arguments: ["zsh", "bash"])
    func completionCommandPrintsTheScriptOnStdoutOnly(shell: String) async throws {
        let fx = LibraryFixture.standard()
        let r = await fx.run(["completion", shell])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.err.isEmpty, "nothing on stderr: \(r.err)")
        #expect(r.out == (try CLI.completionScript(shell: shell, commands: fx.table())))
        #expect(!r.out.isEmpty)
    }

    @Test func completionCommandRejectsUnknownShellsAndBadArguments() async {
        let fx = LibraryFixture.standard()
        let bad = await fx.run(["completion", "fish"])
        #expect(bad.code == 1 && bad.out.isEmpty, "no script on stdout: \(bad.out.prefix(80))")
        #expect(bad.err.hasPrefix("Error: "), "\(bad.err)")
        #expect(bad.err.contains("fish") && bad.err.contains("zsh") && bad.err.contains("bash"), "\(bad.err)")
        let none = await fx.run(["completion"])
        #expect(none.code == 1 && none.out.isEmpty)
        #expect(none.err == "Error: accepts 1 arg(s), received 0\n")
        let extra = await fx.run(["completion", "zsh", "bash"])
        #expect(extra.code == 1 && extra.out.isEmpty)
        #expect(extra.err == "Error: accepts 1 arg(s), received 2\n")
    }

    @Test func completionIsListedInRootHelp() async {
        let r = await LibraryFixture.standard().run(["--help"])
        #expect(r.out.contains("completion"), Comment(rawValue: r.out))
    }
}
