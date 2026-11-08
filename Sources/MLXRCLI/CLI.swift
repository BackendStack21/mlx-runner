import Foundation
import MLXEngine

public enum CLI {
    /// Parse a command line against a table without running anything.
    public static func parse(_ args: [String], commands: [Command] = CLI.commands) throws -> Invocation {
        guard let first = args.first else { throw CLIError.usage("accepts a command") }
        guard let command = findCommand(first, in: commands) else { throw unknownCommand(first, commands) }
        return try parse(args: Array(args.dropFirst()), command: command)
    }

    static func unknownCommand(_ name: String, _ commands: [Command]) -> CLIError {
        if name.hasPrefix("-") { return .usage("unknown flag: \(name)") }
        var msg = "unknown command \"\(name)\" for \"\(rootName)\""
        let near = suggestions(for: name, in: commands)
        if !near.isEmpty { msg += "\n\nDid you mean this?\n" + near.map { "\t\($0)\n" }.joined() }
        return .usage(msg)
    }

    /// Parse and run; returns the process exit code. Errors print `Error: <message>` on stderr.
    public static func run(_ args: [String], console: Console, commands: [Command] = CLI.commands) async -> Int32 {
        do {
            if let first = args.first {
                switch first {
                case "-h", "--help": console.out(rootHelp(commands: commands)); return 0
                case "-v", "--version", "version": console.out("mlxr version is \(MLXRunnerVersion.current)\n"); return 0
                case "help":
                    guard args.count > 1 else { console.out(rootHelp(commands: commands)); return 0 }
                    guard let c = findCommand(args[1], in: commands) else {
                        throw CLIError.usage("unknown help topic \"\(args[1])\"")
                    }
                    console.out(commandHelp(c)); return 0
                default: break
                }
            } else {
                console.out(rootHelp(commands: commands)); return 0
            }
            let inv = try parse(args, commands: commands)
            if inv.helpRequested { console.out(commandHelp(inv.command)); return 0 }
            try await inv.command.run(inv, console)
            return 0
        } catch CLIError.interrupted {
            return 130
        } catch {
            console.err("Error: \(message(for: error))\n")
            return 1
        }
    }

    /// The message printed for an error (Hugging Face 401/404 become one hint).
    public static func message(for error: Error) -> String {
        if let e = error as? CLIError {
            switch e {
            case .usage(let m), .failure(let m): return m
            case .interrupted: return "interrupted"
            }
        }
        let ns = error as NSError
        if ns.domain == "HuggingFace", ns.code == 401 || ns.code == 404 {
            return "not found, or private/gated (set HF_TOKEN)"
        }
        return error.localizedDescription
    }

    /// Entry point of the `mlxr` executable.
    public static func main(_ args: [String]) async -> Int32 {
        setvbuf(stdout, nil, _IONBF, 0)
        let console = Console(
            out: { FileHandle.standardOutput.write(Data($0.utf8)) },
            err: { FileHandle.standardError.write(Data($0.utf8)) },
            stdoutIsTTY: isatty(1) != 0, stderrIsTTY: isatty(2) != 0, stdinIsTTY: isatty(0) != 0,
            env: ProcessInfo.processInfo.environment)
        return await run(args, console: console)
    }
}
