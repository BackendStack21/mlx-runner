import Foundation

/// What an option or positional argument accepts. Drives validation, help and completions.
public enum ValueKind {
    case text
    case int(ClosedRange<Int>)
    case uint64
    case real(ClosedRange<Double>)
    /// A finite number greater than 0 and at most `max`.
    case realAboveZero(max: Double)
    case choice([String])
    case choiceOrInt([String])
    case model
    case file
}

/// One option of a command. `value == nil` is a boolean flag (`--x`, `--x=true|false`).
public struct Opt {
    public var long: String
    public var aliases: [String] = []
    public var short: Character? = nil
    public var value: ValueKind? = nil
    /// Value taken when the option is given alone (cobra NoOptDefVal): `--think` means `--think=true`.
    public var implicit: String? = nil
    public var repeatable = false
    public var env: String? = nil
    public var defaultText: String? = nil
    public var hidden = false
    public var help: String = ""

    public init(long: String, aliases: [String] = [], short: Character? = nil, value: ValueKind? = nil,
                implicit: String? = nil, repeatable: Bool = false, env: String? = nil,
                defaultText: String? = nil, hidden: Bool = false, help: String = "") {
        self.long = long; self.aliases = aliases; self.short = short; self.value = value
        self.implicit = implicit; self.repeatable = repeatable; self.env = env
        self.defaultText = defaultText; self.hidden = hidden; self.help = help
    }
}

/// One positional argument. A variadic argument takes all remaining positionals.
public struct Arg {
    public var name: String
    public var kind: ValueKind = .text
    public var optional = false
    public var variadic = false

    public init(_ name: String, kind: ValueKind = .text, optional: Bool = false, variadic: Bool = false) {
        self.name = name; self.kind = kind; self.optional = optional; self.variadic = variadic
    }
}

/// Failures a command can report. `usage` and `failure` print `Error: <message>` and exit 1.
public enum CLIError: Error, Equatable {
    case usage(String)
    case failure(String)
    case interrupted
}

/// The streams and terminal facts a command sees; injected so commands are testable.
public final class Console: @unchecked Sendable {
    public var out: (String) -> Void
    public var err: (String) -> Void
    public var stdoutIsTTY: Bool
    public var stderrIsTTY: Bool
    public var stdinIsTTY: Bool
    public var width: Int
    public var env: [String: String]
    /// Reads one line of input (a confirmation answer); nil at end of input.
    public var readLine: () -> String? = { Swift.readLine() }

    public init(out: @escaping (String) -> Void, err: @escaping (String) -> Void,
                stdoutIsTTY: Bool = false, stderrIsTTY: Bool = false, stdinIsTTY: Bool = false,
                width: Int = 80, env: [String: String] = [:]) {
        self.out = out; self.err = err
        self.stdoutIsTTY = stdoutIsTTY; self.stderrIsTTY = stderrIsTTY; self.stdinIsTTY = stdinIsTTY
        self.width = width; self.env = env
    }

    /// Reads everything piped to stdin (only called when stdin is not a terminal).
    public var readAllStdin: () -> String = { String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self) }
}

/// The validated result of parsing one command line.
public struct Invocation {
    public var command: Command
    public var positionals: [String]
    /// Raw validated values by long option name, in the order given.
    public var values: [String: [String]]

    public init(command: Command, positionals: [String] = [], values: [String: [String]] = [:]) {
        self.command = command; self.positionals = positionals; self.values = values
    }

    /// True when `-h` / `--help` was given: the command does not run.
    public var helpRequested = false

    /// Last value given for the option, or nil.
    public func string(_ long: String) -> String? { values[long]?.last }
    public func strings(_ long: String) -> [String] { values[long] ?? [] }
    public func int(_ long: String) -> Int? { string(long).flatMap { Int($0) } }
    public func double(_ long: String) -> Double? { string(long).flatMap { Double($0) } }
    /// A boolean flag: true when given alone or as `=true`, false for `=false` or when absent.
    public func bool(_ long: String) -> Bool { string(long).map { $0 != "false" } ?? false }
    public func has(_ long: String) -> Bool { values[long] != nil }
}

public struct Command {
    public var name: String
    public var aliases: [String] = []
    public var summary: String
    public var args: [Arg] = []
    public var options: [Opt] = []
    public var examples: [String] = []
    public var environment: [(name: String, help: String)] = []
    /// Help heading in the root command list; nil means the main list.
    public var group: String? = nil
    public var hidden = false
    public var run: (Invocation, Console) async throws -> Void

    public init(name: String, aliases: [String] = [], summary: String, args: [Arg] = [],
                options: [Opt] = [], examples: [String] = [],
                environment: [(name: String, help: String)] = [], group: String? = nil,
                hidden: Bool = false, run: @escaping (Invocation, Console) async throws -> Void = { _, _ in }) {
        self.name = name; self.aliases = aliases; self.summary = summary; self.args = args
        self.options = options; self.examples = examples; self.environment = environment
        self.group = group; self.hidden = hidden; self.run = run
    }
}
