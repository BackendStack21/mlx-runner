import Foundation

/// The command-line grammar of cobra/pflag: `--x v`, `--x=v`, `-x v`, `--` ends options,
/// options and positionals mix freely, the last value of an option wins.
extension CLI {
    static func findCommand(_ name: String, in commands: [Command]) -> Command? {
        commands.first { $0.name == name || $0.aliases.contains(name) }
    }

    static func parse(args: [String], command: Command) throws -> Invocation {
        var inv = Invocation(command: command)
        var i = 0
        var onlyPositionals = false
        func option(long name: String) -> Opt? {
            command.options.first { $0.long == name || $0.aliases.contains(name) }
        }
        while i < args.count {
            let a = args[i]
            i += 1
            if onlyPositionals || a == "-" || !a.hasPrefix("-") { inv.positionals.append(a); continue }
            if a == "--" { onlyPositionals = true; continue }
            var opt: Opt?
            var inline: String?
            var shown: String
            if a.hasPrefix("--") {
                let body = String(a.dropFirst(2))
                let parts = body.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
                let name = String(parts[0])
                if parts.count == 2 { inline = String(parts[1]) }
                if name == "help" { inv.helpRequested = true; continue }
                guard let o = option(long: name) else { throw CLIError.usage("unknown flag: --\(name)") }
                opt = o; shown = "--\(o.long)"
            } else {
                let chars = Array(a.dropFirst())
                let c = chars[0]
                if c == "h" && chars.count == 1 { inv.helpRequested = true; continue }
                guard let o = command.options.first(where: { $0.short == c }) else {
                    throw CLIError.usage("unknown shorthand flag: '\(c)' in -\(String(chars))")
                }
                if chars.count > 1 {
                    // -pVALUE and -p=VALUE
                    var rest = String(chars.dropFirst())
                    if rest.hasPrefix("=") { rest.removeFirst() }
                    inline = rest
                }
                opt = o; shown = o.short.map { "-\($0), --\(o.long)" } ?? "--\(o.long)"
            }
            let o = opt!
            let raw: String
            if let kind = o.value {
                if let inline { raw = inline }
                else if let implicit = o.implicit { raw = implicit }
                else if i < args.count { raw = args[i]; i += 1 }
                else { throw CLIError.usage(a.hasPrefix("--") ? "flag needs an argument: \(shown)" : "flag needs an argument: '\(o.short.map(String.init) ?? "")' in -\(o.short.map(String.init) ?? "")") }
                try validate(raw, kind: kind, flag: shown)
            } else {
                if let inline {
                    guard ["true", "false", "1", "0", "t", "f", "TRUE", "FALSE", "True", "False", "T", "F"].contains(inline) else {
                        throw CLIError.usage("invalid argument \"\(inline)\" for \"\(shown)\" flag: strconv.ParseBool: parsing \"\(inline)\": invalid syntax")
                    }
                    raw = ["true", "1", "t", "TRUE", "True", "T"].contains(inline) ? "true" : "false"
                } else { raw = "true" }
            }
            if o.repeatable { inv.values[o.long, default: []].append(raw) } else { inv.values[o.long] = [raw] }
        }
        if !inv.helpRequested { try checkArgCount(inv.positionals.count, command: command) }
        return inv
    }

    static func validate(_ raw: String, kind: ValueKind, flag: String) throws {
        func bad(_ why: String) -> CLIError { .usage("invalid argument \"\(raw)\" for \"\(flag)\" flag: \(why)") }
        switch kind {
        case .text, .model, .file: break
        case .int(let r):
            guard let n = Int(raw) else { throw bad("strconv.ParseInt: parsing \"\(raw)\": invalid syntax") }
            guard r.contains(n) else { throw bad("value \(n) is outside \(r.lowerBound)...\(r.upperBound)") }
        case .uint64:
            guard UInt64(raw) != nil else { throw bad("strconv.ParseUint: parsing \"\(raw)\": invalid syntax") }
        case .real(let r):
            guard let d = Double(raw), d.isFinite else { throw bad("strconv.ParseFloat: parsing \"\(raw)\": invalid syntax") }
            guard r.contains(d) else { throw bad("value \(raw) is outside \(r.lowerBound)...\(r.upperBound)") }
        case .choice(let c):
            guard c.contains(raw) else { throw bad("must be one of \(c.joined(separator: ", "))") }
        case .choiceOrInt(let c):
            guard c.contains(raw) || (Int(raw).map { $0 > 0 } ?? false) else {
                throw bad("must be one of \(c.joined(separator: ", ")) or a positive number")
            }
        }
    }

    static func checkArgCount(_ n: Int, command: Command) throws {
        let required = command.args.filter { !$0.optional }.count
        let variadic = command.args.contains { $0.variadic }
        if variadic {
            if n < required { throw CLIError.usage("requires at least \(required) arg(s), only received \(n)") }
        } else if required == command.args.count {
            if n != required { throw CLIError.usage("accepts \(required) arg(s), received \(n)") }
        } else if n < required || n > command.args.count {
            throw CLIError.usage("accepts between \(required) and \(command.args.count) arg(s), received \(n)")
        }
    }

    /// Commands whose name is close to a mistyped one (cobra: edit distance <= 2 or a prefix).
    static func suggestions(for name: String, in commands: [Command]) -> [String] {
        let lower = name.lowercased()
        return commands.filter { !$0.hidden }.compactMap { c -> String? in
            let hit = ([c.name] + c.aliases).contains { n in
                editDistance(lower, n.lowercased()) <= 2 || n.lowercased().hasPrefix(lower)
            }
            return hit ? c.name : nil
        }
    }

    static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        var prev = Array(0...b.count)
        for i in 1...max(a.count, 1) where !a.isEmpty {
            var cur = [i] + [Int](repeating: 0, count: b.count)
            for j in 1...max(b.count, 1) where !b.isEmpty {
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
            }
            prev = cur
        }
        return a.isEmpty ? b.count : prev[b.count]
    }
}
