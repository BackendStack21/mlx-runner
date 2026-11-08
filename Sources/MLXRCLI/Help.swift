import Foundation

/// Help text in the cobra layout.
extension CLI {
    static let rootName = "mlxr"
    static let rootSummary = "Run MLX models from Hugging Face on Apple Silicon, from the command line."

    static func rootHelp(commands: [Command]) -> String {
        let shown = commands.filter { !$0.hidden }
        let width = max(11, shown.map { $0.name.count }.max() ?? 0) + 1
        func rows(_ list: [Command]) -> String {
            list.map { "  " + $0.name.padding(toLength: width, withPad: " ", startingAt: 0) + " " + $0.summary + "\n" }.joined()
        }
        var out = rootSummary + "\n\nUsage:\n  \(rootName) [flags]\n  \(rootName) [command]\n\n"
        out += "Available Commands:\n" + rows(shown.filter { $0.group == nil }) + "\n"
        var groups: [String] = []
        for c in shown { if let g = c.group, !groups.contains(g) { groups.append(g) } }
        for g in groups { out += "\(g) Commands:\n" + rows(shown.filter { $0.group == g }) + "\n" }
        out += "Flags:\n  -h, --help      help for \(rootName)\n  -v, --version   show version information\n\n"
        out += "Use \"\(rootName) [command] --help\" for more information about a command.\n"
        return out
    }

    static func commandHelp(_ c: Command) -> String {
        var out = c.summary + "\n\nUsage:\n  \(rootName) \(c.name)"
        for a in c.args {
            let n = a.variadic ? a.name + "..." : a.name
            out += a.optional ? " [\(n)]" : " \(n)"
        }
        out += " [flags]\n"
        if !c.aliases.isEmpty { out += "\nAliases:\n  \(([c.name] + c.aliases).joined(separator: ", "))\n" }
        if !c.examples.isEmpty { out += "\nExamples:\n" + c.examples.joined(separator: "\n") + "\n" }
        var rows: [(String, String)] = c.options.filter { !$0.hidden }.map { o in
            let short = o.short.map { "-\($0), " } ?? "    "
            var left = short + "--" + o.long
            if let kind = o.value { left += " " + typeName(kind) }
            var help = o.help
            if let d = o.defaultText { help += " (default \(d))" }
            return (left, help)
        }
        rows.append(("-h, --help", "help for \(c.name)"))
        rows = rows.map { ($0.0.hasPrefix("-") && !$0.0.hasPrefix("--") ? $0.0 : "    " + $0.0.trimmingCharacters(in: .init(charactersIn: " ")), $0.1) }
        let sorted = rows.sorted { key($0.0) < key($1.0) }
        let w = sorted.map { $0.0.count }.max() ?? 0
        out += "\nFlags:\n" + sorted.map { "  " + $0.0.padding(toLength: w, withPad: " ", startingAt: 0) + "   " + $0.1 + "\n" }.joined()
        if !c.environment.isEmpty {
            out += "\nEnvironment Variables:\n" + c.environment.map {
                "      " + $0.name.padding(toLength: 27, withPad: " ", startingAt: 0) + "   " + $0.help + "\n"
            }.joined()
        }
        return out
    }

    private static func key(_ left: String) -> String {
        left.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "-h, ", with: "").replacingOccurrences(of: "-", with: "")
            .lowercased()
    }

    static func typeName(_ k: ValueKind) -> String {
        switch k {
        case .int: return "int"
        case .uint64: return "uint"
        case .real: return "float"
        case .text, .choice, .choiceOrInt, .model, .file: return "string"
        }
    }
}
