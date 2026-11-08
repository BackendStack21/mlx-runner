import Foundation

/// Shell completion scripts generated from the command tables.
extension CLI {
    static let completionShells = ["zsh", "bash"]

    /// The completion script for `shell` ("zsh" or "bash"); throws for any other shell.
    static func completionScript(shell: String, commands: [Command]) throws -> String {
        let shown = commands.filter { !$0.hidden }
        switch shell {
        case "zsh": return zshScript(shown)
        case "bash": return bashScript(shown)
        default:
            throw CLIError.usage("unknown shell \"\(shell)\"; supported shells: \(completionShells.joined(separator: ", "))")
        }
    }

    /// `mlxr completion zsh|bash`: prints the script on stdout. `table` holds the other commands;
    /// the script covers them and `completion` itself.
    static func completionCommand(table: [Command] = []) -> Command {
        let shape = Command(name: "completion", summary: "Generate the shell completion script", args: [Arg("SHELL")])
        var command = shape
        command.run = { inv, console in
            console.out(try completionScript(shell: inv.positionals[0], commands: table + [shape]))
        }
        return command
    }

    // MARK: - Shared helpers

    private static func sq(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''").replacingOccurrences(of: "\n", with: " ") + "'"
    }

    private static func visible(_ c: Command) -> [Opt] { c.options.filter { !$0.hidden } }

    private static func flagNames(_ o: Opt) -> [String] {
        ["--" + o.long] + o.aliases.map { "--" + $0 } + (o.short.map { ["-\($0)"] } ?? [])
    }

    private static func words(_ k: ValueKind) -> [String]? {
        switch k {
        case .choice(let w), .choiceOrInt(let w): return w
        default: return nil
        }
    }

    private static func isModel(_ k: ValueKind) -> Bool { if case .model = k { return true } else { return false } }
    private static func isFile(_ k: ValueKind) -> Bool { if case .file = k { return true } else { return false } }

    private static let listModels = "command mlxr list 2>/dev/null | awk 'NR>1 {print $1}'"

    // MARK: - zsh

    private static func zshAction(_ k: ValueKind) -> String {
        if let w = words(k) { return "(" + w.joined(separator: " ") + ")" }
        if isModel(k) { return "_mlxr_models" }
        if isFile(k) { return "_files" }
        return " "
    }

    private static func zshScript(_ commands: [Command]) -> String {
        var out = "#compdef mlxr\n\n_mlxr_models() {\n  local -a models\n  models=(${(f)\"$(\(listModels))\"})\n  compadd -a models\n}\n\n"
        out += "_mlxr() {\n  local -a commands\n  commands=(\n"
        for c in commands {
            for n in [c.name] + c.aliases { out += "    " + sq(n + ":" + c.summary.replacingOccurrences(of: "[", with: "(").replacingOccurrences(of: "]", with: ")")) + "\n" }
        }
        out += "  )\n  if (( CURRENT == 2 )); then\n    _describe 'command' commands\n    return\n  fi\n  case ${words[2]} in\n"
        for c in commands {
            out += "    " + ([c.name] + c.aliases).joined(separator: "|") + ")\n      _arguments \\\n"
            for o in visible(c) {
                let desc = o.help.replacingOccurrences(of: "[", with: "(").replacingOccurrences(of: "]", with: ")")
                for n in flagNames(o) {
                    var spec = n
                    if o.value != nil, o.implicit != nil, n.hasPrefix("--") { spec += "=-" }
                    spec += "[\(desc)]"
                    if let k = o.value { spec += ":value:" + zshAction(k) }
                    out += "        " + sq(spec) + " \\\n"
                }
            }
            out += "        " + sq("(-h --help)--help[help]") + " \\\n"
            for (i, a) in c.args.enumerated() {
                let lead = a.variadic ? "*" : (a.optional ? "\(i + 1):" : "\(i + 1)")
                out += "        " + sq("\(lead):\(a.name.lowercased()):" + zshAction(a.kind)) + " \\\n"
            }
            out += "        && return\n      ;;\n"
        }
        out += "  esac\n}\n\n_mlxr \"$@\"\n"
        return out
    }

    // MARK: - bash

    private static func bashScript(_ commands: [Command]) -> String {
        let names = commands.flatMap { [$0.name] + $0.aliases }
        var out = "_mlxr_models() {\n  \(listModels)\n}\n\n_mlxr() {\n  local cur prev cmd\n  COMPREPLY=()\n"
        out += "  cur=\"${COMP_WORDS[COMP_CWORD]}\"\n  prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n"
        out += "  if [ \"$COMP_CWORD\" -eq 1 ]; then\n    COMPREPLY=($(compgen -W \(sq(names.joined(separator: " "))) -- \"$cur\"))\n    return\n  fi\n"
        out += "  cmd=\"${COMP_WORDS[1]}\"\n  case \"$cmd\" in\n"
        for c in commands {
            out += "    " + ([c.name] + c.aliases).joined(separator: "|") + ")\n      case \"$prev\" in\n"
            for o in visible(c) {
                guard let k = o.value, o.implicit == nil || words(k) != nil else { continue }
                let pattern = flagNames(o).joined(separator: "|")
                if let w = words(k) {
                    out += "        \(pattern)) COMPREPLY=($(compgen -W \(sq(w.joined(separator: " "))) -- \"$cur\")); return;;\n"
                } else if isModel(k) {
                    out += "        \(pattern)) COMPREPLY=($(compgen -W \"$(_mlxr_models)\" -- \"$cur\")); return;;\n"
                } else if isFile(k) {
                    out += "        \(pattern)) COMPREPLY=($(compgen -f -- \"$cur\")); return;;\n"
                } else {
                    out += "        \(pattern)) return;;\n"
                }
            }
            out += "      esac\n"
            let flags = (visible(c).flatMap(flagNames) + ["--help", "-h"]).joined(separator: " ")
            out += "      if [[ \"$cur\" == -* ]]; then\n        COMPREPLY=($(compgen -W \(sq(flags)) -- \"$cur\"))\n        return\n      fi\n"
            if c.args.contains(where: { isModel($0.kind) }) {
                out += "      COMPREPLY=($(compgen -W \"$(_mlxr_models)\" -- \"$cur\"))\n"
            } else if c.args.contains(where: { isFile($0.kind) }) {
                out += "      COMPREPLY=($(compgen -f -- \"$cur\"))\n"
            }
            out += "      ;;\n"
        }
        out += "  esac\n}\n\ncomplete -F _mlxr mlxr\n"
        return out
    }
}
