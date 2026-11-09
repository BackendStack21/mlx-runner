import Foundation
import MLXEngine

extension CLI {
    static let notSupported = ["create", "push", "signin", "signout", "launch"]

    /// list, rm, cp, ps, stop and the commands mlxr does not have.
    static func modelCommands(library: ModelLibrary, presets: PresetStore, catalog: Catalog,
                              now: @escaping @Sendable () -> Date, interrupts: Interrupts = .shared) -> [Command] {
        let resolver = ModelResolver(library: library, presets: presets, catalog: catalog)
        return [
            Command(name: "list", aliases: ["ls"], summary: "List installed models", args: [Arg("PREFIX", optional: true)],
                    options: [Opt(long: "json", help: "print the models as JSON")],
                    run: { inv, console in
                        console.out(renderList(library: library, presets: presets, prefix: inv.positionals.first,
                                               json: inv.bool("json"), now: now()))
                    }),
            Command(name: "rm", summary: "Remove a model", args: [Arg("MODEL", kind: .model, variadic: true)],
                    options: [Opt(long: "yes", short: "y", help: "do not ask before deleting from the Hugging Face cache")],
                    run: { inv, console in
                        try remove(inv, console, library: library, presets: presets, resolver: resolver, interrupts: interrupts)
                    }),
            Command(name: "cp", summary: "Copy a model", args: [Arg("SOURCE", kind: .model), Arg("DESTINATION")],
                    run: { inv, console in
                        let dest = inv.positionals[1]
                        try PresetStore.validateName(dest)
                        let source = inv.positionals[0]
                        let r = try resolver.resolve(source)
                        guard let repo = r.repo else { throw CLIError.failure("cannot copy a model path; use a model name") }
                        try presets.save(Preset(name: dest, repo: repo, system: r.preset?.system, parameters: r.preset?.parameters ?? [:]))
                        console.out("copied '\(source)' to '\(dest)'\n")
                        console.err("success\n")
                    }),
            Command(name: "ps", summary: "List running models", args: [Arg("PREFIX", optional: true)],
                    run: { _, console in
                        console.out(renderTable(header: ["NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "RUNNER", "UNTIL"], rows: []))
                    }),
            Command(name: "stop", summary: "Stop a running model", args: [Arg("MODEL", kind: .model)],
                    run: { inv, _ in
                        throw CLIError.failure("couldn't find model \"\(inv.positionals[0])\" to stop\n"
                            + "mlxr runs a model only while a command uses it and keeps no background server; use 'mlxr serve MODEL' to start one")
                    }),
        ] + unsupportedCommands
    }

    /// create, push, signin, signout and launch exist in Ollama only.
    static var unsupportedCommands: [Command] {
        func stub(_ name: String, aliases: [String] = [], summary: String, options: [Opt] = []) -> Command {
            let hint = name == "create" ? "; use 'mlxr pull <hf-repo>'" : ""
            return Command(name: name, aliases: aliases, summary: summary, args: [Arg("ARGS", optional: true, variadic: true)],
                           options: options, run: { _, _ in throw CLIError.failure("'\(name)' is not supported by mlxr\(hint)") })
        }
        return [
            stub("create", summary: "Create a model (not supported)", options: [
                Opt(long: "file", short: "f", value: .text, help: "Modelfile"), Opt(long: "quantize", short: "q", value: .text, help: "quantization"),
                Opt(long: "combine", help: "combine"), Opt(long: "draft-quantize", value: .text, help: "draft quantization"),
                Opt(long: "force", help: "force")]),
            stub("push", summary: "Push a model (not supported)", options: [Opt(long: "insecure", help: "use an insecure registry")]),
            stub("signin", aliases: ["login"], summary: "Sign in (not supported)"),
            stub("signout", aliases: ["logout"], summary: "Sign out (not supported)"),
            stub("launch", summary: "Launch an integration (not supported)", options: [
                Opt(long: "model", short: "m", value: .text, help: "model"), Opt(long: "config", help: "config"),
                Opt(long: "restore", help: "restore"), Opt(long: "yes", short: "y", help: "assume yes")]),
        ]
    }

    private static func renderList(library: ModelLibrary, presets: PresetStore, prefix: String?, json: Bool, now: Date) -> String {
        struct Row { var name: String; var id: String; var size: Int?; var modified: Date? }
        let models = library.scan()
        var rows = models.map { Row(name: $0.repo, id: modelID($0.repo), size: $0.sizeBytes, modified: $0.modified) }
        for p in presets.list() {
            let base = models.first { $0.repo.lowercased() == p.repo.lowercased() }
            rows.append(Row(name: p.name, id: modelID(p.repo), size: base?.sizeBytes, modified: nil))
        }
        if let prefix = prefix?.lowercased(), !prefix.isEmpty {
            rows = rows.filter { r in
                r.name.lowercased().hasPrefix(prefix) || (r.name.split(separator: "/").last.map { $0.lowercased().hasPrefix(prefix) } ?? false)
            }
        }
        if json {
            let iso = ISO8601DateFormatter()
            let items: [[String: Any]] = rows.map { r in
                var d: [String: Any] = ["name": r.name, "id": r.id, "size": r.size ?? 0]
                if let m = r.modified { d["modified"] = iso.string(from: m) }
                return d
            }
            let data = (try? JSONSerialization.data(withJSONObject: items, options: [.prettyPrinted, .sortedKeys])) ?? Data("[]".utf8)
            return String(decoding: data, as: UTF8.self) + "\n"
        }
        return renderTable(header: ["NAME", "ID", "SIZE", "MODIFIED"], rows: rows.map {
            [$0.name, $0.id, $0.size.map(humanBytes) ?? "-", $0.modified.map { relativeTime($0, now: now) } ?? "-"]
        })
    }

    private enum Removal { case preset(String), model(LocalModel) }

    private static func remove(_ inv: Invocation, _ console: Console, library: ModelLibrary, presets: PresetStore,
                               resolver: ModelResolver, interrupts: Interrupts) throws {
        let models = library.scan()
        var plan: [(arg: String, what: Removal)] = []
        var seen = Set<String>()
        for arg in inv.positionals {
            if presets.load(arg) != nil {
                if seen.insert("preset:" + arg).inserted { plan.append((arg, .preset(arg))) }
                continue
            }
            let r = try resolver.resolve(arg)
            guard let repo = r.repo, let m = models.first(where: { $0.repo.lowercased() == repo.lowercased() }) else {
                throw CLIError.failure("model '\(arg)' not found")
            }
            if seen.insert("model:" + m.repo).inserted { plan.append((arg, .model(m))) }
        }
        if !inv.bool("yes") && console.stdinIsTTY && console.stderrIsTTY {
            for (arg, what) in plan {
                guard case .model(let m) = what, m.source == .huggingFaceCache else { continue }
                console.err("Delete '\(arg)' from the shared Hugging Face cache? [y/N] ")
                guard let answer = console.readLine()?.trimmingCharacters(in: .whitespaces).lowercased(), ["y", "yes"].contains(answer) else {
                    throw CLIError.failure("aborted")
                }
            }
        }
        // Ctrl-C waits for the delete in progress (a half-deleted model directory can still look
        // installed), then stops before the next one.
        defer { interrupts.enter(.idle) }
        for (arg, what) in plan {
            interrupts.enter(.writing)
            switch what {
            case .preset(let name): try presets.delete(name)
            case .model(let m): try library.delete(m)
            }
            console.out("deleted '\(arg)'\n")
            if interrupts.finishWriting() { throw CLIError.interrupted }
        }
    }
}
