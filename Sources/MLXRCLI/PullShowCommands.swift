import Foundation
import MLXEngine

extension CLI {
    /// `pull MODEL` (alias `download`): Hugging Face download with Ollama's progress on stderr; the
    /// saved path goes to stdout.
    static func pullCommand(library: ModelLibrary, presets: PresetStore, catalog: Catalog, hub: HubServices) -> Command {
        let resolver = ModelResolver(library: library, presets: presets, catalog: catalog)
        return Command(
            name: "download", aliases: ["pull"], summary: "Download a model from Hugging Face", args: [Arg("MODEL", kind: .model)],
            options: [Opt(long: "insecure", help: "accepted for Ollama compatibility; has no effect")],
            run: { inv, console in
                if inv.bool("insecure") { console.err("warning: --insecure has no effect: mlxr always verifies Hugging Face downloads\n") }
                let r = try resolver.resolve(inv.positionals[0])
                guard let repo = r.repo else { throw CLIError.failure("'\(inv.positionals[0])' is a path; pull takes a model name") }
                if let dir = r.directory {
                    console.err("success\n")
                    console.out(dir.path + "\n")
                    return
                }
                console.err("pulling manifest\n")
                let details = try await hub.details(repo)
                if !details.isSupported {
                    console.err("warning: \(details.modelType ?? "unknown") architecture is not supported by the engine\n")
                }
                let destination = try hub.destination(repo)
                let renderer = PullRenderer(console: console)
                let hooks = DownloadHooks(progress: { renderer.update(done: $0, total: $1, speed: $2) }, event: { _ in })
                let dir: URL
                do {
                    dir = try await performDownload(start: { try await hub.download(details, destination, hooks) })
                } catch {
                    renderer.finish()
                    throw error
                }
                renderer.finish()
                console.err("verifying sha256 digest\nwriting manifest\nsuccess\n")
                console.out(dir.path + "\n")
            })
    }

    /// `show MODEL`: model information read from the model directory, without loading weights.
    static func showCommand(library: ModelLibrary, presets: PresetStore, catalog: Catalog) -> Command {
        let resolver = ModelResolver(library: library, presets: presets, catalog: catalog)
        let sections = ["license", "modelfile", "parameters", "system", "template"]
        return Command(
            name: "show", summary: "Show information for a model", args: [Arg("MODEL", kind: .model)],
            options: [
                Opt(long: "license", help: "show the license of a model"),
                Opt(long: "modelfile", help: "show the Modelfile of a model"),
                Opt(long: "parameters", help: "show the parameters of a model"),
                Opt(long: "template", help: "show the chat template of a model"),
                Opt(long: "system", help: "show the system message of a model"),
                Opt(long: "verbose", short: "v", help: "show detailed information"),
                Opt(long: "json", help: "print the information as JSON"),
            ],
            run: { inv, console in
                let chosen = sections.filter { inv.bool($0) }
                if chosen.count > 1 {
                    throw CLIError.failure("only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be specified")
                }
                let name = inv.positionals[0]
                let r = try resolver.resolve(name)
                guard let dir = r.directory else {
                    throw CLIError.failure("model '\(name)' not found (pull it with 'mlxr pull \(name)')")
                }
                let info = try ShowInfo.read(directory: dir, repo: r.repo, preset: r.preset)
                info.warnings.forEach { console.err($0 + "\n") }
                if inv.bool("json") { console.out(info.json()); return }
                switch chosen.first {
                case "license": console.out(info.license.map(Self.endingInNewline) ?? "No license was specified for this model.\n")
                case "template": console.out(info.template.map(Self.endingInNewline) ?? "No prompt template was specified for this model.\n")
                case "system": console.out(info.system.map(Self.endingInNewline) ?? "No system message was specified for this model.\n")
                case "parameters":
                    console.out(info.parameters.map { $0.key.padding(toLength: 30, withPad: " ", startingAt: 0) + " " + $0.value + "\n" }.joined())
                case "modelfile": console.out(info.modelfile())
                default: console.out(info.tables())
                }
            })
    }

    static func endingInNewline(_ s: String) -> String { s.hasSuffix("\n") ? s : s + "\n" }
}

/// What `show` reports about a model directory.
struct ShowInfo {
    var repo: String?
    var directory: URL
    var architecture: String
    var contextLength: Int?
    var embeddingLength: Int?
    var quantization: String?
    var capabilities: [String]
    /// Sorted by key.
    var parameters: [(key: String, value: String)]
    var system: String?
    var template: String?
    var license: String?
    var warnings: [String]

    private static let fileLimit = 8 << 20

    private static func readFile(_ url: URL) -> Data? {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue,
              size <= fileLimit else { return nil }
        return try? Data(contentsOf: url)
    }

    private static func text(_ url: URL) -> String? { readFile(url).flatMap { String(data: $0, encoding: .utf8) } }

    static func read(directory: URL, repo: String?, preset: Preset?) throws -> ShowInfo {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = readFile(configURL) else { throw CLIError.failure("cannot read config.json in \(directory.path)") }
        guard let config = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw CLIError.failure("config.json in \(directory.path) is not valid JSON")
        }
        let nested = config["text_config"] as? [String: Any]
        func int(_ key: String) -> Int? {
            ((config[key] as? NSNumber) ?? (nested?[key] as? NSNumber)).flatMap { Int(exactly: $0.doubleValue) }
        }
        let modelType = ((config["model_type"] as? String) ?? "unknown").lowercased()
        let q = (config["quantization"] as? [String: Any]) ?? (config["quantization_config"] as? [String: Any])
        let bits = (q?["bits"] as? NSNumber).map(\.intValue)

        // Chat template and BOS from tokenizer_config.json, then the standalone template files.
        var template: String?
        var bos: String?
        if let d = readFile(directory.appendingPathComponent("tokenizer_config.json")),
           let cfg = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] {
            if let t = cfg["chat_template"] as? String { template = t }
            else if let arr = cfg["chat_template"] as? [[String: Any]] {
                template = (arr.first { $0["name"] as? String == "default" } ?? arr.first)?["template"] as? String
            }
            bos = (cfg["bos_token"] as? String) ?? (cfg["bos_token"] as? [String: Any])?["content"] as? String
        }
        if template == nil { template = text(directory.appendingPathComponent("chat_template.jinja")) }
        if template == nil, let d = readFile(directory.appendingPathComponent("chat_template.json")),
           let o = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] {
            template = o["chat_template"] as? String
        }
        var capabilities = ["completion"]
        if let template {
            let t = ChatTemplate(template: template, modelType: modelType, bosToken: bos)
            if t.supportsTools { capabilities.append("tools") }
            if t.supportsThinkingToggle { capabilities.append("thinking") }
        }

        // Sampling defaults (validated), then the preset's own values.
        let g = GenerationDefaults.load(directory: directory)
        var params: [String: String] = [:]
        if let v = g.temperature { params["temperature"] = "\(v)" }
        if let v = g.topK { params["top_k"] = "\(v)" }
        if let v = g.topP { params["top_p"] = "\(v)" }
        if let v = g.minP { params["min_p"] = "\(v)" }
        if let v = g.repetitionPenalty { params["repeat_penalty"] = "\(v)" }
        for (k, v) in preset?.parameters ?? [:] { params[k] = v }

        var license: String?
        for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"] {
            if let t = text(directory.appendingPathComponent(name)), !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                license = t; break
            }
        }
        let system = preset?.system.flatMap { $0.isEmpty ? nil : $0 }
        return ShowInfo(repo: repo, directory: directory, architecture: modelType, contextLength: int("max_position_embeddings"),
                        embeddingLength: int("hidden_size"), quantization: bits.map { "\($0)-bit" }, capabilities: capabilities,
                        parameters: params.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }, system: system,
                        template: template, license: license, warnings: g.warnings)
    }

    private func section(_ title: String, _ rows: [String]) -> String {
        "  \(title)\n" + rows.map { "    \($0)\n" }.joined() + "\n"
    }

    private func table(_ rows: [(String, String)]) -> [String] {
        let width = (rows.map(\.0.count).max() ?? 0) + 4
        return rows.map { $0.0.padding(toLength: width, withPad: " ", startingAt: 0) + $0.1 }
    }

    /// The first two lines of a text, then `...` when there are more.
    private func excerpt(_ text: String) -> [String] {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let trimmed = lines.last == "" ? Array(lines.dropLast()) : lines
        return trimmed.count > 2 ? Array(trimmed.prefix(2)) + ["..."] : trimmed
    }

    func tables() -> String {
        var model: [(String, String)] = [("architecture", architecture)]
        if let contextLength { model.append(("context length", "\(contextLength)")) }
        if let embeddingLength { model.append(("embedding length", "\(embeddingLength)")) }
        if let quantization { model.append(("quantization", quantization)) }
        var out = section("Model", table(model)) + section("Capabilities", capabilities)
        if !parameters.isEmpty { out += section("Parameters", table(parameters.map { ($0.key, $0.value) })) }
        if let system { out += section("System", excerpt(system)) }
        if let license { out += section("License", excerpt(license)) }
        return out
    }

    func modelfile() -> String {
        var out = "# Modelfile generated by \"mlxr show\"\n\nFROM \(repo ?? directory.path)\n"
        if let template { out += "TEMPLATE \"\"\"\(template)\"\"\"\n" }
        if let system { out += "SYSTEM \"\"\"\(system)\"\"\"\n" }
        for (k, v) in parameters { out += "PARAMETER \(k) \(v)\n" }
        return out
    }

    func json() -> String {
        var details: [String: Any] = ["architecture": architecture]
        if let contextLength { details["context_length"] = contextLength }
        if let embeddingLength { details["embedding_length"] = embeddingLength }
        if let quantization { details["quantization"] = quantization }
        var obj: [String: Any] = ["details": details, "capabilities": capabilities,
                                  "parameters": Dictionary(uniqueKeysWithValues: parameters.map { ($0.key, $0.value) })]
        if let repo { obj["model"] = repo }
        if let template { obj["template"] = template }
        if let system { obj["system"] = system }
        if let license { obj["license"] = license }
        let data = (try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])) ?? Data("{}".utf8)
        return String(decoding: data, as: UTF8.self) + "\n"
    }
}
