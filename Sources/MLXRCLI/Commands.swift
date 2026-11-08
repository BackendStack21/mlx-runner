import Foundation
import MLXEngine

extension CLI {
    static let sparseHelp = "sparse decode attention: on, off or a number of long-range keys"
    static let draftHelp = "draft model for speculative decoding: a model or auto"

    /// The command table of `mlxr`.
    public static var commands: [Command] { commands() }

    /// The command table with its outside world injected.
    public static func commands(services: CLIServices = .live, library: ModelLibrary = .live, presets: PresetStore = .live,
                                catalog: Catalog = .standard, now: @escaping @Sendable () -> Date = { Date() },
                                hub: HubServices = .live, engine: GenerationEngine = LiveGenerationEngine(),
                                interrupts: Interrupts = .shared, lineSource: LineSource? = nil,
                                chatHistory: ChatHistory = .live) -> [Command] {
        let runDeps = RunDependencies(resolver: ModelResolver(library: library, presets: presets, catalog: catalog), hub: hub,
                                      engine: engine, interrupts: interrupts, lineSource: lineSource, history: chatHistory)
        let ctxRange = SamplingLimits.context
        let table: [Command] = [
            Command(name: "run", summary: "Run a model", args: [Arg("MODEL", kind: .model), Arg("PROMPT", optional: true, variadic: true)],
                    options: [
                        Opt(long: "prompt", short: "p", value: .text, help: "prompt"),
                        Opt(long: "max-tokens", aliases: ["max"], short: "n", value: .int(1...SamplingLimits.maxTokens), help: "maximum tokens to generate (default: until the model stops)"),
                        Opt(long: "temp", aliases: ["temperature"], value: .real(SamplingLimits.temperature), help: "sampling temperature (default: the model's)"),
                        Opt(long: "top-p", value: .realAboveZero(max: SamplingLimits.topPMax), help: "nucleus sampling probability"),
                        Opt(long: "top-k", value: .int(SamplingLimits.topK), help: "sample from the k likeliest tokens (0 turns it off)"),
                        Opt(long: "min-p", value: .real(SamplingLimits.minP), help: "minimum probability relative to the likeliest token"),
                        Opt(long: "repeat-penalty", value: .realAboveZero(max: SamplingLimits.repeatPenaltyMax), help: "repetition penalty (1 = off)"),
                        Opt(long: "repeat-last-n", value: .int(SamplingLimits.repeatLastN), help: "tokens the repetition penalty looks back over"),
                        Opt(long: "seed", value: .uint64, help: "random seed"),
                        Opt(long: "system", value: .text, help: "system prompt"),
                        Opt(long: "ctx", value: .int(ctxRange), defaultText: "4096", help: "context length"),
                        Opt(long: "raw", help: "send the prompt without the chat template"),
                        Opt(long: "think", value: .choice(["true", "false", "high", "medium", "low"]), implicit: "true", help: "think before answering"),
                        Opt(long: "no-think", help: "answer without thinking"),
                        Opt(long: "ids", help: "print the prompt token ids"),
                        Opt(long: "draft", value: .model, help: draftHelp),
                        Opt(long: "sparse", value: .text, help: sparseHelp),
                        Opt(long: "spec", help: "speculative decoding with n-gram drafts"),
                        Opt(long: "verbose", help: "print timing statistics to stderr"),
                        Opt(long: "hidethinking", help: "do not show the model's thinking"),
                        Opt(long: "nowordwrap", help: "do not wrap words to the next line automatically"),
                        Opt(long: "keepalive", value: .text, help: "accepted for Ollama compatibility; has no effect"),
                        Opt(long: "insecure", help: "accepted for Ollama compatibility; has no effect"),
                        Opt(long: "format", value: .text, help: "not supported by the engine"),
                        Opt(long: "truncate", help: "not supported (embedding models)"),
                        Opt(long: "dimensions", value: .int(0...Int.max), help: "not supported (embedding models)"),
                        Opt(long: "voice", value: .text, help: "not supported (speech models)"),
                        Opt(long: "ask", value: .text, repeatable: true, help: "not supported"),
                        Opt(long: "choice", value: .text, repeatable: true, help: "not supported"),
                        Opt(long: "level", value: .text, repeatable: true, help: "not supported"),
                        Opt(long: "presence-penalty", value: .text, help: "not supported by the engine"),
                        Opt(long: "frequency-penalty", value: .text, help: "not supported by the engine"),
                    ],
                    examples: ["  mlxr run mlx-community/Qwen3-4B-4bit -p \"Hello\" --temp 0"],
                    environment: [("MLXR_SPARSE_ATTN", "sparse decode attention default (0 turns it off)")],
                    run: { inv, console in try await executeRun(inv, console, runDeps) }),
            Command(name: "serve", aliases: ["start"], summary: "Start the OpenAI-compatible server", args: [Arg("MODEL", kind: .model, optional: true)],
                    options: [
                        Opt(long: "port", value: .int(SamplingLimits.port), defaultText: "1234", help: "port to listen on"),
                        Opt(long: "ctx", value: .int(ctxRange), defaultText: "8192", help: "context length"),
                        Opt(long: "lan", help: "listen on the local network (needs an API key)"),
                        Opt(long: "api-key", value: .text, help: "API key clients must send"),
                        Opt(long: "draft", value: .model, help: draftHelp),
                        Opt(long: "sparse", value: .text, help: sparseHelp),
                    ],
                    environment: [("MLXR_API_KEY", "API key, when --api-key is not given")],
                    run: { inv, _ in
                        guard let model = inv.positionals.first else {
                            throw CLIError.failure("serve needs a model for now: mlxr serve MODEL (serving without a model comes with the background server in a later release)")
                        }
                        try await legacyServe(model: model, port: UInt16(inv.int("port") ?? 1234),
                                              ctx: inv.int("ctx") ?? 8192, lan: inv.bool("lan"), apiKey: inv.string("api-key"),
                                              draft: inv.string("draft"), sparse: inv.string("sparse"))
                    }),
            pullCommand(library: library, presets: presets, catalog: catalog, hub: hub),
            showCommand(library: library, presets: presets, catalog: catalog),
            Command(name: "search", summary: "Search Hugging Face for MLX models", args: [Arg("QUERY", variadic: true)],
                    options: [
                        Opt(long: "limit", value: .int(1...100), defaultText: "25", help: "maximum number of results"),
                        Opt(long: "json", help: "print the results as JSON"),
                    ],
                    exactArgWording: true,
                    run: { inv, console in
                        let models = try await services.search(inv.positionals.joined(separator: " "), inv.int("limit") ?? 25)
                        console.out(renderSearch(models, json: inv.bool("json")))
                    }),
            Command(name: "tokenize", summary: "Tokenize text with a model's tokenizer", args: [Arg("MODEL", kind: .model), Arg("TEXT", variadic: true)],
                    options: [Opt(long: "ids", help: "print only the token ids")],
                    group: "Developer", run: { inv, console in
                        try await services.tokenize(TokenizeOptions(model: inv.positionals[0], text: inv.positionals.dropFirst().joined(separator: " "),
                                                                    idsOnly: inv.bool("ids")), console)
                    }),
            Command(name: "bench", summary: "Benchmark a model", args: [Arg("MODEL", kind: .model)],
                    options: [
                        Opt(long: "decode", help: "decode benchmark only"),
                        Opt(long: "ctx", value: .text, help: "long-context contexts, comma separated"),
                    ],
                    environment: [("MLXR_BENCH_DECODE", "decode benchmark only"), ("MLXR_BENCH_CTX", "long-context contexts, comma separated")],
                    group: "Developer", run: { inv, console in
                        try await services.bench(BenchOptions(model: inv.positionals[0], decodeOnly: inv.bool("decode"), contexts: inv.string("ctx")), console)
                    }),
        ] + modelCommands(library: library, presets: presets, catalog: catalog, now: now)
        return table + [completionCommand(table: table)]
    }
}

extension CLI {
    /// Search results: one row per model, or a JSON array with `--json`.
    static func renderSearch(_ models: [HubModel], json: Bool) -> String {
        if json {
            let rows: [[String: Any]] = models.map { m in
                var r: [String: Any] = ["id": m.id]
                if let d = m.downloads { r["downloads"] = d }
                if let l = m.likes { r["likes"] = l }
                if let t = m.modelTypeTag { r["model_type"] = t }
                if let u = m.lastModified { r["last_modified"] = u }
                return r
            }
            let data = (try? JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])) ?? Data("[]".utf8)
            return String(decoding: data, as: UTF8.self) + "\n"
        }
        return models.map {
            String(format: "%-60@ %10d ↓  %@", $0.id as NSString, $0.downloads ?? 0, ($0.modelTypeTag ?? "") as NSString) + "\n"
        }.joined()
    }
}
