import Foundation
import MLXEngine

extension CLI {
    static let sparseHelp = "sparse decode attention: on, off or a number of long-range keys"
    static let draftHelp = "draft model for speculative decoding: a model or auto"

    /// The command table of `mlxr`.
    public static var commands: [Command] {
        [
            Command(name: "run", summary: "Run a model", args: [Arg("MODEL", kind: .model), Arg("PROMPT", optional: true, variadic: true)],
                    options: [
                        Opt(long: "prompt", short: "p", value: .text, help: "prompt"),
                        Opt(long: "max-tokens", aliases: ["max"], short: "n", value: .int(1...32768), defaultText: "256", help: "maximum tokens to generate"),
                        Opt(long: "temp", aliases: ["temperature"], value: .real(0...2), defaultText: "0", help: "sampling temperature"),
                        Opt(long: "ctx", value: .int(256...1_048_576), defaultText: "4096", help: "context length"),
                        Opt(long: "raw", help: "send the prompt without the chat template"),
                        Opt(long: "think", value: .choice(["true", "false", "high", "medium", "low"]), implicit: "true", help: "think before answering"),
                        Opt(long: "no-think", help: "answer without thinking"),
                        Opt(long: "ids", help: "print the prompt token ids"),
                        Opt(long: "draft", value: .model, help: draftHelp),
                        Opt(long: "sparse", value: .text, help: sparseHelp),
                        Opt(long: "spec", help: "speculative decoding with n-gram drafts"),
                    ],
                    examples: ["  mlxr run mlx-community/Qwen3-4B-4bit -p \"Hello\" --temp 0"],
                    environment: [("MLXR_SPARSE_ATTN", "sparse decode attention default (0 turns it off)")],
                    run: { inv, _ in
                        let words = inv.positionals.dropFirst().joined(separator: " ")
                        let thinking: Bool? = inv.bool("no-think") ? false : inv.has("think") ? inv.string("think") != "false" : nil
                        try await legacyRun(LegacyRun(
                            model: inv.positionals[0],
                            prompt: inv.string("prompt") ?? (words.isEmpty ? "Write a haiku about Apple Silicon." : words),
                            maxTokens: inv.int("max-tokens") ?? 256, temp: Float(inv.double("temp") ?? 0),
                            ctx: inv.int("ctx") ?? 4096, raw: inv.bool("raw"), thinking: thinking, showIds: inv.bool("ids"),
                            draft: inv.string("draft"), sparse: inv.string("sparse"), spec: inv.bool("spec")))
                    }),
            Command(name: "serve", aliases: ["start"], summary: "Start the OpenAI-compatible server", args: [Arg("MODEL", kind: .model)],
                    options: [
                        Opt(long: "port", value: .int(1...65535), defaultText: "1234", help: "port to listen on"),
                        Opt(long: "ctx", value: .int(256...1_048_576), defaultText: "8192", help: "context length"),
                        Opt(long: "lan", help: "listen on the local network (needs an API key)"),
                        Opt(long: "api-key", value: .text, help: "API key clients must send"),
                        Opt(long: "draft", value: .model, help: draftHelp),
                        Opt(long: "sparse", value: .text, help: sparseHelp),
                    ],
                    environment: [("MLXR_API_KEY", "API key, when --api-key is not given")],
                    run: { inv, _ in
                        try await legacyServe(model: inv.positionals[0], port: UInt16(inv.int("port") ?? 1234),
                                              ctx: inv.int("ctx") ?? 8192, lan: inv.bool("lan"), apiKey: inv.string("api-key"),
                                              draft: inv.string("draft"), sparse: inv.string("sparse"))
                    }),
            Command(name: "download", aliases: ["pull"], summary: "Download a model from Hugging Face", args: [Arg("MODEL", kind: .model)],
                    run: { inv, _ in try await legacyDownload(inv.positionals[0]) }),
            Command(name: "list", aliases: ["ls"], summary: "List installed models", run: { _, _ in legacyList() }),
            Command(name: "show", summary: "Show information for a model", args: [Arg("MODEL", kind: .model)],
                    run: { _, _ in throw CLIError.failure("show is not available yet") }),
            Command(name: "search", summary: "Search Hugging Face for MLX models", args: [Arg("QUERY", variadic: true)],
                    run: { inv, _ in try await legacySearch(inv.positionals.joined(separator: " ")) }),
            Command(name: "tokenize", summary: "Tokenize text with a model's tokenizer", args: [Arg("MODEL", kind: .model), Arg("TEXT", variadic: true)],
                    group: "Developer", run: { inv, _ in try legacyTokenize(inv.positionals[0], inv.positionals.dropFirst().joined(separator: " ")) }),
            Command(name: "bench", summary: "Benchmark a model", args: [Arg("MODEL", kind: .model)],
                    environment: [("MLXR_BENCH_DECODE", "decode benchmark only"), ("MLXR_BENCH_CTX", "long-context contexts, comma separated")],
                    group: "Developer", run: { inv, _ in try legacyBench(inv.positionals[0]) }),
        ]
    }
}
