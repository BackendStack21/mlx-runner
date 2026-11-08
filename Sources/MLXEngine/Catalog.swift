import Foundation

/// One curated model with the Ollama-style names it answers to (`qwen3:8b`, `qwen3`, `qwen3:latest`).
public struct CatalogModel: Identifiable, Hashable, Sendable {
    /// The Hugging Face repo id.
    public let id: String
    public let title: String
    public let blurb: String
    public let gigabytes: Double
    /// Lowercase alias names, including the bare family name and `:latest` for the family default.
    public let aliases: [String]

    public init(id: String, title: String, blurb: String, gigabytes: Double, aliases: [String] = []) {
        self.id = id; self.title = title; self.blurb = blurb; self.gigabytes = gigabytes; self.aliases = aliases
    }
}

/// Curated models, shared by the app's recommendations and the CLI's name resolver.
public struct Catalog: Sendable {
    public var models: [CatalogModel]

    public init(models: [CatalogModel]) { self.models = models }

    /// The built-in catalog.
    public static let standard = Catalog(models: [
        .init(id: "mlx-community/Qwen3-0.6B-4bit", title: "Qwen3 0.6B", blurb: "Tiny and blazing fast, with thinking mode", gigabytes: 0.35, aliases: ["qwen3:0.6b"]),
        .init(id: "mlx-community/Qwen3.5-0.8B-4bit", title: "Qwen3.5 0.8B", blurb: "Newest tiny Qwen: hybrid linear attention, quick direct answers", gigabytes: 0.65, aliases: ["qwen3.5:0.8b", "qwen3.5", "qwen3.5:latest"]),
        .init(id: "mlx-community/Llama-3.2-3B-Instruct-4bit", title: "Llama 3.2 3B", blurb: "Meta's compact all-rounder", gigabytes: 1.8, aliases: ["llama3.2:3b", "llama3.2", "llama3.2:latest"]),
        .init(id: "mlx-community/Qwen3-4B-4bit", title: "Qwen3 4B", blurb: "Strong reasoning in a small footprint", gigabytes: 2.3, aliases: ["qwen3:4b"]),
        .init(id: "mlx-community/Phi-3.5-mini-instruct-4bit", title: "Phi-3.5 mini", blurb: "Microsoft's efficient 3.8B model", gigabytes: 2.2, aliases: ["phi3.5", "phi3.5:latest"]),
        .init(id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit", title: "Mistral 7B v0.3", blurb: "Classic, fluent 7B instruct model", gigabytes: 4.1, aliases: ["mistral:7b", "mistral", "mistral:latest"]),
        .init(id: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", title: "Qwen2.5 Coder 7B", blurb: "Code generation and review", gigabytes: 4.3, aliases: ["qwen2.5-coder:7b", "qwen2.5-coder", "qwen2.5-coder:latest"]),
        .init(id: "mlx-community/Qwen3-8B-4bit", title: "Qwen3 8B", blurb: "Best quality under 5 GB", gigabytes: 4.6, aliases: ["qwen3:8b", "qwen3", "qwen3:latest"]),
        .init(id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", title: "Llama 3.1 8B", blurb: "Popular 8B with 128K context", gigabytes: 4.5, aliases: ["llama3.1:8b", "llama3.1", "llama3.1:latest"]),
        .init(id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", title: "DeepSeek R1 Distill 7B", blurb: "Step-by-step reasoning", gigabytes: 4.3, aliases: ["deepseek-r1:7b", "deepseek-r1", "deepseek-r1:latest"]),
    ])

    /// The repo id an alias names (case-insensitive), or nil.
    public func repo(forAlias alias: String) -> String? {
        let key = alias.lowercased()
        return models.first { $0.aliases.contains(key) }?.id
    }

    /// Every alias, for suggestions and completions.
    public var aliases: [String] { models.flatMap(\.aliases) }
}
