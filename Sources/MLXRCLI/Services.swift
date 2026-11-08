import Foundation
import MLXEngine

/// Options of `mlxr bench`.
public struct BenchOptions: Equatable, Sendable {
    public var model: String
    /// `--decode`: decode benchmark only (MLXR_BENCH_DECODE).
    public var decodeOnly = false
    /// `--ctx 8192,32768`: long-context contexts (MLXR_BENCH_CTX).
    public var contexts: String? = nil
    public init(model: String, decodeOnly: Bool = false, contexts: String? = nil) {
        self.model = model; self.decodeOnly = decodeOnly; self.contexts = contexts
    }
}

/// Options of `mlxr tokenize`.
public struct TokenizeOptions: Equatable, Sendable {
    public var model: String
    public var text: String
    /// `--ids`: print only the token ids.
    public var idsOnly = false
    public init(model: String, text: String, idsOnly: Bool = false) {
        self.model = model; self.text = text; self.idsOnly = idsOnly
    }
}

/// What the commands reach outside the process (network, models); tests inject fakes.
public struct CLIServices: Sendable {
    public var search: @Sendable (_ query: String, _ limit: Int) async throws -> [HubModel]
    public var bench: @Sendable (BenchOptions, Console) async throws -> Void
    public var tokenize: @Sendable (TokenizeOptions, Console) async throws -> Void

    public init(search: @escaping @Sendable (String, Int) async throws -> [HubModel],
                bench: @escaping @Sendable (BenchOptions, Console) async throws -> Void,
                tokenize: @escaping @Sendable (TokenizeOptions, Console) async throws -> Void) {
        self.search = search; self.bench = bench; self.tokenize = tokenize
    }

    public static let live = CLIServices(
        search: { q, limit in try await HubClient.search(q, limit: limit) },
        bench: { o, _ in
            if o.decodeOnly { setenv("MLXR_BENCH_DECODE", "1", 1) }
            if let c = o.contexts { setenv("MLXR_BENCH_CTX", c, 1) }
            try legacyBench(o.model)
        },
        tokenize: { o, _ in try legacyTokenize(o.model, o.text, idsOnly: o.idsOnly) })
}
