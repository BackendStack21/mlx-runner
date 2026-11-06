import Foundation

/// Sparse decode attention: on by default for long contexts. It changes outputs (from `minContext` on).
///
/// Decoding one token reads the whole KV cache of every layer, so past a few thousand tokens the
/// cache, not the weights, sets the speed (Qwen3 0.6B: from about 3K tokens; a 4-bit 7B: from about
/// 31K). In this mode a full-attention layer's decode query attends only to the first `sink` keys,
/// the last `window` keys and the `topK` long-range keys it scores highest, after the training-free
/// variant of Latent Sparse Attention (arXiv 2610.42147) without its clustering and tree search:
///
/// - Every `pool` (4) consecutive keys get one index key, their mean (f16), maintained whenever
///   positions are written. It costs a quarter of the K cache of each such layer.
/// - The query scores the index blocks between the sink and the window, per KV head with the sum
///   over that head's query heads of their dot products, so each KV head keeps its own blocks
///   (`perHead`, the default; the same bytes as a shared list). With `perHead` false the sum runs
///   over every head and one selection serves them all, as in the paper.
/// - The best `topK / pool` blocks are kept (`sparse_select`, exact top-k, ties to the earlier block)
///   and attention runs over the sink, the kept keys and the window, in position order.
///
/// Only decode steps (one token) of layers that attend over a full-context cache use it (sliding-window
/// layers are bounded already); prefill and speculative verification stay dense. Below `minContext`
/// decode stays dense too. With a budget that covers the whole context, the result is bit-identical
/// to split decode (`attention_split`; `ModelTests.sparseDecodeWithFullBudgetIsBitIdentical`), which
/// itself equals the single-threadgroup decode kernel for head sizes up to 128.
///
/// `MLXR_SPARSE_ATTN=0` (or off) turns it off, or sets the number of long-range keys, with `MLXR_SPARSE_WINDOW`,
/// `MLXR_SPARSE_MIN`, `MLXR_SPARSE_SHARED=1` (one selection for every head); see
/// docs/sparse-decode-attention.md.
public struct SparseAttention: Equatable, Sendable {
    /// Keys per index block.
    public static let pool = 4
    /// Long-range keys kept per decode step (whole blocks).
    public var topK: Int
    /// The most recent keys, always attended (at least `window`, up to `window + pool - 1`).
    public var window: Int
    /// The first keys (attention sinks), always attended (rounded up to whole blocks).
    public var sink: Int
    /// Decode switches to sparse attention from this context length (position + 1) on.
    public var minContext: Int
    /// One selection per KV head (true) or one shared by every head (false).
    public var perHead: Bool

    /// Values are clamped into range; `minContext` defaults to twice the attended keys.
    public init(topK: Int = 2048, window: Int = 512, sink: Int = 4, minContext: Int? = nil, perHead: Bool = true) {
        let p = Self.pool
        self.topK = max(p, min(topK, 1 << 22)) / p * p
        self.window = max(p, min(window, 1 << 22))
        self.sink = max(0, min(sink, 1 << 16))
        self.minContext = max(1, min(minContext ?? 2 * (self.sink + self.topK + self.window), 1 << 30))
        self.perHead = perHead
    }

    /// The default budget: 2,048 long-range keys, a 512-key window and 4 sink keys, per KV head.
    public static let standard = SparseAttention()

    /// From the environment (sparse attention is on by default): see `parse(environment:)`.
    public static let fromEnvironment: SparseAttention? = parse(environment: ProcessInfo.processInfo.environment)

    /// `MLXR_SPARSE_ATTN`: unset, empty, `1`, `on`, `true` or `yes` is the default budget; a larger
    /// number is the long-range key budget; `0`, `off`, `false`, `no` or anything else is nil
    /// (dense decode), so a mistyped value never turns it on.
    static func parse(environment env: [String: String]) -> SparseAttention? {
        let s = (env["MLXR_SPARSE_ATTN"] ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        if ["", "1", "on", "true", "yes"].contains(s) { return configured(topK: standard.topK, environment: env) }
        guard let v = Int(s), v > 1 else { return nil }
        return configured(topK: v, environment: env)
    }

    /// A budget of `topK` long-range keys with the rest from the environment: `MLXR_SPARSE_WINDOW`
    /// and `MLXR_SPARSE_MIN` (the window and the minimum context), `MLXR_SPARSE_SHARED=1` (one
    /// selection shared between the heads).
    public static func configured(topK: Int, environment env: [String: String] = ProcessInfo.processInfo.environment) -> SparseAttention {
        SparseAttention(topK: topK,
                        window: env["MLXR_SPARSE_WINDOW"].flatMap { Int($0.trimmingCharacters(in: .whitespaces)) } ?? standard.window,
                        minContext: env["MLXR_SPARSE_MIN"].flatMap { Int($0.trimmingCharacters(in: .whitespaces)) },
                        perHead: env["MLXR_SPARSE_SHARED"]?.trimmingCharacters(in: .whitespaces) != "1")
    }

    /// The decode step whose query sits at `position`: nil when it stays dense.
    func plan(position: Int) -> SparsePlan? {
        let pool = Self.pool
        let L = position + 1
        guard position >= 0, L >= minContext else { return nil }
        let sinkEnd = (sink + pool - 1) / pool * pool
        guard L - window > sinkEnd else { return nil }
        let winStart = (L - window) / pool * pool   // L - window > 0: rounds down, so the window grows
        let b0 = sinkEnd / pool, nb = winStart / pool - b0
        guard nb > 0 else { return nil }
        let p = min(topK / pool, nb)
        let count = sinkEnd + p * pool + (L - winStart)   // <= L
        let params = SparseParams(pool: Int32(pool), b0: Int32(b0), nb: Int32(nb), p: Int32(p),
                                  sinkEnd: Int32(sinkEnd), winStart: Int32(winStart), L: Int32(L))
        return SparsePlan(params: params, count: count)
    }
}

/// One sparse decode step: the scored blocks and the attended positions (layer fields of `params`,
/// nKV / D / maxSeq / nH, are filled per layer).
struct SparsePlan {
    var params: SparseParams
    /// Entries of the position list `sparse_select` writes.
    let count: Int
}

extension Transformer {
    /// Largest nKV · head_dim the scoring kernel holds (`SPARSE_MAX_W`).
    static let sparseMaxWidth = 4096
    /// `sparse_score` launch shape: SSC_NSG simdgroups per threadgroup, each scoring SSC_BPS blocks.
    static let sparseScoreThreads = 8 * 32, sparseScoreBlocks = 8 * 8

    /// Can this layer keep a sparse index (with a sparse kernel for its head size)?
    static func sparseEligible(_ spec: ModelConfig.LayerSpec, cacheLen: Int, context: Int) -> Bool {
        spec.hasOwnKV && !spec.sliding && cacheLen == context && spec.numKVHeads * spec.headDim <= sparseMaxWidth
    }

    /// Upper bound on the bytes `init` allocates for sparse decode attention at a context (the
    /// load-time memory check adds it when the model loads with sparse attention): the index of every eligible
    /// layer, and the block scores and position lists of one row per KV head.
    public static func sparseBytes(config c: ModelConfig, context: Int) -> Int {
        let seq = max(256, context), pool = SparseAttention.pool
        let eligible = c.layers.filter { sparseEligible($0, cacheLen: c.cacheRows($0, context: seq), context: seq) }
        let rows = max(1, eligible.map(\.numKVHeads).max() ?? 1)
        var bytes = rows * ((seq / pool + 1) * 4 + seq * 4)
        for l in eligible {
            bytes += seq / pool * l.numKVHeads * l.headDim * 2
        }
        return bytes
    }
}
