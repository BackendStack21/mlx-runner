import Foundation

/// Sparse decode attention: an experimental, opt-in mode for long contexts. It changes outputs.
///
/// Decoding one token reads the whole KV cache of every layer, so past a few thousand tokens the
/// cache, not the weights, sets the speed (Qwen3 0.6B: from about 3K tokens; a 4-bit 7B: from about
/// 31K). In this mode a full-attention layer's decode query attends only to the first `sink` keys,
/// the last `window` keys and the `topK` long-range keys it scores highest, after the training-free
/// variant of Latent Sparse Attention (arXiv 2610.42147) without its clustering and tree search:
///
/// - Every `pool` (4) consecutive keys get one index key, their mean (f16), maintained whenever
///   positions are written. It costs a quarter of the K cache of each such layer.
/// - The query scores the index blocks between the sink and the window with the sum over its heads
///   of their dot products, so one selection serves every head (as in the paper).
/// - The best `topK / pool` blocks are kept (`sparse_select`, exact top-k, ties to the earlier block)
///   and attention runs over the sink, the kept keys and the window, in position order.
///
/// Only decode steps (one token) of layers that attend over a full-context cache use it (sliding-window
/// layers are bounded already); prefill and speculative verification stay dense. Below `minContext`
/// decode stays dense too. With a budget that covers the whole context, the result is bit-identical
/// to dense decode (`ModelTests.sparseDecodeWithFullBudgetIsBitIdentical`).
///
/// Enable it with `MLXR_SPARSE_ATTN=1` (or the number of long-range keys), `MLXR_SPARSE_WINDOW`,
/// `MLXR_SPARSE_MIN`; see docs/sparse-decode-attention.md.
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

    /// Values are clamped into range; `minContext` defaults to twice the attended keys.
    public init(topK: Int = 2048, window: Int = 512, sink: Int = 4, minContext: Int? = nil) {
        let p = Self.pool
        self.topK = max(p, min(topK, 1 << 22)) / p * p
        self.window = max(p, min(window, 1 << 22))
        self.sink = max(0, min(sink, 1 << 16))
        self.minContext = max(1, min(minContext ?? 2 * (self.sink + self.topK + self.window), 1 << 30))
    }

    /// From the environment: `MLXR_SPARSE_ATTN` = 1 (default budget) or the number of long-range
    /// keys; `MLXR_SPARSE_WINDOW` and `MLXR_SPARSE_MIN` set the window and the minimum context.
    /// nil (dense decode) when unset or 0.
    public static let fromEnvironment: SparseAttention? = {
        let env = ProcessInfo.processInfo.environment
        guard let s = env["MLXR_SPARSE_ATTN"], let v = Int(s), v > 0 else { return nil }
        return SparseAttention(topK: v == 1 ? 2048 : v,
                               window: env["MLXR_SPARSE_WINDOW"].flatMap { Int($0) } ?? 512,
                               minContext: env["MLXR_SPARSE_MIN"].flatMap { Int($0) })
    }()

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
    /// load-time memory check adds it when `MLXR_SPARSE_ATTN` is set): the index of every eligible
    /// layer, the block scores and the position list.
    public static func sparseBytes(config c: ModelConfig, context: Int) -> Int {
        let seq = max(256, context), pool = SparseAttention.pool
        var bytes = (seq / pool + 1) * 4 + seq * 4
        for l in c.layers where sparseEligible(l, cacheLen: c.cacheRows(l, context: seq), context: seq) {
            bytes += seq / pool * l.numKVHeads * l.headDim * 2
        }
        return bytes
    }
}
