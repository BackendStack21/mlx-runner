# Sparse decode attention (experimental)

Branch `feat/sparse-decode-attention`. Status: **written without a Swift toolchain or a Metal GPU.
None of it has been compiled, run or benchmarked yet.** Expect compile fixes before the first test run.

## Why

Decoding one token reads every weight once and the whole KV cache of every layer. On an M2 Pro
(197 GB/s) the README's own numbers put each 1K tokens of context at about 0.77 ms per token on
Qwen3 0.6B and 1.06 ms on Mistral 7B: about 150 and 130 GB/s, so this is memory traffic. Extending
that line (an estimate, not a measurement):

| Model (4-bit) | Context | Dense decode (est.) | Attending ~4.6K keys, upper bound |
|---|---|---|---|
| Qwen3 0.6B | 8K | ~102 tok/s | ~142 tok/s |
| Qwen3 0.6B | 32K | ~35 tok/s | ~142 tok/s |
| Mistral 7B | 8K | ~30 tok/s | ~34 tok/s |
| Mistral 7B | 32K | ~17 tok/s | ~34 tok/s |

The KV cache costs as much as the weights from about 3K tokens on Qwen3 0.6B and about 31K on a
4-bit 7B model. The default budget (2,048 long-range keys + 512 window + 4 sink) attends to about
2.6K keys, so the upper bound is higher than this table's ~4.6K column.

## What it is

Based on *Latent Sparse Attention* (arXiv 2610.42147v2), training-free variant, reduced to what pays
at this engine's context sizes (8K default, about 32K–64K for a 7B model on 16 GB):

| Paper | Here |
|---|---|
| Sink + window + top-k long-range keys; one index list per query shared by all heads | Same |
| Index keys: pooled 4:1, projected to 64 dims (trained, or principal directions) | Pooled 4:1 (mean), **full head size**, f16; no projection yet |
| Ward clustering into buckets, flat or tree search | Not used: a flat scan over pooled blocks, exact top-k |
| Index query: sum over heads | Same: the sum over heads of q · k̄ (per KV group) |
| Prefill sparse too | Prefill and speculative verification stay dense |
| Distillation (D-LSA) / native training (N-LSA) | None: works on any checkpoint |

Per decode step and eligible layer (full-context cache with a split-attention kernel for its head size):

1. `sparse_index`: whenever positions are written (prefill and decode), the blocks they complete get
   their index key: the mean of the block's 4 keys, per KV head, f16. A rewritten position (a new
   turn, a rejected draft) rebuilds its block when the block's last position is written again.
2. `sparse_score`: the decode query scores blocks `[sinkEnd, winStart)` with the sum over its query
   heads of `q_h · k̄_block`.
3. `sparse_select`: one threadgroup takes the exact top-p blocks (radix select on the score bits;
   ties go to the earlier block) and writes the attended positions in ascending order: sink, kept
   blocks, window.
4. `attention_sparse` + `attention_combine`: `attention_split` with key n of the list in place of
   position n. With the list `0..<L` it is bit-identical to `attention_split`.

Dense decode and every default code path are unchanged. Sparse mode is off unless enabled.

## Using it

```bash
MLXR_SPARSE_ATTN=1 mlxr run <model> ...            # default budget: 2048 long-range keys
MLXR_SPARSE_ATTN=1024 MLXR_SPARSE_WINDOW=256 ...   # long-range keys / window
MLXR_SPARSE_MIN=8192 ...                           # dense below this context (default 2·(sink+topK+window) = 5128)
```

It is read once per process: by `Transformer.init` (which allocates the index) and by the engine's
load-time memory check (which counts it). There is no app setting or CLI flag yet.

Measuring (nothing else may use the GPU):

```bash
# dense baseline at long contexts (checks the table above)
MLXR_BENCH_DECODE=1 MLXR_BENCH_CTX=8192,30000 mlxr bench mlx-community/Mistral-7B-Instruct-v0.3-4bit
# dense vs sparse at the same contexts, plus agreement with dense decode on real text
MLXR_SPARSE_ATTN=1 MLXR_BENCH_TEXT=~/book.txt MLXR_BENCH_DECODE=1 MLXR_BENCH_CTX=8192,30000 mlxr bench <model>
```

The long-context section prints prefill time, then `dense @N` and `sparse @N` decode rates. It also
prints how many greedy tokens match dense decode, and the teacher-forced next-token agreement over
64 positions. Use a context below the model's maximum (Mistral 7B: 32,768). Without
`MLXR_BENCH_TEXT` the prompt is pseudo-random tokens: fine for speed, meaningless for agreement.

## Checklist on the Mac

1. `swift build -c release`: fix compile errors (Swift and the runtime-compiled Metal source;
   `GPU.shared` reports Metal errors at the first pipeline lookup).
2. `swift test --filter "Sparse attention"`: plan, index, scores, selection (ties, odd thread counts),
   attention over a list (bitwise vs `attention_split`, a subset vs a CPU reference, clamping).
3. `swift test --filter sparseDecode` with models installed: full-budget sparse decode is
   bit-identical to split decode on Qwen3 0.6B, Llama 3.2 1B, Phi-3.5 and Mistral 7B; below the
   minimum context nothing sparse runs.
4. `swift test`: the existing gates (greedy token identity, prefill vs incremental decode) with
   `MLXR_SPARSE_ATTN` unset.
5. Check that kernel compile time at startup did not grow noticeably (25 new `attention_sparse`
   instantiations).
6. Bench as above: dense first, to confirm the cost model, then sparse.

## Keep or drop

Keep only if decode at ≥ 8K context clearly improves, and agreement with dense decode is high on real
text (long documents, needle retrieval). Sparse mode changes greedy output, which AGENTS.md's
correctness gate forbids for default behaviour. Shipping it means an explicitly opt-in mode with its
own accuracy checks. That decision is open.

## Known gaps / next steps (by expected value)

- **Index cost.** The full-size index reads 1/8 of the KV bytes every step (K only, pooled 4:1). At 32K
  that is about as much as the sparse attention itself. If the profile shows it, add the paper's
  projection: 64 index dims from principal directions of the keys (computable from the prompt, no
  training), or keep only the highest-energy RoPE planes per head, or an int8 index.
- **Selection latency.** One threadgroup per layer does 4 radix rounds plus a compaction. It is
  probably tens of µs per layer, which matters most on small models. Options: the paper's sampled
  threshold estimate (one pass, Algorithm 4), a larger `SSC_BPS`, or sharing one selection across
  neighbouring layers.
- **Dispatch count.** Score + select add two dispatches per layer per decode step.
- **Memory.** The index adds 1/8 of each eligible layer's KV cache, which lowers the largest context
  that fits on a 16 GB Mac by about 11%. A projected index would make this negligible.
- **Coverage.** Gemma's sliding-window layers are bounded already and stay dense. Gemma 4's KV-shared
  layers use their source layer's index (untested). Head sizes without a split kernel stay dense.
- **Wiring.** No app setting, CLI flag or server option yet: add them only once the numbers justify it.
- **Prefill.** Still O(T²) attention: at 32K it is roughly a third of prefill FLOPs on a 7B model.
  Sparse prefill would need the selection per query block; it is out of scope until decode proves out.

## Files

- `Sources/MLXEngine/Kernels.swift`: `SparseParams`, `sparse_index`, `sparse_score`, `sparse_select`,
  `attention_sparse` and its instantiations.
- `Sources/MLXEngine/SparseAttention.swift`: the public config (`SparseAttention`, env parsing), the
  per-step plan, eligibility and memory accounting.
- `Sources/MLXEngine/Transformer.swift`: `init(directory:contextLength:sparse:)`, index allocation, index
  maintenance after `qkv_post`, the sparse branch of decode attention, `sparse` / `sparseAvailable`.
- `Sources/MLXEngine/GPU.swift`: the Swift `SparseParams`.
- `Sources/MLXEngine/Engine.swift`: memory check counts the index when `MLXR_SPARSE_ATTN` is set.
- `Sources/mlxr/main.swift`: `MLXR_BENCH_CTX` long-context decode section (dense / sparse / agreement).
- `Tests/MLXEngineTests/SparseAttentionTests.swift`, `ModelTests.swift`: tests above.
