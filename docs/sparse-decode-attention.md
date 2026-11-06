# Sparse decode attention (experimental)

Branch `feat/sparse-decode-attention`. Status: **compiled, tested and measured on an M2 Pro (16 GB,
2026-11-06).** It was written without a Swift toolchain or a Metal GPU, then picked up on a Mac:
- The Swift and the Metal source compiled as written, with no fixes.
- `swift test --filter SparseAttentionTests` passes: 6 tests, 18 cases.
- `sparseDecodeWithFullBudgetIsBitIdentical` passes on Qwen3 0.6B, Llama 3.2 1B, Phi-3.5 mini and
  Mistral 7B, and `sparseDecodeStaysDenseBelowItsMinimumContext` passes.
- With `MLXR_SPARSE_ATTN` unset, the full `swift test` passes: 161 tests in 28 suites.
- Start-up is unchanged: a one-token `mlxr run` on Qwen3 0.6B takes 0.46 s on this branch and 0.47 s
  on `main` (warm).
- The decision to keep, tune or drop it is open. See "Measured" below.

## Measured

M2 Pro (16-core GPU, 16 GB), 4-bit models, nothing else on the GPU. The prompt is *Pride and Prejudice*
(Project Gutenberg #1342, about 177K tokens, so no context repeats). Default budget: 2,048 long-range
keys, a 512-key window and 4 sink keys. Each figure is the better of 2 runs; the runs agreed within 2%.
"Dense" is a separate process with `MLXR_SPARSE_ATTN` unset. The in-process dense line of the sparse
runs was within 0.5% of it, so keeping the index costs nothing measurable. Agreement is over 64 decode
steps: "greedy" counts how many tokens match dense decode before the first difference, and "teacher-forced"
counts how often the next-token argmax matches when both are fed dense decode's tokens.

| Model | Context | Dense tok/s | Sparse tok/s | Speed-up | Greedy prefix | Teacher-forced |
|---|---|---|---|---|---|---|
| Qwen3 0.6B | 8,192 | 112.9 | 142.7 | 1.26× | 64/64 | 64/64 (100%) |
| Qwen3 0.6B | 16,384 | 70.9 | 129.0 | 1.82× | 12/64 | 59/64 (92%) |
| Qwen3 0.6B | 30,000 | 43.8 | 104.7 | 2.39× | 4/64 | 62/64 (97%) |
| Mistral 7B | 8,192 | 31.6 | 35.0 | 1.11× | 4/64 | 62/64 (97%) |
| Mistral 7B | 16,384 | 25.7 | 33.9 | 1.32× | 13/64 | 57/64 (89%) |
| Mistral 7B | 30,000 | 19.7 | 31.6 | 1.61× | 9/64 | 59/64 (92%) |

GPU time per token, dense → sparse: Qwen3 0.6B 8.85 → 7.00, 14.09 → 7.74 and 22.83 → 9.54 ms;
Mistral 7B 31.61 → 28.58, 38.87 → 29.46 and 50.83 → 31.63 ms. For reference, dense decode at 16
tokens of context is 3.33 ms (Qwen3 0.6B) and 25.85 ms (Mistral 7B). Prefill, dense in both modes:
Qwen3 0.6B 1,076 / 630 / 370 tok/s and Mistral 7B 185 / 145 / 106 tok/s at 8K / 16K / 30K.

Against the bar in step 6:
- Sparse is 1.61× dense at 30K on Mistral 7B, above the 1.3× bar.
- At 8K it is faster than dense, not slower.
- Teacher-forced agreement reaches 95% only at 8K on both models and at 30K on Qwen3 0.6B. It falls
  to 89–92% at 16K and at Mistral's 30K. With 64 positions, each token is 1.6 points, so 95% means
  at least 61 of 64.

A larger budget (`MLXR_SPARSE_ATTN=4096 MLXR_SPARSE_WINDOW=1024`, so about 5.1K keys attended; one
run, same prompt):

| Model | Context | Dense tok/s | Sparse tok/s | Speed-up | Greedy prefix | Teacher-forced |
|---|---|---|---|---|---|---|
| Qwen3 0.6B | 16,384 | 71.1 | 105.1 | 1.48× | 12/64 | 59/64 (92%) |
| Qwen3 0.6B | 30,000 | 43.8 | 90.6 | 2.07× | 7/64 | 61/64 (95%) |
| Mistral 7B | 16,384 | 25.7 | 31.3 | 1.22× | 30/64 | 61/64 (95%) |
| Mistral 7B | 30,000 | 19.6 | 29.7 | 1.52× | 21/64 | 61/64 (95%) |

At this budget Mistral 7B meets all three bars at 30K: 1.52× and 61 of 64. The 5 misses on Qwen3
0.6B at 16K do not change with the budget. They are probably near-ties that any change in summation
order flips, rather than keys the selection dropped; this is not checked yet. Agreement measured over
64 positions of one text is a coarse signal. A larger evaluation (several texts, more positions,
needle retrieval) should come before any decision to ship.

The "Why" table below is the estimate made before any measurement. Measured dense decode fell more
slowly with context than it predicted (Qwen3 0.6B at 30K: 43.8 tok/s, estimated about 37).

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

## Continuing on a Mac: instructions for a local agent

You are picking this branch up on an Apple Silicon Mac, where it can finally be compiled, tested and
measured. Read `AGENTS.md` first; its rules apply throughout. In particular: Swift only, no
dependencies, every performance claim needs `mlxr bench` numbers taken with nothing else on the
GPU, and the default code path must keep greedy output token-identical.

**Setup**

```bash
git fetch origin && git checkout feat/sparse-decode-attention
make cli                                  # builds .build/release/mlxr
M=.build/release/mlxr
$M download mlx-community/Qwen3-0.6B-4bit # the model tests skip models that are not installed
$M download mlx-community/Llama-3.2-1B-Instruct-4bit
$M download mlx-community/Mistral-7B-Instruct-v0.3-4bit
$M download mlx-community/Phi-3.5-mini-instruct-4bit   # optional, but the full-budget test covers it
```

**Step 1: make it compile.** `swift build -c release` checks only the Swift. The Metal source in
`Kernels.swift` compiles at runtime when `GPU.shared` is first used. One Metal error fails every
kernel, so run `swift test --filter "Sparse attention"` early: a `kernelCompile` error there prints
the Metal compiler's message. Fix what is wrong and keep each fix minimal. Do not redesign while
fixing compile errors.

**Step 2: unit tests.** `swift test --filter "Sparse attention"` must pass completely. It covers:
- the plan;
- the index (mean of 4 keys);
- the scores (sum over heads);
- the selection (ties, negative scores, 64 and 96 threads, p = N);
- attention over a list: bitwise against `attention_split`, a subset against a CPU reference, and
  clamping.

If a selection case fails, the algorithm itself was checked in a CPU simulation before the
handoff. Suspect the MSL translation (barriers, `simd_prefix_exclusive_sum`, atomics), not the
method.

**Step 3: end-to-end exactness.** `swift test --filter sparseDecode`:
- `sparseDecodeWithFullBudgetIsBitIdentical` must give identical logits to split decode on every
  installed model. If logits differ but are close, the list order or the query-head grouping differs
  from `attention_split`; find it rather than loosening the test.
- `sparseDecodeStaysDenseBelowItsMinimumContext` must show that no sparse dispatch happened.

**Step 4: nothing else changed.** With `MLXR_SPARSE_ATTN` unset, run `swift test` in full and fix
any regression. Also check that start-up did not get noticeably slower: the Metal source
compiles at every launch, and this branch adds 25 `attention_sparse` instantiations. Time a
one-token run (`time $M run <model> -p hi --max 1`, twice, keep the second) on `main` and on this
branch.

**Step 5: measure.** Close other GPU users. Get a long real text, for example a public-domain book
saved as `~/book.txt`. For each model (start with Qwen3 0.6B and Mistral 7B), run:

```bash
# dense baseline: does decode slow down with context as the "Why" table estimates?
MLXR_BENCH_DECODE=1 MLXR_BENCH_CTX=8192,16384,30000 $M bench <model>
# dense vs sparse, with agreement on real text
MLXR_SPARSE_ATTN=1 MLXR_BENCH_TEXT=~/book.txt MLXR_BENCH_DECODE=1 MLXR_BENCH_CTX=8192,16384,30000 $M bench <model>
```

Run each at least twice and keep the better run, as the README's numbers do. Contexts must be
below the model's maximum (Mistral 7B: 32,768; Qwen3 0.6B: 40,960). Record for each context: dense
tok/s, sparse tok/s, the greedy-prefix match and the teacher-forced agreement. If the 16 GB memory
check refuses a context, lower it and note that.

**Step 6: decide, then tune or stop.** Propose a verdict to the user with the numbers. Do not
decide alone. A reasonable bar:
- sparse at least 1.3× dense at about 30K on Mistral 7B;
- not slower than dense at 8K;
- teacher-forced agreement of at least 95% on real text.

If speed falls short, find out which step costs the time before changing anything:
1. Compare GPU time per token (`gpu … ms` in the bench line) with the dense line.
2. Rerun with `MLXR_SPARSE_ATTN=2048` against smaller budgets such as `512` to separate the
   attention cost from the fixed index and selection costs.
3. Try `MLXR_SPARSE_WINDOW=128`.

Then apply the matching item from "Known gaps" below, one change at a time, keeping a change only if
the bench shows a win. If agreement falls short, try a larger `topK` and window before anything
else.

**Step 7: record and hand back.**
- Commit fixes and measurements to this branch with clear messages.
- Put the measured numbers into this document. Replace the "Why" estimates, keeping the estimates
  labelled as such if you leave them.
- Do not merge, tag, bump `Version.swift` or edit the README's performance tables. Do not open a
  pull request unless the user asks.
- Report to the user: what failed to compile and how it was fixed, the test results, the
  measurement table, and your recommendation.

**Things that must stay true**
- `MLXR_SPARSE_ATTN` unset means no sparse buffers, no sparse dispatches, and outputs bit-identical
  to `main`.
- Every GPU index derives from host-checked values. Positions in the list are clamped in
  `attention_sparse`, and the plan guarantees `count <= L <= maxSeq`. Keep both if you change the
  selection.
- The index must be rebuilt for any block whose positions are rewritten (new turn, rejected draft):
  see the comment above the `sparse_index` dispatch in `Transformer.encode`.

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
  layers use their source layer's index when the shapes match (untested). Head sizes without a split
  kernel stay dense. A draft model for speculative decoding always loads dense.
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
