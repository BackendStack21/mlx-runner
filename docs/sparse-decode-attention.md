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

A second pass the same day (below) carried out steps 1–3 of the review panel's plan:
- a stronger agreement bench;
- split and sparse attention for Qwen3 14B;
- one selection per KV head, now the default (better on Qwen3 0.6B and 8B, worse on Mistral 7B).

With `MLXR_SPARSE_ATTN` unset the full `swift test` passes: 164 tests in 29 suites. Start-up is still
0.46–0.47 s warm.

## Second pass (2026-11-06): panel steps 1–3

**Bench.** The `MLXR_BENCH_CTX` section of `mlxr bench` now also reports the following.
- **Teacher-forced on the text after the prompt** (`MLXR_BENCH_POSITIONS`, default 512):
  KL(dense‖sparse), the text's NLL under both, and top-1 agreement bucketed by dense decode's
  top-1/top-2 logit margin. Each position runs sparse first and dense second, so the cache stays
  dense decode's.
- **A control:** the dense forward pass, teacher-forced on dense decode's greedy tokens, must agree
  64 of 64. It always did.
- **`MLXR_BENCH_TEXT_OFFSETS`:** prompts from several places in the text, so context length and text
  region are no longer confounded.
- **Five passkey needles** at 10–90% depth. Each question is prefilled dense except its last token,
  so the answer's first token is decoded in the mode being measured.
- **The cost of a dense verification pass** of T = 1, 2, 5 and 8 tokens.

**The misses are not all near-ties.** At 8K, every one of Qwen3 0.6B's 4 misses in 128 positions
had a dense top-1 margin below 0.5 nats; mean KL was 0.0035 and the text NLL was unchanged.
At 30K, though, the shared selection cost real quality. Qwen3 0.6B, 512 positions in each of two
regions of the book:

| Context | Region | Mean KL (nats) | Text NLL Δ (nats/token) | Top-1 agreement | Agreement at margin ≥ 1 |
|---|---|---|---|---|---|
| 16K | start | 0.035 | +0.022 (perplexity ×1.022) | 91.4% | 248/255 |
| 16K | +60K tokens | 0.012 | +0.004 | 93.4% | 230/230 |
| 30K | start | 0.033 | +0.049 (×1.050) | 88.9% | 229/232 |
| 30K | +60K tokens | 0.046 | +0.053 (×1.055) | 88.3% | 258/261 |

Needle recall was the same as dense in every run.

**One selection per KV head (now the default; `MLXR_SPARSE_SHARED=1` restores the shared one).**
- `sparse_score` writes one row of block scores per KV head: the sum over that head's query heads of
  q · k̄.
- `sparse_select` runs one threadgroup per row.
- `attention_sparse` reads its own KV head's list.

Attention reads the same bytes as before. The score maps each KV head to its own group of lanes, so
it costs the same as the shared score: 216 against 226 µs per layer at 30K, in the kernel timing
below. Same runs as the table above:

| Context | Region | Mean KL, shared → per head | Text NLL Δ, shared → per head | Top-1 agreement |
|---|---|---|---|---|
| 16K | start | 0.035 → 0.004 | +0.022 → −0.011 | 91.4% → 96.1% |
| 16K | +60K tokens | 0.012 → 0.005 | +0.004 → −0.019 | 93.4% → 96.5% |
| 30K | start | 0.033 → 0.010 | +0.049 → +0.016 | 88.9% → 93.9% |
| 30K | +60K tokens | 0.046 → 0.013 | +0.053 → +0.012 | 88.3% → 92.2% |

Where dense decode's top-1 margin is at least 1 nat, per-head sparse agreed at 977 of 978
positions. Decode speed was unchanged (30K: 105.7 → 104.2 tok/s; 16K: 130.8 → 128.2).

**Qwen3 14B.** The split kernel now has G = 5 and 8 for head size 128 (Qwen3 and Qwen2.5 14B and
32B). Split decode on 14B is bit-identical to the single kernel, and the prefill and sparse
full-budget tests pass. The 14B test cases run only with `MLXR_TEST_LARGE=1`: next to the Gemma
suite, 14B pushes a 16 GB Mac into swap. Dense decode before the change: 68.7 ms/token at 8K and
91.0 ms at 16K over a 47.2 ms base, i.e. about 61 GB/s of KV traffic against about 178 GB/s on the
other models.

**Kernel timing** (`MLXR_KERNEL_TIMING=1 swift test --filter SparseTimingTests`). One sparse layer at
Qwen3 0.6B's shapes, with a distinct cache and index per layer, so nothing stays cached between
layers:

| Context | Score | Select | Attention + combine (12-key floor) | Sparse total | Dense split + combine |
|---|---|---|---|---|---|
| 8K | 34 µs | 15 µs | 76 µs (8) | 126 µs | 196 µs |
| 30K | 106 µs | 33 µs | 78 µs (8) | 217 µs | 690 µs |

217 µs × 28 layers = 6.1 ms, which matches the 6.2 ms measured end to end. Scoring at 30K reads
15.1 MB at about 142 GB/s; attention reads 10.5 MB at about 157 GB/s, compared with dense attention's
179 GB/s.

**Tried and rejected** (the timing above, before against after):
- **Aggregating the selection histogram per simdgroup** (simd_min / simd_sum leaders): select went
  from 24–27 to 37 µs at 30K, i.e. slower. Atomic contention was not the bottleneck, so the review
  panel's estimate of −0.9 to −1.6 ms did not hold. Removing the selection entirely would save at
  most about 0.8 ms at 30K.
- **Prefetching list entries in `attention_sparse`** (each lane loads one of the next 32 entries and
  `simd_shuffle` hands them out): 78.2 → 76.8 µs, i.e. no change.
- **Fewer blocks per score simdgroup** (`SSC_BPS` 4, 2 and 1 instead of 8): within noise.

What remains is bytes: the index (15 MB per layer at 30K) and the attended keys (10.5 MB). Cutting
either one means a smaller index or reusing a selection (panel items 6 and 7), which changes outputs.

**Verification cost.** A dense verification pass of 5 tokens costs about 3.1× one decode step:

| Model | Context | T = 1 | T = 2 | T = 5 | T = 8 |
|---|---|---|---|---|---|
| Qwen3 0.6B | 16K | 14.1 ms | 19.6 ms | 43.0 ms | 61.9 ms |
| Qwen3 0.6B | 30K | 22.8 ms | 31.5 ms | 70.5 ms | 105.2 ms |

The sparse-draft, dense-verify route (panel item 8) therefore needs a verification attention kernel
that reads K/V once for all T tokens before it can pay.

### Re-bench after the second pass

Setup:
- `bc95da5` + `1c01e65`: per-head selection, default budget (2,048 + 512 + 4).
- M2 Pro, nothing else on the GPU.
- *Pride and Prejudice*, 512 teacher-forced positions per context.
- One run each. Earlier repeats agreed within 2%, except where noted.
- "Decisive" is agreement where dense decode's top-1 margin is at least 1 nat.
- Needle recall was the same as dense in every run (5/5, or 4/5 for both on Qwen3 0.6B).

**Qwen3 14B dense decode, before and after the split kernel** (same settings,
`MLXR_BENCH_POSITIONS=0`). The output is bit-identical.

| Context | Before | After | Change |
|---|---|---|---|
| 8K | 68.7 ms (14.6 tok/s) | 57.1 ms (17.5 tok/s) | +20% |
| 16K | 91.0 ms (11.0 tok/s) | 71.4 ms (14.0 tok/s) | +27% |

**Sparse against dense:**

| Model | Context | Dense tok/s | Sparse tok/s | Speed-up | Mean KL | Text NLL Δ | Top-1 | Decisive |
|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | 8K | 113.2 | 144.5 | 1.28× | 0.0013 / 0.0030 | −0.004 / −0.015 | 96.9% / 95.7% | 433/433 |
| Qwen3 0.6B | 16K | 71.0 | 130.7 | 1.84× | 0.0042 / 0.0052 | −0.011 / −0.019 | 96.1% / 96.5% | 485/485 |
| Qwen3 0.6B | 30K | 43.8 | 103.5 | 2.36× | 0.0103 / 0.0132 | +0.016 / +0.012 | 93.9% / 92.2% | 492/493 |
| Qwen3 8B | 8K | 29.0 | 32.3 | 1.11× | 0.0055 | −0.012 | 97.3% | 400/400 |
| Qwen3 8B | 16K | 23.5 | 31.2 | 1.33× | 0.0109 | +0.016 | 95.5% | 320/320 |
| Qwen3 8B | 30K | 17.8 | 29.0 | 1.63× | 0.0113 | +0.007 | 95.1% | 304/304 |
| Qwen3 14B | 8K | 17.5 | 19.1 | 1.09× | 0.0022 | −0.007 | 99.0% | 422/422 |
| Qwen3 14B | 16K | 14.9 | 18.7 | 1.25× | 0.0086 | +0.014 | 93.8% | 359/361 |
| Mistral 7B | 8K | 31.6 | 34.9 | 1.10× | 0.0152 | −0.005 | 97.5% | 358/359 |
| Mistral 7B | 16K | 25.4 | 33.9 | 1.33× | 0.0131 | +0.009 | 95.3% | 372/372 |
| Mistral 7B | 30K | 19.7 | 31.5 | 1.60× | 0.0403 | +0.030 | 92.4% | 295/296 |

Notes on the table:
- **Speeds at 16K** (Qwen3 0.6B and Mistral 7B) are from a rerun. The full run's 16K lines were about 1 ms
  per token slow for dense and sparse alike (66.4 / 112.9 and 24.5 / 32.1 tok/s).
- **Qwen3 0.6B** figures are for two regions of the book (start / +60K tokens). The "Decisive"
  column sums both.
- **Qwen3 14B** at 30K was not tried: weights, KV cache and index come to about 14 GB, more than a
  16 GB Mac leaves for the GPU.
  Its text NLL on the book is very low (0.94 nats at 8K), so it has probably memorised the book.
  That flatters its agreement.
- **Mistral 7B's KL at 8K** is one outlier position (5.4 nats; the median is 0.0000).

**Shared against per-head selection at 30K on the same prompt.** Qwen3 0.6B is in the "Second pass"
table above.

| Model | Selection | Sparse tok/s | Mean KL | Text NLL Δ | Top-1 |
|---|---|---|---|---|---|
| Qwen3 8B | shared | 29.1 | 0.0148 | +0.014 | 93.2% |
| Qwen3 8B | per head | 29.0 | 0.0113 | +0.007 | 95.1% |
| Mistral 7B | shared | 30.6 | 0.0171 | +0.013 | 95.7% |
| Mistral 7B | per head | 31.5 | 0.0403 | +0.030 | 92.4% |

**On Mistral 7B, per-head selection is worse.** At 30K its KL is 2.4× the shared selection's and its
text NLL cost is 2.3×, and the gap holds at the median too (0.0171 against 0.0091), not just in
outliers. Qwen3 0.6B (2 query heads per KV head) and Qwen3 8B (4) improve with per-head selection.
Mistral (4, but no QK-norm) does not.

A plausible explanation, not checked: without QK-norm, one KV head's mean-pooled score is a noisy
proxy, and summing over every head averages that noise out. Not measured yet:
- Mistral, shared against per-head, at 8K and 16K;
- Qwen3 14B, shared against per-head.

The panel's follow-ups for item 5 are the natural next step: a per-head score blended with the
shared one, or max-normalised aggregation within the group. Until then per-head stays the default,
and `MLXR_SPARSE_SHARED=1` is better on Mistral.

**Dense verification pass**, GPU ms per pass for T = 1 / 2 / 5 / 8 tokens:

| Model | Context | T = 1 | T = 2 | T = 5 | T = 8 |
|---|---|---|---|---|---|
| Qwen3 0.6B | 30K | 23.2 | 31.3 | 71.1 | 105.2 |
| Qwen3 8B | 30K | 56.1 | 96.6 | 202.0 | 273.3 |
| Qwen3 14B | 16K | 67.3 | 110.5 | 229.5 | 277.9 |
| Mistral 7B | 30K | 50.9 | 84.8 | 182.4 | 245.9 |

T = 5 costs 3.1–3.6× T = 1 everywhere.

## Measured (first pass: one selection shared by every head)

M2 Pro (16-core GPU, 16 GB), 4-bit models, nothing else on the GPU. The prompt is *Pride and Prejudice*
(Project Gutenberg #1342, about 177K tokens, so no context repeats). Default budget: 2,048 long-range
keys, a 512-key window and 4 sink keys. Each figure is the better of 2 runs. The runs agreed within 2%, except two dense lines that differed by 4–5%.
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
| Mistral 7B | 30,000 | 19.7 | 31.6 | 1.60× | 9/64 | 59/64 (92%) |

GPU time per token, dense → sparse: Qwen3 0.6B 8.85 → 7.00, 14.09 → 7.74 and 22.83 → 9.54 ms;
Mistral 7B 31.61 → 28.58, 38.87 → 29.46 and 50.83 → 31.63 ms. For reference, dense decode at 16
tokens of context is 3.33 ms (Qwen3 0.6B) and 25.85 ms (Mistral 7B). Prefill, dense in both modes:
Qwen3 0.6B 1,076 / 630 / 370 tok/s and Mistral 7B 185 / 145 / 107 tok/s at 8K / 16K / 30K.

Against the bar in step 6:
- Sparse is 1.60× dense at 30K on Mistral 7B, above the 1.3× bar.
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

At this budget Mistral 7B meets the speed and agreement bars at 30K: 1.52× and 61 of 64. The 8K bar
is met trivially, because the default minimum context becomes 10,248 and 8K decodes dense.

Qwen3 0.6B at 16K barely responds to the budget, even with `MLXR_SPARSE_MIN=8192`:
- 2,048 keys: 59 of 64;
- 4,096 keys: 59 of 64;
- 8,192 keys: 59 of 64;
- 12,288 keys (78% of the context): 61 of 64, which is also slower than dense (66.6 tok/s against
  70.9).

These positions flip under almost any change to the attention, so a larger budget does not fix them.
The cause is not identified yet. Both models also agree least at 16K, not at 30K, which more likely
reflects that region of the book than the context length.

Agreement over 64 positions of one text is a coarse signal. A larger evaluation (several texts, more
positions, needle retrieval) should come before any decision to ship.

Qwen3 8B, default budget, one run. Dense is the in-process line; it was within 0.5% of a separate
process on the other models.

| Context | Dense tok/s | Sparse tok/s | Speed-up | Greedy prefix | Teacher-forced |
|---|---|---|---|---|---|
| 8,192 | 29.1 | 32.5 | 1.12× | 18/64 | 62/64 (97%) |
| 16,384 | 23.2 | 31.4 | 1.35× | 14/64 | 57/64 (89%) |
| 30,000 | 17.8 | 29.2 | 1.64× | 6/64 | 58/64 (91%) |

**Fixed cost.** With almost nothing attended (`MLXR_SPARSE_ATTN=4 MLXR_SPARSE_WINDOW=4 MLXR_SPARSE_MIN=16`),
Qwen3 0.6B takes:
- 4.82 ms at 16 tokens, against 3.33 ms dense: score, select and two extra dispatches cost about
  53 µs per layer before any index bytes;
- 4.89 ms at 8K;
- 7.74 ms at 30K.

At 30K the step therefore splits into:
- 3.33 ms of weights;
- about 1.5 ms of fixed per-layer work;
- about 2.9 ms that grows with the number of blocks (the index read and the selection's
  per-block work);
- about 1.8 ms of attention over the 2,564 kept keys.

## Review panel (2026-11-06): ranked next steps

Three independent reviews: GPU cost, selection accuracy and the literature. They only read the code;
the measurements above are the inputs. Qwen first, then Mistral 7B. Expected gains are their cost
model's estimates, not measured.

**Measure first**
1. A stronger agreement bench in `benchLongContext`:
   - a dense teacher-forced control;
   - KL(dense‖sparse), and the change in NLL on the book's real next tokens;
   - argmax agreement bucketed by dense's top-1/top-2 margin;
   - at least 512 positions;
   - several text offsets at the same length (16K and 30K currently read different parts of the
     book);
   - a needle test.

   The likely explanation for the misses that ignore the budget is near-ties, which argmax counts as
   flips.

   *Done.* All of it except the needles being at one depth per door. Near-ties explain the 8K
   misses but not the 30K ones (see "Second pass").
2. The cost of a verification pass at 30K (T = 5 against T = 1). Verification runs T separate
   `attention_split` passes (Transformer.swift:1061), so it may read the KV cache T times. This
   decides whether the lossless route (item 8) can pay.

   *Measured.* T = 5 costs 3.1× T = 1 at 16K and 30K, so the KV cache is read about once per token.

**Exact, or lossless with respect to dense**
3. **Qwen3 14B has no split or sparse kernel.** For head size 128, `maxG` is 4 (Transformer.swift:667),
   and Qwen3 14B has 5 query heads per KV head (32B has 8). So 14B decodes long contexts with the
   single-threadgroup `attention` kernel, and `MLXR_SPARSE_ATTN` does nothing on it.
   - Fix: add `INST_ASPLIT(4, 5)`, `INST_ASPLIT(4, 8)` and the matching `INST_ASPARSE` lines, and raise
     `maxG` for E = 4.
   - This is on the default path. It should be bit-identical; this is not measured yet.

   *Done.* It is bit-identical; see "Second pass" for the speed.
4. **`sparse_select` cost: −0.9 ms at 8K, −1.6 ms at 30K on 0.6B.** Two changes:
   - Aggregate the round-1 histogram per simdgroup with ballots. Scores share their top byte, so
     about N threadgroup atomics hit a few bins today.
   - After 16 bits are decided, compact the candidates instead of re-scanning `scores` twice more.

   The output is identical to the current sparse mode.

   *Rejected.* The selection is only 15–33 µs per layer, and aggregating its histogram made it
   slower. Compacting the candidates was not tried: the most it could save is about 0.8 ms at 30K.

**Changes the current sparse output (re-measure agreement)**
5. **Per-KV-head lists.** Score and select per KV head instead of one list summed over all 8 heads.
   - Attention bytes stay the same (each head reads only its own rows).
   - The select runs as 8 threadgroups instead of 1.
   - It removes one head's veto over another's keys.
   - Follow-ups: max-normalised aggregation within a group of query heads, and SparQ-style
     compensation for the attention mass that was not selected.

   *Done; now the default.* The select runs as 8 threadgroups but takes no less time. On Qwen3 0.6B
   at 30K, KL fell 3–4× and the text NLL cost fell from +0.05 to +0.012–0.016 nats/token. The
   follow-ups are not done. On Mistral 7B at 30K, per-head selection was worse than shared (KL
   0.040 against 0.017); see the re-bench.
6. **Reuse a selection.**
   - Across 2–4 decode steps: `attention_sparse` also covers positions after the last selection.
     Estimated −1.6 to −3.5 ms on 0.6B.
   - Or across adjacent layers, as in TidalDecode and OmniKV.
7. **A smaller index.** int8 with a per-row scale, or pool 8: about −1.1 ms at 30K on 0.6B. Later, a
   channel subset ranked by `q_norm·k_norm` (the Double Sparsity approach) or Quest min/max bounds for
   recall of single "needle" keys.

**Larger bets**
8. **Sparse drafts, dense verification** (MagicDec, TriForce).
   - Greedy output stays identical to dense by construction, so this is the only route by which this
     work could ever become default behaviour.
   - Teacher-forced agreement is the acceptance rate.
   - It needs a multi-query verification attention kernel that reads K/V once for all T tokens.
     Estimate with that kernel: about 1.4× dense at 30K on Qwen3 0.6B, exact. 7–8B models lose until
     the batched matvec is memory-bound.
   - Separately, loading draft models with sparse attention (Engine.swift:268) is a one-line change
     and lossless.
9. **An int8 or int4 KV cache** (KIVI, mlx-lm's `QuantizedKVCache`). Opt-in.
   - It cuts every long-context read.
   - Estimate: 0.6B at 30K, 43.8 → about 114 tok/s with int4.
   - It lets Qwen3 14B hold a 30K context in 16 GB, and it composes with sparsity.
   - About 400 lines, touching every attention kernel and prefill.
10. **Prefill.**
    - Lossless: `flash_prefill` runs at about 1.5 of the 3.5 TFLOPS the GEMM tile reaches.
    - Opt-in sparse prefill (XAttention): attention is about 85% of Qwen3 0.6B's prefill at 30K.

Also noted:
- Speculative decoding with a sparse model verifies densely, so it is slower than plain sparse decode
  at long context.
- `minContext` should scale with model size: at 8K a 7–8B model gains about 10% while changing greedy
  output.

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
| Index query: sum over heads | Per KV head by default: the sum over that head's query heads of q · k̄, one list per KV head (`MLXR_SPARSE_SHARED=1`: the paper's single list) |
| Prefill sparse too | Prefill and speculative verification stay dense |
| Distillation (D-LSA) / native training (N-LSA) | None: works on any checkpoint |

Per decode step and eligible layer (full-context cache with a split-attention kernel for its head size):

1. `sparse_index`: whenever positions are written (prefill and decode), the blocks they complete get
   their index key: the mean of the block's 4 keys, per KV head, f16. A rewritten position (a new
   turn, a rejected draft) rebuilds its block when the block's last position is written again.
2. `sparse_score`: the decode query scores blocks `[sinkEnd, winStart)`, per KV head, with the sum
   over that head's query heads of `q_h · k̄_block` (shared: the sum over every head).
3. `sparse_select`: one threadgroup per KV head (shared: one) takes the exact top-p blocks (radix
   select on the score bits; ties go to the earlier block) and writes the attended positions in
   ascending order: sink, kept blocks, window.
4. `attention_sparse` + `attention_combine`: `attention_split` with key n of the KV head's list in
   place of position n. With the list `0..<L` it is bit-identical to `attention_split`.

Dense decode and every default code path are unchanged. Sparse mode is off unless enabled.

## Using it

```bash
MLXR_SPARSE_ATTN=1 mlxr run <model> ...            # default budget: 2048 long-range keys
MLXR_SPARSE_ATTN=1024 MLXR_SPARSE_WINDOW=256 ...   # long-range keys / window
MLXR_SPARSE_MIN=8192 ...                           # dense below this context (default 2·(sink+topK+window) = 5128)
MLXR_SPARSE_SHARED=1 ...                           # one selection for every head (default: one per KV head)
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

The long-context section prints:
- the prefill time;
- `dense @N` and `sparse @N` decode rates;
- how many greedy tokens match dense decode;
- teacher-forced argmax agreement on dense decode's tokens, with a dense control;
- the cost of a dense verification pass;
- with `MLXR_BENCH_TEXT`: KL, text NLL and margin-bucketed agreement over `MLXR_BENCH_POSITIONS`
  (default 512) positions of the text, and passkey needles.

`MLXR_BENCH_TEXT_OFFSETS=0,60000` repeats the measurements from other places in the text. Use a context
below the model's maximum minus the positions (Mistral 7B: 32,768). Without `MLXR_BENCH_TEXT` the
prompt is pseudo-random tokens: fine for speed, meaningless for agreement. Per-kernel GPU time:
`MLXR_KERNEL_TIMING=1 swift test --filter SparseTimingTests`.

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
- `Tests/MLXEngineTests/SparseTimingTests.swift`: GPU time per sparse kernel and per layer step
  (`MLXR_KERNEL_TIMING=1`; `MLXR_SPARSE_SHARED=1` times the shared selection).
