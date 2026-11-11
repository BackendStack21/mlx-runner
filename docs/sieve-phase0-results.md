# Sieve phase 0: results (steps 0 to 4)

Measured on 2026-11-11 for [sieve-handoff.md](sieve-handoff.md). Steps 5 to 7 wait for the owner's
decisions and are listed at the end as not tested. Every number here is a measurement on the machine
below unless it is called a projection.

**Machine.** Apple M2 Pro, 10 CPU cores, 16-core GPU, 16 GB, macOS 27.2 (26B123), Swift 6.4.2. The GPU
may use 12.7 GB (`recommendedMaxWorkingSetSize`). Nothing else used the GPU during a measurement.

> **Update, 2026-11-11.** The uncertified row screen of step 4 has since been built as the opt-in
> `--sieve balanced`: about +8% decode on four models, measured. See [sieve-head.md](sieve-head.md).
> The projections below assumed the head at 27% of its bytes at the dense byte rate; the built
> screen reads 19–32% of them at a lower rate, so its measured gain is below the "with Head"
> columns. The other hypotheses were then measured on the models under 1B
> ([steps 5 and 6](#steps-5-and-6-h2-h3-h4-and-h5-on-the-models-under-1b)): Experts and Pyramid
> failed, Stride passed its gate with little projected gain. The `fast` and `max` projections further
> down assumed operating points that those measurements rule out on these models.

## In short

| # | Hypothesis | Gate | Measured | Verdict |
|---|---|---|---|---|
| 0 | The cost model | none | Its constants are replaced (below). The per-layer cost outside the weight reads is 56 to 101 µs, not 46; the fixed cost per step is about 0.05 ms, not 0.25; a readback between two command buffers costs 0.18 ms, not 0.05; a 2-token pass costs 1.40 steps, not 1.5. | done |
| H1 | Head's certificate is tight | median certified rows ≤ 2% of V and bytes ≤ 40% of the dense head, for some r ≤ d/4 | On four models and every rank and sketch width, the certificate keeps **100% of the vocabulary**. Its bound is ±24 to ±56 logits around a screen logit; the top two logits are 1.3 to 1.8 apart. Sound: no violation in 4,960 held-out states per model. | **failed**, by a factor of about 25 in the bound |
| H1b | The uncertified screen (the fallback) | none set | A rank-d/4, 4-bit sketch (26% of the head's bytes) ranks the dense argmax first in 56 to 67% of states and within its top 700 to 1,800 rows in 99.9%. By blocks of 64 rows it is much worse. | usable only as a Bounded axis, and only by row |
| H3 | The multi-token head drafts well | acceptance ≥ 0.6 | **No installed export holds the head's weights** (0 `mtp.` tensors in the mlx-community 4-bit exports of Qwen3.5 0.8B, 2B, 4B and 9B). The original `Qwen/Qwen3.5-0.8B` holds 15 (41 MB in bf16). Acceptance was not measured. | not tested: needs decision 4, and an export that keeps the tensors |
| H2, H4, H5, H6 | Experts, Pyramid, composition | | | not tested: need decisions 2 and 4 |

**What this does to the plan.** The `exact` profile has nothing in it on the models as they are
installed: Head is not exact, and Stride has no weights to load. Phase 1 as the specification
describes it (a lossless gain of 8% for every user) does not exist. Every remaining axis changes
output, so decision 2 now decides whether there is a project at all, not only whether steps 6 and 7
run.

## Step 0. Build, test, baseline

- `swift build -c release`: passes. `swift test`: 237 tests in 38 suites pass before any change, and
  239 in 40 with this work (the two Sieve suites are skipped unless their variable is set; the 14B and
  Gemma 4 12B cases need `MLXR_TEST_LARGE`). The documentation test passes on all three Sieve files.
- Models: `mlx-community/Qwen3-0.6B-4bit`, `Llama-3.2-1B-Instruct-4bit`, `Qwen3.5-0.8B-4bit`,
  `Qwen3.5-4B-4bit` and, since step 2 showed that it fits, `Qwen3.5-9B-4bit`.
- `mlxr bench` three times per model, the five models in turn in each round. Medians of the three
  (the spread is under 1% except where noted):

| Model | Text weights | Decode @16 | Decode @1K | Prefill 512 | Decode step (GPU) | README |
|---|---|---|---|---|---|---|
| Qwen3 0.6B | 335 MB | 288 tok/s | 235 | 3,025 | 3.47 ms | 286 / 3,020 |
| Qwen3.5 0.8B | 424 MB | 253 ¹ | 245 | 2,779 | 3.94 ms | 271 / 2,780 |
| Llama 3.2 1B | 695 MB | 203 | 185 | 1,724 | 4.93 ms | 202 / 1,720 |
| Qwen3.5 4B | 2.37 GB | 61.8 | 60.5 | 454 ² | 16.18 ms | 62.3 / — |
| Qwen3.5 9B | 5.04 GB | 36.6 | 36.1 | 244 | 27.30 ms | — |

¹ 219, 253 and 254 in the three rounds; the first was disturbed. The README's 271 was not reached
today. ² 425, 454 and 454.

Greedy `mlxr run … "Explain RoPE" --temp 0 --repeat-penalty 1 --seed 42 --max 256`: 284, 255, 199,
62.0 and 36.8 tok/s in the same order (one run each).

## Step 1. The cost model

### Where a decode step goes

From the profile lines of `mlxr bench` (step at 16 tokens of context; medians of three; "without X" is
the step's time less the step with that kernel class skipped, so the parts do not add up exactly):

| Model | Step | Head | Matvecs of the layers | Attention, mixers, norms, activation | Head's rate |
|---|---|---|---|---|---|
| Qwen3 0.6B | 3.37 ms | 0.50 | 2.10 | 0.45 | 171 GB/s |
| Qwen3.5 0.8B ⁴ | 3.87 ms | 0.76 | 2.48 | 0.74 (0.21 is the mixers' state) | 183 |
| Llama 3.2 1B | 4.84 ms | 0.78 | 3.58 | 0.23 | 186 |
| Qwen3.5 4B | 15.97 ms | 1.84 | 12.45 | 1.11 (0.83 is the mixers' state) | 185 |
| Qwen3.5 9B | 27.17 ms | 3.03 | 22.64 | 1.37 (0.83 is the mixers' state) | 188 |

⁴ This model's profile lines varied most between rounds (its head reads 0.50, 0.76 and 0.99 ms); the
isolated `lm_head matvec` line is steady at 0.78 ms.

**What c1 is made of.** Each matvec dispatch costs a fixed time plus its bytes. Time per dispatch
against bytes per dispatch, across the five models (the profile's sweeps over every layer's weights):

| Matvec | Fixed | Marginal rate |
|---|---|---|
| gate and up (one fused dispatch), 3.5 to 57 MB | 16 µs | 191 GB/s |
| down, 1.8 to 28 MB | 11 µs | 175 GB/s |
| o, 1.2 to 9.4 MB | 10 µs | 187 GB/s |

So the bytes of the body are read at about 183 GB/s (93% of the 197 GB/s of a plain read; the
specification's η was 0.91), and each layer pays about 47 µs for its four matvec dispatches whatever
their size. That, not the small kernels, is most of the specification's 46 µs. The small kernels and
dispatch gaps add 14 to 31 µs per layer, and a linear-attention layer of the 4B and 9B reads and
writes 4 MB of recurrent state (35 µs). A dependent dispatch on its own costs 1.5 to 1.8 µs: dispatch
latency is not where the time goes.

With the body at 183 GB/s, the head at its measured rate and 0.05 ms fixed per step (the sampler
measures 0.03), what is left of a measured step, per layer: **56 µs (Qwen3 0.6B), 66 (Qwen3.5 0.8B),
68 (Llama 3.2 1B), 101 (Qwen3.5 4B), 92 (Qwen3.5 9B)**. The specification's single c0 = 0.25 ms and
c1 = 46 µs are replaced by these.

Consequence for the design: Experts shrinks the bytes of two of a layer's four matvecs and none of
their fixed cost, and adds dispatches of its own (router, down by channel, combine). On Qwen3 0.6B a
layer's MLP matvecs take 55 µs, of which 27 µs are fixed.

### A multi-token pass

`verify T=2` against `verify T=1`, GPU time (not measured on hybrid models, which cannot rerun a
position): Qwen3 0.6B 4.89 / 3.51 ms = **1.39**; Llama 3.2 1B 6.86 / 4.91 = **1.40**. The
specification used 1.5. A 4-token pass costs 2.6 to 2.8 steps on all five models (`verify step T=4`).

### The GEMM rate

`layer GEMMs at T=512`: 3.46 (Qwen3 0.6B), 3.48 (Qwen3.5 0.8B), 3.57 (Llama 3.2 1B), 3.53 (Qwen3.5 4B)
and 3.55 TFLOPS (Qwen3.5 9B). The specification used 3.3.

The projection GEMMs are 77% of a 512-token prefill on Qwen3 0.6B, 78% on Qwen3.5 0.8B, 94% on Llama
3.2 1B, 92% on the 4B and 95% on the 9B (the specification assumed 85% everywhere). On the two
smallest models attention or the mixers take 16 to 17%.

### A readback between two command buffers

`SieveTimingTests` (opt-in): the MLP matvecs of every layer at T = 1, as one command buffer, and
split in the middle with the first half committed and waited for, 4 bytes read, and the second half
committed. 300 alternating rounds, wall-clock medians:

| Model | One buffer | Split with a readback | Cost of the readback |
|---|---|---|---|
| Qwen3 0.6B | 1.566 ms | 1.746 ms | **0.180 ms** |
| Qwen3.5 0.8B | 1.484 | 1.665 | **0.181** |
| Llama 3.2 1B | 3.147 | 3.333 | **0.187** |
| Qwen3.5 4B | 8.236 | 8.488 | **0.252** |

- Encoding the second half before the wait does not help (0.180 to 0.226 ms): the cost is the wait
  and the wake-up, not the encoding.
- Splitting without a readback (both committed at once) costs nothing (−0.04 to −0.01 ms), and GPU
  time is unchanged in every variant.
- The specification assumed 0.05 ms. At 0.18 ms, the two exit tests of a token that reaches the top
  cost 0.36 ms: 10% of a Qwen3 0.6B step.
- This is on top of a step that already waits. Generation does not: it queues the next step before
  reading the current one. A step that must wait before the next is encoded also loses that overlap;
  `verify T=1` (which waits) takes 0.38 to 0.48 ms more wall-clock time than GPU time on the two
  models that print it.

## Step 2. The real model shapes

`swift scripts/sieve-shapes.swift <model directory>` reads `config.json` and the safetensors headers.
Bytes of the text model (the Qwen 3.5 files also hold a vision tower of 201 to 912 MB that the engine
does not load):

| Model | d | MLP | Layers (linear + full) | V | Head | Mixers and attention | MLP | MLP share of the body |
|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | 1,024 | 3,072 | 28 | 151,936 | 87.5 MB | 99.1 MB | 148.6 MB | 60% |
| Qwen3.5 0.8B | 1,024 | 3,584 | 24 (18 + 6) | 248,320 | 143.0 | 132.2 | 148.6 | 53% |
| Llama 3.2 1B | 2,048 | 8,192 | 16 | 128,256 | 147.8 | 94.4 | 453.0 | 83% |
| Qwen3.5 2B ³ | 2,048 | 6,144 | 24 (18 + 6) | 248,320 | 286.1 | 263.5 | 509.6 | 66% |
| Qwen3.5 4B | 2,560 | 9,216 | 32 (24 + 8) | 248,320 | 357.6 | 735.2 | 1,274.0 | 63% |
| Qwen3.5 9B | 4,096 | 12,288 | 32 (24 + 8) | 248,320 | 572.1 | 1,175.3 | 2,717.9 | 70% |

³ From the header of the export on huggingface.co; the model is not installed.

Against the specification's assumptions:

- **Qwen3.5 0.8B's body is 47% mixers and attention, 53% MLP**, not 35 / 65. Experts has less to cut
  there. The 4B (37 / 63) is as assumed.
- **The 4B's text model is 2.37 GB**, not 3.03 GB: the specification took the file size, which
  includes 667 MB of vision tower. Its body is 2.01 GB, not 2.67.
- **The 9B has 32 layers, a hidden size of 4,096, a head of 572 MB and a body of 3.89 GB** (assumed:
  40 layers, 500 MB, 5.1 GB). Its head is not tied: the file holds a separate embedding of another
  572 MB, of which a step reads one row.
- Linear-attention layers: 16 key heads of 128; 16 value heads (0.8B, 2B) or 32 (4B, 9B). Full
  attention every fourth layer, head size 256, 2 or 4 KV heads.
- **What a frozen position still pays**: k and v are 14.3% of a full-attention layer's projections;
  the input projections are 80% of a linear layer's mixer (0.8B) or 75% (4B, 9B). Over the body's
  mixers and attention that is 68% for the 0.8B and 62% for the 4B and 9B (assumed: 70%).
- **Multi-token-prediction weights: none in any 4-bit export examined** (installed 0.8B, 4B, 9B; the
  2B's header). Every config declares `mtp_num_hidden_layers: 1`. The original `Qwen/Qwen3.5-0.8B`
  holds 15 `mtp.` tensors: `mtp.fc` [1024, 2048], one transformer layer with q/k norms, three norms.
  Stride on Qwen 3.5 therefore needs its own export, or a sidecar that carries these tensors.
- **The 9B fits this Mac**: 5.95 GB of files, 0.66 GB of interleaved quantization pairs, 0.13 GB of
  KV cache at 4,096 tokens and 0.05 GB of recurrent state, against 12.7 GB. It loads and runs
  (36.6 tok/s).

## Step 3. The tables, reproduced and corrected

`swift scripts/sieve-budget.swift --spec` reproduces the decode and prefill tables the specification
was first written with, within rounding (the largest differences are 2,970 against 2,980 and 320
against 315). Its arithmetic was right; its inputs were not.

`swift scripts/sieve-budget.swift --measured` reruns it with the constants of step 1, the shapes of
step 2, today's dense rates, and step 4's result for Head. **Projections, M2 Pro, 2026-11-11.** The
operating points (experts kept, exits, prefill depth) are still the specification's assumptions, and
so is the 0.8 efficiency of GEMM tiles over token subsets.

Decode, tok/s. "With Head" assumes the uncertified, output-changing screen at 27% of the head's bytes
and 0.04 ms of extra dispatches per evaluation; "dense head" leaves Head out.

| Model | Dense (measured) | `balanced` | `fast` | `max` | `fast`, dense head | `max`, dense head | First written as (`fast` / `max`) |
|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | 288 | 350 | 460 | 620 | 360 | 470 | 500 / 680 |
| Qwen3.5 0.8B | 253 | 320 | 410 | 550 | 290 | 380 | 620 / 830 |
| Llama 3.2 1B | 203 | 300 | 400 | 560 | 290 | 380 | 440 / 610 |
| Qwen3.5 4B | 61.8 | 95 | 130 | 180 | 97 | 130 | 180 / 260 |
| Qwen3.5 9B | 36.6 | 61 | 84 | 120 | 63 | 87 | 88 / 130 |

Prefill, tok/s:

| Model | Dense (measured) | `balanced` | `fast` | `max` | First written as (`fast` / `max`) |
|---|---|---|---|---|---|
| Qwen3 0.6B | 3,025 | 4,250 | 6,900 | 7,760 | 6,990 / 7,850 |
| Qwen3.5 0.8B | 2,779 | 3,530 | 4,930 | 5,310 | 5,670 / 6,180 |
| Llama 3.2 1B | 1,724 | 3,140 | 5,910 | 7,080 | 5,530 / 6,580 |
| Qwen3.5 4B | 454 | 650 | 940 | 1,030 | 820 / 900 |
| Qwen3.5 9B | 244 | 370 | 560 | 620 | 430 / 470 |

Where the decode projections fell, and why:

- Qwen3.5 0.8B loses the most (620 → 410 in `fast`): no Stride (−20%), a body with less MLP, a lower
  dense rate today, and exit tests at 0.18 ms.
- The exit tests now cost 0.25 to 0.32 ms per token (1.4 to 1.75 tests at 0.18 ms), four times the
  first estimate.
- `max` touches 104 MB (Qwen3 0.6B) to 1.19 GB (9B) per token: A0.2 to A2.1, as before.

## Step 4. H1: Head's certificate

`SieveHeadTests` (opt-in), per model: 50 chunks of 512 tokens, the first 16 positions of each dropped,
24,800 final hidden states read from the engine (`xn` after `forward`, which the code confirms holds
the final norm's output for every row when the pass computes logits). Half from prose (Wikipedia
articles on 2026 events, fetched today, which none of these models can have seen in full), half from
code (this repository's Swift sources). Every fifth chunk is held out: 19,840 states fitted, 4,960
held out. The head is dequantized on the CPU; its logits match the engine's to four decimals on every
chunk's last position. The fit is in double precision (LAPACK's symmetric eigensolver).

### The certificate

Certified rows for k = 1, 8-bit sketch, median over the held-out states. k = 2 and k = 64, the 4-bit
and the unquantized sketch all give the same: the whole vocabulary.

| Model | r | Variance in the basis | Mean ρ | Mean unseen norm n | Bound on the argmax row (median) | Top-1 minus top-2 (median) | Certified rows | Bytes against the dense head |
|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | d/16 | 71.7% | 52.8 | 0.894 | ±39.2 | 1.54 | 100% | 113% |
| | d/8 | 78.5% | 46.8 | 0.863 | ±32.6 | | 100% | 125% |
| | d/4 | 85.8% | 39.4 | 0.802 | ±24.6 | | 100% (mean 99.6%) | 149% |
| Qwen3.5 0.8B | d/16 | 59.6% | 77.3 | 0.608 | ±47.5 | 1.39 | 100% | 113% |
| | d/8 | 70.0% | 68.1 | 0.589 | ±39.7 | | 100% | 125% |
| | d/4 | 81.2% | 55.9 | 0.552 | ±30.0 | | 100% | 149% |
| Llama 3.2 1B | d/16 | 61.2% | 57.4 | 0.891 | ±44.3 | 1.30 | 100% | 113% |
| | d/8 | 71.1% | 51.2 | 0.860 | ±36.9 | | 100% | 124% |
| | d/4 | 81.8% | 42.7 | 0.801 | ±27.9 | | 100% | 148% |
| Qwen3.5 4B | d/16 | 63.4% | 90.8 | 0.634 | ±55.8 | 1.81 | 100% | 113% |
| | d/8 | 73.4% | 80.4 | 0.614 | ±46.6 | | 100% | 124% |
| | d/4 | 83.8% | 66.4 | 0.574 | ±34.9 | | 100% | 148% |

- **Gate: failed on all four models, at every rank and sketch width.** The median of certified rows
  is 100% of V against a gate of 2%, and the expected bytes are 113 to 149% of the dense head against
  a gate of 40% (the screen is paid and saves nothing).
- **By how much.** The bound has to be narrower than the gap between the leading logits, about 1.4.
  At r = d/4 it is ±25 to ±35: 20 to 25 times too wide. The quantization term is small (0.13 to 0.28
  logits at 8 bits, 2.4 to 4.0 at 4 bits); the unseen term n · ρ is the whole of it.
- **Why.** The final hidden states are not low-rank: a quarter of the dimensions hold 81 to 86% of
  their variance, so ρ stays at 40 to 66 while a head row's unseen part keeps 87 to 96% of its norm.
  Cauchy–Schwarz assumes the two unseen parts are aligned; in 770 to 1,920 dimensions they almost
  never are, which is why the screen itself is far better than its bound. No rank the screen could
  afford closes a factor of 25: the bound falls by about a fifth for each doubling of r.
- **Soundness: holds.** No dense top-k row outside the certified set, for k = 1, 2 and 64, in any of
  the 36 configurations. With everything certified this says little; the mathematics is not in doubt.
- Prose and code behave the same.

### The uncertified variant

The handoff's fallback: evaluate the blocks that the screen ranks highest and take the best row. How
far down the screen's order is the dense argmax? (4,960 held-out states per model, so "99.99%" is the
worst state seen.) 8-bit sketch; 4 bits changes these by a few percent.

| Model | r | Screen bytes against the head (8-bit / 4-bit) | Screen's argmax is the dense argmax | Blocks of 64 for 99% / 99.9% / worst | Of | Rows for 99% / 99.9% / worst |
|---|---|---|---|---|---|---|
| Qwen3 0.6B | d/8 | 25% / 14% | 45% | 282 / 760 / 1,041 | 2,374 | 867 / 3,057 / 8,813 |
| | d/4 | 49% / 26% | 57% | 88 / 283 / 603 | | 166 / 684 / 3,556 |
| Qwen3.5 0.8B | d/8 | 25% / 14% | 45% | 269 / 1,105 / 1,949 | 3,880 | 889 / 9,269 / 60,399 |
| | d/4 | 49% / 26% | 59% | 81 / 381 / 1,387 | | 161 / 1,672 / 7,926 |
| Llama 3.2 1B | d/8 | 24% / 13% | 50% | 247 / 725 / 1,261 | 2,004 | 577 / 4,051 / 23,423 |
| | d/4 | 48% / 26% | 63% | 82 / 328 / 614 | | 141 / 764 / 2,955 |
| Qwen3.5 4B | d/8 | 24% / 13% | 54% | 139 / 663 / 1,014 | 3,880 | 316 / 2,506 / 9,052 |
| | d/4 | 48% / 26% | 68% | 45 / 232 / 1,132 | | 70 / 789 / 6,107 |

- **By blocks, as specified, it does not pay.** Holding the dense argmax in the worst held-out state
  takes 25 to 50% of all blocks on top of the screen: 70 to 100% of the dense head. Tokens that
  compete for a position are not neighbours in the vocabulary, so a block of 64 consecutive rows
  holds one candidate and 63 passengers.
- **By rows it might.** A 4-bit sketch of rank d/4 is 26% of the head. The 700 to 1,800 rows that
  hold the dense argmax in 99.9% of states are another 0.5 to 0.7%; the worst state seen needs 3,000
  to 8,000 rows (2 to 3%). That is a head at 27 to 29% of its bytes that picks the dense argmax in at
  least 99.9% of states. It needs a gather of single rows (a kernel that does not exist), it is
  **Bounded, not Exact**, and what it does to text NLL and to sampling (where the top 40 matter, not
  the top 1) is not measured.
- A clustered row order (specification 14.2, question 6) would attack the block result. Not tried.

## The forecast for the three targets, on this Mac

Projections from the measured constants; the operating points are still assumptions that H2, H4 and
H5 would have to confirm.

- **T1, 500 tok/s decode at 1B.** Not in `fast` any more: 410 (Qwen3.5 0.8B), 400 (Llama 3.2 1B),
  460 (Qwen3 0.6B). Only `max` passes (550 to 620), and only with the uncertified Head: with the
  dense head `max` gives 380 to 470. T1 now needs every Bounded axis at its furthest assumed point.
- **T2, 5,000 tok/s prefill at 1B.** Still in range in `fast` for Llama 3.2 1B (5,910) and Qwen3 0.6B
  (6,900); Qwen3.5 0.8B falls just short (4,930) because half its body is mixers that a frozen
  position mostly still pays. Head does not enter prefill.
- **T3.** Unchanged in kind and further in degree: the 4B in `max` projects to 180 tok/s (was 260)
  and the 9B to 120 (was 130; its dense rate is higher than assumed, its gains lower). Three `max`
  answers from the 4B would take about as long as one dense answer, not 70% of it.

## Steps 5 and 6. H2, H3, H4 and H5 on the models under 1B

Measured on 2026-11-11 on the same Mac, after the owner asked for the remaining axes, on models
under 1B only: Qwen3 0.6B (28 layers) and Qwen3.5 0.8B (24 layers). All in Swift in this engine, with
the experiment hooks of `SieveProbe` (tests only). Text: this repository's own prose (README,
CHANGELOG, docs) and code (the engine's Swift sources), 512-token chunks, the first 16 positions of
each not scored. Quality only; nothing here is a speed measurement unless it says so.

| Hypothesis | Gate | Result |
|---|---|---|
| H2, Experts | NLL ≤ +0.02 at ≤ 50% of the MLP kept | **Failed.** A perfect selector needs 55–67% of single neurons, and 96–99% of 64-neuron groups. |
| H4, Pyramid in decode | mean depth ≤ 0.8 at δ = 1% | **Failed.** Mean depth 0.98–0.99. |
| H5, Pyramid in prefill | NLL ≤ +0.02 at depth ≤ 0.7 | **Failed.** Needs depth 0.79 on both models. |
| H3, Stride | acceptance ≥ 0.6 | **Passed**: 0.68 on the model's own answers. Its cost side leaves a few percent at best (below). |

### H2: Experts (`SieveExpertsTests`)

After the activation, a neuron (or a group of 64 consecutive neurons, the only unit the down
projection's quantization lets a kernel skip without changing a weight) is kept when its score
|h_n| · ‖down column n‖ reaches a share τ of the row's best. This is an oracle: it reads the true
activations. 7,936 scored positions per domain; ΔNLL is prose / code.

| Model | Unit | Kept | ΔNLL | Top-1 agreement with dense |
|---|---|---|---|---|
| Qwen3 0.6B | neuron | 61.4% | 0.000 / +0.003 | 91.4% / 94.2% |
| | neuron | 54.9% | +0.008 / +0.016 | 88.6% / 92.1% |
| | neuron | 47.8% | +0.027 / +0.040 | 85.3% / 89.5% |
| | neuron | 32.7% | +0.150 / +0.159 | 73.5% / 79.7% |
| | group of 64 | 96.3% | −0.004 / +0.001 | 95.1% / 96.3% |
| | group of 64 | 90.7% | +0.025 / +0.026 | 87.6% / 90.3% |
| | group of 64 | 73.8% | +0.21 / +0.23 | 69.2% / 76.3% |
| Qwen3.5 0.8B | neuron | 66.9% | +0.013 / +0.018 | 92.7% / 94.0% |
| | neuron | 60.2% | +0.023 / +0.026 | 89.8% / 91.7% |
| | neuron | 53.1% | +0.043 / +0.052 | 86.4% / 88.8% |
| | group of 64 | 96.5% | +0.026 / +0.027 | 90.9% / 92.1% |
| | group of 64 | 79.6% | +0.18 / +0.22 | 75.2% / 79.3% |

- Cheap predictors of the oracle's top half (chance is 50% recall): the previous token's set 59%, a
  static set 60%, a rank-256 projection of the MLP input through the exact gate/up 71%, at 2.4 to 2.7
  times the oracle's output error.
- On Qwen3 0.6B the upper half of the layers alone tolerates 36% kept at no cost; Qwen3.5 0.8B shows
  no such difference.
- Not tested: clustered (non-consecutive) groups, which would need the down projection re-quantized;
  per-layer thresholds; the specification's ridge router.

### H4 and H5: Pyramid (`SievePyramidTests`)

A row that exits at layer E contributes nothing from there on; the layers above still compute their
K/V rows (or recurrent state) from its frozen state (the specification's frozen fill, with β = 0).

**H5.** The first 384 rows of each chunk exit at E, the last 128 run every layer and are scored
(4,096 tokens per domain).

| Depth E / L | Qwen3 0.6B ΔNLL prose / code | Qwen3.5 0.8B ΔNLL prose / code |
|---|---|---|
| 0.25 | +0.545 / +0.781 | +0.453 / +0.676 |
| 0.46 | +0.268 / +0.424 | +0.189 / +0.340 |
| 0.57–0.58 | +0.124 / +0.194 | +0.203 / +0.335 |
| 0.67–0.68 | +0.061 / +0.096 | +0.051 / +0.092 |
| 0.75 | +0.016 / +0.027 | +0.034 / +0.032 |
| 0.79 | +0.008 / +0.016 | +0.013 / +0.012 |
| 0.86–0.88 | +0.003 / +0.004 | +0.012 / +0.009 |

+0.02 needs 79% of the layers on both models, +0.05 needs 75%. Keeping the last 8 or 32 prompt tokens
dense moves that by at most one layer.

**H4.** Exits decided causally from the logit lens at candidate layers.

- The raw lens is unusable: at three quarters of the depth it agrees with the full model on 30–32% of
  tokens (Qwen3 0.6B) and 4–10% (Qwen3.5 0.8B).
- With a fitted affine lens (fitted on other chunks of the same text, so optimistic), the points
  inside the targets let 4–11% of tokens exit, all in the top quarter of the stack: mean depth
  0.98–0.99.

Not tested: the β-extrapolated fill, attention-scored keeping of prompt tokens, passkey recall,
contexts beyond 512 tokens, text outside this repository.

### H3: Stride (`SieveStrideTests`)

No mlx-community export keeps the multi-token head (mlx-lm's converter drops `mtp.*`). A third-party
export does: `fcobos/Qwen3.5-0.8B-oQ4-mtp` (661 MB, mixed 4/5/6-bit). The head is
`fc([norm(embed(x_t+1)) | norm(h_t)])`, one full-attention block with its own context, a norm and the
tied output projection (per vLLM and llama.cpp; run here on the CPU in the test, with the block
checked against the engine on all six full-attention layers of the backbone to 3.4e-4).

| Text | Drafts | Acceptance |
|---|---|---|
| The model's own greedy answers (8 prompts) | 1,797 | 0.679 (0.476 to 0.887 per prompt) |
| Repository prose, teacher-forced | 7,904 | 0.616 |
| Repository code, teacher-forced | 7,904 | 0.672 |

- h_t is the final norm's output: the pre-norm state scores 0.574 and 0.267.
- The head's own confidence is calibrated: drafting only at ≥ 0.5 covers 59% of tokens at 0.898.
- It needs its context: with a window of 8 positions acceptance falls to 0.573, with 1 to 0.444.
- The head's wiring has no reference check (no other inference stack was used).

**The cost side, measured** (`SieveTimingTests.twoTokenPassAgainstOneTokenStep`, body only):

| Model | One token | Two tokens | Ratio |
|---|---|---|---|
| Qwen3.5 0.8B, 4-bit (mlx-community) | 3.36 ms | 4.75 ms | 1.42 |
| Qwen3.5 0.8B, mixed 6-bit (the export with the head) | 3.98 ms | 16.23 ms | 4.07 |
| Qwen3 0.6B, 4-bit | 3.66 ms | 5.09 ms | 1.39 |

A Stride cycle is one two-token pass, three head evaluations (two rows and the draft) and the extra
block, for 1.68 tokens. **Projected, not measured**, for the 4-bit Qwen3.5 0.8B with the head's
weights added as a sidecar:

- with dense heads (the lossless form): about 6.6 ms for 1.68 tokens, 3.9 ms per token, the same as
  dense decode today. No gain.
- with Sieve Head on all three evaluations: about 5.7 ms, 3.4 ms per token, 295 tok/s against 275
  with Sieve Head alone (+7%). Drafting only when the head is confident projects to about +11%.
- less the CPU round trip each cycle needs to learn whether the draft was accepted (0.18 ms,
  step 1), and with the acceptance on a 4-bit backbone unmeasured.

On the only export that holds the head, a two-token pass costs four single steps (no multi-token
kernel exists for 6-bit weights), so Stride loses there as it stands.

### What this leaves

On models under 1B, on this Mac: Head is built and worth about 8%. Experts and Pyramid fail their
gates. Stride passes its gate and projects to a few percent on one model family, for the largest
engineering cost of the five axes. The targets T1 (500 tok/s) and T2 (5,000 tok/s prefill) are not
reachable by this design on these models. Larger models were not tested in this round.

## Not tested

| Step | Why |
|---|---|
| H2, H4, H5 on models of 1B and above | The owner asked for models under 1B first. |
| 7 (H6) | Needs two of H2, H4, H5 to pass. None did. |

## Reproducing

```bash
swift build -c release
mlxr bench mlx-community/Qwen3-0.6B-4bit
swift scripts/sieve-shapes.swift "$HOME/Library/Application Support/MLX Runner/models/mlx-community/Qwen3.5-4B-4bit"
swift scripts/sieve-budget.swift
MLXR_SIEVE_TIMING=mlx-community/Qwen3-0.6B-4bit swift test -c release -Xswiftc -enable-testing --filter SieveTiming
MLXR_SIEVE_H1=mlx-community/Qwen3-0.6B-4bit MLXR_SIEVE_TEXT=prose.txt,code.txt \
    swift test -c release -Xswiftc -enable-testing --filter SieveHead
MLXR_SIEVE_EXPERTS=mlx-community/Qwen3-0.6B-4bit swift test -c release -Xswiftc -enable-testing --filter SieveExperts
MLXR_SIEVE_PYRAMID=mlx-community/Qwen3-0.6B-4bit MLXR_SIEVE_PART=h5 MLXR_SIEVE_CHUNKS=32 \
    swift test -c release -Xswiftc -enable-testing --filter SievePyramid
MLXR_SIEVE_PYRAMID=mlx-community/Qwen3-0.6B-4bit MLXR_SIEVE_PART=h4 MLXR_SIEVE_CHUNKS=16 MLXR_SIEVE_LENS=fit \
    swift test -c release -Xswiftc -enable-testing --filter SievePyramid
MLXR_SIEVE_STRIDE=fcobos/Qwen3.5-0.8B-oQ4-mtp swift test -c release -Xswiftc -enable-testing --filter SieveStride
```

Every one of these tests is skipped unless its variable is set. H1 takes one to three minutes per model in a
release build, and needs about 6 GB for the 4B.
