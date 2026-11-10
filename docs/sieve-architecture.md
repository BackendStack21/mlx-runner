# Sieve: a touch-sparse model architecture for MLX (specification)

**Status: specification only. Nothing here is implemented, and nothing here was measured for this
document.** It was written on 2026-11-10 without access to a Mac. Every speed figure is either one of
this repository's own measurements (README, CHANGELOG, [sparse-decode-attention.md](sparse-decode-attention.md);
M2 Pro, 16 GB, 4-bit weights) or a projection from them, and is labelled as one or the other.
[Section 14](#14-sources-and-what-was-not-verified) lists what could not be verified.

Contents: [1 Verdict](#1-the-request-and-the-verdict) ·
[2 Terms](#2-terms-and-reference-setup) ·
[3 Cost model](#3-cost-model) ·
[4 What the targets require](#4-what-the-targets-require) ·
[5 Architecture](#5-architecture) ·
[6 Container](#6-container-format) ·
[7 Importer](#7-importer) ·
[8 Engine integration](#8-engine-integration) ·
[9 Correctness gates](#9-correctness-gates) ·
[10 Measuring](#10-measuring) ·
[11 Intelligence](#11-the-intelligence-target) ·
[12 Plan](#12-plan-and-gates) ·
[13 Risks](#13-risks-open-questions-and-what-was-ruled-out) ·
[14 Sources](#14-sources-and-what-was-not-verified)

## 1. The request and the verdict

The request: a new model architecture for MLX in which a model of about 1B parameters

- **T1** decodes at 500 tok/s or more,
- **T2** prefills at 5,000 tok/s or more,
- **T3** scores above a 4B Qwen model on intelligence benchmarks,

under two constraints:

- **C1** it is built for MLX (MLX weight files, this engine);
- **C2** it takes its weights from existing open models, with no retraining.

**The three targets cannot all be met under C2, on any Mac. T3 is the one that fails.** T1 and T2 are
set by the Mac's memory bandwidth and GPU arithmetic rate: on this repository's reference Mac (M2 Pro)
neither is within reach for a full 1B model, and on a Max-class Mac both are plausible.

| Target | M2 Pro (197 GB/s, 3.3 TFLOPS effective) | Max-class Mac (500 GB/s assumed) | What decides it |
|---|---|---|---|
| T1 decode ≥ 500 tok/s | **No for a 1.0–1.2B model**: 202 tok/s today, 310–350 projected with the output-changing mode. Within reach only for models of 0.8B or less, and only if every part lands at the optimistic end. | Projected 530–580 for a 1.2B model with the output-changing mode. | Bytes read per token ([3](#3-cost-model), [4.1](#41-decode-500-toks-is-a-byte-budget)) |
| T2 prefill ≥ 5,000 tok/s | **No.** Exact prefill needs 2 floating-point operations per body weight per token. 5,000 tok/s is possible up to a 0.33B-parameter body; a 1B model's body needs about three times this GPU's measured rate. | Yes if the GPU sustains about 10 TFLOPS in this engine's GEMM. Not measured. | GPU arithmetic rate ([4.2](#42-prefill-5000-toks-is-an-arithmetic-budget)) |
| T3 above Qwen 4B | **No.** | **No.** | No ~1B open model is close, and no training-free transform adds capability ([4.3](#43-intelligence-above-a-4b-qwen-is-not-reachable-without-training), [11](#11-the-intelligence-target)) |

What this specification does define is the architecture that gets closest, and the measurements that
decide each part early:

- **Sieve** runs an imported model unchanged in its dense limit and reads fewer weight bytes per
  token everywhere else. Three mechanisms, each of the form *screen cheaply, then compute exactly on
  what passes*:
  1. **Sieve-Head**: the output projection reads a small sketch of the vocabulary, proves which rows
     cannot be sampled, and computes the rest exactly. **Output is unchanged.** Projected +10–20%
     decode on 1B-class models (the head is 20–35% of their bytes).
  2. **Sieve-FFN**: each MLP computes only the channels a cheap proxy scores as salient. **Output
     changes**, like sparse decode attention, so it is opt-in and gated on measured divergence.
     Projected 1.15–1.5× on top, most where the MLP is the largest share of the model.
  3. **Stride**: two or more tokens per pass, drafted by the multi-token-prediction head that Qwen 3.5
     checkpoints ship, verified exactly. **Output is unchanged.** The gain depends on how often the
     draft is accepted and on the cost of a 2-token pass: 1.0–1.5× ([3.2](#32-multi-token-passes)).
- A model enters by **import**: the original MLX files plus a sidecar that an importer computes from
  them. No gradient step, no change to the original weights.

**Decisions this needs from the owner** (recommendations in brackets):

1. **Which Mac is the reference?** [Keep the M2 Pro as the measured baseline, and state T1/T2 per
   machine with the formulas in section 3. On an M2 Pro, commit to the projections in
   [4.4](#44-projections), not to 500 / 5,000.]
2. **What replaces T3?** [T3′ in [11.3](#113-recommended-replacement-t3): exact reproduction of the
   source model in the lossless profile, and a measured Pareto gain in the lossy ones. Keep T3 on
   record as unmet.]
3. **Is an output-changing, opt-in mode acceptable?** It is where most of the decode gain is.
   [Yes, with the same treatment as sparse decode attention: off means untouched, the lossless
   profile stays token-identical, and the divergence of the lossy ones is measured and published.]
4. **Which model is the flagship import?** [Qwen3.5-0.8B: the only ~1B-class candidate with a
   multi-token-prediction head, the largest head share, and the one nearest 500 tok/s on an M2 Pro.
   Llama 3.2 1B as the plain-transformer control.]

## 2. Terms and reference setup

- **1B-class model**: 0.8–1.25B parameters, 0.42–0.73 GB at MLX 4-bit. In this repository: Qwen3.5
  0.8B (424 MB), Llama 3.2 1B (695 MB), Gemma 3 1B (732 MB).
- **Body**: every weight except the embedding / output projection. **Head**: the output projection
  (`lm_head`, or the tied `embed_tokens`).
- **Touched bytes**: weight bytes a decode step reads. A dense step touches the whole model except
  the embedding table (one row) or, with tied embeddings, touches that table once as the head.
- **Reference Mac**: M2 Pro, 16-core GPU, 16 GB. Measured in this repository: 197 GB/s with a plain
  read kernel (README), about 3.5 TFLOPS on the 64×64 quantized GEMM tile (README, CHANGELOG 1.3.0).
- **Bytes per weight**: 0.5625 for MLX 4-bit with group size 64 and f16 scales and biases
  (4 bits + 32 bits per 64 weights). Model sizes computed this way match the README within 1 MB.
- **Fidelity classes**, used for every mechanism:
  - **Exact**: output is identical to the engine without Sieve (greedy token-identical; the GPU
    sampler draws the same token for the same seed).
  - **Bounded**: output differs; the divergence is measured the way sparse decode attention's is (KL,
    text NLL, top-1 agreement) and must pass the gates in [section 9](#9-correctness-gates).

## 3. Cost model

### 3.1 Decode

A decode step reads every weight it uses once, so its time is close to

    t_step = B / (η · BW) + c0 + c1 · L

with B the touched bytes, BW the memory bandwidth, L the layer count, η the fraction of the bandwidth
the matvecs reach, c0 a fixed cost per token (embedding, token selection, command buffer) and c1 a
cost per layer that does not depend on weight size (about nine dependent dispatches per layer:
q/k/v, post-projection, attention, o, add + norm, gate/up, activation, down, add + norm).

A fit to seven decode rates from the README (decode at 16 tokens of context) gives **η = 0.91,
c0 = 0.25 ms, c1 = 46 µs per layer**:

| Model | Weights | Layers | Measured step | Bandwidth floor B / BW | Model | Error |
|---|---|---|---|---|---|---|
| Qwen2.5 0.5B | 278 MB | 24 | 2.92 ms | 1.41 ms | 2.90 ms | −0.4% |
| Qwen3 0.6B | 336 MB | 28 | 3.50 ms | 1.71 ms | 3.41 ms | −2.4% |
| Qwen3.5 0.8B | 424 MB | 24 | 3.69 ms | 2.15 ms | 3.72 ms | +0.8% |
| Llama 3.2 1B | 695 MB | 16 | 4.95 ms | 3.53 ms | 4.86 ms | −1.8% |
| Gemma 3 1B | 732 MB | 26 | 5.29 ms | 3.72 ms | 5.53 ms | +4.5% |
| Phi-3.5 mini | 2.15 GB | 32 | 14.29 ms | 10.91 ms | 13.72 ms | −4.0% |
| Mistral 7B | 4.08 GB | 32 | 24.39 ms | 20.71 ms | 24.48 ms | +0.4% |

Two things follow.

- **Small models are half overhead.** Qwen3 0.6B spends 1.71 ms reading weights and 1.79 ms on
  everything else; Llama 3.2 1B, 3.53 and 1.42. Cutting bytes alone cannot double their speed.
- **The constants are a fit, not a measurement.** Three parameters, seven points, one machine, and
  the hybrid Qwen 3.5 layers have a different dispatch count from the others. `mlxr bench` prints the
  real split (`decode step @16: … without qkv … lmhead …`, `tiny dependent dispatch`, `lm_head matvec`).
  Replacing η, c0 and c1 with those lines is the first task on a Mac
  ([12](#12-plan-and-gates), phase 0).

### 3.2 Multi-token passes

A pass that advances T tokens reads the weights once. Its cost relative to a single step is c_v(T).
Measured here (CHANGELOG 1.3.0): **c_v(2) ≈ 1.5** (Qwen3 0.6B: 5.4 ms against 3.5; Qwen3 8B: 40.7 ms
against 26.2) and **c_v(3…8) ≈ 3** on an M2 Pro, where the small-batch kernel is bound by arithmetic,
not by loads.

With γ drafted tokens per pass, each accepted with probability α, and a draft cost of c_d steps:

    tokens per pass  E = (1 − α^(γ+1)) / (1 − α)
    speed-up           = E / (c_v(γ+1) + c_d)

| Acceptance α | γ = 1, c_v = 1.5 (today) | γ = 1, c_v = 1.2 | γ = 3, c_v = 3.0 (today) | γ = 3, c_v = 1.6 |
|---|---|---|---|---|
| 0.45 | 0.94× | 1.16× | 0.55× | 1.00× |
| 0.60 | 1.03× | 1.28× | 0.69× | 1.24× |
| 0.75 | 1.13× | 1.40× | 0.87× | 1.56× |
| 0.85 | 1.19× | 1.48× | 1.01× | 1.82× |

(c_d = 0.05 per drafted token.) So multi-token decode pays only with a nearly free drafter, an
acceptance above about 0.6, and a 2-token pass much nearer 1.2 than 1.5.

### 3.3 Prefill

A prefill pass multiplies every body weight by every token: 2 · P_body floating-point operations per
token, whatever the layout. With an effective GEMM rate F,

    prefill tok/s ≈ F / (2 · P_body)

The README's prefill rates at 512 tokens imply F = 3.06 TFLOPS (Qwen2.5 0.5B, 0.358B body, 4,270
tok/s), 2.66 (Qwen3 0.6B, 0.440B, 3,020) and 3.35 (Llama 3.2 1B, 0.973B, 1,720), against the 3.5
TFLOPS the 64×64 tile measures on its own. Attention and the small kernels take the rest. This
document uses **F = 3.3 TFLOPS** for the M2 Pro.

## 4. What the targets require

### 4.1 Decode: 500 tok/s is a byte budget

500 tok/s is 2.0 ms per token. From section 3.1, the bytes a step may touch:

| Mac | Layers | c1 = 46 µs (today) | c1 = 23 µs (halved) |
|---|---|---|---|
| M2 Pro, 197 GB/s | 16 | 180 MB | 245 MB |
| M2 Pro | 24 | 115 MB | 212 MB |
| M2 Pro | 28 | 82 MB | 196 MB |
| 500 GB/s (assumed), same c0 and c1 | 16 | 456 MB | 622 MB |
| 500 GB/s | 24 | 291 MB | 539 MB |
| 500 GB/s | 28 | 208 MB | 498 MB |

A 1B model is 560–700 MB at 4 bits. On an M2 Pro a single-token step may therefore touch **a quarter
to a third of it at most, and only after the per-layer overhead is also halved.** The alternatives
are fewer bits per weight (ruled out in [13.3](#133-ruled-out-by-analysis)) or more than one token
per pass (section 3.2).

On a 500 GB/s Mac the same model needs to shed a third to a half of its bytes, which section 5 can
plausibly do. Bandwidth figures reported for current chips: 546 GB/s (M4 Max, 40-core GPU), 614 GB/s
(M5 Max, 40-core), 460 GB/s (M5 Max, 32-core). None was measured with this engine, and c0 and c1 on
those GPUs are unknown.

### 4.2 Prefill: 5,000 tok/s is an arithmetic budget

From section 3.3, 5,000 tok/s needs F ≥ 10,000 · P_body:

| Effective GEMM rate F | Largest body that prefills at 5,000 tok/s |
|---|---|
| 3.3 TFLOPS (M2 Pro, this engine) | 0.33B |
| 4.5 TFLOPS | 0.45B |
| 7.5 TFLOPS | 0.75B |
| 10 TFLOPS | 1.00B |

- Llama 3.2 1B (0.97B body) needs 9.7 TFLOPS: 2.9 times what this engine reaches on an M2 Pro, and
  above the 5.7–6.8 TFLOPS usually quoted as that GPU's peak (quoted figure, not verified).
- Qwen2.5 0.5B (0.36B body) is already at 4,270 tok/s. It is the size the target fits on this Mac.
- Exact prefill has no way to skip arithmetic: every prompt token needs every layer's output for the
  tokens after it. The only lossless saving is the last layer's MLP and output projection for
  non-final prompt tokens, about 1 / L of the work ([5.7](#57-prefill)).
- Sieve-FFN does not help a batch: each token keeps different channels, and a 64-token tile's union
  of 35% masks is nearly everything.

So **T2 is a statement about the Mac.** For a 1B model it needs a GPU that sustains about 10 TFLOPS
in this engine's GEMM. One report puts M5 Max prompt processing at 2.4 times the M4 Max's (model not
stated); whether this engine's `simdgroup_matrix` kernels see that gain is an open question
([13.2](#132-open-questions)).

### 4.3 Intelligence: above a 4B Qwen is not reachable without training

Scores from the Qwen 3.5 model cards, as returned by web search summaries (vendor-reported, not
verified against the cards; where a card gives two rows, both are shown, presumably without and with
thinking):

| Model | MMLU-Pro | GPQA Diamond | IFEval |
|---|---|---|---|
| Qwen3.5-9B | 82.5 | 81.7 | |
| **Qwen3.5-4B** | **79.1** | **76.2** | **89.8** |
| Qwen3.5-2B | 55.3 / 66.5 | 51.6 | 61.2 / 78.6 |
| Qwen3.5-0.8B | 29.7 / 42.3 | 11.9 | 52.1 / 44.0 |

Three facts close the question.

1. **An imported model is at best its source.** Sieve's exact profile reproduces the source model;
   the bounded profiles are slightly worse. Nothing in a training-free transform adds knowledge or
   reasoning.
2. **No 1B-class source is close.** With thinking, Qwen3.5-0.8B trails Qwen3.5-4B by 37 MMLU-Pro
   points and 64 GPQA points. Even the 2B, twice the size this project targets, trails by 13 and 25.
   This document has no like-for-like scores for the other 1B-class models (Llama 3.2 1B, Gemma 3 1B,
   LFM2.5-1.2B), and found no claim that any of them approaches the 4B.
3. **A larger source cannot be run at 1B cost.** That needs a step that touches about a tenth of a
   4B-or-larger model. Published training-free results stop well short of that: 40–50% sparsity
   model-wide (TEAL) and 60–70% of the MLP alone (Prox, whose abstract says earlier methods degrade
   substantially at high sparsity). Mixture-of-experts checkpoints are sparse by training, but the
   one nearest 1B active parameters that this document found (LFM2-8B-A1B: 8.3B total, 1.5B active)
   trades wins and losses with Qwen3-4B, the previous generation, in its own vendor's table. No claim
   that such a model beats Qwen3.5-4B was found.

Training does not rescue the full set of targets either. On an M2 Pro, T1 and T2 together describe a
model with about 0.3B *active* parameters per token (sections 4.1 and 4.2). A natively trained Sieve
model of 1B total and 0.3B active parameters would be expected to behave like a dense model well under
1B, which is an order of magnitude less compute per token than the 4B model it would have to beat.

[Section 11](#11-the-intelligence-target) lists what can be tested anyway and what should replace T3.

### 4.4 Projections

Decode, from the cost model with unchanged c0 and c1 (assumptions under the table):

| Model | Today (README) | Sieve-Head (exact) | + Sieve-FFN, 50% of channels | + Sieve-FFN, 35% | 35% and c1 = 30 µs |
|---|---|---|---|---|---|
| **M2 Pro** | | | | | |
| Qwen2.5 0.5B | 343 | 384 | 455 | 487 | 599 |
| Qwen3 0.6B | 286 | 326 | 367 | 384 | 464 |
| Qwen3.5 0.8B | 271 | 324 | needs its config | | |
| Llama 3.2 1B | 202 | 233 | 307 | 347 | 381 |
| Gemma 3 1B | 189 | 219 | needs its bit widths | | |
| Qwen3 1.7B | 143 (model) | 160 | 204 | 227 | 252 |
| **500 GB/s, assumed** | | | | | |
| Qwen2.5 0.5B | 507 (model) | 542 | 593 | 614 | 803 |
| Qwen3 0.6B | 438 (model) | 467 | 499 | 511 | 663 |
| Llama 3.2 1B | 395 (model) | 437 | 532 | 577 | 677 |
| Qwen3 1.7B | 271 (model) | 294 | 349 | 374 | 450 |

Assumptions, all to be replaced by measurements:
- Sieve-Head touches 26% of the head (an 8-bit sketch of rank d / 8, two bounds per row, 4,096 exact
  rows). Whether the certificate is that tight is hypothesis H1 ([12](#12-plan-and-gates)).
- Sieve-FFN touches a 2-bit proxy over a quarter of the inputs, the kept rows of gate and up, the
  kept rows of the transposed down projection and all of its scales and biases.
- Sieve adds dispatches (four for the head, a net one or two per layer), which the table ignores.
- Stride is not included. Where a model has a multi-token-prediction head, multiply by the entry of
  the table in section 3.2 that matches its measured acceptance and pass cost.

Reading the M2 Pro rows: a 1.2B model reaches about 350 tok/s, not 500. Qwen3.5 0.8B is the only
1B-class model with a route to 500: 324 from the head, an unknown gain from Sieve-FFN (its MLP sizes
were not available to this document), and 1.1–1.4× from Stride.

Prefill is unchanged by Sieve apart from the last-layer saving: Llama 3.2 1B stays near 1,800 tok/s
on an M2 Pro and Qwen3.5 0.8B near 2,900.

## 5. Architecture

### 5.1 Definition

A **Sieve model** is a decoder-only transformer whose output projection and MLP blocks are evaluated
in two stages:

1. a **screen** that reads a small, separately stored summary of the weights and decides which rows
   (vocabulary rows, MLP channels) matter for this token;
2. an **exact** stage that evaluates those rows with the original weights and the original
   arithmetic.

Its **dense limit** (every row passes) is the source architecture, weight for weight. That is what
makes import possible without training: the importer adds screens, it never changes a weight. The
existing sparse decode attention is the same idea applied to the KV cache (score block means, select,
attend exactly), and Sieve keeps it as is.

Everything else is unchanged: attention and Gated DeltaNet mixers, normalisation, RoPE, tokenizers,
chat templates, the KV cache, the sampler.

Profiles:

| Profile | Mechanisms | Class | Default |
|---|---|---|---|
| `off` | none: the engine as it is today | | yes |
| `exact` | Sieve-Head, Stride | Exact | candidate default once proven |
| `balanced` | `exact` + Sieve-FFN at the largest sparsity that passes the `balanced` gate | Bounded | opt-in |
| `fast` | `exact` + Sieve-FFN at the largest sparsity that passes the `fast` gate | Bounded | opt-in |

### 5.2 Sieve-Head

The head computes z_i = w_i · x for every vocabulary row w_i (V rows of d values) and the final
normalised hidden state x. It is 20–35% of a 1B-class model's bytes: 148 MB of 695 for Llama 3.2 1B,
about 143 MB of 424 for Qwen3.5 0.8B, 170 MB for Gemma 3 1B. The sampler then uses very few of the V
logits: the argmax when greedy, otherwise the exact top k (`sample_token` builds its pool from the top
`top_k` logits, or the top 512 when top-k is off, capped at `SMP_CAP`).

Sieve-Head computes exactly those logits and proves that no other row can be among them.

**Stored by the importer** (section 6):
- μ, the mean final hidden state over a calibration text (d values);
- Q, an orthonormal basis of the r directions that hold most of the variance of x − μ over that text
  (r × d; r is chosen by the importer, expected d / 16 to d / 4);
- for each row: the prior p_i = w_i · μ, the sketch s_i = Qᵀ w_i quantized to 8 or 4 bits (ŝ_i), and
  two error norms rounded up: q_i ≥ the in-subspace part of w_i − Q ŝ_i (sketch quantization), and
  n_i ≥ the out-of-subspace part of w_i (what the sketch cannot see).

**Per step:**

1. `head_project`: x̃ = x − μ; x_r = Qᵀ x̃; ν = ‖x_r‖; ρ = ‖x̃ − Q x_r‖.
2. `head_screen`, every row: ẑ_i = p_i + ŝ_i · x_r, and the bound b_i = q_i ν + n_i ρ + ε. By
   Cauchy–Schwarz on the two orthogonal parts, **l_i = ẑ_i − b_i ≤ z_i ≤ ẑ_i + b_i = u_i**. It writes
   l and u, and per block of 64 rows the maxima of l and of u.
3. `head_cut`: a threshold θ with at least k rows whose lower bound reaches it (k = 1: θ = max l).
   Found by bisecting on the block maxima, the way `sample_token` finds its pool. Every row with
   u_i < θ is certainly outside the top k.
4. `head_exact`: for each block whose maximum of u reaches θ, the exact logits of its rows, with the
   dense head's per-row arithmetic (`qmv`), written to the ordinary `logits` buffer. Every other row
   gets −∞.
5. The sampler runs unchanged on `logits`.

Properties:
- **Exact.** The top k rows all pass, and their logits are the dense head's bit for bit, so greedy
  output and seeded sampling are identical. Repetition penalty is a monotone change to a few rows:
  `head_cut` applies it to those rows' bounds before choosing θ.
- **Self-limiting.** If the bounds are loose, more blocks pass. In the worst case all pass and the
  step costs a dense head plus the screen. There is no separate fallback path and no CPU round trip.
  The importer measures the expected cost and leaves the head sieve off for a model when it is above
  80% of the dense head.
- **Determinism.** Blocks are selected by a comparison each threadgroup makes for itself. No atomics
  order any floating-point sum.
- **ε** covers float rounding in the screen and in the dense logit, and the f16 rounding of Q. The
  importer measures both and checks, on held-out calibration states, that every dense top-k row is
  inside the certified set. One violation aborts the import.

Where it does not apply: passes that need every logit (`allLogits`, teacher-forced scoring) and CPU
sampling over a constrained token set (tool-call names) use the dense head. Gemma's final-logit
soft-cap is monotone, so it is applied to bounds and logits alike.

**Why a basis of the hidden states, not of the weights.** A low-rank sketch of W alone would leave an
error of ‖w_i‖ · ‖x‖ scale in every logit, far wider than the gaps between the top tokens. With a
basis fitted to x, the unseen part is n_i · ρ, and ρ is small exactly when x lies in the subspace it
usually lies in. Whether ρ is small enough at a rank that is cheap to scan is hypothesis **H1**:
phase 0 measures it before any kernel is written. If it fails, the fallback is a fixed-budget screen
(top blocks by ẑ, no certificate), which is Bounded and belongs to the opt-in profiles.

Projected touched bytes with r = d / 8 and an 8-bit sketch:

| Model | Dense head | Sieve-Head | Saved per token on an M2 Pro |
|---|---|---|---|
| Qwen3 0.6B | 88 MB | 23 MB | 0.36 ms |
| Llama 3.2 1B | 148 MB | 39 MB | 0.61 ms |
| Qwen3.5 0.8B (V ≈ 248K) | 143 MB | 36 MB | 0.60 ms |
| Gemma 3 1B | 170 MB | 43 MB | 0.72 ms |
| Qwen3.5 4B (V ≈ 248K) | 357 MB | 89 MB | 1.51 ms |

### 5.3 Sieve-FFN

A gated MLP computes y = D · (act(G x) ⊙ (U x)): I channels, each with a gate row g_i and an up row
u_i (d values each) and a column d_i of the down projection. Channel i contributes a_i · d_i with
a_i = act(g_i · x)(u_i · x). The MLP is 45–65% of a 1B-class model's bytes (453 MB of 695 for Llama
3.2 1B). Channel contributions are heavy-tailed, and published work keeps 30–60% of the channels at
small loss on 7B and larger models (TEAL and Prox without any training, CATS with little or none;
section 14). Whether that holds on 1B-class models is hypothesis **H2**.

Sieve-FFN follows Prox's two stages, laid out for this engine's kernels.

**Stored by the importer**, per layer:
- a **proxy** of gate and up: 2 bits per weight, stored column-major (all 2I channel codes of input
  dimension j are contiguous), with a scale and offset per channel;
- ‖d_i‖ for every channel;
- **Dᵀ**: the down projection's 4-bit codes re-laid row-major by channel, so a channel's column is
  one contiguous run, and its (scale, bias) pairs re-laid as [channel group][output]. Same codes,
  same scales, same biases: only the byte order changes;
- per profile: a threshold τ, the input fraction φ and a scale correction κ.

**Per step:**

1. `ffn_topm`: J = the ⌈φ d⌉ input dimensions with the largest |x_j| (ties to the lower index).
   φ = 1/4 holds about 70% of the energy of a Gaussian-like x.
2. `ffn_proxy`: for every channel, ĝ_i and û_i = κ × the proxy dot product over J; salience
   ĉ_i = |act(ĝ_i) · û_i| · ‖d_i‖; **keep bit** i = (ĉ_i ≥ τ). Lanes own channels and loop over the
   columns in J, so the read is ⌈φ d⌉ contiguous runs of I / 2 bytes.
3. `ffn_exact`: for each block of 32 channels, the kept channels' gate and up rows with the original
   tensors and `qmv`'s row arithmetic, and a_i = act(g)·u in the epilogue (the same expression as the
   `silu` kernel). Channels that were not kept get a_i = 0. Blocks with no kept channel return at once.
4. `ffn_down_t`: y = Σ over kept channels of a_i · Dᵀ[i], in ascending channel order, split across
   threadgroups by channel range into partial sums.
5. `add_combine`: h += the partial sums in fixed order (and the next norm, as `add_rms_norm` does now).

Properties:
- **Bounded.** With τ = −∞ every channel is kept and the result equals the dense MLP up to the
  summation order of the down projection (step 4 sums by channel, `qmv` by lane). It is therefore
  *not* bit-identical to today's kernels even at full budget, so **the `exact` profile does not use
  Sieve-FFN at all**: it runs the existing gate/up, activation and down kernels.
- **Deterministic.** Keep bits are threshold comparisons, and the sums run in a fixed order.
- **Thresholds, not counts.** A calibrated τ per layer gives each token as many channels as it needs
  and avoids a selection kernel. The worst case is every channel: a dense MLP plus the proxy.
- **Quality-targeted profiles.** The importer finds, per layer, the largest sparsity whose
  end-to-end divergence passes the profile's gate ([7](#7-importer)). `balanced` and `fast` name
  gates, not sparsity levels.

Touched bytes per layer, Llama 3.2 1B (d = 2,048, I = 8,192; dense MLP 28.3 MB):

| Part | 50% kept | 35% kept |
|---|---|---|
| Proxy (φ = 1/4, 2 bits) | 2.1 MB | 2.1 MB |
| Gate and up rows | 9.4 MB | 6.6 MB |
| Dᵀ rows | 4.2 MB | 2.9 MB |
| Dᵀ scales and biases (all groups) | 1.0 MB | 1.0 MB |
| Total | 16.8 MB (59%) | 12.7 MB (45%) |

The down projection's scales and biases are per group of 64 channels and per output, so every group
is touched whatever is kept: the last row does not shrink with sparsity.

Memory: the proxy (134 MB) and Dᵀ (151 MB) add 285 MB to Llama 3.2 1B's 695 MB. The original down
projection stays for prefill until the prefill GEMM can read Dᵀ ([13.2](#132-open-questions)).

Activations: SiLU (Llama, Qwen, Mistral, Phi) and GELU (Gemma). Phi-3's fused `gate_up` and Gemma 4's
per-layer inputs do not change the scheme.

### 5.4 Attention and mixers

Unchanged. Sparse decode attention keeps its own switch, budget and threshold (5,128 tokens) and
composes with every Sieve profile. Qwen 3.5's Gated DeltaNet layers keep their kernels; their
projections are ordinary matvecs and stay dense.

### 5.5 Stride: multi-token verified decode

The engine already verifies drafted tokens in one batched pass with an exact acceptance rule
(`Engine.swift`: prompt lookup with `--spec`, draft models with `--draft`; batches of 2–8 with
`qmvt4` / `qmvx4`). It is off by default because a draft model costs more than it saves, and it is
disabled for Qwen 3.5 because the recurrent state cannot drop a rejected token.

Stride adds the drafter that costs almost nothing, and the state handling Qwen 3.5 needs.

**The drafter.** Qwen 3.5 and 3.6 checkpoints ship a multi-token-prediction head
(`mtp_num_hidden_layers: 1`). As described in mlx-lm pull request 990: it predicts token t+2 from the
backbone's hidden state at t and the embedding of token t+1, through `pre_fc_norm_hidden`,
`pre_fc_norm_embedding`, a linear projection (`mtp.fc`), one full-attention transformer layer, and the
shared `lm_head`. The engine currently leaves `mtp.*` tensors on disk
(`SafeTensorFile.unusedPrefixes`). Stride loads them when the profile needs them.

**The cycle** (draft depth 1): a pass at position t yields the hidden state h_t and the token x_{t+1}.
The head drafts x̂_{t+2}. The next pass runs both x_{t+1} and x̂_{t+2} (T = 2): its first row gives the
true x_{t+2}; if that equals the draft, its second row gives x_{t+3} as well.

**What makes it pay here and not in other MLX runtimes** (reported gains there: +57% on a 27B, +11%
on a 9B, +4–6% on a 9B on another machine, and a loss on fp16 weights):
- the draft needs a head pass, which Sieve-Head makes cheap (36 MB instead of 143 on Qwen3.5 0.8B);
- the 2-token pass reads each weight once (`qmvt4`), at 1.5 steps today. The table in section 3.2
  says the target is 1.2 to 1.3.

**Recurrent state.** A T-token pass computes the state after each token in turn. Stride keeps the
state after every token of the pass in its own buffer (two buffers at depth 1) and continues from the
one after the last accepted token. No snapshot copy, no replay. Memory: one extra state per drafted
token (`LinearAttention.stateBytes` per linear layer).

**Constraints and risks.**
- Not every export has the weights: `mlx-community/Qwen3.5-27B-4bit` is reported to declare the head
  and ship none. The importer checks for the tensors.
- Reported acceptance varies widely: 44% to 88% in most reports, far lower in some quantized
  exports, and the reports on quantizing the head conflict. Acceptance is measured per export at
  import, and Stride is enabled only where the table in section 3.2 predicts a gain.
- The head's transformer layer attends over earlier positions, so it needs K/V rows of its own for
  every position. Either it runs on every token, prefill included (about 1 / L more work), or it
  follows whatever the reference implementation does. Settle this against mlx-lm's implementation,
  and check the head's logits against it, before building the kernels.
- The cycle needs the accept decision before the next pass is encoded, so it gives up the pipelined
  decode's overlap (+4–10% in CHANGELOG 1.1.1) unless positions move to a GPU buffer. Phase 3
  measures both.
- Sampled output: the existing exact acceptance rule keeps the target distribution.

Models without such a head get Stride only through prompt lookup (`NGramDraft`), which is already
there and pays on copy-heavy text only.

### 5.6 Per-layer overhead

Section 3.1 puts 46 µs per layer outside the weight reads: about 1.3 ms of Qwen3 0.6B's 3.5 ms step.
Sieve does not reduce it, and its extra dispatches add to it. Three experiments, each kept only if
`mlxr bench` shows a gain (AGENTS.md):

- **E1: norm by partial sums.** The matvec that produces a residual update adds it to h in its
  epilogue and writes one partial sum of squares per threadgroup. The next matvec's threadgroups sum
  those partials (a few hundred floats) and scale x as they load it. Removes both `add_rms_norm`
  dispatches of a layer. This differs from the fusion CHANGELOG 1.3.0 rejected, where every
  threadgroup recomputed the whole norm. It changes the norm's rounding, so it is **not
  token-identical** and may only ship inside the Bounded profiles.
- **E2: activation in the exact stage.** Already part of `ffn_exact`. The dense equivalent measured
  4–11% slower (CHANGELOG 1.2.0), so the masked form must be re-measured, not assumed.
- **E3: post-projection and attention in one dispatch at short context.** Part of the same rejected
  three-way fusion; never measured on its own.

Target: c1 ≤ 30 µs per layer. Section 4.4's last column shows what that is worth.

### 5.7 Prefill

Prefill stays exact and dense.

- **Last-layer saving (Exact).** When a pass does not need logits for a row, the last layer's MLP and
  output projection are not computed for it; when it needs only the last row's logits, they are
  computed for that row alone. About 1 / L of the pass.
- **Long prompts.** `flash_prefill` is the known gap against mlx-lm at 8K–30K tokens
  (sparse-decode-attention.md, "Against mlx-lm"). That work is independent of Sieve and worth more to
  T2 at long prompts than anything in this document.
- **Not prefilling.** A prefix that was processed before costs nothing: the engine already reuses the
  KV cache across turns. Saving the cache (and the recurrent state) of a system prompt to disk would
  extend that across launches. It improves time to first token, not the T2 number.

### 5.8 Native Sieve models (out of scope)

A model trained for this architecture could make the screens part of the model: ReLU-like MLPs whose
inactive channels are exactly zero, a trained channel predictor, a hierarchical output layer. It would
touch far fewer bytes at the same quality than an import can. It is not specified here, because C2
excludes training and section 4.3 shows it would still not meet T3.

## 6. Container format

A Sieve model is the original MLX model directory plus two files. The original files are never
modified.

    <model>/
      config.json, model*.safetensors, tokenizer files     the MLX export, unchanged
      sieve.json                                           manifest
      sieve.safetensors                                    screens and re-laid tensors

`sieve.json`:

    {
      "format": "mlxr-sieve", "version": 1,
      "source": { "config_sha256": "…", "weights": [ { "file": "model.safetensors", "sha256": "…" } ] },
      "calibration": { "kind": "self-generated" | "file", "tokens": 65536, "seed": 42, "sha256": "…" },
      "head": { "enabled": true, "rank": 256, "block": 64, "sketch_bits": 8, "epsilon": 0.002,
                "expected_fraction": 0.26 },
      "ffn": { "proxy_bits": 2,
               "layers": [ { "phi": 0.25, "kappa": 1.18,
                             "tau": { "balanced": 0.031, "fast": 0.074 } } ] },
      "stride": { "mtp": true, "depth": 1, "acceptance": 0.71 },
      "profiles": { "balanced": { "nll_delta": 0.008, "kl": 0.004, "kept": 0.52 },
                    "fast":     { "nll_delta": 0.027, "kl": 0.015, "kept": 0.36 } }
    }

(The numbers are placeholders showing the shape, not results.)

`sieve.safetensors`:

| Tensor | Shape | Type | Content |
|---|---|---|---|
| `head.mean` | [d] | f32 | μ |
| `head.basis` | [r, d] | f16 (f32 if its rounding widens the bounds measurably) | Q, rows orthonormal |
| `head.sketch.weight`, `.scales`, `.biases` | [V, r·bits/8], [V, r/g] | MLX quantization | ŝ |
| `head.prior` | [V] | f32 | p |
| `head.qerr`, `head.rnorm` | [V] | f16 | q and n, rounded up |
| `ffn.<l>.proxy.codes` | [d, I/2] | u8 | 2-bit codes, column-major, gate channels then up channels |
| `ffn.<l>.proxy.scale`, `.offset` | [2I] | f16 | per channel |
| `ffn.<l>.dnorm` | [I] | f16 | ‖d_i‖ |
| `ffn.<l>.down_t.weight` | [I, d/2] | u8 | the down projection's 4-bit codes by channel |
| `ffn.<l>.down_t.pairs` | [I/g, d, 2] | f16 | its (scale, bias) by channel group and output |

Rules for the loader (AGENTS.md, untrusted input; the engine builds with `-Ounchecked`):
- The manifest's format, version and source hashes must match; a sidecar for other weights is ignored
  with a note, never used.
- Every shape is checked against `config.json` before any buffer is bound: r ≤ d and a multiple of
  32, block ∈ {16, 32, 64}, layer count, I, d, group sizes.
- Every scalar is finite and in range: 0 < φ ≤ 1, κ > 0, τ ≥ 0, ε ≥ 0, bounds non-negative.
- No index in the sidecar is trusted: the sidecar holds no token ids, row lists or offsets. Lists
  are produced at run time by kernels from host-checked sizes.
- A sidecar that fails any check loads the model with Sieve off and says why
  (`LoadedModelInfo.sieveNote`), as a sparse index that does not fit does today.

The files use MLX's own conventions (safetensors, MLX affine quantization), so a reference
implementation in mlx-lm can read them. That implementation would live outside this repository, which
is Swift only.

## 7. Importer

A new `sieve` command writes the sidecar: `sieve import <model>`, with `--calib <text file>` and
`--profiles balanced,fast` as options. Swift only; linear algebra through Accelerate (LAPACK symmetric
eigendecomposition, BLAS products). Nothing is trained: there is no loss function and no gradient.

(Commands that do not exist yet are written without the `mlxr` prefix throughout this document. The
documentation tests parse every command line under `docs/` that starts with `mlxr` against the real
command table.)

1. **Check the source.** A supported architecture, 4-bit weights with group sizes the kernels take.
   Record the hashes.
2. **Calibration text.** A text file given with `--calib` (at least about 50K tokens), or by default
   text the model writes itself: sequences of 512 tokens sampled at temperature 1 from the chat
   template with a fixed seed. Split into a fitting part and a held-out part.
3. **Head.** Run the dense model over the fitting part and collect the final normalised hidden
   states. Compute μ and the covariance, take the top-r eigenvectors as Q for r in {d/16, d/8, d/4}.
   For each: the sketches, their quantization at 8 and at 4 bits, q and n, and, on the held-out part,
   the number of rows the certificate keeps for k = 1 and k = 64. Keep the cheapest combination, or
   disable the head sieve if none is under 80% of the dense head. Check that every held-out dense
   top-k row is inside the certified set.
4. **MLP.** Per layer: quantize the proxy (2 bits, per-channel scale and offset minimising squared
   error), compute ‖d_i‖, re-lay Dᵀ. Fit κ so the proxy's mean salience matches the true one on the
   fitting part.
5. **Thresholds.** For each requested profile, find per-layer τ by coordinate descent on the
   held-out part: start every layer at the same kept fraction, then move budget between layers while
   the end-to-end NLL change per byte saved improves, until the profile's gate
   ([9](#9-correctness-gates)) is just met. Record the divergence reached.
6. **Stride.** If the weights hold `mtp.*` tensors, measure greedy acceptance on the held-out part
   and record it.
7. **Self-check, then write.** Sieve-Head against the dense head (identical tokens on the held-out
   part); every kernel at full budget against its dense reference; the profile metrics. Write the
   files atomically. An import that fails a check writes nothing.

Expected run time on a 1B-class model: minutes, dominated by step 5's repeated forward passes
(estimate, not measured).

## 8. Engine integration

Files (planned):

- `Sources/MLXEngine/Sieve.swift`: `SieveProfile`, manifest parsing and validation, environment
  parsing, memory accounting.
- `Sources/MLXEngine/SieveImport.swift`: the importer.
- `Sources/MLXEngine/Kernels.swift`: `head_project`, `head_screen`, `head_cut`, `head_exact`;
  `ffn_topm`, `ffn_proxy`, `ffn_exact`, `ffn_down_t`, `add_combine`; the multi-token-prediction head.
- `Sources/MLXEngine/GPU.swift`: parameter structs; `mtp.*` no longer skipped when Stride is on.
- `Sources/MLXEngine/Transformer.swift`: the sidecar buffers; the Sieve branches of `encode` (after
  the last layer for the head, in the MLP block for Sieve-FFN); per-token recurrent state outputs.
- `Sources/MLXEngine/Engine.swift`: `load(directory:draftDirectory:contextLength:sparse:sieve:)`, the
  memory check including the sidecar, the Stride cycle next to the existing speculation loop, the
  `sieve…` fields of `LoadedModelInfo`.
- `Sources/MLXRCLI`, `Sources/mlxr/main.swift`: the `sieve` command (`import`, `info`, `verify`), the
  `--sieve` option, the bench section, the `eval` command. Each new command and option also needs its
  entry in the command table, help and completions, and the documentation tests then cover it.
- `Sources/MLXRunner`: an inspector setting, after the CLI path is proven.
- Tests: `SieveKernelTests.swift`, `SieveModelTests.swift`, `SieveManifestTests.swift`.

Switches, mirroring sparse attention:

| Where | Setting |
|---|---|
| CLI | `mlxr run` and `mlxr serve` take `--sieve` with `off`, `exact`, `balanced` or `fast` |
| Environment | `MLXR_SIEVE` (same values; `0` = off) |
| Code | `InferenceEngine.load(…, sieve: SieveProfile?)`; nil means off |
| App | inspector, applies at the next load |

A model without a sidecar loads as today whatever the setting, and says so once.

Kernels reuse what exists: `head_exact` and `ffn_exact` are `qmv`'s row arithmetic over a set of row
blocks (`qmvp` already walks blocks); `head_cut` is `sample_token`'s bisection on block maxima;
`ffn_down_t` and `ffn_proxy` are one new primitive, "add a list of scaled rows", with 4-bit and 2-bit
codes.

## 9. Correctness gates

In addition to AGENTS.md's gates (`swift test`, token-identical greedy output on every existing
path, `prefillMatchesIncrementalDecode`):

- **Off means untouched.** Without `--sieve`, or with `off`, there are no Sieve buffers and no Sieve
  dispatches, and output is bit-identical to the build before Sieve. A model directory that holds a
  sidecar behaves the same as one that does not.
- **`exact` is exact.** On every installed model with a sidecar: greedy output
  (`--temp 0 --repeat-penalty 1 --seed 42 --max 256`) is token-identical to `off`; seeded sampled
  output is token-identical; the logits of every certified row are bit-identical
  (`sieveExactMatchesDense`). Stride on and off give the same tokens (`strideIsTokenIdentical`).
- **Full budget.** Sieve-Head with θ = −∞ is bit-identical to the dense head. Sieve-FFN with τ = −∞
  matches the dense MLP within 1e-4 relative on every output and gives the same argmax at every
  position of a test text that is not a near-tie (`sieveFullBudgetMatchesDense`).
- **Each kernel against a CPU reference**, including empty masks, all-ones masks, masks with one
  channel, ties in `ffn_topm`, a tie at θ, non-finite hidden states, and V not a multiple of the block.
- **Certificate.** On random and adversarial hidden states (including x far outside the calibration
  subspace), no dense top-k row is outside the certified set (`sieveHeadCertificateHolds`).
- **Determinism.** The same prompt and seed give the same tokens across runs in every profile.
- **Prefill and decode agree** in every profile (`prefillMatchesIncrementalDecode` with Sieve on;
  prefill is dense, so in the Bounded profiles this compares within the profile's tolerance and is
  documented as such).
- **Untrusted sidecars.** Truncated files, wrong shapes, out-of-range scalars, hashes that do not
  match, and a sidecar from another model each load with Sieve off and never reach a kernel.
- **Bounded profiles pass their gates** on the held-out calibration text and on a second text the
  importer did not see:

  | Profile | Text NLL change | Mean KL to dense | Top-1 agreement where dense's margin ≥ 1 nat |
  |---|---|---|---|
  | `balanced` | ≤ +0.01 nats / token (about 1% perplexity) | ≤ 0.01 | ≥ 99.5% |
  | `fast` | ≤ +0.03 | ≤ 0.03 | ≥ 99% |

  The limits are set to the range sparse decode attention already ships in (at most about 3%
  perplexity). They are proposals: the owner sets the final values.

## 10. Measuring

Nothing else may use the GPU while measuring (AGENTS.md).

Speed (`MLXR_SIEVE` is planned; the commands themselves exist today):

    mlxr bench <model>                                   # today's numbers and the per-kernel profile
    MLXR_SIEVE=exact mlxr bench <model>                  # the same with a profile
    MLXR_BENCH_DECODE=1 MLXR_SIEVE=fast mlxr bench <model>
    MLXR_SIEVE=exact mlxr run <model> "<prompt>" --temp 0 --repeat-penalty 1 --seed 42 --max 256 --verbose

The bench section for Sieve adds, per profile: decode tok/s at 16 and 1,024 tokens of context,
**touched bytes per token** (summed from the masks, so the achieved GB/s is visible), certified rows
per step and its distribution, kept channel fraction per layer, Stride acceptance and tokens per pass,
and the cost of each new kernel.

Fidelity (Bounded profiles), with the long-context bench's text metrics at a short context:

    MLXR_BENCH_TEXT=book.txt MLXR_BENCH_POSITIONS=2048 MLXR_SIEVE=balanced mlxr bench <model>

KL, text NLL under both modes, top-1 agreement by margin bucket, and the greedy prefix shared with
dense. Use a text the model cannot have memorised, and a second domain (code).

Intelligence: a new developer command, `eval <model> --suite <dir>`, with `--sieve <profile>` and
`--limit N` as options:
- A suite is a directory of JSONL files the user provides, one item per line: `question`, `choices`,
  `answer`. No dataset ships with the repository.
- Scoring: the model answers through its chat template with the model's own thinking default and a
  fixed token budget; the answer letter is extracted from the reply. Greedy, seed 42.
- Report accuracy per file, the paired difference between two runs with a bootstrap 95% interval, and
  wall-clock time per item.
- Suite for this project: MMLU-Pro (a stratified 2,000-item subset), GPQA Diamond, GSM8K, ARC-Challenge.
  The comparison model is `mlx-community/Qwen3.5-4B-4bit` in this engine with `--sieve off`.

## 11. The intelligence target

### 11.1 What is known

Section 4.3: under C2, a 1B-class Sieve model is its source model, and the nearest source in the same
family (Qwen3.5-0.8B) is 37 MMLU-Pro points below Qwen3.5-4B.

### 11.2 What can still be tested, and what to expect

| Route | What it is | Expectation | Decided by |
|---|---|---|---|
| I1. Spend the speed | A 1B-class model at 2–4× the 4B's speed thinks longer, or answers several times and votes, in the same wall-clock time. | Gains of a few points on math, little on knowledge. Does not close a 37-point gap. | The `eval` command at equal time per item. One day once it exists. |
| I2. A larger model, touched sparsely | Import Qwen3.5-2B or 4B and run the `fast` profile. | A 4B at perhaps 1.5–2× its dense speed with a small loss (an estimate): a better trade than the dense 2B, still far from 1B cost. | The Pareto measurement in 11.3. |
| I3. Mixture-of-experts imports | Support models that are sparse by training (about 1–1.5B active parameters). | Engineering, not research; the engine lists MoE as unsupported today. No such model is claimed to beat Qwen3.5-4B, and at 1.5B active it would not reach 500 tok/s on an M2 Pro. | A separate decision; not part of Sieve. |
| I4. Train for it | Distil a larger model into a native Sieve model. | The only route with a mechanism for raising capability per byte. Violates C2, needs a training stack this repository does not have, and by section 4.3 still would not reach T3 at T1/T2's cost. | Out of scope. |

### 11.3 Recommended replacement: T3′

Replace T3 with two claims that an import can meet and a measurement can refute:

- **T3′a, fidelity.** In the `exact` profile a Sieve model reproduces its source token for token, so
  it scores exactly what the source scores.
- **T3′b, Pareto gain.** On one Mac, plot every dense Qwen 3.5 size (0.8B, 2B, 4B) as a point
  (measured ms per token, suite accuracy) and join them. Every Sieve profile of every imported size
  must lie above that line: at its speed, it beats the dense model one could have run instead.

  Worked example with the public scores and this repository's M2 Pro speeds: the line from the dense
  2B (7.5 ms, 66.5 MMLU-Pro) to the dense 4B (16.1 ms, 79.1) passes 69.4 at 9.5 ms. A 4B in the `fast`
  profile at 9.5 ms per token meets T3′b if it keeps more than 69.4, that is, loses fewer than 9.7
  points. (The 9.5 ms is an estimate; the 4B's MLP sizes were not available to this document.)

T3 itself stays in this document as a target that was examined and found unreachable under C2.

## 12. Plan and gates

Each phase ends with a measurement that either releases the next phase or stops the line of work.
Durations are estimates for one person on a Mac.

**Phase 0: measure before building (2–4 days).** No shipped code.

| # | Measurement | How | Gate |
|---|---|---|---|
| 0.1 | The cost model's constants | `mlxr bench` on Qwen3 0.6B, Llama 3.2 1B, Qwen3.5 0.8B; record the profile lines | Replaces section 3.1. If c1 is mostly dispatch latency, E1–E3 matter more than Sieve-FFN on small models. |
| 0.2 | **H1**, head certificate | Dump 20K final hidden states (a debug read of `xn` after a forward with all logits); on the CPU, fit μ and Q and count certified rows for k = 1 and 64, r ∈ {d/16, d/8, d/4}, unquantized sketch first | Median certified rows ≤ 2% of V and expected touched bytes ≤ 40% of the dense head for some r ≤ d/4. Otherwise Sieve-Head is not Exact: stop, or continue with the fixed-budget variant as Bounded. |
| 0.3 | **H2**, MLP concentration | Oracle selection: keep the top fraction of channels by true salience per token (zero the rest of the activation buffer before the down projection); text NLL and KL at 70, 50, 35 and 25% kept | NLL change ≤ +0.02 at ≤ 50% kept on the 1B-class candidates. Otherwise no proxy can do better: drop Sieve-FFN for that size class. |
| 0.4 | **H3**, multi-token head | Which installed Qwen 3.5 exports hold `mtp.*`; greedy acceptance on chat text with the head evaluated on the CPU or in mlx-lm | Acceptance ≥ 0.6. Otherwise Stride waits for a better head or a cheaper pass. |
| 0.5 | The 2-token pass | `verify step` lines of `mlxr bench` | Sets c_v(2) in section 3.2. |

**Phase 1: Sieve-Head (1–2 weeks).** Importer steps 1–3 and 7, the four head kernels, `--sieve exact`.
Gate: `exact` is exact on every installed model, and decode improves by at least 8% on Qwen3.5 0.8B
and Llama 3.2 1B. Deliverable: a lossless speed-up on every model, largest where the vocabulary is
large and the model small.

**Phase 2: Sieve-FFN (2–4 weeks).** Proxy, the two list kernels, thresholds, the bench section.
Gate: the `balanced` profile passes its fidelity gate and decodes at least 1.15× `exact` on the
flagship; `fast` passes its gate. If neither does on 1B-class models, keep Sieve-FFN for 2B and
larger only.

**Phase 3: Stride (1–2 weeks).** The multi-token-prediction head, per-token state outputs, the cycle.
Gate: token-identical, and at least 1.1× on Qwen3.5 0.8B. Then the 2-token pass (c_v from 1.5 toward
1.2) as its own performance pass.

**Phase 4: per-layer overhead (open-ended, one experiment at a time).** E1–E3, each measured alone.

**Phase 5: prefill.** The last-layer saving; then `flash_prefill`.

**Phase 6: the `eval` command and the intelligence measurements.** I1 and T3′b. Can start as soon as phase 1
lands.

Forecast for the flagship (Qwen3.5 0.8B, M2 Pro, decode): 271 today → about 320 after phase 1 →
360–420 after phase 2 (if H2 holds; its MLP sizes are needed for a tighter number) → 400–540 after
phase 3. **500 tok/s is at the optimistic end of that range and is not a commitment.** Prefill stays
near 2,900 tok/s.

## 13. Risks, open questions and what was ruled out

### 13.1 Risks

- **H1 may fail.** If final hidden states are not concentrated in a subspace that is cheap to scan,
  the certificate keeps most rows and Sieve-Head saves nothing. Phase 0.2 costs a day and decides it.
- **H2 is unproven at this size.** The training-free sparsity results are on 7B–70B models. Small
  models usually tolerate less. Phase 0.3 decides it per model.
- **More dispatches.** Sieve adds four dispatches for the head and one or two per layer. At 5–7 µs
  each (to be measured) that is 0.1–0.3 ms per token, a tenth of the 2 ms budget.
- **Memory.** The sidecar adds roughly 40% to a model's size (proxy and Dᵀ). On a 16 GB Mac that is
  immaterial for 1B-class models and about 1.3 GB for a 4B.
- **Calibration dependence.** Thresholds and the head basis are fitted to a text. Sieve-Head stays
  exact on any input (off-distribution input only makes it slower). Sieve-FFN's divergence on text
  unlike the calibration text is not bounded by anything but measurement.
- **Two kernels per bit width.** Only 4-bit has optimised decode kernels today; Sieve inherits that.

### 13.2 Open questions

1. How do c0 and c1 split between dispatch latency, the small kernels and the matvecs' ramp-up?
2. Can the prefill GEMM read Dᵀ, so the original down projection need not stay in memory?
3. Does a clustered row order for the head (similar tokens in the same block) reduce certified
   blocks enough to justify remapping token ids at the engine boundary?
4. Is a 1-bit row-major proxy over all inputs better per byte than the 2-bit column-major one?
   A low-rank predictor fitted to activations is a third option (arXiv 2603.14110).
5. Do M5-class GPUs' new accelerators speed up `simdgroup_matrix` kernels, or only code that goes
   through Apple's tensor APIs?
6. Draft depth 2 for the multi-token head: mlx-lm's authors report it only recently level with depth 1.
7. Position state on the GPU, so Stride keeps the pipelined decode.
8. Whether Sieve-FFN's keep bits can be reused across the two tokens of a Stride pass.

### 13.3 Ruled out by analysis

- **Fewer bits per weight as the main lever.** Meeting the M2 Pro byte budget for a 1B model needs
  about 2 bits per weight. Training-free quantization at 2 bits costs far more quality than T3 can
  afford, and only 4-bit has optimised kernels. Mixed precision by sensitivity remains a separate,
  smaller lever.
- **A draft model.** Measured here: Qwen3 0.6B drafting for Qwen3 8B is accepted 45% of the time and
  makes decode slower (CHANGELOG 1.3.0). For a 1B target the draft would have to run well above 1,000
  tok/s.
- **Self-drafting with the lossy path, verified by the dense path.** The draft is 40–50% of a dense
  step; with any realistic acceptance the cycle is no faster than dense.
- **A cascade with a larger verifier** (a 1B model writes, a 9B model checks or takes the hard
  tokens). The larger model must still read every token; on an M2 Pro the result is slower than
  running the 4B.
- **A certified MLP.** Bounds on a channel's contribution from a partial dot product are too loose to
  prove anything; SiLU has no exact zeros. Sieve-FFN is Bounded by nature.
- **Sparse or approximate prefill** (token pruning, masks per tile). Either lossy with no certificate
  or no saving (section 4.2).
- **Fusing dispatches the way CHANGELOG 1.3.0 tried.** Same output, 3.5 → 6.6 ms. E1–E3 are narrower
  variants and must each be measured alone.
- **The Neural Engine.** Reachable only through Core ML with static shapes; no path to this engine's
  quantized weights and KV cache.

## 14. Sources and what was not verified

From this repository (measured, M2 Pro): README "Performance" and "The 7B ceiling"; CHANGELOG 1.1.1,
1.2.0, 1.3.0, 1.5.0; docs/sparse-decode-attention.md.

External, read through web search summaries or a fetched page on 2026-11-10:

- mlx-lm pull request 990, "Native MTP speculative decoding (Qwen3.5/3.6 reference implementation)"
  (open when read): <https://github.com/ml-explore/mlx-lm/pull/990>. The head's structure, the
  reported acceptance rates and speed-ups, missing weights in some exports.
- Rapid-MLX issue 2175 (85% acceptance, 1.04–1.06× on a 9B):
  <https://github.com/raullenchai/Rapid-MLX/issues/2175>.
- Prox, "Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs",
  arXiv 2607.27591: <https://arxiv.org/abs/2607.27591>. Abstract only: two stages (proxy from input
  sparsity and quantized weights, then exact computation of the selected channels), 60–70% MLP
  sparsity, up to 1.99× decode, ten models.
- TEAL, "Training-Free Activation Sparsity in Large Language Models", arXiv 2408.14690:
  <https://arxiv.org/abs/2408.14690>. Abstract, recalled: 40–50% model-wide sparsity on 7B–70B models.
- CATS: <https://scalingintelligence.stanford.edu/blogs/cats>.
- "SVD Contextual Sparsity Predictors for Fast LLM Inference", arXiv 2603.14110.
- Qwen 3.5 model cards: <https://huggingface.co/Qwen/Qwen3.5-4B>, `…-2B`, `…-0.8B`.
- LFM2-8B-A1B: <https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts>.
- M-series bandwidth and prompt-processing figures:
  <https://www.hardware-corner.net/m5-pro-m5-max-local-llm-4x-faster-prefill-202061108/> and search
  summaries of other coverage.

Not verified, and to be checked before any number here is quoted elsewhere:

- **Nothing was run.** No kernel, no importer, no benchmark. Every projection is arithmetic on the
  README's numbers.
- **Qwen 3.5 configurations.** huggingface.co was not reachable while writing. Hidden sizes (1,024
  for the 0.8B, 2,560 for the 4B) and the vocabulary size (about 248K) are taken from this
  repository's CHANGELOG; MLP sizes and layer counts for the 2B and 4B are unknown here. They are in
  each model's `config.json`.
- **Benchmark scores** come from search summaries of the model cards, not from the cards.
- **The papers** were not read: arxiv.org was not reachable. Their figures are from abstracts as
  quoted by search results or from memory, and none reports on a 1B-class model as far as is known.
- **GPU peak rates and other Macs' bandwidths** are quoted figures, not measurements with this
  engine.
- **The cost model** is a three-parameter fit to seven rows of one table.
