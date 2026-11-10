# Sieve: a touch-sparse model architecture for MLX (specification)

**Status: a design for an architecture that does not exist yet. Nothing here is implemented, and
nothing was measured for this document.** It was written on 2026-11-10 without access to a Mac. Every
speed figure is either one of this repository's own measurements (README, CHANGELOG,
[sparse-decode-attention.md](sparse-decode-attention.md); M2 Pro, 16 GB, 4-bit weights) or arithmetic
on them, and is labelled. The numbers called "on paper" assume the operating points of
[section 6](#6-flagships-and-on-paper-operating-points) can be reached at acceptable quality; whether
they can is what [section 13](#13-plan-and-gates) tests first.
[Section 15](#15-sources-and-what-was-not-verified) lists what could not be verified.

Contents: [1 The idea](#1-the-idea-in-one-page) ·
[2 Terms](#2-terms-and-reference-setup) ·
[3 Cost model](#3-cost-model) ·
[4 Budgets](#4-budgets-what-each-target-demands) ·
[5 Architecture](#5-architecture) ·
[6 Flagships](#6-flagships-and-on-paper-operating-points) ·
[7 Container](#7-container-format) ·
[8 Importer](#8-importer) ·
[9 Engine](#9-engine-integration) ·
[10 Gates](#10-correctness-gates) ·
[11 Measuring](#11-measuring) ·
[12 Intelligence](#12-the-intelligence-target) ·
[13 Plan](#13-plan-and-gates) ·
[14 Risks](#14-risks-open-questions-and-what-was-ruled-out) ·
[15 Sources](#15-sources-and-what-was-not-verified)

## 1. The idea in one page

**The targets.** A model of about 1B parameters that

- **T1** decodes at 500 tok/s or more,
- **T2** prefills at 5,000 tok/s or more,
- **T3** scores above a 4B Qwen model on intelligence benchmarks,

built for MLX (**C1**) and taking its weights from existing open models with no retraining (**C2**).

**The thesis.** Today's engines pay for a model by its size: a decode step reads every weight, a
prefill pass multiplies every weight by every token. Sieve pays by what a token needs. A Sieve model
is a large store of weights that each token *touches* sparsely along five axes:

| Axis | Mechanism | What a token skips | Output |
|---|---|---|---|
| Vocabulary | **Head**: a sketch of the output projection proves which rows cannot be sampled | most vocabulary rows | unchanged |
| Width | **Experts**: each MLP's channels are grouped into experts with a fitted router | the experts this token does not use | changes, bounded |
| Depth | **Pyramid**: fitted lenses let a token leave the stack at an early layer; the layers above get its state by extrapolation | the upper layers, in decode and in prefill | changes, bounded |
| Time | **Stride**: several tokens per pass over the weights, drafted by the model's own multi-token head | a weight pass per token | unchanged |
| Context | sparse decode attention (already in the engine) | most of the KV cache | changes, bounded |

The cost of a token becomes a product of fractions:

    bytes per token  ≈ ( head · f_V  +  depth · ( attention + experts · MLP ) ) / tokens per pass
    FLOPs per prompt token ≈ 2 · depth · ( attention + experts · MLP )

**Import, not training.** With every fraction at 1 the model is its source, weight for weight. The
importer never changes a weight: it adds what the screens need (a sketch, routers, lenses), all fitted
in closed form by least squares or eigendecomposition on text the model reads or writes itself. No
loss function, no gradient step.

**Where that puts the targets, on paper.** Speeds below are this repository's measured rates times
the cost model's ratio at the operating point named (section 6). They are what the design aims at,
not results.

| Model | Source | Profile | Decode, M2 Pro | Prefill, M2 Pro | Decode, 500 GB/s Mac | Prefill, 10 TFLOPS Mac | Intelligence |
|---|---|---|---|---|---|---|---|
| Sieve-0.8B | Qwen3.5-0.8B | `fast` | 620 | 5,670 | 880 | 17,000 | its source's, less the profile's loss |
| Sieve-1B | Llama 3.2 1B | `fast` | 440 | 5,530 | 740 | 16,800 | its source's, less the profile's loss |
| Sieve-4B-A1 | Qwen3.5-4B | `max` | 260 | 900 | 450 | 2,700 | the 4B's, less the loss; above it only by spending speed ([12](#12-the-intelligence-target)) |
| Sieve-9B-A2 | Qwen3.5-9B | `max` | 130 | 470 | 280 | 1,400 | the 9B's, less the loss; beats the 4B if it loses under 3.4 MMLU-Pro points |

Read as targets:

- **T1 and T2 at 1B, on the reference M2 Pro: in range.** Qwen3.5-0.8B at 620 / 5,670 and Llama 3.2
  1B at 440 / 5,530 in `fast`, 610 / 6,580 in `max`. Today they run at 271 / 2,780 and 202 / 1,720.
- **T3 needs a source that is already better than the 4B.** An import is at best its source, and no
  1B-parameter open model is near a 4B Qwen ([4.3](#43-intelligence-a-1b-budget-not-a-1b-source)).
  The architecture's answer is to separate *stored* from *touched*: import a 4B or 9B model and
  touch it like a 1B. A model's **A-number** is the bytes it touches per token divided by the bytes
  of a dense 1B model (0.56 GB). "The 1B model" of T3 is then an A1 model.
- **All three at once** need the 4B or 9B import to decode at 500 and prefill at 5,000: on paper a
  Mac with about 600 GB/s and 18 effective TFLOPS. No single row does all three on an M2 Pro.

**What is a bet.** Each axis rests on a hypothesis with a number attached (section 13). Evidence today:

| Axis | Evidence | Risk |
|---|---|---|
| Head | The proof is elementary; how tight it is on real hidden states is unmeasured. | Low; one day decides it. |
| Stride | Reported gains of 4–57% in other MLX runtimes, acceptance 44–88%. | Medium; depends on a cheaper 2-token pass. |
| Experts | Channel-level sparsity of 40–70% is published for 7B+ models, expert-level routing for ReLU models; neither for 1B-class SiLU models. | Medium to high. |
| Pyramid, prefill | A published training-free method reports 2.2× on larger models; `fast` asks for a mean depth of 0.4, which is 2.5× from depth alone, on smaller ones. | High. |
| Pyramid, decode | Early exit is published mostly with training; exits through fitted lenses on untouched weights are untested. | Highest. |
| All together | None. | The compounding of four approximations is the central unknown. |

**Decisions this needs from the owner** (recommendations in brackets):

1. **Is "1B" the stored size or the touched size?** T3 is only coherent for touched size. [Both:
   Sieve-0.8B and Sieve-1B carry T1 and T2; an A1 import of a larger model carries T3.]
2. **Is an output-changing, opt-in mode acceptable?** Everything except Head and Stride changes
   output. [Yes, with sparse decode attention's treatment: off means untouched, the lossless profile
   stays token-identical, the divergence of the others is measured and published.]
3. **Which Mac is the reference?** [Keep the M2 Pro as the measured baseline and state every target
   per machine with the formulas of section 3.]
4. **A research harness outside this repository.** Four of the six hypotheses can be tested in days
   with mlx-lm in Python, against weeks of Metal kernels. AGENTS.md keeps Python out of this
   repository. [A separate scratch repository for phase 0 only; nothing from it ships.]

## 2. Terms and reference setup

- **1B-class model**: 0.8–1.25B parameters, 0.42–0.73 GB at MLX 4-bit. In this repository: Qwen3.5
  0.8B (424 MB), Llama 3.2 1B (695 MB), Gemma 3 1B (732 MB).
- **Body**: every weight except the embedding / output projection. **Head**: the output projection
  (`lm_head`, or the tied `embed_tokens`).
- **Touched bytes**: weight bytes a decode step reads. **A-number**: touched bytes per emitted token
  divided by 0.5625 GB, the size of 1B weights at 4 bits. A dense Llama 3.2 1B is A1.2; a dense
  Qwen3.5 4B is A5.4.
- **Reference Mac**: M2 Pro, 16-core GPU, 16 GB. Measured in this repository: 197 GB/s with a plain
  read kernel (README), about 3.5 TFLOPS on the 64×64 quantized GEMM tile (README, CHANGELOG 1.3.0).
- **Bytes per weight**: 0.5625 for MLX 4-bit with group size 64 and f16 scales and biases
  (4 bits + 32 bits per 64 weights). Model sizes computed this way match the README within 1 MB.
- **Fidelity classes**, used for every mechanism:
  - **Exact**: output is identical to the engine without Sieve (greedy token-identical; the GPU
    sampler draws the same token for the same seed).
  - **Bounded**: output differs; the divergence is measured the way sparse decode attention's is (KL,
    text NLL, top-1 agreement) and must pass the gates in [section 10](#10-correctness-gates).
- **Fitted**: computed by the importer in closed form (least squares, eigendecomposition, k-means)
  from the source model's own activations. Never by gradient descent, and never by changing a source
  weight.

## 3. Cost model

### 3.1 Decode

A decode step reads every weight it uses once, so its time is close to

    t_step = B / (η · BW) + c0 + c1 · L

with B the touched bytes, BW the memory bandwidth, L the layers that run, η the fraction of the
bandwidth the matvecs reach, c0 a fixed cost per token (embedding, token selection, command buffer)
and c1 a cost per layer that does not depend on weight size (about nine dependent dispatches per
layer: q/k/v, post-projection, attention, o, add + norm, gate/up, activation, down, add + norm).

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
  everything else. Cutting bytes alone cannot double its speed; cutting the layers that run cuts
  both terms, which is why depth is an axis.
- **The constants are a fit, not a measurement.** Three parameters, seven points, one machine, and
  the hybrid Qwen 3.5 layers have a different dispatch count from the others. `mlxr bench` prints the
  real split (`decode step @16: … without qkv … lmhead …`, `tiny dependent dispatch`, `lm_head matvec`).
  Replacing η, c0 and c1 with those lines is the first task on a Mac.

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

(c_d = 0.05 per drafted token.) Multi-token decode pays with a nearly free drafter, an acceptance
above about 0.6, and a 2-token pass much nearer 1.2 than 1.5.

### 3.3 Prefill

A dense prefill pass multiplies every body weight by every token: 2 · P_body floating-point
operations per token. With an effective GEMM rate F,

    prefill tok/s ≈ F / (2 · P_body)

The README's prefill rates at 512 tokens imply F = 3.06 TFLOPS (Qwen2.5 0.5B, 0.358B body, 4,270
tok/s), 2.66 (Qwen3 0.6B, 0.440B, 3,020) and 3.35 (Llama 3.2 1B, 0.973B, 1,720), against the 3.5
TFLOPS the 64×64 tile measures on its own. This document uses **F = 3.3 TFLOPS** for the M2 Pro.

## 4. Budgets: what each target demands

### 4.1 Decode: 500 tok/s is a byte budget

500 tok/s is 2.0 ms per token. The bytes a single-token step may touch, by the layers it runs:

| Mac | Layers run | c1 = 46 µs (today) | c1 = 23 µs (halved) |
|---|---|---|---|
| M2 Pro, 197 GB/s | 8 | 248 MB | 281 MB |
| M2 Pro | 16 | 182 MB | 248 MB |
| M2 Pro | 24 | 116 MB | 215 MB |
| 500 GB/s (assumed), same c0 and c1 | 8 | 629 MB | 713 MB |
| 500 GB/s | 16 | 462 MB | 629 MB |
| 500 GB/s | 24 | 294 MB | 545 MB |

A 1B model is 560–700 MB at 4 bits. On an M2 Pro a token may touch about **a third of it**, which no
single mechanism delivers and the product of three does: a quarter of the head, 60% of the depth, 45%
of the MLP in the layers that run.

Bandwidth figures reported for current chips: 546 GB/s (M4 Max, 40-core GPU), 614 GB/s (M5 Max,
40-core), 460 GB/s (M5 Max, 32-core). None was measured with this engine, and c0 and c1 on those GPUs
are unknown.

### 4.2 Prefill: 5,000 tok/s is an arithmetic budget

5,000 tok/s needs F ≥ 10,000 × the body weights each prompt token is multiplied by:

| Effective GEMM rate F | Weights a prompt token may be multiplied by |
|---|---|
| 3.3 TFLOPS (M2 Pro, this engine) | 0.33B |
| 7.5 TFLOPS | 0.75B |
| 10 TFLOPS | 1.00B |
| 18 TFLOPS | 1.80B |

- Dense, Llama 3.2 1B (0.97B body) would need 9.7 TFLOPS: 2.9 times what this engine reaches on an
  M2 Pro, and above the 5.7–6.8 TFLOPS usually quoted as that GPU's peak (quoted, not verified). **So
  exact prefill cannot meet T2 on this Mac.** The only lossless saving is the last layer's MLP and
  output projection for non-final prompt tokens, about 1 / L.
- **A prompt token does not need the whole body.** What later tokens use of it is its key and value
  at each layer. Pyramid gives most prompt tokens a few layers and extrapolates the rest; Experts
  gives each token only the experts it routes to, grouped so the GEMM tiles stay dense. At a mean
  depth of 0.4 and 40% of the experts, a prompt token of Llama 3.2 1B is multiplied by about 0.22B
  weights: inside the M2 Pro's budget.

### 4.3 Intelligence: a 1B budget, not a 1B source

Scores from the Qwen 3.5 model cards, as returned by web search summaries (vendor-reported, not
verified against the cards; where a card gives two rows, both are shown, presumably without and with
thinking):

| Model | MMLU-Pro | GPQA Diamond | IFEval |
|---|---|---|---|
| Qwen3.5-9B | 82.5 | 81.7 | |
| **Qwen3.5-4B** | **79.1** | **76.2** | **89.8** |
| Qwen3.5-2B | 55.3 / 66.5 | 51.6 | 61.2 / 78.6 |
| Qwen3.5-0.8B | 29.7 / 42.3 | 11.9 | 52.1 / 44.0 |

- **An import is at best its source.** No fitted screen adds knowledge. With thinking, Qwen3.5-0.8B
  trails the 4B by 37 MMLU-Pro points and 64 GPQA points; the 2B by 13 and 25. No 1B-parameter open
  model was found that is claimed to approach the 4B. **A 1B source cannot meet T3, with any
  architecture.**
- **So T3 is a statement about touched size.** A source that already beats the 4B (the 9B, by 3.4
  MMLU-Pro points and 5.5 GPQA points) or equals it (the 4B itself) has to run on a 1B model's
  budget of bytes and arithmetic, and keep its lead. That is what the five axes are for, and it is
  the hardest claim in this document: an A1 import of a 9B model touches about a tenth of it per
  token.
- **Speed can be spent on answers.** A 4B import at four times its dense speed can answer three
  times and vote in less time than the dense 4B answers once. That is the one route by which an
  import can score above its own source at equal wall-clock time
  ([12](#12-the-intelligence-target)).

## 5. Architecture

### 5.1 Definition, contract and profiles

A **Sieve model** is a decoder-only transformer in which every stage that reads weights is evaluated
in two steps:

1. a **screen** that reads a small, fitted summary and decides what this token needs (which
   vocabulary rows, which experts, how many layers);
2. an **exact** step that evaluates what passed with the original weights and the original arithmetic.

**The import contract.** The dense limit (everything passes) is the source architecture, weight for
weight. The importer adds screens; it never changes a weight. Attention and Gated DeltaNet mixers,
normalisation, RoPE, tokenizers, chat templates, the KV cache and the sampler are the source's.

Profiles name a quality target, not a sparsity level. The importer finds, per model, the operating
point that just meets each one ([8](#8-importer)).

| Profile | Mechanisms | Class | Target | Default |
|---|---|---|---|---|
| `off` | none: the engine as it is today | | | yes |
| `exact` | Head, Stride | Exact | identical output | candidate default once proven |
| `balanced` | `exact` + Experts + Pyramid | Bounded | text NLL within +0.02 nats | opt-in |
| `fast` | the same, further | Bounded | text NLL within +0.05 nats | opt-in |
| `max` | the same, as far as the model's benchmark score allows | Bounded | a stated loss on the evaluation suite | opt-in |

### 5.2 Head: the vocabulary axis

The head computes z_i = w_i · x for every vocabulary row w_i (V rows of d values) and the final
normalised hidden state x. It is 20–35% of a 1B-class model's bytes: 148 MB of 695 for Llama 3.2 1B,
about 143 MB of 424 for Qwen3.5 0.8B, 170 MB for Gemma 3 1B. The sampler then uses very few of the V
logits: the argmax when greedy, otherwise the exact top k (`sample_token` builds its pool from the top
`top_k` logits, or the top 512 when top-k is off, capped at `SMP_CAP`).

Head computes exactly those logits and proves that no other row can be among them.

**Fitted by the importer:**
- μ, the mean final hidden state over a calibration text (d values);
- Q, an orthonormal basis of the r directions that hold most of the variance of x − μ over that text
  (r × d; expected d / 16 to d / 4);
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
  The importer measures the expected cost and leaves Head off for a model when it is above 80% of
  the dense head.
- **Deterministic.** Blocks are selected by a comparison each threadgroup makes for itself. No
  atomics order any floating-point sum.
- **ε** covers float rounding in the screen and in the dense logit, and the rounding of the stored
  Q. The importer measures both and checks, on held-out calibration states, that every dense top-k
  row is inside the certified set. One violation aborts the import.

Where it does not apply: passes that need every logit (`allLogits`, teacher-forced scoring) and CPU
sampling over a constrained token set (tool-call names) use the dense head. Gemma's final-logit
soft-cap is monotone, so it is applied to bounds and logits alike.

**Why a basis of the hidden states, not of the weights.** A low-rank sketch of W alone would leave an
error of ‖w_i‖ · ‖x‖ scale in every logit, far wider than the gaps between the top tokens. With a
basis fitted to x, the unseen part is n_i · ρ, and ρ is small exactly when x lies in the subspace it
usually lies in. Whether ρ is small enough at a rank that is cheap to scan is hypothesis **H1**. If it
fails, the fallback is a fixed-budget screen (top blocks by ẑ, no certificate), which is Bounded.

**Head is what makes the other axes affordable.** A Pyramid exit test and a Stride draft each need
the top of a logit vector. With a dense head, two exit tests would cost more than the layers they
save on a 1B-class model. With Head they cost a quarter of that.

Projected touched bytes with r = d / 8 and an 8-bit sketch:

| Model | Dense head | Head | Saved per evaluation on an M2 Pro |
|---|---|---|---|
| Qwen3 0.6B | 88 MB | 23 MB | 0.36 ms |
| Llama 3.2 1B | 148 MB | 39 MB | 0.61 ms |
| Qwen3.5 0.8B (V ≈ 248K) | 143 MB | 36 MB | 0.60 ms |
| Gemma 3 1B | 170 MB | 43 MB | 0.72 ms |
| Qwen3.5 4B (V ≈ 248K) | 357 MB | 89 MB | 1.51 ms |

### 5.3 Experts: the width axis

A gated MLP computes y = D · (act(G x) ⊙ (U x)): I channels, each with a gate row g_i and an up row
u_i (d values each) and a column d_i of the down projection. Channel i contributes c_i = a_i · d_i
with a_i = act(g_i · x)(u_i · x). The MLP is 45–65% of a 1B-class model's bytes (453 MB of 695 for
Llama 3.2 1B), and a token's contributions are heavy-tailed: a minority of channels carries most of
‖y‖.

Published training-free methods pick *channels* per token (TEAL, Prox; section 15). That saves bytes
in decode and nothing in prefill, because every token of a batch picks different channels and the
GEMM tiles fill up again. Sieve picks **experts**: fixed groups of channels that tend to fire
together. A token routes to a few experts; a batch groups its tokens by expert, and every tile stays
dense. One mechanism then serves decode (bytes) and prefill (arithmetic).

**Fitted by the importer**, per layer:
- **Experts.** The I channels are partitioned into E = I / G groups of G channels (G = 64 by
  default) by balanced k-means on their contribution patterns over the calibration text, so that
  channels that fire on the same tokens share an expert.
- **Router.** A matrix R (E × d) and a bias, fitted by ridge regression from the MLP's normalised
  input x to the logarithm of each expert's contribution energy. If the linear router's held-out
  recall is short, the fallback is **probe channels**: the few channels of each expert that best
  predict its energy are evaluated exactly for every token (and reused if the expert is kept).
- Per profile, a threshold τ: expert e is kept when its routed score reaches τ.

**At load** (in the Bounded profiles only): the gate and up rows are reordered by expert in place
(`Linear.permuteRows`; rows are independent), and the down projection is re-laid as **Dᵀ**: its 4-bit
codes row-major by channel, in expert order, with a table from each channel to its original
quantization group. Same codes, same scales, same biases: only the byte order changes. The sidecar
holds the permutation and the router, not a second copy of the weights.

**Per decode step:**

1. `expert_route`: scores = R x + bias; one keep bit per expert.
2. `expert_exact`: the kept experts' gate and up rows (contiguous row ranges) with `qmv`'s row
   arithmetic, and a_i = act(g) · u in the epilogue (the same expression as the `silu` kernel).
3. `expert_down`: y = Σ over kept channels of a_i · Dᵀ[i]. The kernel walks the original
   quantization groups, so each group's scales and biases are read once, and takes the kept channels
   of the group in ascending order. Split across threadgroups into partial sums.
4. `add_combine`: h += the partial sums in fixed order, and the next norm (as `add_rms_norm` does now).

**Per prefill batch:** route every token; for each expert, the list of tokens that keep it; a GEMM of
that expert's rows against those tokens (the existing tiles over a token index list), the activation,
and the same for Dᵀ. A token is multiplied only by the experts it keeps.

Properties:
- **Bounded.** With every expert kept, the result equals the dense MLP up to the summation order of
  the down projection (`expert_down` sums by channel, `qmv` by lane). It is not bit-identical to
  today's kernels even then, so **the `exact` profile does not use Experts.**
- **Deterministic.** Keep bits are threshold comparisons, and the sums run in a fixed order.
- **Thresholds, not counts.** A token gets as many experts as it needs. The worst case is all of
  them: a dense MLP plus the router.
- **Cheap screen.** The router is 0.5 MB per layer for Llama 3.2 1B, against 28.3 MB for the MLP.

Touched bytes at 40% of the experts kept: 40% of the gate, up and Dᵀ rows, all of Dᵀ's scales and
biases (3.7% of the MLP: they are per group of 64 channels and per output, so every group is touched
whatever is kept) and the router (2%): about 44% of the MLP.

Whether 35–50% of the experts carry enough of the output on 1B-class SiLU models, and whether a
fitted router finds them, is hypothesis **H2**. Channel-level selection inside the kept experts is a
decode-only refinement ([14.2](#142-open-questions)).

Activations: SiLU (Llama, Qwen, Mistral, Phi) and GELU (Gemma). Phi-3's fused `gate_up` and Gemma 4's
per-layer inputs do not change the scheme.

### 5.4 Pyramid: the depth axis

Every token runs every layer today. Pyramid lets a token stop at an **exit** partway up the stack. It
has three parts: lenses that read a prediction out of a middle layer, a rule for leaving, and a way to
keep the layers above consistent for the tokens that come later.

**Lenses, fitted by the importer.** For each exit layer e (two or three per model, at full-attention
layers), a lens predicts the final residual from the residual after layer e:

    ĥ = h_e + U_e (V_eᵀ h_e) + c_e          U_e, V_e: d × r, r about 128

fitted by reduced-rank ridge regression of (h_L − h_e) on h_e over the calibration text. The lens
output goes through the model's own final norm and Head. No new output layer is trained: the model's
own head reads a linearly corrected middle state.

**Frozen state.** A token that leaves at e never runs the layers above. Those layers still need it:
their attention must be able to look back at its key and value, and a Gated DeltaNet layer must
advance its recurrent state through it. For a layer l above e, the token's input is extrapolated as

    h̃_l = h_e + β(e, l) · (ĥ − h_e)         β fitted, one scalar per (e, l)

and the layer computes only what later tokens need from it: its input norm, the k and v projections,
`qkv_post` (per-head norm, RoPE, cache append), or for a linear-attention layer its projections,
convolution and state recurrence. No q, no attention, no output projection, no MLP. For Llama 3.2 1B
that is 3.5% of a layer's weights.

The fill is **lazy and batched**: positions wait in a ring of at most 64 (h_e and ĥ, 2d floats each)
and a layer fills all of them in one small batch the next time a token actually reaches it. A layer
no later token reaches is never filled.

**Decode.** After the layers up to exit e, the step runs an exit test: the lens, the final norm, and
Head with k = 2. The token leaves when the margin between the lens's top two logits reaches m_e; it
is then selected from the lens logits by the ordinary sampler (when sampling, Head first certifies
the sampler's own k for that step). Otherwise the step continues to the next exit or the top.

- m_e is calibrated, not chosen: the smallest margin at which, on held-out text, the lens's argmax
  equals the full model's for at least 1 − δ of the positions that pass (δ = 1% for `balanced`, 2%
  for `fast`). The share of tokens that leave at each exit, and so the mean depth, is whatever the
  model gives at that δ.
- A token that leaves costs the layers below its exit and one Head evaluation. A token that reaches
  the top costs every layer and up to three Head evaluations, still less than one dense head.
- Per-layer overhead (c1 in section 3.1) falls with the mean depth, which is half the step on small
  models.
- Control flow: the exit decision is read by the CPU before the next segment is encoded (one small
  readback per exit test, which gives up part of the pipelined decode's overlap). Indirect dispatch,
  where the exit kernel writes the next segment's threadgroup counts, keeps everything on the GPU
  and is the second step ([14.2](#142-open-questions)).

**Prefill.** A prompt token needs no prediction at all, only its keys, values and states. Prefill
runs in stages bounded by the same exits:

1. Every token runs the layers up to the first exit.
2. `key_score`: each position is scored by the attention it receives at that layer from the last 64
   positions of the batch.
3. The top fraction ρ_e by score continues, together with the last 64 positions and the first 4.
   The rest freeze at e and are filled above as frozen states, in batch.
4. Repeat at each exit. The last position always reaches the top and yields the first logits.

A token that freezes at a quarter of the depth costs a quarter of the stack plus the fills. The
fractions ρ_e are found by the importer against the profile's target ([8](#8-importer)). Attention in
the upper layers shrinks with them: fewer queries, the same keys.

Properties and limits:
- **Bounded**, in both phases. A wrong early exit is a wrong token; an extrapolated key is an
  approximate memory of that position for the layers above.
- **The dense limit is the source**: m_e = ∞ and ρ_e = 1 run every layer for every token.
- **Deterministic**: exits and kept sets are threshold comparisons and exact selections, with ties
  to the lower position.
- Hybrid models gain less from each skipped layer: a frozen token still pays a linear-attention
  layer's projections (batched), which this document assumes to be 70% of that layer's mixer weights.
- Sliding-window layers, KV-shared layers and sparse decode attention work unchanged on filled rows.
- Chat reuse: a frozen row is as reusable as an exact one. The KV prefix rule and Qwen 3.5's state
  checkpoint are unchanged.

Whether fitted lenses on untouched weights agree with the full model often enough (**H4**) and
whether extrapolated keys are good enough memories (**H5**) are the two riskiest hypotheses here.
Related published work: early exit with copied states (CALM, LayerSkip, both with training), linear
read-outs of middle layers (the tuned lens, trained by gradient descent), and training-free pruning
of keys in deeper layers during prefill (PyramidInfer, 2.2×).

### 5.5 Stride: the time axis

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

**What makes it pay here** (reported elsewhere: +57% on a 27B, +11% on a 9B, +4–6% on a 9B on another
machine, and a loss on fp16 weights):
- the draft needs a head pass, which Head makes cheap (36 MB instead of 143 on Qwen3.5 0.8B);
- the 2-token pass reads each weight once (`qmvt4`), at 1.5 steps today. The table in section 3.2
  says the target is 1.2 to 1.3.

**Recurrent state.** A T-token pass computes the state after each token in turn. Stride keeps the
state after every token of the pass in its own buffer (two buffers at depth 1) and continues from the
one after the last accepted token. No snapshot copy, no replay. Memory: one extra state per drafted
token (`LinearAttention.stateBytes` per linear layer).

**With Pyramid.** A pass carries rows, and each row may leave at an exit on its own; the pass
continues with the rows that remain. In `exact`, every row runs the full depth and the acceptance
rule is today's.

**Constraints and risks.**
- Not every export has the weights: `mlx-community/Qwen3.5-27B-4bit` is reported to declare the head
  and ship none. The importer checks for the tensors.
- Reported acceptance varies widely: 44% to 88% in most reports, far lower in some quantized
  exports, and the reports on quantizing the head conflict. Acceptance is measured per export at
  import, and Stride is enabled only where the table in section 3.2 predicts a gain (**H3**).
- The head's transformer layer attends over earlier positions, so it needs K/V rows of its own for
  every position. Either it runs on every token, prefill included (about 1 / L more work), or it
  follows whatever the reference implementation does. Settle this against mlx-lm's implementation,
  and check the head's logits against it, before building the kernels.
- The cycle needs the accept decision before the next pass is encoded, so it gives up the pipelined
  decode's overlap (+4–10% in CHANGELOG 1.1.1) unless positions move to a GPU buffer.
- Sampled output: the existing exact acceptance rule keeps the target distribution.

Models without such a head get Stride only through prompt lookup (`NGramDraft`), which is already
there and pays on copy-heavy text only.

### 5.6 Context

Sparse decode attention is unchanged. It keeps its own switch, budget and threshold (5,128 tokens)
and composes with every profile. It is the same idea applied to the KV cache: score block means,
select, attend exactly.

### 5.7 Per-layer overhead

Section 3.1 puts 46 µs per layer outside the weight reads: about 1.3 ms of Qwen3 0.6B's 3.5 ms step.
Pyramid removes it for the layers a token skips. For the layers that run, Experts and the exit tests
add dispatches. Three experiments, each kept only if `mlxr bench` shows a gain (AGENTS.md):

- **E1: norm by partial sums.** The matvec that produces a residual update adds it to h in its
  epilogue and writes one partial sum of squares per threadgroup. The next matvec's threadgroups sum
  those partials (a few hundred floats) and scale x as they load it. Removes both `add_rms_norm`
  dispatches of a layer. This differs from the fusion CHANGELOG 1.3.0 rejected, where every
  threadgroup recomputed the whole norm. It changes the norm's rounding, so it is **not
  token-identical** and may only ship inside the Bounded profiles.
- **E2: activation in the exact stage.** Already part of `expert_exact`. The dense equivalent measured
  4–11% slower (CHANGELOG 1.2.0), so the masked form must be re-measured, not assumed.
- **E3: paired layers.** Two adjacent middle layers read the same input and their outputs are added,
  so their matvecs fuse into one dispatch each. Published work reports that middle layers tolerate
  this; it is Bounded and untested here.

### 5.8 How the axes compose

- Head serves the top of the stack, every exit test and every Stride draft.
- Experts applies inside each layer a token runs, in decode and in prefill.
- Pyramid decides which layers a token runs; the layers it skips are filled, not run.
- Stride divides whatever a pass costs by the tokens it yields.
- The importer treats the Bounded axes as one budget: it moves each profile's NLL allowance between
  Experts thresholds, exit margins and prefill fractions towards whichever saves the most bytes per
  nat lost.

### 5.9 Native Sieve models (out of scope)

A model trained for this architecture could make the screens part of itself: MLPs whose inactive
experts are exactly zero, exits trained to be right, a hierarchical output layer. It would touch far
fewer bytes at the same quality than an import can. It is not specified here, because C2 excludes
training. The container and the kernels would serve it unchanged.

## 6. Flagships and on-paper operating points

The operating points below are design assumptions, chosen to show what each profile is aiming at. The
importer will find the real ones, model by model.

| Profile | Experts kept | Decode: tokens leaving at the first / second exit (at depth) | Mean decode depth | Mean prefill depth |
|---|---|---|---|---|
| `balanced` | 50% | 25% at 0.50, 25% at 0.75 | 0.81 | 0.70 |
| `fast` | 40% | 45% at 0.375, 25% at 0.625 | 0.63 | 0.40 |
| `max` | 35% | 60% at 0.25, 25% at 0.50 | 0.43 | 0.35 |

Decode, tok/s. M2 Pro rows are the README's measured dense rate times the cost model's ratio; the 500
GB/s rows are the model alone (same c0 and c1 assumed, nothing measured). Touched bytes per token and
the A-number are for `max`.

| Model | Dense | `balanced` | `fast` | `max` | `max` touches |
|---|---|---|---|---|---|
| **M2 Pro** | | | | | |
| Qwen3 0.6B | 286 | 390 | 500 | 680 | 102 MB, A0.2 |
| Qwen3.5 0.8B | 271 | 490 | 620 | 830 | 104 MB, A0.2 |
| Llama 3.2 1B | 202 | 330 | 440 | 610 | 176 MB, A0.3 |
| Qwen3.5 4B | 62 | 130 | 180 | 260 | 666 MB, A1.2 |
| A 9B model (size assumed) | 30 (model) | 64 | 88 | 130 | 1.2 GB, A2.2 |
| **500 GB/s, assumed** | | | | | |
| Qwen3 0.6B | 440 (model) | 550 | 690 | 920 | |
| Qwen3.5 0.8B | 437 (model) | 710 | 880 | 1,160 | |
| Llama 3.2 1B | 398 (model) | 580 | 740 | 980 | |
| Qwen3.5 4B | 119 (model) | 230 | 315 | 450 | |
| A 9B model (size assumed) | 69 (model) | 140 | 190 | 280 | |

Prefill, tok/s on an M2 Pro (multiply by F / 3.3 for a GPU with effective rate F):

| Model | Dense (README) | `balanced` | `fast` | `max` |
|---|---|---|---|---|
| Qwen3 0.6B | 3,020 | 4,350 | 6,990 | 7,850 |
| Qwen3.5 0.8B | 2,780 | 3,910 | 5,670 | 6,180 |
| Llama 3.2 1B | 1,720 | 2,980 | 5,530 | 6,580 |
| Qwen3.5 4B | 403 | 570 | 820 | 900 |

Assumptions behind both tables, all to be replaced by measurements:
- Head touches 26% of the head per evaluation; an exit test costs one evaluation and 0.05 ms of
  synchronisation (assumed).
- Experts touches the kept rows, all of Dᵀ's scales and biases, and the router.
- Qwen 3.5 bodies are split 35% mixer and attention, 65% MLP (assumed: their configurations were not
  available), with 24 and 32 layers; the 9B's size (5.6 GB), head (0.5 GB) and 40 layers are assumed.
- Qwen 3.5 imports include Stride at 1.25× (an acceptance near 0.75 with a 2-token pass near 1.3).
- Prefill: GEMMs are 85% of today's prefill time; tiles over token subsets run at 80% of today's
  efficiency (assumed; small batches are slower today, see README "Limitations").
- Extra dispatches for the router and exits are not counted beyond the synchronisation cost.

What the tables say:

- **Sieve-0.8B** (Qwen3.5-0.8B, `fast`): 620 tok/s decode and 5,670 tok/s prefill on an M2 Pro. T1
  and T2, on the reference Mac, with the smallest model that has a multi-token head.
- **Sieve-1B** (Llama 3.2 1B): 440 / 5,530 in `fast`, 610 / 6,580 in `max`. The plain-transformer
  control: no Stride, the largest MLP share.
- **Sieve-4B-A1** (Qwen3.5-4B, `max`): a 4B model touching 666 MB per token, 260 tok/s on an M2 Pro
  against 62 dense. T1 at about 600 GB/s; T2 at about 18 effective TFLOPS.
- **Sieve-9B-A2** (Qwen3.5-9B, `max`): the T3 candidate that needs no voting. 130 tok/s on an M2 Pro
  on paper, against a 4B's 62.

## 7. Container format

A Sieve model is the original MLX model directory plus two files. The original files are never
modified.

    <model>/
      config.json, model*.safetensors, tokenizer files     the MLX export, unchanged
      sieve.json                                           manifest
      sieve.safetensors                                    fitted screens

`sieve.json`:

    {
      "format": "mlxr-sieve", "version": 1,
      "source": { "config_sha256": "…", "weights": [ { "file": "model.safetensors", "sha256": "…" } ] },
      "calibration": { "kind": "self-generated" | "file", "tokens": 131072, "seed": 42, "sha256": "…" },
      "head": { "enabled": true, "rank": 256, "block": 64, "sketch_bits": 8, "epsilon": 0.002,
                "expected_fraction": 0.26 },
      "experts": { "group": 64, "router": "linear",
                   "layers": [ { "tau": { "balanced": 0.031, "fast": 0.074, "max": 0.11 } } ] },
      "pyramid": { "exits": [ { "layer": 6, "rank": 128,
                                "margin": { "balanced": 4.1, "fast": 2.9, "max": 1.8 },
                                "keep":   { "balanced": 0.8, "fast": 0.45, "max": 0.3 } } ] },
      "stride": { "mtp": true, "depth": 1, "acceptance": 0.71 },
      "profiles": { "balanced": { "nll_delta": 0.018, "kl": 0.011, "experts_kept": 0.52,
                                  "decode_depth": 0.83, "prefill_depth": 0.71 } }
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
| `experts.<l>.order` | [I] | u16 | channel order by expert (a permutation) |
| `experts.<l>.router.weight`, `.bias` | [E, d], [E] | f16 | R and its bias |
| `pyramid.<e>.u`, `.v`, `.c` | [d, r], [d, r], [d] | f16 | the lens of exit e |
| `pyramid.<e>.beta` | [L] | f32 | β(e, l) for the layers above e |

The sidecar is small: the head sketch (36–90 MB for the models above) dominates; routers and lenses
are a few megabytes. Dᵀ and the reordered rows are built at load from the original weights.

Rules for the loader (AGENTS.md, untrusted input; the engine builds with `-Ounchecked`):
- The manifest's format, version and source hashes must match; a sidecar for other weights is ignored
  with a note, never used.
- Every shape is checked against `config.json` before any buffer is bound: r ≤ d and a multiple of
  32, block ∈ {16, 32, 64}, layer and exit indices, I, d, E · G = I, group sizes.
- `experts.<l>.order` must be a permutation of 0 ..< I: checked in full before any row moves.
- Every scalar is finite and in range: τ ≥ 0, margins ≥ 0, 0 < keep ≤ 1, 0 ≤ β ≤ 1, ε ≥ 0, bounds
  non-negative.
- Nothing else in the sidecar is an index. Lists of rows, tokens and positions are produced at run
  time by kernels from host-checked sizes.
- A sidecar that fails any check loads the model with Sieve off and says why
  (`LoadedModelInfo.sieveNote`), as a sparse index that does not fit does today.

The files use MLX's own conventions (safetensors, MLX affine quantization), so a reference
implementation in mlx-lm can read them.

## 8. Importer

A new `sieve` command writes the sidecar: `sieve import <model>`, with `--calib <text file>` and
`--profiles balanced,fast,max` as options. Swift only; linear algebra through Accelerate (LAPACK
symmetric eigendecomposition and least squares, BLAS products). Nothing is trained: there is no loss
function and no gradient.

(Commands that do not exist yet are written without the `mlxr` prefix throughout this document. The
documentation tests parse every command line under `docs/` that starts with `mlxr` against the real
command table.)

1. **Check the source.** A supported architecture, 4-bit weights with group sizes the kernels take.
   Record the hashes.
2. **Calibration text.** A text file given with `--calib` (at least about 130K tokens), or by default
   text the model writes itself: sequences of 512 tokens sampled at temperature 1 from the chat
   template with a fixed seed. Split into a fitting part and a held-out part.
3. **Collect.** One dense pass over the fitting part, recording per layer the MLP input, a sample of
   channel contributions, the residual after each candidate exit layer, the final residual and the
   final normalised hidden state.
4. **Head.** μ and the covariance of the final hidden states; Q for r in {d/16, d/8, d/4}; sketches
   at 8 and 4 bits with q and n; on the held-out part, the rows the certificate keeps for k = 1, 2
   and 64. Keep the cheapest combination, or disable Head if none is under 80% of the dense head.
   Check that every held-out dense top-k row is inside the certified set.
5. **Experts.** Per layer: balanced k-means of channels on contribution patterns; the router by ridge
   regression; its held-out recall of the experts that carry 95% of the energy. Fall back to probe
   channels where recall is short.
6. **Pyramid.** Candidate exits at full-attention layers near 1/4, 1/2 and 3/4 of the depth. Per
   exit: the lens by reduced-rank ridge regression; β(e, l) by scalar least squares; the margin–
   agreement curve on the held-out part.
7. **Budget.** For each requested profile, coordinate descent on the held-out part over the Experts
   thresholds, the exit margins and the prefill fractions: move the profile's NLL allowance to
   whichever knob saves the most bytes (decode) or arithmetic (prefill) per nat, until the target is
   just met. Decode and prefill are scored separately: decode by the text's NLL under teacher
   forcing with exits and frozen states active; prefill by the NLL of a continuation after a
   Pyramid-prefilled prompt and by passkey recall at several depths.
8. **Stride.** If the weights hold `mtp.*` tensors, measure greedy acceptance on the held-out part.
9. **Self-check, then write.** Head against the dense head (identical tokens on the held-out part);
   every kernel at full budget against its dense reference; the profile metrics on a second text.
   Write the files atomically. An import that fails a check writes nothing.

Expected run time on a 1B-class model: tens of minutes, dominated by step 7's repeated passes
(estimate, not measured).

## 9. Engine integration

Files (planned):

- `Sources/MLXEngine/Sieve.swift`: `SieveProfile`, manifest parsing and validation, environment
  parsing, memory accounting.
- `Sources/MLXEngine/SieveImport.swift`: the importer.
- `Sources/MLXEngine/Kernels.swift`: `head_project`, `head_screen`, `head_cut`, `head_exact`;
  `expert_route`, `expert_exact`, `expert_down`, `add_combine`, the token-list GEMM tiles; `lens`,
  `exit_test`, `frozen_fill`, `key_score`; the multi-token-prediction head.
- `Sources/MLXEngine/GPU.swift`: parameter structs; `mtp.*` no longer skipped when Stride is on.
- `Sources/MLXEngine/Transformer.swift`: the sidecar buffers; `encode` split into segments between
  exits; the Experts branch of the MLP block; the pending ring and the lazy fills; staged prefill;
  per-token recurrent state outputs.
- `Sources/MLXEngine/Engine.swift`: `load(directory:draftDirectory:contextLength:sparse:sieve:)`, the
  memory check, the segment loop with exit readbacks, the Stride cycle next to the existing
  speculation loop, the `sieve…` fields of `LoadedModelInfo`.
- `Sources/MLXRCLI`, `Sources/mlxr/main.swift`: the `sieve` command (`import`, `info`, `verify`), the
  `--sieve` option, the bench section, the `eval` command. Each new command and option also needs its
  entry in the command table, help and completions, and the documentation tests then cover it.
- `Sources/MLXRunner`: an inspector setting, after the CLI path is proven.
- Tests: `SieveKernelTests.swift`, `SieveModelTests.swift`, `SieveManifestTests.swift`.

Switches, mirroring sparse attention:

| Where | Setting |
|---|---|
| CLI | `mlxr run` and `mlxr serve` take `--sieve` with `off`, `exact`, `balanced`, `fast` or `max` |
| Environment | `MLXR_SIEVE` (same values; `0` = off) |
| Code | `InferenceEngine.load(…, sieve: SieveProfile?)`; nil means off |
| App | inspector, applies at the next load |

A model without a sidecar loads as today whatever the setting, and says so once.

Kernels reuse what exists: `head_exact` and `expert_exact` are `qmv`'s row arithmetic over row ranges
(`qmvp` already walks blocks); `head_cut` is `sample_token`'s bisection on block maxima;
`frozen_fill` is the k/v half of the existing projection and `qkv_post` on a small batch; the staged
prefill is the existing tiles over a token index list.

## 10. Correctness gates

In addition to AGENTS.md's gates (`swift test`, token-identical greedy output on every existing
path, `prefillMatchesIncrementalDecode`):

- **Off means untouched.** Without `--sieve`, or with `off`, there are no Sieve buffers and no Sieve
  dispatches, and output is bit-identical to the build before Sieve. A model directory that holds a
  sidecar behaves the same as one that does not.
- **`exact` is exact.** On every installed model with a sidecar: greedy output
  (`--temp 0 --repeat-penalty 1 --seed 42 --max 256`) is token-identical to `off`; seeded sampled
  output is token-identical; the logits of every certified row are bit-identical
  (`sieveExactMatchesDense`). Stride on and off give the same tokens (`strideIsTokenIdentical`).
- **The dense limit is the source.** With every expert kept, no exit taken and every prompt token
  kept, a Bounded profile matches `off` within 1e-4 relative on every logit and gives the same argmax
  at every position of a test text that is not a near-tie (`sieveDenseLimitMatchesSource`). Head with
  θ = −∞ is bit-identical to the dense head.
- **Each kernel against a CPU reference**, including no expert kept, all kept, one kept, a tie at θ
  or at a threshold, non-finite hidden states, V not a multiple of the block, an empty and a full
  pending ring, and a prefill stage that keeps one token.
- **Certificate.** On random and adversarial hidden states (including x far outside the calibration
  subspace), no dense top-k row is outside the certified set (`sieveHeadCertificateHolds`).
- **Frozen rows are complete.** After any mix of exits, every layer's cache holds a row for every
  position before that layer is next read, and Qwen 3.5's recurrent state has advanced through every
  position exactly once (`sievePyramidStateIsComplete`).
- **Determinism.** The same prompt and seed give the same tokens across runs in every profile.
- **Untrusted sidecars.** Truncated files, wrong shapes, out-of-range scalars, a non-permutation in
  `order`, hashes that do not match, and a sidecar from another model each load with Sieve off and
  never reach a kernel.
- **Bounded profiles pass their targets** on the held-out calibration text and on a second text the
  importer did not see:

  | Profile | Text NLL change | Top-1 agreement where dense's margin ≥ 1 nat | Passkey recall after a Pyramid prefill |
  |---|---|---|---|
  | `balanced` | ≤ +0.02 nats / token (about 2% perplexity) | ≥ 99% | equal to dense |
  | `fast` | ≤ +0.05 | ≥ 97% | at most one of five lost |
  | `max` | set by the evaluation suite ([12](#12-the-intelligence-target)) | reported | reported |

  `balanced` is set to the range sparse decode attention already ships in (at most about 3%
  perplexity). The limits are proposals: the owner sets the final values.
- **Prefill and decode differ by design** in the Bounded profiles (a prompt token and a generated
  token take different paths), so `prefillMatchesIncrementalDecode` is asserted for `off` and `exact`
  and reported, not asserted, for the others.

## 11. Measuring

Nothing else may use the GPU while measuring (AGENTS.md).

Speed (`MLXR_SIEVE` is planned; the commands themselves exist today):

    mlxr bench <model>                                   # today's numbers and the per-kernel profile
    MLXR_SIEVE=exact mlxr bench <model>                  # the same with a profile
    MLXR_BENCH_DECODE=1 MLXR_SIEVE=fast mlxr bench <model>
    MLXR_SIEVE=exact mlxr run <model> "<prompt>" --temp 0 --repeat-penalty 1 --seed 42 --max 256 --verbose

The bench section for Sieve adds, per profile: decode tok/s at 16 and 1,024 tokens of context;
**touched bytes per token and the A-number**, summed from what the kernels actually read; certified
rows per Head evaluation; experts kept per layer; the share of tokens leaving at each exit and the
mean depth; fills per token; prefill tok/s at 512 and 4,096 tokens with the tokens alive per stage;
Stride acceptance and tokens per pass; and the cost of each new kernel.

Fidelity (Bounded profiles), with the long-context bench's text metrics:

    MLXR_BENCH_TEXT=book.txt MLXR_BENCH_POSITIONS=2048 MLXR_SIEVE=balanced mlxr bench <model>

KL, text NLL under both modes, top-1 agreement by margin bucket, the greedy prefix shared with dense,
and the passkey needles after a Pyramid prefill. Use a text the model cannot have memorised, and a
second domain (code).

Intelligence: a new developer command, `eval <model> --suite <dir>`, with `--sieve <profile>`,
`--votes N` and `--limit N` as options:
- A suite is a directory of JSONL files the user provides, one item per line: `question`, `choices`,
  `answer`. No dataset ships with the repository.
- Scoring: the model answers through its chat template with the model's own thinking default and a
  fixed token budget; the answer letter is extracted from the reply. Greedy with seed 42, or N
  sampled replies and a majority vote with `--votes`.
- Report accuracy per file, the paired difference between two runs with a bootstrap 95% interval, and
  wall-clock time per item.
- Suite for this project: MMLU-Pro (a stratified 2,000-item subset), GPQA Diamond, GSM8K, ARC-Challenge.
  The comparison model is `mlx-community/Qwen3.5-4B-4bit` in this engine with `--sieve off`.

## 12. The intelligence target

T3, restated for an architecture that separates stored from touched: **a model that touches no more
per token than a dense 1B model (A1) scores above the dense Qwen3.5-4B on the suite of section 11.**

Three routes, in order of how much they ask:

| Route | Model | What must hold | Standing |
|---|---|---|---|
| R1. Spend the speed | Sieve-4B-A1, three sampled answers and a vote, against the dense 4B answering once | The A1 profile costs less than the vote gains, and three A1 answers take less time than one dense answer (on paper they take 70% of it on an M2 Pro) | The most reachable. Voting gains a few points on reasoning and math items, little on recall items. |
| R2. A stronger source | Sieve-9B, single answer | The profile loses under 3.4 MMLU-Pro points and 5.5 GPQA points of the 9B's lead, at A2.2 in `max` | The cleanest claim. Reaching A1 with the 9B means touching about a tenth of it per token, beyond the `max` operating point. |
| R3. Both | Sieve-9B at A1 with votes when time allows | R1 and R2 | The full target. |

What does not work, and stays ruled out: a 1B-parameter source ([4.3](#43-intelligence-a-1b-budget-not-a-1b-source)),
however it is run. Spending speed on a 0.8B model's answers does not close 37 points.

What the measurement is: one plot per Mac of suite accuracy against measured milliseconds per token,
with the dense Qwen 3.5 sizes (0.8B, 2B, 4B, 9B) as one curve and every Sieve profile of every import
as points. The architecture is right if its points lie above the dense curve; T3 is met by the first
point at or under A1 that lies above the dense 4B.

## 13. Plan and gates

Each phase ends with a measurement that either releases the next phase or stops that axis. Durations
are estimates for one person.

**Phase 0: the six hypotheses (1–2 weeks). No shipped code.** H2, H4, H5 and H6 need a model that can
be cut open at any layer; the quickest way is mlx-lm in a scratch repository outside this one
(decision 4 in section 1). H1, H3 and the constants need only this engine.

| # | Hypothesis | Measurement | Gate |
|---|---|---|---|
| 0 | The cost model | `mlxr bench` on Qwen3 0.6B, Llama 3.2 1B, Qwen3.5 0.8B; record the profile lines and the `verify step` lines | Replaces η, c0, c1 and c_v(2). |
| H1 | Head's certificate is tight | Dump 20K final hidden states; fit μ and Q; count certified rows for k = 1, 2 and 64, r ∈ {d/16, d/8, d/4} | Median certified rows ≤ 2% of V and expected bytes ≤ 40% of the dense head for some r ≤ d/4. Otherwise Head becomes a fixed-budget, Bounded screen. |
| H2 | A minority of experts carries the MLP | Cluster channels; with oracle selection, then with the fitted router, text NLL at 50, 40 and 35% of experts kept; the same per channel for comparison | NLL change ≤ +0.02 at ≤ 50% kept with the fitted router on the 1B-class candidates. Otherwise Experts is for 2B and larger only, or falls back to channel selection in decode. |
| H3 | The multi-token head drafts well | Which installed Qwen 3.5 exports hold `mtp.*`; greedy acceptance on chat text | Acceptance ≥ 0.6. |
| H4 | Tokens can leave early | Fit lenses at three depths; agreement of the lens argmax with the full model's against the margin; the share of positions that pass at δ = 1% and 2%; then the same with frozen states feeding later tokens | A mean depth ≤ 0.8 at δ = 1% and ≤ 0.65 at δ = 2%, with text NLL inside the profile's target. Otherwise Pyramid is prefill-only. |
| H5 | Prompt tokens need few layers | Staged prefill with attention-scored keeping; NLL of the continuation and passkey recall against the mean depth | NLL change ≤ +0.02 at mean depth ≤ 0.7 and ≤ +0.05 at ≤ 0.4. Otherwise T2 is out of reach on an M2 Pro above a 0.33B body. |
| H6 | The axes compose | All Bounded axes together at the `fast` and `max` points on Qwen3.5-0.8B, -4B and -9B; the suite of section 11 on the two larger ones | `fast` inside its NLL target. For T3: route R1 or R2 of section 12 holds on at least MMLU-Pro and GPQA. |

**Phase 1: Head and the `exact` profile (1–2 weeks).** Importer steps 1–4 and 9, the four head
kernels, `--sieve exact`. Gate: `exact` is exact on every installed model, and decode improves by at
least 8% on Qwen3.5 0.8B and Llama 3.2 1B.

**Phase 2: Stride (1–2 weeks).** The multi-token head, per-token state outputs, the cycle; then the
2-token pass (c_v from 1.5 toward 1.2) as its own performance pass. Gate: token-identical, and at
least 1.1× on Qwen3.5 0.8B.

**Phase 3: Experts (3–4 weeks).** Clustering, router, the reordered load, the decode kernels, then the
token-list tiles for prefill. Gate: `balanced` inside its target with decode at least 1.15× `exact`
and prefill at least 1.3× dense.

**Phase 4: Pyramid (4–6 weeks).** Lenses, exits, the pending ring and fills, staged prefill, the
combined budget. Gate: `fast` inside its target; on Qwen3.5 0.8B, decode ≥ 500 tok/s and prefill
≥ 5,000 tok/s on the M2 Pro. **This is where T1 and T2 are met or not.**

**Phase 5: the `eval` command, the `max` profile and T3.** Can start with phase 1. Gate: section 12.

**Phase 6: per-layer overhead (open-ended).** E1–E3, each measured alone.

Order of value if the plan is cut short: phase 1 is a lossless gain for every model and every user;
phases 3 and 4 are where the targets live.

## 14. Risks, open questions and what was ruled out

### 14.1 Risks

- **Compounding.** Four approximations at once. Each one's published or expected loss is small at
  its own operating point; nothing is known about their product. H6 exists for this.
- **Small models have less to give.** Redundancy in depth and width grows with model size. The
  speed targets are stated for the smallest models, where every hypothesis is weakest. The larger
  imports are where the architecture is most likely to work and least likely to reach 500 tok/s on
  an M2 Pro.
- **Wrong early exits are visible.** A lens that is confident and wrong emits a wrong token. δ bounds
  how often on calibration text, not on the user's text.
- **Extrapolated memories.** A frozen row is a shallower memory of its position. Long-range recall
  (needles, exact quotes, code identifiers) is where that would show first; the passkey gate is the
  minimum, not the proof.
- **Calibration dependence.** Routers, lenses, margins and fractions are fitted to a text. Head
  stays exact on any input (unusual input only makes it slower). The other axes' divergence on text
  unlike the calibration text is bounded by measurement only.
- **Control flow.** Exits and Stride both need a decision before the next segment is encoded. Each
  readback gives up some of the pipelined decode. If a readback costs much more than the assumed
  0.05 ms, small models lose most of the depth gain until indirect dispatch works.
- **Engineering size.** Segmented encoding, a pending ring, lazy fills and staged prefill touch the
  centre of `Transformer.encode`. Every existing path must stay bit-identical while they land.
- **Memory.** Dᵀ replaces the down projection in the Bounded profiles, so the model's size is
  unchanged; the sidecar adds 5–15%; the pending ring and per-token states are megabytes.
- **One bit width.** Only 4-bit has optimised decode kernels today; Sieve inherits that.

### 14.2 Open questions

1. How do c0 and c1 split between dispatch latency, the small kernels and the matvecs' ramp-up?
2. Exits by indirect dispatch: what does an empty indirect dispatch cost?
3. Channel selection inside the kept experts in decode: is the finer screen worth its bytes?
4. Should a token that reaches the top refresh the frozen rows of the positions just before it?
5. Is attention received from the last 64 positions the right keeping score for prefill, and does
   it need to be recomputed when a later batch arrives?
6. Does a clustered row order for the head (similar tokens in the same block) reduce certified
   blocks enough to justify remapping token ids at the engine boundary?
7. Do M5-class GPUs' new accelerators speed up `simdgroup_matrix` kernels, or only code that goes
   through Apple's tensor APIs?
8. Draft depth 2 for the multi-token head: mlx-lm's authors report it only recently level with depth 1.
9. The server: Sieve is set per load, like sparse attention. A per-request profile needs both
   layouts of the MLP in memory.

### 14.3 Ruled out by analysis

- **A 1B-parameter source for T3.** Section 4.3.
- **Exact prefill for T2 on an M2 Pro.** Section 4.2.
- **Fewer bits per weight as the main lever.** Meeting the M2 Pro byte budget for a 1B model needs
  about 2 bits per weight; training-free quantization at 2 bits costs far more than any profile
  allows, and only 4-bit has optimised kernels. Mixed precision by sensitivity remains a separate,
  smaller lever.
- **Bit planes as a coarse tier** (high bits for every channel, low bits only for the salient). The
  high plane of every weight is already half the bytes, and a coarse value for a minor channel is
  worth less than the bytes it costs. Dropping the minor experts wins at every budget examined.
- **A draft model.** Measured here: Qwen3 0.6B drafting for Qwen3 8B is accepted 45% of the time and
  makes decode slower (CHANGELOG 1.3.0).
- **Verifying early exits with the upper layers** (self-speculation). The draft is the lower layers,
  which are not cheap; with any realistic acceptance the cycle is barely faster than dense. Pyramid's
  gain comes from *not* verifying, which is why it is Bounded.
- **A cascade with a separate larger model.** The larger model must still read every token; on an M2
  Pro the result is slower than running the larger model through Sieve.
- **A certified MLP.** Bounds on a channel's contribution from a partial dot product are too loose to
  prove anything; SiLU has no exact zeros. Experts is Bounded by nature.
- **Fusing dispatches the way CHANGELOG 1.3.0 tried.** Same output, 3.5 → 6.6 ms. E1 is a narrower
  variant and must be measured alone.
- **The Neural Engine.** Reachable only through Core ML with static shapes; no path to this engine's
  quantized weights and KV cache.

## 15. Sources and what was not verified

From this repository (measured, M2 Pro): README "Performance" and "The 7B ceiling"; CHANGELOG 1.1.1,
1.2.0, 1.3.0, 1.5.0; docs/sparse-decode-attention.md.

External, read through web search summaries or a fetched page on 2026-11-10:

- mlx-lm pull request 990, "Native MTP speculative decoding (Qwen3.5/3.6 reference implementation)"
  (open when read): <https://github.com/ml-explore/mlx-lm/pull/990>. The head's structure, the
  reported acceptance rates and speed-ups, missing weights in some exports.
- Rapid-MLX issue 2175 (85% acceptance, 1.04–1.06× on a 9B):
  <https://github.com/raullenchai/Rapid-MLX/issues/2175>.
- Prox, "Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs",
  arXiv 2607.27591: <https://arxiv.org/abs/2607.27591>. Abstract only: 60–70% MLP sparsity, up to
  1.99× decode, ten models.
- Qwen 3.5 model cards: <https://huggingface.co/Qwen/Qwen3.5-4B>, `…-2B`, `…-0.8B`, `…-9B`.
- LFM2-8B-A1B (8.3B stored, 1.5B active, the naming this document borrows):
  <https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts>.
- M-series bandwidth and prompt-processing figures:
  <https://www.hardware-corner.net/m5-pro-m5-max-local-llm-4x-faster-prefill-202061108/> and search
  summaries of other coverage.

Cited from memory, not read for this document (titles and figures to be checked):

- TEAL, "Training-Free Activation Sparsity in Large Language Models", arXiv 2408.14690: 40–50%
  model-wide sparsity on 7B–70B models.
- MoEfication, "Transformer Feed-forward Layers are Mixtures of Experts" (2022): splitting a trained
  MLP into experts by co-activation with a router, on ReLU models.
- PyramidInfer, "Pyramid KV Cache Compression for High-throughput LLM Inference" (2024):
  training-free, 2.2× throughput.
- CALM, "Confident Adaptive Language Modeling" (2022), and LayerSkip (2024): early exit with copied
  states, both with training.
- The tuned lens, "Eliciting Latent Predictions from Transformers with the Tuned Lens" (2023).
- "Transformer Layers as Painters" (2024), for the tolerance of middle layers to running in parallel.

Not verified, and to be checked before any number here is quoted elsewhere:

- **Nothing was run.** No kernel, no importer, no benchmark, no experiment. Every "on paper" figure
  is arithmetic on the README's numbers at assumed operating points.
- **The operating points are assumptions.** No source shows a 1B-class model holding its quality at
  40% of its experts and 63% of its depth together.
- **Qwen 3.5 configurations.** huggingface.co was not reachable while writing. Hidden sizes (1,024
  for the 0.8B, 2,560 for the 4B) and the vocabulary size (about 248K) are taken from this
  repository's CHANGELOG; MLP sizes, layer counts and everything about the 9B are assumed.
- **Benchmark scores** come from search summaries of the model cards, not from the cards.
- **The papers** were not read: arxiv.org was not reachable.
- **GPU peak rates and other Macs' bandwidths** are quoted figures, not measurements with this
  engine.
- **The cost model** is a three-parameter fit to seven rows of one table.
