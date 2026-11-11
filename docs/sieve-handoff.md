# Sieve: handoff to the agent on the Mac

This is the task for whoever continues the Sieve work on a Mac. It was written by an agent that had
this repository but no Mac, no Swift toolchain and no Metal. Read it before
[sieve-architecture.md](sieve-architecture.md); it tells you what that document is worth.

## State of the branch

**Update, 2026-11-11:** steps 0 to 4 are done on an M2 Pro. Read
[sieve-phase0-results.md](sieve-phase0-results.md) first: H1 failed, no installed export holds the
multi-token head, and the specification is corrected from the measurements. Steps 5 to 7 wait for
the owner's decisions. The rest of this file is the task as it was handed over.

Later the same day the owner asked for Sieve to be integrated into the CLI and the app: the row
screen of step 4 is built as the opt-in `--sieve balanced` ([sieve-head.md](sieve-head.md)). That
answers decision 2 (an output-changing, opt-in mode) with a yes; decisions 1, 3 and 4 are still open.

- Branch `feat/sieve-architecture-spec`, two commits on top of `main`, both documentation:
  `docs/sieve-architecture.md` (the specification) and this file.
- **Nothing is implemented.** No kernel, no importer, no command, no test.
- **Nothing was run.** Not `swift build`, not `swift test`, not a single benchmark. The test that
  parses `mlxr` command lines under `docs/` was emulated in Python and passed there; that is not the
  same as passing.
- **Every speed figure in the specification is arithmetic** on the README's M2 Pro measurements at
  operating points that were assumed, not found. Treat each as a hypothesis. Where a measurement of
  yours disagrees with the specification, the measurement wins and the specification is corrected.

## Read first

1. `AGENTS.md`: the rules of this repository. They apply to everything below.
2. `docs/sieve-architecture.md`: sections 1 (the idea, the targets, the open decisions), 3 (cost
   model), 5 (the five axes) and 13 (the hypotheses and their gates). The rest when you need it.
3. `docs/sparse-decode-attention.md`: the one output-changing feature the engine already ships. Its
   invariants, its measuring section and its bench output are the model for Sieve's.
4. `Sources/MLXEngine/Transformer.swift`, `encode(T:offset:…)` and `profile()`: one forward pass and
   the per-kernel timing that `mlxr bench` prints.

## Rules for this work

- Swift only, no third-party packages, no Python in this repository (AGENTS.md). Measurement code
  that must live here goes in opt-in tests, enabled by an environment variable, like
  `Tests/MLXEngineTests/SparseTimingTests.swift` (`MLXR_KERNEL_TIMING`).
- Existing paths stay token-identical. Phase 0 adds no shipped code and changes no existing kernel.
- Nothing else may use the GPU while you measure. Alternate the things you compare, and say how many
  runs each figure comes from.
- Never write a projected number where a reader could take it for a measured one. In the documents,
  say which is which, with the machine and the date.
- `swift test` must pass before each commit. One of its tests parses every command line under `docs/`
  that starts with `mlxr`: a command that does not exist yet is written without the prefix.
- Commit to this branch and push. Do not open a pull request unless the owner asks.

## Decisions that belong to the owner

Section 1 of the specification lists four. They are still open. Ask; do not choose.

| Decision | What waits for it |
|---|---|
| 1. Is "1B" the stored size or the touched size? | Which models steps 6 and 7 use for the intelligence target. |
| 2. Is an output-changing, opt-in mode acceptable? | Whether Experts and Pyramid are worth measuring at all (steps 6 and 7). |
| 3. Which Mac is the reference? | How the results are stated. Record your machine either way (step 0). |
| 4. May phase 0 use mlx-lm in Python, in a scratch directory outside this repository? | How steps 5 to 7 are done. With it: days. Without it: debug kernels in this engine, which is weeks. |

Steps 0 to 4 need none of these. Start there.

## The task: phase 0

Decide by measurement which of Sieve's axes are worth building, and replace the specification's
assumptions with numbers from this Mac. The gates are in section 13 of the specification; they are
repeated here in short.

### Step 0. Build, test, baseline

```bash
swift build -c release
swift test
make cli
```

- If the documentation test fails on `docs/sieve-architecture.md` or this file, fix the document, not
  the test.
- Record the machine: chip, GPU core count, memory, macOS version.
- Install the models (the Qwen 3.5 4B is 3 GB; pull the 9B only once step 2 says it fits):

```bash
mlxr pull mlx-community/Qwen3-0.6B-4bit
mlxr pull mlx-community/Llama-3.2-1B-Instruct-4bit
mlxr pull mlx-community/Qwen3.5-0.8B-4bit
mlxr pull mlx-community/Qwen3.5-4B-4bit
```

- Run the benchmark on each, with nothing else on the GPU, and keep the whole output:

```bash
mlxr bench mlx-community/Qwen3-0.6B-4bit
mlxr run mlx-community/Qwen3-0.6B-4bit "Explain RoPE" --temp 0 --repeat-penalty 1 --seed 42 --max 256
```

### Step 1. The cost model

The specification's section 3.1 fits `t_step = B / (η · BW) + c0 + c1 · L` to seven rows of the README
and gets η = 0.91, c0 = 0.25 ms, c1 = 46 µs per layer. Replace the fit with measurements:

- From the profile lines of `mlxr bench` (`decode step @16: … without …`, `tiny dependent dispatch`,
  `lm_head matvec`, the matvec GB/s lines): how a decode step splits into weight reads, the head,
  attention, the small kernels and dispatch latency. Say what c1 is made of.
- From the `verify step T=…` lines: the cost of a 2-token pass relative to a step (c_v(2), 1.5 in the
  specification).
- From `layer GEMMs at T=512`: the effective GEMM rate F (3.3 TFLOPS in the specification).
- One thing the bench does not print: the cost of reading a 4-byte result back between two command
  buffers (commit, wait, read, encode, commit). The specification assumes 0.05 ms per Pyramid exit
  test. Measure it in an opt-in test.

Gate: none. This step only replaces constants.

### Step 2. The real model shapes

The first author could not reach huggingface.co. For Qwen 3.5 the specification assumes a body that
is 35% mixer and attention and 65% MLP, 24 and 32 layers, and invents the 9B's size.

- From each installed model's `config.json` and safetensors headers: hidden size, MLP size, layers,
  layer types, vocabulary, the linear-attention dimensions, and the bytes of the head, of each
  layer's mixer or attention projections, and of each layer's MLP.
- Which exports hold multi-token-prediction weights: tensor names starting with `mtp.`, `model.mtp.`
  or `language_model.mtp.` (a safetensors file starts with an 8-byte little-endian header length,
  then a JSON header). The engine skips these at load today (`SafeTensorFile.unusedPrefixes`).
- Whether a 9B export fits this Mac (weights plus KV cache at 4K context under the memory guard).

### Step 3. Reproduce the tables, then correct them

Write `scripts/sieve-budget.swift` (Foundation only) from Appendix A.

1. With the specification's constants it must reproduce the decode and prefill tables of the
   specification's section 6 within rounding. If it does not, the specification's arithmetic is wrong
   somewhere: find out where and say so.
2. Rerun it with the constants of step 1 and the shapes of step 2. Update sections 1, 3, 4 and 6 of
   the specification with the result, marked as projections from this Mac's measurements.

### Step 4. H1: is Head's certificate tight?

Head (specification 5.2) is the lossless axis, and every other axis leans on it. It needs no decision
and no Python. Do it as an opt-in test.

1. **Hidden states.** After `Transformer.forward(tokens, offset: 0)` on one batch of up to 512 tokens,
   the scratch buffer `xn` should hold the final norm's output for every row of the batch (the last
   `encodeAdd` of `encode` writes it; confirm in the code before relying on it). Add an internal
   accessor for tests. Run about 50 chunks of real text for 20,000 states or more; keep a fifth as
   held-out. Use a text the model has not memorised (see the notes on memorisation in
   sparse-decode-attention.md) and a second domain, such as code.
2. **Head rows.** Dequantize the output projection on the CPU (`w = scale · q + bias` per group).
3. **Fit**, in double precision with Accelerate: the mean μ, the covariance of the states, its top-r
   eigenvectors Q, for r in {d/16, d/8, d/4}.
4. **Per row:** the prior `p = w · μ`, the sketch `s = Qᵀ w`, the unseen norm `n = ‖w − Q s‖`. First
   with the sketch unquantized, then quantized to 8 and to 4 bits with its error norm `q`.
5. **Per held-out state**, for k in {1, 2, 64}: the screen logits, the bounds `q ν + n ρ`, the
   threshold θ (the k-th largest lower bound), the rows whose upper bound reaches θ, and the blocks
   of 64 rows that hold any of them.
6. **Soundness.** Every one of the dense head's top k rows must be in the certified set, for every
   state. A single miss means the mathematics or the code is wrong; stop and report it.

Report the median, mean and 95th percentile of certified rows and blocks, and the expected touched
bytes (sketch, two bounds per row, certified blocks) against the dense head.

Gate: median certified rows ≤ 2% of the vocabulary and expected bytes ≤ 40% of the dense head for
some r ≤ d/4. If it fails, say by how much, and measure the uncertified variant (the top blocks by
screen logit): how many blocks it takes to hold the dense argmax in 99.99% of states.

### Step 5. H3: does the multi-token head draft well?

Needs decision 4 for the quick route. mlx-lm pull request 990 (open when the specification was
written) implements the head; outside this repository, measure greedy acceptance of its one-token
draft on chat text for each export that step 2 found weights in. Without Python, this waits for the
head to exist in this engine.

Gate: acceptance ≥ 0.6. Read the result with step 1's c_v(2) in the table of specification 3.2.

### Step 6. H2, H4, H5: width and depth

Needs decisions 2 and 4. Each is an offline experiment on a model cut open at its layers. The
specification's sections 5.3 and 5.4 define the mechanisms; section 13 has the gates. In short:

- **H2, Experts.** Record channel contributions `|a_i| · ‖d_i‖` per layer on a few thousand positions.
  Text NLL with the top 50, 40 and 35% kept (a) per channel with oracle selection, (b) per expert of
  64 co-firing channels with oracle selection, (c) per expert with the fitted linear router.
  Gate: NLL change ≤ +0.02 nats at ≤ 50% kept with the fitted router, on the 1B-class models.
- **H4, Pyramid in decode.** Record the residual after layers near 1/4, 1/2 and 3/4 of the depth and
  the final residual. Fit the lenses. Agreement of the lens's argmax with the model's against the
  lens's top-two margin; the share of positions that pass at 1% and 2% disagreement; then the same
  with exited positions feeding extrapolated keys and values to later tokens.
  Gate: mean depth ≤ 0.8 at 1% and ≤ 0.65 at 2%, with text NLL inside +0.02 and +0.05.
- **H5, Pyramid in prefill.** Staged prefill with attention-scored keeping. NLL of the continuation
  and passkey recall at several depths, against the mean depth.
  Gate: NLL change ≤ +0.02 at mean depth ≤ 0.7 and ≤ +0.05 at ≤ 0.4.

Run each on Qwen3.5 0.8B and Llama 3.2 1B first (the speed targets live there), then on Qwen3.5 4B
(where the redundancy should be larger).

### Step 7. H6: do the axes compose?

Only if at least two of H2, H4 and H5 passed. All the passing axes together at the specification's
`fast` and `max` operating points: text NLL, then the evaluation suite of specification section 11 on
the 4B and, if it fits, the 9B, against the dense 4B. The suite's data is not in this repository; the
owner provides it or says where to get it.

## What to deliver

1. `docs/sieve-phase0-results.md`: the machine; the raw numbers with run counts; for each hypothesis
   the gate, the measurement and a verdict of passed, failed or not tested (with the reason); the
   updated forecast for the three targets on this Mac.
2. The specification corrected by what you found: constants, shapes, tables, and any axis that
   failed its gate marked as such, not removed.
3. The measurement code as opt-in tests in this repository, and `scripts/sieve-budget.swift`.
4. A short report to the owner: one line per hypothesis, the forecast, and what you recommend
   building first. Section 13 of the specification orders the phases; say whether your numbers
   change that order.

Then stop and wait for the owner. Phase 1 (Head and the `exact` profile, specification section 13) is
the expected next task if H1 passed, and it is lossless and off by default, but starting it is the
owner's call.

## Appendix A: how the specification's tables were computed

Constants: bytes per weight `BPW = 0.5625`; η = 0.91; c0 = 0.25 ms; c1 = 0.046 ms per layer; exit
test synchronisation `cx = 0.05 ms`; Head touches `hf = 0.26` of the dense head per evaluation;
bandwidth BW = 197 GB/s (M2 Pro) or 500 GB/s (assumed).

Bytes of a plain transformer with hidden size d, MLP size I, L layers, nH heads, nKV key/value heads,
head dimension hd, vocabulary V:

    H  = V · d · BPW
    A  = L · (nH·hd·d + 2·nKV·hd·d + d·nH·hd) · BPW
    M  = L · 3 · d · I · BPW
    fill = 2·nKV·hd·d / (nH·hd·d + 2·nKV·hd·d + d·nH·hd)

Llama 3.2 1B: d 2048, I 8192, L 16, nH 32, nKV 8, hd 64, V 128256. Qwen3 0.6B: d 1024, I 3072, L 28,
nH 16, nKV 8, hd 128, V 151936. Qwen 3.5 (assumed): H = 248000 · hidden · BPW, body = size − H,
A = 0.35 · body, M = 0.65 · body, fill = 0.7; 0.8B: size 424 MB, hidden 1024, L 24; 4B: size 3030 MB,
hidden 2560, L 32; 9B: H 500 MB, body 5100 MB, L 40.

A profile is the kept share of experts k, two exits (depth d1 with share p1, depth d2 with share p2)
and a mean prefill depth ρ:

| Profile | k | (d1, p1) | (d2, p2) | ρ |
|---|---|---|---|---|
| `balanced` | 0.50 | (0.50, 0.25) | (0.75, 0.25) | 0.70 |
| `fast` | 0.40 | (0.375, 0.45) | (0.625, 0.25) | 0.40 |
| `max` | 0.35 | (0.25, 0.60) | (0.50, 0.25) | 0.35 |

Decode:

    fd = d1·p1 + d2·p2 + (1 − p1 − p2)            mean depth
    nx = 1 + (1 − p1)                             exit tests per token
    nh = 1 + (1 − p1) + (1 − p1 − p2)             Head evaluations per token
    fW = 0.963·k + 0.037 + 0.02                   share of the MLP's bytes touched
    B_pass  = H·hf·nh + fd·(A + fW·M)
    t_pass  = B_pass / (η·BW) + c0 + cx·nx + c1·L·fd
    t_dense = (H + A + M) / (η·BW) + c0 + c1·L
    ratio   = (t_dense / t_pass) · stride         stride = 1.25 for Qwen 3.5 imports, else 1
    touched per token = B_pass / stride           A-number = touched per token / 562.5 MB

M2 Pro rows multiply the README's measured dense rate by `ratio`; 500 GB/s rows use `1 / t_dense` at
that bandwidth as the dense rate.

Prefill, with a = A / (A + M), m = 1 − a, the GEMM share of today's prefill time g = 0.85, and a
factor 0.8 for tiles over token subsets:

    frac  = ρ·(g·(a + m·k) + (1 − g)) + (1 − ρ)·g·a·fill
    tok/s = README prefill / frac · 0.8

For a GPU with effective rate F, multiply by F / 3.3.

README rates used (decode at 16 tokens / prefill at 512, tok/s): Qwen3 0.6B 286 / 3,020; Qwen3.5 0.8B
271 / 2,780; Llama 3.2 1B 202 / 1,720; Qwen3.5 4B 62.3 / 403 (the 403 is the 490-token prompt of the
README's Ollama comparison). The 9B has no measured rate: its dense decode is `1 / t_dense`, and its
dense prefill is the 4B's 403 scaled by the ratio of the bodies (2,670 / 5,100).

## Appendix B: what the first author could not verify

- Anything that needs a Mac: see "State of the branch".
- Qwen 3.5's shapes (step 2), and whether any installed export holds the multi-token head's weights.
- The benchmark scores in specification 4.3: taken from web search summaries of the model cards.
- The papers cited in specification section 15: arxiv.org was unreachable; figures are from
  abstracts quoted by search results or from memory.
- That `xn` holds the final hidden states after a forward pass (step 4): read from the code, not run.
- That the documentation test passes with these two files: emulated, not run.
