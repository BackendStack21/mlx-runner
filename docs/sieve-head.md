# Sieve Head: the output layer by rows

Status: built, opt-in, off by default. Measured on a MacBook Pro M2 Pro (16-core GPU, 16 GB),
macOS 27.2, on 2026-11-11. It is the first, and so far the only, part of
[Sieve](sieve-architecture.md) in the engine.

**It changes output.** Without `--sieve`, nothing in the engine changes: there is no Sieve buffer and
no Sieve dispatch, and greedy output is byte-identical to the build before it.

## What it does

The output projection (the head) computes one logit per vocabulary row, and reads the whole matrix
to do it: 88 MB on Qwen3 0.6B, 357 MB on Qwen3.5 4B, 12 to 20% of a decode step. The sampler then uses
a handful of those logits.

With Sieve on, a step that samples on the GPU does this instead, in the same command buffer:

1. Projects the final hidden state onto r fitted directions (r is an eighth or a quarter of the
   hidden size).
2. Scores every row with a 4-bit sketch of the head in those directions (`head_sketch`).
3. Finds the cut below which at most 16,384 rows remain (`head_cut`, `head_hist`, `head_cut2`).
4. Computes the head's own logit for those rows (`head_select`, `head_exact`).

The other rows keep their sketch value. The sampler runs unchanged on the result.

Phase 0 showed that no certificate can make this exact
([sieve-phase0-results.md](sieve-phase0-results.md)), so it is fitted and measured per model.

Passes that hand logits to the CPU use the dense head as before: `forward`, speculative
verification, scoring, tool-call sampling over a constrained set.

## Using it

Each model is fitted once. That takes one to four minutes with nothing else on the GPU, and writes
`sieve/head.safetensors` into the model's directory (11 to 46 MB).

```bash
mlxr sieve mlx-community/Qwen3-0.6B-4bit
mlxr run mlx-community/Qwen3-0.6B-4bit --sieve balanced -p "Hello"
mlxr serve mlx-community/Qwen3-0.6B-4bit --sieve balanced
mlxr sieve mlx-community/Qwen3-0.6B-4bit --info
mlxr sieve mlx-community/Qwen3-0.6B-4bit --remove
```

- `MLXR_SIEVE=balanced` is the same as `--sieve balanced`, and also applies to `mlxr bench`.
- In the app: the inspector's Sieve section has the switch and a "Fit" button. Fitting unloads the
  model, fits it and loads it again with Sieve on.
- A model that has not been fitted, or whose sidecar was fitted to other weights, loads dense and
  says so.
- `balanced` is the only profile. `exact` does not exist; `fast` and `max` need Experts and Pyramid,
  which are not built.

## The fit

`mlxr sieve MODEL` collects about 9,000 final hidden states from the model's own answers to 40
built-in prompts (temperature 0.8; every fourth prompt is held out), or from text files given with
`--text`. It then:

- takes the leading eigenvectors of the states' second moment as the directions;
- multiplies the dequantized head by them and quantizes the result to 4 bits in groups of 64;
- measures, on the held-out states, the screen as the kernels run it against the dense head.

It tries a rank of an eighth of the hidden size first, then a quarter, then a half, and keeps the
first that meets the target: the dense argmax kept in at least 99.5% of held-out states, and at most
1% of sampled tokens changed (the total variation between the two sampling distributions at
top-k 40, temperature 1). These limits are a proposal; the owner sets the final values.

## Measured

Decode at 16 tokens of context, `mlxr bench`, median of three alternating pairs. Quality is on 2,048
held-out states per model.

| Model | Rank | Dense | Sieve | Gain | Dense argmax kept | Sampled tokens changed | KL mean (median) |
|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | 128 | 288.2 tok/s | 312.3 | +8.4% | 100.00% | 0.01% | 0.0029 (0.00011) |
| Qwen3.5 0.8B | 256 | 253.8 | 275.5 | +8.5% | 99.66% | 0.29% | 0.0453 (0.00011) |
| Llama 3.2 1B | 256 | 201.1 | 218.4 | +8.6% | 100.00% | 0.37% | 0.0698 (0.00010) |
| Qwen3.5 4B | 320 | 61.5 | 66.5 | +8.1% | 99.76% | 0.20% | 0.0333 (0.00003) |

At 1,024 tokens of context the gain is +6.8%, +8.0%, +7.3% and +8.3% in the same order.

Greedy output (`--temp 0 --repeat-penalty 1 --seed 42 --max 256`, one prompt) was identical to dense
on all four models. That is one prompt, not a guarantee: at 99.66% per token, a 256-token greedy
answer differs somewhere in about half of the cases.

What the numbers do not show:

- **Perplexity is affected more than sampling.** The mean KL is carried by a few flat states, where
  the dense head spreads probability over more rows than the budget (in 1% of Llama's states, over a
  quarter of it). With top-k sampling those rows are never drawn; a perplexity benchmark would count
  them. Text NLL on a long document was not measured.
- **The calibration is the model's own writing.** Agreement on text unlike it (other languages,
  unusual formats) was not measured beyond what the 40 prompts cover.
- **No benchmark score was measured** with Sieve on.
- **Prefill is unchanged.** Only the sampled last row of a prompt goes through the screen.

## Why it is 8% and not more

The specification projected the head at a quarter of its bytes. Measured on Qwen3 0.6B, where the
dense head costs about 0.5 ms of a 3.4 ms step (timed by leaving stages out of the full step):

- A sketch narrow enough to be cheap has rows of 64 to 160 bytes. No existing kernel reads such rows
  efficiently: the decode matvec ran the rank-256 sketch at 73 GB/s. `head_sketch` (one thread per
  row) reaches about 115 GB/s, against about 190 GB/s for the dense head.
- Lower ranks are faster and less accurate. Rank d/8 is the lowest that meets the target on any
  model, and three of four models needed more rows or a higher rank than phase 0's 700 to 1,800 rows
  suggested.
- The cut needs four small dispatches (about 0.04 ms), and the exact rows about 0.05 ms.

A faster sketch kernel is the remaining lever: at the dense head's byte rate the gain would be
about 11% (an estimate, not a measurement).

## Correctness gates

`SieveTests`:

- The kernels against a CPU reference on a synthetic head whose vocabulary is not a multiple of the
  block size: every row carries either the head's logit or the sketch's, on the right side of the cut,
  and the list never exceeds the budget.
- With a budget of every row, the result is the dense head.
- A tie of every row at the maximum lists nothing (which rows it would list is not deterministic).
- A sidecar with another model's fingerprint, an unknown format, a wrong rank, a row budget above the
  vocabulary, a non-finite margin, or a truncated file is refused before any kernel runs.
- On an installed, fitted model: `forward` is bit-identical with Sieve on and off, and the screened
  path gives the same tokens twice.

The list of selected rows is built with an atomic counter, so its order varies between runs. Each
row's logit goes to its own slot, so the logits do not.
