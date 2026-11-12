# Sieve

The last layer of a model (the output projection, or head) scores every token of the vocabulary on
every step, and reads its whole weight matrix to do it. The sampler then uses a handful of those
scores. Sieve scores every token with a small approximation of the head, and computes the real
score only for the most likely ones.

It needs no training, but each model has to be **fitted** once: a step of one to four minutes that
builds the approximation from the model's own weights and writes it next to the model.

Status:
- **Off by default.** With Sieve off, nothing changes: there is no Sieve buffer and no Sieve
  dispatch, and greedy output is byte-identical to the build before it.
- **It changes output** when on. That is an exception to the token-identical default path that
  AGENTS.md otherwise requires.
- **Speed** (M2 Pro, 4-bit): about 8% faster greedy decode on the four models measured.
- **Quality:** on the fit's own test states, the likeliest token is kept in 99.66 to 100% of them,
  and the sampling distribution at top-k 40 shifts by 0.01 to 0.37%. See "Measured" for how these
  were obtained and what was not measured.
- **Only fitted models.** A model that has not been fitted runs at its normal speed.
- **Not with a draft model.** Speculative decoding runs the full output layer on every step, so
  Sieve does nothing there, and the app and the CLI say so.

The method is SVD-softmax (Shim et al., NIPS 2017) with two changes: the low-rank directions are
fitted to the model's hidden states, not taken from the weight matrix alone, and the low-rank copy
is itself quantized to 4 bits.

Measured on a MacBook Pro M2 Pro (16-core GPU, 16 GB), macOS 27.2, on 2026-11-11.

## Using it

**In the app:** the chat inspector's **Sieve** section has:
- the **Sieve** switch. Like context length, it applies the next time a model is loaded;
- a **Fit** button for the loaded model. Fitting unloads the model, fits it and loads it again with
  Sieve on;
- a warning when the switch is on and Sieve does nothing or does it badly: the loaded model is not
  fitted, its fit cannot be used, its fit missed the quality target, or a draft model is set.

The inspector's **Model** section shows whether Sieve is active for the loaded model.

**On the command line:** `mlxr sieve` fits a model, and `mlxr run` and `mlxr serve` take
`--sieve off|balanced`. `balanced` is the only profile.

```bash
mlxr sieve mlx-community/Qwen3-0.6B-4bit
mlxr run mlx-community/Qwen3-0.6B-4bit --sieve balanced -p "Hello"
mlxr serve mlx-community/Qwen3-0.6B-4bit --sieve balanced
mlxr sieve mlx-community/Qwen3-0.6B-4bit --info
mlxr sieve mlx-community/Qwen3-0.6B-4bit --remove
```

- `--info` shows what the fit recorded, and `--remove` deletes it. The figures are read from the
  file: for a sidecar that came with a downloaded model, they are that file's claim, not a
  measurement made on this machine.
- `MLXR_SIEVE=balanced` is the same as `--sieve balanced`, and also applies to `mlxr bench`.
- A model that has not been fitted, or whose fit belongs to other weights, loads without Sieve and
  says so.

## Fitting a model

The fit writes `sieve/head.safetensors` into the model's directory (11 to 46 MB). Keep the GPU free
while it runs. Another quantization or fine-tune of the same model needs its own fit. The loader
checks a fingerprint of the output layer (its shape, its quantization and the first and last 64 KB
of its weights) and refuses a sidecar that does not match; a variant that differs only elsewhere in
that layer would not be caught.

`mlxr sieve MODEL` collects about 9,000 final hidden states from the model's own answers to 40
built-in prompts (temperature 0.8; every fourth prompt is held out), or from text files given with
`--text`. It then:

- takes the leading eigenvectors of the states' second moment as the directions;
- multiplies the dequantized head by them and quantizes the result to 4 bits in groups of 64 (the
  sketch);
- sets the margin of the cut on the held-out states;
- measures, on the same held-out states, the screen as the kernels run it against the dense head.

It tries a rank of an eighth of the hidden size first, then a quarter, then a half, and keeps the
first that meets the target:

- the dense head's likeliest token is kept in at least 99.5% of held-out states;
- at most 1% of sampled tokens change (the total variation between the two sampling distributions
  at top-k 40, temperature 1).

A fit that misses the target at every rank is still written. The command, `--info` and every load
say that it missed, and the app leaves the Sieve switch off after such a fit.

The held-out figures are optimistic in two ways. The margin and the rank are chosen on the same
states the figures are reported on; only the directions are fitted on other data. And the first
2,048 held-out states come from at most ten of the forty prompts, all or nearly all of them
English.

## How a step runs

The head is 88 MB on Qwen3 0.6B and 357 MB on Qwen3.5 4B, and reading it is 12 to 20% of a decode
step. With Sieve on, a step that samples on the GPU does this instead, in the same command buffer:

1. Projects the final hidden state onto the r fitted directions.
2. Scores every vocabulary row with the sketch (`head_sketch`).
3. Finds the cut below which at most 16,384 rows remain (8,192 for a vocabulary under 65,536;
   `head_cut`, `head_hist`, `head_cut2`).
4. Computes the head's own logit for those rows (`head_select`, `head_exact`).

The other rows keep their sketch value, and the sampler runs unchanged on the result.

The screen is not exact: a row the sketch ranks below the cut keeps its sketch value even when the
head would have ranked it higher. That is why it is fitted and measured per model.

Passes that hand logits to the CPU use the dense head as before: `forward`, scoring, and tool-call
sampling over a constrained set. A generation with a draft model or prompt lookup (speculative
decoding) is dense on every step, not only when it verifies. Prefill is unchanged too: only the
sampled last row of a prompt goes through the screen.

## Measured

Greedy decode (temperature 0, no repetition penalty) at 16 tokens of context, `mlxr bench`, median
of three alternating pairs. Quality is on 2,048 held-out states per model. "Sampling shift" is
computed, not observed: the total variation between the two sampling distributions at top-k 40 and
temperature 1, averaged over the states. It is the smallest share of steps on which a sampler could
pick a different token.

| Model | Rank | Dense | Sieve | Gain | Dense argmax kept | Sampling shift | KL mean (median) |
|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | 128 | 288.2 tok/s | 312.3 | +8.4% | 100.00% | 0.01% | 0.0029 (0.00011) |
| Qwen3.5 0.8B | 256 | 253.8 | 275.5 | +8.5% | 99.66% | 0.29% | 0.0453 (0.00011) |
| Llama 3.2 1B | 256 | 201.1 | 218.4 | +8.6% | 100.00% | 0.37% | 0.0698 (0.00010) |
| Qwen3.5 4B | 320 | 61.5 | 66.5 | +8.1% | 99.76% | 0.20% | 0.0333 (0.00003) |

At 1,024 tokens of context the gain is +6.8%, +8.0%, +7.3% and +8.3% in the same order.

Greedy output (`--temp 0 --repeat-penalty 1 --seed 42 --max 256`, one prompt) was identical to dense
on all four models. That is one prompt, not a guarantee: at 99.66% per token, a 256-token greedy
answer differs somewhere in about half of the cases. Sampled answers diverge sooner: once one token
differs, the rest of the answer does.

What the numbers do not show:

- **Perplexity is affected more than sampling.** The mean KL is carried by a few flat states, where
  the dense head spreads probability over more rows than the budget (in 1% of Llama's states, over a
  quarter of it). With top-k sampling those rows are never drawn; a perplexity benchmark would count
  them. Text NLL on a long document was not measured.
- **Sampler settings other than top-k 40 at temperature 1 were not measured.** That includes the
  app's default repetition penalty (1.05), higher temperatures, and top-p or min-p without top-k,
  where the sampler considers 512 rows and is more likely to reach rows that kept their
  sketch value.
- **Sampled-mode speed was not measured.** The gain above is for greedy decode.
- **The calibration is the model's own writing:** short, single-turn answers without a system
  prompt, tools or thinking, mostly in English. Agreement on anything else (other languages, long
  contexts, tool calls, thinking traces) was not measured.
- **Gemma's logit softcap** is applied after the screen; the fit's figures do not include it, and no
  Gemma model was fitted.
- **No benchmark score was measured** with Sieve on.
- **Other models were not measured.** The four above are the only ones fitted so far.

## Why it is 8% and not more

The head is 12 to 20% of a step, and the screen recovers about half of that. Measured on Qwen3 0.6B,
where the dense head costs about 0.5 ms of a 3.4 ms step (timed by leaving stages out of the full
step):

- A sketch narrow enough to be cheap has rows of 64 to 160 bytes. No existing kernel reads such rows
  efficiently: the decode matvec ran the rank-256 sketch at 73 GB/s. `head_sketch` (one thread per
  row) reaches about 115 GB/s, against about 190 GB/s for the dense head.
- Lower ranks are faster and less accurate. An eighth of the hidden size is the lowest rank the fit
  tries, and Qwen3.5 0.8B needed a quarter.
- The cut needs four small dispatches (about 0.04 ms), and the exact rows about 0.05 ms.

A faster sketch kernel is the remaining lever: at the dense head's byte rate the gain would be
about 11% (an estimate, not a measurement).

## Correctness gates

`SieveTests`:

- The kernels against a CPU reference on a synthetic head whose vocabulary is not a multiple of the
  block size: every row carries either the head's logit or the sketch's, on the right side of the cut,
  and the list never exceeds the budget.
- With a budget of every row and a margin that covers them all, the result matches a CPU reference
  of the dense head to 2e-3.
- A tie of every row at the maximum lists nothing (which rows it would list is not deterministic).
- A sidecar with another model's fingerprint, an unknown format, a wrong rank, a row budget above the
  vocabulary, a non-finite margin, or a truncated file is refused before any kernel runs.
- On an installed, fitted model: `forward` is bit-identical with Sieve on and off, and the screened
  path gives the same tokens twice. This test skips when no fitted model is installed.

Not covered by a test: the fit itself (directions, quantization, the held-out figures), head
quantizations other than 4-bit with half-precision scales, and the softcap branch.

The list of selected rows is built with an atomic counter, so its order varies between runs. Each
row's logit goes to its own slot, so the logits do not.
