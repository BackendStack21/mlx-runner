# Sieve

The last layer of a model (the output projection, or head) scores every token of the vocabulary on
every step, and reads its whole weight matrix to do it. The sampler then uses a handful of those
scores. Sieve scores every token with a small approximation of the head, and computes the real
score only for the most likely ones.

It needs no training, but each model has to be **fitted** once: a step of one to six minutes that
builds the approximation from the model's own weights and writes it next to the model.

Status:
- **Off by default.** With Sieve off, nothing changes: there is no Sieve buffer and no Sieve
  dispatch, and greedy output is byte-identical to the build before it.
- **It changes output** when on. That is an exception to the token-identical default path that
  AGENTS.md otherwise requires.
- **Speed** (M2 Pro, 4-bit, ten models): from under 1% to 15% faster decode, greedy and sampled
  alike. The gain follows the head's share of the model: 8 to 15% on models up to 4B with a large
  vocabulary, 4% on Qwen3 8B, and under 1% on Phi 3.5 mini and Mistral 7B, whose vocabulary is
  small.
- **Quality:** on test states that the fit chose nothing on, the likeliest token is kept in 99.51
  to 100% of them. In real generations with the default sampler, 0 to 0.3% of tokens differ from
  the dense answer before the first difference. See "Measured" for how these were obtained and
  what was not measured.
- **Only fitted models.** A model that has not been fitted runs at its normal speed.
- **Not with a draft model.** Speculative decoding runs the full output layer on every step, so
  Sieve does nothing there, and the app and the CLI say so.

The method is SVD-softmax (Shim et al., NIPS 2017) with two changes: the low-rank directions are
fitted to the model's hidden states, not taken from the weight matrix alone, and the low-rank copy
is itself quantized to 4 bits.

Measured on a MacBook Pro M2 Pro (16-core GPU, 16 GB), macOS 27.2, on 2026-11-12.

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
  says so. So does one whose sidecar does not fit in memory at the requested context length.
- `mlxr run` and `mlxr serve` print what the load did with the profile, with the fit's figures.

**In code:** `InferenceEngine.load(directory:draftDirectory:contextLength:sparse:sieve:)` takes a
`SieveProfile?`; nil means the dense head. `LoadedModelInfo.sieveHead` holds the fit's figures when
Sieve is active, and `LoadedModelInfo.sieveNote` says why when it was asked for and is not.
`InferenceEngine.fitSieve(directory:options:progress:)` runs the fit.

## Fitting a model

The fit writes `sieve/head.safetensors` into the model's directory (9 to 58 MB on the models
below). Keep the GPU free while it runs. Another quantization or fine-tune of the same model needs
its own fit. The loader checks a fingerprint of the output layer and refuses a sidecar that does
not match. The fingerprint covers the layer's shape and quantization, all of its scales and biases,
and 4 KB of its packed weights taken at 1,024 evenly spaced places. It is a check against a wrong
or stale file, not against a forged one.

The fit holds the dequantized head and its product with the directions in memory. It refuses to
start when that does not fit next to the model: on 16 GB, Qwen3.5 9B and Qwen3 14B are refused.

`mlxr sieve MODEL` collects final hidden states from the model's own answers to 40 built-in prompts
(temperature 0.8), or from text files given with `--text`. The prompts are split three ways, and
no state is used for more than one purpose:

- **fit** (25 prompts): the directions;
- **tune** (5 prompts, up to 1,024 states): the margin of the cut and the choice of rank;
- **test** (10 prompts, two of them in French and German, up to 2,048 states): the reported
  figures.

It then:

- takes the leading eigenvectors of the fit states' second moment as the directions;
- multiplies the dequantized head by them and quantizes the result to 4 bits in groups of 64 (the
  sketch);
- sets the row budget to the largest of 512 to 16,384 that is at most a quarter of the vocabulary,
  and the margin of the cut from the tune states;
- measures the screen as the kernels run it against the dense head, on the tune states and on the
  test states.

Only the rows the sampler can pick are screened and measured: the tokenizer's vocabulary, when the
head has more rows than that.

It tries a rank of an eighth of the hidden size first, then a quarter, then a half, and stops at
the first that meets the target on both the tune and the test states:

- the dense head's likeliest token is kept in at least 99.5% of states;
- at most 1% of sampled tokens change (the total variation between the two sampling distributions
  at top-k 40, temperature 1).

The test states take part in that one decision: a rank that fails on them is replaced by a higher
one, never by a lower one. Apart from that, the reported figures are on states nothing was chosen
on.

Options of `mlxr sieve`, for experiments (the defaults are what was measured below):

| Option | Meaning | Default |
|---|---|---|
| `--text FILE` | calibration text, repeatable | the model's answers to the built-in prompts |
| `--tokens N` | hidden states to collect, 2,000 to 60,000 | 12,000 |
| `--rank N` | one rank for the sketch, 64 to 8,192, with no search | the lowest of 1/8, 1/4 and 1/2 of the hidden size that meets the target |
| `--rows N` | row budget: 512, 1,024, ... 16,384 | the largest that is at most a quarter of the vocabulary |

A fit that misses the target at every rank is still written. The command, `--info` and every load
say that it missed, and the app leaves the Sieve switch off after such a fit.

## How a step runs

With Sieve on, a step that samples on the GPU does this instead of the dense head, in the same
command buffer:

1. Projects the final hidden state onto the r fitted directions.
2. Scores every vocabulary row with the sketch (`head_sketch`).
3. Finds the cut below which at most the row budget remains (`head_cut`, `head_hist`,
   `head_cut2`).
4. Computes the head's own logit for those rows (`head_select`, `head_exact`).

The other rows keep their sketch value, and the sampler runs unchanged on the result.

The screen is not exact: a row the sketch ranks below the cut keeps its sketch value even when the
head would have ranked it higher. That is why it is fitted and measured per model.

Passes that hand logits to the CPU use the dense head as before: `forward`, scoring, and tool-call
sampling over a constrained set. A generation with a draft model or prompt lookup (speculative
decoding) is dense on every step, not only when it verifies. Prefill is unchanged too: only the
sampled last row of a prompt goes through the screen.

## Measured

### Speed

`MLXR_BENCH_DECODE=1 mlxr bench`, tokens per second at 16 tokens of context, median of three
alternating dense and Sieve runs. "Greedy" is temperature 0 without repetition penalty. "Sampled"
is the app's default sampler (temperature 0.7, top-k 40, top-p 0.95, repetition penalty 1.05).

| Model | Vocabulary | Greedy dense | Greedy Sieve | Gain | Sampled dense | Sampled Sieve | Gain |
|---|---|---|---|---|---|---|---|
| Gemma 3 1B QAT | 262K | 188.1 | 216.5 | +15.1% | 185.6 | 213.3 | +14.9% |
| Qwen3.5 0.8B | 248K | 254.8 | 286.3 | +12.4% | 251.2 | 282.4 | +12.4% |
| Gemma 4 E2B OptiQ | 262K | 73.3 | 82.2 | +12.1% | 72.9 | 81.8 | +12.2% |
| Llama 3.2 1B | 128K | 202.4 | 219.6 | +8.5% | 201.0 | 217.7 | +8.3% |
| Qwen3 0.6B | 152K | 287.9 | 311.6 | +8.2% | 284.0 | 306.3 | +7.9% |
| Qwen3 0.6B LR-DWQ | 152K | 288.7 | 311.8 | +8.0% | 285.7 | 306.1 | +7.1% |
| Qwen3.5 4B | 248K | 61.8 | 66.8 | +8.1% | 61.6 | 66.6 | +8.1% |
| Qwen3 8B | 152K | 38.3 | 39.8 | +3.9% | 38.2 | 39.7 | +3.9% |
| Phi 3.5 mini | 32K | 70.2 | 70.8 | +0.9% | 70.0 | 70.5 | +0.7% |
| Mistral 7B v0.3 | 33K | 41.1 | 41.4 | +0.7% | 40.9 | 41.0 | +0.2% |

At 1,024 tokens of context the greedy gain is +13.7%, +11.9%, +11.7%, +7.6%, +6.5%, +6.2%, +8.1%,
+4.1%, +0.7% and +0.3% in the same order, and the sampled gain is within 0.4 points of it.

On Phi 3.5 mini and Mistral 7B the gain is within the range a user would not notice, and Sieve is
not worth turning on for them.

### Quality

Two kinds of figures.

**Computed by the fit**, on up to 2,048 test states per model. "Sampling shift" is the total
variation between the dense and the screened sampling distribution at top-k 40 and temperature 1,
averaged over the states. It is the smallest share of steps on which a sampler could pick a
different token.

**Observed in generations.** The ten test prompts are answered dense, dense a second time (the
control) and with Sieve, up to 96 tokens each: once greedy, and with the default sampler under
three seeds (30 answers). Dense against itself was identical in every case, so each difference is
Sieve's. "Per token" is the number of answers that changed, divided by the tokens compared up to
and including each first change; after the first changed token, the two answers are different
texts and are not compared.

| Model | Rank | Row budget | Sidecar | Likeliest token kept | Sampling shift | KL mean (median) | Greedy answers identical | Sampled answers identical | Changed per token, sampled |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 3 1B QAT | 192 | 16,384 | 28.8 MB | 99.51% | 0.60% | 0.1170 (0.00000) | 8 of 10 | 24 of 30 | 0.22% |
| Qwen3.5 0.8B | 128 | 16,384 | 18.1 MB | 99.61% | 0.68% | 0.1056 (0.00026) | 8 of 10 | 23 of 30 | 0.29% |
| Gemma 4 E2B OptiQ | 384 | 16,384 | 57.8 MB | 99.90% | 0.11% | 0.0366 (0.00000) | 9 of 10 | 28 of 30 | 0.07% |
| Llama 3.2 1B | 256 | 16,384 | 19.5 MB | 99.95% | 0.14% | 0.0244 (0.00010) | 10 of 10 | 25 of 30 | 0.19% |
| Qwen3 0.6B | 128 | 16,384 | 11.2 MB | 100.00% | 0.10% | 0.0228 (0.00014) | 10 of 10 | 25 of 30 | 0.20% |
| Qwen3 0.6B LR-DWQ | 128 | 16,384 | 11.2 MB | 99.95% | 0.13% | 0.0269 (0.00004) | 8 of 10 | 25 of 30 | 0.20% |
| Qwen3.5 4B | 320 | 16,384 | 46.3 MB | 99.85% | 0.16% | 0.0239 (0.00002) | 10 of 10 | 28 of 30 | 0.07% |
| Qwen3 8B | 512 | 16,384 | 48.0 MB | 100.00% | 0.00% | 0.0002 | 10 of 10 | 30 of 30 | 0.00% |
| Phi 3.5 mini | 384 | 4,096 | 9.3 MB | 99.71% | 0.42% | 0.0864 | 7 of 10 | 26 of 30 | 0.15% |
| Mistral 7B v0.3 | 512 | 8,192 | 13.6 MB | 100.00% | 0.00% | 0.0007 | 10 of 10 | 30 of 30 | 0.00% |

The greedy rate per token was 0 to 0.34% (Phi 3.5 mini the highest), in line with the fit's
"likeliest token kept". All ten fits meet the target. Gemma 4 E2B missed it on the test states at
the lowest rank (98.93%) and was fitted at the next one.

An answer that differs is not a worse answer: the changed token is one the dense model also rated
likely. But this was not scored. What the numbers do not show:

- **No benchmark score was measured** with Sieve on, and no long answer was judged for quality.
- **Perplexity is affected more than sampling.** The mean KL is carried by a few flat states, where
  the dense head spreads probability over more rows than the budget. With top-k sampling those rows
  are never drawn; a perplexity benchmark would count them. Text NLL on a long document was not
  measured.
- **Other sampler settings were not measured:** higher temperatures, and top-p or min-p without
  top-k, where the sampler considers 512 rows and is more likely to reach rows that kept their
  sketch value.
- **The calibration is the model's own writing:** short, single-turn answers without a system
  prompt, tools or thinking, in English except for two test prompts. The observed comparison uses
  the same ten test prompts. Agreement on long contexts, tool calls and thinking traces was not
  measured.
- **Gemma's logit softcap** is applied after the screen. The fit's figures do not include it; the
  observed comparison does.
- **Models over 8B** could not be fitted on this machine.

## Why the gain is what it is

The screen saves part of the cost of the head, so the gain is bounded by the head's share of a
step. That share grows with the vocabulary and shrinks with the depth of the model: the head of
Gemma 3 1B has 262K rows under 26 layers of width 1,152, and the head of Mistral 7B has 33K rows
under 32 layers of width 4,096.

Measured on Qwen3 0.6B, where the dense head costs about 0.5 ms of a 3.4 ms step (timed by leaving
stages out of the full step), the screen recovers about half of the head's cost:

- A sketch narrow enough to be cheap has rows of 64 to 256 bytes. No existing kernel reads such
  rows efficiently: the decode matvec ran a rank-256 sketch at 73 GB/s. `head_sketch` (one thread
  per row) reaches about 115 GB/s, against about 190 GB/s for the dense head.
- Lower ranks are faster and less accurate. An eighth of the hidden size is the lowest rank the fit
  tries.
- The cut needs four small dispatches (about 0.04 ms), and the exact rows about 0.05 ms.

A faster sketch kernel is the remaining lever: at the dense head's byte rate the gain on Qwen3 0.6B
would be about 11% (an estimate, not a measurement).

## Correctness gates

`SieveTests`:

- The kernels against a CPU reference on a synthetic head whose vocabulary is not a multiple of the
  block size: every row carries either the head's logit or the sketch's, on the right side of the cut,
  and the list never exceeds the budget.
- With a budget of every row and a margin that covers them all, the result matches a CPU reference
  of the dense head to 2e-3.
- A tie of every row at the maximum lists nothing (which rows it would list is not deterministic).
- Rows of the head beyond the sampler's vocabulary are neither screened nor written.
- A sidecar with another model's fingerprint, an unknown format, a wrong rank, a row budget above the
  vocabulary, a non-finite margin, or a truncated file is refused before any kernel runs.
- The fingerprint changes when a scale, a bias or a sampled weight changes.
- Text from a sidecar's metadata is cut to printable characters before it is shown, and the writer
  refuses to write through a symbolic link.
- The fit's parts: the directions of a synthetic set of states come out in order of weight, the
  quantized sketch is the one the engine reads back, the three splits do not overlap, and a row
  budget outside the supported list is refused.
- On an installed model, a fit into a temporary directory loads, meets its own report and agrees
  with the dense head.
- On an installed, fitted model: `forward` is bit-identical with Sieve on and off, and the screened
  path gives the same tokens twice.

The last two skip when the model is not installed. `MLXR_SIEVE_OBSERVE=<repo>` runs the generation
comparison of "Measured" for one fitted model (`SieveObserved`, off otherwise).

Not covered by a test: head quantizations other than 4-bit with half-precision scales, and the
softcap branch.

The list of selected rows is built with an atomic counter, so its order varies between runs. Each
row's logit goes to its own slot, so the logits do not.

## Files

- `Sources/MLXEngine/Sieve.swift`: `SieveProfile`, `SieveHeadInfo` with the quality target, the
  sidecar's format and loader, and the fingerprint.
- `Sources/MLXEngine/SieveImport.swift`: the fit (`SieveImporter`, `SieveImportOptions`,
  `SieveImportReport`): calibration, directions, the quantized sketch, the row budget and margin,
  the measurements and the writer.
- `Sources/MLXEngine/Kernels.swift`: `head_sketch`, `head_cut`, `head_hist`, `head_cut2`,
  `head_select` and `head_exact`.
- `Sources/MLXEngine/Transformer.swift`: the Sieve buffers and the screened branch of the sampled
  step; `sieveInfo` / `sieveOffReason`.
- `Sources/MLXEngine/Engine.swift`: `load(..., sieve:)`, the memory check with its fallback to the
  dense head, `fitSieve`, and the Sieve fields of `LoadedModelInfo`.
- `Sources/MLXRCLI/SieveCommand.swift`: `mlxr sieve` and the load summary. `Commands.swift`:
  `--sieve` on `run` and `serve`.
- `Sources/MLXRunner/AppState.swift`, `ChatView.swift`: the Sieve setting, the fit and the
  inspector's Sieve section.
- `Tests/MLXEngineTests/SieveTests.swift`: the tests above.
