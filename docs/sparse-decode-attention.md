# Sparse decode attention

At long contexts, decoding one token reads the whole KV cache of every layer. Sparse decode
attention reads only part of it. Each full-attention layer's decode query attends to:
- the first few keys (the sink);
- the most recent keys (the window);
- the long-range keys whose 4-key blocks score highest against the query.

It is training-free and works on every architecture the engine supports.

Status:
- **On by default** from 5,128 tokens of context. Below that, decode is dense and output is unchanged.
  It can be turned off in the app, with `--sparse off`, or with `MLXR_SPARSE_ATTN=0`.
- **It changes outputs** past that context length. That is an exception to the token-identical
  default path that AGENTS.md otherwise requires.
- **Speed** (M2 Pro, 4-bit, default budget): 1.06–1.4× dense decode at 8K, 1.2–2.1× at 16K and
  1.4–3.0× at 30K, on every architecture.
- **Quality:** on real text the perplexity cost is at most about 3% on every model, at every context
  measured, except Gemma 4 E2B (+8% at 30K) and, to a lesser degree, Gemma 1.1 (KL 0.08). See
  "Every architecture".

## Using it

**In the app:** the chat inspector's **Long context** section has three settings:
- **Sparse attention**, on or off;
- **Long-range keys**: 1,024, 2,048, 4,096 or 8,192;
- **Key selection**: per KV head, or shared.

Like context length, they apply the next time a model is loaded. The inspector's **Model** section
shows how many layers run sparse and from which context length.

**On the command line:** `mlxr run` and `mlxr serve` take `--sparse on|off|N`, where N is the
long-range key budget. Without it they use `MLXR_SPARSE_ATTN`, and print the result on load:

```bash
mlxr run <model> --sparse off ...
mlxr serve <model> --sparse 4096 ...
```

**Environment variables:**

| Variable | Meaning | Default |
|---|---|---|
| `MLXR_SPARSE_ATTN` | `0` turns it off, `1` is the default budget, a larger number is the long-range key budget | on (2,048 keys) |
| `MLXR_SPARSE_WINDOW` | recent keys always attended | 512 |
| `MLXR_SPARSE_MIN` | decode stays dense below this context length | 2 · (sink + topK + window) = 5,128 |
| `MLXR_SPARSE_SHARED` | `1`: one selection shared by every head instead of one per KV head | per KV head |

The sink is 4 keys and the long-range budget defaults to 2,048 keys (512 blocks of 4). With the
default budget a step attends to 2,564 keys whatever the context length.

**In code:** `InferenceEngine.load(directory:draftDirectory:contextLength:sparse:sieve:)` takes a
`SparseAttention?`; nil means dense. If the index would not fit in memory at the requested context
length, the model loads with dense decode and `LoadedModelInfo.sparseNote` says why. A
`Transformer` set up with sparse attention can switch per step through `Transformer.sparse`. One
loaded without it allocates nothing, and setting `sparse` on it has no effect.

**Choosing the selection.** Per-head selection (the default) was better than shared or level on every
model compared: Qwen3 0.6B and 8B, and Mistral 7B (see "Per-head and shared selection").

### Coverage

Every architecture runs sparse decode on all of its full-attention layers. Sliding-window layers are
already bounded by their window, so they stay dense.

| Model | Layers | Full-attention | Sparse |
|---|---|---|---|
| Qwen3 0.6B / 8B / 14B | 28 / 36 / 40 | all | all |
| Qwen2.5 0.5B | 24 | 24 | 24 |
| Llama 3.2 1B | 16 | 16 | 16 |
| Mistral 7B | 32 | 32 | 32 |
| Phi-3.5 mini | 32 | 32 | 32 |
| Gemma 1.1 2B | 18 | 18 | 18 |
| Gemma 2 2B | 26 | 13 | 13 |
| Gemma 3 270M / 1B / 4B | 18 / 26 / 34 | 3 / 4 / 5 | 3 / 4 / 5 |
| Gemma 4 E2B | 35 | 7 | 7 |
| Gemma 4 12B | 48 | 8 | 8 |

Gemma 4 E2B indexes 3 layers; its 4 KV-shared full-attention layers use their source layer's index.

- **Shapes with split and sparse kernels:**
  - head sizes 64, 96 and 128;
  - head sizes 256 and 512 in sub-groups;
  - for head size 128, G = 1–5 query heads per KV head. That includes Qwen3 / Qwen2.5 14B. G = 8
    (Qwen3 / Qwen2.5 32B) is not instantiated: its split kernel was not bit-identical to the
    single-threadgroup kernel. G = 5 also made Qwen3 14B's dense long-context decode 20% faster at
    8K and 27% at 16K, with bit-identical output.

  Other head sizes stay dense.
- **Soft-capped attention** (Gemma 2) has its own kernel instantiations.
- **Steps that stay dense:** prefill, speculative verification passes and draft models.
- **Memory.** The index adds an eighth of each eligible layer's KV cache, about 11% less maximum
  context on a 16 GB Mac. The scores and per-head lists add about 1.25 × context × KV heads × 4
  bytes.

## How it works

Per decode step, for each eligible layer:

1. **`sparse_index`.** Whenever positions are written (prefill or decode), each block of 4 keys they
   complete gets an index key: the mean of the 4 post-RoPE keys, per KV head, in f16. A rewritten
   position (a new turn, a rejected draft) rebuilds its block when the block's last position is
   written again.
2. **`sparse_score`.** The decode query scores every block between the sink and the window.
   - Per KV head (the default): the score is the sum of that head's query heads' `q · k̄`. Each KV
     head maps to its own group of lanes, so this costs the same as the shared score.
   - Shared: the score is summed over every head instead.
3. **`sparse_select`.** One threadgroup per KV head (shared: one) keeps the exact top `topK / 4`
   blocks. It uses a radix select on the score bits, and ties go to the earlier block. It writes the
   attended positions in ascending order: sink, kept blocks, window.
4. **`attention_sparse` + `attention_combine`.** This is `attention_split`, with key n of the KV
   head's list in place of position n. With the list `0..<L` it is bit-identical to
   `attention_split`, which is itself bit-identical to the single-threadgroup decode kernel for head
   sizes up to 128.

Relation to *Latent Sparse Attention* (arXiv 2610.42147v2, training-free variant):

| Paper | Here |
|---|---|
| Sink + window + top-k long-range keys | Same |
| One index list shared by all heads | One list per KV head by default; the paper's single list with `MLXR_SPARSE_SHARED=1` |
| Index keys pooled 4:1, projected to 64 dimensions | Pooled 4:1 (mean), full head size, f16 |
| Ward clustering, flat or tree search | A flat scan over blocks, exact top-k |
| Sparse prefill | Prefill stays dense |
| Distillation / native training | None |

### Invariants

- **Off means untouched.** With sparse attention off (`MLXR_SPARSE_ATTN=0`, `--sparse off`, the app
  setting, or `sparse: nil`) there are no sparse buffers, no sparse dispatches, and outputs are
  bit-identical to dense decode. With it on, decode below `minContext` is dense too.
- **Bounded indices.** Every GPU index derives from host-checked values:
  - the plan guarantees `count <= L <= maxSeq`;
  - each KV head's list sits `listStride` (the context length) entries apart in a buffer sized for
    the largest KV head count;
  - `attention_sparse` clamps every list entry to the context.
- **Stale blocks are rebuilt.** Any block whose positions are rewritten has its index rebuilt. See the
  comment above the `sparse_index` dispatch in `Transformer.encode`.

## Performance and accuracy

**Setup.**
- M2 Pro (16-core GPU, 197 GB/s), 16 GB, 4-bit weights, nothing else on the GPU, 2026-11-06.
- Default budget, per-head selection.
- Prompt: *Pride and Prejudice* (Project Gutenberg #1342, about 177K tokens, so no context repeats
  text), starting with BOS where the tokenizer has one. (Without it Gemma produces nonsense, and the
  first measurements of Mistral and Gemma were redone with it.)

**Columns.**
- **Mean KL** is KL(dense‖sparse) over 512 positions of the text after the prompt. Both modes are
  teacher-forced on the text, and the cache stays dense decode's.
- **NLL Δ** is the change in the text's negative log-likelihood, in nats per token. +0.01 is about
  1% perplexity.
- **Decisive** is top-1 agreement at positions where dense decode's top-1/top-2 logit margin is at
  least 1 nat.

**Runs.** Mostly one run each. Repeats agreed within 2%, except for a few dense lines that were 4–7%
slow.

| Model | Context | Dense tok/s | Sparse tok/s | Speed-up | Mean KL | NLL Δ | Top-1 | Decisive |
|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | 8K | 113.2 | 144.5 | 1.28× | 0.001–0.003 | −0.004 / −0.015 | 96–97% | 433/433 |
| Qwen3 0.6B | 16K | 71.0 | 130.7 | 1.84× | 0.004–0.005 | −0.011 / −0.019 | 96–97% | 485/485 |
| Qwen3 0.6B | 30K | 43.8 | 103.5 | 2.36× | 0.010–0.013 | +0.016 / +0.012 | 92–94% | 492/493 |
| Qwen3 8B | 8K | 29.0 | 32.3 | 1.11× | 0.006 | −0.012 | 97.3% | 400/400 |
| Qwen3 8B | 16K | 23.5 | 31.2 | 1.33× | 0.011 | +0.016 | 95.5% | 320/320 |
| Qwen3 8B | 30K | 17.8 | 29.0 | 1.63× | 0.011 | +0.007 | 95.1% | 304/304 |
| Qwen3 14B | 8K | 17.5 | 19.1 | 1.09× | 0.002 | −0.007 | 99.0% | 422/422 |
| Qwen3 14B | 16K | 14.9 | 18.7 | 1.25× | 0.009 | +0.014 | 93.8% | 359/361 |
| Mistral 7B | 8K | 31.6 | 34.9 | 1.10× | 0.014* | −0.007 | 97.3% | 363/364 |
| Mistral 7B | 16K | 25.4 | 33.9 | 1.33× | 0.004 | +0.004 | 96.3% | 371/371 |
| Mistral 7B | 30K | 19.7 | 31.5 | 1.60× | 0.011 | −0.014 | 94.5% | 302/302 |

Notes:
- **Qwen3 0.6B** is measured at two places in the book (from token 0 and from token 60,000).
- **Mistral 7B at 8K (\*):** the mean KL is one outlier position (5.2 nats); the median is 0.0000.
- **Qwen3 14B at 30K** was not run. Weights, KV cache and index come to about 14 GB.
- **Needle recall.** Five passkeys at 10–90% depth. Sparse decode recalled as many as dense decode in
  every run but two, and more in two (Llama 3.2 1B at 30K, Phi-3.5 at 16K). The two exceptions are
  Gemma 3 1B at 8K and 16K, with one needle fewer each.
- **Near-ties.** Almost every top-1 disagreement is at a position where dense decode itself was
  nearly undecided. The "Decisive" column shows agreement elsewhere is effectively complete.
- **Memorisation.** Dense NLL on the book is 3.35–3.66 nats for Qwen3 0.6B but 0.9–2.2 for the 7B+
  models. They have probably memorised parts of it, which flatters their agreement.
- **Prefill** is dense in both modes, at 8K / 16K / 30K:
  - Qwen3 0.6B: 1,065 / 611 / 363 tok/s;
  - Qwen3 8B: 178 / 135 / 99 tok/s;
  - Qwen3 14B: 100 / 80 tok/s (8K / 16K only);
  - Mistral 7B: 185 / 142 / 107 tok/s.

### Every architecture

The rest of the supported architectures, with the same setup.
- **Gemma 1.1 and Gemma 2** support 8K tokens, so they were measured at 7,000.
- **Phi-3.5 mini** has a large KV cache (384 KB per token), so it stops at 16K on 16 GB.
- **Gemma** gains less because only some of its layers attend over the full context; the rest use a
  sliding window (see "Coverage").
- **Speeds** are from a run with nothing else on the GPU. **Accuracy** is from the runs with BOS and,
  for Gemma, the corrected tokenizer (see the CHANGELOG).

| Model | Context | Dense tok/s | Sparse tok/s | Speed-up | Mean KL | NLL Δ | Top-1 | Decisive |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5 0.5B | 8K | 137.7 | 188.3 | 1.37× | 0.002 | +0.004 | 98.0% | 234/234 |
| Qwen2.5 0.5B | 16K | 87.1 | 179.9 | 2.07× | 0.007 | −0.007 | 95.9% | 202/202 |
| Qwen2.5 0.5B | 30K | 54.5 | 163.8 | 3.01× | 0.015 | +0.005 | 91.4% | 232/232 |
| Llama 3.2 1B | 8K | 123.1 | 149.2 | 1.21× | 0.002 | −0.009 | 98.4% | 275/275 |
| Llama 3.2 1B | 16K | 89.5 | 144.4 | 1.61× | 0.008 | +0.016 | 96.1% | 219/219 |
| Llama 3.2 1B | 30K | 59.4 | 126.4 | 2.13× | 0.017 | +0.012 | 94.3% | 217/217 |
| Phi-3.5 mini | 8K | 31.4 | 39.2 | 1.25× | 0.001 | −0.003 | 99.6% | 349/349 |
| Phi-3.5 mini | 16K | 20.4 | 33.3 | 1.63× | 0.006 | +0.001 | 96.9% | 279/279 |
| Gemma 1.1 2B | 7K | 60.1 | 65.6 | 1.09× | 0.078 | +0.036 | 93.8% | 316/321 |
| Gemma 2 2B | 7K | 60.2 | 65.8 | 1.09× | 0.000 | −0.001 | 99.4% | 313/313 |
| Gemma 3 1B | 8K | 152.6 | 162.0 | 1.06× | 0.002 | +0.004 | 98.6% | 351/351 |
| Gemma 3 1B | 16K | 134.7 | 161.0 | 1.20× | 0.003 | +0.027 | 97.5% | 329/329 |
| Gemma 3 1B | 30K | 113.0 | 158.1 | 1.40× | 0.005 | +0.020 | 94.5% | 271/271 |
| Gemma 4 E2B | 8K | 60.4 | 65.0 | 1.08× | 0.068 | −0.049 | 92.6% | 390/396 |
| Gemma 4 E2B | 16K | 53.1 | 64.5 | 1.21× | 0.150 | −0.024 | 86.9% | 367/384 |
| Gemma 4 E2B | 30K | 43.8 | 63.5 | 1.45× | 0.191 | +0.079 | 85.2% | 362/383 |

**Gemma 4 E2B is the outlier, and Gemma 1.1 the lesser one.** Gemma 4 E2B's KL is 5–20× the other
models', it disagrees at 4–6% of the positions where dense decode is decisive (others: under 1%,
Gemma 1.1: 1.6%), and perplexity rises 8% at 30K (Gemma 1.1: 3.6% at 7K). The NLL
change swings from −0.05 to +0.08, and the dense model itself scores poorly on this text (4.3–5.5
nats against 2–3.5 for models of similar size, and mlx-lm agrees: 4.9). So the signal is noisy,
but the KL and decisive-position figures are not. Possible causes, unverified:
- 4 of its 7 full-attention layers share their KV cache with another layer, and select with that
  layer's index;
- its 512-wide global heads are scored and attended in sub-groups.

To turn sparse attention off for it, use the inspector's switch or `--sparse off`.

### Against mlx-lm

mlx-lm 0.32.2 with mlx 0.32.4 (the latest on PyPI), on the same machine, model files and text,
2026-11-06:
- Each run prompts mlx-lm with the first N tokens of the book, as tokenized by its own tokenizer, and
  generates 64 greedy tokens with `stream_generate`.
- The table shows mlx-lm's own `generation_tps` and `prompt_tps`, from a single run.
- Our figures are from the runs in the table above.
- mlx-lm has no sparse attention. Its option for long contexts is a quantized KV cache (`kv_bits`).

Decode, tok/s:

| Model | Context | mlx-lm | Ours, dense | Ours, sparse | Sparse / mlx-lm |
|---|---|---|---|---|---|
| Qwen3 0.6B | 8K | 105.5 | 113.2 | 144.5 | 1.37× |
| Qwen3 0.6B | 16K | 68.3 | 71.0 | 130.7 | 1.91× |
| Qwen3 0.6B | 30K | 43.6 | 43.8 | 103.5 | 2.37× |
| Qwen3 8B | 8K | 31.2 | 29.0 | 32.3 | 1.04× |
| Qwen3 8B | 16K | 25.2 | 23.5 | 31.2 | 1.24× |
| Qwen3 8B | 30K | 19.7 | 17.8 | 29.0 | 1.47× |
| Qwen3 14B | 8K | 18.1 | 17.5 | 19.1 | 1.06× |
| Qwen3 14B | 16K | 15.7 | 14.9 | 18.7 | 1.19× |
| Mistral 7B | 8K | 34.7 | 31.6 | 34.9 | 1.01× |
| Mistral 7B | 16K | 28.3 | 25.4 | 33.9 | 1.20× |
| Mistral 7B | 30K | 22.1 | 19.7 | 31.5 | 1.43× |

- **Sparse decode** is level with mlx-lm at 8K on 7–14B models and ahead everywhere else: 1.2–1.9×
  at 16K and 1.4–2.4× at 30K. It changes output, though, and mlx-lm's dense decode does not.
- **Our dense decode** is behind mlx-lm by 3–11% on the 7–14B models. That is the gap in reading the
  weights that the earlier performance passes recorded. On Qwen3 0.6B it is level or ahead.
- **mlx-lm's quantized KV cache at 30K.** On Qwen3 0.6B it decodes 53.4 tok/s with `kv_bits=8` and
  65.0 with `kv_bits=4`, against our sparse 103.5. Its peak memory is 8.2 GB, against 4.3 GB for its
  dense cache. On Qwen3 8B and Mistral 7B both settings ran the GPU out of memory at 30K on this 16 GB
  Mac, so they have no figure.

Prefill, tok/s. It is dense in both engines, and at long contexts mlx-lm is well ahead:

| Model | Context | mlx-lm | Ours | mlx-lm / ours |
|---|---|---|---|---|
| Qwen3 0.6B | 8K / 16K / 30K | 1,579 / 1,115 / 712 | 1,065 / 611 / 363 | 1.5 / 1.8 / 2.0× |
| Qwen3 8B | 8K / 16K / 30K | 195 / 171 / 139 | 178 / 135 / 99 | 1.1 / 1.3 / 1.4× |
| Qwen3 14B | 8K / 16K | 107 / 97 | 100 / 80 | 1.1 / 1.2× |
| Mistral 7B | 8K / 16K / 30K | 272 / 241 / 202 | 185 / 142 / 107 | 1.5 / 1.7 / 1.9× |

The gap grows with the prompt length, so it is most likely in our prefill attention (`flash_prefill`), not the
GEMMs. Ingesting a 30K-token prompt takes about 83 s on Qwen3 0.6B and 280 s on Mistral 7B, against
42 and 149 s with mlx-lm. At long contexts this is now a larger cost than decode.

### Per-head and shared selection

| Model | Context | Selection | Mean KL | NLL Δ | Top-1 |
|---|---|---|---|---|---|
| Qwen3 0.6B | 16K (two regions) | shared | 0.035 / 0.012 | +0.022 / +0.004 | 91.4% / 93.4% |
| Qwen3 0.6B | 16K (two regions) | per head | 0.004 / 0.005 | −0.011 / −0.019 | 96.1% / 96.5% |
| Qwen3 0.6B | 30K (two regions) | shared | 0.033 / 0.046 | +0.049 / +0.053 | 88.9% / 88.3% |
| Qwen3 0.6B | 30K (two regions) | per head | 0.010 / 0.013 | +0.016 / +0.012 | 93.9% / 92.2% |
| Qwen3 8B | 30K | shared | 0.015 | +0.014 | 93.2% |
| Qwen3 8B | 30K | per head | 0.011 | +0.007 | 95.1% |
| Mistral 7B | 30K | shared | 0.014 | +0.011 | 94.3% |
| Mistral 7B | 30K | per head | 0.011 | −0.014 | 94.5% |

Speed is the same in both modes.

On Qwen3, per-head selection cuts KL 1.3–9× and the perplexity cost 2–4.5×. On Mistral 7B it is
level to slightly better (one run at 30K).

An earlier Mistral run, whose prompts lacked BOS, had shared selection ahead (KL 0.017 against
0.040). With BOS that reversal disappeared, so it was an artifact of the out-of-distribution prompt.

### Where the time goes

GPU time for one sparse layer at Qwen3 0.6B's shapes (8 KV heads, head size 128). Each layer has its
own cache and index, so nothing stays cached between layers.

| Context | Score | Select | Attention + combine | Sparse layer | Dense split + combine |
|---|---|---|---|---|---|
| 8K | 34 µs | 15 µs | 76 µs | 126 µs | 196 µs |
| 30K | 106 µs | 33 µs | 78 µs | 217 µs | 690 µs |

217 µs × 28 layers = 6.1 ms per token, against 6.2 ms measured end to end over the 3.3 ms weight
read.

Bytes dominate:
- Scoring reads the index: 15 MB per layer at 30K, at about 142 GB/s.
- Attention reads the 2,564 kept keys: 10.5 MB, at about 157 GB/s (dense attention: 179 GB/s).
- Attention and combine have a fixed floor of about 8 µs per layer.

The selection is a small and nearly fixed cost. With almost nothing attended, sparse mode costs
about 50 µs per layer even at a 16-token context: the selection plus two extra dependent dispatches.

**Budget.** Attention cost grows with the number of attended keys. With 4,096 keys and a 1,024-key
window (about 5.1K attended, shared selection), Qwen3 0.6B at 30K ran at 90.6 tok/s against 104.7 at
the default budget. Mistral 7B at 30K ran at 29.7 against 31.6. With 12,288 keys at 16K, sparse was
slower than dense on Qwen3 0.6B.

### Verification passes

A dense verification pass of T tokens (speculative decoding) at long context, GPU ms per pass:

| Model | Context | T = 1 | T = 2 | T = 5 | T = 8 |
|---|---|---|---|---|---|
| Qwen3 0.6B | 30K | 23.2 | 31.3 | 71.1 | 105.2 |
| Qwen3 8B | 30K | 56.1 | 96.6 | 202.0 | 273.3 |
| Qwen3 14B | 16K | 67.3 | 110.5 | 229.5 | 277.9 |
| Mistral 7B | 30K | 50.9 | 84.8 | 182.4 | 245.9 |

T = 5 costs 3.1–3.6× T = 1, because each verification token reads the KV cache largely on its own.

## Measuring

Nothing else may use the GPU while measuring.

```bash
# dense decode at long contexts
MLXR_SPARSE_ATTN=0 MLXR_BENCH_DECODE=1 MLXR_BENCH_CTX=8192,16384,30000 mlxr bench <model>
# dense against sparse, with agreement on real text from two places in it
MLXR_BENCH_TEXT=book.txt MLXR_BENCH_TEXT_OFFSETS=0,60000 \
  MLXR_BENCH_DECODE=1 MLXR_BENCH_CTX=8192,16384,30000 mlxr bench <model>
```

For each context, the `MLXR_BENCH_CTX` section of `mlxr bench` prints:
- **Speed.** Prefill time, then `dense @N` and `sparse @N` greedy decode rates over 64 steps (wall
  and GPU ms per token).
- **Greedy prefix.** How many greedy tokens sparse decode shares with dense decode before the first
  difference.
- **Agreement on dense decode's tokens.** Teacher-forced argmax agreement with dense decode's
  greedy tokens, plus a control (dense forward against the GPU sampler's tokens) that must be 64/64.
- **Text metrics** (with `MLXR_BENCH_TEXT`). KL, the text's NLL under both modes, and top-1 agreement
  bucketed by dense decode's top-1 margin. These run over `MLXR_BENCH_POSITIONS` positions (default
  512). Each position runs sparse first and dense second, so the cache stays dense decode's.
- **Verification cost.** A dense verification pass of T = 1, 2, 5 and 8 tokens.
- **Needles** (with `MLXR_BENCH_TEXT`). Five passkeys written into the prompt at 10–90% depth and
  recalled in both modes. The question is prefilled dense except its last token, so the answer
  starts in the mode being measured.

`MLXR_BENCH_TEXT_OFFSETS` repeats the measurements from other places in the text, which separates
context length from text region. Speed runs only at the first offset.

The context must leave room for the measured positions: context + max(64, positions) + 96 must be
below the model's maximum (Mistral 7B: 32,768). Without `MLXR_BENCH_TEXT` the prompt is
pseudo-random tokens, which is fine for speed but meaningless for agreement.

Other measurements:
- **Per-kernel GPU time:** `MLXR_KERNEL_TIMING=1 swift test --filter SparseTimingTests`. It times
  the score, select and attention kernels and a whole layer step at Qwen3 0.6B's shapes, at 8K and
  30K. Add `MLXR_SPARSE_SHARED=1` to time the shared selection.
- **Correctness tests:**
  - `swift test --filter SparseAttentionTests`: the plan, the index, per-head and shared scores
    against a CPU reference, the selection, and attention over shared and per-head lists.
  - `swift test --filter sparseDecode`: full-budget sparse decode, in both modes, gives logits
    bit-identical to dense on every installed model.
  - The Qwen3 14B cases run only with `MLXR_TEST_LARGE=1`, since alongside the Gemma suite they push
    a 16 GB Mac into swap.

## Tried and ruled out

Measured with the kernel timing; none changed the output.
- **Aggregating the selection histogram per simdgroup.** Scores share their top bits, so most keys
  land in one histogram bin. Leader lanes adding counts with `simd_min` / `simd_sum` made the
  selection slower: 37 µs against 24–27 µs at 30K. Atomic contention is not the bottleneck.
- **Prefetching list entries in `attention_sparse`.** Each lane loaded one of the next 32 entries and
  `simd_shuffle` handed them out. This gave 76.8 µs against 78.2 µs, i.e. no change.
- **Fewer blocks per scoring simdgroup** (`SSC_BPS` 1, 2 or 4 instead of 8): within noise.
- **A larger budget to fix agreement.** On Qwen3 0.6B at 16K with shared selection, top-1 misses did
  not respond to the budget up to 8,192 keys. Per-head selection fixed what the budget could not.

## Open questions and next steps

Roughly by expected value.

1. **Gemma 4 E2B's quality.** Find out why its sparse decode is lossier than every other model's:
   the KV-shared layers, the 512-wide heads in sub-groups, or the model itself. Until then, consider
   leaving it dense by default.
2. **Blend per-head and shared scores.** For example, normalise each query head's score before
   summing within the group, or mix in the shared score. Per-head is already better or level on every
   model compared, so this is about the 4–8% of misses that remain at 30K.
3. **A broader evaluation.** Use a text the models cannot have memorised (code, or something
   recent), more needle depths and more than one run.
4. **Reuse a selection** across 2–4 decode steps, with `attention_sparse` also covering positions
   written since the last selection, or across adjacent layers (TidalDecode, OmniKV). This removes
   most scoring and selection work: an estimated 1.6–3.5 ms per token on Qwen3 0.6B at 30K.
5. **A smaller index.** int8 with a per-row scale, or pooling 8:1, would halve the scoring bytes (about
   −1 ms at 30K on Qwen3 0.6B) and the index memory. Later options:
   - a channel subset ranked by `q_norm · k_norm` (the Double Sparsity approach);
   - Quest-style min/max bounds, for single "needle" keys that a 4-key mean dilutes.
6. **A multi-query verification kernel.** It would read K/V once for all T tokens. That makes
   sparse-drafted, dense-verified self-speculation (MagicDec, TriForce) possible, which is greedy
   token-identical to dense and could become default behaviour. With the agreement measured above as
   the acceptance rate, the estimate is about 1.4× dense at 30K on Qwen3 0.6B. 7–8B models also need
   a memory-bound batched matvec.
7. **Load draft models with sparse attention** (`Engine.swift`, where drafts load with
   `sparse: nil`). Verification keeps the output exact. Today speculation with a sparse model is
   slower than plain sparse decode at long context.
8. **An int8 or int4 KV cache** (KIVI; mlx-lm's `QuantizedKVCache`), opt-in. It cuts every
   long-context read, composes with sparsity, and would let Qwen3 14B hold 30K in 16 GB. It touches
   every attention kernel and prefill.
9. **Scale `MLXR_SPARSE_MIN` with model size.** At 8K a 7–14B model gains only about 10%.
10. **Prefill.** It is still O(T²), and at 30K mlx-lm prefills 1.4–2× faster than we do (see "Against
   mlx-lm"). At 30K attention is most of Qwen3 0.6B's prefill time.
   - Lossless headroom: `flash_prefill` runs well below the GEMM tile's throughput.
   - Opt-in sparse prefill (XAttention) would need a selection per query block.
11. **Server option.** Sparse attention is set per load, so a server request cannot ask for dense
    decode: `mlxr serve --sparse off`, or the app setting, applies to every request.

## Files

- `Sources/MLXEngine/SparseAttention.swift`: the public config (`SparseAttention`, environment
  parsing), the per-step plan, eligibility and memory accounting.
- `Sources/MLXEngine/Kernels.swift`: `SparseParams`, `sparse_index`, `sparse_score`, `sparse_select`,
  `attention_sparse` and its instantiations.
- `Sources/MLXEngine/Transformer.swift`:
  - `init(directory:contextLength:sparse:sieve:)`, `sparseLayerCount`;
  - the index, score and list buffers;
  - index maintenance after `qkv_post`;
  - the sparse branch of decode attention;
  - `sparse` / `sparseAvailable`.
- `Sources/MLXEngine/GPU.swift`: the Swift `SparseParams`.
- `Sources/MLXEngine/Engine.swift`: `load(..., sparse:)`, the memory check with its dense fallback,
  and the sparse fields of `LoadedModelInfo`.
- `Sources/MLXRunner/AppState.swift`, `ChatView.swift`: the app settings and the inspector's Long
  context section.
- `Sources/MLXRCLI/Commands.swift`: `--sparse` on `run` and `serve`. `Sources/MLXRCLI/Legacy.swift`:
  `LongBench` and `benchLongContext` (the long-context bench section).
- `Tests/MLXEngineTests/SparseAttentionTests.swift`: kernel tests against CPU references.
- `Tests/MLXEngineTests/ModelTests.swift`: end-to-end tests on installed models.
- `Tests/MLXEngineTests/SparseTimingTests.swift`: opt-in kernel timing.
