# Changelog

## Unreleased

Qwen 3.5 support: the hybrid Gated DeltaNet architecture. Decode is 6–7% ahead of mlx-lm on the 0.8B model, 2% behind on the 2B and 5% behind on the 4B; prefill is ahead.

- **Qwen 3.5** (`qwen3_5`, `qwen3_5_text`; the dense models; verified on the 0.8B, 2B and 4B 4-bit
  exports). Three of every four layers are
  Gated DeltaNet linear attention: a causal convolution and a recurrent state per head instead of a
  KV cache. The others are full attention with an output gate and partial RoPE (64 of 256 dims).
  - New Metal kernels: decode runs each linear-attention layer's convolution, q/k normalization,
    gates, recurrence and gated output norm in one dispatch; prefill uses four batch kernels with the
    same arithmetic, so prefill and token-by-token decode agree bit for bit in those layers.
  - Logits match mlx-lm (float32) within 0.004 at every position of a 70-token text, with the same
    argmax everywhere; the tokenizer and the chat template match transformers.
  - Thinking toggle: with thinking on, the prompt opens the `<think>` block and the reply starts with
    the tag, so the app's "Thought for N s" block and the API's `reasoning_content` work as for Qwen3.
  - Thinking follows each model's own default: the 0.8B and 2B answer directly unless thinking is
    turned on (their templates make it opt-in, and Qwen's model cards warn that it often loops on them:
    with the app's default sampling, 4 of 6 seeded replies on the 0.8B and 5 of 6 on the 2B were still
    thinking after 4,096 tokens; mlx-lm: 3 and 6 of 6). The 4B and larger think unless it is turned off.
  - Chat follow-ups: the recurrent state cannot be rolled back like a KV cache, so the engine saves a
    checkpoint of it 8 tokens before the end of each prompt (before the part the next turn renders
    differently) and the next turn resumes there, with output identical to a fresh run.
  - Long contexts: sparse decode attention runs on the full-attention layers. On real text at 16K
    tokens decode goes from 151 to 215 tok/s with a mean KL of 0.003 nats to dense decode and 5/5
    passkeys found.
  - Not yet: tool calling (Qwen 3.5 uses an XML `<function=…>` format; tool requests get a 400),
    mixture-of-experts variants, multi-token prediction. Speculative decoding is off for these models.
- **Speed** on an M2 Pro against mlx-lm 0.32.2 (alternating runs of `mlx_lm.benchmark` and `mlxr bench`):
  - Qwen3.5-0.8B-4bit: decode 270.5 vs 253.5 tok/s at 16 tokens of context (+6.7%), 260.0 vs 244.5 at
    1,024 (+6.3%); a 256-token chat reply 269.4 vs 251.6 tok/s (`mlxr run` vs `mlx_lm.generate`, +7.1%);
    prefill of 512 tokens 2,780 vs 2,420 tok/s.
  - Qwen3.5-2B-4bit: decode 133.1 vs 135.5 tok/s (−1.8%), 130.5 vs 133.7 at 1,024; prefill 1,158 vs 1,030.
  - Qwen3.5-4B-4bit: decode 62.3 vs 65.7 tok/s (−5%), 61.4 vs 64.5 at 1,024 (hidden size 2,560: none of
    the K = 1,024 or paired kernels apply; the larger matvecs lag mlx-lm as on 7–14B models).
  - Where the time goes: the linear-attention layers' recurrent state runs at the memory bandwidth floor
    (0.19 ms per token for 18 layers); the rest is the 4-bit matvecs and about 200 dependent dispatches.
- **Kernels** (hybrid models only unless noted; greedy output unchanged by each):
  - The 4-bit decode matvecs read each row's quantization scales and biases as interleaved pairs built at
    load (one stream instead of two arrays); for the hidden-size projections (K = 1,024) a kernel issues
    a block's loads before its arithmetic and the next block's before finishing the current one.
  - Long-context decode attention for head size 256 with 4 query heads per KV head: each lane reads 8
    contiguous dims (one 16-byte load per key), 64 virtual simdgroups per KV head, and the output gate
    applied in the combine kernel (+1–3% at 1,024 tokens).
  - GPU token selection, every model: a first pass over blocks of 256 logits on the whole GPU, so the
    one-threadgroup selection scans only the blocks that can hold a candidate (the default sampler went
    from 3.7 to about 2.2 vocabulary scans per token on Qwen 3.5's 248K vocabulary). The selected tokens
    are unchanged.
- Multimodal exports no longer copy their vision tower or multi-token-prediction weights into GPU
  memory (201 MB less for Qwen3.5-0.8B; Gemma 3 4B also benefits).
- **Thinking default per model.** A chat that doesn't say whether to think gets the template's
  default, as with transformers and mlx-lm: an API request without `enable_thinking`,
  `chat_template_kwargs` or a reasoning effort, `mlxr run` without `--think`/`--no-think` (new), and
  the app's Think toggle, which now remembers its setting per model. Settings saved by earlier
  versions with thinking off keep it off for every model. Qwen3 thinks by default, as before. A
  `reasoning_effort` other than `none` now turns thinking on.
- **App settings survive a relaunch.** Loading the saved settings at launch wrote defaults over the
  ones not read yet (system prompt, context length, draft model, sparse attention, last model), so only
  the sampling settings were kept.

## 1.4.0

Sparse decode attention for long contexts, on by default; faster long-context decode on Qwen3 14B; and
fixes to Gemma tokenization and raw prompts.

- **Sparse decode attention**, on by default from 5,128 tokens of context
  ([docs/sparse-decode-attention.md](docs/sparse-decode-attention.md)).
  - **How it works.** Each full-attention layer keeps a mean key per block of 4 keys. Each KV head
    scores those blocks against its query heads and attends to the first 4 keys, its 2,048
    best-scoring earlier keys and the latest 512, not the whole cache.
  - **Speed** on an M2 Pro (4-bit), against dense decode:
    - Qwen3 0.6B: 43.8 → 103.5 tok/s at 30K;
    - Qwen3 8B: 17.8 → 29.0 tok/s at 30K;
    - Mistral 7B: 19.7 → 31.5 tok/s at 30K;
    - across every model measured, 1.06–1.4× at 8K, 1.2–2.1× at 16K and 1.4–3.0× at 30K.
  - **It changes output** past 5,128 tokens. Perplexity on real text rises by at most about 3% on every
    model measured except Gemma 4 E2B (+8% at 30K, and clearly lossier at every length) and Gemma 1.1
    (+3.6% at 7K). Top-1
    agreement with dense decode is 91–99% elsewhere, and almost all disagreements are where dense
    decode was nearly undecided. Below 5,128 tokens decode is dense and output is unchanged.
  - **Settings.**
    - App: the inspector's new **Long context** section (on/off, the long-range key budget, and
      per-KV-head or shared selection). The **Model** section shows the layers that run sparse.
    - CLI: `mlxr run` / `mlxr serve --sparse <on|off|N>`.
    - Environment: `MLXR_SPARSE_ATTN` (0 turns it off), `MLXR_SPARSE_WINDOW`, `MLXR_SPARSE_MIN`,
      `MLXR_SPARSE_SHARED`.
  - **Against mlx-lm 0.32.2** (dense attention; it has no sparse mode), on the same machine and
    files:
    - Sparse decode is level at 8K on 7–14B models and 1.4–2.4× faster at 30K. Qwen3 8B at 30K runs
      at 29.0 tok/s against 19.7; Mistral 7B at 30K at 31.5 against 22.1.
    - Our dense decode is 3–11% behind mlx-lm on 7–14B models.
    - Prefill is 1.4–2× slower than mlx-lm at 30K.
    - mlx-lm's quantized KV cache reaches 65 tok/s on Qwen3 0.6B at 30K (our sparse: 103.5). It ran
      out of GPU memory on the 7–8B models at 30K on 16 GB.
  - **Memory.** If the index (an eighth of each full-attention layer's KV cache) does not fit, the
    model loads with dense decode.
  - **Coverage.** Every architecture: Qwen3, Qwen2.5, Llama, Mistral, Phi-3 and Gemma 1–4.
    Sliding-window layers stay dense. Prefill, speculative verification and draft models stay dense.
- **Qwen3 14B decodes long contexts 20–27% faster** (8K: 14.6 → 17.5 tok/s; 16K: 11.0 → 14.0) with
  identical output. Its 5 query heads per KV head had no split-attention kernel, so decode used the
  single-threadgroup kernel. G = 5 is now instantiated for head size 128, which also covers Qwen2.5
  14B. (G = 8, for the 32B models, was left out: its split kernel was not bit-identical.)
- **Fixes Gemma tokenization of carriage returns and long space runs.** Whole-book tokenizations now
  match the reference (transformers) tokenizer on Gemma 1.1–4, Qwen3, Qwen2.5, Llama 3.2, Mistral
  and Phi-3.5.
  - **Carriage returns.** Foundation's JSON parser drops a U+FEFF that starts a string, so Gemma's
    vocabulary entry for U+FEFF + CR replaced the one for CR. Every carriage return was encoded as
    the wrong token, and Gemma 1.1 read CRLF text as near-nonsense.
  - **Space runs.** BPE now merges in Hugging Face's order: one pair at a time, the lowest rank first.
    Gemma's merge list ranks whitespace runs early, so a run of 32 or more spaces was split
    differently.
- **Gemma 4 raw completions start with BOS** (`mlxr run --raw`, `/v1/completions`). Its tokenizer
  files never say to add it, and without it the model produces nonsense. A chat template that writes
  `bos_token` now implies BOS on raw prompts, unless `add_bos_token` says otherwise. Chat was
  unaffected.
- **`mlxr bench` with `MLXR_BENCH_CTX`** reports, besides speed:
  - KL divergence, text NLL and margin-bucketed top-1 agreement between sparse and dense decode,
    over `MLXR_BENCH_POSITIONS` positions of `MLXR_BENCH_TEXT`, from each of
    `MLXR_BENCH_TEXT_OFFSETS`;
  - a dense control;
  - five passkey needles;
  - the cost of a verification pass.

  `MLXR_KERNEL_TIMING=1 swift test --filter SparseTimingTests` times the sparse kernels.
- **Tests.**
  - Each sparse kernel against a CPU reference.
  - Full-budget sparse decode bit-identical to dense on every architecture.
  - The 14B cases (Qwen3 14B, Gemma 4 12B) run with `MLXR_TEST_LARGE=1`, because next to the Gemma
    suite they do not fit a 16 GB Mac.

## 1.3.0

Faster prefill, a tokenizer fix for Gemma, speculative decoding with a draft model, and cheaper 2-token verification.

- **Prefill is about 1.45× faster.** The quantized prefill GEMM is one template over the per-simdgroup accumulator grid: the 32×32 tile stays for short batches, and a 64×64 tile (4×4 accumulators, two multiply-accumulates per fragment load instead of one, fully unrolled) runs long ones, reaching about 3.5 TFLOPS instead of 2.4. Both keep every output's arithmetic chain. Padded threadgroup rows and register prefetch measured slower on an M2 Pro and are not used. `flash_prefill` stages K/V as half (exact: the cache is half) with 16-byte loads of the next block during the current one's multiplies, rescales element-wise instead of through a diagonal matrix multiply, and schedules the longest query blocks first. Qwen3 0.6B prefills 512 tokens at 3,020 tok/s (was 2,040) and Mistral 7B at 247 (was 166). Decode is unchanged.
- **Fixes wrong and unstable tokens on tokenizers with canonically equivalent entries.** Swift compares strings by canonical equivalence, but `tokenizer.json` keys are exact. Gemma lists both `;` and the Greek question mark U+037E, so they collided: which id a `;` got changed from run to run, and the losing id decoded to nothing. Vocabulary, merge, added-token and word-cache lookups now compare exact text.
- **Draft-model speculative decoding** (`mlxr run --draft <model|auto>`, `mlxr serve --draft`, a "Draft model" picker in the app's inspector, `draft_tokens` / `accepted_draft_tokens` in the server's `usage`). A smaller model of the same family drafts tokens with the GPU sampler, queued back to back, and the target verifies them in one batched pass with the existing exact acceptance rule. Greedy output stays identical and sampled output keeps the target's distribution. The draft has its own KV cache, rolls back by position, and catches up on the prompt after the first token, so it does not delay the first token. Loading refuses a draft unless the tokenizers match, and the memory check sums both models. `auto` pairs the smallest installed model of the same type and tokenizer at most a quarter of the size. The app drops a saved pick that does not fit the model being loaded. **It is off by default and currently slower:** Qwen3 8B with a Qwen3 0.6B draft goes from 38.1 to 27.9 tok/s on a short prompt and from 35.6 to 21.3 at a 1,890-token context, because verifying 3–8 tokens costs about three decode steps and about 45% of drafts are accepted on the tested prompts.
- **Verifying 2 tokens costs half of what it did** (Qwen3 8B: 78.6 → 40.7 ms; Qwen3 0.6B: 10.9 → 5.4 ms), through a 2-token matvec that reads each weight once. Batches of 3–8 tokens use a new simdgroup-matrix kernel (`qmvx4`) that dequantizes each weight once. On an M2 Pro it is ALU-bound at about 3× a single-token step (replacing the weight loads with constants costs the same), so 3–4 tokens are no faster than before, but the batched path now reaches 8. Batches of 2–8 tokens use the per-token decode attention instead of the prefill attention.
- Keeps a chat message's Copy, Edit and Regenerate buttons visible while the pointer is on them, for a report of them hiding on hover and becoming unclickable.
- `mlxr bench`: profile sweeps warm up first and take the best of three (the first sweep after loading read 2–3× slow), verification is reported at a short and a 1,900-token context, and `MLXR_BENCH_NOPROFILE=1` skips the profile.
- Tests: GEMM tiles against the CPU reference up to 517 tokens, tiles agreeing bitwise across batch sizes, prefill vs incremental decode for a 600-token prompt, verification batches of 2–8 tokens vs one-at-a-time decode, exact speculative acceptance on synthetic logits, self-draft greedy identity, and tokenizer entries that are canonically equivalent.

Tried in the same pass and left out: folding `qkv_post` into the decode attention, SiLU into the down projection and the residual add + RMSNorm into the next matvec (same output, but the step got slower on Qwen3 0.6B: 3.5 → 6.6 ms).

## 1.2.2

- Fixes the chat window freezing during a long reply (any model) and then showing the rest at once, and the transcript going blank when a reply finished. Sampled on an M2 Pro, the main thread was 100% busy in SwiftUI/AppKit layout while the engine (which streams at 6–7 ms per token) had already finished the reply into the stream buffer. Five causes, all in the chat view:
  - Every paragraph of the streaming reply was a selectable, AppKit-backed text view that SwiftUI re-created and hit-tested (hover tracking) on each update. A streaming reply is plain text and becomes selectable once it is final; rows have no hover tracking while a reply streams.
  - The live-stats dot, the "Thinking…" shimmer and the typing dots were repeat-forever animations, rendering 60 frames a second, each frame with a full AppKit layout and hit-test pass. They are stepped now.
  - Every token mutated the conversation store, so the sidebar, the inspector (with its AppKit sliders) and the whole chat re-rendered per update. The reply streams into live state that only the in-progress row reads; the conversation is written once at the end.
  - Finished paragraphs keep their view; only the paragraph being written re-renders.
  - Updates are paced by their measured cost on the main thread (at most 20 a second, never more often than twice what the previous one took), so the UI keeps most of its time whatever the transcript size.
  - The message list is a plain stack: a lazy one could be left scrolled past its stale content when the reply's layout changed at the end, which showed an empty transcript.
## 1.2.1

- Fixes the GPU out-of-memory error when running a large model such as Gemma 4 12B on a 16 GB Mac. MLX writes `safetensors` without the 16-byte alignment the kernels need, so every tensor was copied into its own Metal buffer while the mapped file stayed referenced: the GPU held two copies of the weights (18 GB for a 9 GB model). Shards are now copied once, in file order, into a page-aligned buffer and the mapping is released. Loading is faster too: Gemma 4 12B 156 s → 25 s from disk, 11 s from the page cache; Mistral 7B 3.3 → 2.9 s.
- Sliding-window layers (Gemma 2, 3 and 4) keep a ring of window + 512 K/V rows instead of the whole context: Gemma 4 12B at 8K context needs 0.5 GB of cache instead of 2.7 GB, and runs at 32K on 16 GB. A request whose shared prefix with the previous one ends more than 512 tokens before that request's end no longer reuses the cache (its rows are gone); it is processed from the start.
- A GPU out-of-memory failure is reported as such, with the advice to lower the context length, instead of as a kernel compilation error; the load-time memory check includes the scratch buffers.

## 1.2.0

Gemma support (`gemma`, `gemma2`, `gemma3` / `gemma3_text`, `gemma4` / `gemma4_text`, `gemma4_unified`):

- Gemma 1–3: `(1 + w)` RMSNorm weights, scaled embeddings, GeGLU (exact or tanh GELU), sandwich norms fused with the residual add, attention and final-logit soft-capping (Gemma 2), sliding-window layers with their own RoPE base (Gemma 3). Multimodal `gemma3` exports run their language model.
- Gemma 4 (dense models): 256/512 head sizes in one model, proportional RoPE, KV cache shared with an earlier layer, K = V global attention (12B), per-layer input embeddings (E2B/E4B), `layer_scalar`, mixed 4/8-bit (OptiQ) checkpoints. Mixture-of-experts Gemma 4 (26B-A4B) is not supported. Thinking mode is not wired up; models run with it off.
- Chat formats: `<start_of_turn>` (Gemma 1–3) and `<|turn>` (Gemma 4), checked against transformers' `apply_chat_template`. No tool calling for Gemma.
- Tokenizer: Gemma's SentencePiece tokenizer adds no dummy prefix, and runs of spaces (added tokens) decode correctly.
- Verified against mlx-lm, teacher-forced over 40 positions after a 664-token prompt: Gemma 1.1 2B, Gemma 2 2B, Gemma 3 270M/1B/4B, Gemma 4 E2B and 12B produce the same argmax at every position (one near-tie aside), logits within 0.035 of a float32 mlx-lm run.
- Attention kernels take a sliding window and soft-cap; new kernel tests compare decode, split decode, prefill and flash prefill against a CPU reference.

Gemma performance (M2 Pro, 3 alternating pairs against the first Gemma build; mlx-lm 0.32.2 on the same files, 200 greedy tokens):

- The 4-bit decode matvec gets a kernel per remainder of `hidden_size` mod 1024 (Gemma's 1152 / 2304 / 2560 / 1536 left a pass with 4–16 of 32 lanes busy): decode +12% on Gemma 3 1B, +5% on Gemma 2 2B, +4% on Gemma 3 4B. The tail lives in its own instantiations: inside the one kernel it cost every model 2–11%.
- Flash prefill for head_dim 256 (8-key blocks fit threadgroup memory): prefill +7% on Gemma 3 1B, +6% on Gemma 4 E2B (attention 66 → 43 ms and 177 → 121 ms per 512 tokens).
- Split decode attention for head sizes 256 and 512 (query heads in sub-groups of 2, each K/V row read once per pair): decode at 1K context +3.5% on Gemma 3 1B, +3.7% on Gemma 3 4B, +6.1% on Gemma 4 E2B.
- Gemma 4's per-layer input gate folds its GELU into the gate matvec's epilogue (one dispatch fewer per layer).
- Against mlx-lm (mlx-lm → MLX Runner), decode after a short prompt: Gemma 3 270M 396 → 481 tok/s, Gemma 3 1B 187 → 193, Gemma 2 2B 97 → 94, Gemma 3 4B 71 → 67, Gemma 4 E2B 78 → 73; after a 664-token prompt: 1B 177 → 179, 2B 93 → 87, 4B 67 → 64, E2B 75 → 70. Prefill (raw forward pass of 664 tokens vs `mlxr bench` at 512): 1B 1949 vs 1453, 2B 768 vs 538, 4B 378 vs 345, E2B 627 vs 545 tok/s; the remaining prefill gap is the engine's fp32 GEMM, not Gemma-specific.
- Rejected after measuring (kept out): a gate/up matvec with the activation in its epilogue (−4% to −11% on every model), split-K for few-row matrices (−3% to −11% on Gemma), 8-byte lane loads for a 512 remainder.
- Existing models: greedy output is token-identical and decode/prefill throughput is unchanged (3 alternating pairs on Qwen3 0.6B, Llama 3.2 1B, Mistral 7B, all within noise).
- `mlxr bench` adds a per-kernel-class prefill breakdown; the profile sweep no longer crashes on 8-bit layers.

Review fixes (from an adversarial review of the Gemma branch): Gemma replies keep their leading spaces and indentation (the dummy-prefix rule now applies only to tokenizers that have one); raw completions add `<bos>` when the tokenizer's post-processor does (Gemma 2, Llama 3); Gemma 3 encoding of long HTML/code chats is 100× faster (added tokens are matched with a trie instead of a 6,000-way regex); non-special added tokens such as Gemma's HTML tags are kept inside message content, as the reference tokenizer does; Gemma 4's per-layer-input norm uses the reference's exact eps; Gemma 4 mixture-of-experts repos are marked not runnable in Discover and My Models; Gemma 4 OptiQ exports no longer map or budget their 1 GB vision sidecar; flat `gemma3` / transformers 5 `rope_parameters` configs, the outer config's stop tokens, Gemma 2's default soft-caps and an out-of-range `sliding_window` are handled; later system messages render as `<|turn>system`; `layer_scalar` is read only for Gemma 4; model tests cover every installed Gemma checkpoint (Gemma 1.1 2B, 3 4B and 4 12B included) instead of passing silently when one is missing.

## 1.1.1

Performance (M2 Pro, measured against 1.1.0 by alternating the two builds, 4 pairs; greedy output is token-identical on Qwen3 0.6B, Qwen2.5 0.5B, Llama 3.2 1B, Phi-3.5 and Mistral 7B):

- The next token is selected on the GPU, inside the same command buffer as the forward pass: repetition penalty, exact top-k, min-p, top-p and a seeded draw. The following decode step is committed before the current one is read back, so the GPU runs step after step without waiting for the CPU to sample, detokenize or launch. Decode at 16 tokens of context: Qwen3 0.6B +10.7%, Llama 3.2 1B +8.5%, Mistral 7B +4.1%; at 1K context +9.0%, +7.8%, +4.0%. Generation as the app and `mlxr run` see it, which previously also paid for CPU sampling: Qwen3 0.6B 238 → 282 tok/s, Llama 3.2 1B 180 → 201 tok/s, Mistral 7B 39.3 → 40.9 tok/s.
- RMSNorm kernels use vector loads with up to 1024 threads per row; the three norms per layer cost 0.21 ms instead of 0.62 ms on Mistral 7B.
- The GPU sampler draws from the same distribution as the CPU sampler (tests compare them), but with its own xorshift stream: a given seed produces a different sequence than 1.1.0, and top-k values above 1024 are capped at 1024. Speculative decoding and constrained tool-call names still sample on the CPU.
- `mlxr bench` reports the pipelined decode, the app's default sampler and the token-selection cost; `profile` adds a per-kernel-class breakdown of one decode step.
- Prefill is unchanged.

## 1.1.0

Tool calling in the local server (OpenAI-compatible):

- Function tools in Chat Completions (`tools`, `tool_choice`, `parallel_tool_calls`, assistant `tool_calls`, `tool` messages) and the Responses API (function tools, `function_call` / `function_call_output` items, `function_call` output items with their streaming events), streaming and non-streaming.
- Prompts use each model's native tool format: Hermes (Qwen2.5, Qwen3), Llama 3.1+ and Mistral v3. They match the official chat templates and `mistral-common` byte for byte, which tests enforce.
- `tool_choice`: `auto`, `none`, `required` (the function name is constrained to the defined tools during decoding) and a named function.
- Requests are validated: tool count and names, JSON object arguments, tool results that must answer an earlier call, and a nesting limit. Models without a tool format (Phi-3.5) answer tool requests with a 400.
- Tested end to end with the official `openai` Python SDK on Qwen3 0.6B/8B, Qwen2.5 0.5B, Llama 3.2 1B and Mistral 7B v0.3.

## 1.0.3

Performance (M2 Pro, measured against 1.0.2 by alternating the two builds; greedy output is token-identical):

- Decode overlaps CPU work with the GPU: the next token's command buffer is encoded while the current one runs.
- Decode attention at 256+ tokens of context reads each K/V row once per grouped-query head group instead of once per query head. Decode at 1K context: Qwen3 0.6B +6.7%, Llama 3.2 1B +3.8%, Mistral 7B +2.2%. At 16 tokens of context: +2.8%, +0.8%, +0.5%.
- Short prompts (5–16 tokens, such as a follow-up chat message) use an 8- or 16-token GEMM tile instead of padding to 32: time to process them is 27–50% lower (Mistral 7B: 198 → 99 ms for 5 tokens).
- Prefill of long prompts is unchanged.

## 1.0.2

- Discover: the model list takes 40% of the area (at least 380 pt, detail at least 400) instead of whatever width the split view started with, which left it squeezed in large windows. The divider is still draggable and its position is remembered.
- Discover: "Runnable only" and the model count stay on one line in narrow windows.

## 1.0.1

- The window opens at 1280×820 on first launch instead of at its minimum width, which squeezed the Discover screen and the chat.
- The sidebar keeps at least 240 pt and the main area at least 820 pt (chat 560 + settings inspector 260). The window can't shrink below the sum, so neither side gets compressed.
- Discover's model badges move to a second row when the pane is narrow instead of breaking mid-word.

## 1.0.0

First public release.

- Native macOS app (SwiftUI, Apple silicon) to discover, download and chat with MLX models from Hugging Face.
- From-scratch Metal inference engine for `llama`, `mistral`, `qwen2`, `qwen3` and `phi3` models with 2–8-bit MLX quantization: fused 4-bit decode kernels, simdgroup-matrix prefill, flash attention, KV-cache prefix reuse. Mistral 7B decodes at 38.6 tok/s on an M2 Pro (82% of its memory bandwidth).
- Qwen3 thinking mode, Markdown chat with code highlighting, edit/regenerate, searchable history.
- OpenAI-compatible local server: Responses API, Chat Completions, Completions; Host/Origin checks, optional API key (required in LAN mode), limits and timeouts.
- `mlxr` CLI: search, download, run, serve, bench.
- Downloads resume with HTTP ranges and are SHA-256 verified; the Hugging Face token lives in the Keychain.
- Signed in-app updates (Ed25519-signed SHA256SUMS, SHA-256, code-signature check, atomic swap).
