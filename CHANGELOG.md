# Changelog

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
