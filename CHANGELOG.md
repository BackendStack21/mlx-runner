# Changelog

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
