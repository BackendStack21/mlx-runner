# Changelog

## 1.0.0

First public release.

- Native macOS app (SwiftUI, Apple silicon) to discover, download and chat with MLX models from Hugging Face.
- From-scratch Metal inference engine for `llama`, `mistral`, `qwen2`, `qwen3` and `phi3` models with 2–8-bit MLX quantization: fused 4-bit decode kernels, simdgroup-matrix prefill, flash attention, KV-cache prefix reuse. Mistral 7B decodes at 38.6 tok/s on an M2 Pro (82% of its memory bandwidth).
- Qwen3 thinking mode, Markdown chat with code highlighting, edit/regenerate, searchable history.
- OpenAI-compatible local server: Responses API, Chat Completions, Completions; Host/Origin checks, optional API key (required in LAN mode), limits and timeouts.
- `mlxr` CLI: search, download, run, serve, bench.
- Downloads resume with HTTP ranges and are SHA-256 verified; the Hugging Face token lives in the Keychain.
- Signed in-app updates (Ed25519-signed SHA256SUMS, SHA-256, code-signature check, atomic swap).
