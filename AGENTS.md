# AGENTS.md

How to work on MLX Runner (a native macOS app by 21no.de).

## Rules

- **Swift only, zero third-party dependencies.** Apple system frameworks are fine (SwiftUI, AppKit,
  Foundation, Metal, Network, CryptoKit, Security). No SwiftPM packages, no vendored code, no Python —
  scripts are Swift (`swift scripts/<name>.swift`) or POSIX shell.
- **Performance changes need numbers.** Measure with `mlxr bench <model>` (`MLXR_BENCH_DECODE=1` for
  decode only; for runs use `mlxr run <model> "<prompt>" --temp 0 --seed 42 --max 256`) before and after; keep a change only if it wins. Nothing else may use the GPU while
  benchmarking.
- **Correctness gates:** `swift test` passes; greedy output (`--temp 0 --seed 42 --max 256`) stays token-identical; prefill matches
  incremental decode for every architecture (`prefillMatchesIncrementalDecode`). Model tests run when the
  model is installed (`mlxr download <repo>`), otherwise they skip. The 14B cases need
  `MLXR_TEST_LARGE=1` (with the Gemma suite they do not fit 16 GB).
- **One deliberate exception: sparse decode attention**, on by default from 5,128 tokens of context,
  changes output past that point. Every other path stays token-identical, and so does sparse decode
  with a budget covering the whole context (`sparseDecodeWithFullBudgetIsBitIdentical`). Check a change
  to it with the long-context bench (`MLXR_BENCH_CTX` with `MLXR_BENCH_TEXT`: KL, text NLL, agreement);
  `MLXR_SPARSE_ATTN=0` turns it off. See docs/sparse-decode-attention.md.
- **Untrusted input:** anything read from a model directory, the network or the local server is
  validated before use. MLXEngine builds with `-Ounchecked`, so overflow and bounds traps are not a
  safety net: check sizes explicitly.

## Layout

- `Sources/MLXEngine/` — engine (kernels, model, tokenizer, templates, sampler), Hub client, local
  server, updater. `Version.swift` holds the release version; `UpdateKey.swift` is generated.
- `Sources/MLXRunner/` — SwiftUI app. `MLXRUNNER_SNAPSHOT=<dir>` renders screens headlessly.
- `Sources/mlxr/` — CLI.
- `site/` — landing page (mlx-runner.21no.de, Cloudflare Workers static assets via `wrangler.jsonc`).
- `scripts/` — `bundle.sh` (app/DMG/CLI/SHA256SUMS), `sign.swift`, `release-key.swift`, `make-icon.swift`.

## Releasing

See the "Releasing" section of README.md: bump `Version.swift`, add a CHANGELOG section, tag `vX.Y.Z`.
The release workflow signs `SHA256SUMS`; never publish a release by hand, or installed apps will refuse it.
