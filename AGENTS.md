# AGENTS.md

How to work on MLX Runner (a native macOS app by 21no.de).

## Rules

- **Swift only, zero third-party dependencies.** Apple system frameworks are fine (SwiftUI, AppKit,
  Foundation, Metal, Network, CryptoKit, Security). No SwiftPM packages, no vendored code, no Python —
  scripts are Swift (`swift scripts/<name>.swift`) or POSIX shell.
- **Performance changes need numbers.** Measure with `mlxr bench <model>` (`MLXR_BENCH_DECODE=1` for
  decode only) before and after; keep a change only if it wins. Nothing else may use the GPU while
  benchmarking.
- **Correctness gates:** `swift test` passes; greedy output stays token-identical; prefill matches
  incremental decode for every architecture (`prefillMatchesIncrementalDecode`). Model tests run when the
  model is installed (`mlxr download <repo>`), otherwise they skip.
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
