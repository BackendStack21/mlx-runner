# Security policy

MLX Runner runs untrusted model files from the internet and can expose a local HTTP server, so
security issues are taken seriously.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through GitHub's
[private vulnerability reporting](https://github.com/BackendStack21/mlx-runner/security/advisories/new).
Include the version, macOS version, and the steps to reproduce. Expect an acknowledgement within a few days.

## Privacy

MLX Runner collects nothing. Models run locally; prompts and replies never leave the machine. Outbound
requests go to Hugging Face (search, model cards, downloads you start) and to GitHub's releases API
for the update check, which the "Check for updates automatically" setting turns off. Any other network
access by MLX Runner itself would be a bug: please report it as a vulnerability.

## Model files are untrusted input

`config.json`, `tokenizer.json` and `.safetensors` files are validated before anything reaches the GPU:
header sizes and tensor offsets are bounds- and overflow-checked, every dimension is range-checked,
token ids must fit the embedding table and every weight's shape must match the config. Chat message
text is tokenized as plain text, so a message cannot inject control tokens. Model-card links open only
for https (confirmed for hosts other than huggingface.co) and mailto.

## Downloads and tokens

Downloads come from huggingface.co over https; LFS files are checked against the SHA-256 the Hub
publishes. The Hugging Face token is stored in the Keychain and sent only to huggingface.co / hf.co
hosts over https, never to the CDN a download redirects to.

## Local server

The OpenAI-compatible server binds to `127.0.0.1` by default. It rejects requests whose `Host` isn't a
loopback name (DNS rebinding) and browser requests from origins outside its allowlist, accepts JSON
bodies only, and enforces header/body/write deadlines, connection and queue caps and size limits.
An API key is optional on localhost and required when serving on the local network.

## Updates

The in-app updater reads the latest release from GitHub's API and downloads only from GitHub over https,
with size caps. It installs a release only if `SHA256SUMS.sig` is a valid Ed25519 signature of
`SHA256SUMS` by the key compiled into the app (`packaging/update-key.pub`; its private half exists only
as a release-workflow secret), the downloaded file's SHA-256 matches the signed list, the version is
strictly newer, and the staged app passes `codesign --verify --deep --strict` with the expected bundle
identifier and version. The swap is atomic with rollback; nothing downloaded runs before every check
passes. Each release also carries Sigstore build provenance (`gh attestation verify`).
