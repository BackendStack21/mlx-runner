import Foundation
import MLXEngine

// Error-reporting seams for the legacy command bodies: they throw instead of calling exit(1),
// so every failure reaches CLI.run and prints `Error: <message>`.

/// A model by path, installed repo id or short name; throws when nothing matches.
func resolveModel(_ s: String) throws -> URL {
    if FileManager.default.fileExists(atPath: s) { return URL(fileURLWithPath: s) }
    if let m = ModelStore.scan().first(where: { $0.repo == s || $0.name == s }) { return m.directory }
    throw CLIError.failure("model not found: \(s) (try `mlxr download \(s)`)")
}

/// `--sparse <on|off|N>`: sparse decode attention for long contexts (N: long-range keys; the window,
/// minimum context and shared selection still come from MLXR_SPARSE_*). Without the
/// option: MLXR_SPARSE_ATTN, which defaults to on.
func sparseSetting(_ s: String?) throws -> SparseAttention? {
    guard let s else { return SparseAttention.fromEnvironment }
    switch s.lowercased() {
    case "on", "1": return .configured(topK: SparseAttention.standard.topK)
    case "off", "0": return nil
    default:
        guard let n = Int(s), n > 0 else { throw CLIError.failure("--sparse takes on, off or a number of long-range keys") }
        return .configured(topK: n)
    }
}

/// Repo ids must be `owner/name` and pass `RepoID.isValid`.
func validateRepoID(_ s: String) throws {
    guard RepoID.isValid(s) else { throw CLIError.failure("invalid repo id: \(s) (expected owner/name)") }
}

/// Runs a download. Hugging Face errors pass through (CLI.run maps 401/404 to the hint); others
/// become `download failed: <reason>`.
func performDownload(start: () async throws -> URL) async throws -> URL {
    do { return try await start() } catch {
        if (error as NSError).domain == "HuggingFace", [401, 404].contains((error as NSError).code) { throw error }
        if error is CLIError { throw error }
        throw CLIError.failure("download failed: \(error.localizedDescription)")
    }
}

/// Starts the server; a failure becomes `cannot start server: <reason>`.
func startServer(_ start: () throws -> Void) throws {
    do { try start() } catch { throw CLIError.failure("cannot start server: \(error.localizedDescription)") }
}
