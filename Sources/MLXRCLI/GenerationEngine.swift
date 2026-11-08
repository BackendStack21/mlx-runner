import Foundation
import MLXEngine

/// The generation engine as the commands use it, injected so tests never touch a GPU or a model.
/// Cancellation does not throw: a cancelled stream ends with `.finished(stats)` and
/// `stats.stopReason == "cancelled"`.
public protocol GenerationEngine: Sendable {
    func load(directory: URL, draftDirectory: URL?, contextLength: Int, sparse: SparseAttention?) async throws -> LoadedModelInfo
    func chat(_ messages: [ChatMessage], parameters: SamplingParameters, enableThinking: Bool?,
              token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error>
    func complete(_ prompt: String, parameters: SamplingParameters,
                  token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error>
    func renderChat(_ messages: [ChatMessage], enableThinking: Bool?) -> String
    func encode(_ text: String) -> [Int]
    /// The installed model that pairs with `target` as a draft model, if any (`--draft auto`).
    func autoDraft(for target: URL) -> LocalModel?
}
