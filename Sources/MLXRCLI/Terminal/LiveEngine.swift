import Foundation
import MLXEngine

/// The real engine: `InferenceEngine.shared`.
public struct LiveGenerationEngine: GenerationEngine {
    public init() {}
    private var engine: InferenceEngine { InferenceEngine.shared }

    public func load(directory: URL, draftDirectory: URL?, contextLength: Int, sparse: SparseAttention?) async throws -> LoadedModelInfo {
        try await engine.load(directory: directory, draftDirectory: draftDirectory, contextLength: contextLength, sparse: sparse)
    }

    public func chat(_ messages: [ChatMessage], parameters: SamplingParameters, enableThinking: Bool?,
                     token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        engine.chat(messages, parameters: parameters, enableThinking: enableThinking, token: token)
    }

    public func complete(_ prompt: String, parameters: SamplingParameters, token: CancelToken) -> AsyncThrowingStream<GenerationEvent, Error> {
        engine.complete(prompt, parameters: parameters, token: token)
    }

    public func renderChat(_ messages: [ChatMessage], enableThinking: Bool?) -> String {
        engine.renderChat(messages, enableThinking: enableThinking)
    }

    public func encode(_ text: String) -> [Int] { engine.encode(text) }
    public func autoDraft(for target: URL) -> LocalModel? { InferenceEngine.autoDraft(for: target) }
}
