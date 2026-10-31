import Foundation
import Testing
@testable import MLXEngine

/// End-to-end checks against real downloaded models. Each test is skipped when its
/// model is not installed (`mlxr download <repo>`).
func installedModel(_ repo: String) -> URL? {
    ModelStore.scan().first { $0.repo == repo }?.directory
}

@Suite("Real models", .serialized)
struct ModelTests {
    static let qwen3 = "mlx-community/Qwen3-0.6B-4bit"
    static let llama = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    @Test(.enabled(if: installedModel(qwen3) != nil))
    func qwenTokenizerKnownIdsAndRoundTrip() throws {
        let tok = try Tokenizer(directory: installedModel(Self.qwen3)!)
        #expect(tok.encode("Hello world") == [9707, 1879])
        #expect(tok.encode("<|im_start|>user") == [151644, 872])
        for s in ["Hello, world!", "Ünïcödé 你好 🚀", "  leading spaces\n\ttabs", "it's 2024—ok?", "x=1;y=22;z=333"] {
            #expect(tok.decode(tok.encode(s)) == s)
        }
    }

    @Test(.enabled(if: installedModel(llama) != nil))
    func llamaTokenizerKnownIds() throws {
        let tok = try Tokenizer(directory: installedModel(Self.llama)!)
        #expect(tok.encode("Hello world") == [9906, 1917])
        #expect(tok.decode(tok.encode("Ünïcödé 你好 🚀")) == "Ünïcödé 你好 🚀")
    }

    /// Prefill (GEMM + flash attention) and token-by-token decode (matvec + decode attention)
    /// must produce the same next-token distribution.
    @Test(.enabled(if: installedModel(qwen3) != nil), arguments: [qwen3, llama])
    func prefillMatchesIncrementalDecode(repo: String) throws {
        guard let dir = installedModel(repo) else { return }
        let model = try Transformer(directory: dir, contextLength: 1024)
        let tok = try Tokenizer(directory: dir)
        let ids = tok.encode("The quick brown fox jumps over the lazy dog. Then it ran into the forest, where").map { Int32($0) }
        #expect(ids.count > 16)
        let V = model.vocabSize
        let pre = Array(UnsafeBufferPointer(start: try model.forward(ids, offset: 0), count: V))
        var inc: [Float] = []
        for (i, t) in ids.enumerated() {
            let l = try model.forward([t], offset: i)
            if i == ids.count - 1 { inc = Array(UnsafeBufferPointer(start: l, count: V)) }
        }
        let argPre = pre.indices.max { pre[$0] < pre[$1] }!
        let argInc = inc.indices.max { inc[$0] < inc[$1] }!
        #expect(argPre == argInc)
        let maxDiff = zip(pre, inc).map { abs($0 - $1) }.max()!
        #expect(maxDiff < 0.1, "max logit difference \(maxDiff)")
    }

    @Test(.enabled(if: installedModel(qwen3) != nil))
    func chatAnswersAndReusesPromptCache() async throws {
        let engine = InferenceEngine()
        _ = try await engine.load(directory: installedModel(Self.qwen3)!, contextLength: 2048)
        var p = SamplingParameters()
        p.temperature = 0
        p.maxTokens = 24
        var msgs = [ChatMessage(role: .user, content: "What is the capital of France? One word.")]
        var text = ""
        for try await ev in engine.chat(msgs, parameters: p, enableThinking: false) {
            if case .text(let t) = ev { text += t }
        }
        #expect(text.contains("Paris"))
        msgs.append(ChatMessage(role: .assistant, content: text))
        msgs.append(ChatMessage(role: .user, content: "And of Italy?"))
        var cached = 0
        for try await ev in engine.chat(msgs, parameters: p, enableThinking: false) {
            if case .promptProcessed(let s) = ev { cached = s.cachedPromptTokens }
        }
        #expect(cached > 10, "second turn should reuse the KV cache prefix")
    }

    @Test(.enabled(if: installedModel(qwen3) != nil))
    func longConversationIsTrimmedToFitContext() async throws {
        let engine = InferenceEngine()
        _ = try await engine.load(directory: installedModel(Self.qwen3)!, contextLength: 256)
        var p = SamplingParameters()
        p.temperature = 0
        p.maxTokens = 16
        let filler = String(repeating: "This is an old message about nothing in particular. ", count: 20)
        var msgs = [ChatMessage(role: .system, content: "You are terse.")]
        for _ in 0..<6 {
            msgs.append(ChatMessage(role: .user, content: filler))
            msgs.append(ChatMessage(role: .assistant, content: "Ok."))
        }
        msgs.append(ChatMessage(role: .user, content: "Say hi."))
        var finished = false
        for try await ev in engine.chat(msgs, parameters: p, enableThinking: false) {
            if case .finished = ev { finished = true }
        }
        #expect(finished)
    }
}
