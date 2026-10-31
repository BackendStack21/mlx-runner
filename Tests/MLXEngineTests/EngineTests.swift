import Foundation
import Testing
@testable import MLXEngine

@Suite("Text handling")
struct TextTests {
    @Test func streamingDecoderHoldsBackPartialUTF8() {
        var d = StreamingDecoder()
        let rocket = Array("🚀".utf8)            // 4 bytes
        #expect(d.append(Array("Hi ".utf8)) == "Hi ")
        #expect(d.append(Array(rocket[0..<2])) == "")
        #expect(d.append(Array(rocket[2..<3])) == "")
        #expect(d.append(Array(rocket[3..<4]) + Array("!".utf8)) == "🚀!")
        #expect(d.flush() == "")
    }

    @Test func chatMLTemplateWithThinkingToggle() {
        let t = ChatTemplate(template: "<|im_start|>{{ enable_thinking }}", modelType: "qwen3", bosToken: nil)
        #expect(t.format == .chatML)
        #expect(t.supportsThinkingToggle)
        let s = t.render([ChatMessage(role: .system, content: "Be brief."), ChatMessage(role: .user, content: "Hi")],
                         enableThinking: false)
        #expect(s == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test func llama3Template() {
        let t = ChatTemplate(template: "<|start_header_id|>", modelType: "llama", bosToken: "<|begin_of_text|>")
        let s = t.render([ChatMessage(role: .user, content: "Hi")])
        #expect(s == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    }

    @Test func mistralTemplateFoldsSystemIntoLastUserTurn() {
        let t = ChatTemplate(template: "[INST]", modelType: "mistral", bosToken: "<s>")
        let s = t.render([ChatMessage(role: .system, content: "S"), ChatMessage(role: .user, content: "A"),
                          ChatMessage(role: .assistant, content: "B"), ChatMessage(role: .user, content: "C")])
        #expect(s == "<s>[INST] A [/INST] B</s>[INST] S\n\nC [/INST]")
    }
}

@Suite("Sampling")
struct SamplerTests {
    func logits(_ v: [Float]) -> UnsafeMutablePointer<Float> {
        let p = UnsafeMutablePointer<Float>.allocate(capacity: v.count)
        p.initialize(from: v, count: v.count)
        return p
    }

    @Test func greedyPicksArgmax() {
        var params = SamplingParameters()
        params.temperature = 0
        params.repetitionPenalty = 1
        var s = Sampler(params: params, rng: XorShift(seed: 1))
        let l = logits([0.1, 3, -2, 2.9])
        #expect(s.sample(l, count: 4, recent: []) == 1)
    }

    @Test func repetitionPenaltyDemotesRecentToken() {
        var params = SamplingParameters()
        params.temperature = 0
        params.repetitionPenalty = 1.5
        var s = Sampler(params: params, rng: XorShift(seed: 1))
        let l = logits([0.1, 3, -2, 2.9])
        #expect(s.sample(l, count: 4, recent: [1]) == 3)
    }

    @Test func topKOneIsGreedyAndSeedIsDeterministic() {
        var params = SamplingParameters()
        params.temperature = 1
        params.topK = 1
        var s = Sampler(params: params, rng: XorShift(seed: 3))
        #expect(s.sample(logits([1, 5, 2]), count: 3, recent: []) == 1)

        params.topK = 0
        params.topP = 1
        var a = Sampler(params: params, rng: XorShift(seed: 42))
        var b = Sampler(params: params, rng: XorShift(seed: 42))
        let v: [Float] = (0..<1000).map { Float($0 % 17) / 4 }
        let ra = (0..<20).map { _ in a.sample(logits(v), count: v.count, recent: []) }
        let rb = (0..<20).map { _ in b.sample(logits(v), count: v.count, recent: []) }
        #expect(ra == rb)
    }

    @Test func topPKeepsOnlyTheNucleus() {
        var params = SamplingParameters()
        params.temperature = 1
        params.topK = 0
        params.topP = 0.5
        params.repetitionPenalty = 1
        var s = Sampler(params: params, rng: XorShift(seed: 9))
        // token 0 carries > 50% of the mass, so it is the only candidate.
        for _ in 0..<50 { #expect(s.sample(logits([5, 1, 1, 1]), count: 4, recent: []) == 0) }
    }
}

@Suite("Safetensors")
struct SafetensorsTests {
    @Test func loadsTensorsZeroCopyAndRealignsMisalignedOnes() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("st-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        // a: 3 x f16 (6 bytes) so b starts misaligned; b: 4 x f32
        let a: [Float16] = [1, 2, 3]
        let b: [Float] = [0.5, -1, 2.25, 8]
        var header = #"{"a":{"dtype":"F16","shape":[3],"data_offsets":[0,6]},"b":{"dtype":"F32","shape":[4],"data_offsets":[6,22]}}"#
        while (header.utf8.count + 8) % 8 != 0 { header += " " }
        var data = Data()
        var len = UInt64(header.utf8.count)
        data.append(Data(bytes: &len, count: 8))
        data.append(header.data(using: .utf8)!)
        a.withUnsafeBytes { data.append(contentsOf: $0) }
        b.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: dir.appendingPathComponent("model.safetensors"))

        let store = try WeightStore(directory: dir, device: GPU.shared.device)
        let tb = try store.require("b")
        #expect(tb.shape == [4] && tb.dtype == .f32)
        #expect(tb.offset % 16 == 0)
        let got = Array(UnsafeBufferPointer(start: (tb.buffer.contents() + tb.offset).bindMemory(to: Float.self, capacity: 4), count: 4))
        #expect(got == b)
        #expect(store.get("missing") == nil)
    }
}
