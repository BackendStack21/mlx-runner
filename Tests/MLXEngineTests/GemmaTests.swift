import Foundation
import Testing
@testable import MLXEngine

/// Writes `json` as config.json in a fresh directory.
private func configDir(_ json: String) throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("cfg-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try json.data(using: .utf8)!.write(to: dir.appendingPathComponent("config.json"))
    return dir
}

@Suite("Gemma chat formats")
struct GemmaTemplateTests {
    let conversation = [
        ChatMessage(role: .system, content: "You are terse."), ChatMessage(role: .user, content: " Hello \n"),
        ChatMessage(role: .assistant, content: "Hi! "), ChatMessage(role: .user, content: "Bye"),
    ]

    // Expected strings: transformers' apply_chat_template on the models' own templates.
    @Test func gemma3FoldsSystemIntoFirstUserTurn() {
        let t = ChatTemplate(template: "{{ bos_token }}<start_of_turn>", modelType: "gemma3_text", bosToken: "<bos>")
        #expect(t.format == .gemma && !t.supportsTools && !t.supportsThinkingToggle)
        #expect(t.render(conversation) == "<bos><start_of_turn>user\nYou are terse.\n\nHello<end_of_turn>\n<start_of_turn>model\nHi!<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n")
        #expect(t.render([ChatMessage(role: .user, content: "Hi there")]) == "<bos><start_of_turn>user\nHi there<end_of_turn>\n<start_of_turn>model\n")
        #expect(t.stopTokens.contains("<end_of_turn>"))
    }

    @Test func gemma4KeepsLaterSystemTurns() {
        let t = ChatTemplate(template: "<|turn>", modelType: "gemma4", bosToken: "<bos>")
        let msgs = [ChatMessage(role: .user, content: "Hi"), ChatMessage(role: .assistant, content: "Hello"),
                    ChatMessage(role: .system, content: "Be brief."), ChatMessage(role: .user, content: "Why?")]
        #expect(t.render(msgs) == "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello<turn|>\n<|turn>system\nBe brief.<turn|>\n<|turn>user\nWhy?<turn|>\n<|turn>model\n")
    }

    @Test func gemma4HasASystemTurnAndNoThinkingToggle() {
        let t = ChatTemplate(template: "{{ bos_token }}<|turn>{% if enable_thinking %}<|think|>{% endif %}", modelType: "gemma4", bosToken: "<bos>")
        #expect(t.format == .gemma4 && !t.supportsTools && !t.supportsThinkingToggle)
        #expect(t.render(conversation) == "<bos><|turn>system\nYou are terse.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\nHi!<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n")
        #expect(t.render([ChatMessage(role: .user, content: "Hi there")]) == "<bos><|turn>user\nHi there<turn|>\n<|turn>model\n")
        #expect(t.stopTokens.contains("<turn|>"))
    }

    /// The 12B's template closes an empty thought channel when thinking is off.
    @Test func gemma4EmptyThoughtPrefill() {
        let t = ChatTemplate(template: "<|turn>{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}", modelType: "gemma4", bosToken: "<bos>")
        #expect(t.render([ChatMessage(role: .user, content: "Hi there")]) == "<bos><|turn>user\nHi there<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    }

    @Test func gemmaIsDetectedWithoutATemplate() {
        #expect(ChatFormat.detect(template: nil, modelType: "gemma2") == .gemma)
        #expect(ChatFormat.detect(template: nil, modelType: "gemma4_text") == .gemma4)
    }
}

@Suite("Gemma configs")
struct GemmaConfigTests {
    @Test func gemma3MultimodalExportGetsTextDefaults() throws {
        let dir = try configDir(#"{"model_type":"gemma3","vocab_size":262208,"text_config":{"model_type":"gemma3_text","hidden_size":2560,"intermediate_size":10240,"num_hidden_layers":12,"rope_scaling":{"factor":8.0,"rope_type":"linear"},"sliding_window":1024}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        let c = try ModelConfig(directory: dir)
        #expect(c.modelType == "gemma3_text" && c.numHeads == 8 && c.numKVHeads == 4 && c.headDim == 256)
        #expect(c.queryScale == 1 / Float(256).squareRoot() && c.normWeightOffset && c.sandwichNorms && c.activation == .geluTanh)
        #expect(c.slidingWindow == 1024 && c.layers.map(\.sliding) == (0..<12).map { ($0 + 1) % 6 != 0 })
        // Global layers: base 1e6 with linear scaling 8; local layers: base 1e4, unscaled.
        #expect(abs(c.layers[5].invFreq[1] - Float(1 / pow(1e6, 2.0 / 256) / 8)) < 1e-9)
        #expect(abs(c.layers[0].invFreq[1] - Float(1 / pow(1e4, 2.0 / 256))) < 1e-9)
        #expect(c.embedScale == 50.5)   // sqrt(2560) = 50.596 rounded to bf16
    }

    /// transformers >= 5 writes a flat gemma3_text config with rope_parameters only.
    @Test func gemma3FlatTransformers5Config() throws {
        let dir = try configDir(#"{"model_type":"gemma3_text","hidden_size":1152,"intermediate_size":6912,"num_hidden_layers":12,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":256,"sliding_window":512,"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"],"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},"eos_token_id":[1,106],"text_config":{"eos_token_id":1}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        let c = try ModelConfig(directory: dir)
        #expect(c.ropeTheta == 1_000_000 && c.ropeLocalTheta == 10_000 && c.slidingWindow == 512)
        #expect(c.layers[2].invFreq[1] < c.layers[0].invFreq[1])
        #expect(c.attnLogitSoftcap == 0 && c.finalLogitSoftcap == 0)
    }

    /// A flat `gemma3` config (no text_config) is the text model; Gemma 2 without the
    /// soft-capping keys gets transformers' defaults, with null meaning off.
    @Test func gemmaVariantsOfTheSameConfig() throws {
        let flat = try configDir(#"{"model_type":"gemma3","hidden_size":2560,"num_hidden_layers":12,"num_attention_heads":8,"num_key_value_heads":4,"head_dim":256,"sliding_window":1024,"sliding_window_pattern":6,"rope_theta":1e6,"rope_local_base_freq":1e4}"#)
        defer { try? FileManager.default.removeItem(at: flat) }
        let c = try ModelConfig(directory: flat)
        #expect(c.modelType == "gemma3_text" && c.slidingWindow == 1024 && c.ropeLocalTheta == 1e4 && c.layers.filter(\.sliding).count == 10)
        let g2 = try configDir(#"{"model_type":"gemma2","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"final_logit_softcapping":null}"#)
        defer { try? FileManager.default.removeItem(at: g2) }
        let c2 = try ModelConfig(directory: g2)
        #expect(c2.attnLogitSoftcap == 50 && c2.finalLogitSoftcap == 0)
    }

    @Test func gemma4LayerGeometry() throws {
        let dir = try configDir(#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text","hidden_size":256,"intermediate_size":512,"num_hidden_layers":6,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":64,"global_head_dim":128,"num_kv_shared_layers":2,"hidden_size_per_layer_input":32,"final_logit_softcapping":30.0,"sliding_window":16,"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"],"rope_parameters":{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        let c = try ModelConfig(directory: dir)
        #expect(!c.normWeightOffset && c.queryScale == 1 && c.valueNorm && c.finalLogitSoftcap == 30 && c.perLayerInputSize == 32)
        #expect(c.layers.map(\.headDim) == [64, 64, 128, 64, 64, 128])
        // Layers 4 and 5 reuse the caches of the last earlier layer of their type.
        #expect(c.layers.map(\.kvSource) == [-1, -1, -1, -1, 3, 2])
        // Sliding layers keep window + one batch of rows, full layers the whole context.
        #expect(c.kvCacheBytes(context: 4096) == 2 * 2 * (3 * 64 * (16 + 512) + 128 * 4096))
        #expect(c.kvCacheBytes(context: 256) == 2 * 2 * (3 * 64 + 128) * 256)
        // Proportional RoPE: the whole head rotates, a quarter of the frequencies are non-zero,
        // with exponents over the full head dim.
        let g = c.layers[2]
        #expect(g.rotDim == 128 && g.invFreq.count == 64)
        #expect(g.invFreq[15] == Float(1 / pow(1e6, 30.0 / 128)) && g.invFreq[16...].allSatisfy { $0 == 0 })
        #expect(c.layers[0].rotDim == 64 && c.layers[0].invFreq.allSatisfy { $0 > 0 })
    }

    @Test(arguments: [
        #"{"model_type":"gemma3_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"layer_types":["sliding_attention"]}"#,
        #"{"model_type":"gemma3_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"query_pre_attn_scalar":0}"#,
        #"{"model_type":"gemma2","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"final_logit_softcapping":-1}"#,
        #"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_kv_shared_layers":2}"#,
        #"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_kv_shared_layers":1,"layer_types":["sliding_attention","full_attention"]}"#,
        #"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"enable_moe_block":true}"#,
        #"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"global_head_dim":4096}"#,
        #"{"model_type":"gemma3_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"sliding_window":4294967316}"#,
        #"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"sliding_window":2147483653}"#,
    ])
    func rejectsInvalidGemmaConfigs(json: String) throws {
        let dir = try configDir(json)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: EngineError.self) { _ = try ModelConfig(directory: dir) }
    }
}

@Suite("Gemma models", .serialized)
struct GemmaModelTests {
    static let gemma1 = "mlx-community/gemma-1.1-2b-it-4bit"
    static let gemma2 = "mlx-community/gemma-2-2b-it-4bit"
    static let gemma3 = "mlx-community/gemma-3-1b-it-qat-4bit"
    static let gemma3Multimodal = "mlx-community/gemma-3-text-4b-it-4bit"   // `gemma3` export: text model only
    static let gemma4 = "mlx-community/gemma-4-e2b-it-OptiQ-4bit"
    static let gemma4Unified = "mlx-community/gemma-4-12B-it-OptiQ-4bit"   // K = V global attention
    /// Only installed checkpoints become test cases (a missing one must not pass silently).
    static let installed = [gemma1, gemma2, gemma3, gemma3Multimodal, gemma4, gemma4Unified].filter { installedModel($0) != nil }
    static let installedTokenizers = [gemma2, gemma3, gemma4].filter { installedModel($0) != nil }

    /// Ids from transformers' tokenizer for the same files.
    @Test(arguments: installedTokenizers)
    func tokenizerKnownIdsAndRoundTrip(repo: String) throws {
        guard let dir = installedModel(repo) else { return }
        let tok = try Tokenizer(directory: dir)
        let g2 = repo == Self.gemma2
        #expect(tok.encode("Hello world") == (g2 ? [4521, 2134] : [9259, 1902]))
        #expect(tok.encode(" leading space") == (g2 ? [8133, 3641] : [5830, 2557]))
        #expect(tok.encode("two  spaces\tand tab\n\nnew paragraph") == (g2 ? [9721, 139, 46633, 226, 639, 6684, 109, 1404, 16957]
                                                                          : [13498, 138, 35220, 255968, 624, 6937, 108, 1951, 15649]))
        for s in ["two  spaces\tand tab\n\nnew paragraph", "emoji 🚀 ünïcödé", "  indented\n    code"] {
            #expect(tok.decode(tok.encode(s)) == s)
        }
    }

    /// With a context longer than window + batch the sliding layers keep a ring of rows: a prompt
    /// past that point must still prefill and decode alike, and a run that went on for more than
    /// a batch past the divergence point cannot reuse its cache.
    @Test(.enabled(if: installedModel(gemma3) != nil))
    func ringCacheMatchesAndLimitsPrefixReuse() throws {
        guard let dir = installedModel(Self.gemma3) else { return }
        let model = try Transformer(directory: dir, contextLength: 2048)   // window 512 → 1024 cache rows
        let tok = try Tokenizer(directory: dir)
        let text = (0..<120).map { "Line \($0) of a long document about rings, windows and caches." }.joined(separator: " ")
        let ids = Array(tok.encode(text).map { Int32($0) }.prefix(1200))
        #expect(ids.count == 1200)
        let V = model.vocabSize
        let pre = Array(UnsafeBufferPointer(start: try model.forward(ids, offset: 0), count: V))
        _ = try model.forward(Array(ids[0..<1100]), offset: 0)
        var inc: [Float] = []
        for i in 1100..<ids.count {
            let l = try model.forward([ids[i]], offset: i)
            if i == ids.count - 1 { inc = Array(UnsafeBufferPointer(start: l, count: V)) }
        }
        #expect(pre.indices.max { pre[$0] < pre[$1] } == inc.indices.max { inc[$0] < inc[$1] })
        #expect(zip(pre, inc).map { abs($0 - $1) }.max()! < 0.1)
        #expect(model.reusablePrefix(900, previousLength: 1200) == 900)   // 300 newer rows: the window's rows survive
        #expect(model.reusablePrefix(500, previousLength: 1200) == 0)     // 700 newer rows overwrote them
    }

    /// Prefill and token-by-token decode agree past the sliding window (512 tokens for Gemma 3
    /// 1B and Gemma 4), so the windowed prefill and decode kernels are both exercised.
    @Test(arguments: installed)
    func prefillMatchesIncrementalDecodePastTheWindow(repo: String) throws {
        guard let dir = installedModel(repo) else { return }
        let model = try Transformer(directory: dir, contextLength: 1024)
        let tok = try Tokenizer(directory: dir)
        #expect(!tok.stripsLeadingSpace)   // Gemma's leading spaces are real: the engine must not drop them
        let text = (0..<60).map { "Sentence number \($0) talks about the quick brown fox and the lazy dog." }.joined(separator: " ")
        let ids = Array(tok.encode(text).map { Int32($0) }.prefix(600))
        #expect(ids.count == 600)
        let V = model.vocabSize
        let pre = Array(UnsafeBufferPointer(start: try model.forward(ids, offset: 0), count: V))
        _ = try model.forward(Array(ids[0..<500]), offset: 0)
        var inc: [Float] = []
        for i in 500..<ids.count {
            let l = try model.forward([ids[i]], offset: i)
            if i == ids.count - 1 { inc = Array(UnsafeBufferPointer(start: l, count: V)) }
        }
        let argPre = pre.indices.max { pre[$0] < pre[$1] }!
        let argInc = inc.indices.max { inc[$0] < inc[$1] }!
        #expect(argPre == argInc)
        let maxDiff = zip(pre, inc).map { abs($0 - $1) }.max()!
        #expect(maxDiff < 0.1, "max logit difference \(maxDiff)")
    }
}
