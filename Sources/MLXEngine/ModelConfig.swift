import Foundation

/// Architectures the engine can run. All are pre-norm decoder-only transformers
/// with RMSNorm, RoPE, grouped-query attention and a gated MLP (SwiGLU, or GeGLU for Gemma).
/// `gemma3` is the multimodal export: only its language model is used.
public enum SupportedArchitectures {
    public static let modelTypes: Set<String> = ["llama", "mistral", "qwen2", "qwen3", "phi3",
                                                 "gemma", "gemma2", "gemma3", "gemma3_text",
                                                 "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"]

    public static func isSupported(_ modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        return modelTypes.contains(t)
    }

    /// Like `isSupported`, also ruling out variants the type alone does not reveal: Gemma 4
    /// mixture-of-experts exports (`enable_moe_block` in the text config).
    public static func isSupported(_ modelType: String?, config: [String: Any]?) -> Bool {
        guard isSupported(modelType) else { return false }
        return !isMoE(config)
    }

    public static func isMoE(_ config: [String: Any]?) -> Bool {
        guard let c = config else { return false }
        let text = c["text_config"] as? [String: Any]
        return (c["enable_moe_block"] as? Bool) == true || (text?["enable_moe_block"] as? Bool) == true
    }
}

public struct ModelConfig {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numLayers: Int
    public var numHeads: Int
    public var numKVHeads: Int
    public var headDim: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeTraditional: Bool
    public var partialRotaryFactor: Float
    public var ropeScaling: [String: Any]?
    public var tieWordEmbeddings: Bool
    public var maxPositionEmbeddings: Int
    public var originalMaxPosition: Int
    public var eosTokenIds: [Int]
    public var bits: Int?
    public var groupSize: Int?

    // Gemma family (all neutral for the other architectures).
    public enum Activation: String { case silu, geluTanh, gelu }
    /// Gate activation of the MLP.
    public var activation: Activation = .silu
    /// RMSNorm weights are stored as offsets from 1 (`x * (1 + w)`).
    public var normWeightOffset = false
    /// The token embeddings are multiplied by this (Gemma: sqrt(hidden_size), rounded to bf16).
    public var embedScale: Float?
    /// Gemma 2/3: the attention and MLP outputs are normalized before the residual add
    /// (`post_attention_layernorm`, `post_feedforward_layernorm`), and the MLP input uses
    /// `pre_feedforward_layernorm`.
    public var sandwichNorms = false
    /// Multiplier of q·k (default 1/sqrt(head_dim); Gemma: query_pre_attn_scalar^-0.5).
    public var queryScale: Float
    /// tanh soft-capping of attention scores and of the final logits (0 = off).
    public var attnLogitSoftcap: Float = 0
    public var finalLogitSoftcap: Float = 0
    /// Sliding-window size of the local layers (0 = every layer attends to the whole context).
    public var slidingWindow = 0
    /// Per layer: true for sliding-window (local) attention.
    public var slidingLayers: [Bool] = []
    /// RoPE base of the local layers (Gemma 3), which also use no RoPE scaling.
    public var ropeLocalTheta: Float?

    public var isGemma: Bool { modelType.hasPrefix("gemma") }
    public var isGemma4: Bool { modelType.hasPrefix("gemma4") }

    /// Geometry of one layer. Uniform for every architecture except Gemma 4, whose global layers
    /// have their own head size, KV head count and RoPE, and whose last layers reuse the KV cache
    /// of an earlier layer.
    public struct LayerSpec {
        public var headDim: Int
        public var numKVHeads: Int
        public var sliding: Bool
        /// The layer whose K/V cache this layer attends over (itself unless KV-shared).
        public var kvSource: Int
        /// V is the raw k_proj output (no v_proj; Gemma 4 `attention_k_eq_v`).
        public var kEqV = false
        /// Inverse RoPE frequencies (rotDim / 2 entries) and the number of rotated dims.
        public var invFreq: [Float]
        public var rotDim: Int
        public var hasOwnKV: Bool { kvSource < 0 }
    }
    public private(set) var layers: [LayerSpec] = []
    /// Gemma 4: per-layer input embeddings (`hidden_size_per_layer_input`, 0 = none) and the
    /// weightless RMSNorm on V.
    public var perLayerInputSize = 0
    public var valueNorm = false

    /// KV cache bytes per token of context (f16 K and V, layers with their own cache).
    public var kvBytesPerToken: Int { layers.reduce(0) { $0 + ($1.hasOwnKV ? 2 * $1.numKVHeads * $1.headDim * 2 : 0) } }

    public init(directory: URL) throws {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              var obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw EngineError.invalidFile("config.json not found or unreadable")
        }
        // The outer config's stop tokens (a multimodal export's text_config may list fewer).
        var eos: [Int] = []
        if let e = obj["eos_token_id"] as? NSNumber { eos.append(e.intValue) }
        if let e = obj["eos_token_id"] as? [NSNumber] { eos += e.map(\.intValue) }
        // Some multimodal exports nest the language model config.
        if obj["hidden_size"] == nil, let text = obj["text_config"] as? [String: Any] {
            obj.merge(text) { _, new in new }
        }
        var type = (obj["model_type"] as? String ?? "").lowercased()
        if type == "gemma4_text" || type == "gemma4_unified_text" {
            obj.merge(["num_attention_heads": 8, "num_key_value_heads": 1]) { old, _ in old }   // as mlx-lm
        }
        if type == "gemma3" { type = "gemma3_text" }   // a flat multimodal config: the text model's own type
        if type == "gemma3_text" {
            // Gemma 3 configs (multimodal exports, newer transformers) leave the text model's
            // defaults out (transformers' Gemma3TextConfig, with the head counts mlx-lm assumes).
            let defaults: [String: Any] = [
                "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256, "rms_norm_eps": 1e-6,
                "rope_local_base_freq": 10_000.0, "query_pre_attn_scalar": 256,
                "sliding_window": 4096, "sliding_window_pattern": 6, "max_position_embeddings": 131_072,
                "hidden_activation": "gelu_pytorch_tanh",
            ]
            obj.merge(defaults) { old, _ in old }
        }
        // transformers >= 5 stores RoPE under rope_parameters[.full_attention] instead of
        // rope_theta / rope_scaling / rope_local_base_freq.
        if obj["rope_theta"] == nil, let rp = obj["rope_parameters"] as? [String: Any], !type.hasPrefix("gemma4") {
            let full = (rp["full_attention"] as? [String: Any]) ?? rp
            if let t = full["rope_theta"] { obj["rope_theta"] = t }
            if let kind = full["rope_type"] as? String, kind != "default", obj["rope_scaling"] == nil {
                obj["rope_scaling"] = full
            }
            if let local = (rp["sliding_attention"] as? [String: Any])?["rope_theta"], obj["rope_local_base_freq"] == nil {
                obj["rope_local_base_freq"] = local
            }
        }
        if type == "gemma3_text" { obj.merge(["rope_theta": 1_000_000.0]) { old, _ in old } }
        func int(_ k: String) -> Int? { (obj[k] as? NSNumber)?.intValue }
        func float(_ k: String) -> Float? { (obj[k] as? NSNumber)?.floatValue }

        modelType = type
        guard SupportedArchitectures.isSupported(modelType) else {
            throw EngineError.unsupported("architecture '\(modelType)' is not supported yet (supported: \(SupportedArchitectures.modelTypes.sorted().joined(separator: ", ")))")
        }
        guard let h = int("hidden_size"), let l = int("num_hidden_layers"), let nh = int("num_attention_heads") else {
            throw EngineError.invalidFile("config.json is missing core dimensions")
        }
        hiddenSize = h
        numLayers = l
        numHeads = nh
        intermediateSize = int("intermediate_size") ?? 4 * h
        numKVHeads = int("num_key_value_heads") ?? nh
        headDim = int("head_dim") ?? h / nh
        vocabSize = int("vocab_size") ?? 0
        rmsNormEps = float("rms_norm_eps") ?? 1e-6
        ropeTheta = float("rope_theta") ?? 10000
        ropeTraditional = (obj["rope_traditional"] as? Bool) ?? false
        partialRotaryFactor = float("partial_rotary_factor") ?? 1
        ropeScaling = obj["rope_scaling"] as? [String: Any]
        tieWordEmbeddings = (obj["tie_word_embeddings"] as? Bool) ?? false
        maxPositionEmbeddings = int("max_position_embeddings") ?? 4096
        originalMaxPosition = int("original_max_position_embeddings") ?? maxPositionEmbeddings

        // config.json is untrusted: bound every dimension so later size math cannot overflow.
        let dims: [(String, Int, ClosedRange<Int>)] = [
            ("hidden_size", hiddenSize, 1...65536), ("intermediate_size", intermediateSize, 1...262_144),
            ("num_hidden_layers", numLayers, 1...512), ("num_attention_heads", numHeads, 1...1024),
            ("num_key_value_heads", numKVHeads, 1...1024), ("head_dim", headDim, 1...1024),
            ("max_position_embeddings", maxPositionEmbeddings, 1...(1 << 24)),
        ]
        for (name, v, range) in dims where !range.contains(v) {
            throw EngineError.invalidFile("config.json: \(name) = \(v) is out of range")
        }
        guard numHeads % numKVHeads == 0 else {
            throw EngineError.invalidFile("config.json: num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard rmsNormEps.isFinite, rmsNormEps > 0, ropeTheta.isFinite, ropeTheta > 0,
              partialRotaryFactor > 0, partialRotaryFactor <= 1 else {
            throw EngineError.invalidFile("config.json: invalid rms_norm_eps, rope_theta or partial_rotary_factor")
        }
        queryScale = 1 / Float(headDim).squareRoot()

        if modelType.hasPrefix("gemma") && !modelType.hasPrefix("gemma4") {
            normWeightOffset = true
            // Rounded to the checkpoint's bf16, as transformers and mlx-lm do.
            embedScale = Float(BFloat16Rounding.round(Float(hiddenSize).squareRoot()))
            let act = (obj["hidden_activation"] as? String) ?? (obj["hidden_act"] as? String) ?? "gelu_pytorch_tanh"
            // mlx-lm runs Gemma 1 with the exact GELU; the later models use the tanh approximation.
            activation = modelType == "gemma" && act == "gelu" ? .gelu : .geluTanh
            if modelType != "gemma" { sandwichNorms = true }
            if let s = float("query_pre_attn_scalar") {
                guard s.isFinite, s > 0 else { throw EngineError.invalidFile("config.json: invalid query_pre_attn_scalar") }
                queryScale = 1 / s.squareRoot()
            }
            // null: off. Absent: transformers' Gemma2Config defaults (50 / 30); off for Gemma 1 and 3.
            func softcap(_ key: String, default d: Float) throws -> Float {
                guard let raw = obj[key] else { return type == "gemma2" ? d : 0 }
                guard let v = (raw as? NSNumber)?.floatValue else { return 0 }
                guard v.isFinite, v > 0 else { throw EngineError.invalidFile("config.json: invalid \(key)") }
                return v
            }
            attnLogitSoftcap = try softcap("attn_logit_softcapping", default: 50)
            finalLogitSoftcap = try softcap("final_logit_softcapping", default: 30)
            if modelType == "gemma2" || modelType == "gemma3_text" {
                let window = int("sliding_window") ?? 0
                guard (0...(1 << 24)).contains(window) else { throw EngineError.invalidFile("config.json: sliding_window = \(window) is out of range") }
                if let types = obj["layer_types"] as? [String] {
                    guard types.count == numLayers else { throw EngineError.invalidFile("config.json: layer_types has \(types.count) entries for \(numLayers) layers") }
                    slidingLayers = types.map { $0 == "sliding_attention" }
                } else if modelType == "gemma2" {
                    slidingLayers = (0..<numLayers).map { $0 % 2 == 0 }
                } else {
                    let pattern = int("sliding_window_pattern") ?? int("_sliding_window_pattern") ?? 6
                    guard pattern >= 1 else { throw EngineError.invalidFile("config.json: invalid sliding_window_pattern") }
                    slidingLayers = (0..<numLayers).map { ($0 + 1) % pattern != 0 }
                }
                if window > 0 && slidingLayers.contains(true) { slidingWindow = window } else { slidingLayers = [] }
            }
            if modelType == "gemma3_text", let local = float("rope_local_base_freq") {
                guard local.isFinite, local > 0 else { throw EngineError.invalidFile("config.json: invalid rope_local_base_freq") }
                ropeLocalTheta = local
            }
        }

        if let e = obj["eos_token_id"] as? NSNumber { eos.append(e.intValue) }
        if let e = obj["eos_token_id"] as? [NSNumber] { eos += e.map(\.intValue) }
        if let gdata = try? Data(contentsOf: directory.appendingPathComponent("generation_config.json")),
           let g = try? JSONSerialization.jsonObject(with: gdata) as? [String: Any] {
            if let e = g["eos_token_id"] as? NSNumber { eos.append(e.intValue) }
            if let e = g["eos_token_id"] as? [NSNumber] { eos += e.map(\.intValue) }
        }
        eosTokenIds = Array(Set(eos))

        let q = (obj["quantization"] as? [String: Any]) ?? (obj["quantization_config"] as? [String: Any])
        bits = (q?["bits"] as? NSNumber)?.intValue
        groupSize = (q?["group_size"] as? NSNumber)?.intValue

        if modelType.hasPrefix("gemma4") { try parseGemma4(obj) }
        if layers.isEmpty {
            let global = inverseFrequencies(), local = ropeLocalTheta != nil ? inverseFrequencies(local: true) : global
            layers = (0..<numLayers).map { i in
                LayerSpec(headDim: headDim, numKVHeads: numKVHeads, sliding: isSliding(i), kvSource: -1,
                          invFreq: isSliding(i) ? local : global, rotDim: rotaryDim)
            }
        }
    }

    /// Gemma 4 (`gemma4_text`, and `gemma4_unified_text` which mlx-lm runs with the same model).
    private mutating func parseGemma4(_ obj: [String: Any]) throws {
        func int(_ k: String) -> Int? { (obj[k] as? NSNumber)?.intValue }
        func float(_ k: String) -> Float? { (obj[k] as? NSNumber)?.floatValue }
        func bad(_ what: String) -> EngineError { .invalidFile("config.json: \(what)") }
        if (obj["enable_moe_block"] as? Bool) == true {
            throw EngineError.unsupported("Gemma 4 mixture-of-experts models are not supported yet")
        }
        // Plain RMSNorm weights; q_norm / k_norm replace the 1/sqrt(d) attention scale.
        embedScale = BFloat16Rounding.round(Float(hiddenSize).squareRoot())
        activation = .geluTanh
        sandwichNorms = true
        valueNorm = true
        queryScale = 1
        if let v = float("final_logit_softcapping") {
            guard v.isFinite, v > 0 else { throw bad("invalid final_logit_softcapping") }
            finalLogitSoftcap = v
        }
        let window = int("sliding_window") ?? 512
        guard (1...(1 << 24)).contains(window) else { throw bad("sliding_window = \(window) is out of range") }
        slidingWindow = window
        let types: [String]
        if let t = obj["layer_types"] as? [String] {
            guard t.count == numLayers else { throw bad("layer_types has \(t.count) entries for \(numLayers) layers") }
            types = t
        } else {
            let pattern = int("sliding_window_pattern") ?? 5
            guard pattern >= 1 else { throw bad("invalid sliding_window_pattern") }
            types = (0..<numLayers).map { ($0 + 1) % pattern == 0 ? "full_attention" : "sliding_attention" }
        }
        guard types.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" }) else { throw bad("unknown layer type") }
        slidingLayers = types.map { $0 == "sliding_attention" }
        let globalHeadDim = int("global_head_dim") ?? headDim
        let globalKV = int("num_global_key_value_heads")
        let kEqV = (obj["attention_k_eq_v"] as? Bool) ?? false
        for (name, v) in [("global_head_dim", globalHeadDim), ("num_global_key_value_heads", globalKV ?? numKVHeads)] {
            guard (1...1024).contains(v) else { throw bad("\(name) = \(v) is out of range") }
        }
        let shared = int("num_kv_shared_layers") ?? 0
        guard (0..<numLayers).contains(shared) else { throw bad("num_kv_shared_layers = \(shared) is out of range") }
        perLayerInputSize = int("hidden_size_per_layer_input") ?? 0
        guard (0...4096).contains(perLayerInputSize) else { throw bad("hidden_size_per_layer_input is out of range") }

        let rope = obj["rope_parameters"] as? [String: Any] ?? [:]
        func ropeFor(_ type: String, dim: Int) throws -> (freqs: [Float], rotDim: Int) {
            let p = rope[type] as? [String: Any] ?? [:]
            let theta = (p["rope_theta"] as? NSNumber)?.doubleValue ?? (type == "full_attention" ? 1_000_000 : 10_000)
            let kind = (p["rope_type"] as? String) ?? "default"
            let partial = (p["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 1
            let factor = (p["factor"] as? NSNumber)?.doubleValue ?? 1
            guard theta.isFinite, theta > 0, factor.isFinite, factor > 0, partial > 0, partial <= 1 else { throw bad("invalid rope_parameters") }
            switch kind {
            case "default":
                return ((0..<dim / 2).map { Float(1 / pow(theta, Double(2 * $0) / Double(dim))) }, dim)
            case "proportional":
                // mlx-lm ProportionalRoPE: the whole head rotates (i pairs with i + dim/2), exponents
                // over the full dim; only the first rotated/2 frequencies are non-zero.
                let rotated = Int(Double(dim) * partial)
                return ((0..<dim / 2).map { i in 2 * i < rotated ? Float(1 / (factor * pow(theta, Double(2 * i) / Double(dim)))) : 0 }, dim)
            default:
                throw EngineError.unsupported("Gemma 4 RoPE type '\(kind)'")
            }
        }
        let firstShared = numLayers - shared
        var lastOfType: [Bool: Int] = [:]
        for i in 0..<numLayers {
            let sliding = slidingLayers[i]
            let dim = sliding ? headDim : globalHeadDim
            guard dim % 32 == 0 else { throw EngineError.unsupported("head_dim \(dim)") }
            let nkv = !sliding && kEqV ? (globalKV ?? numKVHeads) : numKVHeads
            guard numHeads % nkv == 0 else { throw bad("num_attention_heads must be a multiple of the KV head count") }
            var source = -1
            if i >= firstShared {
                guard let s = lastOfType[sliding] else { throw bad("KV-shared layer \(i) has no earlier layer of its type") }
                source = s
            } else {
                lastOfType[sliding] = i
            }
            let r = try ropeFor(sliding ? "sliding_attention" : "full_attention", dim: dim)
            var spec = LayerSpec(headDim: dim, numKVHeads: nkv, sliding: sliding, kvSource: source, kEqV: !sliding && kEqV,
                                 invFreq: r.freqs, rotDim: r.rotDim)
            if source >= 0 { spec.numKVHeads = layers[source].numKVHeads; spec.kEqV = layers[source].kEqV }
            layers.append(spec)
        }
    }

    /// Is layer `i` a sliding-window (local attention) layer?
    func isSliding(_ i: Int) -> Bool { slidingWindow > 0 && i < slidingLayers.count && slidingLayers[i] }

    /// Inverse RoPE frequencies, including Llama 3 / linear scaling. `local`: the frequencies of
    /// Gemma 3's sliding-window layers (their own base, no scaling).
    func inverseFrequencies(local: Bool = false) -> [Float] {
        let rotDim = Int(Float(headDim) * partialRotaryFactor) / 2 * 2
        let half = rotDim / 2
        let theta = local ? (ropeLocalTheta ?? ropeTheta) : ropeTheta
        var freqs = (0..<half).map { i in 1.0 / pow(Double(theta), Double(2 * i) / Double(rotDim)) }
        if !local, let rs = ropeScaling {
            let type = (rs["rope_type"] as? String) ?? (rs["type"] as? String) ?? ""
            let factor = (rs["factor"] as? NSNumber)?.doubleValue ?? 1
            if type == "llama3" {
                let low = (rs["low_freq_factor"] as? NSNumber)?.doubleValue ?? 1
                let high = (rs["high_freq_factor"] as? NSNumber)?.doubleValue ?? 4
                let oldCtx = (rs["original_max_position_embeddings"] as? NSNumber)?.doubleValue ?? 8192
                let lowWavelen = oldCtx / low, highWavelen = oldCtx / high
                freqs = freqs.map { f in
                    let wavelen = 2 * Double.pi / f
                    if wavelen < highWavelen { return f }
                    if wavelen > lowWavelen { return f / factor }
                    let smooth = (oldCtx / wavelen - low) / (high - low)
                    return (1 - smooth) * f / factor + smooth * f
                }
            } else if type == "linear" {
                freqs = freqs.map { $0 / factor }
            } else if type == "longrope" || type == "su", let long = rs["long_factor"] as? [NSNumber], long.count == half {
                // Phi-3 LongRoPE (matches mlx-lm SuScaledRotaryEmbedding: long factors always).
                freqs = zip(freqs, long).map { $0 / $1.doubleValue }
            }
        }
        return freqs.map { Float($0) }
    }

    /// Multiplier applied to q and k by scaled RoPE variants (LongRoPE), folded into the attention scale.
    var ropeAttentionFactor: Float {
        guard let rs = ropeScaling else { return 1 }
        let type = (rs["rope_type"] as? String) ?? (rs["type"] as? String) ?? ""
        guard type == "longrope" || type == "su" else { return 1 }
        if let m = (rs["long_mscale"] as? NSNumber)?.floatValue { return m }
        let orig = (rs["original_max_position_embeddings"] as? NSNumber)?.floatValue ?? Float(originalMaxPosition)
        let scale = Float(maxPositionEmbeddings) / orig
        return scale <= 1 ? 1 : (1 + log(scale) / log(orig)).squareRoot()
    }

    var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) / 2 * 2 }
}

/// Round-to-nearest-even to bfloat16 precision.
enum BFloat16Rounding {
    static func round(_ x: Float) -> Float {
        guard x.isFinite else { return x }
        let b = x.bitPattern
        let rounded = (b &+ 0x7FFF &+ ((b >> 16) & 1)) & 0xFFFF_0000
        return Float(bitPattern: rounded)
    }
}
