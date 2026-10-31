import Foundation

/// Architectures the engine can run. All are pre-norm decoder-only transformers
/// with RMSNorm, RoPE, grouped-query attention and a SwiGLU MLP.
public enum SupportedArchitectures {
    public static let modelTypes: Set<String> = ["llama", "mistral", "qwen2", "qwen3", "phi3"]

    public static func isSupported(_ modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        return modelTypes.contains(t)
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
    public var eosTokenIds: [Int]
    public var bits: Int?
    public var groupSize: Int?

    public init(directory: URL) throws {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              var obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw EngineError.invalidFile("config.json not found or unreadable")
        }
        // Some multimodal exports nest the language model config.
        if obj["hidden_size"] == nil, let text = obj["text_config"] as? [String: Any] {
            obj.merge(text) { _, new in new }
        }
        func int(_ k: String) -> Int? { (obj[k] as? NSNumber)?.intValue }
        func float(_ k: String) -> Float? { (obj[k] as? NSNumber)?.floatValue }

        modelType = (obj["model_type"] as? String ?? "").lowercased()
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

        var eos: [Int] = []
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
    }

    /// Inverse RoPE frequencies, including Llama 3 / linear scaling.
    func inverseFrequencies() -> [Float] {
        let rotDim = Int(Float(headDim) * partialRotaryFactor) / 2 * 2
        let half = rotDim / 2
        var freqs = (0..<half).map { i in 1.0 / pow(Double(ropeTheta), Double(2 * i) / Double(rotDim)) }
        if let rs = ropeScaling {
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
            }
        }
        return freqs.map { Float($0) }
    }

    var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) / 2 * 2 }
}
