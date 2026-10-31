import Foundation

public struct SamplingParameters: Codable, Hashable, Sendable {
    public var temperature: Float = 0.7
    public var topP: Float = 0.95
    public var topK: Int = 40
    public var minP: Float = 0.0
    public var repetitionPenalty: Float = 1.05
    public var repetitionWindow: Int = 64
    public var maxTokens: Int = 4096
    public var seed: UInt64? = nil
    public init() {}
}

public struct GenerationStats: Sendable, Hashable, Codable {
    public var promptTokens: Int = 0
    public var cachedPromptTokens: Int = 0
    public var generatedTokens: Int = 0
    public var timeToFirstToken: Double = 0
    public var promptTokensPerSecond: Double = 0
    public var tokensPerSecond: Double = 0
    public var stopReason: String = ""
    public init() {}
}

public enum GenerationEvent: Sendable {
    case promptProcessed(GenerationStats)
    case text(String)
    case finished(GenerationStats)
}

public struct LoadedModelInfo: Sendable, Hashable {
    public let name: String
    public let directory: URL
    public let modelType: String
    public let quantization: String
    public let parameterBytes: Int
    public let contextLength: Int
    public let layers: Int
    public let hiddenSize: Int
    public let chatFormat: String
    public let supportsThinking: Bool
    public let loadSeconds: Double
}

/// Minimal xorshift RNG so sampling is seedable.
struct XorShift: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
    mutating func next() -> UInt64 {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17
        return state
    }
}

/// Picks the next token from logits.
struct Sampler {
    var params: SamplingParameters
    var rng: XorShift

    mutating func sample(_ logits: UnsafeMutablePointer<Float>, count: Int, recent: ArraySlice<Int>) -> Int {
        if params.repetitionPenalty != 1, !recent.isEmpty {
            let p = params.repetitionPenalty
            for t in Set(recent) where t < count {
                logits[t] = logits[t] > 0 ? logits[t] / p : logits[t] * p
            }
        }
        if params.temperature <= 0.001 {
            var best = 0
            var bestV = -Float.infinity
            for i in 0..<count where logits[i] > bestV { bestV = logits[i]; best = i }
            return best
        }
        // Top-k candidates by partial selection (k capped to keep this O(V)).
        let k = params.topK > 0 ? min(params.topK, count) : min(512, count)
        var cand: [(Int, Float)] = []
        cand.reserveCapacity(k + 1)
        var minV = -Float.infinity
        for i in 0..<count {
            let v = logits[i]
            if cand.count < k {
                cand.append((i, v))
                if cand.count == k { cand.sort { $0.1 > $1.1 }; minV = cand[k - 1].1 }
            } else if v > minV {
                // insert keeping descending order
                var lo = 0, hi = k - 1
                while lo < hi { let mid = (lo + hi) / 2; if cand[mid].1 >= v { lo = mid + 1 } else { hi = mid } }
                cand.insert((i, v), at: lo)
                cand.removeLast()
                minV = cand[k - 1].1
            }
        }
        if cand.count < k { cand.sort { $0.1 > $1.1 } }
        let invT = 1 / params.temperature
        let maxV = cand[0].1
        var probs = cand.map { exp(($0.1 - maxV) * invT) }
        var total = probs.reduce(0, +)
        // min-p and top-p filtering
        var cut = probs.count
        if params.minP > 0 {
            let threshold = params.minP * probs[0]
            cut = probs.firstIndex(where: { $0 < threshold }) ?? cut
        }
        if params.topP < 1 {
            var acc: Float = 0
            for i in 0..<cut {
                acc += probs[i] / total
                if acc >= params.topP { cut = i + 1; break }
            }
        }
        probs = Array(probs[0..<max(1, cut)])
        total = probs.reduce(0, +)
        var r = Float.random(in: 0..<1, using: &rng) * total
        for (i, p) in probs.enumerated() {
            r -= p
            if r <= 0 { return cand[i].0 }
        }
        return cand[probs.count - 1].0
    }
}

/// Owns a loaded model and runs generation on a dedicated high-priority thread.
public final class InferenceEngine: @unchecked Sendable {
    public static let shared = InferenceEngine()

    private let queue = DispatchQueue(label: "mlxrunner.inference", qos: .userInitiated)
    private var model: Transformer?
    private var tokenizer: Tokenizer?
    private var template: ChatTemplate?
    private var stopIds: Set<Int> = []
    private var cachedTokens: [Int] = []
    private let cancelLock = NSLock()
    private var cancelled = false
    public private(set) var info: LoadedModelInfo?

    public init() {}

    public var isLoaded: Bool { queue.sync { model != nil } }

    public func load(directory: URL, contextLength: Int = 8192) async throws -> LoadedModelInfo {
        try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let start = Date()
                    self.model = nil
                    self.tokenizer = nil
                    self.info = nil
                    self.cachedTokens = []
                    let tok = try Tokenizer(directory: directory)
                    let model = try Transformer(directory: directory, contextLength: contextLength)
                    let tpl = ChatTemplate(template: tok.chatTemplate, modelType: model.config.modelType, bosToken: tok.bosToken)
                    var stops = Set(model.config.eosTokenIds)
                    if let e = tok.eosToken, let id = tok.tokenId(e) { stops.insert(id) }
                    for s in tpl.stopTokens { if let id = tok.tokenId(s) { stops.insert(id) } }
                    // Warm-up pass compiles pipelines and faults in weights.
                    _ = try model.forward([Int32(tok.tokenId(tok.bosToken ?? "") ?? 0)], offset: 0)
                    self.model = model
                    self.tokenizer = tok
                    self.template = tpl
                    self.stopIds = stops
                    let info = LoadedModelInfo(
                        name: directory.lastPathComponent, directory: directory,
                        modelType: model.config.modelType, quantization: model.quantDescription,
                        parameterBytes: model.weightBytes, contextLength: model.maxSeq,
                        layers: model.config.numLayers, hiddenSize: model.config.hiddenSize,
                        chatFormat: tpl.format.rawValue, supportsThinking: tpl.supportsThinkingToggle,
                        loadSeconds: Date().timeIntervalSince(start))
                    self.info = info
                    cont.resume(returning: info)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    public func unload() {
        cancel()
        queue.async {
            self.model = nil
            self.tokenizer = nil
            self.template = nil
            self.info = nil
            self.cachedTokens = []
        }
    }

    public func cancel() {
        cancelLock.lock(); cancelled = true; cancelLock.unlock()
    }

    private var isCancelled: Bool {
        cancelLock.lock(); defer { cancelLock.unlock() }
        return cancelled
    }

    /// Streams a chat completion.
    public func chat(_ messages: [ChatMessage], parameters: SamplingParameters = .init(), enableThinking: Bool = true)
        -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { cont in
            cancelLock.lock(); cancelled = false; cancelLock.unlock()
            queue.async {
                guard let tok = self.tokenizer, let tpl = self.template else {
                    cont.finish(throwing: EngineError.invalidFile("No model loaded"))
                    return
                }
                let prompt = tpl.render(messages, enableThinking: enableThinking)
                do {
                    try self.run(tokens: tok.encode(prompt), parameters: parameters) { cont.yield($0) }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
        }
    }

    /// Streams a raw completion for a prompt string (special tokens are parsed).
    public func complete(_ prompt: String, parameters: SamplingParameters = .init()) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { cont in
            cancelLock.lock(); cancelled = false; cancelLock.unlock()
            queue.async {
                guard let tok = self.tokenizer else {
                    cont.finish(throwing: EngineError.invalidFile("No model loaded"))
                    return
                }
                do {
                    try self.run(tokens: tok.encode(prompt), parameters: parameters) { cont.yield($0) }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
        }
    }

    /// Token-level API used by the CLI and tests. Must not be called concurrently with generation.
    public func encode(_ text: String) -> [Int] { queue.sync { tokenizer?.encode(text) ?? [] } }
    public func decode(_ ids: [Int]) -> String { queue.sync { tokenizer?.decode(ids) ?? "" } }
    public func renderChat(_ messages: [ChatMessage], enableThinking: Bool = true) -> String {
        queue.sync { template?.render(messages, enableThinking: enableThinking) ?? "" }
    }

    private func run(tokens: [Int], parameters: SamplingParameters, emit: (GenerationEvent) -> Void) throws {
        guard let model, let tok = tokenizer else { throw EngineError.invalidFile("No model loaded") }
        var stats = GenerationStats()
        stats.promptTokens = tokens.count
        guard !tokens.isEmpty else { throw EngineError.invalidFile("Empty prompt") }
        guard tokens.count < model.maxSeq else { throw EngineError.contextFull }

        // Reuse the KV cache for the shared prefix with the previous request.
        var common = 0
        let limit = min(cachedTokens.count, tokens.count - 1)
        while common < limit && cachedTokens[common] == tokens[common] { common += 1 }
        stats.cachedPromptTokens = common

        cachedTokens = Array(tokens[0..<common])
        let t0 = Date()
        var logits = try model.forward(tokens[common...].map { Int32($0) }, offset: common)
        var history = tokens
        cachedTokens = tokens
        let tPrompt = Date().timeIntervalSince(t0)
        stats.timeToFirstToken = tPrompt
        stats.promptTokensPerSecond = Double(tokens.count - common) / max(tPrompt, 1e-6)
        emit(.promptProcessed(stats))

        var sampler = Sampler(params: parameters, rng: XorShift(seed: parameters.seed ?? UInt64.random(in: 1...UInt64.max)))
        var decoder = StreamingDecoder()
        let tGen = Date()
        stats.stopReason = "length"
        let maxNew = min(parameters.maxTokens, model.maxSeq - tokens.count)
        for _ in 0..<max(0, maxNew) {
            if isCancelled { stats.stopReason = "cancelled"; break }
            let window = history.suffix(parameters.repetitionWindow)
            let next = sampler.sample(logits, count: min(model.vocabSize, tok.vocabularySize), recent: window)
            if stopIds.contains(next) { stats.stopReason = "stop"; break }
            stats.generatedTokens += 1
            let text = decoder.append(tok.bytes(for: next))
            if !text.isEmpty { emit(.text(text)) }
            history.append(next)
            if history.count >= model.maxSeq { stats.stopReason = "context"; break }
            logits = try model.forward([Int32(next)], offset: history.count - 1)
            cachedTokens = history
        }
        let rest = decoder.flush()
        if !rest.isEmpty { emit(.text(rest)) }
        let dt = Date().timeIntervalSince(tGen)
        stats.tokensPerSecond = Double(stats.generatedTokens) / max(dt, 1e-6)
        emit(.finished(stats))
    }
}
