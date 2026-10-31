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
    private var currentToken: CancelToken?
    private var trimState: (anchor: String, dropped: Int)?
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
                    try Self.checkMemory(directory: directory, contextLength: contextLength)
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

    /// Stops whichever request is currently generating (queued requests are unaffected).
    public func cancel() {
        cancelLock.lock(); let t = currentToken; cancelLock.unlock()
        t?.cancel()
    }

    /// Streams a chat completion. Cancel with `token`, or simply stop consuming the stream.
    public func chat(_ messages: [ChatMessage], parameters: SamplingParameters = .init(), enableThinking: Bool = true,
                     token: CancelToken = CancelToken()) -> AsyncThrowingStream<GenerationEvent, Error> {
        stream(token, parameters: parameters) { tok, tpl in
            guard let tpl else { throw EngineError.invalidFile("No model loaded") }
            return self.fitToContext(messages, template: tpl, tokenizer: tok,
                                     reserve: min(parameters.maxTokens, 1024), enableThinking: enableThinking)
        }
    }

    /// Streams a raw completion for a prompt string (special tokens are parsed).
    public func complete(_ prompt: String, parameters: SamplingParameters = .init(),
                         token: CancelToken = CancelToken()) -> AsyncThrowingStream<GenerationEvent, Error> {
        stream(token, parameters: parameters) { tok, _ in tok.encode(prompt) }
    }

    private func stream(_ token: CancelToken, parameters: SamplingParameters,
                        _ makeTokens: @escaping (Tokenizer, ChatTemplate?) throws -> [Int]) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { cont in
            cont.onTermination = { _ in token.cancel() }
            queue.async {
                guard !token.isCancelled else { cont.finish(); return }
                guard let tok = self.tokenizer else {
                    cont.finish(throwing: EngineError.invalidFile("No model loaded"))
                    return
                }
                self.cancelLock.lock(); self.currentToken = token; self.cancelLock.unlock()
                defer { self.cancelLock.lock(); self.currentToken = nil; self.cancelLock.unlock() }
                do {
                    let tokens = try makeTokens(tok, self.template)
                    try self.run(tokens: tokens, parameters: parameters, token: token) { cont.yield($0) }
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

    /// Encodes the conversation, dropping the oldest turns (never the system prompt or the
    /// latest message) so it fits in the context window with room to answer.
    ///
    /// The cut point is sticky: it is remembered per conversation and only advances, and when
    /// it does it drops down to a low watermark (70% of the window). Consecutive turns then
    /// share a stable prefix, so the KV cache keeps being reused instead of re-prefilling
    /// the whole window on every turn.
    private func fitToContext(_ messages: [ChatMessage], template: ChatTemplate, tokenizer: Tokenizer,
                              reserve: Int, enableThinking: Bool) -> [Int] {
        let limit = (model?.maxSeq ?? Int.max) - reserve
        let system = messages.filter { $0.role == .system }
        let turns = messages.filter { $0.role != .system }
        guard let first = turns.first else {
            return tokenizer.encode(template.render(messages, enableThinking: enableThinking))
        }
        let anchor = first.role.rawValue + "\u{0}" + first.content
        var dropped = (trimState?.anchor == anchor) ? min(trimState!.dropped, turns.count - 1) : 0

        func render(_ d: Int) -> [Int] {
            tokenizer.encode(template.render(system + turns[d...], enableThinking: enableThinking))
        }
        /// Next cut index > d that keeps the history starting with a user turn.
        func advance(_ d: Int) -> Int? {
            var n = d + 1
            while n < turns.count - 1 && turns[n].role != .user { n += 1 }
            return n < turns.count ? n : nil
        }
        var tokens = render(dropped)
        if tokens.count > limit {
            let target = Int(Double(limit) * 0.7)
            while tokens.count > target, let n = advance(dropped), n <= turns.count - 1 {
                dropped = n
                tokens = render(dropped)
                if dropped == turns.count - 1 { break }
            }
        }
        trimState = (anchor, dropped)
        return tokens
    }

    private func run(tokens: [Int], parameters: SamplingParameters, token: CancelToken, emit: (GenerationEvent) -> Void) throws {
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
        var atStart = true
        for _ in 0..<max(0, maxNew) {
            if token.isCancelled { stats.stopReason = "cancelled"; break }
            let window = history.suffix(parameters.repetitionWindow)
            let next = sampler.sample(logits, count: min(model.vocabSize, tok.vocabularySize), recent: window)
            if stopIds.contains(next) { stats.stopReason = "stop"; break }
            stats.generatedTokens += 1
            var text = decoder.append(tok.bytes(for: next))
            // SentencePiece models start the reply with a word-boundary space.
            if atStart, tok.mode == .sentencePiece { text = String(text.drop(while: { $0 == " " })) }
            if !text.isEmpty { atStart = false; emit(.text(text)) }
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

extension InferenceEngine {
    /// Refuses to load a model whose weights plus KV cache would not fit in GPU-addressable memory.
    static func checkMemory(directory: URL, contextLength: Int) throws {
        let fm = FileManager.default
        var weights = 0
        for f in (try? fm.contentsOfDirectory(atPath: directory.path)) ?? [] where f.hasSuffix(".safetensors") {
            let p = directory.appendingPathComponent(f).resolvingSymlinksInPath().path
            weights += (try? fm.attributesOfItem(atPath: p)[.size] as? Int) ?? 0
        }
        let cfg = try ModelConfig(directory: directory)
        let ctx = min(contextLength, max(cfg.maxPositionEmbeddings, 256))
        let kv = cfg.numLayers * 2 * cfg.numKVHeads * cfg.headDim * 2 * ctx
        let need = UInt64(weights + kv)
        let budget = GPU.shared.device.recommendedMaxWorkingSetSize
        guard need < budget else {
            let f = ByteCountFormatter()
            throw EngineError.unsupported("needs \(f.string(fromByteCount: Int64(need))) (weights + \(ctx)-token KV cache) but the GPU can use about \(f.string(fromByteCount: Int64(budget))). Try a smaller quantization or a shorter context length.")
        }
    }
}

/// Cancels one generation request. Thread-safe.
public final class CancelToken: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    public init() {}
    public func cancel() { lock.lock(); flag = true; lock.unlock() }
    public var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return flag }
}
