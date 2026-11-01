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
    /// Speculative decoding with prompt-lookup drafts (nil = off). Output is unchanged; it pays
    /// off only once a verify pass costs well under k decode steps (see README).
    public var speculative: Bool? = nil
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
    /// Speculative decoding: drafted tokens and how many the model accepted.
    public var draftTokens: Int? = nil
    public var acceptedDraftTokens: Int? = nil
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
    /// The chat template defines native tool calling.
    public var supportsTools: Bool = false
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

    /// The final sampling distribution after the repetition penalty, temperature and
    /// top-k / top-p / min-p filters (descending probability). Greedy decoding gives a
    /// single candidate. Returns nil for non-finite logits.
    mutating func distribution(_ logits: UnsafeMutablePointer<Float>, count: Int, recent: ArraySlice<Int>) -> [(id: Int, p: Float)]? {
        if params.repetitionPenalty != 1, !recent.isEmpty {
            let p = params.repetitionPenalty
            for t in Set(recent) where t >= 0 && t < count {
                logits[t] = logits[t] > 0 ? logits[t] / p : logits[t] * p
            }
        }
        if params.temperature <= 0.001 {
            var best = 0
            var bestV = -Float.infinity
            for i in 0..<count where logits[i] > bestV { bestV = logits[i]; best = i }
            return bestV.isFinite ? [(id: best, p: Float(1))] : nil
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
        guard maxV.isFinite else { return nil }  // NaN or infinite logits
        var probs = cand.map { exp(($0.1 - maxV) * invT) }
        var total = probs.reduce(0, +)
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
        cut = max(1, cut)
        total = probs[0..<cut].reduce(0, +)
        return (0..<cut).map { (cand[$0].0, probs[$0] / total) }
    }

    mutating func draw(_ dist: [(id: Int, p: Float)]) -> Int {
        var r = Float.random(in: 0..<1, using: &rng)
        for c in dist {
            r -= c.p
            if r <= 0 { return c.id }
        }
        return dist[dist.count - 1].id
    }

    mutating func sample(_ logits: UnsafeMutablePointer<Float>, count: Int, recent: ArraySlice<Int>) -> Int {
        guard let d = distribution(logits, count: count, recent: recent) else { return -1 }
        return draw(d)
    }

    /// Exact speculative acceptance for a deterministic draft: accept `draft` with
    /// probability p(draft), otherwise draw from p with the draft removed. The output
    /// distribution is identical to sampling from p directly (and to argmax when greedy).
    mutating func verify(_ dist: [(id: Int, p: Float)], draft: Int) -> (accepted: Bool, token: Int) {
        let pd = dist.first(where: { $0.id == draft })?.p ?? 0
        if pd >= 1 || (pd > 0 && Float.random(in: 0..<1, using: &rng) < pd) { return (true, draft) }
        let rest = dist.filter { $0.id != draft }
        let total = rest.reduce(Float(0)) { $0 + $1.p }
        guard total > 0 else { return (true, draft) }
        return (false, draw(rest.map { ($0.id, $0.p / total) }))
    }
}

/// Prompt-lookup drafting: if the last few generated tokens already occurred in the
/// context, propose the tokens that followed them. Free to compute, and very effective
/// when the answer quotes or edits the prompt (code, documents, RAG).
enum NGramDraft {
    static func propose(_ history: [Int], maxTokens: Int, maxNGram: Int = 3, minNGram: Int = 2, searchWindow: Int = 8192) -> [Int] {
        guard maxTokens > 0 else { return [] }
        let n = history.count
        let lo = max(0, n - searchWindow)
        for g in stride(from: maxNGram, through: minNGram, by: -1) where n > g {
            let tail = n - g
            var i = tail - 1
            while i >= lo {
                var match = true
                for j in 0..<g where history[i + j] != history[tail + j] { match = false; break }
                if match {
                    let start = i + g
                    let end = min(start + maxTokens, n)
                    if end > start { return Array(history[start..<end]) }
                }
                i -= 1
            }
        }
        return []
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
    /// Sticky trim cut per conversation (key: caller-supplied id, else the first turn).
    private var trimState: [String: Int] = [:]
    /// Decoded text of every token, indexed by first character (built on first use per model;
    /// used to constrain forced tool-call names).
    private var tokenTexts: (texts: [String], byFirst: [Character: [Int]])?
    private let infoLock = NSLock()
    private var _info: LoadedModelInfo?
    /// The loaded model; safe to read from any thread.
    public private(set) var info: LoadedModelInfo? {
        get { infoLock.lock(); defer { infoLock.unlock() }; return _info }
        set { infoLock.lock(); _info = newValue; infoLock.unlock() }
    }

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
                    self.trimState = [:]
                    self.tokenTexts = nil
                    let tok = try Tokenizer(directory: directory)
                    try Self.checkMemory(directory: directory, contextLength: contextLength)
                    let model = try Transformer(directory: directory, contextLength: contextLength)
                    let tpl = ChatTemplate(template: tok.chatTemplate, modelType: model.config.modelType, bosToken: tok.bosToken,
                                           toolTemplate: tok.toolUseTemplate)
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
                        loadSeconds: Date().timeIntervalSince(start), supportsTools: tpl.supportsTools)
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
            self.trimState = [:]
        }
    }

    /// Stops whichever request is currently generating (queued requests are unaffected).
    public func cancel() {
        cancelLock.lock(); let t = currentToken; cancelLock.unlock()
        t?.cancel()
    }

    /// Streams a chat completion. Cancel with `token`, or simply stop consuming the stream.
    /// `conversationID` keeps context trimming stable across turns of the same chat.
    /// With `tools`, the tool definitions are rendered into the prompt (unless the choice is
    /// "none") and the template's tool-call marker tokens appear in the output text. `prefill`
    /// (from `ChatTemplate.toolCallPrefill`) starts the assistant turn.
    /// `constrainToolNames`: the reply continues a prefill that ends inside a function name
    /// (`tool_choice: required`); until the name is complete, only tokens that keep it one of
    /// these names can be sampled.
    public func chat(_ messages: [ChatMessage], parameters: SamplingParameters = .init(), enableThinking: Bool = true,
                     conversationID: String? = nil, tools: ToolRequest? = nil, prefill: String? = nil,
                     constrainToolNames: [String]? = nil,
                     token: CancelToken = CancelToken()) -> AsyncThrowingStream<GenerationEvent, Error> {
        let defs = (tools?.active ?? false) ? tools?.tools ?? [] : []
        return stream(token, parameters: parameters, revealToolMarkers: !defs.isEmpty, nameChoices: constrainToolNames) { tok, tpl in
            guard let tpl else { throw EngineError.invalidFile("No model loaded") }
            return self.fitToContext(messages, template: tpl, tokenizer: tok,
                                     reserve: min(parameters.maxTokens, 1024), enableThinking: enableThinking,
                                     conversationID: conversationID, tools: defs, prefill: prefill)
        }
    }

    /// Streams a raw completion for a prompt string (special tokens are parsed).
    public func complete(_ prompt: String, parameters: SamplingParameters = .init(),
                         token: CancelToken = CancelToken()) -> AsyncThrowingStream<GenerationEvent, Error> {
        stream(token, parameters: parameters) { tok, _ in
            var ids = tok.encode(prompt)
            // Models such as Llama and Mistral expect BOS on raw prompts (add_bos_token).
            if tok.addBosToken, let b = tok.bosToken, let id = tok.tokenId(b), ids.first != id { ids.insert(id, at: 0) }
            return ids
        }
    }

    private func stream(_ token: CancelToken, parameters: SamplingParameters, revealToolMarkers: Bool = false,
                        nameChoices: [String]? = nil,
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
                    let reveal = revealToolMarkers
                        ? Set((self.template?.toolCallMarkerTokens ?? []).compactMap { tok.tokenId($0) }) : []
                    try self.run(tokens: tokens, parameters: parameters, reveal: reveal, nameChoices: nameChoices,
                                 token: token) { cont.yield($0) }
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
                              reserve: Int, enableThinking: Bool, conversationID: String?,
                              tools: [ToolDefinition] = [], prefill: String? = nil) -> [Int] {
        let limit = (model?.maxSeq ?? Int.max) - reserve
        // Content is tokenized as plain text (see renderMarked), normalised up front if needed.
        let msgs = tokenizer.usesNFC ? messages.map {
            var m = $0
            m.content = m.content.precomposedStringWithCanonicalMapping
            return m
        } : messages
        let system = msgs.filter { $0.role == .system }
        let turns = msgs.filter { $0.role != .system }
        guard let first = turns.first else {
            let r = template.renderMarked(msgs, enableThinking: enableThinking, tools: tools, prefill: prefill)
            return tokenizer.encode(r.text, plainRanges: r.content)
        }
        let key = conversationID ?? (first.role.rawValue + "\u{0}" + first.content)
        if trimState.count > 256 { trimState = [:] }

        func render(_ d: Int) -> [Int] {
            let r = template.renderMarked(system + turns[d...], enableThinking: enableThinking, tools: tools, prefill: prefill)
            return tokenizer.encode(r.text, plainRanges: r.content)
        }
        /// Next cut index > d that keeps the history starting with a user turn.
        func advance(_ d: Int) -> Int? {
            var n = d + 1
            while n < turns.count - 1 && turns[n].role != .user { n += 1 }
            return n < turns.count ? n : nil
        }
        // Whole conversation fits: never trim (a stale cut must not drop history).
        var tokens = render(0)
        var dropped = 0
        if tokens.count > limit {
            dropped = min(trimState[key] ?? 0, turns.count - 1)
            if dropped > 0 { tokens = render(dropped) }
        }
        if tokens.count > limit {
            let target = Int(Double(limit) * 0.7)
            while tokens.count > target, let n = advance(dropped), n <= turns.count - 1 {
                dropped = n
                tokens = render(dropped)
                if dropped == turns.count - 1 { break }
            }
        }
        trimState[key] = dropped
        return tokens
    }

    /// `reveal`: special tokens whose text is emitted (tool-call markers); others decode to nothing.
    private func run(tokens: [Int], parameters: SamplingParameters, reveal: Set<Int> = [], nameChoices: [String]? = nil,
                     token: CancelToken, emit: (GenerationEvent) -> Void) throws {
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

        let vocab = model.vocabSize
        let count = min(vocab, tok.vocabularySize)
        let seed = parameters.seed ?? UInt64.random(in: 1...UInt64.max)
        // Constrained name: the text generated so far while a forced tool-call name is open.
        var nameText = ""
        var nameOpen = !(nameChoices ?? []).isEmpty
        let texts = nameOpen ? tokenTextIndex(tok) : nil
        // Speculative decoding (prompt lookup): verifying k drafted tokens costs one batched
        // pass, so every accepted draft is a token generated without its own weight read.
        let speculative = (parameters.speculative ?? false) && model.supportsBatchedVerify && !nameOpen
        // Plain decoding selects tokens on the GPU, so each step is queued before the previous one
        // is read back and the GPU never waits for sampling or detokenisation.
        let gpuSampling = !speculative && !nameOpen
        let gpuParams = SampleParams(parameters, vocab: count)

        cachedTokens = Array(tokens[0..<common])
        let t0 = Date()
        var logits: UnsafeMutablePointer<Float>?   // CPU sampling only
        var firstToken: Int?
        do {
            let prompt = tokens[common...].map { Int32($0) }
            if gpuSampling {
                model.seedSampler(seed)
                firstToken = try model.prefill(prompt, offset: common, sample: gpuParams, isCancelled: { token.isCancelled })
            } else {
                logits = try model.forward(prompt, offset: common, isCancelled: { token.isCancelled })
            }
        } catch EngineError.cancelled {
            stats.stopReason = "cancelled"
            emit(.finished(stats))
            return
        }
        var history = tokens
        cachedTokens = tokens
        let tPrompt = Date().timeIntervalSince(t0)
        stats.timeToFirstToken = tPrompt
        stats.promptTokensPerSecond = Double(tokens.count - common) / max(tPrompt, 1e-6)
        emit(.promptProcessed(stats))

        var sampler = Sampler(params: parameters, rng: XorShift(seed: seed))
        var decoder = StreamingDecoder()
        let tGen = Date()
        stats.stopReason = "length"
        let maxNew = min(parameters.maxTokens, model.maxSeq - tokens.count)
        var atStart = true
        var drafted = 0, acceptedDrafts = 0
        var draftLen = 4

        /// Emits a sampled token. Returns false when generation must stop.
        func accept(_ t: Int) -> Bool {
            if stopIds.contains(t) { stats.stopReason = "stop"; return false }
            stats.generatedTokens += 1
            var text = decoder.append(tok.bytes(for: t, includeSpecial: reveal.contains(t)))
            // SentencePiece models start the reply with a word-boundary space.
            if atStart, tok.mode == .sentencePiece { text = String(text.drop(while: { $0 == " " })) }
            if !text.isEmpty { atStart = false; emit(.text(text)) }
            history.append(t)
            if nameOpen, let texts {
                nameText += texts.texts[t]
                if (nameChoices ?? []).contains(where: { nameText.hasPrefix($0 + "\",") }) { nameOpen = false }
            }
            if stats.generatedTokens >= maxNew { stats.stopReason = "length"; return false }
            if history.count >= model.maxSeq { stats.stopReason = "context"; return false }
            return true
        }
        func dist(_ row: UnsafeMutablePointer<Float>) throws -> [(id: Int, p: Float)] {
            if nameOpen, let texts {
                let allowed = Self.allowedNameTokens(nameChoices ?? [], typed: nameText, texts: texts)
                if allowed.isEmpty { nameOpen = false } else {
                    let keep = allowed.compactMap { $0 < count ? ($0, row[$0]) : nil }
                    for i in 0..<count { row[i] = -.infinity }
                    for (i, v) in keep { row[i] = v }
                }
            }
            guard let d = sampler.distribution(row, count: count, recent: history.suffix(parameters.repetitionWindow)) else {
                throw EngineError.invalidOutput("non-finite logits")
            }
            return d
        }

        var running = maxNew > 0 && !token.isCancelled
        if let first = firstToken {
            // GPU sampling: the step for the newest token is committed before the previous step is
            // read back, so the GPU runs them back to back. A step that turns out to process a stop
            // token only writes a K/V row that `cachedTokens` does not claim.
            if running { running = accept(first) }
            var cachedCount = tokens.count   // positions whose K/V rows the cache holds
            var inflight: Transformer.DecodeStep? = running ? try model.beginDecodeStep(offset: history.count - 1, sample: gpuParams) : nil
            while let cur = inflight {
                inflight = nil
                let nextOffset = cur.offset + 1
                if !token.isCancelled, stats.generatedTokens + 1 < maxNew, nextOffset + 1 < model.maxSeq {
                    do { inflight = try model.beginDecodeStep(offset: nextOffset, sample: gpuParams) }
                    catch { _ = try? model.finish(cur); throw error }
                }
                let t: Int
                do { t = try model.finish(cur) } catch { if let f = inflight { _ = try? model.finish(f) }; throw error }
                cachedCount = cur.offset + 1
                if token.isCancelled { stats.stopReason = "cancelled"; break }
                if !accept(t) { break }
            }
            if let f = inflight, (try? model.finish(f)) != nil { cachedCount = f.offset + 1 }
            cachedTokens = Array(history.prefix(min(history.count, cachedCount)))
            running = false
        }
        // CPU sampling (speculative decoding, constrained tool names). Invariant: `history.last`
        // (the pending token) is not yet in the KV cache; everything before it is, and
        // `cachedTokens` mirrors exactly that.
        if running, let logits { running = accept(sampler.draw(try dist(logits))) }
        while running {
            if token.isCancelled { stats.stopReason = "cancelled"; break }
            let pending = history[history.count - 1]
            let room = min(maxNew - stats.generatedTokens, model.maxSeq - history.count) - 1
            let draft = speculative ? NGramDraft.propose(history, maxTokens: min(draftLen, room, Transformer.maxVerifyTokens - 1)) : []
            if draft.isEmpty {
                let row = try model.forward([Int32(pending)], offset: history.count - 1)
                cachedTokens = history
                running = accept(sampler.draw(try dist(row)))
                continue
            }
            let rows = try model.forward(([pending] + draft).map { Int32($0) }, offset: history.count - 1, allLogits: true)
            cachedTokens = history
            drafted += draft.count
            var accepted = 0
            var corrected = false
            for (i, d) in draft.enumerated() {
                let (ok, t) = sampler.verify(try dist(rows + i * vocab), draft: d)
                running = accept(t)
                if ok { accepted += 1; if running { cachedTokens = history } }
                else { corrected = true }
                if !ok || !running { break }
            }
            acceptedDrafts += accepted
            // Bonus token: every draft matched, so the last row predicts one more for free.
            if running && !corrected { running = accept(sampler.draw(try dist(rows + draft.count * vocab))) }
            draftLen = accepted == draft.count ? min(draftLen + 1, Transformer.maxVerifyTokens - 1) : max(1, accepted)
        }
        let rest = decoder.flush()
        if !rest.isEmpty { emit(.text(rest)) }
        let dt = Date().timeIntervalSince(tGen)
        stats.tokensPerSecond = Double(stats.generatedTokens) / max(dt, 1e-6)
        if drafted > 0 { stats.draftTokens = drafted; stats.acceptedDraftTokens = acceptedDrafts }
        emit(.finished(stats))
    }
}

extension InferenceEngine {
    /// Token texts, built once per loaded model (runs on the inference queue).
    private func tokenTextIndex(_ tok: Tokenizer) -> (texts: [String], byFirst: [Character: [Int]]) {
        if let t = tokenTexts { return t }
        var texts: [String] = []
        texts.reserveCapacity(tok.vocabularySize)
        var byFirst: [Character: [Int]] = [:]
        for id in 0..<tok.vocabularySize {
            let s = String(decoding: tok.bytes(for: id), as: UTF8.self)
            texts.append(s)
            if let c = s.first { byFirst[c, default: []].append(id) }
        }
        tokenTexts = (texts, byFirst)
        return (texts, byFirst)
    }

    /// Tokens that keep `typed` on the way to one of `names`, its closing quote and the comma
    /// that must follow (`"name": "get_weather", "arguments": …` in every supported format).
    /// A token may run past the comma only with whitespace, so the call cannot close early.
    static func allowedNameTokens(_ names: [String], typed: String,
                                  texts: (texts: [String], byFirst: [Character: [Int]])) -> Set<Int> {
        var allowed = Set<Int>()
        for n in names {
            let target = n + "\","
            guard target.hasPrefix(typed) else { continue }
            let rest = target.dropFirst(typed.count)
            guard let c = rest.first else { continue }
            for id in texts.byFirst[c] ?? [] {
                let t = texts.texts[id]
                if rest.hasPrefix(t) || (t.hasPrefix(rest) && t.dropFirst(rest.count).allSatisfy(\.isWhitespace)) { allowed.insert(id) }
            }
        }
        return allowed
    }

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
        let kv = cfg.kvBytesPerToken * ctx
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
