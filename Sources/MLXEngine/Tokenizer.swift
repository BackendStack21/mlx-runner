import Foundation

/// A token's exact text. `String` equality is canonical equivalence, but tokenizer.json keys are
/// exact: Gemma has both ";" and U+037E, which would otherwise collide (and which of the two ids
/// won depended on the process's hash seed).
struct ExactKey: Hashable, Sendable {
    let text: String
    init(_ text: String) { self.text = text }
    static func == (a: ExactKey, b: ExactKey) -> Bool { a.text.utf8.elementsEqual(b.text.utf8) }
    func hash(into h: inout Hasher) {
        var s = text
        s.withUTF8 { h.combine(bytes: UnsafeRawBufferPointer($0)) }
    }
}

/// BPE tokenizer driven by a Hugging Face `tokenizer.json`.
/// Supports byte-level BPE (GPT-2, Llama 3, Qwen, Phi) and
/// SentencePiece-style BPE with byte fallback (Llama 2, Mistral).
public final class Tokenizer: @unchecked Sendable {
    enum Mode { case byteLevel, sentencePiece }

    private(set) var mode: Mode = .byteLevel
    private var vocab: [ExactKey: Int] = [:]
    private var idToToken: [String] = []
    private var mergeRanks: [ExactKey: Int] = [:]
    private var specialIds: Set<Int> = []
    private var addedTokens: [ExactKey: Int] = [:]
    /// Leftmost-longest matcher over every added token (a 6,000-way regex alternation made
    /// Gemma 3 encoding 100× slower).
    private var addedMatcher: AddedTokenMatcher?
    private var preRegex: NSRegularExpression?
    private var ignoreMerges = false
    private var byteFallback = false
    /// SentencePiece word-boundary prefix: Metaspace `first` (first segment only, if missing),
    /// Metaspace `always` (every segment, if missing), or a `Prepend` normalizer (every segment).
    private var spPrepend = true
    private var spPrependEverySegment = false
    private var spPrependUnconditional = false
    /// Added tokens that swallow adjacent whitespace (`lstrip` / `rstrip` in tokenizer.json).
    private var addedStrip: [ExactKey: (left: Bool, right: Bool)] = [:]
    /// The model expects a BOS token at the start of raw prompts.
    public private(set) var addBosToken = false
    /// Decoding drops the dummy-prefix space a SentencePiece model puts in front of the first
    /// word (Llama 2, Mistral, Phi). Gemma's tokenizer has no dummy prefix: its leading spaces are real.
    public var stripsLeadingSpace: Bool { mode == .sentencePiece && spPrepend }
    /// The normaliser applies Unicode NFC.
    public var usesNFC: Bool { nfc }
    private var nfc = false
    private var unkId: Int?
    private var cache: [ExactKey: [Int]] = [:]
    private let cacheLock = NSLock()

    private static let byteEncoder: [Character] = {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(Unicode.Scalar(UInt32(c))!) }
        return table
    }()
    private static let byteDecoder: [Character: UInt8] = {
        var d: [Character: UInt8] = [:]
        for (i, c) in byteEncoder.enumerated() { d[c] = UInt8(i) }
        return d
    }()

    public private(set) var bosToken: String?
    public private(set) var eosToken: String?
    public private(set) var chatTemplate: String?
    /// A separate tool-use template, when `chat_template` is a named list (Mistral v0.3).
    public private(set) var toolUseTemplate: String?

    public var vocabularySize: Int { idToToken.count }

    public init(directory: URL) throws {
        guard let raw = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              case let (data, stand) = Self.shieldingBOM(raw),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any] else {
            throw EngineError.unsupported("tokenizer.json is missing or unreadable")
        }
        guard (model["type"] as? String ?? "BPE") == "BPE", let v = model["vocab"] as? NSDictionary else {
            throw EngineError.unsupported("only BPE tokenizers are supported")
        }
        func unshield(_ s: String) -> String {
            guard let stand, s.unicodeScalars.contains(stand) else { return s }
            return String(String.UnicodeScalarView(s.unicodeScalars.map { $0 == stand ? "\u{FEFF}" : $0 }))
        }
        var maxId = 0
        vocab.reserveCapacity(v.count)
        // Ids index arrays and GPU tables: reject anything out of range (untrusted file).
        let idRange = 0..<(1 << 21)
        // Read as an NSDictionary: bridging to [String: _] would itself merge equivalent keys.
        for (key, id) in v {
            guard let k = key as? String, let i = (id as? NSNumber)?.intValue else { throw EngineError.invalidFile("tokenizer.json: malformed vocabulary entry") }
            guard idRange.contains(i) else { throw EngineError.invalidFile("tokenizer.json: token id \(i) is out of range") }
            vocab[ExactKey(unshield(k))] = i; maxId = max(maxId, i)
        }

        if let merges = model["merges"] as? [Any] {
            mergeRanks.reserveCapacity(merges.count)
            for (rank, m) in merges.enumerated() {
                if let s = (m as? String).map(unshield) {
                    if let sp = s.utf8.firstIndex(of: 0x20) {
                        mergeRanks[ExactKey(String(s[..<sp]) + "\u{0}" + String(s[s.utf8.index(after: sp)...]))] = rank
                    }
                } else if let pair = (m as? [String])?.map(unshield), pair.count == 2 {
                    mergeRanks[ExactKey(pair[0] + "\u{0}" + pair[1])] = rank
                }
            }
        }
        ignoreMerges = model["ignore_merges"] as? Bool ?? false
        byteFallback = model["byte_fallback"] as? Bool ?? false
        if let unk = (model["unk_token"] as? String).map(unshield) { unkId = vocab[ExactKey(unk)] }

        var added: [(String, Int)] = []
        for t in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = (t["content"] as? String).map(unshield), let id = (t["id"] as? NSNumber)?.intValue else { continue }
            guard idRange.contains(id) else { throw EngineError.invalidFile("tokenizer.json: added token id \(id) is out of range") }
            vocab[ExactKey(content)] = id
            maxId = max(maxId, id)
            addedTokens[ExactKey(content)] = id
            let l = t["lstrip"] as? Bool ?? false, r = t["rstrip"] as? Bool ?? false
            if l || r { addedStrip[ExactKey(content)] = (l, r) }
            added.append((content, id))
            if t["special"] as? Bool ?? false { specialIds.insert(id) }
        }
        idToToken = Array(repeating: "", count: maxId + 1)
        for (k, i) in vocab { idToToken[i] = k.text }
        if !added.isEmpty { addedMatcher = AddedTokenMatcher(added.map(\.0)) }
        // A TemplateProcessing post-processor that opens with a special token (Llama 3, Gemma 2)
        // adds BOS to raw prompts even when tokenizer_config.json says add_bos_token: false.
        var postBos: String?
        func visitPost(_ p: [String: Any]) {
            switch p["type"] as? String {
            case "Sequence": for sub in (p["processors"] as? [[String: Any]]) ?? [] { visitPost(sub) }
            case "TemplateProcessing":
                if let first = (p["single"] as? [[String: Any]])?.first, let st = first["SpecialToken"] as? [String: Any],
                   let id = (st["id"] as? String).map(unshield) { postBos = id }
            default: break
            }
        }
        if let post = json["post_processor"] as? [String: Any] { visitPost(post) }

        // Pre-tokenizer & mode detection.
        var isByteLevel = false
        var hasPrefixRule = false   // Metaspace or a Prepend normalizer decides the ▁ prefix
        func visitPre(_ p: [String: Any]) {
            switch p["type"] as? String {
            case "Sequence":
                for sub in (p["pretokenizers"] as? [[String: Any]]) ?? [] { visitPre(sub) }
            case "Split":
                if let pat = p["pattern"] as? [String: Any] {
                    if let r = (pat["Regex"] as? String).map(unshield) { preRegex = try? NSRegularExpression(pattern: r) }
                    else if let s = (pat["String"] as? String).map(unshield) { preRegex = try? NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: s)) }
                }
            case "ByteLevel":
                isByteLevel = true
                if (p["use_regex"] as? Bool ?? true) && preRegex == nil {
                    preRegex = try? NSRegularExpression(pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)
                }
            case "Metaspace":
                let scheme = p["prepend_scheme"] as? String ?? ((p["add_prefix_space"] as? Bool ?? true) ? "always" : "never")
                spPrepend = scheme != "never"
                spPrependEverySegment = scheme == "always"
                hasPrefixRule = true
            default: break
            }
        }
        if let pre = json["pre_tokenizer"] as? [String: Any] { visitPre(pre) }
        if let dec = json["decoder"] as? [String: Any] {
            let s = String(describing: dec)
            if s.contains("ByteLevel") { isByteLevel = true }
        }
        func visitNorm(_ n: [String: Any]) {
            switch n["type"] as? String {
            case "Sequence": for sub in (n["normalizers"] as? [[String: Any]]) ?? [] { visitNorm(sub) }
            case "NFC": nfc = true
            case "Prepend": spPrepend = true; spPrependEverySegment = true; spPrependUnconditional = true; hasPrefixRule = true
            default: break
            }
        }
        var hasSPNormalizer = false
        if let norm = json["normalizer"] as? [String: Any] {
            visitNorm(norm)
            hasSPNormalizer = String(describing: norm).contains("▁")
        }
        mode = isByteLevel ? .byteLevel : .sentencePiece
        // A ▁ normalizer without a prefix rule (Gemma: Replace " " → ▁, Split on spaces) adds no
        // dummy prefix.
        if mode == .sentencePiece && hasSPNormalizer && !hasPrefixRule { spPrepend = false }

        // tokenizer_config.json: special tokens and chat template.
        var bosSetting: Bool?   // add_bos_token, when the config states it
        if let cdata = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
           let cfg = try? JSONSerialization.jsonObject(with: cdata) as? [String: Any] {
            func tok(_ k: String) -> String? {
                if let s = cfg[k] as? String { return s }
                if let d = cfg[k] as? [String: Any] { return d["content"] as? String }
                return nil
            }
            bosToken = tok("bos_token")
            bosSetting = cfg["add_bos_token"] as? Bool
            addBosToken = bosSetting ?? false
            eosToken = tok("eos_token")
            if let t = cfg["chat_template"] as? String { chatTemplate = t }
            else if let arr = cfg["chat_template"] as? [[String: Any]] {
                chatTemplate = arr.first(where: { $0["name"] as? String == "default" })?["template"] as? String
                    ?? arr.first?["template"] as? String
                toolUseTemplate = arr.first(where: { $0["name"] as? String == "tool_use" })?["template"] as? String
            }
        }
        if let b = postBos, vocab[ExactKey(b)] != nil {
            addBosToken = true
            if bosToken == nil { bosToken = b }
        }
        if chatTemplate == nil,
           let t = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8) {
            chatTemplate = t
        }
        if chatTemplate == nil,
           let d = try? Data(contentsOf: directory.appendingPathComponent("chat_template.json")),
           let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            chatTemplate = o["chat_template"] as? String
        }
        // A chat template that writes BOS (`{{ bos_token }}`, any whitespace control) shows the model
        // was trained with it first, unless the config says otherwise. Gemma 4's tokenizer files do
        // not say whether to add it to raw prompts,
        // and without it the model produces nonsense.
        if !addBosToken, bosSetting == nil, let b = bosToken, vocab[ExactKey(b)] != nil, let t = chatTemplate, Self.templateWritesBos(t) {
            addBosToken = true
        }
    }

    /// JSONSerialization drops a U+FEFF that starts a string, raw or escaped, so Gemma's vocabulary
    /// entry U+FEFF + "\r" became a second "\r" and took its id: every carriage return got the wrong
    /// token. Every string that starts with U+FEFF (raw, or escaped in any case) gets a private-use
    /// stand-in there instead, one the file does not use, and `stand` maps it back after parsing.
    /// U+FEFF anywhere else (which JSONSerialization keeps) and the file's own BOM are left alone.
    /// One pass over the bytes, tracking whether it is inside a JSON string.
    static func shieldingBOM(_ data: Data) -> (data: Data, stand: Unicode.Scalar?) {
        var starts: [(range: Range<Int>, escaped: Bool)] = []
        var used = [Bool](repeating: false, count: 256)   // U+E000...U+E0FF already in the file
        data.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
            let n = p.count
            func hexDigit(_ b: UInt8) -> Int? {
                switch b {
                case 0x30...0x39: return Int(b - 0x30)
                case 0x41...0x46: return Int(b - 0x41 + 10)
                case 0x61...0x66: return Int(b - 0x61 + 10)
                default: return nil
                }
            }
            var inString = false, atStart = false   // atStart: the next character is the string's first
            var i = 0
            while i < n {
                let b = p[i]
                guard inString else {
                    if b == 0x22 { inString = true; atStart = true }
                    i += 1
                    continue
                }
                if b == 0x22 { inString = false; i += 1; continue }
                if b == 0x5C {   // an escape: \uXXXX, or two bytes (\" \\ \n ...)
                    if i + 5 < n, p[i + 1] == 0x75, let a = hexDigit(p[i + 2]), let c = hexDigit(p[i + 3]),
                       let d = hexDigit(p[i + 4]), let e = hexDigit(p[i + 5]) {
                        let v = a << 12 | c << 8 | d << 4 | e
                        if v == 0xFEFF && atStart { starts.append((i..<(i + 6), true)) }
                        if v >> 8 == 0xE0 { used[v & 0xFF] = true }
                        i += 6
                    } else {
                        i += 2
                    }
                    atStart = false
                    continue
                }
                if b == 0xEF, i + 2 < n, p[i + 1] == 0xBB, p[i + 2] == 0xBF {
                    if atStart { starts.append((i..<(i + 3), false)) }
                    i += 3; atStart = false
                    continue
                }
                if b == 0xEE, i + 2 < n, (0x80...0x83).contains(p[i + 1]) {   // UTF-8 of U+E000...U+E0FF
                    used[Int(p[i + 1] - 0x80) << 6 | Int(p[i + 2] & 0x3F)] = true
                }
                i += 1; atStart = false
            }
        }
        guard !starts.isEmpty, let free = used.firstIndex(of: false),
              let stand = Unicode.Scalar(UInt32(0xE000 + free)) else { return (data, nil) }
        let raw = Data(String(stand).utf8), escaped = Data(String(format: "\\u%04x", stand.value).utf8)
        var out = Data(capacity: data.count)
        var pos = 0
        for s in starts {
            out.append(data[pos..<s.range.lowerBound])
            out.append(s.escaped ? escaped : raw)
            pos = s.range.upperBound
        }
        out.append(data[pos..<data.count])
        return (out, stand)
    }

    /// Does a Jinja chat template output `bos_token` (`{{ bos_token }}`, `{{- bos_token -}}`, ...)?
    static func templateWritesBos(_ template: String) -> Bool {
        template.range(of: #"\{\{-?\s*bos_token\s*-?\}\}"#, options: .regularExpression) != nil
    }

    public func tokenId(_ token: String) -> Int? { vocab[ExactKey(token)] }
    public func token(_ id: Int) -> String? { id >= 0 && id < idToToken.count ? idToToken[id] : nil }
    public func isSpecial(_ id: Int) -> Bool { specialIds.contains(id) }

    // MARK: Encoding

    /// Encodes text; added/special tokens written literally in `text` map to their ids.
    /// `plainRanges` (UTF-16) are treated as ordinary text: special tokens inside them are not parsed.
    public func encode(_ text: String, plainRanges: [NSRange] = []) -> [Int] {
        var text = text
        // NFC can shift UTF-16 offsets, so with ranges the caller normalises content first.
        if nfc && plainRanges.isEmpty { text = text.precomposedStringWithCanonicalMapping }
        var out: [Int] = []
        let ns = text as NSString
        var cursor = 0
        var firstSegment = true
        var matches = addedMatcher?.matches(in: ns) ?? []
        if !plainRanges.isEmpty {
            // Message content must not forge control tokens; non-special added tokens (Gemma's
            // HTML tags, Qwen's <tool_call>) are matched there as the reference tokenizer does.
            matches.removeAll { m in
                specialIds.contains(addedTokens[ExactKey(ns.substring(with: m.range))] ?? -1)
                    && plainRanges.contains { NSIntersectionRange($0, m.range).length > 0 }
            }
        }
        for m in matches {
            let strip = addedStrip[ExactKey(ns.substring(with: m.range))]
            var segEnd = m.range.location
            if strip?.left == true { while segEnd > cursor && Self.isSpace(ns.character(at: segEnd - 1)) { segEnd -= 1 } }
            if segEnd > cursor {
                encodeOrdinary(ns.substring(with: NSRange(location: cursor, length: segEnd - cursor)),
                               first: firstSegment, into: &out)
            }
            if let id = addedTokens[ExactKey(ns.substring(with: m.range))] { out.append(id) }
            cursor = m.range.location + m.range.length
            if strip?.right == true { while cursor < ns.length && Self.isSpace(ns.character(at: cursor)) { cursor += 1 } }
            firstSegment = false
        }
        if cursor < ns.length {
            encodeOrdinary(ns.substring(from: cursor), first: firstSegment, into: &out)
        }
        return out
    }

    private static func isSpace(_ c: unichar) -> Bool {
        c == 0x20 || c == 0x0A || c == 0x09 || c == 0x0D || c == 0x0B || c == 0x0C
    }

    private func encodeOrdinary(_ text: String, first: Bool, into out: inout [Int]) {
        switch mode {
        case .byteLevel:
            for piece in preTokenize(text) {
                let mapped = String(piece.utf8.map { Self.byteEncoder[Int($0)] })
                out += bpeCached(mapped)
            }
        case .sentencePiece:
            var s = text.replacingOccurrences(of: " ", with: "▁")
            if spPrepend && (first || spPrependEverySegment) && (spPrependUnconditional || !s.hasPrefix("▁")) { s = "▁" + s }
            // Split into words that begin at a ▁ following non-▁ text, keeping runs of ▁ attached.
            var words: [String] = []
            var cur = ""
            var prevWasSpace = false
            for ch in s.unicodeScalars {
                let isSpace = ch == "▁"
                if isSpace && !prevWasSpace && !cur.isEmpty { words.append(cur); cur = "" }
                cur.unicodeScalars.append(ch)
                prevWasSpace = isSpace
            }
            if !cur.isEmpty { words.append(cur) }
            for w in words { out += bpeCached(w) }
        }
    }

    private func preTokenize(_ text: String) -> [String] {
        guard let re = preRegex else { return [text] }
        let ns = text as NSString
        var pieces: [String] = []
        var cursor = 0
        for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            if m.range.location > cursor { pieces.append(ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))) }
            if m.range.length > 0 { pieces.append(ns.substring(with: m.range)) }
            cursor = m.range.location + m.range.length
        }
        if cursor < ns.length { pieces.append(ns.substring(from: cursor)) }
        return pieces
    }

    private func bpeCached(_ word: String) -> [Int] {
        cacheLock.lock()
        if let c = cache[ExactKey(word)] { cacheLock.unlock(); return c }
        cacheLock.unlock()
        let r = bpe(word)
        cacheLock.lock()
        if cache.count > 100_000 { cache.removeAll(keepingCapacity: true) }
        cache[ExactKey(word)] = r
        cacheLock.unlock()
        return r
    }

    /// BPE merges in Hugging Face's order: one merge at a time, always the adjacent pair with the
    /// lowest rank, leftmost first, so a pair the merge creates competes at once. (Merging every
    /// occurrence of the best pair in one pass, as GPT-2's reference code does, agrees only when the
    /// merge list is in training order. Gemma's is not: its whitespace-run merges rank early, and
    /// 32 spaces came out as 16 + 16 instead of 31 + one that joins the next word.) A heap keyed by
    /// (rank, position) keeps it O(n log n) for long words.
    static func mergePairs(_ start: [String], rank: (String, String) -> Int?) -> [String] {
        var symbols = start
        let n = symbols.count
        guard n > 1 else { return symbols }
        var next = Array(1...n), prev = Array(-1..<(n - 1)), version = [Int](repeating: 0, count: n)
        var alive = [Bool](repeating: true, count: n)
        struct Candidate { let rank: Int, pos: Int, right: Int, leftVersion: Int, rightVersion: Int }
        var heap: [Candidate] = []
        func before(_ a: Candidate, _ b: Candidate) -> Bool { a.rank != b.rank ? a.rank < b.rank : a.pos < b.pos }
        func push(_ c: Candidate) {
            heap.append(c)
            var i = heap.count - 1
            while i > 0 {
                let p = (i - 1) / 2
                guard before(heap[i], heap[p]) else { break }
                heap.swapAt(i, p); i = p
            }
        }
        func pop() -> Candidate {
            let top = heap[0]
            heap[0] = heap[heap.count - 1]
            heap.removeLast()
            var i = 0
            while true {
                let l = 2 * i + 1, r = l + 1
                var m = i
                if l < heap.count, before(heap[l], heap[m]) { m = l }
                if r < heap.count, before(heap[r], heap[m]) { m = r }
                if m == i { break }
                heap.swapAt(i, m); i = m
            }
            return top
        }
        func consider(_ i: Int) {   // the pair (i, next[i])
            let j = next[i]
            guard i >= 0, j < n, let r = rank(symbols[i], symbols[j]) else { return }
            push(Candidate(rank: r, pos: i, right: j, leftVersion: version[i], rightVersion: version[j]))
        }
        for i in 0..<(n - 1) { consider(i) }
        while !heap.isEmpty {
            let c = pop()
            // Stale: either side was merged into something else since this pair was queued.
            guard alive[c.pos], alive[c.right], next[c.pos] == c.right,
                  version[c.pos] == c.leftVersion, version[c.right] == c.rightVersion else { continue }
            symbols[c.pos] += symbols[c.right]
            version[c.pos] += 1
            alive[c.right] = false
            next[c.pos] = next[c.right]
            if next[c.pos] < n { prev[next[c.pos]] = c.pos }
            if prev[c.pos] >= 0 { consider(prev[c.pos]) }
            consider(c.pos)
        }
        return symbols.indices.filter { alive[$0] }.map { symbols[$0] }
    }

    private func bpe(_ word: String) -> [Int] {
        if ignoreMerges, let id = vocab[ExactKey(word)] { return [id] }
        let symbols = Self.mergePairs(word.unicodeScalars.map { String($0) }) { a, b in mergeRanks[ExactKey(a + "\u{0}" + b)] }
        var ids: [Int] = []
        ids.reserveCapacity(symbols.count)
        for s in symbols {
            if let id = vocab[ExactKey(s)] {
                ids.append(id)
            } else if byteFallback {
                for b in s.utf8 {
                    if let id = vocab[ExactKey(String(format: "<0x%02X>", b))] { ids.append(id) }
                }
            } else if let u = unkId {
                ids.append(u)
            }
        }
        return ids
    }

    // MARK: Decoding

    /// Raw bytes for a token (special tokens decode to nothing unless `includeSpecial`).
    public func bytes(for id: Int, includeSpecial: Bool = false) -> [UInt8] {
        guard let tok = token(id) else { return [] }
        if specialIds.contains(id) { return includeSpecial ? Array(tok.utf8) : [] }
        // Non-special added tokens still go through the decoder in SentencePiece mode (Gemma
        // lists runs of ▁ as added tokens).
        if addedTokens[ExactKey(tok)] != nil, mode == .byteLevel { return Array(tok.utf8) }
        switch mode {
        case .byteLevel:
            var out: [UInt8] = []
            out.reserveCapacity(tok.count)
            for ch in tok {
                if let b = Self.byteDecoder[ch] { out.append(b) } else { out += Array(String(ch).utf8) }
            }
            return out
        case .sentencePiece:
            if tok.count == 6, tok.hasPrefix("<0x"), tok.hasSuffix(">"),
               let b = UInt8(tok.dropFirst(3).dropLast(), radix: 16) {
                return [b]
            }
            return Array(tok.replacingOccurrences(of: "▁", with: " ").utf8)
        }
    }

    public func decode(_ ids: [Int], includeSpecial: Bool = false) -> String {
        var bytes: [UInt8] = []
        for id in ids { bytes += self.bytes(for: id, includeSpecial: includeSpecial) }
        var s = String(decoding: bytes, as: UTF8.self)
        if mode == .sentencePiece && spPrepend && s.hasPrefix(" ") { s.removeFirst() }
        return s
    }
}

/// Incrementally turns token bytes into text, holding back incomplete UTF-8 sequences.
public struct StreamingDecoder {
    private var pending: [UInt8] = []
    public init() {}

    public mutating func append(_ bytes: [UInt8]) -> String {
        pending += bytes
        guard !pending.isEmpty else { return "" }
        // Find the longest prefix that ends on a complete UTF-8 sequence.
        var end = pending.count
        var i = pending.count - 1
        var back = 0
        while i >= 0 && back < 4 {
            let b = pending[i]
            if b & 0xC0 == 0x80 { i -= 1; back += 1; continue }
            let need: Int
            if b & 0x80 == 0 { need = 1 } else if b & 0xE0 == 0xC0 { need = 2 } else if b & 0xF0 == 0xE0 { need = 3 } else if b & 0xF8 == 0xF0 { need = 4 } else { need = 1 }
            if pending.count - i < need { end = i }
            break
        }
        let out = String(decoding: pending[0..<end], as: UTF8.self)
        pending.removeFirst(end)
        return out
    }

    public mutating func flush() -> String {
        let out = String(decoding: pending, as: UTF8.self)
        pending.removeAll()
        return out
    }
}

/// Leftmost-longest literal matching of a token set over UTF-16 text (a trie), equivalent to
/// a regex alternation of the tokens sorted longest first.
struct AddedTokenMatcher {
    struct Match { let range: NSRange }
    private var next: [[UInt16: Int32]] = [[:]]
    private var terminal: [Bool] = [false]

    init(_ tokens: [String]) {
        for t in tokens {
            var node = 0
            for u in t.utf16 {
                if let n = next[node][u] { node = Int(n) } else {
                    next.append([:]); terminal.append(false)
                    next[node][u] = Int32(next.count - 1)
                    node = next.count - 1
                }
            }
            terminal[node] = true
        }
    }

    func matches(in s: NSString) -> [Match] {
        var out: [Match] = []
        let n = s.length
        var i = 0
        while i < n {
            var node = 0, j = i, end = -1
            while j < n, let nx = next[node][s.character(at: j)] {
                node = Int(nx); j += 1
                if terminal[node] { end = j }
            }
            if end > i { out.append(Match(range: NSRange(location: i, length: end - i))); i = end } else { i += 1 }
        }
        return out
    }
}
