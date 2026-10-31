import Foundation

/// BPE tokenizer driven by a Hugging Face `tokenizer.json`.
/// Supports byte-level BPE (GPT-2, Llama 3, Qwen, Phi) and
/// SentencePiece-style BPE with byte fallback (Llama 2, Mistral).
public final class Tokenizer: @unchecked Sendable {
    enum Mode { case byteLevel, sentencePiece }

    private(set) var mode: Mode = .byteLevel
    private var vocab: [String: Int] = [:]
    private var idToToken: [String] = []
    private var mergeRanks: [String: Int] = [:]
    private var specialIds: Set<Int> = []
    private var addedTokens: [String: Int] = [:]
    private var addedRegex: NSRegularExpression?
    private var preRegex: NSRegularExpression?
    private var ignoreMerges = false
    private var byteFallback = false
    /// SentencePiece word-boundary prefix: Metaspace `first` (first segment only, if missing),
    /// Metaspace `always` (every segment, if missing), or a `Prepend` normalizer (every segment).
    private var spPrepend = true
    private var spPrependEverySegment = false
    private var spPrependUnconditional = false
    /// Added tokens that swallow adjacent whitespace (`lstrip` / `rstrip` in tokenizer.json).
    private var addedStrip: [String: (left: Bool, right: Bool)] = [:]
    /// The model expects a BOS token at the start of raw prompts.
    public private(set) var addBosToken = false
    /// The normaliser applies Unicode NFC.
    public var usesNFC: Bool { nfc }
    private var nfc = false
    private var unkId: Int?
    private var cache: [String: [Int]] = [:]
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
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any] else {
            throw EngineError.unsupported("tokenizer.json is missing or unreadable")
        }
        guard (model["type"] as? String ?? "BPE") == "BPE", let v = model["vocab"] as? [String: NSNumber] else {
            throw EngineError.unsupported("only BPE tokenizers are supported")
        }
        var maxId = 0
        vocab.reserveCapacity(v.count)
        // Ids index arrays and GPU tables: reject anything out of range (untrusted file).
        let idRange = 0..<(1 << 21)
        for (k, id) in v {
            let i = id.intValue
            guard idRange.contains(i) else { throw EngineError.invalidFile("tokenizer.json: token id \(i) is out of range") }
            vocab[k] = i; maxId = max(maxId, i)
        }

        if let merges = model["merges"] as? [Any] {
            mergeRanks.reserveCapacity(merges.count)
            for (rank, m) in merges.enumerated() {
                if let s = m as? String {
                    if let sp = s.firstIndex(of: " ") {
                        mergeRanks[String(s[..<sp]) + "\u{0}" + String(s[s.index(after: sp)...])] = rank
                    }
                } else if let pair = m as? [String], pair.count == 2 {
                    mergeRanks[pair[0] + "\u{0}" + pair[1]] = rank
                }
            }
        }
        ignoreMerges = model["ignore_merges"] as? Bool ?? false
        byteFallback = model["byte_fallback"] as? Bool ?? false
        if let unk = model["unk_token"] as? String { unkId = vocab[unk] }

        var added: [(String, Int)] = []
        for t in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = t["content"] as? String, let id = (t["id"] as? NSNumber)?.intValue else { continue }
            guard idRange.contains(id) else { throw EngineError.invalidFile("tokenizer.json: added token id \(id) is out of range") }
            vocab[content] = id
            maxId = max(maxId, id)
            addedTokens[content] = id
            let l = t["lstrip"] as? Bool ?? false, r = t["rstrip"] as? Bool ?? false
            if l || r { addedStrip[content] = (l, r) }
            added.append((content, id))
            if t["special"] as? Bool ?? false { specialIds.insert(id) }
        }
        idToToken = Array(repeating: "", count: maxId + 1)
        for (k, i) in vocab { idToToken[i] = k }
        if !added.isEmpty {
            let pattern = added.map(\.0).sorted { $0.count > $1.count }
                .map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")
            addedRegex = try? NSRegularExpression(pattern: pattern)
        }

        // Pre-tokenizer & mode detection.
        var isByteLevel = false
        func visitPre(_ p: [String: Any]) {
            switch p["type"] as? String {
            case "Sequence":
                for sub in (p["pretokenizers"] as? [[String: Any]]) ?? [] { visitPre(sub) }
            case "Split":
                if let pat = p["pattern"] as? [String: Any] {
                    if let r = pat["Regex"] as? String { preRegex = try? NSRegularExpression(pattern: r) }
                    else if let s = pat["String"] as? String { preRegex = try? NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: s)) }
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
            case "Prepend": spPrepend = true; spPrependEverySegment = true; spPrependUnconditional = true
            default: break
            }
        }
        var hasSPNormalizer = false
        if let norm = json["normalizer"] as? [String: Any] {
            visitNorm(norm)
            hasSPNormalizer = String(describing: norm).contains("▁")
        }
        mode = isByteLevel ? .byteLevel : .sentencePiece
        if mode == .sentencePiece && !hasSPNormalizer && json["pre_tokenizer"] == nil { spPrepend = true }

        // tokenizer_config.json: special tokens and chat template.
        if let cdata = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
           let cfg = try? JSONSerialization.jsonObject(with: cdata) as? [String: Any] {
            func tok(_ k: String) -> String? {
                if let s = cfg[k] as? String { return s }
                if let d = cfg[k] as? [String: Any] { return d["content"] as? String }
                return nil
            }
            bosToken = tok("bos_token")
            addBosToken = cfg["add_bos_token"] as? Bool ?? false
            eosToken = tok("eos_token")
            if let t = cfg["chat_template"] as? String { chatTemplate = t }
            else if let arr = cfg["chat_template"] as? [[String: Any]] {
                chatTemplate = arr.first(where: { $0["name"] as? String == "default" })?["template"] as? String
                    ?? arr.first?["template"] as? String
                toolUseTemplate = arr.first(where: { $0["name"] as? String == "tool_use" })?["template"] as? String
            }
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
    }

    public func tokenId(_ token: String) -> Int? { vocab[token] }
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
        var matches = addedRegex?.matches(in: text, range: NSRange(location: 0, length: ns.length)) ?? []
        if !plainRanges.isEmpty {
            matches.removeAll { m in plainRanges.contains { NSIntersectionRange($0, m.range).length > 0 } }
        }
        for m in matches {
            let strip = addedStrip[ns.substring(with: m.range)]
            var segEnd = m.range.location
            if strip?.left == true { while segEnd > cursor && Self.isSpace(ns.character(at: segEnd - 1)) { segEnd -= 1 } }
            if segEnd > cursor {
                encodeOrdinary(ns.substring(with: NSRange(location: cursor, length: segEnd - cursor)),
                               first: firstSegment, into: &out)
            }
            if let id = addedTokens[ns.substring(with: m.range)] { out.append(id) }
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
        if let c = cache[word] { cacheLock.unlock(); return c }
        cacheLock.unlock()
        let r = bpe(word)
        cacheLock.lock()
        if cache.count > 100_000 { cache.removeAll(keepingCapacity: true) }
        cache[word] = r
        cacheLock.unlock()
        return r
    }

    private func bpe(_ word: String) -> [Int] {
        if ignoreMerges, let id = vocab[word] { return [id] }
        var symbols = word.unicodeScalars.map { String($0) }
        if symbols.count > 1 {
            while true {
                var bestRank = Int.max
                var bestPair = ("", "")
                for i in 0..<(symbols.count - 1) {
                    if let r = mergeRanks[symbols[i] + "\u{0}" + symbols[i + 1]], r < bestRank {
                        bestRank = r
                        bestPair = (symbols[i], symbols[i + 1])
                    }
                }
                if bestRank == Int.max { break }
                var merged: [String] = []
                merged.reserveCapacity(symbols.count)
                var i = 0
                while i < symbols.count {
                    if i < symbols.count - 1 && symbols[i] == bestPair.0 && symbols[i + 1] == bestPair.1 {
                        merged.append(bestPair.0 + bestPair.1)
                        i += 2
                    } else {
                        merged.append(symbols[i])
                        i += 1
                    }
                }
                symbols = merged
                if symbols.count == 1 { break }
            }
        }
        var ids: [Int] = []
        ids.reserveCapacity(symbols.count)
        for s in symbols {
            if let id = vocab[s] {
                ids.append(id)
            } else if byteFallback {
                for b in s.utf8 {
                    if let id = vocab[String(format: "<0x%02X>", b)] { ids.append(id) }
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
        if addedTokens[tok] != nil { return Array(tok.utf8) }
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
