import CryptoKit
import Foundation

// MARK: - JSON that keeps key order

/// A JSON value that keeps object key order. Tool definitions are rendered into prompts the
/// way the models' Jinja templates do (`tojson`, i.e. Python's `json.dumps`), which keeps the
/// client's key order; Foundation's JSONSerialization does not.
public indirect enum JSONValue: Equatable, Sendable {
    case null
    case bool(Bool)
    /// The number's source text (validated JSON number syntax).
    case number(String)
    case string(String)
    case array([JSONValue])
    case object([(String, JSONValue)])

    public static func == (a: JSONValue, b: JSONValue) -> Bool {
        switch (a, b) {
        case (.null, .null): return true
        case let (.bool(x), .bool(y)): return x == y
        case let (.number(x), .number(y)): return x == y
        case let (.string(x), .string(y)): return x == y
        case let (.array(x), .array(y)): return x == y
        case let (.object(x), .object(y)):
            return x.count == y.count && zip(x, y).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        default: return false
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let kv) = self { return kv.first(where: { $0.0 == key })?.1 }
        return nil
    }

    public var stringValue: String? { if case .string(let s) = self { return s }; return nil }
    public var isObject: Bool { if case .object = self { return true }; return false }

    /// The object with `key` removed (no-op for other values).
    func removing(_ key: String) -> JSONValue {
        if case .object(let kv) = self { return .object(kv.filter { $0.0 != key }) }
        return self
    }

    public enum ParseError: Error, Equatable { case invalid(String), tooDeep }

    /// Parses exactly one JSON value (surrounding whitespace allowed). Nesting beyond `maxDepth`
    /// is rejected rather than recursed into.
    public static func parse(_ text: String, maxDepth: Int = 64) throws -> JSONValue {
        var p = JSONParser(Array(text.utf8), maxDepth: maxDepth)
        let v = try p.value(depth: 0)
        p.skipWhitespace()
        guard p.atEnd else { throw ParseError.invalid("trailing characters at byte \(p.i)") }
        return v
    }

    /// Parses one JSON value at the start of `text` and returns it with the unparsed remainder.
    static func parsePrefix(_ text: Substring, maxDepth: Int = 64) throws -> (JSONValue, Substring) {
        let bytes = Array(text.utf8)
        var p = JSONParser(bytes, maxDepth: maxDepth)
        let v = try p.value(depth: 0)
        let rest = String(decoding: bytes[p.i...], as: UTF8.self)
        return (v, Substring(rest))
    }

    /// Converts a JSONSerialization value. Key order is lost, so keys are sorted.
    static func from(_ any: Any?, depth: Int = 0) throws -> JSONValue {
        guard depth <= 64 else { throw ParseError.tooDeep }
        switch any {
        case nil, is NSNull: return .null
        case let n as NSNumber:
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return .bool(n.boolValue) }
            return .number(CFNumberIsFloatType(n) ? JSONValue.pythonFloat(n.doubleValue) : n.stringValue)
        case let s as String: return .string(s)
        case let a as [Any]: return .array(try a.map { try from($0, depth: depth + 1) })
        case let d as [String: Any]: return .object(try d.keys.sorted().map { ($0, try from(d[$0], depth: depth + 1)) })
        default: throw ParseError.invalid("unsupported value")
        }
    }

    // MARK: Serialization

    public enum Style: Sendable {
        /// `json.dumps(x)`: `", "` and `": "` separators (Jinja `tojson` in HF templates).
        case python
        /// `json.dumps(x, indent: n)`.
        case pythonIndent(Int)
        /// `,` and `:` with no spaces (OpenAI's `arguments` strings).
        case compact
    }

    public func serialized(_ style: Style = .python) -> String {
        var out = ""
        write(&out, style: style, level: 0)
        return out
    }

    private func write(_ out: inout String, style: Style, level: Int) {
        switch self {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .number(let n): out += JSONValue.normalizedNumber(n)
        case .string(let s): JSONValue.writeString(s, to: &out)
        case .array(let a):
            if a.isEmpty { out += "[]"; return }
            out += "["
            for (i, v) in a.enumerated() {
                if i > 0 { out += JSONValue.itemSeparator(style) }
                JSONValue.newline(&out, style: style, level: level + 1)
                v.write(&out, style: style, level: level + 1)
            }
            JSONValue.newline(&out, style: style, level: level)
            out += "]"
        case .object(let kv):
            if kv.isEmpty { out += "{}"; return }
            out += "{"
            for (i, (k, v)) in kv.enumerated() {
                if i > 0 { out += JSONValue.itemSeparator(style) }
                JSONValue.newline(&out, style: style, level: level + 1)
                JSONValue.writeString(k, to: &out)
                out += { if case .compact = style { return ":" }; return ": " }()
                v.write(&out, style: style, level: level + 1)
            }
            JSONValue.newline(&out, style: style, level: level)
            out += "}"
        }
    }

    private static func itemSeparator(_ style: Style) -> String {
        switch style {
        case .python: return ", "
        case .pythonIndent, .compact: return ","
        }
    }

    private static func newline(_ out: inout String, style: Style, level: Int) {
        if case .pythonIndent(let n) = style { out += "\n" + String(repeating: " ", count: n * level) }
    }

    /// Python's `json.dumps(ensure_ascii=False)` string escaping.
    static func writeString(_ s: String, to out: inout String) {
        out += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if u.value < 0x20 { out += String(format: "\\u%04x", u.value) } else { out.unicodeScalars.append(u) }
            }
        }
        out += "\""
    }

    /// Integers keep their digits; other numbers print as Python's `repr(float)` would.
    static func normalizedNumber(_ n: String) -> String {
        if !n.contains(where: { $0 == "." || $0 == "e" || $0 == "E" }) { return n }
        guard let d = Double(n), d.isFinite else { return n }
        return pythonFloat(d)
    }

    /// Shortest round-trip text, as Python's `repr(float)` prints it (`18.5`, `1e-05`, `1e+16`).
    static func pythonFloat(_ d: Double) -> String { "\(d)" }
}

private struct JSONParser {
    let b: [UInt8]
    var i = 0
    let maxDepth: Int
    init(_ b: [UInt8], maxDepth: Int) { self.b = b; self.maxDepth = maxDepth }

    var atEnd: Bool { i >= b.count }

    mutating func skipWhitespace() {
        while i < b.count, b[i] == 0x20 || b[i] == 0x0A || b[i] == 0x0D || b[i] == 0x09 { i += 1 }
    }

    func fail(_ m: String) -> JSONValue.ParseError { .invalid("\(m) at byte \(i)") }

    mutating func value(depth: Int) throws -> JSONValue {
        guard depth <= maxDepth else { throw JSONValue.ParseError.tooDeep }
        skipWhitespace()
        guard i < b.count else { throw fail("unexpected end") }
        switch b[i] {
        case UInt8(ascii: "{"):
            i += 1
            var kv: [(String, JSONValue)] = []
            skipWhitespace()
            if i < b.count, b[i] == UInt8(ascii: "}") { i += 1; return .object(kv) }
            while true {
                skipWhitespace()
                guard i < b.count, b[i] == UInt8(ascii: "\"") else { throw fail("expected a key") }
                let k = try string()
                skipWhitespace()
                guard i < b.count, b[i] == UInt8(ascii: ":") else { throw fail("expected ':'") }
                i += 1
                kv.append((k, try value(depth: depth + 1)))
                skipWhitespace()
                guard i < b.count else { throw fail("unexpected end") }
                if b[i] == UInt8(ascii: ",") { i += 1; continue }
                if b[i] == UInt8(ascii: "}") { i += 1; return .object(kv) }
                throw fail("expected ',' or '}'")
            }
        case UInt8(ascii: "["):
            i += 1
            var a: [JSONValue] = []
            skipWhitespace()
            if i < b.count, b[i] == UInt8(ascii: "]") { i += 1; return .array(a) }
            while true {
                a.append(try value(depth: depth + 1))
                skipWhitespace()
                guard i < b.count else { throw fail("unexpected end") }
                if b[i] == UInt8(ascii: ",") { i += 1; continue }
                if b[i] == UInt8(ascii: "]") { i += 1; return .array(a) }
                throw fail("expected ',' or ']'")
            }
        case UInt8(ascii: "\""): return .string(try string())
        case UInt8(ascii: "t"): try literal("true"); return .bool(true)
        case UInt8(ascii: "f"): try literal("false"); return .bool(false)
        case UInt8(ascii: "n"): try literal("null"); return .null
        default: return .number(try number())
        }
    }

    mutating func literal(_ s: String) throws {
        let u = Array(s.utf8)
        guard i + u.count <= b.count, Array(b[i..<i + u.count]) == u else { throw fail("invalid literal") }
        i += u.count
    }

    mutating func number() throws -> String {
        let start = i
        func digits() -> Int { let s = i; while i < b.count, b[i] >= 0x30, b[i] <= 0x39 { i += 1 }; return i - s }
        if i < b.count, b[i] == UInt8(ascii: "-") { i += 1 }
        guard i < b.count else { throw fail("invalid number") }
        if b[i] == UInt8(ascii: "0") { i += 1 } else { guard digits() > 0 else { throw fail("invalid number") } }
        if i < b.count, b[i] == UInt8(ascii: ".") { i += 1; guard digits() > 0 else { throw fail("invalid number") } }
        if i < b.count, b[i] == UInt8(ascii: "e") || b[i] == UInt8(ascii: "E") {
            i += 1
            if i < b.count, b[i] == UInt8(ascii: "+") || b[i] == UInt8(ascii: "-") { i += 1 }
            guard digits() > 0 else { throw fail("invalid number") }
        }
        return String(decoding: b[start..<i], as: UTF8.self)
    }

    mutating func hex4() throws -> UInt32 {
        guard i + 4 <= b.count else { throw fail("bad \\u escape") }
        var v: UInt32 = 0
        for _ in 0..<4 {
            let c = b[i]
            let d: UInt32
            switch c {
            case 0x30...0x39: d = UInt32(c - 0x30)
            case 0x41...0x46: d = UInt32(c - 0x41 + 10)
            case 0x61...0x66: d = UInt32(c - 0x61 + 10)
            default: throw fail("bad \\u escape")
            }
            v = v * 16 + d
            i += 1
        }
        return v
    }

    mutating func string() throws -> String {
        i += 1  // opening quote
        var bytes: [UInt8] = []
        while true {
            guard i < b.count else { throw fail("unterminated string") }
            let c = b[i]
            if c == UInt8(ascii: "\"") { i += 1; break }
            if c < 0x20 { throw fail("control character in string") }
            if c != UInt8(ascii: "\\") { bytes.append(c); i += 1; continue }
            i += 1
            guard i < b.count else { throw fail("unterminated escape") }
            let e = b[i]
            i += 1
            switch e {
            case UInt8(ascii: "\""): bytes.append(0x22)
            case UInt8(ascii: "\\"): bytes.append(0x5C)
            case UInt8(ascii: "/"): bytes.append(0x2F)
            case UInt8(ascii: "b"): bytes.append(0x08)
            case UInt8(ascii: "f"): bytes.append(0x0C)
            case UInt8(ascii: "n"): bytes.append(0x0A)
            case UInt8(ascii: "r"): bytes.append(0x0D)
            case UInt8(ascii: "t"): bytes.append(0x09)
            case UInt8(ascii: "u"):
                var cp = try hex4()
                if (0xD800...0xDBFF).contains(cp), i + 6 <= b.count, b[i] == UInt8(ascii: "\\"), b[i + 1] == UInt8(ascii: "u") {
                    let save = i
                    i += 2
                    let lo = try hex4()
                    if (0xDC00...0xDFFF).contains(lo) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00) } else { i = save }
                }
                let scalar = Unicode.Scalar(cp) ?? "\u{FFFD}"  // lone surrogates become U+FFFD
                bytes += Array(String(scalar).utf8)
            default: throw fail("invalid escape")
            }
        }
        // Raw bytes come from a Swift String, so they are valid UTF-8.
        return String(decoding: bytes, as: UTF8.self)
    }
}

// MARK: - Tools

/// A function the model may call. `spec` is the OpenAI Chat Completions tool object
/// (`{"type": "function", "function": {...}}`) in the client's key order; templates render it.
public struct ToolDefinition: Sendable, Equatable {
    public let name: String
    public let spec: JSONValue
    public init(name: String, spec: JSONValue) { self.name = name; self.spec = spec }

    /// OpenAI's function-name rule: `^[a-zA-Z0-9_-]{1,64}$`.
    public static func isValidName(_ s: String) -> Bool {
        guard !s.isEmpty, s.utf8.count <= 64 else { return false }
        return s.utf8.allSatisfy { c in
            (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A) || c == 0x5F || c == 0x2D
        }
    }
}

/// A call the model made (or made earlier, in a conversation's history).
public struct ToolCall: Codable, Hashable, Sendable {
    public var id: String
    public var name: String
    /// The arguments as a JSON object text.
    public var arguments: String
    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// OpenAI-style call id: `call_` and 24 random alphanumerics.
    public static func newID() -> String {
        let chars = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        return "call_" + String((0..<24).map { _ in chars.randomElement()! })
    }

    /// Mistral's tokenizer (v3) requires call ids of exactly 9 alphanumerics. Client ids are
    /// mapped deterministically, so a call and its result keep matching ids.
    static func mistralID(_ id: String) -> String {
        if id.utf8.count == 9, id.utf8.allSatisfy({ ($0 >= 0x30 && $0 <= 0x39) || ($0 >= 0x41 && $0 <= 0x5A) || ($0 >= 0x61 && $0 <= 0x7A) }) {
            return id
        }
        let chars = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        let digest = Array(SHA256.hash(data: Data(id.utf8)))
        return String((0..<9).map { chars[Int(digest[$0]) % chars.count] })
    }

    /// The arguments as a JSON value (an empty object if they are not valid JSON).
    var argumentsValue: JSONValue { (try? JSONValue.parse(arguments)) ?? .object([]) }
}

public enum ToolChoice: Sendable, Equatable {
    case none, auto, required
    case function(String)
}

/// Tools offered to the model for one request.
public struct ToolRequest: Sendable {
    public var tools: [ToolDefinition]
    public var choice: ToolChoice
    /// When false, at most one call is returned.
    public var parallelCalls: Bool
    public init(tools: [ToolDefinition], choice: ToolChoice, parallelCalls: Bool) {
        self.tools = tools
        self.choice = choice
        self.parallelCalls = parallelCalls
    }

    /// Tools are rendered into the prompt (and calls parsed) unless the choice is "none".
    public var active: Bool { !tools.isEmpty && choice != .none }
}

// MARK: - Parsing model output

/// Extracts tool calls from a model's reply in its native format:
/// - ChatML/Hermes (Qwen2.5, Qwen3): `<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>`, repeatable.
/// - Llama 3.1+: a bare JSON object `{"name": …, "parameters": {…}}` (optionally after `<|python_tag|>`).
/// - Mistral v3: `[TOOL_CALLS] [{"name": …, "arguments": {…}, "id": …}]`.
/// Only calls to defined tools are accepted; anything else is left as text.
public enum ToolCallParser {
    public struct Result: Equatable {
        public var content: String
        public var calls: [ToolCall]
    }

    /// A JSON object, tolerating what small models copy from Qwen2.5's prompt example: a doubled
    /// opening brace and stray braces after the object (`{{"name": …}}{}`).
    static func lenientObject(_ s: Substring) -> JSONValue? {
        if let v = try? JSONValue.parse(String(s)) { return v }
        var t = s.drop(while: { $0.isWhitespace })
        if t.hasPrefix("{{") { t = t.dropFirst() }
        guard let (v, rest) = try? JSONValue.parsePrefix(t),
              rest.allSatisfy({ $0.isWhitespace || $0 == "{" || $0 == "}" }) else { return nil }
        return v
    }

    /// Special tokens revealed only so calls can be found; removed again from ordinary text.
    static func revealedMarkers(_ format: ChatFormat) -> [String] {
        switch format {
        case .llama3: return ["<|python_tag|>"]
        case .mistral: return ["[TOOL_CALLS]"]
        case .chatML, .phi3, .plain: return []
        }
    }

    /// The marker that starts a tool call in `format` (Llama has none: its calls are bare JSON).
    static func marker(_ format: ChatFormat) -> String? {
        switch format {
        case .chatML: return "<tool_call>"
        case .mistral: return "[TOOL_CALLS]"
        case .llama3: return "<|python_tag|>"
        case .phi3, .plain: return nil
        }
    }

    /// Parses `text` (a reply, or the part of it from the first tool-call marker on).
    /// Returns nil when it holds no well-formed call to a defined tool.
    public static func parse(_ text: String, format: ChatFormat, tools: [ToolDefinition]) -> Result? {
        let names = Set(tools.map(\.name))
        var calls: [ToolCall] = []
        func add(_ call: JSONValue, argsKeys: [String]) -> Bool {
            // Small models sometimes echo the tool-definition shape: {"type": "function", "function": {...}}.
            let obj = call["name"] == nil && (call["function"]?.isObject ?? false) ? call["function"]! : call
            guard obj.isObject, let name = obj["name"]?.stringValue, names.contains(name) else { return false }
            var args: JSONValue = .object([])
            for k in argsKeys { if let a = obj[k] { args = a; break } }
            // Some models emit the arguments as a JSON string.
            if case .string(let s) = args, let v = try? JSONValue.parse(s) { args = v }
            guard args.isObject else { return false }
            calls.append(ToolCall(id: ToolCall.newID(), name: name, arguments: args.serialized(.compact)))
            return true
        }
        switch format {
        case .chatML:
            let open = "<tool_call>", close = "</tool_call>"
            var content = ""
            var rest = Substring(text)
            while let r = rest.range(of: open) {
                content += rest[..<r.lowerBound]
                let body = rest[r.upperBound...]
                let end = body.range(of: close)
                let json = end.map { body[..<$0.lowerBound] } ?? body
                guard let v = lenientObject(json), add(v, argsKeys: ["arguments", "parameters"]) else { return nil }
                rest = end.map { body[$0.upperBound...] } ?? ""
            }
            content += rest
            guard !calls.isEmpty else { return nil }
            return Result(content: content.trimmingCharacters(in: .whitespacesAndNewlines), calls: calls)
        case .llama3:
            var s = Substring(text.trimmingCharacters(in: .whitespacesAndNewlines))
            if s.hasPrefix("<|python_tag|>") { s = s.dropFirst("<|python_tag|>".count) }
            // One object, or several separated by ';' or newlines (Llama 3.1 style).
            while !s.isEmpty {
                s = s.drop(while: { $0.isWhitespace || $0 == ";" })
                if s.isEmpty { break }
                guard s.first == "{", let (v, rest) = try? JSONValue.parsePrefix(s),
                      add(v, argsKeys: ["parameters", "arguments"]) else { return nil }
                s = rest
            }
            return calls.isEmpty ? nil : Result(content: "", calls: calls)
        case .mistral:
            var s = Substring(text.trimmingCharacters(in: .whitespacesAndNewlines))
            guard s.hasPrefix("[TOOL_CALLS]") else { return nil }
            s = s.dropFirst("[TOOL_CALLS]".count)
            guard let (v, rest) = try? JSONValue.parsePrefix(s) else { return nil }
            let items: [JSONValue]
            if case .array(let a) = v { items = a } else { items = [v] }
            for item in items { guard add(item, argsKeys: ["arguments", "parameters"]) else { return nil } }
            guard !calls.isEmpty else { return nil }
            return Result(content: rest.trimmingCharacters(in: .whitespacesAndNewlines), calls: calls)
        case .phi3, .plain:
            return nil
        }
    }
}

/// Splits a streamed reply into text that can be shown now and a tail that may be a tool call.
/// Text before a call marker streams as usual (minus any partial marker at the end); from the
/// marker on everything is held until the reply ends and is then parsed. Llama 3 calls are bare
/// JSON, so a Llama reply that starts with `{` is held from the start.
struct ToolStreamGate {
    let format: ChatFormat
    let marker: String?
    /// Everything is held (a forced call: the reply continues a prefilled call opener).
    let holdAll: Bool
    private(set) var held = ""
    private var pending = ""   // possible start of a marker
    private var holding: Bool
    private var decided: Bool  // Llama: whether the reply starts with JSON is known

    init(format: ChatFormat, prefill: String?) {
        self.format = format
        marker = ToolCallParser.marker(format)
        holdAll = prefill != nil
        holding = prefill != nil
        held = prefill ?? ""
        decided = format != .llama3 || prefill != nil
    }

    /// Returns the text that can be emitted now.
    mutating func feed(_ s: String) -> String {
        if holding { held += s; return "" }
        var text = pending + s
        pending = ""
        if !decided {
            let trimmed = text.drop(while: { $0.isWhitespace })
            if trimmed.isEmpty { pending = text; return "" }
            decided = true
            if trimmed.first == "{" || trimmed.hasPrefix("<|python_tag|>") {
                holding = true
                held = text
                return ""
            }
            if "<|python_tag|>".hasPrefix(trimmed) { pending = text; decided = false; return "" }
        }
        guard let marker else { return text }
        if let r = text.range(of: marker) {
            holding = true
            held = String(text[r.lowerBound...])
            return String(text[..<r.lowerBound])
        }
        // Keep back a suffix that could be the start of the marker.
        var keep = min(marker.count - 1, text.count)
        while keep > 0 {
            if marker.hasPrefix(text.suffix(keep)) { break }
            keep -= 1
        }
        if keep > 0 {
            pending = String(text.suffix(keep))
            text = String(text.dropLast(keep))
        }
        return text
    }

    /// At the end of the reply: the parsed calls, plus text that is still unemitted.
    mutating func finish(tools: [ToolDefinition]) -> (content: String, calls: [ToolCall], parsed: Bool) {
        let tail = held + pending
        held = ""
        pending = ""
        guard holding, let r = ToolCallParser.parse(tail, format: format, tools: tools) else {
            // Not a call: show the text as it would appear without tools (special tokens hidden).
            let text = ToolCallParser.revealedMarkers(format).reduce(tail) { $0.replacingOccurrences(of: $1, with: "") }
            return (holdAll ? tail : text, [], false)
        }
        return (r.content, r.calls, true)
    }
}
