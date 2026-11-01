import Foundation

public struct ChatMessage: Codable, Hashable, Sendable {
    public enum Role: String, Codable, Sendable { case system, user, assistant, tool }
    public var role: Role
    public var content: String
    /// Calls an assistant message made (OpenAI `tool_calls`).
    public var toolCalls: [ToolCall]?
    /// For `tool` messages: the call this is the result of.
    public var toolCallID: String?
    public init(role: Role, content: String, toolCalls: [ToolCall]? = nil, toolCallID: String? = nil) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }
}

/// Prompt formats for the supported model families. The format is detected
/// from the model's Jinja `chat_template` instead of executing it.
public enum ChatFormat: String, Sendable {
    case chatML, llama3, mistral, phi3, gemma, gemma4, plain

    public static func detect(template: String?, modelType: String) -> ChatFormat {
        let t = template ?? ""
        if t.contains("<|im_start|>") { return .chatML }
        if t.contains("<|start_header_id|>") { return .llama3 }
        if t.contains("[INST]") { return .mistral }
        if t.contains("<|user|>") && t.contains("<|end|>") { return .phi3 }
        if t.contains("<start_of_turn>") { return .gemma }
        if t.contains("<|turn>") { return .gemma4 }
        switch modelType {
        case "qwen2", "qwen3": return .chatML
        case "mistral": return .mistral
        case "phi3": return .phi3
        case "gemma", "gemma2", "gemma3", "gemma3_text": return .gemma
        case "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text": return .gemma4
        case "llama": return t.isEmpty ? .plain : .llama3
        default: return .chatML
        }
    }
}

public struct ChatTemplate: Sendable {
    public let format: ChatFormat
    /// The template supports Qwen3-style `enable_thinking`.
    public let supportsThinkingToggle: Bool
    let bos: String
    /// System prompt the template injects when the conversation has none (Qwen2.5).
    let defaultSystem: String?
    /// Llama 3.1+ templates always open with a system block carrying the knowledge cutoff and date.
    let llamaDatePreamble: Bool
    /// The template defines native tool calling (Hermes `<tool_call>`, Llama 3.1+ `ipython`,
    /// Mistral v3 `[AVAILABLE_TOOLS]`).
    public let supportsTools: Bool
    /// Qwen2.5's template prints its tool-call example with doubled braces (`{{"name": …}}`, a
    /// Jinja quirk the model was trained with); Qwen3's prints single braces.
    let hermesDoubleBraces: Bool
    /// Gemma 4 templates that close an empty thought channel after the model turn opener when
    /// thinking is off (the 12B), so the model answers directly.
    let emptyThoughtPrefill: Bool

    /// `toolTemplate`: a separate tool-use template, when the model ships one (checked for tool support).
    public init(template: String?, modelType: String, bosToken: String?, toolTemplate: String? = nil) {
        format = ChatFormat.detect(template: template, modelType: modelType)
        // Gemma 4's thinking (`<|think|>`, `<|channel>thought…<channel|>`) is not wired up: it runs
        // with thinking off, its template's default.
        supportsThinkingToggle = format != .gemma4 && (template?.contains("enable_thinking") ?? false)
        bos = bosToken ?? ""
        let qwen = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        defaultSystem = format == .chatML && (template?.contains(qwen) ?? false) ? qwen : nil
        llamaDatePreamble = format == .llama3 && (template?.contains("Cutting Knowledge Date") ?? false)
        let t = (template ?? "") + (toolTemplate ?? "")
        hermesDoubleBraces = t.contains("{{\\\"name\\\": <function-name>")
        emptyThoughtPrefill = format == .gemma4 && (template ?? "").contains("'<|channel>thought\\n<channel|>'")
        switch format {
        case .chatML: supportsTools = t.contains("<tool_call>")
        case .llama3: supportsTools = llamaDatePreamble && t.contains("ipython")
        case .mistral: supportsTools = t.contains("[AVAILABLE_TOOLS]")
        case .phi3, .gemma, .gemma4, .plain: supportsTools = false
        }
    }

    /// Special tokens that start a tool call and must reach the output parser.
    public var toolCallMarkerTokens: [String] { ToolCallParser.revealedMarkers(format) }

    /// Text that starts the assistant turn so the reply must be a tool call: any call for
    /// `.required`, a call to the named function for `.function`. nil otherwise.
    public static func toolCallPrefill(_ format: ChatFormat, _ choice: ToolChoice) -> String? {
        switch (format, choice) {
        case (.chatML, .required): return "<tool_call>\n{\"name\": \""
        case (.chatML, .function(let n)): return "<tool_call>\n{\"name\": \"\(n)\", \"arguments\": "
        case (.llama3, .required): return "{\"name\": \""
        case (.llama3, .function(let n)): return "{\"name\": \"\(n)\", \"parameters\": "
        case (.mistral, .required): return "[TOOL_CALLS] [{\"name\": \""
        case (.mistral, .function(let n)): return "[TOOL_CALLS] [{\"name\": \"\(n)\", \"arguments\": "
        default: return nil
        }
    }

    static func todayString() -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "dd MMM yyyy"
        return f.string(from: Date())
    }

    /// Renders messages as a prompt ending with the assistant turn opener.
    public func render(_ messages: [ChatMessage], enableThinking: Bool = true, tools: [ToolDefinition] = [],
                       prefill: String? = nil) -> String {
        renderMarked(messages, enableThinking: enableThinking, tools: tools, prefill: prefill).text
    }

    /// Like `render`, plus the UTF-16 ranges holding message content. Those ranges must be
    /// tokenized as plain text so a message cannot inject control tokens (forged turns).
    /// `tools` are rendered the way the model's own template does; `prefill` (see
    /// `toolCallPrefill`) is appended after the assistant turn opener.
    public func renderMarked(_ messages: [ChatMessage], enableThinking: Bool = true, tools: [ToolDefinition] = [],
                             prefill: String? = nil) -> (text: String, content: [NSRange]) {
        var s = ""
        var len = 0
        var ranges: [NSRange] = []
        func lit(_ x: String) { s += x; len += x.utf16.count }
        func content(_ x: String) {
            let n = x.utf16.count
            if n > 0 { ranges.append(NSRange(location: len, length: n)) }
            s += x; len += n
        }
        switch format {
        case .chatML:
            var rest = messages[...]
            if !tools.isEmpty {
                // Hermes tool block, as in the Qwen2.5 / Qwen3 templates.
                lit("<|im_start|>system\n")
                if rest.first?.role == .system { content(rest.removeFirst().content); lit("\n\n") }
                else if let d = defaultSystem { lit(d + "\n\n") }
                lit("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                    + "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
                for t in tools { lit("\n"); content(t.spec.serialized(.python)) }
                lit("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
                    + "<tool_call></tool_call> XML tags:\n<tool_call>\n"
                    + (hermesDoubleBraces ? "{{\"name\": <function-name>, \"arguments\": <args-json-object>}}"
                                          : "{\"name\": <function-name>, \"arguments\": <args-json-object>}")
                    + "\n</tool_call><|im_end|>\n")
            } else if let d = defaultSystem, rest.first?.role != .system {
                lit("<|im_start|>system\n\(d)<|im_end|>\n")
            }
            let list = Array(rest)
            for (i, m) in list.enumerated() {
                switch m.role {
                case .tool:
                    if i == 0 || list[i - 1].role != .tool { lit("<|im_start|>user") }
                    lit("\n<tool_response>\n"); content(m.content); lit("\n</tool_response>")
                    if i == list.count - 1 || list[i + 1].role != .tool { lit("<|im_end|>\n") }
                case .assistant where !(m.toolCalls ?? []).isEmpty:
                    lit("<|im_start|>assistant\n"); content(m.content)
                    for (j, c) in (m.toolCalls ?? []).enumerated() {
                        if j > 0 || !m.content.isEmpty { lit("\n") }
                        lit("<tool_call>\n{\"name\": \""); content(c.name); lit("\", \"arguments\": ")
                        content(c.argumentsValue.serialized(.python)); lit("}\n</tool_call>")
                    }
                    lit("<|im_end|>\n")
                default:
                    lit("<|im_start|>\(m.role.rawValue)\n"); content(m.content); lit("<|im_end|>\n")
                }
            }
            lit("<|im_start|>assistant\n")
            if supportsThinkingToggle && (!enableThinking || prefill != nil) { lit("<think>\n\n</think>\n\n") }
        case .llama3:
            lit("<|begin_of_text|>")
            var rest = messages[...]
            if llamaDatePreamble {
                var system = ""
                if rest.first?.role == .system { system = rest.removeFirst().content.trimmingCharacters(in: .whitespacesAndNewlines) }
                lit("<|start_header_id|>system<|end_header_id|>\n\n")
                if !tools.isEmpty { lit("Environment: ipython\n") }
                lit("Cutting Knowledge Date: December 2023\nToday Date: \(Self.todayString())\n\n")
                content(system); lit("<|eot_id|>")
                // Llama 3.1+ puts custom tools in the first user message (tools_in_user_message).
                if !tools.isEmpty, rest.first?.role == .user {
                    let first = rest.removeFirst()
                    lit("<|start_header_id|>user<|end_header_id|>\n\n")
                    lit("Given the following functions, please respond with a JSON for a function call "
                        + "with its proper arguments that best answers the given prompt.\n\n"
                        + "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}."
                        + "Do not use variables.\n\n")
                    for t in tools { content(t.spec.serialized(.pythonIndent(4))); lit("\n\n") }
                    content(first.content.trimmingCharacters(in: .whitespacesAndNewlines)); lit("<|eot_id|>")
                }
            }
            for m in rest {
                if m.role == .assistant, let c = m.toolCalls?.first {
                    // The template allows one call per assistant message.
                    lit("<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \""); content(c.name)
                    lit("\", \"parameters\": "); content(c.argumentsValue.serialized(.python)); lit("}<|eot_id|>")
                } else if m.role == .tool {
                    // The template JSON-encodes tool results (a string is "iterable" to Jinja).
                    var quoted = ""
                    JSONValue.writeString(m.content, to: &quoted)
                    lit("<|start_header_id|>ipython<|end_header_id|>\n\n"); content(quoted); lit("<|eot_id|>")
                } else {
                    lit("<|start_header_id|>\(m.role.rawValue)<|end_header_id|>\n\n")
                    content(m.content.trimmingCharacters(in: .whitespacesAndNewlines)); lit("<|eot_id|>")
                }
            }
            lit("<|start_header_id|>assistant<|end_header_id|>\n\n")
        case .mistral:
            lit(bos.isEmpty ? "<s>" : bos)
            let system = messages.first(where: { $0.role == .system })?.content
            let turns = messages.filter { $0.role != .system }
            let lastUser = turns.lastIndex(where: { $0.role == .user })
            for (i, m) in turns.enumerated() {
                switch m.role {
                case .user:
                    var c = m.content
                    if let system, i == lastUser { c = system + "\n\n" + c }
                    if !tools.isEmpty, i == lastUser {
                        // mistral-common v3: [AVAILABLE_TOOLS]▁[...][/AVAILABLE_TOOLS] before the last [INST].
                        lit("[AVAILABLE_TOOLS] "); content(JSONValue.array(tools.map(\.spec)).serialized(.python)); lit("[/AVAILABLE_TOOLS]")
                    }
                    lit("[INST] "); content(c); lit("[/INST]")  // mistral-common v3: [INST]▁content[/INST]
                case .assistant where !(m.toolCalls ?? []).isEmpty:
                    let calls = JSONValue.array((m.toolCalls ?? []).map {
                        .object([("name", .string($0.name)), ("arguments", $0.argumentsValue), ("id", .string(ToolCall.mistralID($0.id)))])
                    })
                    lit("[TOOL_CALLS] "); content(calls.serialized(.python)); lit("</s>")
                case .tool:
                    // mistral-common sends JSON results as values, anything else as a string.
                    let value = (try? JSONValue.parse(m.content)) ?? .string(m.content)
                    let result = JSONValue.object([("content", value), ("call_id", .string(ToolCall.mistralID(m.toolCallID ?? "")))])
                    lit("[TOOL_RESULTS] "); content(result.serialized(.python)); lit("[/TOOL_RESULTS]")
                default:
                    lit(" "); content(m.content); lit("</s>")
                }
            }
        case .phi3:
            for m in messages {
                lit("<|\(m.role.rawValue)|>\n"); content(m.content); lit("<|end|>\n")
            }
            lit("<|assistant|>\n")
        case .gemma:
            // Gemma's templates fold a system prompt into the first turn, call the assistant
            // "model" and trim message content. There is no tool role: results go in as user turns.
            lit(bos.isEmpty ? "<bos>" : bos)
            var rest = messages[...]
            var prefix: String?
            if rest.first?.role == .system { prefix = rest.removeFirst().content + "\n\n" }
            for m in rest {
                let role = m.role == .assistant ? "model" : "user"
                lit("<start_of_turn>\(role)\n")
                if let p = prefix { content(p); prefix = nil }
                content(m.content.trimmingCharacters(in: .whitespacesAndNewlines)); lit("<end_of_turn>\n")
            }
            if let p = prefix { lit("<start_of_turn>user\n"); content(p); lit("<end_of_turn>\n") }
            lit("<start_of_turn>model\n")
        case .gemma4:
            // `<|turn>role … <turn|>`; a system prompt gets its own (trimmed) turn; consecutive
            // assistant messages share one model turn. No tools: results go in as user turns.
            lit(bos.isEmpty ? "<bos>" : bos)
            var rest = messages[...]
            if rest.first?.role == .system {
                lit("<|turn>system\n"); content(rest.removeFirst().content.trimmingCharacters(in: .whitespacesAndNewlines)); lit("<turn|>\n")
            }
            let list = Array(rest)
            for (i, m) in list.enumerated() {
                let model = m.role == .assistant
                let role = model ? "model" : m.role == .system ? "system" : "user"
                if !(model && i > 0 && list[i - 1].role == .assistant) { lit("<|turn>\(role)\n") }
                content(m.content.trimmingCharacters(in: .whitespacesAndNewlines))
                if !(model && i + 1 < list.count && list[i + 1].role == .assistant) { lit("<turn|>\n") }
            }
            lit("<|turn>model\n")
            if emptyThoughtPrefill { lit("<|channel>thought\n<channel|>") }
        case .plain:
            lit(bos)
            for m in messages {
                switch m.role {
                case .system: content(m.content); lit("\n\n")
                case .user: lit("User: "); content(m.content); lit("\n")
                case .assistant: lit("Assistant: "); content(m.content); lit("\n")
                case .tool: lit("Tool: "); content(m.content); lit("\n")
                }
            }
            lit("Assistant:")
        }
        // The prefill holds only markers and a validated function name.
        if let prefill { lit(prefill) }
        return (s, ranges)
    }

    /// Extra stop strings for the format (in addition to EOS ids).
    public var stopTokens: [String] {
        switch format {
        case .chatML: return ["<|im_end|>", "<|endoftext|>"]
        case .llama3: return ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]
        case .mistral: return ["</s>"]
        case .phi3: return ["<|end|>", "<|endoftext|>"]
        case .gemma: return ["<end_of_turn>", "<eos>"]
        case .gemma4: return ["<turn|>", "<eos>"]
        case .plain: return []
        }
    }
}
