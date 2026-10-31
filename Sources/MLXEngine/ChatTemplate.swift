import Foundation

public struct ChatMessage: Codable, Hashable, Sendable {
    public enum Role: String, Codable, Sendable { case system, user, assistant }
    public var role: Role
    public var content: String
    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// Prompt formats for the supported model families. The format is detected
/// from the model's Jinja `chat_template` instead of executing it.
public enum ChatFormat: String, Sendable {
    case chatML, llama3, mistral, phi3, plain

    public static func detect(template: String?, modelType: String) -> ChatFormat {
        let t = template ?? ""
        if t.contains("<|im_start|>") { return .chatML }
        if t.contains("<|start_header_id|>") { return .llama3 }
        if t.contains("[INST]") { return .mistral }
        if t.contains("<|user|>") && t.contains("<|end|>") { return .phi3 }
        switch modelType {
        case "qwen2", "qwen3": return .chatML
        case "mistral": return .mistral
        case "phi3": return .phi3
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

    public init(template: String?, modelType: String, bosToken: String?) {
        format = ChatFormat.detect(template: template, modelType: modelType)
        supportsThinkingToggle = template?.contains("enable_thinking") ?? false
        bos = bosToken ?? ""
        let qwen = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        defaultSystem = format == .chatML && (template?.contains(qwen) ?? false) ? qwen : nil
        llamaDatePreamble = format == .llama3 && (template?.contains("Cutting Knowledge Date") ?? false)
    }

    static func todayString() -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "dd MMM yyyy"
        return f.string(from: Date())
    }

    /// Renders messages as a prompt ending with the assistant turn opener.
    public func render(_ messages: [ChatMessage], enableThinking: Bool = true) -> String {
        renderMarked(messages, enableThinking: enableThinking).text
    }

    /// Like `render`, plus the UTF-16 ranges holding message content. Those ranges must be
    /// tokenized as plain text so a message cannot inject control tokens (forged turns).
    public func renderMarked(_ messages: [ChatMessage], enableThinking: Bool = true) -> (text: String, content: [NSRange]) {
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
            if let d = defaultSystem, messages.first?.role != .system {
                lit("<|im_start|>system\n\(d)<|im_end|>\n")
            }
            for m in messages {
                lit("<|im_start|>\(m.role.rawValue)\n"); content(m.content); lit("<|im_end|>\n")
            }
            lit("<|im_start|>assistant\n")
            if supportsThinkingToggle && !enableThinking { lit("<think>\n\n</think>\n\n") }
        case .llama3:
            lit("<|begin_of_text|>")
            var rest = messages[...]
            if llamaDatePreamble {
                var system = ""
                if rest.first?.role == .system { system = rest.removeFirst().content.trimmingCharacters(in: .whitespacesAndNewlines) }
                lit("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: \(Self.todayString())\n\n")
                content(system); lit("<|eot_id|>")
            }
            for m in rest {
                lit("<|start_header_id|>\(m.role.rawValue)<|end_header_id|>\n\n")
                content(m.content.trimmingCharacters(in: .whitespacesAndNewlines)); lit("<|eot_id|>")
            }
            lit("<|start_header_id|>assistant<|end_header_id|>\n\n")
        case .mistral:
            lit(bos.isEmpty ? "<s>" : bos)
            let system = messages.first(where: { $0.role == .system })?.content
            let turns = messages.filter { $0.role != .system }
            let lastUser = turns.lastIndex(where: { $0.role == .user })
            for (i, m) in turns.enumerated() {
                if m.role == .user {
                    var c = m.content
                    if let system, i == lastUser { c = system + "\n\n" + c }
                    lit("[INST] "); content(c); lit("[/INST]")  // mistral-common v3: [INST]▁content[/INST]
                } else {
                    lit(" "); content(m.content); lit("</s>")
                }
            }
        case .phi3:
            for m in messages {
                lit("<|\(m.role.rawValue)|>\n"); content(m.content); lit("<|end|>\n")
            }
            lit("<|assistant|>\n")
        case .plain:
            lit(bos)
            for m in messages {
                switch m.role {
                case .system: content(m.content); lit("\n\n")
                case .user: lit("User: "); content(m.content); lit("\n")
                case .assistant: lit("Assistant: "); content(m.content); lit("\n")
                }
            }
            lit("Assistant:")
        }
        return (s, ranges)
    }

    /// Extra stop strings for the format (in addition to EOS ids).
    public var stopTokens: [String] {
        switch format {
        case .chatML: return ["<|im_end|>", "<|endoftext|>"]
        case .llama3: return ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]
        case .mistral: return ["</s>"]
        case .phi3: return ["<|end|>", "<|endoftext|>"]
        case .plain: return []
        }
    }
}
