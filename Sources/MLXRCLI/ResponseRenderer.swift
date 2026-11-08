import Foundation

/// Ollama's word wrap: at `width - 5` the partial word is erased (`ESC[nD ESC[K`), a newline is
/// written and the word printed again. Characters pass through unchanged otherwise.
struct WordWrapper {
    let width: Int
    private var lineLength = 0
    private var wordBuffer = ""
    private var wordWidth = 0

    init(width: Int) { self.width = width }

    /// Terminal columns of a character: 2 for East Asian wide ranges, 1 otherwise.
    private static func columns(_ u: Unicode.Scalar) -> Int {
        switch u.value {
        case 0x1100...0x115F, 0x2E80...0xA4CF, 0xAC00...0xD7A3, 0xF900...0xFAFF, 0xFE30...0xFE6F,
             0xFF00...0xFF60, 0xFFE0...0xFFE6, 0x1F300...0x1F64F, 0x1F900...0x1F9FF, 0x20000...0x3FFFD: return 2
        default: return 1
        }
    }

    mutating func feed(_ text: String) -> String {
        var out = ""
        for u in text.unicodeScalars {
            let ch = String(u)
            let w = Self.columns(u)
            if lineLength + 1 > width - 5 {
                if wordWidth > width - 10 {
                    // A word longer than the line is split where it is (it is already on screen).
                    out += ch
                    wordBuffer = ""; wordWidth = 0; lineLength = 0
                    continue
                }
                if wordWidth > 0 { out += "\u{1B}[\(wordWidth)D" }
                out += "\u{1B}[K\n" + wordBuffer + ch
                lineLength = wordWidth + w
            } else {
                out += ch
                lineLength += w
                if w >= 2 { wordBuffer = ""; wordWidth = 0; continue }
                switch u {
                case " ", "\t": wordBuffer = ""; wordWidth = 0
                case "\n", "\r": lineLength = 0; wordBuffer = ""; wordWidth = 0
                default: wordBuffer += ch; wordWidth += 1
                }
            }
        }
        return out
    }
}

/// Writes a reply to stdout the way Ollama does: reasoning between `Thinking...` and
/// `...done thinking.` + blank line (dim and bold on a terminal), then the answer, word-wrapped when
/// stdout is a terminal and wrapping is on.
final class ResponseRenderer {
    private let console: Console
    private let hideThinking: Bool
    private let plain: Bool
    private let wrapWidth: Int?
    private var wrapper: WordWrapper?
    private var opened = false
    private var closed = false
    private var reasoningEndsInNewline = false

    private static let grey = "\u{1B}[38;5;245m", bold = "\u{1B}[1m", reset = "\u{1B}[0m"

    init(console: Console, hideThinking: Bool, wordWrap: Bool) {
        self.console = console
        self.hideThinking = hideThinking
        plain = !console.stdoutIsTTY
        wrapWidth = wordWrap && console.stdoutIsTTY && console.width >= 10 ? console.width : nil
        wrapper = wrapWidth.map { WordWrapper(width: $0) }
    }

    private func show(_ text: String) {
        if wrapper != nil { console.out(wrapper!.feed(text)) } else { console.out(text) }
    }

    func reasoning(_ text: String) {
        guard !hideThinking, !text.isEmpty else { return }
        if !opened {
            console.out(plain ? "Thinking...\n" : Self.grey + Self.bold + "Thinking...\n" + Self.reset + Self.grey)
            opened = true; closed = false
        }
        reasoningEndsInNewline = text.hasSuffix("\n")
        show(text)
    }

    func content(_ text: String) {
        guard !text.isEmpty else { return }
        if opened && !closed {
            if !reasoningEndsInNewline { console.out("\n") }
            console.out(plain ? "...done thinking.\n\n" : Self.grey + Self.bold + "...done thinking.\n\n" + Self.reset)
            opened = false; closed = true
            wrapper = wrapWidth.map { WordWrapper(width: $0) }
        }
        show(text)
    }
}
