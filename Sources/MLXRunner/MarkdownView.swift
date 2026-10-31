import AppKit
import MLXEngine
import SwiftUI

/// Block-level Markdown renderer: headings, lists, quotes, code fences, tables, rules.
/// Inline styling (bold, italic, code, links) uses Foundation's AttributedString parser.
struct MarkdownView: View {
    let text: String
    var baseSize: CGFloat = 14

    var body: some View {
        let blocks = MarkdownParser.parse(text)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, b in
                view(for: b)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func view(for block: MDBlock) -> some View {
        switch block {
        case .paragraph(let s):
            inline(s).font(.system(size: baseSize)).lineSpacing(3).fixedSize(horizontal: false, vertical: true)
        case .heading(let level, let s):
            inline(s)
                .font(.system(size: baseSize + CGFloat(max(0, 4 - level)) * 3, weight: .bold, design: .rounded))
                .padding(.top, 4)
        case .code(let lang, let code):
            CodeBlock(language: lang, code: code)
        case .list(let items, let ordered):
            VStack(alignment: .leading, spacing: 5) {
                ForEach(Array(items.enumerated()), id: \.offset) { i, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(ordered ? "\(item.number ?? i + 1)." : "•")
                            .font(.system(size: baseSize, weight: .semibold).monospacedDigit())
                            .foregroundStyle(Theme.accent)
                            .frame(minWidth: 14, alignment: .trailing)
                        inline(item.text).font(.system(size: baseSize)).lineSpacing(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.leading, CGFloat(item.indent) * 16)
                }
            }
        case .quote(let s):
            inline(s).font(.system(size: baseSize)).foregroundStyle(.secondary)
                .padding(.leading, 12)
                .overlay(alignment: .leading) { Rectangle().fill(Theme.accent.opacity(0.6)).frame(width: 3) }
        case .rule:
            Divider().padding(.vertical, 4)
        case .table(let header, let rows):
            TableBlock(header: header, rows: rows, inline: inline)
        }
    }

    private func inline(_ s: String) -> Text {
        if let a = try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            // file:, smb:, custom app schemes etc. are never clickable; the rest goes through
            // the OpenURLAction installed by SafeLinkHandling at the window root.
            return Text(LinkPolicy.sanitize(a))
        }
        return Text(s)
    }
}

struct TableBlock: View {
    let header: [String]
    let rows: [[String]]
    let inline: (String) -> Text
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    ForEach(Array(header.enumerated()), id: \.offset) { _, h in inline(h).font(.system(size: 13, weight: .semibold)) }
                }
                Divider()
                ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                    GridRow {
                        ForEach(Array(r.enumerated()), id: \.offset) { _, c in inline(c).font(.system(size: 13)) }
                    }
                }
            }
            .padding(12)
        }
        .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct CodeBlock: View {
    let language: String
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 11))
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Color.secondary.opacity(0.08))
            ScrollView(.horizontal, showsIndicators: false) {
                Text(SyntaxHighlighter.highlight(code))
                    .font(.system(size: 12.5, design: .monospaced))
                    .lineSpacing(2)
                    .padding(12)
                    .fixedSize(horizontal: true, vertical: true)
            }
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.secondary.opacity(0.15)))
    }
}

/// Tiny language-agnostic highlighter: keywords, strings, numbers, comments.
enum SyntaxHighlighter {
    static let keywords: Set<String> = [
        "func", "let", "var", "if", "else", "for", "while", "return", "import", "struct", "class", "enum", "case",
        "switch", "def", "in", "from", "const", "function", "public", "private", "static", "true", "false", "nil",
        "null", "None", "True", "False", "self", "async", "await", "try", "catch", "throw", "throws", "guard",
        "fn", "pub", "impl", "use", "mut", "int", "void", "new", "this", "type", "interface", "export", "default",
        "break", "continue", "protocol", "extension", "where", "lambda", "yield", "with", "as", "package", "go",
    ]

    static func highlight(_ code: String) -> AttributedString {
        var out = AttributedString()
        let kw = Color(red: 0.85, green: 0.35, blue: 0.75)
        let str = Color(red: 0.95, green: 0.45, blue: 0.35)
        let num = Color(red: 0.55, green: 0.5, blue: 1.0)
        let com = Color.secondary
        let chars = Array(code)
        var i = 0
        func emit(_ s: String, _ c: Color? = nil) {
            var a = AttributedString(s)
            if let c { a.foregroundColor = c }
            out += a
        }
        while i < chars.count {
            let c = chars[i]
            // comments
            if (c == "/" && i + 1 < chars.count && chars[i + 1] == "/") || c == "#" {
                var j = i
                while j < chars.count && chars[j] != "\n" { j += 1 }
                emit(String(chars[i..<j]), com); i = j; continue
            }
            if c == "\"" || c == "'" || c == "`" {
                var j = i + 1
                while j < chars.count && chars[j] != c && chars[j] != "\n" { if chars[j] == "\\" { j += 1 }; j += 1 }
                j = min(j + 1, chars.count)
                emit(String(chars[i..<j]), str); i = j; continue
            }
            if c.isNumber {
                var j = i
                while j < chars.count && (chars[j].isNumber || chars[j] == "." || chars[j] == "x") { j += 1 }
                emit(String(chars[i..<j]), num); i = j; continue
            }
            if c.isLetter || c == "_" {
                var j = i
                while j < chars.count && (chars[j].isLetter || chars[j].isNumber || chars[j] == "_") { j += 1 }
                let w = String(chars[i..<j])
                emit(w, keywords.contains(w) ? kw : nil); i = j; continue
            }
            var j = i
            while j < chars.count, !(chars[j].isLetter || chars[j].isNumber || chars[j] == "_" || chars[j] == "\"" || chars[j] == "'" || chars[j] == "`" || chars[j] == "#" || chars[j] == "/") { j += 1 }
            if j == i { j = i + 1 }
            emit(String(chars[i..<j])); i = j
        }
        return out
    }
}

// MARK: - Parser

struct MDListItem { var text: String; var indent: Int; var number: Int? }

enum MDBlock {
    case paragraph(String)
    case heading(Int, String)
    case code(String, String)
    case list([MDListItem], ordered: Bool)
    case quote(String)
    case rule
    case table([String], [[String]])
}

enum MarkdownParser {
    static func parse(_ text: String) -> [MDBlock] {
        var blocks: [MDBlock] = []
        let lines = text.components(separatedBy: "\n")
        var i = 0
        var para: [String] = []
        func flushPara() {
            if !para.isEmpty { blocks.append(.paragraph(para.joined(separator: "\n"))); para = [] }
        }
        while i < lines.count {
            let line = lines[i]
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("```") {
                flushPara()
                let lang = String(t.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                i += 1
                while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[i]); i += 1
                }
                blocks.append(.code(lang, code.joined(separator: "\n")))
                i += 1
                continue
            }
            if t.isEmpty { flushPara(); i += 1; continue }
            if t.hasPrefix("#") {
                let level = t.prefix(while: { $0 == "#" }).count
                if level <= 6, t.dropFirst(level).first == " " {
                    flushPara()
                    blocks.append(.heading(level, String(t.dropFirst(level + 1))))
                    i += 1; continue
                }
            }
            if t == "---" || t == "***" || t == "___" { flushPara(); blocks.append(.rule); i += 1; continue }
            if t.hasPrefix(">") {
                flushPara()
                var q: [String] = []
                while i < lines.count, lines[i].trimmingCharacters(in: .whitespaces).hasPrefix(">") {
                    q.append(String(lines[i].trimmingCharacters(in: .whitespaces).dropFirst()).trimmingCharacters(in: .whitespaces))
                    i += 1
                }
                blocks.append(.quote(q.joined(separator: "\n")))
                continue
            }
            if t.hasPrefix("|"), i + 1 < lines.count, lines[i + 1].contains("---") {
                flushPara()
                func cells(_ s: String) -> [String] {
                    s.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "|"))
                        .components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
                }
                let header = cells(t)
                var rows: [[String]] = []
                i += 2
                while i < lines.count, lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    rows.append(cells(lines[i])); i += 1
                }
                blocks.append(.table(header, rows))
                continue
            }
            if let item = listItem(line) {
                flushPara()
                var items = [item.0]
                let ordered = item.1
                i += 1
                while i < lines.count {
                    if let next = listItem(lines[i]) { items.append(next.0); i += 1; continue }
                    let nt = lines[i].trimmingCharacters(in: .whitespaces)
                    // continuation line of a list item
                    if !nt.isEmpty, lines[i].hasPrefix("  "), !nt.hasPrefix("```") {
                        items[items.count - 1].text += " " + nt; i += 1; continue
                    }
                    break
                }
                blocks.append(.list(items, ordered: ordered))
                continue
            }
            para.append(line)
            i += 1
        }
        flushPara()
        return blocks
    }

    private static func listItem(_ line: String) -> (MDListItem, Bool)? {
        let indent = line.prefix(while: { $0 == " " }).count / 2
        let t = line.trimmingCharacters(in: .whitespaces)
        for m in ["- ", "* ", "+ "] where t.hasPrefix(m) {
            return (MDListItem(text: String(t.dropFirst(2)), indent: indent, number: nil), false)
        }
        let digits = t.prefix(while: { $0.isNumber })
        if !digits.isEmpty, digits.count < 4 {
            let rest = t.dropFirst(digits.count)
            if rest.hasPrefix(". ") || rest.hasPrefix(") ") {
                return (MDListItem(text: String(rest.dropFirst(2)), indent: indent, number: Int(digits)), true)
            }
        }
        return nil
    }
}
