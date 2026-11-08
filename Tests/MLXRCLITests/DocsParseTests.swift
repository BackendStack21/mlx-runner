import Foundation
import Testing
@testable import MLXRCLI

/// Extracts every `mlxr ...` command line from documentation text and parses it with the real command table.
enum DocsCommands {
    struct Line { var file: String; var number: Int; var text: String }
    struct Failure: CustomStringConvertible { var line: Line; var reason: String
        var description: String { "\(line.file):\(line.number): `\(line.text)`: \(reason)" } }

    /// Lines that mention `mlxr` inside code but are not invocations of it. Explicit, with the reason;
    /// the parser itself is never relaxed to accept them.
    static let notInvocations: [String: String] = [:]

    static func decodeEntities(_ s: String) -> String {
        var out = s
        for (a, b) in [("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#39;", "'"), ("&amp;", "&")] {
            out = out.replacingOccurrences(of: a, with: b)
        }
        return out
    }

    static func stripTags(_ s: String) -> String {
        s.replacingOccurrences(of: "<[^>]*>", with: "", options: .regularExpression)
    }

    /// Candidate command texts of one source line: every inline code span, and the whole line inside a code block.
    static func candidates(_ raw: String, html: Bool, inBlock: Bool) -> [(text: String, inline: Bool)] {
        var line = raw
        var spans: [String] = []
        if html {
            // <code>...</code> spans first, then the tag-stripped whole line (terminal blocks).
            let ns = line as NSString
            let re = try! NSRegularExpression(pattern: "<code>(.*?)</code>")
            for m in re.matches(in: line, range: NSRange(location: 0, length: ns.length)) {
                spans.append(decodeEntities(stripTags(ns.substring(with: m.range(at: 1)))))
            }
            line = decodeEntities(stripTags(line))
        } else {
            let ns = line as NSString
            let re = try! NSRegularExpression(pattern: "`([^`]+)`")
            for m in re.matches(in: line, range: NSRange(location: 0, length: ns.length)) {
                spans.append(ns.substring(with: m.range(at: 1)))
            }
        }
        return spans.map { ($0, true) } + (inBlock ? [(line, false)] : [])
    }

    /// Words of a command line: shell quoting, `# comment` removed, `<placeholder>` made a plain word,
    /// `[optional ...]` groups and a trailing `...` dropped. Nil when the text is not a mlxr invocation.
    static func words(_ candidate: String) -> [String]? {
        var text = candidate.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("$ ") { text.removeFirst(2) }
        text = text.replacingOccurrences(of: "\\[[^\\]]*\\]", with: "", options: .regularExpression)
        var words: [String] = [], cur = "", inWord = false, quote: Character? = nil
        for ch in text {
            if let q = quote {
                if ch == q { quote = nil } else { cur.append(ch) }
            } else if ch == "\"" || ch == "'" {
                quote = ch; inWord = true
            } else if ch == " " || ch == "\t" {
                if inWord { words.append(cur); cur = ""; inWord = false }
            } else if ch == "#" && !inWord {
                break
            } else {
                cur.append(ch); inWord = true
            }
        }
        if inWord { words.append(cur) }
        // Leading VAR=value assignments.
        while let w = words.first, w.range(of: "^[A-Z_][A-Z0-9_]*=", options: .regularExpression) != nil { words.removeFirst() }
        guard words.first == "mlxr" else { return nil }
        words.removeFirst()
        if words.last == "..." { words.removeLast() }
        return words.map {
            $0.hasPrefix("<") && $0.hasSuffix(">") && $0.count > 2 ? String($0.dropFirst().dropLast()) : $0
        }
    }

    /// Parse every mlxr command line found in `text`; returns the number checked and the failures.
    static func check(_ text: String, file: String, html: Bool, commands: [Command]) -> (checked: Int, failures: [Failure]) {
        var checked = 0
        var failures: [Failure] = []
        var seen = Set<String>()
        var inBlock = false
        for (i, raw) in text.components(separatedBy: "\n").enumerated() {
            // A code block: a markdown fence, or an html <pre> element. Prose lines are not command lines.
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            var blockLine = inBlock
            if html {
                if raw.contains("<pre") { inBlock = true; blockLine = true }
                if raw.contains("</pre>") { inBlock = false }
            } else if trimmed.hasPrefix("```") {
                inBlock.toggle(); blockLine = false
            }
            for (cand, inline) in candidates(raw, html: html, inBlock: blockLine) {
                guard let args = words(cand) else { continue }
                let line = Line(file: file, number: i + 1, text: cand.trimmingCharacters(in: .whitespaces))
                guard seen.insert("\(i):\(args.joined(separator: "\u{1}"))").inserted else { continue }
                if notInvocations[line.text] != nil { continue }
                checked += 1
                if args.isEmpty { continue }  // a bare `mlxr` prints help
                // A reference to a command by name (`mlxr run`, `mlxr bench`) must name a real command.
                if args.count == 1, !args[0].hasPrefix("-") {
                    if CLI.findCommand(args[0], in: commands) == nil {
                        failures.append(Failure(line: line, reason: "unknown command \(args[0])"))
                    }
                    continue
                }
                do { _ = try CLI.parse(args, commands: commands) }
                catch {
                    // `mlxr run --spec` in prose names a flag without the arguments around it: pad the
                    // missing positionals, so only a real flag or value problem still fails.
                    if inline, let c = CLI.findCommand(args[0], in: commands) {
                        let need = c.args.filter { !$0.optional }.count
                        if (1...max(1, need)).contains(where: { (try? CLI.parse(args + Array(repeating: "placeholder", count: $0), commands: commands)) != nil }) { continue }
                    }
                    failures.append(Failure(line: line, reason: CLI.message(for: error)))
                }
            }
        }
        return (checked, failures)
    }

    /// The package root, resolved from this file: Tests/MLXRCLITests/DocsParseTests.swift.
    static var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    static func docFiles() -> [URL] {
        var files = [root.appendingPathComponent("README.md"), root.appendingPathComponent("site/index.html")]
        for dir in ["docs"] {
            let base = root.appendingPathComponent(dir)
            if let e = FileManager.default.enumerator(at: base, includingPropertiesForKeys: nil) {
                for case let u as URL in e where ["md", "html"].contains(u.pathExtension) { files.append(u) }
            }
        }
        return files
    }
}

@Suite("Documented mlxr command lines parse (AC-21)")
struct DocsParseTests {
    let table = LibraryFixture.standard().table()

    @Test func goodLinesParseAndBadFlagsAreReported() {
        let good = """
        ```
        mlxr run mlx-community/Qwen3-4B-4bit -p "Explain RoPE" --max 300 [--temp 0.7] [--think|--no-think]
        $ mlxr serve <model> --port 1234 --sparse 4096 ...
        MLXR_BENCH_CTX=8192 mlxr bench <model>   # long context
        Use `mlxr list` and `mlxr download <repo>` then `mlxr completion zsh`.
        mlxr
        sudo install -m 755 dir/mlxr /usr/local/bin/mlxr
        ```
        """
        let r = DocsCommands.check(good, file: "good.md", html: false, commands: table)
        #expect(r.failures.isEmpty, "\(r.failures)")
        #expect(r.checked >= 7, "found \(r.checked) lines")

        let bad = DocsCommands.check("""
        ```
        mlxr run m --no-such-flag 1
        `mlxr frobnicate`
        mlxr run -p hi
        mlxr search --limit 0 q
        ```
        """, file: "bad.md", html: false, commands: table)
        #expect(bad.checked == 4)
        #expect(bad.failures.count == 4, "\(bad.failures)")
        #expect(bad.failures.first?.reason.contains("no-such-flag") == true, "\(bad.failures)")
    }

    @Test func htmlCodeIsDecodedBeforeParsing() {
        let html = """
        <li><code>mlxr run &lt;model&gt; -p "Explain RoPE" --no-think</code></li>
        <pre><code><span class="p">$</span> mlxr run m -p "Write a haiku." --temp 0
        </code></pre>
        <li><code>mlxr serve &lt;model&gt; --bogus 1</code></li>
        <p>mlxr is a prose line, not a command</p>
        """
        let r = DocsCommands.check(html, file: "x.html", html: true, commands: table)
        #expect(r.failures.count == 1 && r.failures[0].line.number == 4, "\(r.failures)")
    }

    @Test func aFlagMissingFromTheTableFails() {
        let reduced = table.map { c -> Command in
            var c = c
            if c.name == "run" { c.options.removeAll { $0.long == "think" } }
            return c
        }
        let r = DocsCommands.check("```\nmlxr run m --think\n```", file: "x.md", html: false, commands: reduced)
        #expect(r.failures.count == 1)
    }

    @Test func docsDirectoryAndSiteAreFound() {
        let names = DocsCommands.docFiles().map { $0.lastPathComponent }
        #expect(names.contains("README.md") && names.contains("index.html") && names.contains("sparse-decode-attention.md"), "\(names)")
        for f in DocsCommands.docFiles() { #expect(FileManager.default.fileExists(atPath: f.path), "\(f.path)") }
    }

    @Test func everyDocumentedMlxrCommandLineParses() throws {
        var total = 0
        var failures: [DocsCommands.Failure] = []
        for url in DocsCommands.docFiles() {
            let text = try String(contentsOf: url, encoding: .utf8)
            let rel = url.path.replacingOccurrences(of: DocsCommands.root.path + "/", with: "")
            let r = DocsCommands.check(text, file: rel, html: url.pathExtension == "html", commands: table)
            total += r.checked
            failures += r.failures
        }
        #expect(total >= 15, "the docs hold many mlxr command lines; found \(total)")
        #expect(failures.isEmpty, "\(failures.map(\.description).joined(separator: "\n"))")
    }
}
