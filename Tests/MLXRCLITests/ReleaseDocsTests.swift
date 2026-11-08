import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

/// SL08: the 2.0.0 release surface: version string, changelog, and the documentation that shows commands.
@Suite("Release 2.0.0 docs and version (SL08)")
struct ReleaseDocsTests {
    static func read(_ path: String) -> String {
        (try? String(contentsOf: DocsCommands.root.appendingPathComponent(path), encoding: .utf8)) ?? ""
    }

    @Test func versionIs2_0_0() {
        #expect(MLXRunnerVersion.current == "2.0.0")
    }

    @Test func dashVPrintsTheVersion() async {
        for flag in ["-v", "--version"] {
            let cap = Capture()
            let code = await CLI.run([flag], console: cap.console())
            #expect(code == 0)
            #expect(cap.stdout == "mlxr version is 2.0.0\n", "\(flag): \(cap.stdout)")
        }
    }

    @Test func changelogHasA2_0_0Section() {
        let text = Self.read("CHANGELOG.md")
        let lines = text.components(separatedBy: "\n")
        guard let i = lines.firstIndex(of: "## 2.0.0") else {
            Issue.record("CHANGELOG.md has no '## 2.0.0' section"); return
        }
        // It is the newest release: first section heading in the file.
        #expect(lines.first(where: { $0.hasPrefix("## ") }) == "## 2.0.0")
        let body = lines[(i + 1)...].prefix { !$0.hasPrefix("## ") }.joined(separator: "\n")
        #expect(body.trimmingCharacters(in: .whitespacesAndNewlines).count > 80, "section is empty")
    }

    @Test func readmeAndAgentsGiveTheGreedyFlags() {
        for f in ["README.md", "AGENTS.md"] {
            #expect(Self.read(f).contains("--temp 0 --seed 42 --max 256"), "\(f) lacks '--temp 0 --seed 42 --max 256'")
        }
    }

    @Test func siteCliInstallCheckLineIsBareMlxrPrintsHelp() {
        let html = Self.read("site/index.html")
        #expect(!html.isEmpty)
        #expect(!html.contains("prints its usage line"), "old comment still present")
        // Template (data-template) and copy text (data-copy) end with a bare `mlxr` line.
        #expect(html.contains("/usr/local/bin/mlxr\nmlxr   # prints help\""), "data-template check line")
        #expect(html.contains("/usr/local/bin/mlxr\nmlxr   # prints help\"><pre>")
                || html.contains("# prints help\"><pre>"), "template attribute must close before <pre>")
        // Visible block: bare mlxr followed by the comment span.
        #expect(html.contains("mlxr   <span class=\"c\"># prints help</span></code></pre>"), "visible check line")
        // The copy text: bare `mlxr` plus the comment, inside data-copy.
        guard let r = html.range(of: "data-copy=\"curl -LO") else {
            Issue.record("no cli-install data-copy"); return
        }
        let copy = html[r.upperBound...].prefix { $0 != "\"" }
        let lastLine = copy.split(separator: "\n", omittingEmptySubsequences: false).last.map(String.init) ?? ""
        #expect(lastLine == "mlxr   # prints help", "data-copy last line: \(lastLine)")
    }

    @Test func docsParseStillCoversEveryMlxrLine() {
        let table = LibraryFixture.standard().table()
        var files = DocsCommands.docFiles()
        files.append(DocsCommands.root.appendingPathComponent("AGENTS.md"))
        var total = 0
        for f in files {
            let text = (try? String(contentsOf: f, encoding: .utf8)) ?? ""
            let r = DocsCommands.check(text, file: f.lastPathComponent, html: f.pathExtension == "html", commands: table)
            total += r.checked
            #expect(r.failures.isEmpty, "\(r.failures)")
        }
        #expect(total > 20, "only \(total) mlxr lines found")
        // The greedy-flags line itself parses against the real command table.
        let greedy = DocsCommands.check("```\nmlxr run <model> -p \"Explain RoPE\" --temp 0 --seed 42 --max 256\n```",
                                        file: "g.md", html: false, commands: table)
        #expect(greedy.checked == 1 && greedy.failures.isEmpty, "\(greedy.failures)")
    }
}
