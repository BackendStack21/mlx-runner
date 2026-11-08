import Foundation
import Testing
@testable import MLXRCLI

/// SL04 (AC-16): the docs describe `mlxr upgrade`, and stay true for 1.x users who do not have it.
@Suite("mlxr upgrade documentation (SL04)")
struct UpgradeDocsTests {
    static func read(_ path: String) -> String { ReleaseDocsTests.read(path) }

    /// Text between `start` and the next line starting with `endPrefix` (exclusive).
    static func section(_ text: String, from start: String, until endPrefix: String) -> String {
        let lines = text.components(separatedBy: "\n")
        guard let i = lines.firstIndex(where: { $0.hasPrefix(start) }) else { return "" }
        return lines[(i + 1)...].prefix { !$0.hasPrefix(endPrefix) }.joined(separator: "\n")
    }

    @Test func readmeCliSectionDescribesUpgrade() {
        let body = Self.section(Self.read("README.md"), from: "## Command line (`mlxr`)", until: "## ")
        #expect(!body.isEmpty, "README has no Command line section")
        #expect(body.contains("mlxr upgrade"), "CLI section does not mention `mlxr upgrade`")
        #expect(body.contains("sudo mlxr upgrade"), "CLI section does not mention `sudo mlxr upgrade` for /usr/local/bin")
        #expect(body.contains("/usr/local/bin"), "CLI section does not say when sudo is needed")
    }

    @Test func readmeCliOnlyInstallStepsDescribeUpgrade() {
        let readme = Self.read("README.md")
        let body = Self.section(readme, from: "The CLI on its own", until: "### ")
        #expect(!body.isEmpty, "README has no CLI-only install steps")
        #expect(body.contains("mlxr upgrade"), "install steps do not mention `mlxr upgrade`")
        #expect(body.contains("sudo mlxr upgrade"), "install steps do not mention `sudo mlxr upgrade`")
        #expect(!body.contains("The CLI doesn't update itself"), "stale claim that the CLI never updates itself")
    }

    @Test func readmeSaysOneDotXHasNoUpgradeCommand() {
        let body = Self.section(Self.read("README.md"), from: "The CLI on its own", until: "### ")
        #expect(body.contains("1.x"), "install steps do not say how 1.x users move to 2.0")
    }

    @Test func siteCliInstallNoteMentionsUpgradeAndStaysTrueFor1x() {
        let html = Self.read("site/index.html")
        guard let r = html.range(of: "id=\"cli-install\"") else {
            Issue.record("no #cli-install"); return
        }
        let rest = html[r.upperBound...]
        let end = rest.range(of: "</section>")?.lowerBound ?? rest.endIndex
        let block = String(rest[..<end])
        #expect(block.contains("mlxr upgrade"), "#cli-install does not mention `mlxr upgrade`")
        #expect(!block.contains("the CLI doesn't update itself"), "stale claim that the CLI never updates itself")
        #expect(!block.contains("doesn't update itself"), "stale claim that the CLI never updates itself")
        #expect(block.contains("1.x"), "#cli-install does not say 1.x has no upgrade command")
        // The 1.x caveat must say the command is missing there (not that it exists).
        let lower = block.lowercased()
        #expect(lower.contains("1.x has no") || lower.contains("1.x doesn't have") || lower.contains("1.x does not have")
                || lower.contains("no upgrade command") || lower.contains("without an upgrade command"),
                "#cli-install does not state that 1.x has no upgrade command")
    }

    @Test func siteInstallCommandsAreUntouched() {
        // The install block must still end with a bare `mlxr` check line (ReleaseDocsTests) and keep `sudo install`.
        let html = Self.read("site/index.html")
        #expect(html.contains("sudo install -m 755 {dir:cli}/mlxr /usr/local/bin/mlxr"))
    }

    @Test func changelog2_0_0HasAnUpgradeBullet() {
        let body = Self.section(Self.read("CHANGELOG.md"), from: "## 2.0.0", until: "## ")
        let bullets = body.components(separatedBy: "\n").filter { $0.hasPrefix("- ") && $0.contains("mlxr upgrade") }
        #expect(!bullets.isEmpty, "no bullet in '## 2.0.0' mentioning `mlxr upgrade`")
        let b = bullets.joined(separator: "\n").lowercased()
        #expect(b.contains("update"), "upgrade bullet does not mention `update` (alias)")
        #expect(b.contains("--check"), "upgrade bullet does not mention --check")
        #expect(b.contains("sudo"), "upgrade bullet does not mention sudo")
    }

    @Test func everyUpgradeLineInTheDocsParsesAgainstTheRealTable() {
        let table = LibraryFixture.standard().table()
        var seen = 0
        for f in ["README.md", "site/index.html"] {
            let text = Self.read(f)
            let r = DocsCommands.check(text, file: f, html: f.hasSuffix(".html"), commands: table)
            #expect(r.failures.isEmpty, "\(r.failures)")
            seen += text.components(separatedBy: "\n").filter { $0.contains("mlxr upgrade") }.count
        }
        #expect(seen >= 2, "only \(seen) lines mention `mlxr upgrade` in README and site")
        let sample = DocsCommands.check("```\nmlxr upgrade\nsudo mlxr upgrade --check\nmlxr update\n```",
                                        file: "u.md", html: false, commands: table)
        #expect(sample.checked >= 2 && sample.failures.isEmpty, "\(sample.failures)")
    }
}
