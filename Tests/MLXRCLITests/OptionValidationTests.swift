import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

/// AC-6 and AC-8: strict option validation in the real command table, before any model loads.
@Suite("Option validation (AC-6, AC-8)")
struct OptionValidationTests {
    /// The error message of a parse, or nil when it parsed.
    func failure(_ args: [String]) -> String? {
        do { _ = try CLI.parse(args); return nil } catch { return CLI.message(for: error) }
    }

    func rejected(_ args: [String], flag: String, sourceLocation: SourceLocation = #_sourceLocation) {
        let m = failure(args)
        #expect(m?.hasPrefix("invalid argument") == true && m?.contains("\(flag)\" flag") == true,
                "\(args): \(m ?? "accepted")", sourceLocation: sourceLocation)
    }

    func accepted(_ args: [String], sourceLocation: SourceLocation = #_sourceLocation) {
        let m = failure(args)
        #expect(m == nil, "\(args): \(m ?? "")", sourceLocation: sourceLocation)
    }

    @Test func maxTokens() {
        rejected(["run", "m", "--max", "abc"], flag: "--max-tokens")
        for bad in ["0", "-1", "32769"] { rejected(["run", "m", "--max", bad], flag: "--max-tokens") }
        for ok in ["1", "32768"] { accepted(["run", "m", "--max", ok]) }
    }

    @Test func port() {
        for bad in ["0", "70000", "65536", "-1", "x"] { rejected(["serve", "m", "--port", bad], flag: "--port") }
        for ok in ["1", "65535"] { accepted(["serve", "m", "--port", ok]) }
    }

    @Test func temperature() {
        for bad in ["nan", "inf", "-inf", "2.1", "-0.1"] { rejected(["run", "m", "--temp", bad], flag: "--temp") }
        for ok in ["0", "2", "0.7"] { accepted(["run", "m", "--temp", ok]) }
    }

    @Test func topP() {
        for bad in ["0", "1.01", "nan", "-0.5"] { rejected(["run", "m", "--top-p", bad], flag: "--top-p") }
        for ok in ["0.0001", "1", "0.95"] { accepted(["run", "m", "--top-p", ok]) }
    }

    @Test func topK() {
        for bad in ["100001", "-1", "1.5"] { rejected(["run", "m", "--top-k", bad], flag: "--top-k") }
        for ok in ["0", "100000"] { accepted(["run", "m", "--top-k", ok]) }
    }

    @Test func minP() {
        for bad in ["-0.1", "1.1", "nan"] { rejected(["run", "m", "--min-p", bad], flag: "--min-p") }
        for ok in ["0", "1"] { accepted(["run", "m", "--min-p", ok]) }
    }

    @Test func repeatPenalty() {
        for bad in ["0", "-1", "100.5", "inf"] { rejected(["run", "m", "--repeat-penalty", bad], flag: "--repeat-penalty") }
        for ok in ["0.001", "1", "100"] { accepted(["run", "m", "--repeat-penalty", ok]) }
    }

    @Test func repeatLastN() {
        for bad in ["4097", "-1"] { rejected(["run", "m", "--repeat-last-n", bad], flag: "--repeat-last-n") }
        for ok in ["0", "4096"] { accepted(["run", "m", "--repeat-last-n", ok]) }
    }

    @Test func seedIsUnsigned64() {
        rejected(["run", "m", "--seed", "-1"], flag: "--seed")
        rejected(["run", "m", "--seed", "18446744073709551616"], flag: "--seed")
        accepted(["run", "m", "--seed", "0"])
        accepted(["run", "m", "--seed", "18446744073709551615"])
    }

    @Test func context() {
        for cmd in ["run", "serve"] {
            for bad in ["255", "1048577", "x"] { rejected([cmd, "m", "--ctx", bad], flag: "--ctx") }
            for ok in ["256", "1048576"] { accepted([cmd, "m", "--ctx", ok]) }
        }
    }

    @Test func systemPromptAndValuesAreRecorded() throws {
        let inv = try CLI.parse(["run", "m", "--system", "Be brief", "--top-p", "0.9", "--top-k", "0", "--seed", "7",
                                 "--min-p", "0.1", "--repeat-penalty", "1.2", "--repeat-last-n", "32"])
        #expect(inv.string("system") == "Be brief")
        #expect(inv.double("top-p") == 0.9)
        #expect(inv.int("top-k") == 0)
        #expect(inv.string("seed") == "7")
        #expect(inv.double("min-p") == 0.1)
        #expect(inv.double("repeat-penalty") == 1.2)
        #expect(inv.int("repeat-last-n") == 32)
    }

    @Test func tableRangesComeFromSharedLimits() throws {
        func range(_ name: String, _ cmd: String = "run") -> ValueKind? {
            CLI.commands.first { $0.name == cmd }?.options.first { $0.long == name }?.value
        }
        if case .int(let r)? = range("max-tokens") { #expect(r.lowerBound == 1 && r.upperBound == SamplingLimits.maxTokens) } else { Issue.record("max-tokens is not an int option") }
        if case .int(let r)? = range("ctx") { #expect(r == SamplingLimits.context) } else { Issue.record("ctx is not an int option") }
        if case .int(let r)? = range("port", "serve") { #expect(r == SamplingLimits.port) } else { Issue.record("port is not an int option") }
        if case .real(let r)? = range("temp") { #expect(r == SamplingLimits.temperature) } else { Issue.record("temp is not a real option") }
    }

    // AC-8: options anywhere, the model stays the model.
    @Test func specAndApiKeyDoNotSwallowTheModel() throws {
        let a = try CLI.parse(["run", "--spec", "M"])
        #expect(a.positionals == ["M"] && a.bool("spec"))
        let b = try CLI.parse(["serve", "--api-key", "K", "M"])
        #expect(b.positionals == ["M"] && b.string("api-key") == "K")
        let c = try CLI.parse(["serve", "M", "--api-key=K", "--port", "8080"])
        #expect(c.positionals == ["M"] && c.string("api-key") == "K" && c.int("port") == 8080)
    }

    @Test func helpFlagsShowHelpInsteadOfRunning() async {
        for cmd in ["tokenize", "search", "bench"] {
            let cap = Capture()
            #expect(await CLI.run([cmd, "-h"], console: cap.console()) == 0, "\(cmd)")
            #expect(cap.stdout.contains("Usage:") && cap.stdout.contains("Flags:"), "\(cmd)")
            #expect(cap.stderr.isEmpty, "\(cmd)")
        }
    }

    @Test func invalidValuesExitOneBeforeRunning() async {
        for args in [["run", "m", "--max", "0"], ["serve", "m", "--port", "70000"], ["run", "m", "--top-k", "100001"],
                     ["run", "m", "--temp", "nan"], ["run", "m", "--repeat-penalty", "0"]] {
            let cap = Capture()
            #expect(await CLI.run(args, console: cap.console()) == 1, "\(args)")
            #expect(cap.stderr.hasPrefix("Error: invalid argument"), "\(args): \(cap.stderr)")
            #expect(cap.stdout.isEmpty, "\(args)")
        }
    }
}
