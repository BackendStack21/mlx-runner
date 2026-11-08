import Foundation
import MLXEngine

/// Go's `time.Duration.String()`: `1.2345s`, `850µs`, `1.5ms`, `1m30.5s`, `1h2m5s`, `0s`.
enum GoDuration {
    static func string(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds > 0 else { return "0s" }
        let ns = Int64(min(seconds * 1e9, 9e18).rounded())
        if ns == 0 { return "0s" }
        if ns < 1_000 { return "\(ns)ns" }
        if ns < 1_000_000 { return fraction(ns, per: 1_000, digits: 3) + "\u{B5}s" }
        if ns < 1_000_000_000 { return fraction(ns, per: 1_000_000, digits: 6) + "ms" }
        let hours = ns / 3_600_000_000_000
        let minutes = ns / 60_000_000_000 % 60
        let secs = fraction(ns % 60_000_000_000, per: 1_000_000_000, digits: 9) + "s"
        if hours > 0 { return "\(hours)h\(minutes)m" + secs }
        if minutes > 0 { return "\(minutes)m" + secs }
        return secs
    }

    /// `value / per` with the fractional digits and no trailing zeros.
    private static func fraction(_ value: Int64, per: Int64, digits: Int) -> String {
        let whole = value / per
        var rest = String(value % per)
        rest = String(repeating: "0", count: digits - rest.count) + rest
        while rest.hasSuffix("0") { rest.removeLast() }
        return rest.isEmpty ? "\(whole)" : "\(whole).\(rest)"
    }
}

/// What `--verbose` reports about one generation, in Ollama's units (seconds, token counts).
struct RunStats: Equatable {
    var totalDuration: Double = 0
    var loadDuration: Double = 0
    var promptEvalCount = 0
    var promptEvalCached = 0
    var promptEvalDuration: Double = 0
    var evalCount = 0
    var evalDuration: Double = 0

    init(totalDuration: Double = 0, loadDuration: Double = 0, promptEvalCount: Int = 0, promptEvalCached: Int = 0,
         promptEvalDuration: Double = 0, evalCount: Int = 0, evalDuration: Double = 0) {
        self.totalDuration = totalDuration; self.loadDuration = loadDuration
        self.promptEvalCount = promptEvalCount; self.promptEvalCached = promptEvalCached
        self.promptEvalDuration = promptEvalDuration; self.evalCount = evalCount; self.evalDuration = evalDuration
    }

    /// From the model's load time and the final stats of a generation: the prompt phase lasted
    /// `timeToFirstToken`, the decode phase `generatedTokens / tokensPerSecond`, and the total is
    /// load + prompt + decode.
    init(loadSeconds: Double, stats: GenerationStats) {
        let decode = stats.tokensPerSecond > 0 ? Double(stats.generatedTokens) / stats.tokensPerSecond : 0
        self.init(totalDuration: loadSeconds + stats.timeToFirstToken + decode, loadDuration: loadSeconds,
                  promptEvalCount: stats.promptTokens, promptEvalCached: stats.cachedPromptTokens,
                  promptEvalDuration: stats.timeToFirstToken, evalCount: stats.generatedTokens, evalDuration: decode)
    }
}

enum StatsFormat {
    /// Ollama's stats block (stderr): only values > 0, in this order, with Go durations:
    /// total duration, load duration, prompt eval count, prompt eval cached, prompt eval duration,
    /// prompt eval rate (uncached tokens / duration), eval count, eval duration, eval rate.
    static func block(_ s: RunStats) -> String {
        var out = ""
        if s.totalDuration > 0 { out += "total duration:       \(GoDuration.string(s.totalDuration))\n" }
        if s.loadDuration > 0 { out += "load duration:        \(GoDuration.string(s.loadDuration))\n" }
        if s.promptEvalCount > 0 { out += "prompt eval count:    \(s.promptEvalCount) token(s)\n" }
        if s.promptEvalCached > 0 { out += "prompt eval cached:   \(s.promptEvalCached) token(s)\n" }
        if s.promptEvalDuration > 0 {
            out += "prompt eval duration: \(GoDuration.string(s.promptEvalDuration))\n"
            let uncached = max(0, s.promptEvalCount - s.promptEvalCached)
            out += String(format: "prompt eval rate:     %.2f tokens/s\n", Double(uncached) / s.promptEvalDuration)
        }
        if s.evalCount > 0 { out += "eval count:           \(s.evalCount) token(s)\n" }
        if s.evalDuration > 0 {
            out += "eval duration:        \(GoDuration.string(s.evalDuration))\n"
            out += String(format: "eval rate:            %.2f tokens/s\n", Double(s.evalCount) / s.evalDuration)
        }
        return out
    }

    /// mlxr's own lines, byte-identical to 1.5.0 minus the leading newline:
    /// `[prompt: %d tokens, %.0f tok/s, ttft %.3fs]`, `[generated %d tokens at %.1f tok/s, stop: %@,
    /// drafts accepted %d/%d%@]` (with ` (draft model a/b)`), and when the draft model ran cycles
    /// `[draft model: %d cycles, %.2f tokens/cycle, draft %.1f ms + verify %.1f ms per cycle]`.
    static func mlxrLines(prompt: GenerationStats?, final s: GenerationStats) -> [String] {
        var lines: [String] = []
        if let p = prompt {
            lines.append(String(format: "[prompt: %d tokens, %.0f tok/s, ttft %.3fs]", p.promptTokens, p.promptTokensPerSecond, p.timeToFirstToken))
        }
        let model = s.modelDraftTokens.map { String(format: " (draft model %d/%d)", s.acceptedModelDraftTokens ?? 0, $0) } ?? ""
        lines.append(String(format: "[generated %d tokens at %.1f tok/s, stop: %@, drafts accepted %d/%d%@]", s.generatedTokens,
                            s.tokensPerSecond, s.stopReason, s.acceptedDraftTokens ?? 0, s.draftTokens ?? 0, model))
        if let c = s.draftCycles, c > 0, let dt = s.draftSeconds, let vt = s.verifySeconds {
            lines.append(String(format: "[draft model: %d cycles, %.2f tokens/cycle, draft %.1f ms + verify %.1f ms per cycle]", c,
                                Double(s.acceptedModelDraftTokens ?? 0) / Double(c) + 1, dt / Double(c) * 1000, vt / Double(c) * 1000))
        }
        return lines
    }
}
