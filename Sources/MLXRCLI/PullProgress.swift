import Foundation

/// The progress line of `mlxr pull`, in Ollama's layout:
/// `"<label> %3.0f%% ▕█…▏ %6s/%6s  %6s/s  %6s"` (done/total, speed, time left).
enum PullProgress {
    private static func pad6(_ s: String) -> String { String(repeating: " ", count: max(0, 6 - s.count)) + s }

    private static func size(_ n: Double) -> String {
        guard n.isFinite else { return CLI.humanBytes(0) }
        return CLI.humanBytes(Int(min(max(n, 0), 1e15)))
    }

    /// One bar line. A finished download shows only the total after the bar.
    static func line(label: String, done: Int64, total: Int64, speed: Double, barWidth: Int = 10) -> String {
        let width = max(barWidth, 1)
        let finished = total > 0 && done >= total
        let fraction = total > 0 ? min(1, max(0, Double(done) / Double(total))) : 0
        let filled = finished ? width : min(width, Int(Double(width) * fraction))
        let bar = String(repeating: "█", count: filled) + String(repeating: " ", count: width - filled)
        let head = label + " " + String(format: "%3.0f%%", fraction * 100) + " ▕" + bar + "▏ "
        if finished { return head + pad6(size(Double(total))) }
        let eta = total > 0 && speed > 0 && speed.isFinite ? duration(Double(total - done) / speed) : ""
        return head + pad6(size(Double(done))) + "/" + pad6(size(Double(total))) + "  " + pad6(size(speed)) + "/s  " + pad6(eta)
    }

    /// Go-style duration rounded to whole seconds: `24s`, `1m30s`, `1h2m3s`.
    static func duration(_ seconds: Double) -> String {
        let total = Int(min(max(seconds.isFinite ? seconds : 0, 0), 1e9).rounded())
        let h = total / 3600, m = total % 3600 / 60, s = total % 60
        if h > 0 { return "\(h)h\(m)m\(s)s" }
        if m > 0 { return "\(m)m\(s)s" }
        return "\(s)s"
    }
}

/// Draws the progress line on stderr: redrawn at most every 100 ms with `\r` when stderr is a TTY,
/// printed once when it finishes otherwise.
final class PullRenderer: @unchecked Sendable {
    private let console: Console
    private let label: String
    private let barWidth: Int
    private let now: @Sendable () -> Date
    private let lock = NSLock()
    private var lastDraw: Date?
    private var wroteLine = false
    private var ended = false

    init(console: Console, label: String = "pulling model:", barWidth: Int = 10, now: @escaping @Sendable () -> Date = { Date() }) {
        self.console = console; self.label = label; self.barWidth = barWidth; self.now = now
    }

    func update(done: Int64, total: Int64, speed: Double) {
        lock.lock(); defer { lock.unlock() }
        guard !ended else { return }
        let finished = total > 0 && done >= total
        let line = PullProgress.line(label: label, done: done, total: total, speed: speed, barWidth: barWidth)
        if console.stderrIsTTY {
            let t = now()
            guard finished || lastDraw == nil || t.timeIntervalSince(lastDraw!) >= 0.1 else { return }
            lastDraw = t
            console.err("\r" + line + "\u{1B}[K")
            wroteLine = true
        } else if finished && !wroteLine {
            console.err(line)
            wroteLine = true
        }
    }

    /// Ends the line (a `\n` after the last redraw).
    func finish() {
        lock.lock(); defer { lock.unlock() }
        guard !ended else { return }
        ended = true
        if wroteLine { console.err("\n") }
    }
}
