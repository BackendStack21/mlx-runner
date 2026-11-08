import Foundation
import Testing
@testable import MLXRCLI

@Suite("pull progress renderer (AC-13)")
struct PullProgressTests {
    static let label = "pulling model:"

    @Test func lineMatchesOllamasLayout() {
        let line = PullProgress.line(label: Self.label, done: 900_000_000, total: 2_000_000_000, speed: 45_000_000, barWidth: 16)
        #expect(line == "pulling model:  45% ▕███████         ▏ 900 MB/2.0 GB   45 MB/s     24s")
    }

    @Test func aFinishedLineShowsOnlyTheTotalAfterTheBar() {
        let line = PullProgress.line(label: Self.label, done: 2_000_000_000, total: 2_000_000_000, speed: 45_000_000, barWidth: 16)
        #expect(line == "pulling model: 100% ▕████████████████▏ 2.0 GB")
        let over = PullProgress.line(label: Self.label, done: 3_000_000_000, total: 2_000_000_000, speed: 1, barWidth: 16)
        #expect(over == line, "more bytes than the total counts as finished")
    }

    @Test func theDefaultBarIsTenCellsWide() {
        let line = PullProgress.line(label: Self.label, done: 500_000_000, total: 1_000_000_000, speed: 100_000_000)
        #expect(line.hasPrefix("pulling model:  50% ▕█████     ▏ "), "\(line)")
    }

    @Test func zeroSpeedAndZeroTotalNeverPrintNanOrInf() {
        for (done, total, speed) in [(Int64(0), Int64(1_000_000), 0.0), (0, 0, 0), (5, 0, 10)] {
            let line = PullProgress.line(label: Self.label, done: done, total: total, speed: speed)
            #expect(line.hasPrefix(Self.label), "\(line)")
            #expect(!line.lowercased().contains("nan") && !line.lowercased().contains("inf"), "\(line)")
        }
    }

    @Test func durationsAreGoStyleAndRoundedToSeconds() {
        #expect(PullProgress.duration(24.4) == "24s")
        #expect(PullProgress.duration(0.2) == "0s")
        #expect(PullProgress.duration(90) == "1m30s")
        #expect(PullProgress.duration(59.6) == "1m0s")
        #expect(PullProgress.duration(3723) == "1h2m3s")
    }

    @Test func aTerminalRedrawsAtMostEvery100msAndAlwaysAtTheEnd() {
        let cap = Capture(), clock = TestClock()
        let r = PullRenderer(console: cap.console(stderrTTY: true), barWidth: 10, now: { clock.now })
        r.update(done: 100_000_000, total: 1_000_000_000, speed: 10_000_000)
        clock.advance(0.01)
        r.update(done: 110_000_000, total: 1_000_000_000, speed: 10_000_000)
        clock.advance(0.01)
        r.update(done: 120_000_000, total: 1_000_000_000, speed: 10_000_000)
        clock.advance(0.13)
        r.update(done: 300_000_000, total: 1_000_000_000, speed: 10_000_000)
        clock.advance(0.01)
        r.update(done: 1_000_000_000, total: 1_000_000_000, speed: 10_000_000)
        r.finish()
        #expect(cap.stderr.filter { $0 == "\r" }.count == 3, "first, after 100 ms, and the finished line")
        #expect(cap.stderr.contains(PullProgress.line(label: Self.label, done: 1_000_000_000, total: 1_000_000_000, speed: 10_000_000)))
        #expect(!cap.stderr.contains(PullProgress.line(label: Self.label, done: 110_000_000, total: 1_000_000_000, speed: 10_000_000)))
        #expect(cap.stderr.filter { $0 == "\n" }.count == 1 && cap.stderr.hasSuffix("\n"))
        #expect(cap.stdout.isEmpty)
    }

    @Test func withoutATerminalNothingIsRedrawnAndTheFinishedLineIsPrintedOnce() {
        let cap = Capture(), clock = TestClock()
        let r = PullRenderer(console: cap.console(stderrTTY: false), barWidth: 10, now: { clock.now })
        for done in stride(from: Int64(100_000_000), through: 900_000_000, by: 200_000_000) {
            clock.advance(1)
            r.update(done: done, total: 1_000_000_000, speed: 10_000_000)
        }
        #expect(cap.stderr.isEmpty, "partial progress prints nothing without a terminal")
        r.update(done: 1_000_000_000, total: 1_000_000_000, speed: 10_000_000)
        r.finish()
        #expect(!cap.stderr.contains("\r"))
        #expect(cap.stderr == PullProgress.line(label: Self.label, done: 1_000_000_000, total: 1_000_000_000, speed: 10_000_000) + "\n")
    }

    @Test func aCancelledPullWithoutATerminalLeavesNoPartialLine() {
        let cap = Capture()
        let r = PullRenderer(console: cap.console(stderrTTY: false))
        r.update(done: 5, total: 10, speed: 1)
        r.finish()
        #expect(cap.stderr.isEmpty)
    }
}
