import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("verbose stats (AC-15)")
struct RunStatsTests {
    @Test func goDurations() {
        let micro = "\u{B5}s"
        let cases: [(Double, String)] = [
            (0, "0s"), (1, "1s"), (2, "2s"), (1.2345, "1.2345s"), (0.5, "500ms"), (0.0015, "1.5ms"),
            (0.00085, "850" + micro), (0.0000015, "1.5" + micro), (0.0000005, "500ns"),
            (90.5, "1m30.5s"), (60, "1m0s"), (3725, "1h2m5s"), (3600, "1h0m0s"),
        ]
        for (seconds, text) in cases { #expect(GoDuration.string(seconds) == text, "\(seconds)") }
    }

    func fullStats() -> GenerationStats { FakeEngine.stats(prompt: 100, cached: 20, ttft: 0.5, generated: 50, rate: 25) }

    @Test func statsFromTheEngineNumbers() {
        let s = RunStats(loadSeconds: 1.5, stats: fullStats())
        #expect(s == RunStats(totalDuration: 4, loadDuration: 1.5, promptEvalCount: 100, promptEvalCached: 20,
                              promptEvalDuration: 0.5, evalCount: 50, evalDuration: 2))
    }

    @Test func fullBlockInOllamasOrderAndLayout() {
        let block = StatsFormat.block(RunStats(loadSeconds: 1.5, stats: fullStats()))
        #expect(block == """
        total duration:       4s
        load duration:        1.5s
        prompt eval count:    100 token(s)
        prompt eval cached:   20 token(s)
        prompt eval duration: 500ms
        prompt eval rate:     160.00 tokens/s
        eval count:           50 token(s)
        eval duration:        2s
        eval rate:            25.00 tokens/s

        """)
    }

    @Test func onlyValuesAboveZeroArePrinted() {
        #expect(StatsFormat.block(RunStats()) == "")
        let onlyEval = StatsFormat.block(RunStats(evalCount: 10, evalDuration: 0.5))
        #expect(onlyEval == "eval count:           10 token(s)\neval duration:        500ms\neval rate:            20.00 tokens/s\n")
        let noCache = StatsFormat.block(RunStats(promptEvalCount: 8, promptEvalDuration: 0.25))
        #expect(noCache == "prompt eval count:    8 token(s)\nprompt eval duration: 250ms\nprompt eval rate:     32.00 tokens/s\n")
        let noRate = StatsFormat.block(RunStats(evalCount: 10))
        #expect(noRate == "eval count:           10 token(s)\n")
    }

    @Test func mlxrLinesAreByteIdenticalToTheOldFormat() {
        let prompt = FakeEngine.stats(prompt: 100, ttft: 0.5, promptRate: 200, generated: 0, rate: 0)
        let final = FakeEngine.stats(generated: 50, rate: 25, stop: "stop")
        #expect(StatsFormat.mlxrLines(prompt: prompt, final: final) == [
            "[prompt: 100 tokens, 200 tok/s, ttft 0.500s]",
            "[generated 50 tokens at 25.0 tok/s, stop: stop, drafts accepted 0/0]",
        ])
        #expect(StatsFormat.mlxrLines(prompt: nil, final: final) == [
            "[generated 50 tokens at 25.0 tok/s, stop: stop, drafts accepted 0/0]",
        ])
    }

    @Test func draftLinesCarryTheModelDraftNumbers() {
        var final = FakeEngine.stats(generated: 50, rate: 25, stop: "length")
        final.draftTokens = 10; final.acceptedDraftTokens = 7
        final.modelDraftTokens = 8; final.acceptedModelDraftTokens = 6
        final.draftCycles = 4; final.draftSeconds = 0.04; final.verifySeconds = 0.08
        #expect(StatsFormat.mlxrLines(prompt: nil, final: final) == [
            "[generated 50 tokens at 25.0 tok/s, stop: length, drafts accepted 7/10 (draft model 6/8)]",
            "[draft model: 4 cycles, 2.50 tokens/cycle, draft 10.0 ms + verify 20.0 ms per cycle]",
        ])
    }
}
