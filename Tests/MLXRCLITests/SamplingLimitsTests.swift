import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("Shared sampling limits (AC-7)")
struct SamplingLimitsTests {
    @Test func boundsMatchTheServerContract() {
        #expect(SamplingLimits.temperature == 0...2)
        #expect(SamplingLimits.topPMax == 1)
        #expect(SamplingLimits.topK == 0...100_000)
        #expect(SamplingLimits.minP == 0...1)
        #expect(SamplingLimits.repeatPenaltyMax == 100)
        #expect(SamplingLimits.maxTokens == 32_768)
        #expect(SamplingLimits.context == 256...1_048_576)
        #expect(SamplingLimits.port == 1...65_535)
        #expect(SamplingLimits.repeatLastN == 0...4096)
    }

    @Test func serverCeilingIsTheSharedConstant() {
        #expect(OpenAIServer(port: 1).maxOutputTokens == SamplingLimits.maxTokens)
    }

    @Test func topPAndRepeatPenaltyExcludeZero() {
        #expect(!SamplingLimits.isValidTopP(0))
        #expect(SamplingLimits.isValidTopP(0.0001))
        #expect(SamplingLimits.isValidTopP(1))
        #expect(!SamplingLimits.isValidTopP(1.0001))
        #expect(!SamplingLimits.isValidTopP(.nan))
        #expect(!SamplingLimits.isValidRepeatPenalty(0))
        #expect(!SamplingLimits.isValidRepeatPenalty(-1))
        #expect(SamplingLimits.isValidRepeatPenalty(0.001))
        #expect(SamplingLimits.isValidRepeatPenalty(100))
        #expect(!SamplingLimits.isValidRepeatPenalty(100.5))
        #expect(!SamplingLimits.isValidRepeatPenalty(.infinity))
    }

    /// The server keeps clamping (not rejecting) what it accepts: exactly the behavior before this change.
    @Test func serverClampsInsteadOfRejecting() {
        #expect(SamplingLimits.clampTopK(200_000) == 100_000)
        #expect(SamplingLimits.clampTopK(40) == 40)
        #expect(SamplingLimits.clampTopK(0) == 0)
        #expect(SamplingLimits.clampTopK(-1) == 0)
        #expect(SamplingLimits.clampRepeatPenalty(500) == 100)
        #expect(SamplingLimits.clampRepeatPenalty(1.05) == 1.05)
        #expect(SamplingLimits.clampMaxTokens(1e18, cap: 32_768) == 32_768)
        #expect(SamplingLimits.clampMaxTokens(100, cap: 32_768) == 100)
        #expect(SamplingLimits.clampMaxTokens(1e18, cap: 1000) == 1000)
    }
}
