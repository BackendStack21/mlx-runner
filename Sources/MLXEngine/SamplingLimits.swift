import Foundation

/// The bounds of every sampling and serving number, shared by the HTTP server and the `mlxr` CLI.
/// The server clamps some values to these bounds; the CLI rejects values outside them.
public enum SamplingLimits {
    public static let temperature: ClosedRange<Double> = 0...2
    /// Top-p is valid in (0, topPMax].
    public static let topPMax: Double = 1
    public static let topK: ClosedRange<Int> = 0...100_000
    public static let minP: ClosedRange<Double> = 0...1
    /// Repeat penalty is valid in (0, repeatPenaltyMax].
    public static let repeatPenaltyMax: Double = 100
    public static let maxTokens: Int = 32_768
    public static let context: ClosedRange<Int> = 256...1_048_576
    public static let port: ClosedRange<Int> = 1...65_535
    public static let repeatLastN: ClosedRange<Int> = 0...4096

    public static func isValidTopP(_ v: Double) -> Bool { v > 0 && v <= topPMax }
    public static func isValidRepeatPenalty(_ v: Double) -> Bool { v > 0 && v <= repeatPenaltyMax }

    /// What the server does with out-of-range values it accepts: top_k and repeat penalty are
    /// clamped to their ceiling, max_tokens to the given per-server cap.
    public static func clampTopK(_ v: Double) -> Int { v <= 0 ? 0 : Int(min(v, Double(topK.upperBound))) }
    public static func clampRepeatPenalty(_ v: Double) -> Double { min(v, repeatPenaltyMax) }
    public static func clampMaxTokens(_ v: Double, cap: Int) -> Int { Int(min(v, Double(cap))) }
}
