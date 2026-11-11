import Foundation
import MLXEngine

extension CLI {
    static let sieveHelp = "Sieve profile: off or balanced (changes output; needs `mlxr sieve MODEL` first)"

    /// `--sieve <off|balanced>`; without the option, `MLXR_SIEVE` (off when unset).
    static func sieveSetting(_ s: String?) throws -> SieveProfile? {
        do { return try SieveProfile.parse(s ?? ProcessInfo.processInfo.environment["MLXR_SIEVE"]) } catch {
            throw CLIError.failure((error as? LocalizedError)?.errorDescription ?? "\(error)")
        }
    }

    /// One line on what a load did with the Sieve profile it was asked for.
    static func sieveSummary(_ info: LoadedModelInfo) -> String? {
        if let h = info.sieveHead { return "sieve: balanced (" + sieveLine(h) + ")" }
        if let why = info.sieveNote { return "sieve: off (\(why))" }
        return nil
    }

    static func sieveLine(_ h: SieveHeadInfo) -> String {
        String(format: "head by rows: rank %d, %.0f exact rows per token, %.0f%% of the head's bytes; on %d held-out states the dense argmax is kept in %.2f%%, %.2f%% of sampled tokens change (top-k 40), KL mean %.4f (median %.5f)",
               h.rank, h.rowsMean, 100 * Double(h.touchedBytes) / Double(max(h.denseBytes, 1)), h.heldOutStates, 100 * h.agreement, 100 * h.sampledChange, h.kl, h.klMedian)
    }

    /// `mlxr sieve MODEL`: fits the model's Sieve sidecar, or shows or removes it.
    static func sieveCommand() -> Command {
        Command(name: "sieve", summary: "Fit a model's Sieve sidecar (faster decode, changed output)", args: [Arg("MODEL", kind: .model)],
                options: [
                    Opt(long: "info", help: "show the sidecar's measurements"),
                    Opt(long: "remove", help: "delete the sidecar"),
                    Opt(long: "text", value: .text, repeatable: true, help: "calibration text file (default: the model writes its own)"),
                    Opt(long: "tokens", value: .int(2000...200_000), defaultText: "12000", help: "calibration states to collect"),
                    Opt(long: "rows", value: .int(64...65536), help: "rows that get the head's own logit per token (default: fitted)"),
                    Opt(long: "rank", value: .int(64...8192), help: "rank of the head sketch (default: an eighth of the hidden size)"),
                ],
                examples: ["  mlxr sieve mlx-community/Qwen3-0.6B-4bit", "  mlxr run mlx-community/Qwen3-0.6B-4bit --sieve balanced -p \"Hello\""],
                group: "Developer", run: { inv, console in
                    let dir = try installedModelDirectory(inv.positionals[0], resolver: .live)
                    let url = dir.appendingPathComponent("sieve").appendingPathComponent("head.safetensors")
                    if inv.bool("remove") {
                        guard FileManager.default.fileExists(atPath: url.path) else { console.out("no Sieve sidecar\n"); return }
                        try FileManager.default.removeItem(at: url.deletingLastPathComponent())
                        console.out("removed \(url.path)\n"); return
                    }
                    if inv.bool("info") {
                        guard let h = SieveImporter.sidecarInfo(directory: dir) else { console.out("no Sieve sidecar (run: mlxr sieve \(inv.positionals[0]))\n"); return }
                        console.out(sieveLine(h) + "\ncalibration: \(h.calibration)\n"); return
                    }
                    var o = SieveImportOptions()
                    o.textFiles = inv.strings("text").map { URL(fileURLWithPath: $0) }
                    if let n = inv.int("tokens") { o.tokens = n }
                    o.rank = inv.int("rank")
                    o.rows = inv.int("rows")
                    let report: SieveImportReport
                    do { report = try SieveImporter.importHead(directory: dir, options: o) { console.err($0 + "\n") } } catch {
                        throw CLIError.failure((error as? LocalizedError)?.errorDescription ?? "\(error)")
                    }
                    for b in report.budgets {
                        console.out(String(format: "  %5d rows: margin %6.2f, dense argmax ranked below them in %d of %d held-out states\n",
                                           b.rows, b.delta, b.missed, report.info.heldOutStates))
                    }
                    for t in report.tried.dropLast() {
                        console.out(String(format: "rank %d misses the target (argmax kept in %.2f%%, %.2f%% of sampled tokens change); trying a higher rank\n", t.rank, 100 * t.agreement, 100 * t.sampledChange))
                    }
                    console.out(sieveLine(report.info) + "\n")
                    if !report.info.meetsTarget {
                        console.out(String(format: "This misses the target (argmax kept in %.1f%%, at most %.0f%% of sampled tokens changed): expect a visible change in output.\n",
                                           100 * SieveHeadInfo.targetAgreement, 100 * SieveHeadInfo.targetSampledChange))
                    }
                    console.out(String(format: "most rows in one step: %d. Sidecar: %@ (%@), fitted in %.0f s\n", report.rowsMax,
                                       url.path as NSString, formatBytes(report.sidecarBytes) as NSString, report.seconds))
                    console.out("Use it with --sieve balanced (it changes output; without the option nothing changes).\n")
                })
    }
}
