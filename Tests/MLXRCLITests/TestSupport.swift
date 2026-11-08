import Foundation
@testable import MLXRCLI

/// Collects what a command writes, per stream.
final class Capture: @unchecked Sendable {
    private let lock = NSLock()
    private var o = "", e = ""
    var stdout: String { lock.withLock { o } }
    var stderr: String { lock.withLock { e } }

    func console(stdoutTTY: Bool = false, stderrTTY: Bool = false, stdinTTY: Bool = false,
                 width: Int = 80, env: [String: String] = [:]) -> Console {
        Console(out: { s in self.lock.withLock { self.o += s } },
                err: { s in self.lock.withLock { self.e += s } },
                stdoutIsTTY: stdoutTTY, stderrIsTTY: stderrTTY, stdinIsTTY: stdinTTY, width: width, env: env)
    }
}

/// Records which commands ran (and with what), so tests can prove help and errors never execute one.
final class RunLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    var ran: [String] { lock.withLock { items } }
    func add(_ s: String) { lock.withLock { items.append(s) } }
}

/// A small table exercising the grammar independent of the real commands.
func miniTable(_ log: RunLog = RunLog()) -> [Command] {
    func rec(_ n: String) -> (Invocation, Console) async throws -> Void { { _, _ in log.add(n) } }
    return [
        Command(name: "run", summary: "Run a model",
                args: [Arg("MODEL", kind: .model), Arg("PROMPT", optional: true, variadic: true)],
                options: [
                    Opt(long: "temp", value: .real(0...2), help: "temperature"),
                    Opt(long: "prompt", short: "p", value: .text, help: "prompt"),
                    Opt(long: "think", value: .choice(["true", "false", "high", "medium", "low"]),
                        implicit: "true", help: "think"),
                    Opt(long: "verbose", help: "stats"),
                    Opt(long: "count", value: .int(1...10), help: "count"),
                ], run: rec("run")),
        Command(name: "list", aliases: ["ls"], summary: "List models", run: rec("list")),
        Command(name: "download", aliases: ["pull"], summary: "Download a model",
                args: [Arg("MODEL", kind: .model)], run: rec("download")),
        Command(name: "bench", summary: "Benchmark", args: [Arg("MODEL", kind: .model)],
                group: "Developer", run: rec("bench")),
    ]
}
