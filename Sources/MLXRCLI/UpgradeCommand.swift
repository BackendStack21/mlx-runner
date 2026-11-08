import Foundation
import MLXEngine

extension CLI {
    /// The updater `mlxr upgrade` uses: the running binary, replaced in place by `CLIInstaller`.
    public static func liveUpdater() -> Updater {
        let exe = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
        return Updater(install: .detectCLI(executable: exe), installer: CLIInstaller())
    }

    /// `upgrade` (alias `update`): installs the latest signed release over the running binary.
    /// The updater is built lazily, so no other command ever reaches the update code.
    static func upgradeCommand(updater: @escaping @Sendable () -> Updater) -> Command {
        Command(name: "upgrade", aliases: ["update"], summary: "Update mlxr to the latest release",
                options: [Opt(long: "check", help: "only report whether a newer release exists")],
                run: { inv, console in try await executeUpgrade(inv, console, updater()) })
    }

    static func executeUpgrade(_ inv: Invocation, _ console: Console, _ updater: Updater) async throws {
        // A source build is updated with git, whether or not a newer release exists.
        if updater.install.kind == .source {
            throw CLIError.failure("mlxr is running from a source build; update it with git pull")
        }
        let current = updater.currentVersion
        let info = try await updater.check(force: true)
        guard info.available, let latest = info.latest else {
            console.out("mlxr is up to date (\(current))\n")
            return
        }
        if inv.bool("check") {
            console.out("mlxr \(latest) is available (you have \(current))\n")
            return
        }
        let renderer = PullRenderer(console: console, label: "upgrading mlxr:")
        defer { renderer.finish() }
        let applied: AppliedUpdate
        do {
            applied = try await updater.apply(progress: { phase in
                if console.stderrIsTTY, case .download(let f) = phase {
                    renderer.update(done: Int64(min(max(f, 0), 1) * 1000), total: 1000, speed: 0)
                }
            })
        } catch { renderer.finish(); throw error }
        console.out("upgraded mlxr \(current) -> \(applied.version)\n")
    }
}
