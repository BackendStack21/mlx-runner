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
    static func upgradeCommand(updater: @escaping @Sendable () -> Updater, interrupts: Interrupts = .shared) -> Command {
        Command(name: "upgrade", aliases: ["update"], summary: "Update mlxr to the latest release",
                options: [Opt(long: "check", help: "only report whether a newer release exists")],
                run: { inv, console in try await executeUpgrade(inv, console, updater(), interrupts: interrupts) })
    }

    static func executeUpgrade(_ inv: Invocation, _ console: Console, _ updater: Updater,
                               interrupts: Interrupts = .shared) async throws {
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
        // The updater reports a percentage only, so the bar shows no sizes or speed.
        let bar = UpgradeBar(console: console)
        defer { bar.finish() }
        // Ctrl-C cancels the download (the updater removes its staging folder); once the install
        // starts it runs to the end, since a half-replaced binary is worse than a late exit. The
        // phase is entered before the task starts, so the task's `.writing` always comes after it.
        let canceller = TaskCanceller()
        // After the first Ctrl-C, further ones are held rather than forcing the exit: the updater removes
        // its staging folder (next to the binary) only when `apply` unwinds.
        interrupts.enter(.downloading, onCancel: { canceller.cancel(); interrupts.enter(.writing) })
        defer { interrupts.enter(.idle) }
        let task = Task {
            try await updater.apply(reuseCheck: true, progress: { phase in
                if case .download(let f) = phase { bar.update(f) }
                if phase == .install { interrupts.enter(.writing) }
            })
        }
        canceller.set { task.cancel() }
        let applied: AppliedUpdate
        do { applied = try await task.value } catch {
            if interrupts.wasInterrupted { throw CLIError.interrupted }
            throw error
        }
        console.out("upgraded mlxr \(current) -> \(applied.version)\n")
    }
}

/// Holds how to cancel a task that starts after the Ctrl-C phase meant to cancel it: a Ctrl-C that
/// arrives first cancels the task as soon as it is set.
final class TaskCanceller: @unchecked Sendable {
    private let lock = NSLock()
    private var action: (@Sendable () -> Void)?
    private var cancelled = false

    func set(_ cancel: @escaping @Sendable () -> Void) {
        let now = lock.withLock { () -> Bool in action = cancel; return cancelled }
        if now { cancel() }
    }

    func cancel() {
        let pending = lock.withLock { () -> (@Sendable () -> Void)? in cancelled = true; return action }
        pending?()
    }
}

/// A percentage-only progress bar on stderr, drawn only when stderr is a terminal.
final class UpgradeBar: @unchecked Sendable {
    private let console: Console
    private let lock = NSLock()
    private var drawn = false

    init(console: Console) { self.console = console }

    static func line(_ fraction: Double, width: Int = 10) -> String {
        let f = fraction.isFinite ? min(max(fraction, 0), 1) : 0
        let filled = Int(Double(width) * f)
        return "upgrading mlxr: " + String(format: "%3.0f%%", f * 100) + " \u{2595}"
            + String(repeating: "\u{2588}", count: filled) + String(repeating: " ", count: width - filled) + "\u{258F}"
    }

    func update(_ fraction: Double) {
        guard console.stderrIsTTY else { return }
        lock.withLock {
            console.err("\r" + Self.line(fraction) + "\u{1B}[K")
            drawn = true
        }
    }

    func finish() {
        lock.withLock {
            if drawn { console.err("\n"); drawn = false }
        }
    }
}
