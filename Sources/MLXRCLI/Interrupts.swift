import Foundation

/// What the process is doing when Ctrl-C arrives.
public enum InterruptPhase: Equatable, Sendable { case idle, loading, generating, downloading, serving }

/// What an interrupt asks for.
public enum InterruptAction: Equatable, Sendable {
    /// Nothing is running that an interrupt can stop.
    case ignore
    /// Leave the process now with this code (loading: 130, serving: 0).
    case exit(Int32)
    /// Cancel the running generation; the command returns by itself (exit 130).
    case cancel
    /// Cancel the running download, keep the partial files; the command returns by itself (exit 130).
    case cancelAndExit(Int32)
    /// A second interrupt while a generation is still being cancelled: leave the process at once.
    case forceExit(Int32)
}

/// The Ctrl-C state machine: loading -> exit 130; generating -> cancel, and a second interrupt
/// -> forced exit 130; downloading -> cancel and exit 130; serving -> exit 0.
public struct InterruptMachine: Sendable {
    public private(set) var phase: InterruptPhase = .idle
    public private(set) var count = 0

    public init() {}

    /// Starts a phase; the interrupt count starts again.
    public mutating func enter(_ phase: InterruptPhase) { self.phase = phase; count = 0 }

    public mutating func interrupt() -> InterruptAction {
        count += 1
        switch phase {
        case .idle: return .ignore
        case .loading: return .exit(130)
        case .generating: return count == 1 ? .cancel : .forceExit(130)
        case .downloading: return .cancelAndExit(130)
        case .serving: return .exit(0)
        }
    }
}

/// Routes Ctrl-C to the running phase. The live `SIGINT` source (Signals.swift) calls `fire()`;
/// tests call it directly. `exit` is `_exit` in the real process.
public final class Interrupts: @unchecked Sendable {
    public static let shared = Interrupts()

    private let lock = NSLock()
    private var machine = InterruptMachine()
    private var cancelAction: @Sendable () -> Void = {}
    private var interrupted = false
    private let exitProcess: @Sendable (Int32) -> Void

    public init(exit: @escaping @Sendable (Int32) -> Void = { _exit($0) }) { exitProcess = exit }

    /// Enters a phase. `onCancel` is what `.cancel` / `.cancelAndExit` do (stop the generation, the download).
    public func enter(_ phase: InterruptPhase, onCancel: @escaping @Sendable () -> Void = {}) {
        lock.lock(); defer { lock.unlock() }
        machine.enter(phase)
        cancelAction = onCancel
    }

    /// An interrupt arrived: applies the state machine (calls `onCancel` or `exit`).
    public func fire() {
        lock.lock()
        let action = machine.interrupt()
        let cancel = cancelAction
        if action != .ignore { interrupted = true }
        lock.unlock()
        switch action {
        case .ignore: break
        case .exit(let c), .forceExit(let c): exitProcess(c)
        case .cancel, .cancelAndExit: cancel()
        }
    }

    /// True once any interrupt arrived.
    public var wasInterrupted: Bool { lock.lock(); defer { lock.unlock() }; return interrupted }
}
