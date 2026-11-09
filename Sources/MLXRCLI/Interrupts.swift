import Foundation

/// What the process is doing when Ctrl-C arrives. `idle` is where every command starts: a command
/// that never enters another phase still stops on Ctrl-C.
public enum InterruptPhase: Equatable, Sendable { case idle, prompt, loading, generating, downloading, writing, serving }

/// What an interrupt asks for.
public enum InterruptAction: Equatable, Sendable {
    /// Let the running step finish (an update being installed, a model being deleted); the
    /// interrupt is held until `finishWriting` (which then leaves) or the next phase (which drops it).
    case ignore
    /// Leave the process now with this code (idle, loading: 130; serving: 0).
    case exit(Int32)
    /// Cancel the running generation; the command returns by itself (exit 130).
    case cancel
    /// Cancel the running download, keep the partial files; the command returns by itself (exit 130).
    case cancelAndExit(Int32)
    /// A second interrupt while a generation or download is still being cancelled: leave the process at once.
    case forceExit(Int32)
    /// Wake the chat prompt: it shows the interrupt and the chat goes on (not an interruption of the command).
    case wake
}

/// The Ctrl-C state machine: idle and loading -> exit 130; prompt -> wake the line source;
/// generating -> cancel, and a second interrupt -> forced exit 130; downloading -> cancel and exit
/// 130, and a second interrupt -> forced exit 130; writing (files that must not be left half
/// changed) -> ignore; serving -> exit 0.
public struct InterruptMachine: Sendable {
    public private(set) var phase: InterruptPhase = .idle
    public private(set) var count = 0

    public init() {}

    /// Starts a phase; the interrupt count starts again.
    public mutating func enter(_ phase: InterruptPhase) { self.phase = phase; count = 0 }

    public mutating func interrupt() -> InterruptAction {
        count += 1
        switch phase {
        case .idle, .loading: return .exit(130)
        case .prompt: return .wake
        case .generating: return count == 1 ? .cancel : .forceExit(130)
        case .downloading: return count == 1 ? .cancelAndExit(130) : .forceExit(130)
        case .writing: return .ignore
        case .serving: return .exit(0)
        }
    }
}

/// Set by the cancel action of a phase: true only when the user interrupted (the engine also
/// cancels its token whenever a stream ends, so the token cannot tell).
public final class UserCancel: @unchecked Sendable {
    private let lock = NSLock()
    private var set = false
    public init() {}
    public func mark() { lock.lock(); set = true; lock.unlock() }
    public var isSet: Bool { lock.lock(); defer { lock.unlock() }; return set }
}

/// Routes Ctrl-C to the running phase. The live `SIGINT` source (Signals.swift) calls `fire()`;
/// tests call it directly. `exit` is `_exit` in the real process.
public final class Interrupts: @unchecked Sendable {
    public static let shared = Interrupts()

    private let lock = NSLock()
    private var machine = InterruptMachine()
    private var cancelAction: @Sendable () -> Void = {}
    private var interrupted = false
    /// A Ctrl-C held while `.writing`.
    private var held = false
    private let exitProcess: @Sendable (Int32) -> Void

    public init(exit: @escaping @Sendable (Int32) -> Void = { _exit($0) }) { exitProcess = exit }

    /// Enters a phase. `onCancel` is what `.cancel` / `.cancelAndExit` / `.wake` do (stop the
    /// generation or the download, wake the prompt).
    public func enter(_ phase: InterruptPhase, onCancel: @escaping @Sendable () -> Void = {}) {
        lock.lock(); defer { lock.unlock() }
        machine.enter(phase)
        cancelAction = onCancel
        held = false
    }

    /// Ends a `.writing` step: back to `.idle`, and a Ctrl-C held during the step now leaves (130).
    /// Returns whether it did, so a command whose exit is faked in tests stops anyway.
    @discardableResult
    public func finishWriting() -> Bool {
        lock.lock()
        let pending = held
        machine.enter(.idle)
        cancelAction = {}
        held = false
        if pending { interrupted = true }
        lock.unlock()
        if pending { exitProcess(130) }
        return pending
    }

    /// The current phase.
    public var phase: InterruptPhase { lock.lock(); defer { lock.unlock() }; return machine.phase }

    /// An interrupt arrived: applies the state machine (calls `onCancel` or `exit`).
    public func fire() {
        lock.lock()
        let action = machine.interrupt()
        let cancel = cancelAction
        if action != .ignore, action != .wake { interrupted = true }
        if action == .ignore { held = true }
        lock.unlock()
        switch action {
        case .ignore: break
        case .exit(let c), .forceExit(let c): exitProcess(c)
        case .cancel, .cancelAndExit, .wake: cancel()
        }
    }

    /// True once an interrupt stopped (or began stopping) something: Ctrl-C at the chat prompt does not count.
    public var wasInterrupted: Bool { lock.lock(); defer { lock.unlock() }; return interrupted }
}
