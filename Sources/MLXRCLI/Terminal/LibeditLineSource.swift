import EditLine.readline
import Foundation

/// Line editing through the system libedit (readline-compatible API): arrows, Ctrl-A/E/K/U/W, history.
/// Used only when stdin and stdout are terminals.
///
/// Ctrl-C at the prompt: the process ignores SIGINT and routes it through a dispatch source
/// (Signals.swift), and libedit puts that disposition back while it reads, so no signal handler
/// can interrupt a blocked `readline()`. Instead libedit is fed one character at a time through its
/// callback interface, from a `poll` on stdin and a wake pipe that `interrupter` writes to.
final class LibeditLineSource: LineSource {
    // The libedit line callback is a C function: it cannot capture, so the line it completes lands here.
    // Only `read` (one chat loop) touches these, on the thread that runs it.
    nonisolated(unsafe) private static var completed = false
    nonisolated(unsafe) private static var completedLine: String?

    private let console: Console
    private let wake = WakePipe()

    init(history: [String], console: Console) {
        self.console = console
        setlocale(LC_CTYPE, "")
        for line in history { add_history(line) }
    }

    var interrupter: (@Sendable () -> Void)? {
        guard wake.isOpen else { return nil }
        let wake = wake
        return { wake.signal() }
    }

    func read(prompt: String) -> LineEvent {
        wake.drain()   // a Ctrl-C from before this prompt was shown does not count
        Self.completed = false
        Self.completedLine = nil
        rl_callback_handler_install(prompt) { raw in
            LibeditLineSource.completed = true
            LibeditLineSource.completedLine = raw.map { String(cString: $0) }
            free(raw)   // libedit hands over a copy (strdup)
        }
        let outcome = feedUntilDone()
        rl_callback_handler_remove()   // back to the terminal's normal mode
        switch outcome {
        case .interrupted: return Self.event(line: nil, interrupted: true, console: console)
        case .closed: return .eof
        case .completed:
            guard let line = Self.completedLine else {
                // Ctrl-D on an empty line: libedit's callback mode draws `^D`; end that line.
                console.err("\n")
                return .eof
            }
            if !line.trimmingCharacters(in: .whitespaces).isEmpty { add_history(line) }
            return .line(line)
        }
    }

    private enum Outcome { case completed, interrupted, closed }

    /// `FIONREAD` (bytes waiting on a descriptor), a function-like macro Swift cannot import:
    /// `_IOR('f', 127, int)` = IOC_OUT | sizeof(int) << 16 | 'f' << 8 | 127.
    static let fionread = UInt(0x4000_0000) | UInt(MemoryLayout<Int32>.size) << 16 | UInt(UInt8(ascii: "f")) << 8 | 127

    /// Hands libedit each character as it arrives until it completes a line, Ctrl-C wakes the pipe,
    /// or the terminal goes away.
    private func feedUntilDone() -> Outcome {
        while !Self.completed {
            var fds = [pollfd(fd: STDIN_FILENO, events: Int16(POLLIN), revents: 0),
                       pollfd(fd: wake.readEnd, events: Int16(POLLIN), revents: 0)]
            if poll(&fds, 2, -1) < 0 {
                if errno == EINTR { continue }
                return .closed
            }
            if fds[1].revents != 0 { return .interrupted }
            if fds[0].revents & Int16(POLLHUP | POLLERR | POLLNVAL) != 0 { return .closed }
            if fds[0].revents & Int16(POLLIN) != 0 {
                // Readable with nothing to read (end of input without a hangup, a read error):
                // libedit would make no progress and this loop would spin.
                var available: Int32 = 0
                if ioctl(STDIN_FILENO, Self.fionread, &available) == 0, available == 0 { return .closed }
                rl_callback_read_char()
            }
        }
        return .completed
    }

    /// What a `readline` result means: a line, Ctrl-C (interrupted while blocked) or end of input.
    /// The newline after a Ctrl-C goes to stderr, never to stdout.
    static func event(line: String?, interrupted: Bool, console: Console) -> LineEvent {
        if let line { return .line(line) }
        if interrupted { console.err("\n"); return .interrupt }
        return .eof
    }
}

/// A self-pipe: `signal()` from any thread makes the read end readable until `drain()`.
final class WakePipe: @unchecked Sendable {
    let readEnd: Int32
    private let writeEnd: Int32
    private let lock = NSLock()
    private var lastSignal: TimeInterval = 0

    init() {
        var fds: [Int32] = [-1, -1]
        if pipe(&fds) == 0 {
            for fd in fds {
                _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)
                _ = fcntl(fd, F_SETFD, FD_CLOEXEC)
            }
        }
        readEnd = fds[0]
        writeEnd = fds[1]
    }

    deinit {
        if readEnd >= 0 { close(readEnd) }
        if writeEnd >= 0 { close(writeEnd) }
    }

    var isOpen: Bool { readEnd >= 0 && writeEnd >= 0 }

    /// One Ctrl-C at the libedit prompt reaches the SIGINT source twice (libedit re-raises it), so a
    /// signal within 200 ms of the previous one is the same Ctrl-C: it must not wake the next prompt.
    func signal() {
        let now = ProcessInfo.processInfo.systemUptime
        let repeated = lock.withLock { () -> Bool in
            defer { lastSignal = now }
            return now - lastSignal < 0.2
        }
        if repeated { return }
        var byte: UInt8 = 1
        _ = write(writeEnd, &byte, 1)   // non-blocking: a full pipe is already signalled
    }

    func drain() {
        var buffer = [UInt8](repeating: 0, count: 64)
        while readEnd >= 0, Darwin.read(readEnd, &buffer, buffer.count) > 0 {}
    }
}
