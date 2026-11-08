import EditLine.readline
import Foundation

/// Line editing through the system libedit (readline-compatible API): arrows, Ctrl-A/E/K/U/W, history.
/// Used only when stdin and stdout are terminals.
final class LibeditLineSource: LineSource {
    nonisolated(unsafe) private static var sawInterrupt: sig_atomic_t = 0

    init(history: [String]) {
        setlocale(LC_CTYPE, "")
        for line in history { add_history(line) }
    }

    func read(prompt: String) -> LineEvent {
        // A SIGINT handler without SA_RESTART makes the blocked read return so Ctrl-C reaches the loop.
        Self.sawInterrupt = 0
        var action = sigaction()
        var previous = sigaction()
        action.__sigaction_u.__sa_handler = { _ in LibeditLineSource.sawInterrupt = 1 }
        action.sa_flags = 0
        sigemptyset(&action.sa_mask)
        sigaction(SIGINT, &action, &previous)
        defer { sigaction(SIGINT, &previous, nil) }

        guard let raw = readline(prompt) else {
            if Self.sawInterrupt != 0 { print(""); return .interrupt }
            return .eof
        }
        defer { free(raw) }
        let line = String(cString: raw)
        if !line.trimmingCharacters(in: .whitespaces).isEmpty { add_history(raw) }
        return .line(line)
    }
}
