import Foundation

/// Installs the process-wide Ctrl-C handler: `SIGINT` is ignored by default delivery and routed
/// to `Interrupts.shared` through a dispatch source.
enum Signals {
    nonisolated(unsafe) private static var source: DispatchSourceSignal?

    static func install() {
        signal(SIGINT, SIG_IGN)
        let s = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        s.setEventHandler { Interrupts.shared.fire() }
        s.resume()
        source = s
    }
}
