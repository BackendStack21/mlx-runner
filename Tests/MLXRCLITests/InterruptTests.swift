import Foundation
import Testing
@testable import MLXRCLI

@Suite("Ctrl-C state machine (AC-17)")
struct InterruptTests {
    @Test func eachPhaseMapsToItsAction() {
        var m = InterruptMachine()
        m.enter(.loading)
        #expect(m.interrupt() == .exit(130))
        m.enter(.generating)
        #expect(m.interrupt() == .cancel)
        m.enter(.downloading)
        #expect(m.interrupt() == .cancelAndExit(130))
        m.enter(.serving)
        #expect(m.interrupt() == .exit(0))
        m.enter(.idle)
        #expect(m.interrupt() == .ignore)
    }

    @Test func aSecondInterruptWhileGeneratingForcesTheExit() {
        var m = InterruptMachine()
        m.enter(.generating)
        #expect(m.interrupt() == .cancel)
        #expect(m.interrupt() == .forceExit(130))
        #expect(m.interrupt() == .forceExit(130))
    }

    @Test func enteringAPhaseStartsTheCountAgain() {
        var m = InterruptMachine()
        m.enter(.generating)
        #expect(m.interrupt() == .cancel)
        m.enter(.generating)
        #expect(m.interrupt() == .cancel)
        #expect(m.phase == .generating)
    }

    func router() -> (Interrupts, ExitLog, CancelCount) {
        let log = ExitLog()
        return (Interrupts(exit: { log.add($0) }), log, CancelCount())
    }

    @Test func loadingExitsWith130() {
        let (r, log, cancels) = router()
        r.enter(.loading, onCancel: { cancels.hit() })
        #expect(!r.wasInterrupted)
        r.fire()
        #expect(log.codes == [130])
        #expect(cancels.count == 0)
        #expect(r.wasInterrupted)
    }

    @Test func generatingCancelsOnceThenForcesTheExit() {
        let (r, log, cancels) = router()
        r.enter(.generating, onCancel: { cancels.hit() })
        r.fire()
        #expect(cancels.count == 1)
        #expect(log.codes.isEmpty)
        #expect(r.wasInterrupted)
        r.fire()
        #expect(log.codes == [130])
        #expect(cancels.count == 1)
    }

    @Test func downloadingCancelsAndLeavesTheExitToTheCommand() {
        let (r, log, cancels) = router()
        r.enter(.downloading, onCancel: { cancels.hit() })
        r.fire()
        #expect(cancels.count == 1)
        #expect(log.codes.isEmpty)
        #expect(r.wasInterrupted)
    }

    @Test func servingExitsWith0() {
        let (r, log, _) = router()
        r.enter(.serving)
        r.fire()
        #expect(log.codes == [0])
    }

    @Test func idleIgnoresInterrupts() {
        let (r, log, cancels) = router()
        r.enter(.idle, onCancel: { cancels.hit() })
        r.fire()
        #expect(log.codes.isEmpty && cancels.count == 0)
    }

    @Test func aNewPhaseReplacesTheCancelAction() {
        let (r, _, cancels) = router()
        let other = CancelCount()
        r.enter(.generating, onCancel: { cancels.hit() })
        r.enter(.generating, onCancel: { other.hit() })
        r.fire()
        #expect(cancels.count == 0 && other.count == 1)
    }
}
