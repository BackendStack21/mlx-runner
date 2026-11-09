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
        #expect(m.interrupt() == .exit(130))
        m.enter(.prompt)
        #expect(m.interrupt() == .wake)
        m.enter(.writing)
        #expect(m.interrupt() == .ignore)
    }

    @Test func aCommandWithoutAPhaseOfItsOwnStopsOnCtrlC() {
        var m = InterruptMachine()
        #expect(m.phase == .idle)
        #expect(m.interrupt() == .exit(130))
    }

    @Test func aSecondInterruptWhileDownloadingForcesTheExit() {
        var m = InterruptMachine()
        m.enter(.downloading)
        #expect(m.interrupt() == .cancelAndExit(130))
        #expect(m.interrupt() == .forceExit(130))
    }

    @Test func everyCtrlCAtThePromptWakesIt() {
        var m = InterruptMachine()
        m.enter(.prompt)
        #expect(m.interrupt() == .wake)
        #expect(m.interrupt() == .wake)
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

    @Test func idleExits130() {
        let (r, log, cancels) = router()
        r.fire()
        #expect(log.codes == [130], "a command that never entered a phase still stops")
        r.enter(.idle, onCancel: { cancels.hit() })
        r.fire()
        #expect(log.codes == [130, 130] && cancels.count == 0)
        #expect(r.wasInterrupted)
    }

    @Test func thePromptIsWokenAndCtrlCThereIsNotAnInterruption() {
        let (r, log, wakes) = router()
        r.enter(.prompt, onCancel: { wakes.hit() })
        r.fire()
        r.fire()
        #expect(wakes.count == 2)
        #expect(log.codes.isEmpty)
        #expect(!r.wasInterrupted, "a later load must not read it as a Ctrl-C")
    }

    @Test func aDownloadThatDoesNotStopIsLeftOnTheSecondCtrlC() {
        let (r, log, cancels) = router()
        r.enter(.downloading, onCancel: { cancels.hit() })
        r.fire()
        r.fire()
        #expect(cancels.count == 1)
        #expect(log.codes == [130])
    }

    @Test func writingIsNeverCutShort() {
        let (r, log, cancels) = router()
        r.enter(.writing, onCancel: { cancels.hit() })
        r.fire()
        r.fire()
        #expect(log.codes.isEmpty && cancels.count == 0)
        #expect(!r.wasInterrupted)
        // Held until the step ends, then it leaves.
        #expect(r.finishWriting())
        #expect(log.codes == [130] && r.wasInterrupted && r.phase == .idle)
    }

    @Test func aWriteWithoutCtrlCFinishesQuietly() {
        let (r, log, _) = router()
        r.enter(.writing)
        #expect(!r.finishWriting())
        #expect(log.codes.isEmpty && !r.wasInterrupted && r.phase == .idle)
    }

    @Test func aNewPhaseDropsAHeldCtrlC() {
        let (r, log, _) = router()
        r.enter(.writing)
        r.fire()
        r.enter(.idle)   // e.g. an upgrade that finished its install
        #expect(!r.finishWriting())
        #expect(log.codes.isEmpty)
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
