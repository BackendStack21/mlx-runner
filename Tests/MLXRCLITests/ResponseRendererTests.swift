import Foundation
import Testing
@testable import MLXRCLI

@Suite("word wrap and thinking display (AC-15, AC-16)")
struct ResponseRendererTests {
    let esc = "\u{1B}"

    @Test func wrapErasesThePartialWordAndRepeatsItOnTheNextLine() {
        var w = WordWrapper(width: 20)
        // The limit is width - 5 = 15 columns: "hello wonderful" fills them, the space after it wraps.
        #expect(w.feed("hello wonderful world") == "hello wonderful\(esc)[9D\(esc)[K\nwonderful world")
    }

    @Test func shortTextPassesThroughAndNewlinesResetTheLine() {
        var w = WordWrapper(width: 20)
        #expect(w.feed("short\n") == "short\n")
        // Streaming in pieces gives the same result as one piece.
        var a = WordWrapper(width: 20), b = WordWrapper(width: 20)
        let whole = b.feed("hello wonderful world")
        let parts = ["hello wo", "nderful", " wor", "ld"].map { a.feed($0) }.joined()
        #expect(parts == whole)
    }

    func render(_ steps: [(Bool, String)], tty: Bool = false, hide: Bool = false, wrap: Bool = true, width: Int = 80) -> String {
        let cap = Capture()
        let r = ResponseRenderer(console: cap.console(stdoutTTY: tty, width: width), hideThinking: hide, wordWrap: wrap)
        for (isReasoning, text) in steps { isReasoning ? r.reasoning(text) : r.content(text) }
        return cap.stdout
    }

    @Test func plainThinkingBlock() {
        let out = render([(true, "plan"), (true, " it\n"), (false, "Answer")])
        #expect(out == "Thinking...\nplan it\n...done thinking.\n\nAnswer")
    }

    @Test func aNewlineIsAddedWhenTheReasoningDoesNotEndInOne() {
        #expect(render([(true, "plan"), (false, "Answer")]) == "Thinking...\nplan\n...done thinking.\n\nAnswer")
    }

    @Test func hiddenThinkingShowsOnlyTheAnswer() {
        #expect(render([(true, "plan\n"), (false, "Answer")], hide: true) == "Answer")
    }

    @Test func answerWithoutReasoningHasNoMarkers() {
        #expect(render([(false, "Just "), (false, "text")]) == "Just text")
    }

    @Test func thinkingIsDimOnATerminal() {
        let out = render([(true, "plan\n"), (false, "Answer")], tty: true)
        let open = "\(esc)[38;5;245m\(esc)[1mThinking...\n\(esc)[0m\(esc)[38;5;245m"
        let close = "\(esc)[38;5;245m\(esc)[1m...done thinking.\n\n\(esc)[0m"
        #expect(out == open + "plan\n" + close + "Answer")
    }

    @Test func wrapOnlyOnATerminalAndNotWithNoWordWrap() {
        let text = "hello wonderful world"
        #expect(render([(false, text)], tty: true, wrap: true, width: 20) == "hello wonderful\(esc)[9D\(esc)[K\nwonderful world")
        #expect(render([(false, text)], tty: true, wrap: false, width: 20) == text)
        #expect(render([(false, text)], tty: false, wrap: true, width: 20) == text)
    }
}
