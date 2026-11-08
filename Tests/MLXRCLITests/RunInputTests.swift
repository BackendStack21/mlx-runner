import Foundation
import Testing
@testable import MLXRCLI

@Suite("run input assembly (AC-14)")
struct RunInputTests {
    func assemble(stdin: String = "", words: [String] = [], p: String? = nil, stdinTTY: Bool = false, stdoutTTY: Bool = false) throws -> RunInput {
        try CLI.assembleRunInput(stdin: stdin, words: words, promptFlag: p, stdinIsTTY: stdinTTY, stdoutIsTTY: stdoutTTY)
    }

    @Test func pipedInputGoesBeforeThePromptWords() throws {
        #expect(try assemble(stdin: "A", words: ["b", "c"]) == .oneShot("A b c"))
    }

    @Test func emptyInputIsIgnored() throws {
        #expect(try assemble(stdin: "", words: ["b", "c"]) == .oneShot("b c"))
    }

    @Test func noPromptWithTerminalsOnBothEndsIsChat() throws {
        #expect(try assemble(stdinTTY: true, stdoutTTY: true) == .chat)
    }

    @Test func noPromptWithoutBothTerminalsIsAnEmptyOneShot() throws {
        #expect(try assemble(stdinTTY: true, stdoutTTY: false) == .oneShot(""))
        #expect(try assemble(stdinTTY: false, stdoutTTY: true) == .oneShot(""))
        #expect(try assemble(stdinTTY: false, stdoutTTY: false) == .oneShot(""))
    }

    @Test func promptFlagMakesItOneShotEvenOnTerminals() throws {
        #expect(try assemble(p: "hello", stdinTTY: true, stdoutTTY: true) == .oneShot("hello"))
    }

    @Test func wordsMakeItOneShotEvenOnTerminals() throws {
        #expect(try assemble(words: ["hi"], stdinTTY: true, stdoutTTY: true) == .oneShot("hi"))
    }

    @Test func pipedInputGoesBeforeThePromptFlag() throws {
        #expect(try assemble(stdin: "A", p: "b") == .oneShot("A b"))
    }

    @Test func pipedInputAloneIsTheOneShotPrompt() throws {
        #expect(try assemble(stdin: "just this", stdinTTY: false, stdoutTTY: true) == .oneShot("just this"))
    }

    @Test func wordsAndPromptFlagTogetherAreAnError() {
        let m = failureMessage { try self.assemble(words: ["b"], p: "x") }
        #expect(m?.contains("-p") == true, "\(String(describing: m))")
    }
}
