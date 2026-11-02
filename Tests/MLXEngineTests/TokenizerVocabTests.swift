import Foundation
import Testing
@testable import MLXEngine

/// Swift's `String` equality is canonical equivalence, but a vocabulary is keyed by exact text:
/// Gemma has both ";" and U+037E (its canonical equivalent), each with its own id.
@Suite("Tokenizer vocabulary keys")
struct TokenizerVocabTests {
    /// Raw JSON: a Swift dictionary literal would itself collapse the equivalent keys.
    private func tokenizer() throws -> Tokenizer {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tok-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let json = #"""
        {"model": {"type": "BPE", "unk_token": "<unk>",
           "vocab": {"<unk>": 0, ";": 1, ";": 2, "Å": 3, "Å": 4, "Ω": 5, "Ω": 6,
                     "é": 7, "e": 8, "́": 9, "é": 10, "a": 11},
           "merges": [["e", "́"]]},
         "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never"},
         "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never"}}
        """#
        try json.data(using: .utf8)!.write(to: dir.appendingPathComponent("tokenizer.json"))
        return try Tokenizer(directory: dir)
    }

    @Test func canonicallyEquivalentTokensKeepTheirOwnIds() throws {
        let t = try tokenizer()
        let cases: [(String, Int)] = [(";", 1), ("\u{37E}", 2), ("\u{C5}", 3), ("\u{212B}", 4), ("\u{3A9}", 5), ("\u{2126}", 6), ("\u{E9}", 7)]
        for (text, id) in cases {
            #expect(t.encode(text) == [id], "U+\(text.unicodeScalars.map { String($0.value, radix: 16) })")
            #expect(t.bytes(for: id) == Array(text.utf8), "id \(id) must decode to its own bytes")
            #expect(t.tokenId(text) == id)
        }
    }

    @Test func mergesAndTheWordCacheAreExact() throws {
        let t = try tokenizer()
        // Decomposed é merges to its own token; precomposed é is another one, even once cached.
        #expect(t.encode("e\u{301}") == [10])
        #expect(t.encode("\u{E9}") == [7])
        #expect(t.encode("e\u{301}") == [10])
    }

    @Test(.enabled(if: installedModel(GemmaModelTests.gemma3) != nil))
    func gemmaSemicolonIsTheAsciiToken() throws {
        let t = try Tokenizer(directory: installedModel(GemmaModelTests.gemma3)!)
        #expect(t.encode(";").count == 1)
        let semi = try #require(t.tokenId(";"))
        let greek = try #require(t.tokenId("\u{37E}"))
        #expect(semi != greek)
        #expect(t.encode("a; b").contains(semi) && !t.encode("a; b").contains(greek))
        #expect(t.bytes(for: semi) == Array(";".utf8) && t.bytes(for: greek) == Array("\u{37E}".utf8))
    }
}
