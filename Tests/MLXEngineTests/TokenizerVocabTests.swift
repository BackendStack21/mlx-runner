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

    /// JSONSerialization drops a U+FEFF that starts a string (raw or escaped). Gemma lists
    /// "\u{FEFF}\r" next to "\r", so the BOM entry took over every carriage return.
    @Test func entriesStartingWithBOMKeepTheirOwnIds() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tok-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // A file BOM, a raw U+FEFF entry, an escaped one, an escaped backslash before "ufeff" (text,
        // not an escape), and a merge whose left side starts with U+FEFF.
        let json = (#"""
        @BOM@{"model": {"type": "BPE", "vocab": {"\r": 0, "@BOM@\r": 1, "x": 2, "@ESC@x": 3,
                   "\\ufeff": 4, "\\": 5, "u": 6, "f": 7, "e": 8, "a": 9, "@ESC@xa": 10, "@MIX@y": 11, "a@BOM@b": 12, "@PUA@": 13},
                   "merges": [["@ESC@", "x"], ["@ESC@x", "a"]]},
         "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never"}}
        """#).replacingOccurrences(of: "@BOM@", with: "\u{FEFF}").replacingOccurrences(of: "@ESC@", with: "\\u" + "feff")
            .replacingOccurrences(of: "@MIX@", with: "\\u" + "FeFf").replacingOccurrences(of: "@PUA@", with: "\\u" + "e000")
        try json.data(using: .utf8)!.write(to: dir.appendingPathComponent("tokenizer.json"))
        let t = try Tokenizer(directory: dir)
        #expect(t.tokenId("\r") == 0 && t.tokenId("\u{FEFF}\r") == 1)
        #expect(t.tokenId("x") == 2 && t.tokenId("\u{FEFF}x") == 3)
        #expect(t.tokenId("\\ufeff") == 4)
        #expect(t.encode("\r") == [0])
        #expect(t.tokenId("\u{FEFF}xa") == 10)
        #expect(t.encode("\u{FEFF}xa") == [10], "the merge kept its U+FEFF")
        #expect(t.tokenId("\u{FEFF}y") == 11, "a mixed-case escape")
        #expect(t.tokenId("a\u{FEFF}b") == 12, "U+FEFF inside a string is left as it is")
        #expect(t.tokenId("\u{E000}") == 13, "the stand-in is one the file does not use")
        #expect(Tokenizer.shieldingBOM(try Data(contentsOf: dir.appendingPathComponent("tokenizer.json"))).stand == Unicode.Scalar(0xE001))
        // A file without U+FEFF is read unchanged.
        let plain = Data(#"{"a": "b"}"#.utf8)
        #expect(Tokenizer.shieldingBOM(plain).stand == nil && Tokenizer.shieldingBOM(plain).data == plain)
    }

    /// Hugging Face merges one pair at a time, the lowest rank first and leftmost among equals, so a
    /// pair a merge creates competes at once. Merging every occurrence of the best pair per pass
    /// differs when the merge list is not in training order (Gemma's whitespace runs).
    @Test func mergesFollowHuggingFaceOrder() {
        let ranks: [String: Int] = ["a\u{0}a": 2, "aa\u{0}a": 1]
        let merged = Tokenizer.mergePairs(["a", "a", "a", "a"]) { ranks[$0 + "\u{0}" + $1] }
        #expect(merged == ["aaa", "a"])   // every-occurrence merging gives ["aa", "aa"]
        let usual: [String: Int] = ["a\u{0}b": 0, "ab\u{0}c": 1, "b\u{0}c": 2]
        #expect(Tokenizer.mergePairs(["a", "b", "c", "a", "b"]) { usual[$0 + "\u{0}" + $1] } == ["abc", "ab"])
        #expect(Tokenizer.mergePairs(["z"]) { _, _ in 0 } == ["z"])
    }

    /// Gemma 1.1's carriage return and Gemma 3's long space runs, against the reference tokenizer.
    @Test(.enabled(if: installedModel(GemmaModelTests.gemma3) != nil || installedModel(GemmaModelTests.gemma1) != nil))
    func gemmaCarriageReturnsAndSpaceRunsMatchTheReference() throws {
        if let dir = installedModel(GemmaModelTests.gemma1) {
            let t = try Tokenizer(directory: dir)
            #expect(t.encode("a\r\nb") == [235250, 235316, 108, 235268])
        }
        if let dir = installedModel(GemmaModelTests.gemma3) {
            let t = try Tokenizer(directory: dir)
            #expect(t.encode("x" + String(repeating: " ", count: 32) + "LONDON") == [236781, 167, 109372])
            #expect(t.encode("x" + String(repeating: " ", count: 40) + "y") == [236781, 167, 145, 236762])
            #expect(t.encode("a\rb") == [236746, 251, 236763])
        }
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
