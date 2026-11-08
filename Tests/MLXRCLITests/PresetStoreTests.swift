import Foundation
import Testing
@testable import MLXRCLI

@Suite("Preset store (AC-12)")
struct PresetStoreTests {
    func store() -> (PresetStore, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-presets-\(UUID().uuidString)", isDirectory: true)
        return (PresetStore(directory: dir), dir)
    }

    @Test func saveLoadListDeleteRoundTrip() throws {
        let (s, dir) = store()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(s.load("mychat") == nil)
        #expect(s.list().isEmpty)
        let p = Preset(name: "mychat", repo: "acme/X-4bit", system: "be brief", parameters: ["temperature": "0.2", "seed": "7"])
        try s.save(p)
        try s.save(Preset(name: "alpha", repo: "acme/Y-4bit"))
        #expect(s.load("mychat") == p)
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("mychat.json").path), "one <name>.json file per preset")
        #expect(s.list().map(\.name) == ["alpha", "mychat"], "sorted by name")
        try s.save(Preset(name: "mychat", repo: "acme/Z-4bit"))
        #expect(s.load("mychat")?.repo == "acme/Z-4bit", "saving again replaces")
        try s.delete("mychat")
        #expect(s.load("mychat") == nil)
        #expect(s.list().map(\.name) == ["alpha"])
    }

    @Test func nameValidation() {
        for ok in ["mychat", "qwen3-tuned:v2", "a.b_c", "A1"] {
            #expect(failureMessage { try PresetStore.validateName(ok) } == nil, "\(ok)")
        }
        for bad in ["", "../x", "a/b", ".hidden", "a b", "x\0y", "back\\slash", String(repeating: "a", count: 129)] {
            #expect(failureMessage { try PresetStore.validateName(bad) } != nil, "\(bad.debugDescription) accepted")
        }
    }

    @Test func saveRejectsAnUnsafeNameWithoutWritingAnything() throws {
        let (s, dir) = store()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(failureMessage { try s.save(Preset(name: "../evil", repo: "a/b")) } != nil)
        #expect(!FileManager.default.fileExists(atPath: dir.deletingLastPathComponent().appendingPathComponent("evil.json").path))
        #expect(s.list().isEmpty)
        #expect(s.load("../evil") == nil)
    }

    @Test func aCorruptPresetIsMovedAsideNotOverwrittenOrFatal() throws {
        let (s, dir) = store()
        defer { try? FileManager.default.removeItem(at: dir) }
        try s.save(Preset(name: "good", repo: "a/b"))
        try Data("{ not json".utf8).write(to: dir.appendingPathComponent("bad.json"))
        #expect(s.load("bad") == nil)
        #expect(s.list().map(\.name) == ["good"])
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        #expect(files.contains { $0.hasPrefix("bad.corrupt-") }, "\(files)")
        #expect(!files.contains("bad.json"))
    }
}
