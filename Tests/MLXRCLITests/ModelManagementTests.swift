import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("rm and cp (AC-12)")
struct ModelManagementTests {
    static let hf = "someone/Cached-4bit"

    func fixture() -> LibraryFixture {
        LibraryFixture(models: [
            LibraryFixture.model(LibraryFixture.qwen8, bytes: 5_200_000_000, ageSeconds: 60),
            LibraryFixture.model(Self.hf, bytes: 1_000_000_000, ageSeconds: 60, source: .huggingFaceCache),
        ])
    }

    @Test func rmDeletesAndPrintsDeleted() async {
        let f = fixture()
        let r = await f.run(["rm", LibraryFixture.qwen8])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "deleted '\(LibraryFixture.qwen8)'\n")
        #expect(f.deleted == [LibraryFixture.qwen8])
    }

    @Test func rmTakesSeveralModelsByAliasOrShortName() async {
        let f = fixture()
        let r = await f.run(["rm", "qwen3:8b", "cached-4bit", "-y"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "deleted 'qwen3:8b'\ndeleted 'cached-4bit'\n")
        #expect(Set(f.deleted) == [LibraryFixture.qwen8, Self.hf])
    }

    @Test func rmOnlyDeletesInstalledModels() async {
        let f = fixture()
        let r = await f.run(["rm", "qwen3:4b"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: model 'qwen3:4b' not found"), "\(r.err)")
        let mixed = await f.run(["rm", LibraryFixture.qwen8, "nope"])
        #expect(mixed.code == 1)
        #expect(f.deleted.isEmpty, "nothing is deleted when any name is unknown")
        let none = await f.run(["rm"])
        #expect(none.err == "Error: requires at least 1 arg(s), only received 0\n")
    }

    @Test func rmRefusesAnAmbiguousShortName() async {
        let f = LibraryFixture(models: [
            LibraryFixture.model("a/Same-4bit", bytes: 1, ageSeconds: 1),
            LibraryFixture.model("b/Same-4bit", bytes: 1, ageSeconds: 1),
        ])
        let r = await f.run(["rm", "Same-4bit"])
        #expect(r.code == 1)
        #expect(r.err.hasPrefix("Error: ") && r.err.contains("ambiguous") && r.err.contains("a/Same-4bit") && r.err.contains("b/Same-4bit"), "\(r.err)")
        #expect(f.deleted.isEmpty)
    }

    @Test func aTerminalAsksBeforeDeletingFromTheSharedHuggingFaceCache() async {
        let confirmed = fixture()
        let feed = LineFeed(["y"])
        let ok = await confirmed.run(["rm", Self.hf], tty: true, answers: feed)
        #expect(ok.code == 0, "\(ok.err)")
        #expect(ok.err.contains("Hugging Face") && ok.err.contains("[y/N]"), "\(ok.err)")
        #expect(ok.out == "deleted '\(Self.hf)'\n")
        #expect(confirmed.deleted == [Self.hf])

        let yes = fixture()
        #expect(await yes.run(["rm", Self.hf], tty: true, answers: LineFeed(["YES"])).code == 0)
        #expect(yes.deleted == [Self.hf])

        for answer in ["n", "", "maybe"] {
            let declined = fixture()
            let r = await declined.run(["rm", Self.hf], tty: true, answers: LineFeed([answer]))
            #expect(r.code == 1, "\(answer)")
            #expect(r.err.contains("aborted"), "\(r.err)")
            #expect(declined.deleted.isEmpty, "\(answer) must not delete")
            #expect(!r.out.contains("deleted"))
        }
        let eof = fixture()
        #expect(await eof.run(["rm", Self.hf], tty: true, answers: LineFeed([])).code == 1)
        #expect(eof.deleted.isEmpty)
    }

    @Test func yesSkipsTheQuestionAndNoTerminalNeverAsks() async {
        let f = fixture()
        let feed = LineFeed(["n"])
        let r = await f.run(["rm", "-y", Self.hf], tty: true, answers: feed)
        #expect(r.code == 0 && feed.asked == 0)
        #expect(f.deleted == [Self.hf])

        let long = fixture()
        #expect(await long.run(["rm", "--yes", Self.hf], tty: true, answers: LineFeed([])).code == 0)

        let piped = fixture()
        let p = await piped.run(["rm", Self.hf])
        #expect(p.code == 0 && p.err.isEmpty, "\(p.err)")
        #expect(piped.deleted == [Self.hf])
    }

    @Test func libraryModelsAreDeletedWithoutAskingEvenOnATerminal() async {
        let f = fixture()
        let feed = LineFeed([])
        let r = await f.run(["rm", LibraryFixture.qwen8], tty: true, answers: feed)
        #expect(r.code == 0 && feed.asked == 0 && r.err.isEmpty, "\(r.err)")
    }

    @Test func rmOfAPresetNameRemovesOnlyThePreset() async throws {
        let f = fixture()
        try f.presets.save(Preset(name: "mychat", repo: LibraryFixture.qwen8))
        let r = await f.run(["rm", "mychat"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "deleted 'mychat'\n")
        #expect(f.presets.load("mychat") == nil)
        #expect(f.deleted.isEmpty)
    }

    @Test func cpCreatesAPresetThatResolves() async throws {
        let f = fixture()
        let r = await f.run(["cp", LibraryFixture.qwen8, "mychat"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "copied '\(LibraryFixture.qwen8)' to 'mychat'\n")
        #expect(r.err == "success\n")
        #expect(f.presets.load("mychat")?.repo == LibraryFixture.qwen8)
        let resolved = try f.resolver.resolve("mychat")
        #expect(resolved.source == .preset && resolved.repo == LibraryFixture.qwen8)
    }

    @Test func cpAcceptsAnyResolvableSourceAndCopiesPresetSettings() async throws {
        let f = fixture()
        let viaAlias = await f.run(["cp", "qwen3:8b", "short"])
        #expect(viaAlias.code == 0, "\(viaAlias.err)")
        #expect(f.presets.load("short")?.repo == LibraryFixture.qwen8)

        try f.presets.save(Preset(name: "base", repo: LibraryFixture.qwen8, system: "be brief", parameters: ["temperature": "0.2"]))
        let copy = await f.run(["cp", "base", "tuned:v2"])
        #expect(copy.code == 0, "\(copy.err)")
        let saved = try #require(f.presets.load("tuned:v2"))
        #expect(saved == Preset(name: "tuned:v2", repo: LibraryFixture.qwen8, system: "be brief", parameters: ["temperature": "0.2"]))
    }

    @Test func cpRejectsMissingSourcesAndUnsafeNames() async throws {
        let f = fixture()
        let missing = await f.run(["cp", "nope", "x"])
        #expect(missing.code == 1 && missing.out.isEmpty)
        #expect(missing.err.hasPrefix("Error: model 'nope' not found"), "\(missing.err)")
        #expect(f.presets.list().isEmpty)

        for name in ["../evil", "a/b", ".hidden", "has space"] {
            let r = await f.run(["cp", LibraryFixture.qwen8, name])
            #expect(r.code == 1 && r.out.isEmpty, "\(name)")
            #expect(r.err.hasPrefix("Error: "), "\(name): \(r.err)")
        }
        #expect(f.presets.list().isEmpty)
        let escaped = f.presetsDirectory.deletingLastPathComponent().appendingPathComponent("evil.json")
        #expect(!FileManager.default.fileExists(atPath: escaped.path))
        let none = await f.run(["cp", "only-one"])
        #expect(none.err == "Error: accepts 2 arg(s), received 1\n")
    }
}
