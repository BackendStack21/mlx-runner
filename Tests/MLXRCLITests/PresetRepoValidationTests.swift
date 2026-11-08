import Foundation
import Testing
@testable import MLXRCLI

@Suite("A preset's repo id is validated (SL03 review R-01)")
struct PresetRepoValidationTests {
    static let badRepos = ["../x", "a/b/c", "org/../etc", "", "/abs", "no-slash"]

    /// Writes a preset file by hand, as a damaged or hostile file would be.
    func plant(_ name: String, repo: String, in f: LibraryFixture) throws {
        let json = try JSONSerialization.data(withJSONObject: ["name": name, "repo": repo, "parameters": [String: String]()])
        try json.write(to: f.presetsDirectory.appendingPathComponent("\(name).json"))
    }

    func files(_ f: LibraryFixture) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: f.presetsDirectory.path)) ?? []).sorted()
    }

    @Test func aPresetWithAnInvalidRepoDoesNotResolve() throws {
        for (i, repo) in Self.badRepos.enumerated() {
            let f = LibraryFixture.standard()
            let name = "bad\(i)"
            try plant(name, repo: repo, in: f)
            let m = failureMessage { try f.resolver.resolve(name) }
            #expect(m != nil, "preset repo \(repo.debugDescription) resolved")
            #expect(m?.contains("invalid repo id") == true, "\(repo.debugDescription): \(m ?? "no error")")
        }
    }

    @Test func aGoodPresetStillResolves() throws {
        let f = LibraryFixture.standard()
        try plant("good", repo: LibraryFixture.qwen8, in: f)
        let r = try f.resolver.resolve("good")
        #expect(r.source == .preset && r.repo == LibraryFixture.qwen8)
    }

    @Test func cpFromAPresetWithAnInvalidRepoFailsAndWritesNothing() async throws {
        for (i, repo) in Self.badRepos.enumerated() {
            let f = LibraryFixture.standard()
            let name = "bad\(i)"
            try plant(name, repo: repo, in: f)
            let before = files(f)
            let r = await f.run(["cp", name, "copy"])
            #expect(r.code == 1 && r.out.isEmpty, "\(repo.debugDescription)")
            #expect(r.err.hasPrefix("Error: ") && r.err.contains("invalid repo id"), "\(repo.debugDescription): \(r.err)")
            #expect(files(f) == before, "cp wrote a file for \(repo.debugDescription)")
            #expect(f.presets.load("copy") == nil)
        }
    }

    @Test func rmOfABrokenPresetStillRemovesIt() async throws {
        let f = LibraryFixture.standard()
        try plant("broken", repo: "../x", in: f)
        let r = await f.run(["rm", "broken"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "deleted 'broken'\n")
        #expect(!files(f).contains("broken.json"))
        #expect(f.deleted.isEmpty)
    }
}
