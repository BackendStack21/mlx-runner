import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("Model name resolver (AC-10)")
struct ModelResolverTests {
    static let qwen8 = LibraryFixture.qwen8

    func fixture() throws -> LibraryFixture {
        let f = LibraryFixture(models: [
            LibraryFixture.model(Self.qwen8, bytes: 5_200_000_000, ageSeconds: 60),
            LibraryFixture.model(LibraryFixture.custom, bytes: 1_000_000_000, ageSeconds: 60),
        ])
        try f.presets.save(Preset(name: "mychat", repo: Self.qwen8, system: "be brief"))
        return f
    }

    @Test func existingPathWins() throws {
        let f = try fixture()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let r = try f.resolver.resolve(dir.path)
        #expect(r.source == .path)
        #expect(r.directory?.standardizedFileURL.path == dir.standardizedFileURL.path)
        #expect(r.repo == nil)
    }

    @Test func installedRepoIdIsCaseInsensitiveAndCanonical() throws {
        let f = try fixture()
        for typed in [Self.qwen8, "MLX-COMMUNITY/qwen3-8b-4bit"] {
            let r = try f.resolver.resolve(typed)
            #expect(r.source == .installed, "\(typed)")
            #expect(r.repo == Self.qwen8)
            #expect(r.directory?.path == "/fixture/\(Self.qwen8)")
        }
    }

    @Test func huggingFaceUrlsNameARepoWithoutPulling() throws {
        let f = try fixture()
        for prefix in ["hf.co/", "huggingface.co/"] {
            let r = try f.resolver.resolve(prefix + "acme/Other-4bit")
            #expect(r.source == .huggingFace)
            #expect(r.repo == "acme/Other-4bit")
            #expect(r.directory == nil, "not installed")
        }
        let installed = try f.resolver.resolve("hf.co/" + LibraryFixture.custom)
        #expect(installed.repo == LibraryFixture.custom)
        #expect(installed.directory?.path == "/fixture/\(LibraryFixture.custom)")
        let bare = try f.resolver.resolve("acme/NotInstalled")
        #expect(bare.source == .huggingFace && bare.repo == "acme/NotInstalled" && bare.directory == nil)
    }

    @Test func catalogAliasesIncludeTheFamilyDefault() throws {
        let f = try fixture()
        let eight = try f.resolver.resolve("qwen3:8b")
        #expect(eight.source == .catalog && eight.repo == Self.qwen8)
        #expect(eight.directory?.path == "/fixture/\(Self.qwen8)", "installed, so a directory")
        for family in ["qwen3", "qwen3:latest"] {
            let r = try f.resolver.resolve(family)
            #expect(r.source == .catalog && r.repo == Self.qwen8, "\(family)")
        }
        let four = try f.resolver.resolve("qwen3:4b")
        #expect(four.repo == "mlx-community/Qwen3-4B-4bit")
        #expect(four.directory == nil, "not installed")
    }

    @Test func presetsResolveToTheirModelAndKeepTheirSettings() throws {
        let f = try fixture()
        let r = try f.resolver.resolve("mychat")
        #expect(r.source == .preset && r.repo == Self.qwen8)
        #expect(r.preset?.system == "be brief")
        #expect(r.directory?.path == "/fixture/\(Self.qwen8)")
    }

    @Test func installedShortNameComesLast() throws {
        let f = try fixture()
        for typed in ["Custom-Model-4bit", "custom-model-4bit"] {
            let r = try f.resolver.resolve(typed)
            #expect(r.source == .shortName && r.repo == LibraryFixture.custom, "\(typed)")
        }
    }

    @Test func orderIsCatalogBeforePresetBeforeShortName() throws {
        let f = try fixture()
        try f.presets.save(Preset(name: "qwen3:8b", repo: LibraryFixture.custom))
        try f.presets.save(Preset(name: "Custom-Model-4bit", repo: Self.qwen8))
        let catalogWins = try f.resolver.resolve("qwen3:8b")
        #expect(catalogWins.source == .catalog && catalogWins.repo == Self.qwen8)
        let presetWins = try f.resolver.resolve("Custom-Model-4bit")
        #expect(presetWins.source == .preset && presetWins.repo == Self.qwen8)
    }

    @Test func ambiguousShortNameListsTheCandidates() throws {
        let f = LibraryFixture(models: [
            LibraryFixture.model("a/Same-4bit", bytes: 1, ageSeconds: 1),
            LibraryFixture.model("b/Same-4bit", bytes: 1, ageSeconds: 1),
        ])
        let m = try #require(failureMessage { try f.resolver.resolve("Same-4bit") })
        #expect(m.contains("ambiguous") && m.contains("a/Same-4bit") && m.contains("b/Same-4bit"), "\(m)")
    }

    @Test func aMissSuggestsTheClosestNames() throws {
        let f = try fixture()
        let m = try #require(failureMessage { try f.resolver.resolve("qwen3:9b") })
        #expect(m.hasPrefix("model 'qwen3:9b' not found"), "\(m)")
        #expect(m.contains("did you mean") && m.contains("qwen3:8b"), "\(m)")
        let installed = try #require(failureMessage { try f.resolver.resolve("Custom-Model-4bi") })
        #expect(installed.contains("did you mean") && installed.contains(LibraryFixture.custom), "\(installed)")
        let far = try #require(failureMessage { try f.resolver.resolve("zzzzzzzzzz") })
        #expect(far == "model 'zzzzzzzzzz' not found", "\(far)")
    }

    @Test func derivedRepoIdsMustValidate() throws {
        let f = try fixture()
        for typed in ["../x", "a/b/c", "hf.co/a/b/c", "hf.co/org", "hf.co/../x", "huggingface.co//x"] {
            let m = try #require(failureMessage { try f.resolver.resolve(typed) }, "\(typed) resolved")
            #expect(m.contains("invalid repo id"), "\(typed): \(m)")
        }
        #expect(failureMessage { try f.resolver.resolve("") } != nil)
    }

    @Test func aCustomCatalogIsUsed() throws {
        let f = try fixture()
        let tiny = CatalogModel(id: "acme/Tiny-4bit", title: "Tiny", blurb: "b", gigabytes: 0.1, aliases: ["tiny", "tiny:latest"])
        let resolver = ModelResolver(library: f.library, presets: f.presets, catalog: Catalog(models: [tiny]))
        #expect(try resolver.resolve("tiny:latest").repo == "acme/Tiny-4bit")
        #expect(failureMessage { try resolver.resolve("qwen3:8b") } != nil, "the built-in catalog is not consulted")
    }
}
