import Foundation
import Testing
@testable import MLXRCLI

@Suite("serve, tokenize and bench resolve names like pull and run (SL03 review note)")
struct InstalledModelDirectoryTests {
    func resolver(_ lib: LibraryFixture = .standard()) -> ModelResolver { lib.resolver }

    @Test func catalogAliasOfAnInstalledModelResolves() throws {
        let url = try installedModelDirectory("qwen3:8b", resolver: resolver())
        #expect(url.path == "/fixture/mlx-community/Qwen3-8B-4bit")
        #expect(try installedModelDirectory("qwen3:0.6b", resolver: resolver()).path == "/fixture/mlx-community/Qwen3-0.6B-4bit")
    }

    @Test func repoIdsAndHFPrefixesResolve() throws {
        let lib = LibraryFixture(models: [LibraryFixture.model(LibraryFixture.custom, bytes: 1_000_000, ageSeconds: 60)])
        #expect(try installedModelDirectory("acme/Custom-Model-4bit", resolver: lib.resolver).path == "/fixture/acme/Custom-Model-4bit")
        #expect(try installedModelDirectory("hf.co/acme/Custom-Model-4bit", resolver: lib.resolver).path == "/fixture/acme/Custom-Model-4bit")
    }

    @Test func aPathResolvesToItself() throws {
        #expect(try installedModelDirectory(NSTemporaryDirectory(), resolver: resolver()).path == URL(fileURLWithPath: NSTemporaryDirectory()).path)
    }

    @Test func aResolvableButMissingModelPointsToPull() {
        let m = failureMessage { try installedModelDirectory("qwen3:4b", resolver: self.resolver()) }
        #expect(m?.contains("qwen3:4b") == true && m?.contains("mlxr pull") == true, "\(String(describing: m))")
    }

    @Test func aMissGivesDidYouMean() {
        let m = failureMessage { try installedModelDirectory("qwen3:9b", resolver: self.resolver()) }
        #expect(m?.contains("did you mean") == true, "\(String(describing: m))")
    }

    @Test func invalidRepoIdsFailValidation() {
        for bad in ["hf.co/a/b/c", "hf.co/../x"] {
            let m = failureMessage { try installedModelDirectory(bad, resolver: self.resolver()) }
            #expect(m?.contains("invalid repo id") == true, "\(bad): \(String(describing: m))")
        }
    }
}
