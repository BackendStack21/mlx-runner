import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("run auto-pull (AC-24, AC-25)")
struct RunAutoPullTests {
    let model = "qwen3:0.6b"
    let repo = LibraryFixture.qwen06

    @Test func aMissingResolvableModelIsPulledOnceThenRun() async throws {
        let h = try RunHarness(installed: false)
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "Hello world\n\n", "stdout holds no progress: \(r.out)")
        #expect(h.hub.detailsCalls == [repo])
        #expect(h.hub.downloads.count == 1 && h.hub.downloads.first?.repo == repo)
        #expect(r.err.contains("100%"), "progress goes to stderr: \(r.err)")
        #expect(h.engine.loads.count == 1)
        #expect(h.engine.loads.first?.directory == h.hub.destination(repo), "the model loads from where it was pulled")
        #expect(h.engine.chats.count == 1)
    }

    @Test func hfPrefixedNamesArePulledToo() async throws {
        let h = try RunHarness(installed: false)
        let r = await h.run(["run", "hf.co/acme/Custom-Model-4bit", "-p", "hi"])
        #expect(r.code == 0, "\(r.err)")
        #expect(h.hub.downloads.map(\.repo) == ["acme/Custom-Model-4bit"])
        #expect(h.engine.loads.first?.directory == h.hub.destination("acme/Custom-Model-4bit"))
    }

    @Test func anInstalledModelIsNotDownloaded() async throws {
        let h = try RunHarness()
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 0, "\(r.err)")
        #expect(h.hub.detailsCalls.isEmpty && h.hub.downloads.isEmpty)
        #expect(!r.err.contains("%"), "\(r.err)")
    }

    @Test func aNameThatDoesNotResolveDownloadsNothingAndSuggests() async throws {
        let h = try RunHarness(installed: false)
        let r = await h.run(["run", "qwen3:9b", "-p", "hi"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: model 'qwen3:9b' not found, did you mean"), "\(r.err)")
        #expect(h.hub.detailsCalls.isEmpty && h.hub.downloads.isEmpty)
        #expect(h.engine.loads.isEmpty)
    }

    @Test func anUnsupportedArchitectureIsRefusedBeforeAnyWeightFile() async throws {
        let hub = FakeHub()
        hub.modelType = "mamba2"
        let h = try RunHarness(installed: false, hub: hub)
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: ") && r.err.contains("mamba2"), "\(r.err)")
        #expect(hub.downloads.isEmpty, "no weight file requested")
        #expect(h.engine.loads.isEmpty)
    }

    @Test func invalidRepoIdsFailBeforeAnyNetworkCall() async throws {
        for bad in ["hf.co/a/b/c", "hf.co/../x", "huggingface.co/only-one-part/"] {
            let h = try RunHarness(installed: false)
            let r = await h.run(["run", bad, "-p", "hi"])
            #expect(r.code == 1 && r.err.hasPrefix("Error: ") && r.err.contains("invalid repo id"), "\(bad): \(r.err)")
            #expect(h.hub.detailsCalls.isEmpty && h.hub.downloads.isEmpty, "\(bad)")
            #expect(h.engine.loads.isEmpty)
        }
    }

    @Test func aFailedPullIsAnErrorLine() async throws {
        let hub = FakeHub()
        hub.downloadError = NSError(domain: "HuggingFace", code: 404, userInfo: [NSLocalizedDescriptionKey: "HTTP 404"])
        let h = try RunHarness(installed: false, hub: hub)
        let r = await h.run(["run", model, "-p", "hi"])
        #expect(r.code == 1 && r.err.contains("Error: not found, or private/gated (set HF_TOKEN)"), "\(r.err)")
        #expect(h.engine.loads.isEmpty)
    }
}
