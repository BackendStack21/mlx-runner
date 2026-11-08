import Foundation
@testable import MLXRCLI
import MLXEngine

/// A scripted Hugging Face: records every call, writes nothing to the network, and downloads into a
/// temporary directory.
final class FakeHub: @unchecked Sendable {
    private let lock = NSLock()
    private var detailsLog: [String] = []
    private var downloadLog: [(repo: String, destination: URL)] = []
    let root: URL
    /// Progress steps (done, total, bytes per second) the fake download reports.
    var script: [(Int64, Int64, Double)] = [(100_000_000, 1_000_000_000, 50_000_000), (1_000_000_000, 1_000_000_000, 50_000_000)]
    var modelType: String? = "qwen3"
    var detailsError: Error?
    var downloadError: Error?

    init() {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-fakehub-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: root) }

    var detailsCalls: [String] { lock.withLock { detailsLog } }
    var downloads: [(repo: String, destination: URL)] { lock.withLock { downloadLog } }

    func destination(_ repo: String) -> URL { root.appendingPathComponent(repo, isDirectory: true) }

    var services: HubServices {
        HubServices(
            details: { repo in
                self.lock.withLock { self.detailsLog.append(repo) }
                if let e = self.detailsError { throw e }
                return HubModelDetails(
                    id: repo,
                    files: [HubFile(rfilename: "config.json", size: 1_000), HubFile(rfilename: "tokenizer.json", size: 2_000),
                            HubFile(rfilename: "model.safetensors", size: 1_000_000_000)],
                    modelType: self.modelType, bits: 4, readmeURL: URL(string: "https://huggingface.co/\(repo)")!,
                    downloads: 10, likes: 1, tags: [], license: "apache-2.0")
            },
            destination: { self.destination($0) },
            download: { details, destination, hooks in
                self.lock.withLock { self.downloadLog.append((details.id, destination)) }
                for (done, total, speed) in self.script { hooks.progress(done, total, speed) }
                if let e = self.downloadError { throw e }
                return destination
            })
    }
}

/// A model directory with the files `show` reads and a weights file that cannot be read.
struct ShowModelDir {
    let directory: URL

    static let license = "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\nTERMS AND CONDITIONS\n"
    static let template = "<|im_start|>{{ role }}\n{%- if enable_thinking is defined %}{%- endif %}<tool_call>{{ content }}<|im_end|>"

    /// `config`, `tokenizerConfig` and `generation` are the file contents (nil: file absent).
    init(config: String? = #"{"model_type":"qwen3","max_position_embeddings":40960,"hidden_size":4096,"quantization":{"bits":4,"group_size":64}}"#,
         tokenizerConfig: String? = nil, generation: String? = #"{"temperature":0.6,"top_k":20,"top_p":0.95}"#,
         license: String? = ShowModelDir.license, weightsReadable: Bool = false) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-showmodel-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        func write(_ name: String, _ text: String?) throws {
            if let text { try Data(text.utf8).write(to: directory.appendingPathComponent(name)) }
        }
        try write("config.json", config)
        let tc = try tokenizerConfig ?? String(decoding: JSONSerialization.data(withJSONObject: ["chat_template": Self.template]), as: UTF8.self)
        try write("tokenizer_config.json", tc)
        try write("generation_config.json", generation)
        try write("LICENSE", license)
        try write("tokenizer.json", "{}")
        let weights = directory.appendingPathComponent("model.safetensors")
        try Data("not a safetensors file".utf8).write(to: weights)
        if !weightsReadable { try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: weights.path) }
        try Data().write(to: directory.appendingPathComponent(".complete"))
    }

    func remove() {
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: directory.appendingPathComponent("model.safetensors").path)
        try? FileManager.default.removeItem(at: directory)
    }
}

/// Runs command lines against an injected library, presets, catalog and hub.
struct CLIHarness {
    let library: LibraryFixture
    let hub: FakeHub

    init(models: [LocalModel] = [], hub: FakeHub = FakeHub()) {
        library = LibraryFixture(models: models)
        self.hub = hub
    }

    func run(_ args: [String], stderrTTY: Bool = false) async -> (code: Int32, out: String, err: String) {
        let cap = Capture()
        let table = CLI.commands(services: FakeServices().services, library: library.library, presets: library.presets,
                                 catalog: .standard, now: { LibraryFixture.now }, hub: hub.services)
        let code = await CLI.run(args, console: cap.console(stderrTTY: stderrTTY), commands: table)
        return (code, cap.stdout, cap.stderr)
    }
}

/// A clock the test moves by hand.
final class TestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var t = Date(timeIntervalSince1970: 1_800_000_000)
    var now: Date { lock.withLock { t } }
    func advance(_ seconds: TimeInterval) { lock.withLock { t = t.addingTimeInterval(seconds) } }
}
