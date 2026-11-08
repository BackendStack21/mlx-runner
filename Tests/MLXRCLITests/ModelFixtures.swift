import Foundation
@testable import MLXRCLI
import MLXEngine

/// Feeds scripted answers to `Console.readLine`.
final class LineFeed: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String]
    private(set) var asked = 0
    init(_ lines: [String]) { self.lines = lines }
    func next() -> String? {
        lock.withLock { asked += 1; return lines.isEmpty ? nil : lines.removeFirst() }
    }
}

/// An in-memory model library and a presets directory under the temporary directory. Nothing touches
/// ~/Library or the Hugging Face cache.
final class LibraryFixture: @unchecked Sendable {
    static let now = Date(timeIntervalSince1970: 1_800_000_000)
    static let qwen8 = "mlx-community/Qwen3-8B-4bit"
    static let qwen06 = "mlx-community/Qwen3-0.6B-4bit"
    static let custom = "acme/Custom-Model-4bit"

    private let lock = NSLock()
    private var current: [LocalModel]
    private var removed: [String] = []
    let presetsDirectory: URL
    var presets: PresetStore { PresetStore(directory: presetsDirectory) }
    var deleted: [String] { lock.withLock { removed } }

    static func model(_ repo: String, bytes: Int, ageSeconds: TimeInterval, source: LocalModel.Source = .library) -> LocalModel {
        LocalModel(repo: repo, directory: URL(fileURLWithPath: "/fixture/\(repo)"), sizeBytes: bytes,
                   modelType: "qwen3", bits: 4, source: source, modified: now.addingTimeInterval(-ageSeconds))
    }

    init(models: [LocalModel]) {
        current = models
        presetsDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-cli-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: presetsDirectory, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: presetsDirectory) }

    /// Two library models, 0.6B 21 days old and 8B 5 minutes old.
    static func standard() -> LibraryFixture {
        LibraryFixture(models: [
            model(qwen06, bytes: 350_000_000, ageSeconds: 21 * 86_400),
            model(qwen8, bytes: 5_200_000_000, ageSeconds: 300),
        ])
    }

    var library: ModelLibrary {
        ModelLibrary(scan: { self.lock.withLock { self.current } },
                     delete: { m in self.lock.withLock { self.removed.append(m.repo); self.current.removeAll { $0.repo == m.repo } } })
    }

    var resolver: ModelResolver { ModelResolver(library: library, presets: presets) }

    func table(catalog: Catalog = .standard) -> [Command] {
        CLI.commands(services: FakeServices().services, library: library, presets: presets, catalog: catalog, now: { Self.now })
    }

    /// Runs a command line; returns the exit code and what it wrote.
    func run(_ args: [String], tty: Bool = false, answers: LineFeed? = nil, catalog: Catalog = .standard) async -> (code: Int32, out: String, err: String) {
        let cap = Capture()
        let console = cap.console(stdoutTTY: tty, stderrTTY: tty, stdinTTY: tty)
        if let answers { console.readLine = { answers.next() } }
        let code = await CLI.run(args, console: console, commands: table(catalog: catalog))
        return (code, cap.stdout, cap.stderr)
    }
}

/// The message a throwing call fails with, or nil when it does not throw.
func failureMessage(_ body: () throws -> Any) -> String? {
    do { _ = try body(); return nil } catch { return CLI.message(for: error) }
}
