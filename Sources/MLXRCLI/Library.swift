import CryptoKit
import Foundation
import MLXEngine

/// The installed models, injected so tests never scan or delete the real library.
public struct ModelLibrary: Sendable {
    public var scan: @Sendable () -> [LocalModel]
    public var delete: @Sendable (LocalModel) throws -> Void

    public init(scan: @escaping @Sendable () -> [LocalModel], delete: @escaping @Sendable (LocalModel) throws -> Void) {
        self.scan = scan; self.delete = delete
    }

    public static let live = ModelLibrary(scan: { ModelStore.scan() }, delete: { try ModelStore.delete($0) })
}

/// A named alias for a model, with the system prompt and parameters saved alongside it.
public struct Preset: Codable, Equatable, Sendable {
    public var name: String
    public var repo: String
    public var system: String?
    public var parameters: [String: String]

    public init(name: String, repo: String, system: String? = nil, parameters: [String: String] = [:]) {
        self.name = name; self.repo = repo; self.system = system; self.parameters = parameters
    }
}

/// Presets as one `<directory>/<name>.json` file each, read and written safely.
public struct PresetStore: Sendable {
    public var directory: URL

    public init(directory: URL) { self.directory = directory }

    public static let live = PresetStore(directory: ModelStore.root.deletingLastPathComponent().appendingPathComponent("presets", isDirectory: true))

    /// A preset name is 1...128 characters of letters, digits and `. _ - :`, not starting with a dot.
    public static func validateName(_ name: String) throws {
        let ok = !name.isEmpty && name.utf8.count <= 128 && !name.hasPrefix(".") && name.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 58) || ($0 >= 65 && $0 <= 90) || ($0 >= 97 && $0 <= 122) || $0 == 46 || $0 == 95 || $0 == 45
        }
        guard ok else { throw CLIError.failure("invalid model name '\(name)' (use letters, digits and . _ - :)") }
    }

    private func file(_ name: String) -> URL { directory.appendingPathComponent(name + ".json") }

    /// The preset, or nil when missing, unreadable (moved aside) or the name is unsafe.
    public func load(_ name: String) -> Preset? {
        guard (try? Self.validateName(name)) != nil else { return nil }
        if case .loaded(let p) = SafeJSONFile.load(Preset.self, from: file(name)) { return p }
        return nil
    }

    public func list() -> [Preset] {
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? [])
            .filter { $0.hasSuffix(".json") && !$0.contains(".corrupt-") }.map { String($0.dropLast(5)) }
        return names.compactMap { load($0) }.sorted { $0.name < $1.name }
    }

    public func save(_ preset: Preset) throws {
        try Self.validateName(preset.name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(preset).write(to: file(preset.name), options: .atomic)
    }

    public func delete(_ name: String) throws {
        try Self.validateName(name)
        let url = file(name)
        if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
    }
}

/// What a model name resolved to.
public struct Resolution: Equatable, Sendable {
    public enum Source: String, Sendable { case path, installed, huggingFace, catalog, preset, shortName }
    public var source: Source
    /// The Hugging Face repo id; nil for a plain path.
    public var repo: String?
    /// Where the model is on disk; nil when it is not installed.
    public var directory: URL?
    /// The preset a name resolved through.
    public var preset: Preset?

    public init(source: Source, repo: String?, directory: URL?, preset: Preset? = nil) {
        self.source = source; self.repo = repo; self.directory = directory; self.preset = preset
    }
}

/// Turns what the user typed into a model: path, installed repo id, `hf.co/org/repo`, catalog alias,
/// preset, installed short name, in that order.
public struct ModelResolver: Sendable {
    public var library: ModelLibrary
    public var presets: PresetStore
    public var catalog: Catalog

    public init(library: ModelLibrary, presets: PresetStore, catalog: Catalog = .standard) {
        self.library = library; self.presets = presets; self.catalog = catalog
    }

    public func resolve(_ name: String) throws -> Resolution {
        guard !name.isEmpty else { throw CLIError.failure("model name is empty") }
        if FileManager.default.fileExists(atPath: name) {
            return Resolution(source: .path, repo: nil, directory: URL(fileURLWithPath: name))
        }
        let models = library.scan()
        func installed(_ repo: String) -> LocalModel? { models.first { $0.repo.lowercased() == repo.lowercased() } }
        func make(_ source: Resolution.Source, _ repo: String, preset: Preset? = nil) -> Resolution {
            let m = installed(repo)
            return Resolution(source: source, repo: m?.repo ?? repo, directory: m?.directory, preset: preset)
        }
        if let m = installed(name) { return make(.installed, m.repo) }
        let lower = name.lowercased()
        for prefix in ["hf.co/", "huggingface.co/"] where lower.hasPrefix(prefix) {
            let repo = String(name.dropFirst(prefix.count))
            try validateRepo(repo)
            return make(.huggingFace, repo)
        }
        if let repo = catalog.repo(forAlias: name) { return make(.catalog, repo) }
        if let p = presets.load(name) {
            try validateRepo(p.repo)
            return make(.preset, p.repo, preset: p)
        }
        let named = models.filter { $0.name.lowercased() == lower }
        if named.count > 1 {
            throw CLIError.failure("'\(name)' is ambiguous; matches: " + named.map(\.repo).joined(separator: ", "))
        }
        if let m = named.first { return make(.shortName, m.repo) }
        if name.contains("/") {
            try validateRepo(name)
            return make(.huggingFace, name)
        }
        throw CLIError.failure(missMessage(name, models: models))
    }

    private func validateRepo(_ repo: String) throws {
        guard RepoID.isValid(repo) else { throw CLIError.failure("invalid repo id: \(repo) (expected owner/name)") }
    }

    private func missMessage(_ name: String, models: [LocalModel]) -> String {
        let lower = name.lowercased()
        var candidates: [(shown: String, keys: [String])] = catalog.aliases.map { ($0, [$0]) }
        candidates += models.map { ($0.repo, [$0.repo.lowercased(), $0.name.lowercased()]) }
        candidates += presets.list().map { ($0.name, [$0.name.lowercased()]) }
        var scored: [(String, Int)] = []
        for c in candidates where !scored.contains(where: { $0.0 == c.shown }) {
            let d = c.keys.map { CLI.editDistance(lower, $0) }.min() ?? 99
            if d <= 2 { scored.append((c.shown, d)) }
        }
        var msg = "model '\(name)' not found"
        let best = scored.sorted { ($0.1, $0.0) < ($1.1, $1.0) }.prefix(3).map { "'\($0.0)'" }
        if !best.isEmpty { msg += ", did you mean " + best.joined(separator: " or ") + "?" }
        return msg
    }
}

extension CLI {
    /// Base-1000 size: `%d UNIT` from 10, else `%.1f UNIT` (`5.2 GB`, `350 MB`).
    static func humanBytes(_ n: Int) -> String {
        if n < 1000 { return "\(max(n, 0)) B" }
        var value = Double(n) / 1000
        var unit = "KB"
        for next in ["MB", "GB", "TB"] where value >= 1000 { value /= 1000; unit = next }
        return (value >= 10 ? String(Int(value)) : String(format: "%.1f", value)) + " " + unit
    }

    /// Ollama's relative time (`Less than a second`, `About a minute`, `5 minutes`, `3 weeks`) plus ` ago`.
    static func relativeTime(_ date: Date, now: Date) -> String {
        let delta = now.timeIntervalSince(date)
        let seconds = Int(abs(delta))
        let hours = seconds / 3600
        let text: String
        switch seconds {
        case 0: text = "Less than a second"
        case 1: text = "1 second"
        case ..<60: text = "\(seconds) seconds"
        case ..<120: text = "About a minute"
        case ..<3600: text = "\(seconds / 60) minutes"
        default:
            if hours == 1 { text = "About an hour" }
            else if hours < 48 { text = "\(hours) hours" }
            else if hours < 24 * 14 { text = "\(hours / 24) days" }
            else if hours < 24 * 60 { text = "\(hours / 24 / 7) weeks" }
            else if hours < 24 * 365 * 2 { text = "\(hours / 24 / 30) months" }
            else { text = "\(hours / 24 / 365) years" }
        }
        return text + (delta < 0 ? " from now" : " ago")
    }

    /// Borderless table: every column but the last padded to its widest cell, then 4 spaces; the last
    /// column's body cells followed by 4 spaces, its header by 1.
    static func renderTable(header: [String], rows: [[String]]) -> String {
        let last = header.count - 1
        let widths = header.indices.map { i in ([header[i]] + rows.map { $0[i] }).map(\.count).max() ?? 0 }
        func line(_ cells: [String], tail: String) -> String {
            cells.indices.map { i in i == last ? cells[i] + tail : cells[i].padding(toLength: widths[i], withPad: " ", startingAt: 0) + "    " }.joined() + "\n"
        }
        return line(header, tail: " ") + rows.map { line($0, tail: "    ") }.joined()
    }

    /// The 12 hex characters shown as ID: the start of the SHA-256 of the repo id.
    static func modelID(_ repo: String) -> String {
        SHA256.hash(data: Data(repo.utf8)).prefix(6).map { String(format: "%02x", $0) }.joined()
    }
}
