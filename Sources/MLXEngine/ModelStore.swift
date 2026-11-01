import Foundation

/// A model available on disk.
public struct LocalModel: Identifiable, Hashable, Sendable {
    public var id: String { directory.path }
    public let repo: String
    public let directory: URL
    public let sizeBytes: Int
    public let modelType: String?
    public let bits: Int?
    public let source: Source

    public enum Source: String, Sendable { case library, huggingFaceCache }

    public var name: String { repo.split(separator: "/").last.map(String.init) ?? repo }
    public var author: String { repo.contains("/") ? String(repo.split(separator: "/").first!) : "local" }
    public var isSupported: Bool { SupportedArchitectures.isSupported(modelType) && !isMoE }
    /// A mixture-of-experts variant of a supported type (Gemma 4 26B-A4B).
    public var isMoE: Bool = false
}

public enum ModelStore {
    /// ~/Library/Application Support/MLX Runner/models/<org>/<repo>
    public static var root: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("MLX Runner/models", isDirectory: true)
    }

    public static var huggingFaceCache: URL {
        if let h = ProcessInfo.processInfo.environment["HF_HUB_CACHE"] { return URL(fileURLWithPath: h) }
        if let h = ProcessInfo.processInfo.environment["HF_HOME"] { return URL(fileURLWithPath: h).appendingPathComponent("hub") }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
    }

    /// Library folder for a repo id. Throws `HubError.invalidRepoID` unless `repo` is a plain
    /// `owner/name` id, so a crafted id can never escape `root`.
    public static func directory(for repo: String) throws -> URL {
        try RepoID.validate(repo)
        return root.appendingPathComponent(repo, isDirectory: true)
    }

    public static func isInstalled(_ repo: String) -> Bool {
        guard let dir = try? directory(for: repo) else { return false }
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent(".complete").path)
    }

    /// Lists models in the app library and in the Hugging Face cache.
    public static func scan() -> [LocalModel] {
        var out: [LocalModel] = []
        let fm = FileManager.default
        if let orgs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) {
            for org in orgs where org.hasDirectoryPath || isDir(org) {
                for dir in (try? fm.contentsOfDirectory(at: org, includingPropertiesForKeys: nil)) ?? [] where isDir(dir) {
                    guard fm.fileExists(atPath: dir.appendingPathComponent(".complete").path) else { continue }
                    let repo = "\(org.lastPathComponent)/\(dir.lastPathComponent)"
                    if let m = describe(dir, repo: repo, source: .library) { out.append(m) }
                }
            }
        }
        // Reuse models already downloaded by huggingface_hub / mlx-lm.
        if let entries = try? fm.contentsOfDirectory(at: huggingFaceCache, includingPropertiesForKeys: nil) {
            for e in entries where e.lastPathComponent.hasPrefix("models--") {
                let repo = e.lastPathComponent.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
                let snaps = e.appendingPathComponent("snapshots")
                guard let snap = (try? fm.contentsOfDirectory(at: snaps, includingPropertiesForKeys: [.contentModificationDateKey]))?
                    .sorted(by: { mtime($0) > mtime($1) }).first else { continue }
                let files = (try? fm.contentsOfDirectory(atPath: snap.path)) ?? []
                guard files.contains("config.json"), files.contains("tokenizer.json"),
                      files.contains(where: { $0.hasSuffix(".safetensors") }) else { continue }
                if let m = describe(snap, repo: repo, source: .huggingFaceCache), m.bits != nil || repo.hasPrefix("mlx-community/") {
                    out.append(m)
                }
            }
        }
        return out.sorted { $0.repo.lowercased() < $1.repo.lowercased() }
    }

    public static func delete(_ model: LocalModel) throws {
        switch model.source {
        case .library: try FileManager.default.removeItem(at: model.directory)
        case .huggingFaceCache: try FileManager.default.removeItem(at: model.directory.deletingLastPathComponent().deletingLastPathComponent())
        }
    }

    private static func isDir(_ u: URL) -> Bool {
        var d: ObjCBool = false
        return FileManager.default.fileExists(atPath: u.path, isDirectory: &d) && d.boolValue
    }

    private static func mtime(_ u: URL) -> Date {
        (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    static func describe(_ dir: URL, repo: String, source: LocalModel.Source) -> LocalModel? {
        let fm = FileManager.default
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
              let cfg = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let q = (cfg["quantization"] as? [String: Any]) ?? (cfg["quantization_config"] as? [String: Any])
        var size = 0
        for f in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] where f.hasSuffix(".safetensors") {
            let p = dir.appendingPathComponent(f).resolvingSymlinksInPath().path
            size += (try? fm.attributesOfItem(atPath: p)[.size] as? Int) ?? 0
        }
        var m = LocalModel(repo: repo, directory: dir, sizeBytes: size,
                           modelType: (cfg["model_type"] as? String)?.lowercased(),
                           bits: (q?["bits"] as? NSNumber)?.intValue, source: source)
        m.isMoE = SupportedArchitectures.isMoE(cfg)
        return m
    }
}

/// Loads JSON documents without ever losing the user's data: a file that fails to decode is
/// moved aside to `<name>.corrupt-<timestamp>.<ext>` instead of being silently overwritten.
public enum SafeJSONFile {
    public enum LoadResult<T> {
        /// No file yet.
        case missing
        case loaded(T)
        /// Decoding failed. `movedTo` is the quarantined copy, or nil if moving it aside failed
        /// (callers must then avoid overwriting the original).
        case corrupt(error: Error, movedTo: URL?)
    }

    public static func load<T: Decodable>(_ type: T.Type, from url: URL, decoder: JSONDecoder = JSONDecoder(),
                                          now: Date = Date()) -> LoadResult<T> {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return .missing }
        do {
            let data = try Data(contentsOf: url)
            return .loaded(try decoder.decode(T.self, from: data))
        } catch {
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyyMMdd-HHmmss"
            let ext = url.pathExtension
            let stem = url.deletingPathExtension().lastPathComponent
            var target = url.deletingLastPathComponent()
                .appendingPathComponent("\(stem).corrupt-\(f.string(from: now))" + (ext.isEmpty ? "" : ".\(ext)"))
            var n = 1
            while fm.fileExists(atPath: target.path) {
                target = url.deletingLastPathComponent()
                    .appendingPathComponent("\(stem).corrupt-\(f.string(from: now))-\(n)" + (ext.isEmpty ? "" : ".\(ext)"))
                n += 1
            }
            let moved = (try? fm.moveItem(at: url, to: target)) != nil
            return .corrupt(error: error, movedTo: moved ? target : nil)
        }
    }
}
