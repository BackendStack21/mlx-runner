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
    public var isSupported: Bool { SupportedArchitectures.isSupported(modelType) }
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

    public static func directory(for repo: String) -> URL {
        root.appendingPathComponent(repo, isDirectory: true)
    }

    public static func isInstalled(_ repo: String) -> Bool {
        FileManager.default.fileExists(atPath: directory(for: repo).appendingPathComponent(".complete").path)
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
        return LocalModel(repo: repo, directory: dir, sizeBytes: size,
                          modelType: (cfg["model_type"] as? String)?.lowercased(),
                          bits: (q?["bits"] as? NSNumber)?.intValue, source: source)
    }
}
