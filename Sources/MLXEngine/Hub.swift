import Foundation

/// A model listing from the Hugging Face Hub.
public struct HubModel: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public let downloads: Int?
    public let likes: Int?
    public let lastModified: String?
    public let createdAt: String?
    public let tags: [String]?
    public let pipelineTag: String?
    public let config: HubConfig?
    public let safetensors: HubSafetensors?

    enum CodingKeys: String, CodingKey {
        case id, downloads, likes, lastModified, createdAt, tags, config, safetensors
        case pipelineTag = "pipeline_tag"
    }

    public struct HubConfig: Codable, Hashable, Sendable {
        public let model_type: String?
        public let quantization_config: Quant?
        public struct Quant: Codable, Hashable, Sendable { public let bits: Int? }
    }
    public struct HubSafetensors: Codable, Hashable, Sendable { public let total: Int? }

    /// Exact architecture from config.json when the listing includes it.
    public var modelType: String? { config?.model_type?.lowercased() ?? modelTypeTag }
    public var isRunnable: Bool { SupportedArchitectures.isSupported(modelType) }

    /// Parameter count, e.g. "7.6B" (for quantized repos this counts packed elements).
    public var parameterLabel: String? {
        guard let n = safetensors?.total, n > 0 else { return nil }
        return n >= 1_000_000_000 ? String(format: "%.1fB", Double(n) / 1e9) : String(format: "%.0fM", Double(n) / 1e6)
    }

    public var author: String { id.split(separator: "/").first.map(String.init) ?? "" }
    public var name: String { id.split(separator: "/").last.map(String.init) ?? id }

    /// Architecture guessed from tags (exact value comes from config.json in details).
    public var modelTypeTag: String? {
        tags?.first { SupportedArchitectures.modelTypes.contains($0.lowercased()) }
    }

    /// Quantization guessed from the repo name, e.g. "4bit", "8bit", "bf16".
    public var quantizationHint: String? {
        if let b = config?.quantization_config?.bits { return "\(b)bit" }
        let n = name.lowercased()
        for q in ["2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "bf16", "fp16"] where n.contains(q) { return q }
        if n.contains("dwq") { return "dwq" }
        return nil
    }
}

public struct HubFile: Codable, Hashable, Sendable {
    public let rfilename: String
    public let size: Int?
}

public struct HubModelDetails: Sendable {
    public let id: String
    public let files: [HubFile]
    public let modelType: String?
    public let bits: Int?
    public let readmeURL: URL
    public let downloads: Int?
    public let likes: Int?
    public let tags: [String]
    public let license: String?

    public var isSupported: Bool {
        SupportedArchitectures.isSupported(modelType) && files.contains { $0.rfilename == "tokenizer.json" }
            && files.contains { $0.rfilename.hasSuffix(".safetensors") }
    }

    /// Files needed to run the model locally.
    public var requiredFiles: [HubFile] {
        files.filter { f in
            let n = f.rfilename
            if n.contains("/") { return false }
            return n.hasSuffix(".safetensors") || n == "model.safetensors.index.json" || n == "config.json"
                || n == "tokenizer.json" || n == "tokenizer_config.json" || n == "generation_config.json"
                || n == "special_tokens_map.json" || n == "chat_template.jinja" || n == "chat_template.json"
                || n == "README.md"
        }
    }

    public var downloadSize: Int { requiredFiles.reduce(0) { $0 + ($1.size ?? 0) } }
}

public enum HubSort: String, CaseIterable, Sendable {
    case downloads, likes, trending = "trendingScore", recent = "lastModified"
    public var title: String {
        switch self {
        case .downloads: return "Most downloaded"
        case .likes: return "Most liked"
        case .trending: return "Trending"
        case .recent: return "Recently updated"
        }
    }
}

public enum HubClient {
    static let base = URL(string: "https://huggingface.co")!

    static let session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.httpAdditionalHeaders = ["User-Agent": "MLXRunner/1.0 (macOS; Swift)"]
        cfg.requestCachePolicy = .reloadRevalidatingCacheData
        return URLSession(configuration: cfg)
    }()

    static func request(_ url: URL) -> URLRequest {
        var r = URLRequest(url: url)
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"] ?? UserDefaults.standard.string(forKey: "hfToken"),
           !token.isEmpty {
            r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return r
    }

    /// Searches MLX models (library tag `mlx`), text-generation only by default.
    public static func search(_ query: String, sort: HubSort = .downloads, limit: Int = 60,
                              textGenerationOnly: Bool = true) async throws -> [HubModel] {
        var c = URLComponents(url: base.appendingPathComponent("api/models"), resolvingAgainstBaseURL: false)!
        var items = [
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "sort", value: sort.rawValue),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "expand[]", value: "config"),
            URLQueryItem(name: "expand[]", value: "downloads"),
            URLQueryItem(name: "expand[]", value: "likes"),
            URLQueryItem(name: "expand[]", value: "tags"),
            URLQueryItem(name: "expand[]", value: "pipeline_tag"),
            URLQueryItem(name: "expand[]", value: "lastModified"),
        ]
        if textGenerationOnly { items.append(URLQueryItem(name: "pipeline_tag", value: "text-generation")) }
        let q = query.trimmingCharacters(in: .whitespaces)
        if !q.isEmpty { items.append(URLQueryItem(name: "search", value: q)) }
        c.queryItems = items
        let (data, resp) = try await session.data(for: request(c.url!))
        try check(resp)
        return try JSONDecoder().decode([HubModel].self, from: data)
    }

    public static func details(_ id: String) async throws -> HubModelDetails {
        let url = URL(string: "https://huggingface.co/api/models/\(id)?blobs=true")!
        let (data, resp) = try await session.data(for: request(url))
        try check(resp)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw URLError(.cannotParseResponse)
        }
        let siblings = (obj["siblings"] as? [[String: Any]] ?? []).compactMap { s -> HubFile? in
            guard let n = s["rfilename"] as? String else { return nil }
            return HubFile(rfilename: n, size: (s["size"] as? NSNumber)?.intValue)
        }
        let config = obj["config"] as? [String: Any]
        let q = config?["quantization_config"] as? [String: Any]
        let card = obj["cardData"] as? [String: Any]
        return HubModelDetails(
            id: id, files: siblings,
            modelType: config?["model_type"] as? String,
            bits: (q?["bits"] as? NSNumber)?.intValue,
            readmeURL: URL(string: "https://huggingface.co/\(id)")!,
            downloads: (obj["downloads"] as? NSNumber)?.intValue,
            likes: (obj["likes"] as? NSNumber)?.intValue,
            tags: obj["tags"] as? [String] ?? [],
            license: card?["license"] as? String)
    }

    public static func readme(_ id: String) async -> String? {
        let url = URL(string: "https://huggingface.co/\(id)/resolve/main/README.md")!
        guard let (data, resp) = try? await session.data(for: request(url)),
              (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func fileURL(_ id: String, _ file: String) -> URL {
        URL(string: "https://huggingface.co/\(id)/resolve/main/\(file)")!
    }

    static func check(_ resp: URLResponse) throws {
        if let h = resp as? HTTPURLResponse, !(200..<300).contains(h.statusCode) {
            throw NSError(domain: "HuggingFace", code: h.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "Hugging Face returned HTTP \(h.statusCode)"])
        }
    }
}

// MARK: - Downloads

/// Downloads every file of a repo into the local model store with progress and cancellation.
public final class ModelDownload: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    public let repo: String
    public let destination: URL
    private let files: [HubFile]
    private let total: Int64
    private var completedBytes: Int64 = 0
    private var currentBytes: Int64 = 0
    private var session: URLSession!
    private var continuation: CheckedContinuation<URL, Error>?
    private var index = 0
    private var task: URLSessionDownloadTask?
    private let lock = NSLock()
    private var startTime = Date()
    public var onProgress: (@Sendable (Double, Int64, Int64, Double) -> Void)?

    public init(details: HubModelDetails, destination: URL) {
        repo = details.id
        self.destination = destination
        files = details.requiredFiles.sorted { ($0.size ?? 0) < ($1.size ?? 0) }
        total = Int64(details.downloadSize)
        super.init()
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 60 * 60 * 24
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }

    public func start() async throws -> URL {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        startTime = Date()
        return try await withCheckedThrowingContinuation { cont in
            lock.lock(); continuation = cont; lock.unlock()
            next()
        }
    }

    public func cancel() {
        lock.lock()
        let t = task
        let c = continuation
        continuation = nil
        lock.unlock()
        t?.cancel()
        session.invalidateAndCancel()
        c?.resume(throwing: CancellationError())
    }

    private func next() {
        lock.lock()
        // Skip files that already exist with the right size (resume after interruption).
        while index < files.count {
            let f = files[index]
            let path = destination.appendingPathComponent(f.rfilename).path
            if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
               let sz = attrs[.size] as? Int, let want = f.size, sz == want {
                completedBytes += Int64(want)
                index += 1
            } else { break }
        }
        if index >= files.count {
            let c = continuation
            continuation = nil
            lock.unlock()
            try? Data().write(to: destination.appendingPathComponent(".complete"))
            try? repo.write(to: destination.appendingPathComponent(".repo"), atomically: true, encoding: .utf8)
            session.finishTasksAndInvalidate()
            c?.resume(returning: destination)
            return
        }
        let f = files[index]
        currentBytes = 0
        let t = session.downloadTask(with: HubClient.request(HubClient.fileURL(repo, f.rfilename)))
        task = t
        lock.unlock()
        t.resume()
    }

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                           totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        lock.lock()
        currentBytes = totalBytesWritten
        let done = completedBytes + currentBytes
        lock.unlock()
        let elapsed = Date().timeIntervalSince(startTime)
        let speed = Double(done) / max(elapsed, 0.001)
        onProgress?(total > 0 ? min(1, Double(done) / Double(total)) : 0, done, total, speed)
    }

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        lock.lock()
        let f = files[index]
        lock.unlock()
        if let h = downloadTask.response as? HTTPURLResponse, !(200..<300).contains(h.statusCode) {
            fail(NSError(domain: "HuggingFace", code: h.statusCode,
                         userInfo: [NSLocalizedDescriptionKey: "Download of \(f.rfilename) failed (HTTP \(h.statusCode)). The model may be gated — set a Hugging Face token in Settings."]))
            return
        }
        let dest = destination.appendingPathComponent(f.rfilename)
        do {
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: location, to: dest)
        } catch {
            fail(error)
            return
        }
        lock.lock()
        completedBytes += Int64(f.size ?? Int(currentBytes))
        currentBytes = 0
        index += 1
        lock.unlock()
        next()
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { fail(error) }
    }

    private func fail(_ error: Error) {
        lock.lock()
        let c = continuation
        continuation = nil
        lock.unlock()
        c?.resume(throwing: error)
    }
}
