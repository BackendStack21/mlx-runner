import CryptoKit
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
        public let enable_moe_block: Bool?
        public let text_config: TextConfig?
        public struct TextConfig: Codable, Hashable, Sendable { public let enable_moe_block: Bool? }
        /// Gemma 4 mixture-of-experts exports share the dense models' `model_type`.
        public var isMoE: Bool { enable_moe_block == true || text_config?.enable_moe_block == true }
        public struct Quant: Codable, Hashable, Sendable { public let bits: Int? }
    }
    public struct HubSafetensors: Codable, Hashable, Sendable { public let total: Int? }

    /// Exact architecture from config.json when the listing includes it.
    public var modelType: String? { config?.model_type?.lowercased() ?? modelTypeTag }
    public var isRunnable: Bool { SupportedArchitectures.isSupported(modelType) && config?.isMoE != true }

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
    /// SHA-256 of the file contents for LFS files (from `?blobs=true`), lowercase hex.
    public let sha256: String?

    public init(rfilename: String, size: Int?, sha256: String? = nil) {
        self.rfilename = rfilename
        self.size = size
        self.sha256 = sha256
    }
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

    public var isMoE: Bool = false
    public var isSupported: Bool {
        SupportedArchitectures.isSupported(modelType) && !isMoE && files.contains { $0.rfilename == "tokenizer.json" }
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

// MARK: - Errors, validation and security policies

public enum HubError: LocalizedError, Equatable, Sendable {
    case invalidRepoID(String)
    case invalidFileName(String)
    case checksumMismatch(file: String, expected: String, actual: String)
    case sizeMismatch(file: String, expected: Int, actual: Int)
    case http(file: String, status: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidRepoID(let id):
            return "“\(id)” is not a valid Hugging Face repo id (expected owner/name)."
        case .invalidFileName(let n):
            return "Refusing to download a file with an unsafe name: \(n)"
        case .checksumMismatch(let f, let e, let a):
            return "\(f) failed SHA-256 verification (expected \(e.prefix(12))…, got \(a.prefix(12))…). The file was deleted; try again."
        case .sizeMismatch(let f, let e, let a):
            return "\(f) is incomplete (\(a) of \(e) bytes). Try again to resume."
        case .http(let f, let s):
            return "Download of \(f) failed (HTTP \(s)). The model may be gated — set a Hugging Face token in Settings."
        }
    }
}

/// Hugging Face repo ids ("owner/name"). Anything else is rejected before it becomes a URL or a path.
public enum RepoID {
    /// Accepts only `^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$` with no `..` sequence.
    public static func isValid(_ id: String) -> Bool {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, id.utf8.count <= 256 else { return false }
        return parts.allSatisfy { p in
            guard let first = p.utf8.first, isAlnum(first), !p.contains("..") else { return false }
            return p.utf8.allSatisfy { isAlnum($0) || $0 == UInt8(ascii: ".") || $0 == UInt8(ascii: "_") || $0 == UInt8(ascii: "-") }
        }
    }

    public static func validate(_ id: String) throws {
        guard isValid(id) else { throw HubError.invalidRepoID(id) }
    }

    /// A single, non-hidden path component (no separators, no `.`/`..`).
    public static func isSafeFileName(_ name: String) -> Bool {
        !name.isEmpty && !name.hasPrefix(".") && !name.contains("/") && !name.contains("\\")
            && !name.contains("\0") && name.utf8.count <= 255
    }

    private static func isAlnum(_ c: UInt8) -> Bool {
        (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122)
    }
}

/// Decides where the Hugging Face bearer token may travel.
public enum HubAuthPolicy {
    /// https and huggingface.co / *.huggingface.co / hf.co / *.hf.co.
    public static func isTrustedHost(_ url: URL?) -> Bool {
        guard let url, url.scheme?.lowercased() == "https", let host = url.host?.lowercased() else { return false }
        return host == "huggingface.co" || host.hasSuffix(".huggingface.co") || host == "hf.co" || host.hasSuffix(".hf.co")
    }

    /// Whether `Authorization` may be forwarded on a redirect from `current` to `redirect`.
    /// Requires a trusted https host on both ends and the same host: cross-host hops (e.g. the
    /// signed-URL CDN hosts LFS files redirect to) never receive the token — they don't need it.
    public static func shouldForwardAuthorization(from current: URL?, to redirect: URL?) -> Bool {
        guard isTrustedHost(current), isTrustedHost(redirect),
              let a = current?.host?.lowercased(), let b = redirect?.host?.lowercased() else { return false }
        return a == b
    }

    /// The request to actually follow for a redirect: `Authorization` is set from the original
    /// request only when allowed (and removed otherwise); `Range` is carried over for resumes.
    public static func redirectedRequest(_ proposed: URLRequest, from current: URL?, original: URLRequest?) -> URLRequest {
        var r = proposed
        let auth = original?.value(forHTTPHeaderField: "Authorization")
        if let auth, shouldForwardAuthorization(from: current, to: proposed.url) {
            r.setValue(auth, forHTTPHeaderField: "Authorization")
        } else {
            r.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        if let range = original?.value(forHTTPHeaderField: "Range") { r.setValue(range, forHTTPHeaderField: "Range") }
        return r
    }
}

/// Policy for links clicked in rendered Markdown (model cards, model output).
public enum LinkPolicy {
    public enum Decision: Equatable, Sendable { case open, confirm, discard }

    /// https huggingface.co links open directly; other http(s) and mailto links need confirmation;
    /// every other scheme (file:, smb:, custom app schemes, relative links) is discarded.
    public static func decide(_ url: URL) -> Decision {
        switch url.scheme?.lowercased() {
        case "https", "http":
            guard let host = url.host?.lowercased(), !host.isEmpty else { return .discard }
            if url.scheme?.lowercased() == "https", host == "huggingface.co" || host.hasSuffix(".huggingface.co") { return .open }
            return .confirm
        case "mailto":
            return .confirm
        default:
            return .discard
        }
    }

    /// Removes link attributes that `decide` would discard, so they are not clickable at all
    /// (render-time guard, independent of how the view routes link clicks).
    public static func sanitize(_ text: AttributedString) -> AttributedString {
        var out = text
        for run in text.runs {
            if let url = run.link, decide(url) == .discard { out[run.range].link = nil }
        }
        return out
    }
}

/// Streaming SHA-256 of a file (constant memory; files can be many GB).
public enum FileHasher {
    public static func sha256(of url: URL, chunkSize: Int = 8 << 20) throws -> String {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while true {
            let chunk = try autoreleasepool { try h.read(upToCount: chunkSize) }
            guard let chunk, !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func isSHA256Hex(_ s: String) -> Bool {
        s.utf8.count == 64 && s.utf8.allSatisfy { ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102) }
    }
}

/// Pure decisions for HTTP Range resume of a `<file>.partial`.
public enum ResumePlan {
    public enum Start: Equatable, Sendable {
        case fresh
        case resume(from: Int64)
        /// The partial already has every byte; just verify and install it.
        case verifyOnly
    }

    public static func start(partialBytes: Int64, expectedSize: Int?) -> Start {
        guard partialBytes > 0 else { return .fresh }
        if let e = expectedSize {
            if partialBytes == Int64(e) { return .verifyOnly }
            if partialBytes > Int64(e) { return .fresh }
        }
        return .resume(from: partialBytes)
    }

    public enum Response: Equatable, Sendable {
        /// 206 starting exactly at the requested offset: append.
        case append
        /// 2xx full body: truncate the partial and write from byte 0.
        case restart
        /// The server answered the Range oddly (416 or mismatched Content-Range): drop the partial, retry without Range.
        case retryWithoutRange
        case fail(Int)
    }

    public static func response(status: Int, requestedOffset: Int64, contentRange: String?) -> Response {
        switch status {
        case 206:
            guard requestedOffset > 0, let s = contentRangeStart(contentRange), s == requestedOffset else { return .retryWithoutRange }
            return .append
        case 200..<300: return .restart
        case 416: return .retryWithoutRange
        default: return .fail(status)
        }
    }

    /// "bytes 100-199/200" → 100.
    public static func contentRangeStart(_ header: String?) -> Int64? {
        guard let h = header?.trimmingCharacters(in: .whitespaces), h.lowercased().hasPrefix("bytes ") else { return nil }
        let spec = h.dropFirst(6)
        guard let dash = spec.firstIndex(of: "-") else { return nil }
        return Int64(spec[..<dash].trimmingCharacters(in: .whitespaces))
    }
}

// MARK: - Hub API

/// Strips the bearer token from cross-host redirects of API requests.
final class HubRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(HubAuthPolicy.redirectedRequest(request, from: response.url ?? task.currentRequest?.url,
                                                          original: task.originalRequest))
    }
}

public enum HubClient {
    static let base = URL(string: "https://huggingface.co")!

    static let session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.httpAdditionalHeaders = ["User-Agent": "MLXRunner/1.0 (macOS; Swift)"]
        cfg.requestCachePolicy = .reloadRevalidatingCacheData
        return URLSession(configuration: cfg, delegate: HubRedirectDelegate(), delegateQueue: nil)
    }()

    static func request(_ url: URL) -> URLRequest {
        var r = URLRequest(url: url)
        if HubAuthPolicy.isTrustedHost(url), let token = HFToken.current {
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
        try RepoID.validate(id)
        var c = URLComponents(url: base.appendingPathComponent("api/models").appendingPathComponent(id),
                              resolvingAgainstBaseURL: false)!
        c.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        let (data, resp) = try await session.data(for: request(c.url!))
        try check(resp)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw URLError(.cannotParseResponse)
        }
        let siblings = (obj["siblings"] as? [[String: Any]] ?? []).compactMap { s -> HubFile? in
            guard let n = s["rfilename"] as? String else { return nil }
            let lfs = s["lfs"] as? [String: Any]
            let sha = (lfs?["sha256"] as? String)?.lowercased()
            return HubFile(rfilename: n, size: (s["size"] as? NSNumber)?.intValue,
                           sha256: sha.flatMap { FileHasher.isSHA256Hex($0) ? $0 : nil })
        }
        let config = obj["config"] as? [String: Any]
        let q = config?["quantization_config"] as? [String: Any]
        let card = obj["cardData"] as? [String: Any]
        var details = HubModelDetails(
            id: id, files: siblings,
            modelType: config?["model_type"] as? String,
            bits: (q?["bits"] as? NSNumber)?.intValue,
            readmeURL: base.appendingPathComponent(id),
            downloads: (obj["downloads"] as? NSNumber)?.intValue,
            likes: (obj["likes"] as? NSNumber)?.intValue,
            tags: obj["tags"] as? [String] ?? [],
            license: card?["license"] as? String)
        details.isMoE = SupportedArchitectures.isMoE(config)
        return details
    }

    public static func readme(_ id: String) async -> String? {
        guard RepoID.isValid(id),
              let (data, resp) = try? await session.data(for: request(fileURL(id, "README.md"))),
              (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Callers must validate `id` (RepoID.isValid) and `file` (RepoID.isSafeFileName) first.
    static func fileURL(_ id: String, _ file: String) -> URL {
        base.appendingPathComponent(id).appendingPathComponent("resolve/main").appendingPathComponent(file)
    }

    static func check(_ resp: URLResponse) throws {
        if let h = resp as? HTTPURLResponse, !(200..<300).contains(h.statusCode) {
            throw NSError(domain: "HuggingFace", code: h.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "Hugging Face returned HTTP \(h.statusCode)"])
        }
    }
}

// MARK: - Downloads

/// Downloads every required file of a repo into the local model store.
///
/// Each file streams into `.download/<name>.partial`; an interrupted download (even a killed process)
/// resumes with `Range: bytes=N-`. LFS files are SHA-256 verified before being moved into place, and a
/// `.download/<name>.sha256ok` marker records the verified hash so later runs can skip them cheaply.
/// All state lives on one serial queue (also the URLSession delegate queue).
public final class ModelDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    public let repo: String
    public let destination: URL
    private let files: [HubFile]
    private let total: Int64
    private var completedBytes: Int64 = 0
    private var currentBytes: Int64 = 0
    private var sessionBytes: Int64 = 0
    private var session: URLSession!
    private var index = 0
    private var task: URLSessionDataTask?
    private var handle: FileHandle?
    private var requestedOffset: Int64 = 0
    private var allowRange = true
    private var retryPending = false
    private var resumedThisFile = false
    private let queue = DispatchQueue(label: "de.21no.mlxrunner.download", qos: .utility)
    private var startTime = Date()
    // Guarded by `lock` (touched from cancel()).
    private let lock = NSLock()
    private var continuation: CheckedContinuation<URL, Error>?
    private var stopped = false

    public var onProgress: (@Sendable (Double, Int64, Int64, Double) -> Void)?
    /// Human-readable status lines ("Resuming …", "Verifying …").
    public var onEvent: (@Sendable (String) -> Void)?

    public init(details: HubModelDetails, destination: URL) {
        repo = details.id
        self.destination = destination
        files = details.requiredFiles.sorted { ($0.size ?? 0) < ($1.size ?? 0) }
        total = Int64(details.downloadSize)
        super.init()
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 60 * 60 * 24
        cfg.urlCache = nil
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.httpAdditionalHeaders = ["User-Agent": "MLXRunner/1.0 (macOS; Swift)"]
        let ops = OperationQueue()
        ops.maxConcurrentOperationCount = 1
        ops.underlyingQueue = queue
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: ops)
    }

    private var workDir: URL { destination.appendingPathComponent(".download", isDirectory: true) }
    private func partialURL(_ f: HubFile) -> URL { workDir.appendingPathComponent(f.rfilename + ".partial") }
    private func markerURL(_ f: HubFile) -> URL { workDir.appendingPathComponent(f.rfilename + ".sha256ok") }
    private func destURL(_ f: HubFile) -> URL { destination.appendingPathComponent(f.rfilename) }

    public func start() async throws -> URL {
        try RepoID.validate(repo)
        for f in files where !RepoID.isSafeFileName(f.rfilename) { throw HubError.invalidFileName(f.rfilename) }
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        startTime = Date()
        return try await withCheckedThrowingContinuation { cont in
            lock.lock(); continuation = cont; lock.unlock()
            queue.async { self.next() }
        }
    }

    /// Stops the download. Partial bytes are kept so the next attempt resumes.
    public func cancel() {
        lock.lock()
        stopped = true
        let c = continuation
        continuation = nil
        lock.unlock()
        session.invalidateAndCancel()
        c?.resume(throwing: CancellationError())
    }

    private var isStopped: Bool { lock.lock(); defer { lock.unlock() }; return stopped }

    private func next() {
        guard !isStopped else { return }
        while index < files.count, isInstalled(files[index]) {
            completedBytes += Int64(files[index].size ?? 0)
            index += 1
            report()
        }
        if index >= files.count {
            try? Data().write(to: destination.appendingPathComponent(".complete"))
            try? repo.write(to: destination.appendingPathComponent(".repo"), atomically: true, encoding: .utf8)
            lock.lock()
            let c = continuation
            continuation = nil
            stopped = true
            lock.unlock()
            session.finishTasksAndInvalidate()
            c?.resume(returning: destination)
            return
        }
        resumedThisFile = false
        allowRange = true
        startFile(files[index])
    }

    /// True when the final file is present with the right size and (for LFS files) a verified hash.
    private func isInstalled(_ f: HubFile) -> Bool {
        let dest = destURL(f)
        guard let want = f.size, Self.fileSize(dest) == Int64(want) else { return false }
        guard let sha = f.sha256 else { return true }
        let marker = (try? String(contentsOf: markerURL(f), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
        if marker == sha { return true }
        // Installed by an older build (no marker) or upstream changed: verify once.
        onEvent?("Verifying existing \(f.rfilename) (SHA-256)…")
        if let actual = try? FileHasher.sha256(of: dest), actual == sha {
            try? sha.write(to: markerURL(f), atomically: true, encoding: .utf8)
            return true
        }
        try? FileManager.default.removeItem(at: dest)
        try? FileManager.default.removeItem(at: markerURL(f))
        return false
    }

    private func startFile(_ f: HubFile) {
        guard !isStopped else { return }
        let partial = partialURL(f)
        var plan = ResumePlan.start(partialBytes: Self.fileSize(partial) ?? 0, expectedSize: f.size)
        if !allowRange, plan != .verifyOnly { plan = .fresh }
        switch plan {
        case .verifyOnly:
            currentBytes = Self.fileSize(partial) ?? 0
            resumedThisFile = true   // kept bytes: a hash mismatch gets one automatic fresh retry
            finalize(f)
            return
        case .fresh:
            try? FileManager.default.removeItem(at: partial)
            requestedOffset = 0
        case .resume(let n):
            requestedOffset = n
        }
        currentBytes = requestedOffset
        var req = HubClient.request(HubClient.fileURL(repo, f.rfilename))
        if requestedOffset > 0 { req.setValue("bytes=\(requestedOffset)-", forHTTPHeaderField: "Range") }
        let t = session.dataTask(with: req)
        task = t
        t.resume()
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                           newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(HubAuthPolicy.redirectedRequest(request, from: response.url ?? task.currentRequest?.url,
                                                          original: task.originalRequest))
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                           completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard !isStopped, dataTask === task, index < files.count else { completionHandler(.cancel); return }
        let f = files[index]
        let partial = partialURL(f)
        let http = response as? HTTPURLResponse
        let status = http?.statusCode ?? 200
        let decision = ResumePlan.response(status: status, requestedOffset: requestedOffset,
                                           contentRange: http?.value(forHTTPHeaderField: "Content-Range"))
        do {
            switch decision {
            case .append:
                let h = try FileHandle(forWritingTo: partial)
                try h.truncate(atOffset: UInt64(requestedOffset))
                try h.seekToEnd()
                handle = h
                resumedThisFile = true
                onEvent?("Resuming \(f.rfilename) from byte \(requestedOffset) (HTTP 206)")
            case .restart:
                if requestedOffset > 0 { onEvent?("Server ignored the range request for \(f.rfilename) (HTTP \(status)); restarting it") }
                FileManager.default.createFile(atPath: partial.path, contents: nil)
                handle = try FileHandle(forWritingTo: partial)
                requestedOffset = 0
                currentBytes = 0
            case .retryWithoutRange:
                try? FileManager.default.removeItem(at: partial)
                guard allowRange else {
                    // Already retried without Range and the server still answers oddly: give up instead of looping.
                    completionHandler(.cancel)
                    fail(HubError.http(file: f.rfilename, status: status))
                    return
                }
                onEvent?("Unusable range response for \(f.rfilename) (HTTP \(status)); restarting it")
                allowRange = false
                retryPending = true
                completionHandler(.cancel)
                return
            case .fail(let code):
                completionHandler(.cancel)
                fail(HubError.http(file: f.rfilename, status: code))
                return
            }
        } catch {
            completionHandler(.cancel)
            fail(error)
            return
        }
        completionHandler(.allow)
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard dataTask === task, let handle else { return }
        do {
            try handle.write(contentsOf: data)
        } catch {
            dataTask.cancel()
            fail(error)
            return
        }
        currentBytes += Int64(data.count)
        sessionBytes += Int64(data.count)
        report()
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard task === self.task else { return }
        try? handle?.close()
        handle = nil
        self.task = nil
        if retryPending {
            retryPending = false
            if index < files.count { startFile(files[index]) }
            return
        }
        if let error { fail(error); return }   // partial bytes stay on disk for the next attempt
        guard !isStopped, index < files.count else { return }
        finalize(files[index])
    }

    /// Checks size and hash of the finished partial, then moves it into place. Runs on `queue`.
    private func finalize(_ f: HubFile) {
        let fm = FileManager.default
        let partial = partialURL(f)
        let size = Self.fileSize(partial) ?? 0
        if let want = f.size, size != Int64(want) {
            if size > Int64(want) { try? fm.removeItem(at: partial) }
            fail(HubError.sizeMismatch(file: f.rfilename, expected: want, actual: Int(size)))
            return
        }
        if let sha = f.sha256 {
            onEvent?("Verifying \(f.rfilename) (SHA-256)…")
            let actual: String
            do { actual = try FileHasher.sha256(of: partial) } catch { fail(error); return }
            guard actual == sha else {
                try? fm.removeItem(at: partial)
                try? fm.removeItem(at: markerURL(f))
                if resumedThisFile {
                    // The kept bytes may be from an older revision: retry once from scratch.
                    onEvent?("\(f.rfilename) failed verification after resuming; downloading it again from scratch")
                    resumedThisFile = false
                    allowRange = false
                    startFile(f)
                    return
                }
                fail(HubError.checksumMismatch(file: f.rfilename, expected: sha, actual: actual))
                return
            }
            onEvent?("Verified \(f.rfilename) sha256 \(sha.prefix(16))…")
        }
        let dest = destURL(f)
        do {
            try? fm.removeItem(at: dest)
            try fm.moveItem(at: partial, to: dest)
        } catch { fail(error); return }
        if let sha = f.sha256 {
            try? sha.write(to: markerURL(f), atomically: true, encoding: .utf8)
        } else {
            try? fm.removeItem(at: markerURL(f))
        }
        completedBytes += Int64(f.size ?? Int(size))
        currentBytes = 0
        index += 1
        report()
        next()
    }

    private func report() {
        let done = completedBytes + currentBytes
        let elapsed = Date().timeIntervalSince(startTime)
        let speed = Double(sessionBytes) / max(elapsed, 0.001)
        onProgress?(total > 0 ? min(1, Double(done) / Double(total)) : 0, done, total, speed)
    }

    private func fail(_ error: Error) {
        lock.lock()
        stopped = true
        let c = continuation
        continuation = nil
        lock.unlock()
        try? handle?.close()
        handle = nil
        session.invalidateAndCancel()
        c?.resume(throwing: error)
    }

    static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }
}
