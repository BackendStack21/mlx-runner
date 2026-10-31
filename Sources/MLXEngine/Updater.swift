import CryptoKit
import Foundation

// Self-update, verified end to end (a port of Codeviewer's updater):
//
//   1. The latest release is read from GitHub (api.github.com, over TLS). Drafts and
//      pre-releases are ignored; the answer is cached for an hour unless forced.
//   2. SHA256SUMS.sig must be a valid Ed25519 signature of SHA256SUMS by the release key
//      (the private half lives only in the release workflow's secrets; the public half is
//      `UpdateKey.publicKeyBase64`). A compromised download host can't forge it.
//   3. The downloaded file's SHA-256 must match its line in the signed SHA256SUMS, and the
//      signed file name must carry the new version (no rollbacks, no mix-ups).
//   4. The new version must be strictly newer than the running one.
//   5. The staged app must pass `codesign --verify --deep --strict` and carry the expected
//      bundle identifier and version.
//   6. The swap is atomic (rename) with rollback; nothing from the download is executed
//      before every check has passed.

/// What the updater is doing, reported to the UI.
public enum UpdatePhase: Equatable, Sendable {
    case checking
    case signature
    /// Download progress, 0...1 (0 while the size is unknown).
    case download(Double)
    case checksum
    case install
    /// The version that was installed.
    case done(String)
}

/// The latest release compared with the running version.
public struct UpdateInfo: Equatable, Sendable {
    public var current: String
    public var latest: String?
    public var available: Bool
    public var canInstall: Bool
    /// Why an available update can't be installed automatically.
    public var reason: String?
    /// The release page.
    public var url: URL?
    /// Release notes (Markdown, truncated).
    public var notes: String
    public var checkedAt: Date
}

/// How this copy of MLX Runner is installed.
public struct UpdateInstall: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        /// A `.app` bundle: the whole bundle is replaced.
        case macApp
        /// `swift run` / an unbundled build: updated with git.
        case source
    }

    public var kind: Kind
    /// The `.app` bundle that gets replaced.
    public var target: URL

    public init(kind: Kind, target: URL) {
        self.kind = kind
        self.target = target
    }

    /// Works out how the running app was installed from its bundle location.
    public static func detect(bundleURL: URL = Bundle.main.bundleURL) -> UpdateInstall {
        let url = bundleURL.standardizedFileURL
        return UpdateInstall(kind: url.pathExtension == "app" ? .macApp : .source, target: url)
    }

    /// The release asset for a version.
    public static func assetName(for version: String) -> String {
        "MLX-Runner-\(version)-macos-arm64.dmg"
    }
}

/// The result of a successful `Updater.apply`.
public struct AppliedUpdate: Equatable, Sendable {
    public var version: String
    public var target: URL
}

/// An update failure with a user-facing message.
public struct UpdateError: LocalizedError, Equatable, Sendable {
    public var message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// Turns a verified, downloaded package into the installed app. `MacAppInstaller` is the real
/// one; tests stub it.
public protocol UpdateInstalling: Sendable {
    /// `package` is the verified download inside `stage`, a private directory on the same
    /// volume as `target` that the updater deletes afterwards.
    func install(package: URL, version: String, target: URL, stage: URL) async throws
}

public actor Updater {
    /// Hosts release assets may be downloaded from, always over https. GitHub's redirects are
    /// followed, but only to these hosts.
    public static let assetHosts: Set<String> = ["github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"]
    /// SHA256SUMS and its signature are tiny; a hostile host must not make us buffer gigabytes.
    public static let maxSumsBytes = 1 << 20
    /// No release asset is anywhere near this; the download stops here before any verification.
    public static let maxAssetBytes = 1 << 30
    static let maxAPIBytes = 8 << 20
    static let cacheLifetime: TimeInterval = 3600

    public nonisolated let currentVersion: String
    public nonisolated let install: UpdateInstall
    let publicKeyBase64: String
    let apiURL: URL
    let hosts: Set<String>
    let configuration: URLSessionConfiguration
    let installer: any UpdateInstalling
    let now: @Sendable () -> Date

    struct Release {
        var version: String
        var url: URL?
        var notes: String
        var assets: [String: String]
    }

    private var cached: (info: UpdateInfo, release: Release?)?
    private var busy = false

    /// Every dependency is injectable for tests; the defaults are the real app's.
    /// - Parameter publicKey: base64 of the raw 32-byte Ed25519 public key (default: the
    ///   key compiled into this build, `UpdateKey.publicKeyBase64`).
    public init(
        currentVersion: String = MLXRunnerVersion.current,
        publicKey: String? = nil,
        install: UpdateInstall = .detect(),
        apiURL: URL? = nil,
        assetHosts: Set<String>? = nil,
        configuration: URLSessionConfiguration = .ephemeral,
        installer: any UpdateInstalling = MacAppInstaller(),
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.currentVersion = currentVersion
        self.publicKeyBase64 = publicKey ?? UpdateKey.publicKeyBase64
        self.install = install
        self.apiURL = apiURL ?? URL(string: "https://api.github.com/repos/\(MLXRunnerVersion.repository)/releases/latest")!
        self.hosts = assetHosts ?? Self.assetHosts
        self.configuration = configuration
        self.installer = installer
        self.now = now
    }

    public var isInstalling: Bool { busy }

    // MARK: Pure checks (static so they can be tested directly)

    /// Compares "1.2.3" style versions: negative if a < b, 0 if equal, positive if a > b.
    /// A pre-release sorts before its release; pre-release identifiers follow semver §11.
    public static func compareVersions(_ a: String, _ b: String) -> Int {
        func parse(_ v: String) -> (nums: [Int], pre: String) {
            var base = Substring(v)
            if base.hasPrefix("v") { base = base.dropFirst() }
            if let plus = base.firstIndex(of: "+") { base = base[..<plus] }  // build metadata never orders
            let dash = base.firstIndex(of: "-")
            let core = dash.map { base[..<$0] } ?? base
            let pre = dash.map { String(base[base.index(after: $0)...]) } ?? ""
            return (core.split(separator: ".", omittingEmptySubsequences: false).map { Int($0) ?? 0 }, pre)
        }
        let x = parse(a), y = parse(b)
        for i in 0..<3 {
            let p = i < x.nums.count ? x.nums[i] : 0
            let q = i < y.nums.count ? y.nums[i] : 0
            if p != q { return p < q ? -1 : 1 }
        }
        if x.pre == y.pre { return 0 }
        if x.pre.isEmpty { return 1 }
        if y.pre.isEmpty { return -1 }
        let xs = x.pre.split(separator: ".", omittingEmptySubsequences: false)
        let ys = y.pre.split(separator: ".", omittingEmptySubsequences: false)
        func isNumeric(_ s: Substring) -> Bool { !s.isEmpty && s.allSatisfy { $0.isASCII && $0.isNumber } }
        for i in 0..<max(xs.count, ys.count) {
            if i >= xs.count { return -1 }  // a prefix sorts first
            if i >= ys.count { return 1 }
            let p = xs[i], q = ys[i]
            let pn = isNumeric(p), qn = isNumeric(q)
            if pn && qn {
                // Compare digit strings without converting (no overflow): by length, then lexically.
                let ps = p.drop { $0 == "0" }, qs = q.drop { $0 == "0" }
                if ps.count != qs.count { return ps.count < qs.count ? -1 : 1 }
                if ps != qs { return ps < qs ? -1 : 1 }
            } else if pn != qn {
                return pn ? -1 : 1  // numeric identifiers sort before alphanumeric ones
            } else if p != q {
                return p < q ? -1 : 1
            }
        }
        return 0
    }

    /// Whether a release tag (without "v") is a version we accept: x.y.z[-pre][+build], made of
    /// characters that are safe in a file name.
    public static func isValidVersion(_ v: String) -> Bool {
        v.wholeMatch(of: try! Regex(#"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?"#)) != nil
    }

    /// Parses a SHA256SUMS file (`<64 hex>  [*]name` lines) into name → lowercase hex digest.
    public static func parseSums(_ text: String) -> [String: String] {
        var out: [String: String] = [:]
        for raw in text.split(omittingEmptySubsequences: true, whereSeparator: { $0 == "\n" || $0 == "\r\n" }) {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.utf8.count >= 66 else { continue }
            let hex = line.prefix(64)
            guard hex.allSatisfy({ $0.isHexDigit && $0.isASCII }) else { continue }
            var rest = line.dropFirst(64)
            guard let first = rest.first, first == " " || first == "\t" else { continue }
            rest = rest.drop { $0 == " " || $0 == "\t" }
            if rest.first == "*" { rest = rest.dropFirst() }
            guard let c = rest.first, !c.isWhitespace else { continue }
            out[String(rest)] = hex.lowercased()
        }
        return out
    }

    /// Decodes a base64 raw Ed25519 public key; nil when it is empty or malformed.
    public static func publicKey(fromBase64 b64: String) -> Curve25519.Signing.PublicKey? {
        let t = b64.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, let raw = Data(base64Encoded: t), raw.count == 32 else { return nil }
        return try? Curve25519.Signing.PublicKey(rawRepresentation: raw)
    }

    /// Verifies an Ed25519 signature (raw 64 bytes, or base64 of them) over `data`.
    public static func verifySignature(publicKeyBase64: String, data: Data, signature: Data) -> Bool {
        guard let key = publicKey(fromBase64: publicKeyBase64) else { return false }
        var sig = signature
        if sig.count != 64 {
            guard let text = String(data: sig, encoding: .utf8),
                  let decoded = Data(base64Encoded: text.trimmingCharacters(in: .whitespacesAndNewlines))
            else { return false }
            sig = decoded
        }
        guard sig.count == 64 else { return false }
        return key.isValidSignature(sig, for: data)
    }

    /// Release downloads come only from the allowed hosts, over TLS (initial URLs and redirects).
    public static func isAllowedDownloadURL(_ url: URL, hosts: Set<String> = assetHosts) -> Bool {
        guard url.scheme?.lowercased() == "https", let host = url.host?.lowercased(), url.user == nil, url.password == nil
        else { return false }
        return hosts.contains(host)
    }

    // MARK: Check

    /// Latest release vs. the running version. Cached for an hour unless forced.
    public func check(force: Bool = false) async throws -> UpdateInfo {
        if !force, let c = cached, now().timeIntervalSince(c.info.checkedAt) < Self.cacheLifetime { return c.info }
        let base = UpdateInfo(current: currentVersion, latest: nil, available: false, canInstall: false,
                              reason: nil, url: nil, notes: "", checkedAt: now())
        guard apiURL.scheme == "https", let apiHost = apiURL.host?.lowercased() else {
            throw UpdateError("The update URL must be https.")
        }
        let body: Data
        do {
            body = try await fetch(apiURL, hosts: [apiHost], limit: Self.maxAPIBytes, accept: "application/vnd.github+json")
        } catch let e as HTTPStatusError {
            throw UpdateError("Update check failed: GitHub answered \(e.status).")
        }
        struct GHAsset: Decodable { var name: String; var browser_download_url: String }
        struct GHRelease: Decodable {
            var tag_name: String?
            var html_url: String?
            var body: String?
            var draft: Bool?
            var prerelease: Bool?
            var assets: [GHAsset]?
        }
        guard let rel = try? JSONDecoder().decode(GHRelease.self, from: body) else {
            throw UpdateError("Update check failed: unexpected answer from GitHub.")
        }
        var version = rel.tag_name ?? ""
        if version.hasPrefix("v") { version.removeFirst() }
        if !Self.isValidVersion(version) || rel.draft == true || rel.prerelease == true {
            cached = (base, nil)
            return base
        }
        let pageURL = rel.html_url.flatMap(URL.init(string:)).flatMap { u in
            u.scheme == "https" && u.host?.lowercased() == "github.com" ? u : nil
        } ?? URL(string: "https://github.com/\(MLXRunnerVersion.repository)/releases")
        var assets: [String: String] = [:]
        for a in rel.assets ?? [] { assets[a.name] = a.browser_download_url }
        let release = Release(version: version, url: pageURL, notes: String((rel.body ?? "").prefix(8000)), assets: assets)
        var info = base
        info.latest = version
        info.available = Self.compareVersions(version, currentVersion) > 0
        info.url = release.url
        info.notes = release.notes
        if info.available {
            info.reason = cannotInstall(release)
            info.canInstall = info.reason == nil
        }
        cached = (info, release)
        return info
    }

    private func cannotInstall(_ r: Release) -> String? {
        if install.kind == .source { return "Running from source: update with git pull." }
        let name = UpdateInstall.assetName(for: r.version)
        if r.assets[name] == nil { return "This release has no build for this Mac (\(name))." }
        if r.assets["SHA256SUMS"] == nil || r.assets["SHA256SUMS.sig"] == nil {
            return "This release isn't signed, so it can't be installed automatically."
        }
        if Self.publicKey(fromBase64: publicKeyBase64) == nil { return "This build has no update signing key." }
        let path = install.target.path
        if path.contains("/AppTranslocation/") || path.hasPrefix("/Volumes/") {
            return "Move MLX Runner to the Applications folder to update it."
        }
        let parent = install.target.deletingLastPathComponent().path
        if !FileManager.default.isWritableFile(atPath: parent) || !FileManager.default.isWritableFile(atPath: path) {
            return "No permission to replace \(path)."
        }
        return nil
    }

    // MARK: Apply

    /// Downloads, verifies and installs the latest release, reporting each phase. Returns the
    /// installed version; the caller restarts the app (`Updater.relaunch`).
    public func apply(progress: @escaping @Sendable (UpdatePhase) -> Void = { _ in }) async throws -> AppliedUpdate {
        guard !busy else { throw UpdateError("An update is already being installed.") }
        busy = true
        defer { busy = false }

        progress(.checking)
        let info = try await check(force: true)
        guard info.available, let release = cached?.release else { throw UpdateError("MLX Runner is already up to date.") }
        guard info.canInstall else { throw UpdateError(info.reason ?? "This update can't be installed automatically.") }
        let name = UpdateInstall.assetName(for: release.version)
        guard let assetURL = release.assets[name], let sumsURL = release.assets["SHA256SUMS"],
              let sigURL = release.assets["SHA256SUMS.sig"] else { throw UpdateError("The release is missing files.") }

        // 1. Signed checksum list.
        progress(.signature)
        let sums = try await fetchAsset(sumsURL, limit: Self.maxSumsBytes)
        let sig = try await fetchAsset(sigURL, limit: Self.maxSumsBytes)
        guard Self.verifySignature(publicKeyBase64: publicKeyBase64, data: sums, signature: sig) else {
            throw UpdateError("SHA256SUMS signature is invalid: refusing to install.")
        }
        guard let expected = Self.parseSums(String(decoding: sums, as: UTF8.self))[name] else {
            throw UpdateError("\(name) is not in the signed checksum list.")
        }
        guard name.contains("-\(release.version)-") else {
            throw UpdateError("The signed file name doesn't match the release version.")
        }

        // 2. Download into a staging dir next to the target (same volume → atomic rename).
        let target = install.target
        let stage = try Self.makeStage(next: target)
        defer { try? FileManager.default.removeItem(at: stage) }
        let file = stage.appendingPathComponent(name)
        let actual = try await download(assetURL, to: file) { progress(.download($0)) }

        // 3. Checksum.
        progress(.checksum)
        guard actual == expected else {
            throw UpdateError("Checksum mismatch for \(name): expected \(expected.prefix(12))…, got \(actual.prefix(12))…")
        }

        // 4. Install.
        progress(.install)
        try await installer.install(package: file, version: release.version, target: target, stage: stage)
        cached = nil
        progress(.done(release.version))
        return AppliedUpdate(version: release.version, target: target)
    }

    /// Creates `.mlxrunner-update-XXXXXX` next to `target` (so renames stay on one volume).
    static func makeStage(next target: URL) throws -> URL {
        var template = Array(target.deletingLastPathComponent().appendingPathComponent(".mlxrunner-update-XXXXXX").path.utf8CString)
        guard let p = mkdtemp(&template) else {
            throw UpdateError("Could not create a staging folder next to \(target.path): \(String(cString: strerror(errno))).")
        }
        return URL(fileURLWithPath: String(cString: p), isDirectory: true)
    }

    // MARK: Relaunch

    /// Starts the app at `target` again once this process has exited (a detached shell waits a
    /// second, then `open -n`s it). The caller terminates the app right after.
    public static func relaunch(_ target: URL) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        // The path is passed as $0, never interpolated into the script.
        p.arguments = ["-c", "sleep 1; /usr/bin/open -n \"$0\"", target.path]
        p.standardInput = FileHandle.nullDevice
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        try p.run()
    }

    // MARK: Networking

    struct HTTPStatusError: Error { var status: Int }

    private func fetchAsset(_ string: String, limit: Int) async throws -> Data {
        guard let url = URL(string: string) else { throw UpdateError("A release file has an invalid URL.") }
        guard Self.isAllowedDownloadURL(url, hosts: hosts) else {
            throw UpdateError("Refusing to download from \(url.host ?? url.absoluteString).")
        }
        do {
            return try await fetch(url, hosts: hosts, limit: limit, accept: nil)
        } catch let e as HTTPStatusError {
            throw UpdateError("Download failed (\(e.status)): \(url.lastPathComponent)")
        }
    }

    /// Downloads a small file into memory, refusing more than `limit` bytes.
    private func fetch(_ url: URL, hosts: Set<String>, limit: Int, accept: String?) async throws -> Data {
        let sink = DataSink()
        try await stream(url, hosts: hosts, limit: limit, accept: accept) { chunk, _, _ in sink.data.append(chunk) }
        return sink.data
    }

    /// Streams a release asset to disk, hashing as it goes. Returns the lowercase hex SHA-256.
    private func download(_ string: String, to dest: URL, progress: @escaping @Sendable (Double) -> Void) async throws -> String {
        guard let url = URL(string: string) else { throw UpdateError("The release asset has an invalid URL.") }
        guard Self.isAllowedDownloadURL(url, hosts: hosts) else {
            throw UpdateError("Refusing to download from \(url.host ?? url.absoluteString).")
        }
        guard FileManager.default.createFile(atPath: dest.path, contents: nil),
              let handle = try? FileHandle(forWritingTo: dest) else {
            throw UpdateError("Could not write the download to \(dest.deletingLastPathComponent().path).")
        }
        defer { try? handle.close() }
        let sink = HashSink(handle: handle)
        progress(0)
        do {
            try await stream(url, hosts: hosts, limit: Self.maxAssetBytes, accept: "application/octet-stream") { chunk, got, total in
                try sink.handle.write(contentsOf: chunk)
                sink.hasher.update(data: chunk)
                let pct = total > 0 ? min(100, Int(got * 100 / total)) : 0
                if pct != sink.lastPercent {
                    sink.lastPercent = pct
                    progress(Double(pct) / 100)
                }
            }
        } catch let e as HTTPStatusError {
            throw UpdateError("Download failed (\(e.status)): \(url.lastPathComponent)")
        }
        try handle.synchronize()
        return sink.hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// GETs `url`, following redirects only to https `hosts`, handing each body chunk to
    /// `onChunk(chunk, bytesSoFar, expectedTotal)` and failing past `limit` bytes.
    private func stream(_ url: URL, hosts: Set<String>, limit: Int, accept: String?,
                        onChunk: @escaping (Data, Int64, Int64) throws -> Void) async throws {
        var request = URLRequest(url: url)
        request.setValue("MLXRunner/\(currentVersion)", forHTTPHeaderField: "User-Agent")
        if let accept { request.setValue(accept, forHTTPHeaderField: "Accept") }
        let delegate = StreamDelegate(hosts: hosts, limit: Int64(limit), onChunk: onChunk)
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: queue)
        defer { session.finishTasksAndInvalidate() }
        let task = session.dataTask(with: request)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                delegate.continuation = c
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }
}

private final class DataSink: @unchecked Sendable {
    var data = Data()
}

private final class HashSink: @unchecked Sendable {
    let handle: FileHandle
    var hasher = SHA256()
    var lastPercent = -1
    init(handle: FileHandle) { self.handle = handle }
}

/// Feeds one data task's body to a closure on a serial delegate queue, enforcing the redirect
/// host policy and the size cap.
private final class StreamDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let hosts: Set<String>
    let limit: Int64
    let onChunk: (Data, Int64, Int64) throws -> Void
    var continuation: CheckedContinuation<Void, Error>?
    // Only touched on the session's serial delegate queue.
    private var failure: Error?
    private var received: Int64 = 0
    private var expected: Int64 = 0

    init(hosts: Set<String>, limit: Int64, onChunk: @escaping (Data, Int64, Int64) throws -> Void) {
        self.hosts = hosts
        self.limit = limit
        self.onChunk = onChunk
    }

    private var tooLarge: UpdateError {
        limit <= Int64(Updater.maxSumsBytes) ? UpdateError("The checksum file is too large.") : UpdateError("The release asset is too large.")
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard let url = request.url, Updater.isAllowedDownloadURL(url, hosts: hosts) else {
            failure = UpdateError("Refusing to follow a redirect to \(request.url?.host ?? "an invalid URL").")
            completionHandler(nil)
            task.cancel()
            return
        }
        completionHandler(request)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard failure == nil else { return completionHandler(.cancel) }
        guard let http = response as? HTTPURLResponse else {
            failure = UpdateError("Unexpected response from \(response.url?.host ?? "the server").")
            return completionHandler(.cancel)
        }
        guard (200..<300).contains(http.statusCode) else {
            failure = Updater.HTTPStatusError(status: http.statusCode)
            return completionHandler(.cancel)
        }
        // Belt and braces: the final URL must still satisfy the host policy.
        if let url = http.url, !Updater.isAllowedDownloadURL(url, hosts: hosts) {
            failure = UpdateError("Refusing to download from \(url.host ?? url.absoluteString).")
            return completionHandler(.cancel)
        }
        if http.expectedContentLength > limit {
            failure = tooLarge
            return completionHandler(.cancel)
        }
        expected = max(0, http.expectedContentLength)
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard failure == nil else { return }
        received += Int64(data.count)
        do {
            if received > limit { throw tooLarge }
            try onChunk(data, received, expected)
        } catch {
            failure = error
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let c = continuation
        continuation = nil
        if let failure {
            c?.resume(throwing: failure)
        } else if let error {
            c?.resume(throwing: (error as? URLError)?.code == .cancelled ? CancellationError() : error)
        } else {
            c?.resume()
        }
    }
}

// MARK: - macOS app installer

/// Mounts the DMG, copies the app out (signature intact), verifies it, and swaps it into place.
public struct MacAppInstaller: UpdateInstalling {
    public var appName: String
    public var bundleIdentifier: String

    public init(appName: String = "MLX Runner.app", bundleIdentifier: String = "de.21no.mlxrunner") {
        self.appName = appName
        self.bundleIdentifier = bundleIdentifier
    }

    public func install(package: URL, version: String, target: URL, stage: URL) async throws {
        // hdiutil, ditto and codesign block: keep them off the cooperative thread pool.
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                c.resume(with: Result { try installSync(package: package, version: version, target: target, stage: stage) })
            }
        }
    }

    func installSync(package: URL, version: String, target: URL, stage: URL) throws {
        let staged = try stageApp(dmg: package, stage: stage)
        try verify(staged, version: version)
        try Self.swap(target: target, staged: staged, stage: stage)
    }

    /// Mounts the disk image inside `stage` and copies the app out of it.
    func stageApp(dmg: URL, stage: URL) throws -> URL {
        let mnt = stage.appendingPathComponent("mnt", isDirectory: true)
        try FileManager.default.createDirectory(at: mnt, withIntermediateDirectories: false)
        try Self.run("/usr/bin/hdiutil", ["attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", mnt.path, dmg.path])
        defer {
            if (try? Self.run("/usr/bin/hdiutil", ["detach", mnt.path, "-quiet"])) == nil {
                _ = try? Self.run("/usr/bin/hdiutil", ["detach", mnt.path, "-force", "-quiet"])
            }
        }
        let source = mnt.appendingPathComponent(appName, isDirectory: true)
        // Must be a real directory on the image, not a symlink to somewhere else.
        let type = (try? FileManager.default.attributesOfItem(atPath: source.path))?[.type] as? FileAttributeType
        guard type == .typeDirectory else { throw UpdateError("The disk image has no \(appName).") }
        let staged = stage.appendingPathComponent(appName, isDirectory: true)
        try Self.run("/usr/bin/ditto", [source.path, staged.path])
        return staged
    }

    /// The staged app must be validly signed and be the expected app at the expected version.
    func verify(_ staged: URL, version: String) throws {
        do {
            try Self.run("/usr/bin/codesign", ["--verify", "--deep", "--strict", staged.path])
        } catch let e as UpdateError {
            throw UpdateError("The new app's code signature is invalid: \(e.message)")
        }
        // Read the plist directly: Bundle(url:) caches by path.
        let plistURL = staged.appendingPathComponent("Contents/Info.plist")
        guard let data = try? Data(contentsOf: plistURL),
              let plist = (try? PropertyListSerialization.propertyList(from: data, format: nil)) as? [String: Any]
        else { throw UpdateError("The new app has no readable Info.plist.") }
        guard plist["CFBundleIdentifier"] as? String == bundleIdentifier else {
            throw UpdateError("The new app has an unexpected bundle identifier.")
        }
        guard plist["CFBundleShortVersionString"] as? String == version else {
            throw UpdateError("The new app's version doesn't match the release (\(version)).")
        }
    }

    /// Atomically replaces `target` with `staged` (both on one volume); restores the original
    /// if the second rename fails. The previous app is left in `stage/previous`.
    public static func swap(target: URL, staged: URL, stage: URL) throws {
        let old = stage.appendingPathComponent("previous")
        guard rename(target.path, old.path) == 0 else {
            throw UpdateError("Could not move the current app aside: \(String(cString: strerror(errno))).")
        }
        guard rename(staged.path, target.path) == 0 else {
            let err = String(cString: strerror(errno))
            _ = rename(old.path, target.path)
            throw UpdateError("Could not move the new app into place: \(err).")
        }
    }

    /// Runs a tool by absolute path; throws with the first line of its output on failure.
    /// Output goes to a temp file (not a pipe): hdiutil's helpers can hold a pipe open.
    @discardableResult
    static func run(_ tool: String, _ args: [String]) throws -> String {
        let log = FileManager.default.temporaryDirectory.appendingPathComponent("mlxrunner-update-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: log) }
        let out = try FileHandle(forWritingTo: log)
        defer { try? out.close() }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: tool)
        p.arguments = args
        p.standardInput = FileHandle.nullDevice
        p.standardOutput = out
        p.standardError = out
        try p.run()
        p.waitUntilExit()
        let text = (try? String(contentsOf: log, encoding: .utf8)) ?? ""
        guard p.terminationStatus == 0 else {
            let first = text.split(separator: "\n").first.map(String.init) ?? "exit \(p.terminationStatus)"
            throw UpdateError("\((tool as NSString).lastPathComponent) failed: \(first.trimmingCharacters(in: .whitespaces))")
        }
        return text
    }
}
