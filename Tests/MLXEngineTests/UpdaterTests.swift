import CryptoKit
import Foundation
import Testing
@testable import MLXEngine

// MARK: - URLProtocol stub (no test touches the network)

/// Serves canned responses to a URLSession. Each test gets its own route table, selected by
/// the `X-Stub` header its session configuration adds, so suites can run in parallel.
final class UpdateStub: URLProtocol, @unchecked Sendable {
    enum Route {
        case body(Int, Data, extraHeaders: [String: String] = [:])
        case redirect(String)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var tables: [String: [String: Route]] = [:]
    nonisolated(unsafe) private static var logs: [String: [String]] = [:]

    /// A session configuration routed to `routes` (keyed by absolute URL string).
    static func configuration(_ routes: [String: Route]) -> (URLSessionConfiguration, id: String) {
        let id = UUID().uuidString
        lock.withLock { tables[id] = routes }
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [UpdateStub.self]
        c.httpAdditionalHeaders = ["X-Stub": id]
        return (c, id)
    }

    static func setRoute(_ id: String, _ url: String, _ route: Route) { lock.withLock { tables[id]?[url] = route } }
    static func requests(_ id: String) -> [String] { lock.withLock { logs[id] ?? [] } }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        let id = request.value(forHTTPHeaderField: "X-Stub") ?? ""
        let route: Route? = Self.lock.withLock {
            Self.logs[id, default: []].append(url.absoluteString)
            return Self.tables[id]?[url.absoluteString]
        }
        switch route {
        case .redirect(let to)?:
            let resp = HTTPURLResponse(url: url, statusCode: 302, httpVersion: "HTTP/1.1", headerFields: ["Location": to])!
            var next = URLRequest(url: URL(string: to)!)
            next.setValue(id, forHTTPHeaderField: "X-Stub")
            client?.urlProtocol(self, wasRedirectedTo: next, redirectResponse: resp)
        case .body(let status, let data, let extra)?:
            var headers = ["Content-Length": "\(data.count)"]
            headers.merge(extra) { _, b in b }
            let resp = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            var i = 0
            while i < data.count {  // several chunks, like a real download
                let end = min(data.count, i + 4096)
                client?.urlProtocol(self, didLoad: data.subdata(in: i..<end))
                i = end
            }
            client?.urlProtocolDidFinishLoading(self)
        case nil:
            let resp = HTTPURLResponse(url: url, statusCode: 404, httpVersion: "HTTP/1.1", headerFields: [:])!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}

/// Records installer calls instead of touching hdiutil.
final class StubInstaller: UpdateInstalling, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var calls: [(package: URL, version: String, target: URL, stage: URL, bytes: Data)] = []
    func install(package: URL, version: String, target: URL, stage: URL) async throws {
        let bytes = (try? Data(contentsOf: package)) ?? Data()
        lock.withLock { calls.append((package, version, target, stage, bytes)) }
    }
}

final class PhaseLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [UpdatePhase] = []
    func add(_ p: UpdatePhase) { lock.withLock { items.append(p) } }
    var phases: [UpdatePhase] { lock.withLock { items } }
}

// MARK: - Fixtures

private let api = "https://api.github.com/repos/BackendStack21/mlx-runner/releases/latest"
private let dl = "https://github.com/BackendStack21/mlx-runner/releases/download"

private func hex(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }

private func releaseJSON(tag: String = "v1.2.0", draft: Bool = false, prerelease: Bool = false,
                         assets: [String]? = nil, body: String = "## What's new\n- Faster") -> Data {
    let version = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
    let names = assets ?? [UpdateInstall.assetName(for: version), "SHA256SUMS", "SHA256SUMS.sig"]
    let json: [String: Any] = [
        "tag_name": tag, "html_url": "https://github.com/BackendStack21/mlx-runner/releases/tag/\(tag)",
        "body": body, "draft": draft, "prerelease": prerelease,
        "assets": names.map { ["name": $0, "browser_download_url": "\(dl)/\(tag)/\($0)"] },
    ]
    return try! JSONSerialization.data(withJSONObject: json)
}

/// A temp folder holding a writable `MLX Runner.app` directory.
private func tempInstall() -> UpdateInstall {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("upd-\(UUID().uuidString)")
    let app = dir.appendingPathComponent("MLX Runner.app")
    try! FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
    return UpdateInstall(kind: .macApp, target: app)
}

/// A complete, validly signed 1.2.0 release served by the stub.
private struct SignedRelease {
    let key = Curve25519.Signing.PrivateKey()
    let asset: Data
    let sums: Data
    let sig: Data
    var routes: [String: UpdateStub.Route]
    var publicKey: String { key.publicKey.rawRepresentation.base64EncodedString() }

    init(version: String = "1.2.0", assetSize: Int = 50_000, sumsOverride: String? = nil) {
        var bytes = Data(count: assetSize)
        for i in 0..<assetSize { bytes[i] = UInt8(truncatingIfNeeded: i &* 31 &+ 7) }
        asset = bytes
        let name = UpdateInstall.assetName(for: version)
        let text = sumsOverride ?? "\(hex(asset))  \(name)\n\(String(repeating: "a", count: 64))  mlxr-\(version)-macos-arm64.tar.gz\n"
        sums = Data(text.utf8)
        sig = try! key.signature(for: sums)
        routes = [
            api: .body(200, releaseJSON(tag: "v\(version)")),
            "\(dl)/v\(version)/\(name)": .body(200, asset),
            "\(dl)/v\(version)/SHA256SUMS": .body(200, sums),
            "\(dl)/v\(version)/SHA256SUMS.sig": .body(200, sig),
        ]
    }
}

private func makeUpdater(_ routes: [String: UpdateStub.Route], key: String = Curve25519.Signing.PrivateKey().publicKey.rawRepresentation.base64EncodedString(),
                         current: String = "1.0.0", install: UpdateInstall? = nil, installer: any UpdateInstalling = StubInstaller(),
                         now: @escaping @Sendable () -> Date = { Date() }) -> (Updater, String) {
    let (config, id) = UpdateStub.configuration(routes)
    let u = Updater(currentVersion: current, publicKey: key, install: install ?? tempInstall(),
                    configuration: config, installer: installer, now: now)
    return (u, id)
}

// MARK: - Pure checks

@Suite("Updater: versions, checksums, signatures, hosts")
struct UpdaterPureTests {
    @Test func comparesVersions() {
        let c = Updater.compareVersions
        #expect(c("1.0.1", "1.0.0") > 0)
        #expect(c("1.0.0", "1.0.1") < 0)
        #expect(c("2.0.0", "1.99.99") > 0)
        #expect(c("1.10.0", "1.9.0") > 0)
        #expect(c("v1.2.3", "1.2.3") == 0)
        #expect(c("1.2.3+abc", "1.2.3+def") == 0)
        #expect(c("1.0.0-rc.1", "1.0.0") < 0)  // pre-release before release
        #expect(c("1.0.0", "1.0.0-rc.1") > 0)
        #expect(c("1.0.0-rc.10", "1.0.0-rc.9") > 0)  // numerically
        #expect(c("1.0.0-alpha", "1.0.0-alpha.1") < 0)  // prefix first
        #expect(c("1.0.0-alpha.1", "1.0.0-alpha.beta") < 0)  // numeric before alphanumeric
        #expect(c("1.0.0-alpha", "1.0.0-beta") < 0)
        #expect(c("1.0.0-beta.11", "1.0.0-rc.1") < 0)
        #expect(c("1.0.0-rc.99999999999999999999999", "1.0.0-rc.2") > 0)  // no Int overflow
    }

    @Test func validatesVersions() {
        #expect(Updater.isValidVersion("1.2.3"))
        #expect(Updater.isValidVersion("1.2.3-rc.1+build.5"))
        #expect(!Updater.isValidVersion("1.2"))
        #expect(!Updater.isValidVersion("1.2.3/../../x"))
        #expect(!Updater.isValidVersion("latest"))
    }

    @Test func parsesSums() {
        let a = String(repeating: "a", count: 64), b = String(repeating: "B", count: 64)
        let text = "\(a)  MLX-Runner-1.2.0-macos-arm64.dmg\r\n\(b) *mlxr.tar.gz\n\nnot a line\n\(a.prefix(63))  short.dmg\n\(a)nospace\n  \(a)\tspaced name.txt  \n"
        let s = Updater.parseSums(text)
        #expect(s["MLX-Runner-1.2.0-macos-arm64.dmg"] == a)
        #expect(s["mlxr.tar.gz"] == b.lowercased())
        #expect(s["spaced name.txt"] == a)
        #expect(s["short.dmg"] == nil)
        #expect(s.count == 3)
    }

    @Test func verifiesSignatures() throws {
        let key = Curve25519.Signing.PrivateKey()
        let pub = key.publicKey.rawRepresentation.base64EncodedString()
        let data = Data("abc  file\n".utf8)
        let sig = try key.signature(for: data)
        #expect(Updater.verifySignature(publicKeyBase64: pub, data: data, signature: sig))
        // base64 of the raw signature, with a trailing newline.
        #expect(Updater.verifySignature(publicKeyBase64: pub, data: data, signature: Data((sig.base64EncodedString() + "\n").utf8)))
        // Tampered data.
        #expect(!Updater.verifySignature(publicKeyBase64: pub, data: Data("abd  file\n".utf8), signature: sig))
        // Tampered signature.
        var bad = sig
        bad[5] ^= 0x01
        #expect(!Updater.verifySignature(publicKeyBase64: pub, data: data, signature: bad))
        #expect(!Updater.verifySignature(publicKeyBase64: pub, data: data, signature: sig.prefix(63)))
        #expect(!Updater.verifySignature(publicKeyBase64: pub, data: data, signature: Data("not base64".utf8)))
        // Wrong key.
        let other = Curve25519.Signing.PrivateKey().publicKey.rawRepresentation.base64EncodedString()
        #expect(!Updater.verifySignature(publicKeyBase64: other, data: data, signature: sig))
        // Empty or malformed key.
        #expect(!Updater.verifySignature(publicKeyBase64: "", data: data, signature: sig))
        #expect(!Updater.verifySignature(publicKeyBase64: "AAAA", data: data, signature: sig))
        #expect(Updater.publicKey(fromBase64: "") == nil)
    }

    @Test func enforcesHostPolicy() {
        let ok = Updater.isAllowedDownloadURL
        #expect(ok(URL(string: "https://github.com/BackendStack21/mlx-runner/releases/download/v1/x.dmg")!, Updater.assetHosts))
        #expect(ok(URL(string: "https://objects.githubusercontent.com/x")!, Updater.assetHosts))
        #expect(ok(URL(string: "https://release-assets.githubusercontent.com/x")!, Updater.assetHosts))
        #expect(!ok(URL(string: "http://github.com/x.dmg")!, Updater.assetHosts))  // no TLS
        #expect(!ok(URL(string: "https://evil.example.com/x.dmg")!, Updater.assetHosts))
        #expect(!ok(URL(string: "https://github.com.evil.example/x.dmg")!, Updater.assetHosts))
        #expect(!ok(URL(string: "https://api.github.com/x")!, Updater.assetHosts))
        #expect(!ok(URL(string: "https://user:pw@github.com/x")!, Updater.assetHosts))
        #expect(!ok(URL(string: "file:///etc/passwd")!, Updater.assetHosts))
    }

    @Test func detectsInstallKind() {
        #expect(UpdateInstall.detect(bundleURL: URL(fileURLWithPath: "/Applications/MLX Runner.app")).kind == .macApp)
        #expect(UpdateInstall.detect(bundleURL: URL(fileURLWithPath: "/Users/me/mlx-runner/.build/debug")).kind == .source)
        #expect(UpdateInstall.assetName(for: "1.2.0") == "MLX-Runner-1.2.0-macos-arm64.dmg")
    }

    @Test func swapRollsBackOnFailure() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("swap-\(UUID().uuidString)")
        let target = dir.appendingPathComponent("App.app"), stage = dir.appendingPathComponent("stage")
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: target.appendingPathComponent("v"))
        defer { try? FileManager.default.removeItem(at: dir) }
        // The staged app doesn't exist: the second rename fails and the original comes back.
        #expect(throws: UpdateError.self) {
            try MacAppInstaller.swap(target: target, staged: stage.appendingPathComponent("missing.app"), stage: stage)
        }
        #expect(try String(contentsOf: target.appendingPathComponent("v"), encoding: .utf8) == "old")
        // And a successful swap.
        let staged = stage.appendingPathComponent("New.app")
        try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
        try Data("new".utf8).write(to: staged.appendingPathComponent("v"))
        try MacAppInstaller.swap(target: target, staged: staged, stage: stage)
        #expect(try String(contentsOf: target.appendingPathComponent("v"), encoding: .utf8) == "new")
        #expect(try String(contentsOf: stage.appendingPathComponent("previous/v"), encoding: .utf8) == "old")
    }
}

// MARK: - check()

@Suite("Updater: check")
struct UpdaterCheckTests {
    @Test func newerSignedReleaseIsInstallable() async throws {
        let r = SignedRelease()
        let (u, _) = makeUpdater(r.routes, key: r.publicKey)
        let info = try await u.check()
        #expect(info.available && info.canInstall)
        #expect(info.reason == nil)
        #expect(info.latest == "1.2.0" && info.current == "1.0.0")
        #expect(info.notes.contains("What's new"))
        #expect(info.url?.host == "github.com")
    }

    @Test func ignoresDraftsAndPrereleases() async throws {
        for json in [releaseJSON(draft: true), releaseJSON(prerelease: true), releaseJSON(tag: "nightly")] {
            let (u, _) = makeUpdater([api: .body(200, json)])
            let info = try await u.check()
            #expect(!info.available && info.latest == nil)
        }
    }

    @Test func olderOrEqualIsNotAvailable() async throws {
        for current in ["1.2.0", "1.3.0", "2.0.0-rc.1"] {
            let (u, _) = makeUpdater([api: .body(200, releaseJSON(tag: "v1.2.0"))], current: current)
            let info = try await u.check()
            #expect(!info.available && !info.canInstall, "current \(current)")
            #expect(info.latest == "1.2.0")
        }
        // A release is newer than its own release candidate.
        let (u, _) = makeUpdater([api: .body(200, releaseJSON(tag: "v1.2.0"))], current: "1.2.0-rc.3")
        #expect(try await u.check().available)
    }

    @Test func explainsWhyItCannotInstall() async throws {
        let dmg = UpdateInstall.assetName(for: "1.2.0")
        let cases: [([String], String)] = [
            (["SHA256SUMS", "SHA256SUMS.sig"], "no build for this Mac"),
            ([dmg, "SHA256SUMS"], "isn't signed"),
            ([dmg, "SHA256SUMS.sig"], "isn't signed"),
        ]
        let key = Curve25519.Signing.PrivateKey().publicKey.rawRepresentation.base64EncodedString()
        for (assets, reason) in cases {
            let (u, _) = makeUpdater([api: .body(200, releaseJSON(assets: assets))], key: key)
            let info = try await u.check()
            #expect(info.available && !info.canInstall)
            #expect(info.reason?.contains(reason) == true, "\(assets): \(info.reason ?? "nil")")
        }
        // No signing key compiled in.
        let (noKey, _) = makeUpdater([api: .body(200, releaseJSON())], key: "")
        #expect(try await noKey.check().reason == "This build has no update signing key.")
        // Running from source.
        let src = UpdateInstall(kind: .source, target: URL(fileURLWithPath: "/tmp/build"))
        let (fromSource, _) = makeUpdater([api: .body(200, releaseJSON())], key: key, install: src)
        #expect(try await fromSource.check().reason?.contains("git pull") == true)
        // Translocated.
        let tr = UpdateInstall(kind: .macApp, target: URL(fileURLWithPath: "/private/var/folders/x/AppTranslocation/y/d/MLX Runner.app"))
        let (translocated, _) = makeUpdater([api: .body(200, releaseJSON())], key: key, install: tr)
        #expect(try await translocated.check().reason?.contains("Applications") == true)
        // Not writable.
        let ro = UpdateInstall(kind: .macApp, target: URL(fileURLWithPath: "/System/Applications/MLX Runner.app"))
        let (readOnly, _) = makeUpdater([api: .body(200, releaseJSON())], key: key, install: ro)
        #expect(try await readOnly.check().reason?.contains("No permission") == true)
        // And an update that can't be installed is refused by apply().
        await #expect(throws: UpdateError("This build has no update signing key.")) { try await noKey.apply() }
    }

    @Test func cachesForAnHour() async throws {
        final class Clock: @unchecked Sendable { var t = Date(timeIntervalSince1970: 1_000_000) }
        let clock = Clock()
        let (u, id) = makeUpdater([api: .body(200, releaseJSON())], now: { clock.t })
        _ = try await u.check()
        _ = try await u.check()
        #expect(UpdateStub.requests(id).count == 1)
        clock.t += 3599
        _ = try await u.check()
        #expect(UpdateStub.requests(id).count == 1)
        _ = try await u.check(force: true)
        #expect(UpdateStub.requests(id).count == 2)
        clock.t += 3601
        _ = try await u.check()
        #expect(UpdateStub.requests(id).count == 3)
    }

    @Test func reportsAPIErrors() async throws {
        let (u, _) = makeUpdater([api: .body(403, Data("{}".utf8))])
        await #expect(throws: UpdateError("Update check failed: GitHub answered 403.")) { try await u.check() }
        let (garbage, _) = makeUpdater([api: .body(200, Data("<html>".utf8))])
        await #expect(throws: UpdateError.self) { try await garbage.check() }
        // The API may only redirect within api.github.com.
        let (redirected, rid) = makeUpdater([api: .redirect("https://evil.example.com/latest")])
        let error = await #expect(throws: UpdateError.self) { try await redirected.check() }
        #expect(error?.message.contains("Refusing to follow a redirect") == true)
        #expect(UpdateStub.requests(rid) == [api])
    }
}

// MARK: - apply() (download and verification; the hdiutil install is stubbed)

@Suite("Updater: apply")
struct UpdaterApplyTests {
    @Test func downloadsVerifiesAndInstalls() async throws {
        let r = SignedRelease(assetSize: 300_000)
        let installer = StubInstaller()
        let install = tempInstall()
        let (u, _) = makeUpdater(r.routes, key: r.publicKey, install: install, installer: installer)
        let log = PhaseLog()
        let applied = try await u.apply { log.add($0) }
        #expect(applied == AppliedUpdate(version: "1.2.0", target: install.target))
        #expect(installer.calls.count == 1)
        let call = try #require(installer.calls.first)
        #expect(call.bytes == r.asset)
        #expect(call.version == "1.2.0")
        #expect(call.package.lastPathComponent == "MLX-Runner-1.2.0-macos-arm64.dmg")
        #expect(call.stage.deletingLastPathComponent() == install.target.deletingLastPathComponent())
        #expect(call.stage.lastPathComponent.hasPrefix(".mlxrunner-update-"))
        // The staging dir is gone afterwards.
        #expect(!FileManager.default.fileExists(atPath: call.stage.path))
        // Phases in order, with download progress ending at 100%.
        let phases = log.phases
        let downloads = phases.compactMap { if case .download(let f) = $0 { f } else { nil } }
        #expect(downloads.first == 0 && downloads.last == 1)
        #expect(downloads == downloads.sorted())
        let coarse = phases.filter { if case .download = $0 { false } else { true } }
        #expect(coarse == [.checking, .signature, .checksum, .install, .done("1.2.0")])
        #expect(phases.firstIndex(of: .signature)! < phases.firstIndex(of: .download(0))!)
    }

    @Test func followsGitHubRedirectsToAllowedHosts() async throws {
        var r = SignedRelease()
        let name = UpdateInstall.assetName(for: "1.2.0")
        let cdn = "https://release-assets.githubusercontent.com/github-production-release-asset/1/\(name)"
        r.routes[cdn] = r.routes["\(dl)/v1.2.0/\(name)"]
        r.routes["\(dl)/v1.2.0/\(name)"] = .redirect(cdn)
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r.routes, key: r.publicKey, installer: installer)
        _ = try await u.apply()
        #expect(installer.calls.first?.bytes == r.asset)
        #expect(UpdateStub.requests(id).contains(cdn))
    }

    @Test func refusesRedirectsToOtherHosts() async throws {
        for evil in ["https://evil.example.com/x.dmg", "http://objects.githubusercontent.com/x.dmg"] {
            var r = SignedRelease()
            let name = UpdateInstall.assetName(for: "1.2.0")
            r.routes["\(dl)/v1.2.0/\(name)"] = .redirect(evil)
            r.routes[evil] = .body(200, r.asset)
            let installer = StubInstaller()
            let (u, id) = makeUpdater(r.routes, key: r.publicKey, installer: installer)
            let error = await #expect(throws: UpdateError.self) { try await u.apply() }
            #expect(error?.message.contains("Refusing to follow a redirect") == true, "\(error?.message ?? "")")
            #expect(installer.calls.isEmpty)
            #expect(!UpdateStub.requests(id).contains(evil))
        }
    }

    @Test func refusesAssetURLsOnOtherHosts() async throws {
        var r = SignedRelease()
        let name = UpdateInstall.assetName(for: "1.2.0")
        let json: [String: Any] = [
            "tag_name": "v1.2.0", "assets": [
                ["name": name, "browser_download_url": "https://evil.example.com/\(name)"],
                ["name": "SHA256SUMS", "browser_download_url": "\(dl)/v1.2.0/SHA256SUMS"],
                ["name": "SHA256SUMS.sig", "browser_download_url": "\(dl)/v1.2.0/SHA256SUMS.sig"],
            ],
        ]
        r.routes[api] = .body(200, try JSONSerialization.data(withJSONObject: json))
        let installer = StubInstaller()
        let (u, _) = makeUpdater(r.routes, key: r.publicKey, installer: installer)
        await #expect(throws: UpdateError("Refusing to download from evil.example.com.")) { try await u.apply() }
        #expect(installer.calls.isEmpty)
    }

    @Test func rejectsABadSignature() async throws {
        var r = SignedRelease()
        var sig = r.sig
        sig[0] ^= 0xff
        r.routes["\(dl)/v1.2.0/SHA256SUMS.sig"] = .body(200, sig)
        let installer = StubInstaller()
        let (u, _) = makeUpdater(r.routes, key: r.publicKey, installer: installer)
        await #expect(throws: UpdateError("SHA256SUMS signature is invalid: refusing to install.")) { try await u.apply() }
        #expect(installer.calls.isEmpty)
        // A valid signature by a different key is just as bad.
        let (u2, _) = makeUpdater(SignedRelease().routes, key: r.publicKey, installer: installer)
        await #expect(throws: UpdateError("SHA256SUMS signature is invalid: refusing to install.")) { try await u2.apply() }
        #expect(installer.calls.isEmpty)
    }

    @Test func rejectsAChecksumMismatch() async throws {
        var r = SignedRelease()
        var tampered = r.asset
        tampered[100] ^= 0x01
        r.routes["\(dl)/v1.2.0/\(UpdateInstall.assetName(for: "1.2.0"))"] = .body(200, tampered)
        let installer = StubInstaller()
        let install = tempInstall()
        let (u, _) = makeUpdater(r.routes, key: r.publicKey, install: install, installer: installer)
        await #expect(throws: UpdateError.self) { try await u.apply() }
        #expect(installer.calls.isEmpty)
        // No staging dir left behind.
        let left = try FileManager.default.contentsOfDirectory(atPath: install.target.deletingLastPathComponent().path)
        #expect(left == ["MLX Runner.app"])
    }

    @Test func rejectsAnAssetMissingFromTheSignedList() async throws {
        let other = "\(String(repeating: "0", count: 64))  MLX-Runner-1.1.0-macos-arm64.dmg\n"
        let r = SignedRelease(sumsOverride: other)
        let installer = StubInstaller()
        let (u, _) = makeUpdater(r.routes, key: r.publicKey, installer: installer)
        await #expect(throws: UpdateError("MLX-Runner-1.2.0-macos-arm64.dmg is not in the signed checksum list.")) { try await u.apply() }
        #expect(installer.calls.isEmpty)
    }

    @Test func capsChecksumFileSize() async throws {
        var r = SignedRelease()
        r.routes["\(dl)/v1.2.0/SHA256SUMS"] = .body(200, Data(count: Updater.maxSumsBytes + 1))
        let (u, _) = makeUpdater(r.routes, key: r.publicKey)
        await #expect(throws: UpdateError("The checksum file is too large.")) { try await u.apply() }
    }

    @Test func refusesWhenUpToDate() async throws {
        let r = SignedRelease()
        let (u, _) = makeUpdater(r.routes, key: r.publicKey, current: "1.2.0")
        await #expect(throws: UpdateError("MLX Runner is already up to date.")) { try await u.apply() }
    }
}

// MARK: - Real install: fake .app → DMG → hdiutil/ditto/codesign → swap

@Suite("Updater: macOS install (hdiutil)", .serialized)
struct UpdaterInstallIntegrationTests {
    /// Builds an ad-hoc-signed `MLX Runner.app` with the given identity, wrapped in a DMG.
    /// Returns nil if this machine can't create disk images.
    private func makeDMG(in dir: URL, id: String, version: String) throws -> URL? {
        let src = dir.appendingPathComponent("src-\(UUID().uuidString)")
        let app = src.appendingPathComponent("MLX Runner.app/Contents")
        try FileManager.default.createDirectory(at: app.appendingPathComponent("MacOS"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: "/usr/bin/true", toPath: app.appendingPathComponent("MacOS/MLXRunner").path)
        let plist: [String: Any] = [
            "CFBundleIdentifier": id, "CFBundleShortVersionString": version, "CFBundleVersion": version,
            "CFBundleExecutable": "MLXRunner", "CFBundlePackageType": "APPL", "CFBundleName": "MLX Runner",
        ]
        try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
            .write(to: app.appendingPathComponent("Info.plist"))
        let appURL = src.appendingPathComponent("MLX Runner.app")
        guard (try? MacAppInstaller.run("/usr/bin/codesign", ["-f", "-s", "-", appURL.path])) != nil else { return nil }
        let dmg = dir.appendingPathComponent("MLX-Runner-\(version)-\(UUID().uuidString).dmg")
        guard (try? MacAppInstaller.run("/usr/bin/hdiutil", ["create", "-quiet", "-srcfolder", src.path, "-volname", "MLX Runner", "-format", "UDZO", "-ov", dmg.path])) != nil
        else { return nil }
        return dmg
    }

    private func version(of app: URL) -> String? {
        guard let d = try? Data(contentsOf: app.appendingPathComponent("Contents/Info.plist")),
              let p = try? PropertyListSerialization.propertyList(from: d, format: nil) as? [String: Any] else { return nil }
        return p["CFBundleShortVersionString"] as? String
    }

    @Test func stagesVerifiesAndSwaps() async throws {
        guard FileManager.default.isExecutableFile(atPath: "/usr/bin/hdiutil") else { return }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("upd-int-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        guard let dmg = try makeDMG(in: dir, id: "de.21no.mlxrunner", version: "9.9.9") else {
            print("skipping: hdiutil can't create disk images here")
            return
        }
        // The "installed" 1.0.0 app.
        let installDir = dir.appendingPathComponent("Applications")
        try FileManager.default.createDirectory(at: installDir, withIntermediateDirectories: true)
        let target = installDir.appendingPathComponent("MLX Runner.app")
        try FileManager.default.createDirectory(at: target.appendingPathComponent("Contents"), withIntermediateDirectories: true)
        try PropertyListSerialization.data(fromPropertyList: ["CFBundleShortVersionString": "1.0.0"], format: .xml, options: 0)
            .write(to: target.appendingPathComponent("Contents/Info.plist"))

        // Wrong version or bundle id: refused, and the installed app is untouched.
        for (v, inst) in [("9.9.8", MacAppInstaller()), ("9.9.9", MacAppInstaller(bundleIdentifier: "com.example.other"))] {
            let stage = try Updater.makeStage(next: target)
            defer { try? FileManager.default.removeItem(at: stage) }
            do {
                try await inst.install(package: dmg, version: v, target: target, stage: stage)
                Issue.record("expected the install to be refused")
            } catch let e as UpdateError {
                if e.message.contains("hdiutil failed") { print("skipping: \(e.message)"); return }
            }
            #expect(version(of: target) == "1.0.0")
        }

        let stage = try Updater.makeStage(next: target)
        defer { try? FileManager.default.removeItem(at: stage) }
        try await MacAppInstaller().install(package: dmg, version: "9.9.9", target: target, stage: stage)
        #expect(version(of: target) == "9.9.9")
        #expect(version(of: stage.appendingPathComponent("previous")) == "1.0.0")
        // The image was detached.
        #expect(!FileManager.default.fileExists(atPath: stage.appendingPathComponent("mnt/MLX Runner.app").path))
    }
}
