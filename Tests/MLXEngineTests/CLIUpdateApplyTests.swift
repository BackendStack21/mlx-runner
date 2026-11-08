import CryptoKit
import Foundation
import Testing
@testable import MLXEngine

// SL02: Updater.apply for the `.cli` kind. No network: a URLProtocol stub serves the release API,
// SHA256SUMS, SHA256SUMS.sig and a real tarball (built with /usr/bin/tar); a throwaway Ed25519 key
// stands in for UpdateKey.

private let api = "https://api.github.com/repos/BackendStack21/mlx-runner/releases/latest"
private let dl = "https://github.com/BackendStack21/mlx-runner/releases/download"
private let oldBytes = Data("old-binary".utf8)

private func hex(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }

private func cliAsset(_ version: String) -> String { UpdateInstall.assetName(for: version, kind: .cli) }

/// A temp `bin/` directory holding the installed (old) `mlxr`.
private struct Bin {
    let root: URL
    let dir: URL
    let target: URL
    var install: UpdateInstall { UpdateInstall(kind: .cli, target: target) }

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-apply-\(UUID().uuidString)", isDirectory: true)
        dir = root.appendingPathComponent("bin", isDirectory: true)
        target = dir.appendingPathComponent("mlxr")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try oldBytes.write(to: target)
        chmod(target.path, 0o755)
    }

    func cleanup() {
        chmod(dir.path, 0o755)
        try? FileManager.default.removeItem(at: root)
    }

    func listing() -> [String] { ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).sorted() }
    var targetBytes: Data? { try? Data(contentsOf: target) }
}

/// A complete, validly signed CLI release served by the stub. `tarball` is a real .tar.gz.
private struct CLIRelease {
    let version: String
    let key = Curve25519.Signing.PrivateKey()
    let tarball: Data
    let sums: Data
    let sig: Data
    var routes: [String: UpdateStub.Route]
    var publicKey: String { key.publicKey.rawRepresentation.base64EncodedString() }
    var tarballURL: String { "\(dl)/v\(version)/\(cliAsset(version))" }

    init(version: String) throws {
        self.version = version
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-apply-tar-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let folder = "mlxr-\(version)-macos-arm64"
        try FileManager.default.createDirectory(at: tmp.appendingPathComponent(folder), withIntermediateDirectories: true)
        try Data("stand-in for the new mlxr".utf8).write(to: tmp.appendingPathComponent("\(folder)/mlxr"))
        try Data("MIT".utf8).write(to: tmp.appendingPathComponent("\(folder)/LICENSE"))
        let out = tmp.appendingPathComponent("a.tar.gz")
        try MacAppInstaller.run("/usr/bin/tar", ["-czf", out.path, "-C", tmp.path, folder])
        tarball = try Data(contentsOf: out)
        let name = cliAsset(version)
        // The app's disk image is listed too, as in a real release; it must be ignored.
        sums = Data("\(hex(tarball))  \(name)\n\(String(repeating: "b", count: 64))  \(UpdateInstall.assetName(for: version))\n".utf8)
        sig = try key.signature(for: sums)
        let names = [cliAsset(version), UpdateInstall.assetName(for: version), "SHA256SUMS", "SHA256SUMS.sig"]
        let json: [String: Any] = [
            "tag_name": "v\(version)", "html_url": "https://github.com/BackendStack21/mlx-runner/releases/tag/v\(version)",
            "body": "notes", "draft": false, "prerelease": false,
            "assets": names.map { ["name": $0, "browser_download_url": "\(dl)/v\(version)/\($0)"] },
        ]
        routes = [
            api: .body(200, try JSONSerialization.data(withJSONObject: json)),
            "\(dl)/v\(version)/\(name)": .body(200, tarball),
            "\(dl)/v\(version)/SHA256SUMS": .body(200, sums),
            "\(dl)/v\(version)/SHA256SUMS.sig": .body(200, sig),
        ]
    }
}

private func makeUpdater(_ release: CLIRelease, bin: Bin, current: String, install: UpdateInstall? = nil,
                         installer: any UpdateInstalling = StubInstaller(),
                         routes: [String: UpdateStub.Route]? = nil) -> (Updater, String) {
    let (config, id) = UpdateStub.configuration(routes ?? release.routes)
    return (Updater(currentVersion: current, publicKey: release.publicKey, install: install ?? bin.install,
                    configuration: config, installer: installer), id)
}

/// Installer that fails after the updater downloaded the package.
private struct FailingInstaller: UpdateInstalling {
    func install(package: URL, version: String, target: URL, stage: URL) async throws {
        try Data("partial".utf8).write(to: stage.appendingPathComponent("partial"))
        throw UpdateError("install failed on purpose")
    }
}

/// Fails every request like an unreachable or hung host.
final class UnreachableStub: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let code: URLError.Code = request.value(forHTTPHeaderField: "X-Fail") == "timeout" ? .timedOut : .cannotFindHost
        client?.urlProtocol(self, didFailWithError: URLError(code))
    }
    override func stopLoading() {}
}

@Suite("CLI update: apply")
struct CLIUpdateApplyTests {
    // MARK: happy path

    @Test func downloadsVerifiesAndHandsTheTarballToTheInstaller() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let r = try CLIRelease(version: "2.0.1")
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer)
        let info = try await u.check()
        #expect(info.available && info.canInstall, "reason: \(info.reason ?? "nil")")
        let log = PhaseLog()
        let applied = try await u.apply { log.add($0) }
        #expect(applied == AppliedUpdate(version: "2.0.1", target: bin.target))
        let call = try #require(installer.calls.first)
        #expect(installer.calls.count == 1)
        #expect(call.package.lastPathComponent == "mlxr-2.0.1-macos-arm64.tar.gz")
        #expect(call.bytes == r.tarball)
        #expect(call.version == "2.0.1")
        #expect(call.stage.deletingLastPathComponent() == bin.dir, "staging is next to the target")
        #expect(!FileManager.default.fileExists(atPath: call.stage.path), "staging removed")
        // The app's disk image is never downloaded.
        #expect(!UpdateStub.requests(id).contains { $0.hasSuffix(".dmg") })
        #expect(bin.listing() == ["mlxr"])
        let coarse = log.phases.filter { if case .download = $0 { false } else { true } }
        #expect(coarse == [.checking, .signature, .checksum, .install, .done("2.0.1")])
    }

    // MARK: AC-10: up to date and version ordering

    @Test func upToDateDownloadsNothing() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let r = try CLIRelease(version: "2.0.0")
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer)
        let info = try await u.check(force: true)
        #expect(info.latest == "2.0.0")
        #expect(!info.available)
        await #expect(throws: UpdateError.self) { try await u.apply() }
        #expect(UpdateStub.requests(id).allSatisfy { $0 == api }, "only the release API was asked: \(UpdateStub.requests(id))")
        #expect(installer.calls.isEmpty)
        #expect(bin.listing() == ["mlxr"])
        #expect(bin.targetBytes == oldBytes)
    }

    @Test func olderReleaseIsNotAnUpdate() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let r = try CLIRelease(version: "2.0.9")
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r, bin: bin, current: "2.0.10", installer: installer)
        #expect(try await u.check(force: true).available == false, "2.0.9 is older than 2.0.10")
        await #expect(throws: UpdateError.self) { try await u.apply() }
        #expect(UpdateStub.requests(id).allSatisfy { $0 == api })
        #expect(installer.calls.isEmpty)
    }

    @Test func multiDigitPatchIsNewer() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let r = try CLIRelease(version: "2.0.10")
        let installer = StubInstaller()
        let (u, _) = makeUpdater(r, bin: bin, current: "2.0.9", installer: installer)
        let info = try await u.check(force: true)
        #expect(info.available, "2.0.10 is newer than 2.0.9")
        #expect(info.canInstall, "reason: \(info.reason ?? "nil")")
        let applied = try await u.apply()
        #expect(applied.version == "2.0.10")
        #expect(installer.calls.first?.package.lastPathComponent == "mlxr-2.0.10-macos-arm64.tar.gz")
    }

    @Test func networkFailuresAreUpdateErrors() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let r = try CLIRelease(version: "2.0.1")
        // GitHub answers 5xx.
        for status in [500, 503] {
            let (u, _) = makeUpdater(r, bin: bin, current: "2.0.0", routes: [api: .body(status, Data())])
            await #expect(throws: UpdateError("Update check failed: GitHub answered \(status).")) { try await u.check(force: true) }
            await #expect(throws: UpdateError("Update check failed: GitHub answered \(status).")) { try await u.apply() }
        }
        // The release file download answers 5xx.
        var routes = r.routes
        routes[r.tarballURL] = .body(502, Data())
        let installer = StubInstaller()
        let (u, _) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer, routes: routes)
        await #expect(throws: UpdateError.self) { try await u.apply() }
        #expect(installer.calls.isEmpty)
        #expect(bin.listing() == ["mlxr"], "staging removed: \(bin.listing())")
        // Timeouts and unreachable hosts surface as UpdateError, not as a raw URLError.
        for header in ["timeout", "unreachable"] {
            let config = URLSessionConfiguration.ephemeral
            config.protocolClasses = [UnreachableStub.self]
            config.httpAdditionalHeaders = ["X-Fail": header]
            let down = Updater(currentVersion: "2.0.0", publicKey: r.publicKey, install: bin.install, configuration: config)
            for op in ["check", "apply"] {
                var thrown: Error?
                do {
                    if op == "check" { _ = try await down.check(force: true) } else { _ = try await down.apply() }
                } catch { thrown = error }
                #expect(thrown is UpdateError, "\(op) with \(header): \(String(describing: thrown))")
            }
        }
        #expect(bin.targetBytes == oldBytes)
    }

    // MARK: AC-6, AC-7: signature and checksum

    @Test func rejectsABadSignature() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        var r = try CLIRelease(version: "2.0.1")
        var bad = r.sig
        bad[0] ^= 0xff
        r.routes["\(dl)/v2.0.1/SHA256SUMS.sig"] = .body(200, bad)
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer)
        await #expect(throws: UpdateError("SHA256SUMS signature is invalid: refusing to install.")) { try await u.apply() }
        #expect(!UpdateStub.requests(id).contains(r.tarballURL), "the tarball is not downloaded before the signature verifies")
        #expect(installer.calls.isEmpty)
        #expect(bin.listing() == ["mlxr"], "nothing written next to the target: \(bin.listing())")
        #expect(bin.targetBytes == oldBytes)
    }

    @Test func rejectsASignatureByAnotherKey() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let r = try CLIRelease(version: "2.0.1")
        let other = try CLIRelease(version: "2.0.1")
        var routes = r.routes
        routes["\(dl)/v2.0.1/SHA256SUMS.sig"] = .body(200, other.sig)
        let installer = StubInstaller()
        let (u, _) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer, routes: routes)
        await #expect(throws: UpdateError("SHA256SUMS signature is invalid: refusing to install.")) { try await u.apply() }
        #expect(installer.calls.isEmpty)
        #expect(bin.listing() == ["mlxr"])
    }

    @Test func rejectsAChecksumMismatch() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        var r = try CLIRelease(version: "2.0.1")
        var tampered = r.tarball
        tampered[tampered.count / 2] ^= 0x01
        r.routes[r.tarballURL] = .body(200, tampered)
        let installer = StubInstaller()
        let (u, _) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer)
        var thrown: UpdateError?
        do { _ = try await u.apply() } catch let e as UpdateError { thrown = e } catch {}
        #expect(thrown?.message.hasPrefix("Checksum mismatch") == true, "got \(String(describing: thrown))")
        #expect(installer.calls.isEmpty, "the installer is not invoked")
        #expect(bin.listing() == ["mlxr"], "staging removed: \(bin.listing())")
        #expect(bin.targetBytes == oldBytes)
    }

    @Test func rejectsATarballMissingFromTheSignedList() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        var r = try CLIRelease(version: "2.0.1")
        let text = Data("\(String(repeating: "0", count: 64))  mlxr-2.0.0-macos-arm64.tar.gz\n".utf8)
        r.routes["\(dl)/v2.0.1/SHA256SUMS"] = .body(200, text)
        r.routes["\(dl)/v2.0.1/SHA256SUMS.sig"] = .body(200, try r.key.signature(for: text))
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer)
        await #expect(throws: UpdateError("mlxr-2.0.1-macos-arm64.tar.gz is not in the signed checksum list.")) { try await u.apply() }
        #expect(!UpdateStub.requests(id).contains(r.tarballURL))
        #expect(installer.calls.isEmpty)
        #expect(bin.listing() == ["mlxr"])
    }

    // MARK: install failure cleanup

    @Test func aFailingInstallerLeavesNothingBehind() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let r = try CLIRelease(version: "2.0.1")
        let (u, _) = makeUpdater(r, bin: bin, current: "2.0.0", installer: FailingInstaller())
        await #expect(throws: UpdateError("install failed on purpose")) { try await u.apply() }
        #expect(bin.listing() == ["mlxr"], "staging removed: \(bin.listing())")
        #expect(bin.targetBytes == oldBytes)
        // And the updater is usable again.
        #expect(await u.isInstalling == false)
    }

    // MARK: AC-9: source build

    @Test func sourceBuildIsRefusedBeforeDownloading() async throws {
        let bin = try Bin()
        defer { bin.cleanup() }
        let build = bin.root.appendingPathComponent("repo/.build/release", isDirectory: true)
        try FileManager.default.createDirectory(at: build, withIntermediateDirectories: true)
        let exe = build.appendingPathComponent("mlxr")
        try oldBytes.write(to: exe)
        let install = UpdateInstall.detectCLI(executable: exe)
        #expect(install.kind == .source)
        let r = try CLIRelease(version: "2.0.1")
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r, bin: bin, current: "2.0.0", install: install, installer: installer)
        await #expect(throws: UpdateError("mlxr is running from a source build; update it with git pull")) { try await u.apply() }
        #expect(UpdateStub.requests(id).allSatisfy { $0 == api }, "no release file requested: \(UpdateStub.requests(id))")
        #expect(installer.calls.isEmpty)
        #expect(try Data(contentsOf: exe) == oldBytes)
        #expect(((try? FileManager.default.contentsOfDirectory(atPath: build.path)) ?? []) == ["mlxr"])
    }

    // MARK: AC-8: unwritable target

    @Test func unwritableTargetIsRefusedBeforeDownloading() async throws {
        try #require(geteuid() != 0, "root can write anywhere")
        let bin = try Bin()
        defer { bin.cleanup() }
        chmod(bin.dir.path, 0o555)
        let r = try CLIRelease(version: "2.0.1")
        let installer = StubInstaller()
        let (u, id) = makeUpdater(r, bin: bin, current: "2.0.0", installer: installer)
        let expected = UpdateError("cannot replace \(bin.target.path) (permission denied); run: sudo mlxr upgrade")
        await #expect(throws: expected) { try await u.apply() }
        #expect(UpdateStub.requests(id).allSatisfy { $0 == api },
                "no signature file or tarball requested before refusing: \(UpdateStub.requests(id))")
        #expect(installer.calls.isEmpty)
        #expect(bin.listing() == ["mlxr"])
        #expect(bin.targetBytes == oldBytes)
    }
}
