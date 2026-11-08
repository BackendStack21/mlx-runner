import CryptoKit
import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

// SL03: `mlxr upgrade` through the command table. No network: a URLProtocol stub serves the release
// API, SHA256SUMS, SHA256SUMS.sig and the tarball; a throwaway Ed25519 key stands in for UpdateKey;
// the installer is recorded, never run.

private let api = "https://api.github.com/repos/BackendStack21/mlx-runner/releases/latest"
private let dl = "https://github.com/BackendStack21/mlx-runner/releases/download"
private let oldBytes = Data("old-binary".utf8)
private let barGlyph = "\u{2595}"  // the left edge of the pull-style bar

private func hex(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }

/// Serves canned responses to a URLSession, selected per test by the `X-Stub` header.
final class UpgradeStub: URLProtocol, @unchecked Sendable {
    enum Route {
        case body(Int, Data)
        case fail(URLError.Code)
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var tables: [String: [String: Route]] = [:]
    nonisolated(unsafe) private static var logs: [String: [String]] = [:]

    static func configuration(_ routes: [String: Route]) -> (URLSessionConfiguration, id: String) {
        let id = UUID().uuidString
        lock.withLock { tables[id] = routes }
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [UpgradeStub.self]
        c.httpAdditionalHeaders = ["X-Stub": id]
        return (c, id)
    }
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
        case .body(let status, let data)?:
            let resp = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Length": "\(data.count)"])!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            var i = 0
            while i < data.count {
                let end = min(data.count, i + 4096)
                client?.urlProtocol(self, didLoad: data.subdata(in: i..<end))
                i = end
            }
            client?.urlProtocolDidFinishLoading(self)
        case .fail(let code)?:
            client?.urlProtocol(self, didFailWithError: URLError(code))
        case nil:
            let resp = HTTPURLResponse(url: url, statusCode: 404, httpVersion: "HTTP/1.1", headerFields: [:])!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
        }
    }
    override func stopLoading() {}
}

/// Records every call; installs nothing.
final class RecordingInstaller: UpdateInstalling, @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    var versions: [String] { lock.withLock { items } }
    func install(package: URL, version: String, target: URL, stage: URL) async throws {
        lock.withLock { items.append(version) }
    }
}

/// Counts how often the command asked for an updater.
final class FactoryCount: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    var count: Int { lock.withLock { n } }
    func hit() { lock.withLock { n += 1 } }
}

/// A temp `bin/` directory holding the installed (old) `mlxr`.
private struct UpgradeBin {
    let root: URL
    let dir: URL
    let target: URL

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-upgrade-\(UUID().uuidString)", isDirectory: true)
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
    var targetBytes: Data? { try? Data(contentsOf: target) }
    var listing: [String] { ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).sorted() }
}

/// A signed CLI release served by the stub.
private struct UpgradeRelease {
    enum Tamper { case none, signature, digest }
    let version: String
    let key = Curve25519.Signing.PrivateKey()
    var publicKey: String { key.publicKey.rawRepresentation.base64EncodedString() }
    var routes: [String: UpgradeStub.Route]
    var assetName: String { UpdateInstall.assetName(for: version, kind: .cli) }
    var tarballURL: String { "\(dl)/v\(version)/\(assetName)" }

    init(version: String, tamper: Tamper = .none) throws {
        self.version = version
        let tarball = Data((0..<20_000).map { UInt8($0 % 251) })
        let name = UpdateInstall.assetName(for: version, kind: .cli)
        let signedDigest = tamper == .digest ? String(repeating: "0", count: 64) : hex(tarball)
        let sums = Data("\(signedDigest)  \(name)\n".utf8)
        var sig = try key.signature(for: sums)
        if tamper == .signature { sig[0] ^= 0xFF }
        let json: [String: Any] = [
            "tag_name": "v\(version)", "html_url": "https://github.com/BackendStack21/mlx-runner/releases/tag/v\(version)",
            "body": "notes", "draft": false, "prerelease": false,
            "assets": [name, "SHA256SUMS", "SHA256SUMS.sig"].map { ["name": $0, "browser_download_url": "\(dl)/v\(version)/\($0)"] },
        ]
        routes = [
            api: .body(200, try JSONSerialization.data(withJSONObject: json)),
            "\(dl)/v\(version)/\(name)": .body(200, tarball),
            "\(dl)/v\(version)/SHA256SUMS": .body(200, sums),
            "\(dl)/v\(version)/SHA256SUMS.sig": .body(200, sig),
        ]
    }
}

/// Runs `mlxr` command lines with the injected updater.
private final class UpgradeRig: @unchecked Sendable {
    let bin: UpgradeBin
    let installer = RecordingInstaller()
    let factory = FactoryCount()
    let release: UpgradeRelease
    private(set) var stubID = ""
    private let routes: [String: UpgradeStub.Route]
    private let install: UpdateInstall?
    private let current: String

    init(latest: String = "2.0.1", current: String = "2.0.0", tamper: UpgradeRelease.Tamper = .none,
         routes override: [String: UpgradeStub.Route]? = nil, install: UpdateInstall? = nil,
         failTarball status: Int? = nil) throws {
        bin = try UpgradeBin()
        release = try UpgradeRelease(version: latest, tamper: tamper)
        var table = override ?? release.routes
        if let status { table[release.tarballURL] = .body(status, Data()) }
        routes = table
        self.install = install
        self.current = current
    }
    deinit { bin.cleanup() }

    var requests: [String] { UpgradeStub.requests(stubID) }
    var tarballRequested: Bool { requests.contains(release.tarballURL) }

    func table() -> [Command] {
        let h = try! RunHarness()
        let (config, id) = UpgradeStub.configuration(routes)
        stubID = id
        let target = bin.target
        let installer = self.installer, factory = self.factory, release = self.release
        let install = self.install ?? UpdateInstall(kind: .cli, target: target), current = self.current
        return CLI.commands(services: FakeServices().services, library: h.library.library, presets: h.library.presets,
                            catalog: .standard, now: { LibraryFixture.now }, hub: h.hub.services, engine: h.engine,
                            interrupts: h.interrupts,
                            updater: {
                                factory.hit()
                                return Updater(currentVersion: current, publicKey: release.publicKey, install: install,
                                               configuration: config, installer: installer)
                            })
    }

    func run(_ args: [String], stderrTTY: Bool = false, stdoutTTY: Bool = false) async -> (code: Int32, out: String, err: String) {
        let cap = Capture()
        let code = await CLI.run(args, console: cap.console(stdoutTTY: stdoutTTY, stderrTTY: stderrTTY), commands: table())
        return (code, cap.stdout, cap.stderr)
    }
}

@Suite("mlxr upgrade")
struct UpgradeCommandTests {
    // MARK: version

    @Test func theVersionStaysAtTwoZeroZero() {
        #expect(MLXRunnerVersion.current == "2.0.0")
    }

    // MARK: AC-11 up to date

    @Test func upToDatePrintsOneLineAndExitsZero() async throws {
        let rig = try UpgradeRig(latest: "2.0.0")
        let r = await rig.run(["upgrade"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "mlxr is up to date (2.0.0)\n", "\(r.out)")
        #expect(r.err.isEmpty, "\(r.err)")
        #expect(!rig.tarballRequested && rig.installer.versions.isEmpty)
        #expect(rig.bin.targetBytes == oldBytes)
    }

    @Test func anOlderLatestReleaseCountsAsUpToDate() async throws {
        let rig = try UpgradeRig(latest: "1.9.0")
        let r = await rig.run(["upgrade"])
        #expect(r.code == 0 && r.out == "mlxr is up to date (2.0.0)\n", "\(r.out) \(r.err)")
        #expect(rig.installer.versions.isEmpty)
    }

    // MARK: AC-12 successful upgrade

    @Test func aSuccessfulUpgradePrintsTheTransitionOnStdout() async throws {
        let rig = try UpgradeRig()
        let r = await rig.run(["upgrade"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "upgraded mlxr 2.0.0 -> 2.0.1\n", "\(r.out)")
        #expect(rig.installer.versions == ["2.0.1"])
        #expect(rig.requests.contains(rig.release.tarballURL))
    }

    @Test func noBarOrEscapesOnStderrWhenItIsNotATerminal() async throws {
        let rig = try UpgradeRig()
        let r = await rig.run(["upgrade"], stderrTTY: false)
        #expect(r.code == 0, "\(r.err)")
        #expect(!r.err.contains("\r") && !r.err.contains("\u{1B}") && !r.err.contains(barGlyph), "\(r.err.debugDescription)")
        #expect(r.out == "upgraded mlxr 2.0.0 -> 2.0.1\n")
    }

    @Test func theProgressBarGoesToStderrOnlyWhenItIsATerminal() async throws {
        let rig = try UpgradeRig()
        let r = await rig.run(["upgrade"], stderrTTY: true)
        #expect(r.code == 0, "\(r.err)")
        #expect(r.err.contains("\r") && r.err.contains(barGlyph), "\(r.err.debugDescription)")
        #expect(r.out == "upgraded mlxr 2.0.0 -> 2.0.1\n", "progress never reaches stdout: \(r.out.debugDescription)")
    }

    @Test func aTerminalStdoutDoesNotChangeTheResultLine() async throws {
        let rig = try UpgradeRig()
        let r = await rig.run(["upgrade"], stderrTTY: false, stdoutTTY: true)
        #expect(r.out == "upgraded mlxr 2.0.0 -> 2.0.1\n", "\(r.out)")
    }

    @Test func updateIsAnAliasForUpgrade() async throws {
        let inv = try CLI.parse(["update"], commands: try UpgradeRig().table())
        #expect(inv.command.name == "upgrade")
        let rig = try UpgradeRig()
        let r = await rig.run(["update"])
        #expect(r.code == 0 && r.out == "upgraded mlxr 2.0.0 -> 2.0.1\n", "\(r.out) \(r.err)")
        let up = try UpgradeRig(latest: "2.0.0")
        let r2 = await up.run(["update", "--check"])
        #expect(r2.code == 0 && r2.out == "mlxr is up to date (2.0.0)\n", "\(r2.out) \(r2.err)")
    }

    // MARK: AC-13 --check

    @Test func checkReportsANewerReleaseWithoutInstalling() async throws {
        let rig = try UpgradeRig()
        let r = await rig.run(["upgrade", "--check"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "mlxr 2.0.1 is available (you have 2.0.0)\n", "\(r.out)")
        #expect(r.err.isEmpty, "\(r.err)")
        #expect(rig.installer.versions.isEmpty)
        #expect(!rig.tarballRequested)
        #expect(rig.requests.allSatisfy { $0 == api }, "only the release API is asked: \(rig.requests)")
        #expect(rig.bin.targetBytes == oldBytes)
    }

    @Test func checkWhenUpToDatePrintsTheUpToDateLine() async throws {
        let rig = try UpgradeRig(latest: "2.0.0")
        let r = await rig.run(["upgrade", "--check"])
        #expect(r.code == 0 && r.out == "mlxr is up to date (2.0.0)\n", "\(r.out) \(r.err)")
        #expect(rig.installer.versions.isEmpty)
    }

    @Test func checkFailuresAreErrors() async throws {
        let rig = try UpgradeRig(routes: [api: .body(503, Data())])
        let r = await rig.run(["upgrade", "--check"])
        #expect(r.code == 1)
        #expect(r.out.isEmpty)
        #expect(r.err == "Error: Update check failed: GitHub answered 503.\n", "\(r.err)")
    }

    // MARK: AC-14 failures

    @Test func aBadSignatureIsRefusedAndNothingIsWritten() async throws {
        let rig = try UpgradeRig(tamper: .signature)
        let r = await rig.run(["upgrade"])
        #expect(r.code == 1)
        #expect(r.out.isEmpty, "\(r.out)")
        #expect(r.err == "Error: SHA256SUMS signature is invalid: refusing to install.\n", "\(r.err)")
        #expect(rig.installer.versions.isEmpty && !rig.tarballRequested)
        #expect(rig.bin.targetBytes == oldBytes)
        #expect(rig.bin.listing == ["mlxr"], "\(rig.bin.listing)")
    }

    @Test func aChecksumMismatchIsRefused() async throws {
        let rig = try UpgradeRig(tamper: .digest)
        let r = await rig.run(["upgrade"])
        #expect(r.code == 1)
        #expect(r.out.isEmpty, "\(r.out)")
        #expect(r.err.hasPrefix("Error: Checksum mismatch for \(rig.release.assetName)"), "\(r.err)")
        #expect(r.err.hasSuffix("\n") && r.err.filter { $0 == "\n" }.count == 1, "one line: \(r.err.debugDescription)")
        #expect(rig.installer.versions.isEmpty)
        #expect(rig.bin.targetBytes == oldBytes && rig.bin.listing == ["mlxr"])
    }

    @Test(.enabled(if: geteuid() != 0, "root can write anywhere"))
    func anUnwritableDirectoryNamesSudo() async throws {
        let rig = try UpgradeRig()
        chmod(rig.bin.dir.path, 0o555)
        let r = await rig.run(["upgrade"])
        #expect(r.code == 1)
        #expect(r.out.isEmpty, "\(r.out)")
        #expect(r.err == "Error: cannot replace \(rig.bin.target.path) (permission denied); run: sudo mlxr upgrade\n", "\(r.err)")
        #expect(!rig.tarballRequested && rig.installer.versions.isEmpty)
        #expect(rig.bin.targetBytes == oldBytes)
    }

    @Test func aSourceBuildIsRefusedWithAGitHint() async throws {
        let source = UpdateInstall.detectCLI(executable: URL(fileURLWithPath: "/work/mlx-runner/.build/debug/mlxr"))
        #expect(source.kind == .source)
        let rig = try UpgradeRig(install: source)
        let r = await rig.run(["upgrade"])
        #expect(r.code == 1)
        #expect(r.out.isEmpty, "\(r.out)")
        #expect(r.err == "Error: mlxr is running from a source build; update it with git pull\n", "\(r.err)")
        #expect(!rig.tarballRequested && rig.installer.versions.isEmpty)
    }

    @Test(arguments: ["2.0.0", "1.9.0"])
    func aSourceBuildIsRefusedEvenWhenNoNewerReleaseExists(latest: String) async throws {
        let source = UpdateInstall.detectCLI(executable: URL(fileURLWithPath: "/work/mlx-runner/.build/release/mlxr"))
        let rig = try UpgradeRig(latest: latest, install: source)
        let r = await rig.run(["upgrade"])
        #expect(r.code == 1, "\(r.out)")
        #expect(r.out.isEmpty, "\(r.out)")
        #expect(r.err == "Error: mlxr is running from a source build; update it with git pull\n", "\(r.err)")
        #expect(rig.installer.versions.isEmpty && !rig.tarballRequested)
    }

    @Test func aSourceBuildIsRefusedForUpdateToo() async throws {
        let source = UpdateInstall.detectCLI(executable: URL(fileURLWithPath: "/work/mlx-runner/.build/debug/mlxr"))
        let rig = try UpgradeRig(latest: "2.0.0", install: source)
        let r = await rig.run(["update"])
        #expect(r.code == 1 && r.err == "Error: mlxr is running from a source build; update it with git pull\n", "\(r.err)")
    }

    @Test func aServerErrorOnTheReleaseCheckIsAnError() async throws {
        let rig = try UpgradeRig(routes: [api: .body(500, Data())])
        let r = await rig.run(["upgrade"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err == "Error: Update check failed: GitHub answered 500.\n", "\(r.err)")
        #expect(rig.bin.targetBytes == oldBytes)
    }

    @Test func aServerErrorOnTheDownloadIsAnError() async throws {
        let broken = try UpgradeRig(failTarball: 500)
        let r = await broken.run(["upgrade"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: "), "\(r.err)")
        #expect(broken.installer.versions.isEmpty)
        #expect(broken.bin.targetBytes == oldBytes && broken.bin.listing == ["mlxr"])
    }

    @Test(arguments: [URLError.Code.cannotFindHost, .timedOut, .notConnectedToInternet])
    func anUnreachableHostIsAnError(code: URLError.Code) async throws {
        let rig = try UpgradeRig(routes: [api: .fail(code)])
        let r = await rig.run(["upgrade"])
        #expect(r.code == 1 && r.out.isEmpty)
        #expect(r.err.hasPrefix("Error: ") && r.err.hasSuffix("\n"), "\(r.err.debugDescription)")
        #expect(r.err.filter { $0 == "\n" }.count == 1, "\(r.err.debugDescription)")
        #expect(rig.bin.targetBytes == oldBytes)
    }

    // MARK: AC-15 table, help, completions, parser

    @Test func theTableResolvesBothNames() throws {
        let table = CLI.commands
        #expect(try CLI.parse(["upgrade"], commands: table).command.name == "upgrade")
        #expect(try CLI.parse(["update"], commands: table).command.name == "upgrade")
        #expect(try CLI.parse(["upgrade", "--check"], commands: table).bool("check"))
        #expect(!(try CLI.parse(["upgrade"], commands: table).bool("check")))
    }

    @Test func commandHelpListsTheFlagAndTheAlias() async {
        for name in ["upgrade", "update"] {
            let cap = Capture()
            #expect(await CLI.run([name, "--help"], console: cap.console()) == 0)
            #expect(cap.stdout.contains("Usage:\n  mlxr upgrade"), "\(cap.stdout)")
            #expect(cap.stdout.contains("Aliases:\n  upgrade, update"), "\(cap.stdout)")
            #expect(cap.stdout.contains("--check"), "\(cap.stdout)")
        }
        let cap = Capture()
        #expect(await CLI.run(["help", "update"], console: cap.console()) == 0)
        #expect(cap.stdout.contains("Usage:\n  mlxr upgrade"), "\(cap.stdout)")
    }

    @Test func rootHelpListsUpgrade() async {
        let cap = Capture()
        #expect(await CLI.run(["--help"], console: cap.console()) == 0)
        let lines = cap.stdout.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
        #expect(lines.contains { $0.hasPrefix("upgrade ") }, "\(cap.stdout)")
    }

    @Test func helpNeverRunsTheUpdater() async throws {
        let rig = try UpgradeRig()
        let r = await rig.run(["upgrade", "--help"])
        #expect(r.code == 0 && r.out.contains("--check"))
        #expect(rig.factory.count == 0 && rig.requests.isEmpty && rig.installer.versions.isEmpty)
    }

    @Test(arguments: ["zsh", "bash"])
    func completionsOfferTheCommandAndItsFlag(shell: String) async {
        let cap = Capture()
        #expect(await CLI.run(["completion", shell], console: cap.console()) == 0)
        #expect(cap.stdout.contains("upgrade") && cap.stdout.contains("update"), "\(shell)")
        #expect(cap.stdout.contains("--check"), "\(shell)")
    }

    @Test(arguments: [["upgrade", "--bogus"], ["upgrade", "--check=maybe"], ["upgrade", "now"], ["upgrade", "-x"]])
    func unknownFlagsAndArgumentsAreRejectedBeforeAnythingRuns(argv: [String]) async throws {
        let rig = try UpgradeRig()
        let r = await rig.run(argv)
        #expect(r.code == 1, "\(argv)")
        #expect(r.out.isEmpty, "\(r.out)")
        #expect(r.err.hasPrefix("Error: "), "\(r.err)")
        #expect(rig.factory.count == 0 && rig.requests.isEmpty && rig.installer.versions.isEmpty)
    }

    // MARK: no other command updates (plan review R-02)

    @Test func buildingTheTableDoesNotCreateAnUpdater() throws {
        let rig = try UpgradeRig()
        _ = rig.table()
        #expect(rig.factory.count == 0)
    }

    @Test func noOtherCommandTouchesTheUpdaterOrTheNetwork() async throws {
        let rig = try UpgradeRig()
        let argvs: [[String]] = [
            ["run", "qwen3:0.6b", "hi"], ["list"], ["ls"], ["show", "qwen3:0.6b"], ["show", "nope/none"],
            ["pull", "mlx-community/Qwen3-0.6B-4bit"], ["rm", "nope/none"], ["search", "qwen"],
            ["tokenize", "qwen3:0.6b", "hi"], ["--version"], ["version"], ["help"], ["--help"],
            ["completion", "zsh"], ["run", "--help"], ["bogus"],
        ]
        for argv in argvs {
            _ = await rig.run(argv)
            #expect(rig.factory.count == 0, "\(argv) built an updater")
            #expect(rig.requests.isEmpty, "\(argv) made an update request: \(rig.requests)")
            #expect(rig.installer.versions.isEmpty, "\(argv)")
        }
        #expect(rig.bin.targetBytes == oldBytes)
    }
}
