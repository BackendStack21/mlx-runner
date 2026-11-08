import CryptoKit
import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

// Final-review polish for `mlxr upgrade`: no invented byte counts on the progress line, and one
// release-API request per upgrade.

private let api = "https://api.github.com/repos/BackendStack21/mlx-runner/releases/latest"
private let dl = "https://github.com/BackendStack21/mlx-runner/releases/download"

private struct PolishFixture {
    let key = Curve25519.Signing.PrivateKey()
    let routes: [String: UpgradeStub.Route]
    let config: URLSessionConfiguration
    let id: String
    let dir: URL
    let target: URL

    init() throws {
        let tarball = Data((0..<20_000).map { UInt8($0 % 251) })
        let name = UpdateInstall.assetName(for: "2.0.1", kind: .cli)
        let digest = SHA256.hash(data: tarball).map { String(format: "%02x", $0) }.joined()
        let sums = Data("\(digest)  \(name)\n".utf8)
        let sig = try key.signature(for: sums)
        let json: [String: Any] = [
            "tag_name": "v2.0.1", "html_url": "https://github.com/BackendStack21/mlx-runner/releases/tag/v2.0.1",
            "body": "notes", "draft": false, "prerelease": false,
            "assets": [name, "SHA256SUMS", "SHA256SUMS.sig"].map { ["name": $0, "browser_download_url": "\(dl)/v2.0.1/\($0)"] },
        ]
        routes = [
            api: .body(200, try JSONSerialization.data(withJSONObject: json)),
            "\(dl)/v2.0.1/\(name)": .body(200, tarball),
            "\(dl)/v2.0.1/SHA256SUMS": .body(200, sums),
            "\(dl)/v2.0.1/SHA256SUMS.sig": .body(200, sig),
        ]
        (config, id) = UpgradeStub.configuration(routes)
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-polish-\(UUID().uuidString)", isDirectory: true)
        target = dir.appendingPathComponent("mlxr")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: target)
    }

    func run(stderrTTY: Bool) async throws -> (code: Int32, out: String, err: String) {
        defer { try? FileManager.default.removeItem(at: dir) }
        let h = try RunHarness()
        let updater = Updater(currentVersion: "2.0.0", publicKey: key.publicKey.rawRepresentation.base64EncodedString(),
                              install: UpdateInstall(kind: .cli, target: target), configuration: config,
                              installer: RecordingInstaller())
        let table = CLI.commands(services: FakeServices().services, library: h.library.library, presets: h.library.presets,
                                 catalog: .standard, now: { LibraryFixture.now }, hub: h.hub.services, engine: h.engine,
                                 interrupts: h.interrupts, updater: { updater })
        let cap = Capture()
        let code = await CLI.run(["upgrade"], console: cap.console(stderrTTY: stderrTTY), commands: table)
        return (code, cap.stdout, cap.stderr)
    }
}

@Suite("mlxr upgrade polish")
struct UpgradePolishTests {
    @Test func theTerminalBarShowsNoMadeUpByteCountsOrSpeed() async throws {
        let f = try PolishFixture()
        let r = try await f.run(stderrTTY: true)
        #expect(r.code == 0, "\(r.err)")
        #expect(r.err.contains("\u{2595}"), "a bar is still drawn: \(r.err.debugDescription)")
        #expect(r.err.contains("100%"), "\(r.err.debugDescription)")
        #expect(!r.err.contains("/s"), "no speed field: \(r.err.debugDescription)")
        for fake in ["1000 B", "1.0 kB", "1.0 KB", "1000/1000", "0 B"] {
            #expect(!r.err.contains(fake), "\(fake) was never downloaded: \(r.err.debugDescription)")
        }
    }

    @Test func aSuccessfulUpgradeAsksTheReleaseAPIExactlyOnce() async throws {
        let f = try PolishFixture()
        let r = try await f.run(stderrTTY: false)
        #expect(r.code == 0 && r.out == "upgraded mlxr 2.0.0 -> 2.0.1\n", "\(r.out) \(r.err)")
        let hits = UpgradeStub.requests(f.id).filter { $0 == api }.count
        #expect(hits == 1, "release API requests: \(UpgradeStub.requests(f.id))")
    }
}
