import Foundation
import Testing
@testable import MLXEngine

// SL01: the `.cli` install kind, its asset name, and CLIInstaller's extraction and validation.
// Everything is local: real tarballs are built with /usr/bin/tar in a temporary directory.

// MARK: - Fixtures

private let testVersion = "9.9.9"
private let memberDir = "mlxr-\(testVersion)-macos-arm64"
private let oldBytes = Data("old-binary".utf8)

/// A small arm64 Mach-O executable, ad-hoc signed, built once with swiftc. Prints "new-binary".
private let signedBinary: Data? = {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-cli-fixture-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    do {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let src = dir.appendingPathComponent("stub.swift")
        try Data("print(\"new-binary\")\n".utf8).write(to: src)
        let out = dir.appendingPathComponent("stub")
        try MacAppInstaller.run("/usr/bin/swiftc", ["-o", out.path, src.path])
        try MacAppInstaller.run("/usr/bin/codesign", ["-f", "-s", "-", out.path])
        return try Data(contentsOf: out)
    } catch { return nil }
}()

private func requireBinary() throws -> Data {
    try #require(signedBinary, "swiftc/codesign unavailable: cannot build a test Mach-O")
}

private func tamper(_ data: Data) -> Data {
    var d = data
    d[d.count / 2] ^= 0xFF  // inside the signed pages, before the signature blob
    return d
}

private func unsigned(_ data: Data) throws -> Data {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-cli-unsigned-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let f = dir.appendingPathComponent("b")
    try data.write(to: f)
    try MacAppInstaller.run("/usr/bin/codesign", ["--remove-signature", f.path])
    return try Data(contentsOf: f)
}

/// Mach-O header bytes: magic, cputype, then padding.
private func header(magic: [UInt8], cpu: [UInt8], total: Int = 4096) -> Data {
    var d = Data(magic + cpu)
    d.append(Data(count: max(0, total - d.count)))
    return d
}

enum Member {
    case file(Data, mode: UInt16 = 0o755)
    case symlink(String)
}

/// A temp area with a target directory holding the old binary, a stage directory and a tarball.
private struct Area {
    let root: URL
    let targetDir: URL
    let target: URL
    let stage: URL
    var tarball: URL { stage.appendingPathComponent("mlxr-\(testVersion)-macos-arm64.tar.gz") }

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-cli-test-\(UUID().uuidString)", isDirectory: true)
        targetDir = root.appendingPathComponent("bin", isDirectory: true)
        target = targetDir.appendingPathComponent("mlxr")
        stage = targetDir.appendingPathComponent(".mlxrunner-update-TEST", isDirectory: true)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        try oldBytes.write(to: target)
        chmod(target.path, 0o644)
    }

    func cleanup() { try? FileManager.default.removeItem(at: root) }

    /// Builds `tarball` with /usr/bin/tar. Each (stored, archived) pair is written to a scratch tree
    /// under `stored`, then archived under the name `archived` (via `-s` when they differ).
    func makeTarball(_ members: [(stored: String, archived: String, member: Member)]) throws {
        let src = root.appendingPathComponent("src-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: src) }
        var args = ["-P", "-czf", tarball.path, "-C", src.path]
        var names: [String] = []
        for m in members {
            let f = src.appendingPathComponent(m.stored)
            try FileManager.default.createDirectory(at: f.deletingLastPathComponent(), withIntermediateDirectories: true)
            switch m.member {
            case .file(let data, let mode):
                try data.write(to: f)
                chmod(f.path, mode_t(mode))
            case .symlink(let dest):
                try FileManager.default.createSymbolicLink(atPath: f.path, withDestinationPath: dest)
            }
            if m.stored != m.archived { args += ["-s", "|^\(m.stored)$|\(m.archived)|"] }
            names.append(m.stored)
        }
        try MacAppInstaller.run("/usr/bin/tar", args + names)  // options first, then the members
    }

    /// The release layout: `mlxr-<v>-macos-arm64/mlxr` plus LICENSE.
    func makeRelease(binary: Data, withLicense: Bool = true) throws {
        var members: [(String, String, Member)] = [("\(memberDir)/mlxr", "\(memberDir)/mlxr", .file(binary))]
        if withLicense { members.append(("\(memberDir)/LICENSE", "\(memberDir)/LICENSE", .file(Data("MIT".utf8), mode: 0o644))) }
        try makeTarball(members.map { (stored: $0.0, archived: $0.1, member: $0.2) })
    }

    func listing() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: targetDir.path)) ?? []).sorted()
    }

    func stageListing() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: stage.path)) ?? []).sorted()
    }

    func install(_ installer: CLIInstaller = CLIInstaller()) async throws {
        try await installer.install(package: tarball, version: testVersion, target: target, stage: stage)
    }

    /// The install must throw an UpdateError and leave target, its directory and the stage as they were.
    func expectRejected(_ installer: CLIInstaller = CLIInstaller(), sourceLocation: SourceLocation = #_sourceLocation) async {
        var thrown: Error?
        do { try await install(installer) } catch { thrown = error }
        #expect(thrown is UpdateError, "expected an UpdateError, got \(String(describing: thrown))", sourceLocation: sourceLocation)
        #expect((try? Data(contentsOf: target)) == oldBytes, "target changed", sourceLocation: sourceLocation)
        #expect(listing() == [".mlxrunner-update-TEST", "mlxr"], "stray files next to the target: \(listing())", sourceLocation: sourceLocation)
        #expect(stageListing() == [tarball.lastPathComponent], "stage not cleaned: \(stageListing())", sourceLocation: sourceLocation)
    }
}

// MARK: - AC-1: kind detection and asset name

@Suite("CLI update: kind and asset name")
struct CLIKindTests {
    @Test func installedBinaryIsCLI() {
        let i = UpdateInstall.detectCLI(executable: URL(fileURLWithPath: "/usr/local/bin/mlxr"))
        #expect(i.kind == .cli)
        #expect(i.target.path == "/usr/local/bin/mlxr")
    }

    @Test func buildDirectoryIsSource() {
        for path in ["/Users/me/mlx-runner/.build/arm64-apple-macosx/release/mlxr", "/Users/me/mlx-runner/.build/debug/mlxr"] {
            #expect(UpdateInstall.detectCLI(executable: URL(fileURLWithPath: path)).kind == .source, "\(path)")
        }
    }

    @Test func buildLookalikeIsNotSource() {
        for path in ["/Users/me/my.build/bin/mlxr", "/Users/me/.builds/mlxr", "/opt/x.build/mlxr"] {
            #expect(UpdateInstall.detectCLI(executable: URL(fileURLWithPath: path)).kind == .cli, "\(path)")
        }
    }

    @Test func symlinkIsResolvedToTheRealFile() throws {
        let a = try Area()
        defer { a.cleanup() }
        let link = a.root.appendingPathComponent("mlxr-link")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: a.target.path)
        let i = UpdateInstall.detectCLI(executable: link)
        #expect(i.kind == .cli)
        #expect(i.target.resolvingSymlinksInPath().path == a.target.resolvingSymlinksInPath().path)
        #expect(i.target.lastPathComponent == "mlxr", "target is the file the link points to, not the link")
        #expect(i.target.path != link.path)
    }

    @Test func symlinkIntoABuildDirectoryIsSource() throws {
        let a = try Area()
        defer { a.cleanup() }
        let build = a.root.appendingPathComponent("repo/.build/release", isDirectory: true)
        try FileManager.default.createDirectory(at: build, withIntermediateDirectories: true)
        let real = build.appendingPathComponent("mlxr")
        try oldBytes.write(to: real)
        let link = a.root.appendingPathComponent("bin/mlxr-dev")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: real.path)
        #expect(UpdateInstall.detectCLI(executable: link).kind == .source)
    }

    @Test func cliAssetName() {
        #expect(UpdateInstall.assetName(for: "2.0.1", kind: .cli) == "mlxr-2.0.1-macos-arm64.tar.gz")
    }

    @Test func appAssetNameIsUnchanged() {
        #expect(UpdateInstall.assetName(for: "1.2.0") == "MLX-Runner-1.2.0-macos-arm64.dmg")
        #expect(UpdateInstall.assetName(for: "1.2.0", kind: .macApp) == "MLX-Runner-1.2.0-macos-arm64.dmg")
    }

    @Test func appDetectionIsUnchanged() {
        #expect(UpdateInstall.detect(bundleURL: URL(fileURLWithPath: "/Applications/MLX Runner.app")).kind == .macApp)
        #expect(UpdateInstall.detect(bundleURL: URL(fileURLWithPath: "/Users/me/mlx-runner/.build/debug")).kind == .source)
    }
}

// MARK: - AC-2: a valid tarball installs atomically

@Suite("CLI update: install")
struct CLIInstallTests {
    @Test func replacesTheTargetWithTheNewBinary() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        try await a.install()
        #expect(try Data(contentsOf: a.target) == bin)
        let mode = (try FileManager.default.attributesOfItem(atPath: a.target.path)[.posixPermissions] as? NSNumber)?.intValue
        #expect(mode == 0o755)
    }

    @Test func installedBinaryRuns() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        try await a.install()
        let out = try MacAppInstaller.run(a.target.path, [])
        #expect(out.trimmingCharacters(in: .whitespacesAndNewlines) == "new-binary")
    }

    @Test func leavesNothingBehindButTheTarball() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        try await a.install()
        #expect(try Data(contentsOf: a.target) == bin)
        #expect(a.listing() == [".mlxrunner-update-TEST", "mlxr"], "\(a.listing())")
        #expect(a.stageListing() == [a.tarball.lastPathComponent], "\(a.stageListing())")
    }

    @Test func releaseWithoutLicenseInstalls() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin, withLicense: false)
        try await a.install()
        #expect(try Data(contentsOf: a.target) == bin)
    }

    @Test func licenseIsNeverInstalled() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        try await a.install()
        #expect(try Data(contentsOf: a.target) == bin)
        #expect(!a.listing().contains("LICENSE"))
        #expect(!a.stageListing().contains(memberDir))
    }

    @Test func aRestrictiveModeInTheArchiveStillEndsAs755() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeTarball([(stored: "\(memberDir)/mlxr", archived: "\(memberDir)/mlxr", member: .file(bin, mode: 0o600))])
        try await a.install()
        let mode = (try FileManager.default.attributesOfItem(atPath: a.target.path)[.posixPermissions] as? NSNumber)?.intValue
        #expect(mode == 0o755)
    }

    @Test func aTargetThatCannotBeReplacedKeepsTheOldBinary() async throws {
        // rename(2) of a file over a directory fails: the install must report it and clean up.
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try FileManager.default.removeItem(at: a.target)
        try FileManager.default.createDirectory(at: a.target, withIntermediateDirectories: false)
        try oldBytes.write(to: a.target.appendingPathComponent("keep"))
        try a.makeRelease(binary: bin)
        var thrown: Error?
        do { try await a.install() } catch { thrown = error }
        #expect(thrown is UpdateError)
        #expect(try Data(contentsOf: a.target.appendingPathComponent("keep")) == oldBytes)
        #expect(a.stageListing() == [a.tarball.lastPathComponent], "\(a.stageListing())")
    }
}

// MARK: - AC-3: hostile archive layouts

@Suite("CLI update: archive layout")
struct CLILayoutTests {
    @Test func symlinkMemberIsRejected() async throws {
        let a = try Area()
        defer { a.cleanup() }
        try a.makeTarball([(stored: "\(memberDir)/mlxr", archived: "\(memberDir)/mlxr", member: .symlink("/bin/ls"))])
        await a.expectRejected()
    }

    @Test func pathWithDotDotIsRejected() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeTarball([(stored: "zz/mlxr", archived: "\(memberDir)/../mlxr", member: .file(bin))])
        let listing = try MacAppInstaller.run("/usr/bin/tar", ["-tzf", a.tarball.path])
        try #require(listing.contains(".."), "fixture must really contain a .. path: \(listing)")
        await a.expectRejected()
    }

    @Test func dotDotMemberNextToTheRealOneIsRejected() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeTarball([
            (stored: "\(memberDir)/mlxr", archived: "\(memberDir)/mlxr", member: .file(bin)),
            (stored: "zz/evil", archived: "\(memberDir)/../../evil", member: .file(Data("x".utf8))),
        ])
        await a.expectRejected()
        #expect(!FileManager.default.fileExists(atPath: a.root.appendingPathComponent("evil").path))
    }

    @Test func absolutePathMemberIsRejected() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        let abs = a.root.appendingPathComponent("abs").path + "/mlxr"
        try a.makeTarball([(stored: "zz/mlxr", archived: abs, member: .file(bin))])
        await a.expectRejected()
        #expect(!FileManager.default.fileExists(atPath: abs))
    }

    @Test func extraMemberOutsideTheReleaseFolderIsRejected() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeTarball([
            (stored: "\(memberDir)/mlxr", archived: "\(memberDir)/mlxr", member: .file(bin)),
            (stored: "other/payload", archived: "other/payload", member: .file(Data("x".utf8))),
        ])
        await a.expectRejected()
    }

    @Test func memberFromAnotherVersionFolderIsRejected() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeTarball([(stored: "mlxr-1.0.0-macos-arm64/mlxr", archived: "mlxr-1.0.0-macos-arm64/mlxr", member: .file(bin))])
        await a.expectRejected()
    }

    @Test func archiveWithoutTheBinaryIsRejected() async throws {
        let a = try Area()
        defer { a.cleanup() }
        try a.makeTarball([(stored: "\(memberDir)/LICENSE", archived: "\(memberDir)/LICENSE", member: .file(Data("MIT".utf8)))])
        await a.expectRejected()
    }

    @Test func notATarballIsRejected() async throws {
        let a = try Area()
        defer { a.cleanup() }
        try Data("not a tarball at all".utf8).write(to: a.tarball)
        await a.expectRejected()
    }
}

// MARK: - AC-4: Mach-O arm64 check and size cap

@Suite("CLI update: binary validation")
struct CLIValidationTests {
    private static let magic: [UInt8] = [0xCF, 0xFA, 0xED, 0xFE]
    private static let arm64: [UInt8] = [0x0C, 0x00, 0x00, 0x01]
    private static let x86_64: [UInt8] = [0x07, 0x00, 0x00, 0x01]

    private func rejects(_ data: Data, sourceLocation: SourceLocation = #_sourceLocation) async throws {
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: data)
        await a.expectRejected(sourceLocation: sourceLocation)
    }

    @Test func wrongMagicIsRejected() async throws {
        try await rejects(header(magic: [0x7F, 0x45, 0x4C, 0x46], cpu: Self.arm64))  // ELF
        try await rejects(Data("#!/bin/sh\necho pwned\n".utf8))
    }

    @Test func fatBinaryIsRejected() async throws {
        try await rejects(header(magic: [0xCA, 0xFE, 0xBA, 0xBE], cpu: [0x00, 0x00, 0x00, 0x02]))
        try await rejects(header(magic: [0xBE, 0xBA, 0xFE, 0xCA], cpu: [0x02, 0x00, 0x00, 0x00]))
    }

    @Test func x86_64BinaryIsRejected() async throws {
        try await rejects(header(magic: Self.magic, cpu: Self.x86_64))
    }

    @Test func emptyMemberIsRejected() async throws {
        try await rejects(Data())
    }

    @Test func truncatedHeaderIsRejected() async throws {
        try await rejects(Data(Self.magic))
        try await rejects(Data(Self.magic + [0x0C]))
    }

    @Test func defaultSizeCapIs256MiB() {
        #expect(CLIInstaller.defaultMaxBinaryBytes == 256 << 20)
        #expect(CLIInstaller().maxBinaryBytes == 256 << 20)
    }

    @Test func memberOverTheCapIsRejected() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        await a.expectRejected(CLIInstaller(maxBinaryBytes: bin.count - 1))
    }

    @Test func memberExactlyAtTheCapInstalls() async throws {
        let bin = try requireBinary()
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        try await a.install(CLIInstaller(maxBinaryBytes: bin.count))
        #expect(try Data(contentsOf: a.target) == bin)
    }
}

// MARK: - AC-5: code signature

@Suite("CLI update: code signature")
struct CLISignatureTests {
    @Test func unsignedBinaryIsRejected() async throws {
        let bin = try unsigned(try requireBinary())
        // still a valid arm64 Mach-O: only the signature check can stop it
        #expect(Array(bin.prefix(4)) == [0xCF, 0xFA, 0xED, 0xFE])
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        await a.expectRejected()
    }

    @Test func tamperedBinaryIsRejected() async throws {
        let bin = tamper(try requireBinary())
        #expect(Array(bin.prefix(4)) == [0xCF, 0xFA, 0xED, 0xFE])
        let a = try Area()
        defer { a.cleanup() }
        try a.makeRelease(binary: bin)
        await a.expectRejected()
    }
}
