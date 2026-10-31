import CryptoKit
import Foundation
import Testing
@testable import MLXEngine

@Suite("Hub security and downloads")
struct HubTests {
    // MARK: Repo ids

    @Test func acceptsPlainRepoIDs() {
        for id in ["mlx-community/Qwen3-4B-4bit", "a/b", "Org_1/model.v2-final", "0x/9"] {
            #expect(RepoID.isValid(id), "\(id)")
        }
    }

    @Test func rejectsUnsafeRepoIDs() {
        for id in ["", "noslash", "a/b/c", "../etc", "a/..", "../../x", "a/..b", "a..b/c", "/a/b", "a/b/", ".a/b",
                   "a/.b", "-a/b", "a/_b", "a b/c", "a/b?x=1", "a/b#f", "a%2F../b", "a/b\0", "ä/b", "a\\b/c", "~/b"] {
            #expect(!RepoID.isValid(id), "\(id)")
        }
        #expect(throws: HubError.invalidRepoID("../x")) { try ModelStore.directory(for: "../x") }
        #expect((try? ModelStore.directory(for: "org/name"))?.path.hasSuffix("models/org/name") == true)
    }

    @Test func safeFileNames() {
        #expect(RepoID.isSafeFileName("model-00001-of-00002.safetensors"))
        for n in ["", ".", "..", ".complete", "a/b", "../x", "a\\b"] { #expect(!RepoID.isSafeFileName(n), "\(n)") }
    }

    // MARK: Redirect auth policy

    @Test func trustedHosts() {
        for s in ["https://huggingface.co/x", "https://HuggingFace.co/x", "https://cdn-lfs.huggingface.co/x",
                  "https://hf.co/x", "https://cas-bridge.xethub.hf.co/x", "https://us.aws.cdn.hf.co/x"] {
            #expect(HubAuthPolicy.isTrustedHost(URL(string: s)), "\(s)")
        }
        for s in ["http://huggingface.co/x", "https://huggingface.co.evil.com/x", "https://evilhuggingface.co/x",
                  "https://evil.com/huggingface.co", "https://xhf.co/x", "ftp://hf.co/x"] {
            #expect(!HubAuthPolicy.isTrustedHost(URL(string: s)), "\(s)")
        }
        #expect(!HubAuthPolicy.isTrustedHost(nil))
    }

    @Test func authorizationOnlyFollowsSameTrustedHost() {
        let hf = URL(string: "https://huggingface.co/org/m/resolve/main/model.safetensors")
        func fwd(_ to: String) -> Bool { HubAuthPolicy.shouldForwardAuthorization(from: hf, to: URL(string: to)) }
        // Relative hop for small files (gated repos need the token here).
        #expect(fwd("https://huggingface.co/api/resolve-cache/models/org/m/abc/config.json"))
        // Cross-host CDN hops carry a signed URL; the token must not leak there.
        #expect(!fwd("https://us.aws.cdn.hf.co/xet-bridge-us/abc?Signature=x"))
        #expect(!fwd("https://cas-bridge.xethub.hf.co/xet-bridge-us/abc"))
        #expect(!fwd("https://cdn-lfs-us-1.hf.co/repos/abc"))
        #expect(!fwd("http://huggingface.co/x"))
        #expect(!fwd("https://evil.com/x"))
        #expect(!fwd("https://huggingface.co.evil.com/x"))
        #expect(!HubAuthPolicy.shouldForwardAuthorization(from: URL(string: "http://huggingface.co/x"),
                                                         to: URL(string: "https://huggingface.co/y")))
    }

    @Test func redirectedRequestStripsOrRestoresHeaders() {
        var original = URLRequest(url: URL(string: "https://huggingface.co/o/m/resolve/main/model.safetensors")!)
        original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        original.setValue("bytes=100-", forHTTPHeaderField: "Range")

        var cdn = URLRequest(url: URL(string: "https://us.aws.cdn.hf.co/blob?sig=1")!)
        cdn.setValue("Bearer secret", forHTTPHeaderField: "Authorization")   // URLSession's carried-over copy
        let r1 = HubAuthPolicy.redirectedRequest(cdn, from: original.url, original: original)
        #expect(r1.value(forHTTPHeaderField: "Authorization") == nil)
        #expect(r1.value(forHTTPHeaderField: "Range") == "bytes=100-")

        let same = URLRequest(url: URL(string: "https://huggingface.co/api/resolve-cache/x")!)
        let r2 = HubAuthPolicy.redirectedRequest(same, from: original.url, original: original)
        #expect(r2.value(forHTTPHeaderField: "Authorization") == "Bearer secret")

        // Back to huggingface.co from a CDN host: still stripped (the hop came from elsewhere).
        let back = HubAuthPolicy.redirectedRequest(same, from: cdn.url, original: original)
        #expect(back.value(forHTTPHeaderField: "Authorization") == nil)
    }

    // MARK: Links

    @Test func linkPolicy() {
        #expect(LinkPolicy.decide(URL(string: "https://huggingface.co/mlx-community")!) == .open)
        #expect(LinkPolicy.decide(URL(string: "https://www.huggingface.co/x")!) == .open)
        #expect(LinkPolicy.decide(URL(string: "http://huggingface.co/x")!) == .confirm)
        #expect(LinkPolicy.decide(URL(string: "https://github.com/x")!) == .confirm)
        #expect(LinkPolicy.decide(URL(string: "https://huggingface.co.evil.com/x")!) == .confirm)
        #expect(LinkPolicy.decide(URL(string: "mailto:a@b.c")!) == .confirm)
        for s in ["file:///etc/passwd", "smb://host/share", "x-apple.systempreferences:com.apple", "javascript:alert(1)",
                  "ssh://host", "vnc://host", "relative/link.md", "#anchor", "https:///nohost"] {
            #expect(LinkPolicy.decide(URL(string: s)!) == .discard, "\(s)")
        }
    }

    @Test func sanitizeStripsDiscardedLinks() throws {
        let a = try AttributedString(markdown: "[a](file:///etc/passwd) [b](smb://x/y) [c](https://huggingface.co) [d](https://example.com)")
        let s = LinkPolicy.sanitize(a)
        // Only c and d keep a link, and each covers exactly its own label.
        let linked = s.runs.compactMap { run in run.link.map { (String(s[run.range].characters), $0) } }
        #expect(linked.map(\.0) == ["c", "d"])
        #expect(linked.map(\.1) == [URL(string: "https://huggingface.co")!, URL(string: "https://example.com")!])
        #expect(String(s.characters) == String(a.characters))
        #expect(a.runs.compactMap(\.link).count == 4)
    }

    // MARK: Delegate hookup

    final class Box: @unchecked Sendable { var request: URLRequest? }

    private func followRedirect(_ delegate: URLSessionTaskDelegate, to target: String) -> URLRequest? {
        var original = URLRequest(url: URL(string: "https://huggingface.co/o/m/resolve/main/model.safetensors")!)
        original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        let task = URLSession.shared.dataTask(with: original)   // never resumed
        defer { task.cancel() }
        let response = HTTPURLResponse(url: original.url!, statusCode: 302, httpVersion: "HTTP/1.1",
                                       headerFields: ["Location": target])!
        var proposed = URLRequest(url: URL(string: target)!)
        proposed.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        let box = Box()
        delegate.urlSession?(URLSession.shared, task: task, willPerformHTTPRedirection: response,
                             newRequest: proposed) { box.request = $0 }
        return box.request
    }

    @Test func delegatesStripAuthorizationOnCrossHostRedirects() {
        let details = HubModelDetails(id: "o/m", files: [], modelType: nil, bits: nil,
                                      readmeURL: URL(string: "https://huggingface.co/o/m")!,
                                      downloads: nil, likes: nil, tags: [], license: nil)
        let download = ModelDownload(details: details, destination: FileManager.default.temporaryDirectory)
        defer { download.cancel() }
        for delegate in [HubRedirectDelegate() as URLSessionTaskDelegate, download] {
            let cdn = followRedirect(delegate, to: "https://us.aws.cdn.hf.co/blob?Signature=x")
            #expect(cdn != nil)
            #expect(cdn?.value(forHTTPHeaderField: "Authorization") == nil)
            let same = followRedirect(delegate, to: "https://huggingface.co/api/resolve-cache/models/o/m/x/config.json")
            #expect(same?.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        }
    }

    // MARK: Hashing

    @Test func sha256OfFileStreamsInChunks() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-hash-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let abc = dir.appendingPathComponent("abc")
        try Data("abc".utf8).write(to: abc)
        #expect(try FileHasher.sha256(of: abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

        var bytes = [UInt8](repeating: 0, count: 100_003)
        for i in bytes.indices { bytes[i] = UInt8(truncatingIfNeeded: i &* 31 &+ 7) }
        let big = dir.appendingPathComponent("big")
        try Data(bytes).write(to: big)
        let expected = SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
        #expect(try FileHasher.sha256(of: big, chunkSize: 4096) == expected)
        #expect(try FileHasher.sha256(of: big) == expected)

        let empty = dir.appendingPathComponent("empty")
        try Data().write(to: empty)
        #expect(try FileHasher.sha256(of: empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        #expect(FileHasher.isSHA256Hex(expected))
        #expect(!FileHasher.isSHA256Hex("XYZ"))
    }

    // MARK: Range resume

    @Test func resumeStartPlan() {
        #expect(ResumePlan.start(partialBytes: 0, expectedSize: 100) == .fresh)
        #expect(ResumePlan.start(partialBytes: 40, expectedSize: 100) == .resume(from: 40))
        #expect(ResumePlan.start(partialBytes: 100, expectedSize: 100) == .verifyOnly)
        #expect(ResumePlan.start(partialBytes: 140, expectedSize: 100) == .fresh)
        #expect(ResumePlan.start(partialBytes: 40, expectedSize: nil) == .resume(from: 40))
    }

    @Test func resumeResponseHandling() {
        #expect(ResumePlan.response(status: 206, requestedOffset: 40, contentRange: "bytes 40-99/100") == .append)
        #expect(ResumePlan.response(status: 206, requestedOffset: 40, contentRange: "bytes 0-99/100") == .retryWithoutRange)
        #expect(ResumePlan.response(status: 206, requestedOffset: 40, contentRange: nil) == .retryWithoutRange)
        #expect(ResumePlan.response(status: 206, requestedOffset: 0, contentRange: "bytes 0-99/100") == .retryWithoutRange)
        #expect(ResumePlan.response(status: 200, requestedOffset: 40, contentRange: nil) == .restart)
        #expect(ResumePlan.response(status: 200, requestedOffset: 0, contentRange: nil) == .restart)
        #expect(ResumePlan.response(status: 416, requestedOffset: 40, contentRange: "bytes */100") == .retryWithoutRange)
        #expect(ResumePlan.response(status: 401, requestedOffset: 0, contentRange: nil) == .fail(401))
        #expect(ResumePlan.response(status: 503, requestedOffset: 40, contentRange: nil) == .fail(503))
        #expect(ResumePlan.contentRangeStart("bytes 123-456/789") == 123)
        #expect(ResumePlan.contentRangeStart("Bytes  7-8/*") == 7)
        #expect(ResumePlan.contentRangeStart("items 1-2/3") == nil)
    }

    // MARK: Corrupt JSON quarantine

    struct Doc: Codable, Equatable { var name: String }

    @Test func safeJSONFileQuarantinesCorruptFiles() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("mlxr-json-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        let url = dir.appendingPathComponent("chats.json")

        guard case .missing = SafeJSONFile.load([Doc].self, from: url) else { Issue.record("expected missing"); return }

        try JSONEncoder().encode([Doc(name: "a")]).write(to: url)
        guard case .loaded(let docs) = SafeJSONFile.load([Doc].self, from: url) else { Issue.record("expected loaded"); return }
        #expect(docs == [Doc(name: "a")])

        let garbage = Data("[{\"name\": 1".utf8)
        try garbage.write(to: url)
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        guard case .corrupt(_, let moved?) = SafeJSONFile.load([Doc].self, from: url, now: now) else {
            Issue.record("expected corrupt+moved"); return
        }
        #expect(!fm.fileExists(atPath: url.path))
        #expect(moved.lastPathComponent.hasPrefix("chats.corrupt-"))
        #expect(moved.pathExtension == "json")
        #expect(try Data(contentsOf: moved) == garbage)

        // A second corrupt file in the same second gets a distinct name.
        try garbage.write(to: url)
        guard case .corrupt(_, let moved2?) = SafeJSONFile.load([Doc].self, from: url, now: now) else {
            Issue.record("expected corrupt+moved"); return
        }
        #expect(moved2 != moved)
        #expect(fm.fileExists(atPath: moved.path) && fm.fileExists(atPath: moved2.path))
    }

    // MARK: Keychain

    @Test func keychainRoundTrip() {
        let service = "de.21no.mlxrunner.tests.\(UUID().uuidString)"
        defer { Keychain.delete(account: "t", service: service) }
        #expect(Keychain.read(account: "t", service: service) == nil)
        #expect(Keychain.write("one", account: "t", service: service))
        #expect(Keychain.read(account: "t", service: service) == "one")
        #expect(Keychain.write("two", account: "t", service: service))
        #expect(Keychain.read(account: "t", service: service) == "two")
        #expect(Keychain.write("", account: "t", service: service))
        #expect(Keychain.read(account: "t", service: service) == nil)
    }
}
