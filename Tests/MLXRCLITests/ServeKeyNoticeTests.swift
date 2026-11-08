import Testing
@testable import MLXRCLI

@Suite("serve shows only a key it generated")
struct ServeKeyNoticeTests {
    @Test func generatedKeyIsShownOnce() {
        let line = serveKeyNotice(key: "mlxr-generated-123", generated: true)
        #expect(line == "API key (send as 'Authorization: Bearer <key>'): mlxr-generated-123")
    }

    @Test func suppliedKeyIsNeverEchoed() throws {
        let line = try #require(serveKeyNotice(key: "my-own-secret", generated: false))
        #expect(!line.contains("my-own-secret"))
        #expect(line == "API key required: send it as 'Authorization: Bearer <key>'")
    }

    @Test func noKeyNoNotice() {
        #expect(serveKeyNotice(key: nil, generated: false) == nil)
    }
}
