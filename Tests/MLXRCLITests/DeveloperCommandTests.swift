import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

/// Records what the injected services were asked; nothing touches the network or a model.
final class FakeServices: @unchecked Sendable {
    private let lock = NSLock()
    private var s: [(String, Int)] = [], b: [BenchOptions] = [], t: [TokenizeOptions] = []
    var searches: [(String, Int)] { lock.withLock { s } }
    var benches: [BenchOptions] { lock.withLock { b } }
    var tokenizes: [TokenizeOptions] { lock.withLock { t } }
    let catalog: [HubModel]

    init(count: Int = 8) {
        catalog = (0..<count).map { i in
            let json = #"{"id":"mlx-community/Model-\#(i)-4bit","downloads":\#(1000 - i),"tags":["qwen3"],"pipeline_tag":"text-generation"}"#
            return try! JSONDecoder().decode(HubModel.self, from: Data(json.utf8))
        }
    }

    var services: CLIServices {
        CLIServices(
            search: { q, limit in
                self.lock.withLock { self.s.append((q, limit)) }
                return Array(self.catalog.prefix(limit))
            },
            bench: { o, console in
                self.lock.withLock { self.b.append(o) }
                console.out("weights: 1 B\nBENCH-OUTPUT-PASSED-THROUGH\n")
            },
            tokenize: { o, console in
                self.lock.withLock { self.t.append(o) }
                console.out("[1, 2, 3]\n")
            })
    }

    func table() -> [Command] { CLI.commands(services: services) }
}

@Suite("search, tokenize and bench through the table (AC-26)")
struct DeveloperCommandTests {
    @Test func searchJSONHonorsTheLimit() async throws {
        let fake = FakeServices()
        let cap = Capture()
        #expect(await CLI.run(["search", "qwen", "--limit", "5", "--json"], console: cap.console(), commands: fake.table()) == 0, "\(cap.stderr)")
        let rows = try #require(try JSONSerialization.jsonObject(with: Data(cap.stdout.utf8)) as? [[String: Any]])
        #expect(rows.count == 5)
        #expect(rows.first?["id"] as? String == "mlx-community/Model-0-4bit")
        #expect(fake.searches.count == 1 && fake.searches[0].0 == "qwen" && fake.searches[0].1 == 5)
        #expect(cap.stderr.isEmpty)
    }

    @Test func searchPlainTextHasOneRowPerModel() async {
        let fake = FakeServices(count: 3)
        let cap = Capture()
        #expect(await CLI.run(["search", "qwen"], console: cap.console(), commands: fake.table()) == 0, "\(cap.stderr)")
        let lines = cap.stdout.split(separator: "\n").map(String.init)
        #expect(lines.count == 3)
        #expect(lines.allSatisfy { $0.contains("mlx-community/Model-") })
        #expect((try? JSONSerialization.jsonObject(with: Data(cap.stdout.utf8))) == nil)
        #expect(fake.searches.first?.1 == 25, "default limit")
    }

    @Test func searchMultiWordQueryAndFlagBeforeQuery() async {
        let fake = FakeServices()
        let cap = Capture()
        #expect(await CLI.run(["search", "--limit=2", "qwen", "4b"], console: cap.console(), commands: fake.table()) == 0, "\(cap.stderr)")
        #expect(fake.searches.first?.0 == "qwen 4b" && fake.searches.first?.1 == 2)
    }

    @Test func badLimitsFailBeforeAnyRequest() async {
        for bad in ["0", "abc", "101", "-1"] {
            let fake = FakeServices()
            let cap = Capture()
            #expect(await CLI.run(["search", "q", "--limit", bad], console: cap.console(), commands: fake.table()) == 1, "\(bad)")
            #expect(cap.stderr.hasPrefix("Error: invalid argument \"\(bad)\" for \"--limit\" flag"), "\(bad): \(cap.stderr)")
            #expect(fake.searches.isEmpty, "\(bad)")
            #expect(cap.stdout.isEmpty, "\(bad)")
        }
        let fake = FakeServices()
        let cap = Capture()
        #expect(await CLI.run(["search", "q", "--limit", "100"], console: cap.console(), commands: fake.table()) == 0)
        #expect(fake.searches.first?.1 == 100)
    }

    @Test func searchWithoutQueryIsAUsageError() async {
        let fake = FakeServices()
        let cap = Capture()
        #expect(await CLI.run(["search"], console: cap.console(), commands: fake.table()) == 1)
        #expect(cap.stderr == "Error: requires at least 1 arg(s), only received 0\n" || cap.stderr == "Error: accepts 1 arg(s), received 0\n", "\(cap.stderr)")
        #expect(fake.searches.isEmpty)
    }

    @Test func searchFailuresReachErrorOutput() async {
        struct Down: LocalizedError { var errorDescription: String? { "offline" } }
        let services = CLIServices(search: { _, _ in throw Down() }, bench: { _, _ in }, tokenize: { _, _ in })
        let cap = Capture()
        #expect(await CLI.run(["search", "q"], console: cap.console(), commands: CLI.commands(services: services)) == 1)
        #expect(cap.stderr == "Error: offline\n")
    }

    @Test func tokenizeAcceptsFlagsBeforeAndAfterTheModel() async {
        for args in [["tokenize", "--ids", "m", "hello", "world"], ["tokenize", "m", "hello", "world", "--ids"]] {
            let fake = FakeServices()
            let cap = Capture()
            #expect(await CLI.run(args, console: cap.console(), commands: fake.table()) == 0, "\(args): \(cap.stderr)")
            #expect(fake.tokenizes == [TokenizeOptions(model: "m", text: "hello world", idsOnly: true)], "\(args)")
            #expect(cap.stdout == "[1, 2, 3]\n")
        }
        let fake = FakeServices()
        #expect(await CLI.run(["tokenize", "m", "hi"], console: Capture().console(), commands: fake.table()) == 0)
        #expect(fake.tokenizes == [TokenizeOptions(model: "m", text: "hi", idsOnly: false)])
    }

    @Test func tokenizeNeedsAModelAndText() async {
        let fake = FakeServices()
        let cap = Capture()
        #expect(await CLI.run(["tokenize", "m"], console: cap.console(), commands: fake.table()) == 1)
        #expect(cap.stderr.hasPrefix("Error: requires at least 2 arg(s), only received 1"), "\(cap.stderr)")
        #expect(fake.tokenizes.isEmpty)
    }

    @Test func benchReceivesParsedOptionsAndPassesOutputThrough() async {
        let fake = FakeServices()
        let cap = Capture()
        #expect(await CLI.run(["bench", "--decode", "m", "--ctx", "8192,32768"], console: cap.console(), commands: fake.table()) == 0, "\(cap.stderr)")
        #expect(fake.benches == [BenchOptions(model: "m", decodeOnly: true, contexts: "8192,32768")])
        #expect(cap.stdout == "weights: 1 B\nBENCH-OUTPUT-PASSED-THROUGH\n")
        #expect(cap.stderr.isEmpty)

        let plain = FakeServices()
        #expect(await CLI.run(["bench", "m"], console: Capture().console(), commands: plain.table()) == 0)
        #expect(plain.benches == [BenchOptions(model: "m")])
    }

    @Test func benchAndTokenizeStayUnderDeveloperHeading() {
        let table = FakeServices().table()
        #expect(table.first { $0.name == "bench" }?.group == "Developer")
        #expect(table.first { $0.name == "tokenize" }?.group == "Developer")
        #expect(table.first { $0.name == "search" }?.options.contains { $0.long == "limit" } == true)
        #expect(table.first { $0.name == "search" }?.options.contains { $0.long == "json" } == true)
    }
}
