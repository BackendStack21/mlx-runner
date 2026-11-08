import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("list and its formatting (AC-9)")
struct ModelListTests {
    static let header = "NAME                             ID              SIZE      MODIFIED \n"
    static let row06 = "mlx-community/Qwen3-0.6B-4bit    51a1a9bb2c1e    350 MB    3 weeks ago    \n"
    static let row8 = "mlx-community/Qwen3-8B-4bit      9c29d1d07c97    5.2 GB    5 minutes ago    \n"

    @Test func listRendersTheOllamaTableByteForByte() async {
        let r = await LibraryFixture.standard().run(["list"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == Self.header + Self.row06 + Self.row8)
        #expect(r.err.isEmpty)
    }

    @Test func lsIsAnAliasAndEmptyLibraryPrintsOnlyTheHeader() async {
        let r = await LibraryFixture(models: []).run(["ls"])
        #expect(r.code == 0, "\(r.err)")
        #expect(r.out == "NAME    ID    SIZE    MODIFIED \n")
    }

    @Test func prefixFilterIsCaseInsensitiveOnTheNameOrTheShortName() async {
        let f = LibraryFixture.standard()
        let short = await f.run(["list", "QWEN3-8"])
        #expect(short.out.contains("Qwen3-8B-4bit") && !short.out.contains("Qwen3-0.6B"), "\(short.out)")
        let full = await f.run(["list", "MLX-COMMUNITY/qwen3-0"])
        #expect(full.out.contains("Qwen3-0.6B-4bit") && !full.out.contains("Qwen3-8B"), "\(full.out)")
        let none = await f.run(["list", "zzz"])
        #expect(none.code == 0)
        #expect(!none.out.contains("mlx-community"))
    }

    @Test func jsonListsEveryModelAsValidJSON() async throws {
        let r = await LibraryFixture.standard().run(["list", "--json"])
        #expect(r.code == 0, "\(r.err)")
        let rows = try #require(try JSONSerialization.jsonObject(with: Data(r.out.utf8)) as? [[String: Any]])
        #expect(rows.count == 2)
        #expect(rows[0]["name"] as? String == LibraryFixture.qwen06)
        #expect(rows[0]["id"] as? String == "51a1a9bb2c1e")
        #expect(rows[0]["size"] as? Int == 350_000_000)
        #expect(rows[1]["name"] as? String == LibraryFixture.qwen8)
        #expect(rows[1]["modified"] is String)
    }

    @Test func presetsAreListedByName() async throws {
        let f = LibraryFixture.standard()
        try f.presets.save(Preset(name: "mychat", repo: LibraryFixture.qwen8))
        let r = await f.run(["list"])
        let lines = r.out.split(separator: "\n").map(String.init)
        #expect(lines.contains { $0.hasPrefix("mychat") }, "\(r.out)")
        #expect(lines.contains { $0.hasPrefix(LibraryFixture.qwen8) })
    }

    @Test func humanBytesUsesBase1000UnitsWithOneDecimalBelowTen() {
        let cases: [(Int, String)] = [
            (0, "0 B"), (999, "999 B"), (1_500, "1.5 KB"), (350_000_000, "350 MB"), (512_000_000, "512 MB"),
            (2_000_000_000, "2.0 GB"), (5_200_000_000, "5.2 GB"), (12_345_000_000, "12 GB"),
        ]
        for (n, want) in cases { #expect(CLI.humanBytes(n) == want, "\(n)") }
    }

    @Test func relativeTimeReadsLikeOllama() {
        let now = LibraryFixture.now
        let cases: [(TimeInterval, String)] = [
            (0.2, "Less than a second ago"), (60, "About a minute ago"), (300, "5 minutes ago"),
            (3 * 3_600, "3 hours ago"), (3 * 86_400, "3 days ago"), (21 * 86_400, "3 weeks ago"),
            (90 * 86_400, "3 months ago"), (3 * 365 * 86_400, "3 years ago"),
        ]
        for (age, want) in cases { #expect(CLI.relativeTime(now.addingTimeInterval(-age), now: now) == want, "\(age)") }
    }

    @Test func tableAlignsColumnsAndEndsEveryCellWithFourSpaces() {
        let t = CLI.renderTable(header: ["A", "BB"], rows: [["xxx", "1"], ["y", "22"]])
        #expect(t == "A      BB \nxxx    1    \ny      22    \n")
        #expect(CLI.renderTable(header: ["A", "B"], rows: []) == "A    B \n")
    }

    @Test func modelIDIsTheFirstTwelveHexOfTheRepoIdDigest() {
        #expect(CLI.modelID(LibraryFixture.qwen06) == "51a1a9bb2c1e")
        #expect(CLI.modelID(LibraryFixture.qwen8) == "9c29d1d07c97")
    }
}
