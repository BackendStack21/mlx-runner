import Foundation
import Testing
import MLXEngine
@testable import MLXRCLI

@Suite("chat history and line source (AC-18)")
struct ChatHistoryTests {
    func tempFile() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-history-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("history")
    }

    @Test func linesAreAppendedAndReadBackInOrderAcrossInstances() throws {
        let file = tempFile(); defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        let a = ChatHistory(file: file)
        #expect(a.lines().isEmpty, "no file yet")
        a.append("first"); a.append("second")
        #expect(ChatHistory(file: file).lines() == ["first", "second"])
    }

    @Test func theHistoryFileIsPrivate() throws {
        let file = tempFile(); defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        ChatHistory(file: file).append("secret")
        let mode = (try? FileManager.default.attributesOfItem(atPath: file.path)[.posixPermissions] as? NSNumber)?.intValue
        #expect(mode == 0o600, "\(String(describing: mode))")
    }

    @Test func aHistoryWithoutAFileKeepsNothingAndTheChatStillWorks() async throws {
        let h = try ChatHarness()
        let none = ChatHistory(file: nil)
        await h.session(ScriptedLines(["hi"]), history: none).run()
        #expect(none.lines().isEmpty)
        #expect(h.engine.chats.count == 1)
        #expect(!FileManager.default.fileExists(atPath: h.historyFile.path))
    }

    @Test func theLiveHistoryLivesUnderApplicationSupportMLXRunner() {
        let path = ChatHistory.live.file?.path ?? ""
        #expect(path.contains("Application Support/MLX Runner/"), "\(path)")
        #expect(ChatHistory.live.file?.lastPathComponent == "history")
    }

    @Test func eachSubmittedLineIsRecordedIncludingSlashCommands() async throws {
        let h = try ChatHarness()
        await h.chat(["hello", "/set verbose", "world"])
        #expect(ChatHistory(file: h.historyFile).lines() == ["hello", "/set verbose", "world"])
    }

    @Test func blankLinesAreNotRecorded() async throws {
        let h = try ChatHarness()
        await h.chat(["", "  ", "hello"])
        #expect(ChatHistory(file: h.historyFile).lines() == ["hello"])
    }

    @Test func setNohistoryStopsRecordingAndSetHistoryResumes() async throws {
        let h = try ChatHarness()
        await h.chat(["one", "/set nohistory", "two", "/set history", "three"])
        let lines = ChatHistory(file: h.historyFile).lines()
        #expect(lines.contains("one") && lines.contains("three"), "\(lines)")
        #expect(!lines.contains("two"), "\(lines)")
    }

    @Test func ollamaNohistoryEnvironmentKeepsTheFileUntouched() async throws {
        let h = try ChatHarness(env: ["OLLAMA_NOHISTORY": "1"])
        await h.chat(["private question"])
        #expect(!FileManager.default.fileExists(atPath: h.historyFile.path))
        #expect(h.engine.chats.count == 1, "the chat itself still works")
    }

    @Test func existingHistoryIsKeptWhenTheChatAppends() async throws {
        let h = try ChatHarness()
        ChatHistory(file: h.historyFile).append("older")
        await h.chat(["newer"])
        #expect(ChatHistory(file: h.historyFile).lines() == ["older", "newer"])
    }

    // MARK: readLine fallback

    @Test func theFallbackSourceReadsConsoleLinesAndEndsAtNil() {
        let cap = Capture()
        let console = cap.console()
        let feed = LineFeed(["one", "two"])
        console.readLine = { feed.next() }
        let source = ConsoleLineSource(console: console)
        #expect(source.read(prompt: ">>> ") == .line("one"))
        #expect(source.read(prompt: ">>> ") == .line("two"))
        #expect(source.read(prompt: ">>> ") == .eof)
        #expect(cap.stdout.isEmpty, "prompts never go to stdout")
    }
}
