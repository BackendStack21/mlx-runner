import Foundation
import MLXEngine
import Observation
import SwiftUI

// MARK: - Persisted chat model

struct StoredMessage: Codable, Identifiable, Hashable {
    var id = UUID()
    var role: ChatMessage.Role
    var content: String
    var reasoning: String?
    var stats: GenerationStats?
    var model: String?
    var date = Date()
    var error: String?
    var reasoningSeconds: Double?
}

struct Conversation: Codable, Identifiable, Hashable {
    var id = UUID()
    var title: String = "New Chat"
    var messages: [StoredMessage] = []
    var created = Date()
    var updated = Date()
    var systemPrompt: String?
}

enum SidebarItem: Hashable {
    case chat(UUID)
    case discover
    case models
    case server
}

struct DownloadState: Equatable {
    var progress: Double = 0
    var bytes: Int64 = 0
    var total: Int64 = 0
    var speed: Double = 0
    var error: String?
}

enum LoadState: Equatable {
    case idle
    case loading(String)
    case loaded(LoadedModelInfo)
    case failed(String)
}

// MARK: - App state

@MainActor @Observable
final class AppState {
    var selection: SidebarItem? = .discover
    var conversations: [Conversation] = []
    var localModels: [LocalModel] = []
    var loadState: LoadState = .idle
    var downloads: [String: DownloadState] = [:]
    var generatingConversation: UUID?
    var showInspector = true
    var draft: String = ""
    var liveTokens = 0
    var liveTokensPerSecond: Double = 0

    // Settings (persisted in UserDefaults)
    var sampling = SamplingParameters() { didSet { saveSettings() } }
    var systemPrompt: String = "" { didSet { saveSettings() } }
    var contextLength: Int = 8192 { didSet { saveSettings() } }
    var enableThinking: Bool = true { didSet { saveSettings() } }
    var lastModelPath: String? { didSet { saveSettings() } }

    @ObservationIgnored private var activeDownloads: [String: ModelDownload] = [:]
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var saveTask: Task<Void, Never>?
    let engine = InferenceEngine.shared

    static let supportDir: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MLX Runner", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }()
    private var chatsURL: URL { Self.supportDir.appendingPathComponent("chats.json") }

    init() {
        loadSettings()
        loadChats()
        refreshModels()
        if let first = conversations.first { selection = .chat(first.id) }
        // Compile Metal kernels in the background so the first load is instant.
        Task.detached(priority: .utility) { GPU.prepare() }
    }

    var loadedInfo: LoadedModelInfo? {
        if case .loaded(let i) = loadState { return i }
        return nil
    }

    var isGenerating: Bool { generatingConversation != nil }

    // MARK: Models

    func refreshModels() {
        Task.detached(priority: .userInitiated) {
            let models = ModelStore.scan()
            await MainActor.run { self.localModels = models }
        }
    }

    func load(_ model: LocalModel) {
        guard model.isSupported else {
            loadState = .failed("\(model.modelType ?? "This") architecture is not supported yet.")
            return
        }
        stopGeneration()
        loadState = .loading(model.name)
        let ctx = contextLength
        Task {
            do {
                let info = try await engine.load(directory: model.directory, contextLength: ctx)
                loadState = .loaded(info)
                lastModelPath = model.directory.path
            } catch {
                loadState = .failed(error.localizedDescription)
            }
        }
    }

    func eject() {
        stopGeneration()
        engine.unload()
        loadState = .idle
    }

    func reloadLast() {
        guard loadedInfo == nil, let p = lastModelPath,
              let m = localModels.first(where: { $0.directory.path == p }) else { return }
        load(m)
    }

    func delete(_ model: LocalModel) {
        if loadedInfo?.directory == model.directory { eject() }
        try? ModelStore.delete(model)
        refreshModels()
    }

    // MARK: Downloads

    func download(_ details: HubModelDetails) {
        guard activeDownloads[details.id] == nil else { return }
        let dl = ModelDownload(details: details, destination: ModelStore.directory(for: details.id))
        activeDownloads[details.id] = dl
        downloads[details.id] = DownloadState(total: Int64(details.downloadSize))
        let repo = details.id
        let throttle = Throttle(interval: 0.1)
        dl.onProgress = { [weak self] p, done, total, speed in
            guard throttle.ready() || p >= 1 else { return }
            Task { @MainActor in
                self?.downloads[repo] = DownloadState(progress: p, bytes: done, total: total, speed: speed)
            }
        }
        Task {
            do {
                _ = try await dl.start()
                downloads[repo] = nil
                refreshModels()
            } catch is CancellationError {
                downloads[repo] = nil
            } catch {
                downloads[repo]?.error = error.localizedDescription
            }
            activeDownloads[repo] = nil
        }
    }

    func cancelDownload(_ repo: String) {
        activeDownloads[repo]?.cancel()
        activeDownloads[repo] = nil
        downloads[repo] = nil
    }

    // MARK: Conversations

    @discardableResult
    func newChat() -> UUID {
        if let empty = conversations.first(where: { $0.messages.isEmpty }) {
            selection = .chat(empty.id)
            return empty.id
        }
        let c = Conversation()
        conversations.insert(c, at: 0)
        selection = .chat(c.id)
        scheduleSave()
        return c.id
    }

    func deleteChat(_ id: UUID) {
        if generatingConversation == id { stopGeneration() }
        conversations.removeAll { $0.id == id }
        if case .chat(let s) = selection, s == id {
            selection = conversations.first.map { .chat($0.id) } ?? .discover
        }
        scheduleSave()
    }

    func renameChat(_ id: UUID, _ title: String) {
        guard let i = index(id) else { return }
        conversations[i].title = title
        scheduleSave()
    }

    func clearAllChats() {
        stopGeneration()
        conversations.removeAll()
        selection = .discover
        scheduleSave()
    }

    func index(_ id: UUID) -> Int? { conversations.firstIndex { $0.id == id } }

    func send(_ text: String, in id: UUID) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating, let i = index(id) else { return }
        conversations[i].messages.append(StoredMessage(role: .user, content: trimmed))
        if conversations[i].title == "New Chat" {
            conversations[i].title = String(trimmed.split(separator: "\n").first ?? "").prefix(48).description
        }
        generate(in: id)
    }

    func regenerate(in id: UUID) {
        guard !isGenerating, let i = index(id) else { return }
        while let last = conversations[i].messages.last, last.role == .assistant {
            conversations[i].messages.removeLast()
        }
        guard conversations[i].messages.last?.role == .user else { return }
        generate(in: id)
    }

    func editAndResend(_ messageID: UUID, newText: String, in id: UUID) {
        guard !isGenerating, let i = index(id),
              let m = conversations[i].messages.firstIndex(where: { $0.id == messageID }) else { return }
        conversations[i].messages.removeSubrange(m...)
        send(newText, in: id)
    }

    func stopGeneration() {
        engine.cancel()
    }

    private func generate(in id: UUID) {
        guard let i = index(id) else { return }
        guard let info = loadedInfo else {
            conversations[i].messages.append(StoredMessage(role: .assistant, content: "",
                                                           error: "No model loaded. Pick a model from the bar at the top of the window."))
            return
        }
        var history: [ChatMessage] = []
        let sys = (conversations[i].systemPrompt ?? systemPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
        if !sys.isEmpty { history.append(ChatMessage(role: .system, content: sys)) }
        for m in conversations[i].messages where m.error == nil {
            history.append(ChatMessage(role: m.role, content: m.content))
        }
        let reply = StoredMessage(role: .assistant, content: "", model: info.name)
        conversations[i].messages.append(reply)
        conversations[i].updated = Date()
        generatingConversation = id
        let params = sampling
        let thinking = enableThinking
        let replyID = reply.id

        generationTask = Task {
            var raw = ""
            var pending = ""
            var lastFlush = Date()
            var thinkStart: Date?
            var thinkEnd: Date?
            var genStart: Date?
            var tokenCount = 0
            liveTokens = 0
            liveTokensPerSecond = 0
            @MainActor func flush(_ final: Bool = false) {
                raw += pending
                pending = ""
                lastFlush = Date()
                let parsed = ThinkParser.split(raw)
                if parsed.reasoning != nil && thinkStart == nil { thinkStart = Date() }
                if parsed.reasoningComplete && thinkEnd == nil, thinkStart != nil { thinkEnd = Date() }
                if let g = genStart {
                    liveTokens = tokenCount
                    liveTokensPerSecond = Double(tokenCount) / max(Date().timeIntervalSince(g), 0.001)
                }
                update(id, replyID) { m in
                    m.content = final ? parsed.content.trimmingCharacters(in: .whitespacesAndNewlines) : parsed.content
                    m.reasoning = parsed.reasoning
                    if let s = thinkStart { m.reasoningSeconds = (thinkEnd ?? Date()).timeIntervalSince(s) }
                }
            }
            do {
                for try await ev in engine.chat(history, parameters: params, enableThinking: thinking) {
                    switch ev {
                    case .promptProcessed(let s):
                        genStart = Date()
                        update(id, replyID) { $0.stats = s }
                    case .text(let t):
                        pending += t
                        tokenCount += 1
                        if Date().timeIntervalSince(lastFlush) > 0.033 { flush() }
                    case .finished(let s):
                        flush(true)
                        update(id, replyID) { $0.stats = s }
                    }
                }
            } catch {
                flush(true)
                update(id, replyID) { $0.error = error.localizedDescription }
            }
            generatingConversation = nil
            scheduleSave()
        }
    }

    private func update(_ conv: UUID, _ msg: UUID, _ body: (inout StoredMessage) -> Void) {
        guard let i = index(conv), let j = conversations[i].messages.firstIndex(where: { $0.id == msg }) else { return }
        body(&conversations[i].messages[j])
    }

    // MARK: Persistence

    func scheduleSave() {
        saveTask?.cancel()
        let snapshot = conversations
        let url = chatsURL
        saveTask = Task.detached(priority: .background) {
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled else { return }
            if let data = try? JSONEncoder().encode(snapshot) { try? data.write(to: url, options: .atomic) }
        }
    }

    private func loadChats() {
        guard let data = try? Data(contentsOf: chatsURL),
              let c = try? JSONDecoder().decode([Conversation].self, from: data) else { return }
        conversations = c.sorted { $0.updated > $1.updated }
    }

    private func loadSettings() {
        let d = UserDefaults.standard
        if let data = d.data(forKey: "sampling"), let s = try? JSONDecoder().decode(SamplingParameters.self, from: data) { sampling = s }
        systemPrompt = d.string(forKey: "systemPrompt") ?? ""
        contextLength = d.object(forKey: "contextLength") as? Int ?? 8192
        enableThinking = d.object(forKey: "enableThinking") as? Bool ?? true
        lastModelPath = d.string(forKey: "lastModelPath")
    }

    private func saveSettings() {
        let d = UserDefaults.standard
        if let data = try? JSONEncoder().encode(sampling) { d.set(data, forKey: "sampling") }
        d.set(systemPrompt, forKey: "systemPrompt")
        d.set(contextLength, forKey: "contextLength")
        d.set(enableThinking, forKey: "enableThinking")
        d.set(lastModelPath, forKey: "lastModelPath")
    }
}

/// Splits `<think>…</think>` reasoning from the visible answer while streaming.
enum ThinkParser {
    static func split(_ raw: String) -> (reasoning: String?, content: String, reasoningComplete: Bool) {
        let s = raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? raw : raw
        guard let open = s.range(of: "<think>") else {
            // Some models omit the opening tag but emit a closing one.
            if let close = s.range(of: "</think>") {
                return (String(s[..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines),
                        String(s[close.upperBound...]), true)
            }
            return (nil, s, false)
        }
        let afterOpen = s[open.upperBound...]
        if let close = afterOpen.range(of: "</think>") {
            let reasoning = String(afterOpen[..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoning.isEmpty ? nil : reasoning, String(afterOpen[close.upperBound...]), true)
        }
        return (String(afterOpen).trimmingCharacters(in: .whitespacesAndNewlines), "", false)
    }
}

// MARK: - Formatting helpers

enum Fmt {
    static func bytes(_ b: Int) -> String { ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file) }
    static func bytes(_ b: Int64) -> String { ByteCountFormatter.string(fromByteCount: b, countStyle: .file) }
    static func count(_ n: Int?) -> String {
        guard let n else { return "–" }
        switch n {
        case 1_000_000...: return String(format: "%.1fM", Double(n) / 1_000_000)
        case 1_000...: return String(format: "%.1fK", Double(n) / 1_000)
        default: return "\(n)"
        }
    }
    static var physicalMemory: Int { Int(ProcessInfo.processInfo.physicalMemory) }
}

/// Thread-safe rate limiter for progress callbacks.
final class Throttle: @unchecked Sendable {
    private let interval: TimeInterval
    private var last = Date.distantPast
    private let lock = NSLock()
    init(interval: TimeInterval) { self.interval = interval }
    func ready() -> Bool {
        lock.lock(); defer { lock.unlock() }
        let now = Date()
        guard now.timeIntervalSince(last) >= interval else { return false }
        last = now
        return true
    }
}
