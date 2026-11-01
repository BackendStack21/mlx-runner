import AppKit
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

    enum CodingKeys: String, CodingKey {
        case id, role, content, reasoning, stats, model, date, error, reasoningSeconds
    }
}

struct Conversation: Codable, Identifiable, Hashable {
    var id = UUID()
    var title: String = "New Chat"
    var messages: [StoredMessage] = []
    var created = Date()
    var updated = Date()
    var systemPrompt: String?

    enum CodingKeys: String, CodingKey {
        case id, title, messages, created, updated, systemPrompt
    }
}

// Tolerant decoding (in extensions so the memberwise initializers stay): missing or
// reshaped fields fall back to defaults instead of failing the whole chats.json.
extension StoredMessage {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decodeIfPresent(UUID.self, forKey: .id)) ?? UUID()
        role = (try? c.decodeIfPresent(ChatMessage.Role.self, forKey: .role)) ?? .assistant
        content = (try? c.decodeIfPresent(String.self, forKey: .content)) ?? ""
        reasoning = try? c.decodeIfPresent(String.self, forKey: .reasoning)
        stats = try? c.decodeIfPresent(GenerationStats.self, forKey: .stats)
        model = try? c.decodeIfPresent(String.self, forKey: .model)
        date = (try? c.decodeIfPresent(Date.self, forKey: .date)) ?? Date()
        error = try? c.decodeIfPresent(String.self, forKey: .error)
        reasoningSeconds = try? c.decodeIfPresent(Double.self, forKey: .reasoningSeconds)
    }
}

extension Conversation {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decodeIfPresent(UUID.self, forKey: .id)) ?? UUID()
        title = (try? c.decodeIfPresent(String.self, forKey: .title)) ?? "New Chat"
        messages = (try? c.decodeIfPresent([StoredMessage].self, forKey: .messages)) ?? []
        created = (try? c.decodeIfPresent(Date.self, forKey: .created)) ?? Date()
        updated = (try? c.decodeIfPresent(Date.self, forKey: .updated)) ?? created
        systemPrompt = try? c.decodeIfPresent(String.self, forKey: .systemPrompt)
    }
}

/// Serializes chats.json writes; a stale snapshot never overwrites a newer one.
final class ChatWriter: @unchecked Sendable {
    let queue = DispatchQueue(label: "de.21no.mlxrunner.chats", qos: .utility)
    private var lastSeq = 0  // only touched on `queue`

    func write(_ snapshot: [Conversation], seq: Int, to url: URL) {
        guard seq > lastSeq, let data = try? JSONEncoder().encode(snapshot) else { return }
        lastSeq = seq
        try? data.write(to: url, options: .atomic)
    }
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
    /// Non-fatal persistence problem to show the user (e.g. an unreadable chats.json).
    var persistenceError: String?

    // Settings (persisted in UserDefaults)
    var sampling = SamplingParameters() { didSet { saveSettings() } }
    var systemPrompt: String = "" { didSet { saveSettings() } }
    var contextLength: Int = 8192 { didSet { saveSettings() } }
    var enableThinking: Bool = true { didSet { saveSettings() } }
    var lastModelPath: String? { didSet { saveSettings() } }

    @ObservationIgnored private var activeDownloads: [String: ModelDownload] = [:]
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var generationToken: CancelToken?
    @ObservationIgnored private var saveTask: Task<Void, Never>?
    @ObservationIgnored private var saveSeq = 0
    /// Set when an unreadable chats.json could not be moved aside: never overwrite it.
    @ObservationIgnored private var chatsSaveBlocked = false
    @ObservationIgnored private let chatWriter = ChatWriter()
    @ObservationIgnored private var terminateObserver: NSObjectProtocol?
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
        // Flush pending chat edits synchronously before the process exits.
        terminateObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.saveNow(synchronously: true) }
        }
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
        let destination: URL
        do { destination = try ModelStore.directory(for: details.id) } catch {
            downloads[details.id] = DownloadState(error: error.localizedDescription)
            return
        }
        let dl = ModelDownload(details: details, destination: destination)
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
        // Persist the user's message right away, not only when generation finishes.
        saveNow()
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
        generationToken?.cancel()
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
        let token = CancelToken()
        generationToken = token

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
                for try await ev in engine.chat(history, parameters: params, enableThinking: thinking,
                                                 conversationID: id.uuidString, token: token) {
                    switch ev {
                    case .promptProcessed(let s):
                        genStart = Date()
                        update(id, replyID) { $0.stats = s }
                    case .text(let t):
                        pending += t
                        tokenCount += 1
                        if Date().timeIntervalSince(lastFlush) > 0.05 { flush() }   // 20 updates/s
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

    /// Debounced save (coalesces bursts of edits).
    func scheduleSave() {
        guard !DevSnapshot.isActive, !chatsSaveBlocked else { return }
        saveTask?.cancel()
        saveSeq += 1
        let seq = saveSeq
        let snapshot = conversations
        let url = chatsURL
        let writer = chatWriter
        saveTask = Task.detached(priority: .background) {
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled else { return }
            writer.queue.async { writer.write(snapshot, seq: seq, to: url) }
        }
    }

    /// Writes chats now. `synchronously` blocks until the file is on disk (used at termination).
    func saveNow(synchronously: Bool = false) {
        guard !DevSnapshot.isActive, !chatsSaveBlocked else { return }
        saveTask?.cancel()
        saveTask = nil
        saveSeq += 1
        let seq = saveSeq
        let snapshot = conversations
        let url = chatsURL
        let writer = chatWriter
        if synchronously {
            writer.queue.sync { writer.write(snapshot, seq: seq, to: url) }
        } else {
            writer.queue.async { writer.write(snapshot, seq: seq, to: url) }
        }
    }

    private func loadChats() {
        if DevSnapshot.isActive {
            // Snapshot and tour runs never read or touch the user's chats: they show sample ones.
            conversations = DevSnapshot.sampleConversations()
            return
        }
        switch SafeJSONFile.load([Conversation].self, from: chatsURL) {
        case .missing:
            return
        case .loaded(let c):
            conversations = c.sorted { $0.updated > $1.updated }
        case .corrupt(let error, let movedTo):
            if let movedTo {
                persistenceError = "Your chat history could not be read (\(error.localizedDescription)). "
                    + "It was kept as \(movedTo.lastPathComponent) in \(movedTo.deletingLastPathComponent().path), and a new history was started."
            } else {
                chatsSaveBlocked = true
                persistenceError = "Your chat history could not be read (\(error.localizedDescription)) and could not be moved aside. "
                    + "Chats will not be saved this session so \(chatsURL.path) is not overwritten."
            }
        }
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
        guard !DevSnapshot.isActive else { return }
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
