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

/// The load-time settings a model was loaded with. Compared with the current settings (not with
/// `LoadedModelInfo`, whose values the engine clamps or turns off per model) to tell when a reload is due.
struct LoadSettings: Equatable {
    var contextLength: Int
    var draftModel: String
    var sparse: SparseAttention?
    var sieve: SieveProfile?
}

/// A load that failed while another model stayed loaded (the engine checks before it unloads).
struct LoadFailure: Equatable {
    var model: String
    var message: String
}

/// UserDefaults keys for what happens when the app opens.
enum LaunchKeys {
    static let loadLastModel = "launchLoadLastModel"
    static let startServer = "launchStartServer"
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
    /// Unsent composer text, per chat.
    var drafts: [UUID: String] = [:]
    /// The model picker popover (lifted here so ⌘L and the sidebar footer can open it).
    var showModelPicker = false
    /// The user message being edited in place (the composer's ⌘⏎ is off meanwhile).
    var editingMessage: UUID?
    /// A chat whose reply waits for a model load to finish (send with no model loaded).
    var pendingSend: UUID?
    /// The settings the loaded model was loaded with (nil when none is loaded).
    var loadedWith: LoadSettings?
    /// A failed load that left the previous model loaded.
    var loadFailure: LoadFailure?
    /// The first scan of installed models has finished (until then "no models" is unknown, not true).
    var modelsScanned = false
    /// Downloads in progress; changes only when one starts or ends (the sidebar reads it, not `downloads`).
    var activeDownloadCount = 0
    var liveTokens = 0
    var liveTokensPerSecond: Double = 0
    /// The reply being generated. Streamed here, not into `conversations`: every view that
    /// reads `conversations` (sidebar, inspector with its AppKit sliders, the whole chat) would
    /// otherwise re-render on every token, which froze the window for long replies. Only the
    /// in-progress message row reads these; the final text is written to the conversation once.
    var liveReplyID: UUID?
    var liveContent = ""
    var liveReasoning: String?
    var liveReasoningSeconds: Double?
    /// Non-fatal persistence problem to show the user (e.g. an unreadable chats.json).
    var persistenceError: String?

    // Settings (persisted in UserDefaults)
    var sampling = SamplingParameters() { didSet { saveSettings() } }
    var systemPrompt: String = "" { didSet { saveSettings() } }
    var contextLength: Int = 8192 { didSet { saveSettings() } }
    /// Speculative decoding draft model: "" off, "auto", or a model directory path.
    var draftModel: String = "" { didSet { saveSettings() } }
    /// The Think toggle, per model (keyed by directory path). A model without a choice follows its
    /// template: Qwen3 thinks, Qwen 3.5 0.8B and 2B answer directly (thinking often loops on them).
    var thinkingChoices: [String: Bool] = [:] { didSet { saveSettings() } }
    /// Settings saved before choices were per model had thinking off for every model: still the fallback.
    private var thinkingOffEverywhere = false
    var enableThinking: Bool {
        get {
            guard let info = loadedInfo else { return false }
            return thinkingChoices[info.directory.path] ?? (!thinkingOffEverywhere && info.thinksByDefault)
        }
        set {
            guard let info = loadedInfo else { return }
            thinkingChoices[info.directory.path] = newValue
        }
    }
    /// Sparse decode attention for long contexts (on by default), its long-range key budget, and
    /// whether each KV head selects its own keys or every head shares one selection.
    var sparseAttention: Bool = true { didSet { saveSettings() } }
    var sparseKeys: Int = SparseAttention.standard.topK { didSet { saveSettings() } }
    var sparsePerHead: Bool = true { didSet { saveSettings() } }
    /// The sparse attention the next load uses (nil: dense decode).
    var sparseConfig: SparseAttention? {
        sparseAttention ? SparseAttention(topK: sparseKeys, perHead: sparsePerHead) : nil
    }
    /// Sieve (off by default): the output projection by rows, for models that have a fitted sidecar.
    /// Faster decode, changed output.
    var sieve: Bool = false { didSet { saveSettings() } }
    var sieveProfile: SieveProfile? { sieve ? .balanced : nil }
    /// Progress of a running Sieve fit (nil: none running), and the last fit's failure.
    var sieveFitStatus: String?
    var sieveFitError: String?
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

    var isLoading: Bool {
        if case .loading = loadState { return true }
        return false
    }

    /// The model loaded last time (any session), if it is still installed and runnable.
    var lastModel: LocalModel? {
        guard let p = lastModelPath else { return nil }
        return localModels.first { $0.directory.path == p && $0.isSupported }
    }

    /// A reply can start: a model is loaded, loading, or the last one can be loaded on send.
    var canStartReply: Bool { loadedInfo != nil || isLoading || lastModel != nil }

    /// Settings as the next load would use them.
    var currentLoadSettings: LoadSettings {
        LoadSettings(contextLength: contextLength, draftModel: draftModel, sparse: sparseConfig, sieve: sieveProfile)
    }

    /// Load-time settings changed since the loaded model was loaded.
    var needsReload: Bool {
        guard loadedInfo != nil, let used = loadedWith else { return false }
        return used != currentLoadSettings
    }

    /// One VoiceOver announcement (never per streamed token).
    func announce(_ text: String) {
        AccessibilityNotification.Announcement(text).post()
    }

    // MARK: Models

    func refreshModels() {
        Task.detached(priority: .userInitiated) {
            let models = ModelStore.scan()
            await MainActor.run {
                self.localModels = models
                // A draft pick whose model was removed falls back to none (the picker has no such row).
                if !["", "auto"].contains(self.draftModel), !models.contains(where: { $0.directory.path == self.draftModel }) { self.draftModel = "" }
                if !self.modelsScanned {
                    self.modelsScanned = true
                    self.runLaunchActions()
                }
            }
        }
    }

    /// "When MLX Runner opens" settings: load the last model, start the local server. Both off by default.
    private func runLaunchActions() {
        guard !DevSnapshot.isActive else { return }
        let d = UserDefaults.standard
        if d.bool(forKey: LaunchKeys.loadLastModel) { reloadLast() }
        if d.bool(forKey: LaunchKeys.startServer) { ServerController.shared.start() }
    }

    /// Reloads the loaded model with the current settings (context length, draft, sparse attention).
    func reloadModel() {
        guard let dir = loadedInfo?.directory, let m = localModels.first(where: { $0.directory == dir }) else { return }
        load(m)
    }

    func load(_ model: LocalModel) {
        guard model.isSupported else {
            let message = "\(model.modelType ?? "This") architecture is not supported yet."
            loadState = .failed(message)
            if let id = pendingSend {
                pendingSend = nil
                if let i = index(id) {
                    conversations[i].messages.append(StoredMessage(role: .assistant, content: "", error: message))
                    scheduleSave()
                }
            }
            return
        }
        stopGeneration()
        loadFailure = nil
        loadState = .loading(model.name)
        let ctx = contextLength
        let draftSetting = draftModel
        let sparse = sparseConfig
        let sieve = sieveProfile
        let settings = currentLoadSettings
        let models = localModels
        Task {
            do {
                // Resolved off the main actor (tokenizer hashing). A saved pick that no longer fits the
                // model being loaded (another family, removed, or a target with a recurrent state, which
                // cannot drop rejected drafts) is dropped rather than failing the load.
                let draft: URL? = await Task.detached(priority: .userInitiated) {
                    switch draftSetting {
                    case "": return nil
                    case "auto": return InferenceEngine.autoDraft(for: model.directory, among: models)?.directory
                    default:
                        guard let d = models.first(where: { $0.directory.path == draftSetting }), d.isSupported, d.directory != model.directory,
                              d.modelType == model.modelType, !SupportedArchitectures.hasRecurrentState(model.modelType),
                              let fp = InferenceEngine.tokenizerFingerprint(model.directory), InferenceEngine.tokenizerFingerprint(d.directory) == fp
                        else { return nil }
                        return d.directory
                    }
                }.value
                let info = try await engine.load(directory: model.directory, draftDirectory: draft, contextLength: ctx, sparse: sparse, sieve: sieve)
                loadState = .loaded(info)
                loadedWith = settings
                lastModelPath = model.directory.path
                announce("\(info.name) loaded")
                if let id = pendingSend {
                    pendingSend = nil
                    generate(in: id)
                }
            } catch {
                let message = error.localizedDescription
                if let previous = engine.info {
                    // The engine checks a model before unloading the current one: that one is still loaded.
                    loadState = .loaded(previous)
                    loadFailure = LoadFailure(model: model.name, message: message)
                } else {
                    loadState = .failed(message)
                    loadedWith = nil
                }
                announce("Couldn't load \(model.name)")
                if let id = pendingSend {
                    pendingSend = nil
                    if let i = index(id) {
                        conversations[i].messages.append(StoredMessage(role: .assistant, content: "",
                                                                       error: "Couldn't load \(model.name): \(message)"))
                        scheduleSave()
                    }
                }
            }
        }
    }

    /// Fits the loaded model's Sieve sidecar (a minute or more with the GPU to itself: the model is
    /// unloaded for it), turns Sieve on and loads the model again.
    func fitSieve() {
        guard sieveFitStatus == nil, let dir = loadedInfo?.directory,
              let model = localModels.first(where: { $0.directory == dir }) else { return }
        stopGeneration()
        sieveFitError = nil
        sieveFitStatus = "Starting"
        loadState = .loading(model.name)
        loadedWith = nil
        Task {
            do {
                let report = try await engine.fitSieve(directory: dir) { line in
                    Task { @MainActor in if self.sieveFitStatus != nil { self.sieveFitStatus = line } }
                }
                sieveFitStatus = nil
                // A fit that misses its quality target is kept on disk but not switched on for the user.
                if report.info.meetsTarget {
                    sieve = true
                    announce("Sieve fitted for \(model.name)")
                } else {
                    sieve = false
                    sieveFitError = String(format: "The fit for %@ missed its quality target (likeliest token kept in %.1f%% of test states, %.1f%% of sampled tokens changed), so Sieve was left off. Turning it on will visibly change this model's output.",
                                           model.name, 100 * report.info.agreement, 100 * report.info.sampledChange)
                    announce("Sieve fit for \(model.name) missed its quality target")
                }
            } catch {
                sieveFitStatus = nil
                sieveFitError = "Couldn't fit \(model.name): \(error.localizedDescription)"
                announce("Couldn't fit Sieve for \(model.name)")
            }
            // Load it again, unless the user loaded or ejected something else while it was fitting.
            guard loadState == .loading(model.name), loadedWith == nil else { return }
            loadState = .idle
            load(model)
        }
    }

    func eject() {
        stopGeneration()
        engine.unload()
        loadState = .idle
        loadedWith = nil
        loadFailure = nil
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
        activeDownloadCount = activeDownloads.count
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
                announce("Downloaded \(repo.split(separator: "/").last.map(String.init) ?? repo)")
                // Bounces the Dock icon once when the app is in the background; cleared on activation.
                if !NSApp.isActive { NSApp.requestUserAttention(.informationalRequest) }
            } catch is CancellationError {
                downloads[repo] = nil
            } catch {
                downloads[repo]?.error = error.localizedDescription
            }
            activeDownloads[repo] = nil
            activeDownloadCount = activeDownloads.count
        }
    }

    func cancelDownload(_ repo: String) {
        activeDownloads[repo]?.cancel()
        activeDownloads[repo] = nil
        activeDownloadCount = activeDownloads.count
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

    /// Chats in sidebar order (most recently updated first).
    var sortedChats: [Conversation] { conversations.sorted { $0.updated > $1.updated } }

    /// Deletes a chat. With an undo manager, Edit ▸ Undo Delete Chat puts it back where it was.
    func deleteChat(_ id: UUID, undoManager: UndoManager? = nil) {
        guard let i = index(id) else { return }
        if generatingConversation == id { stopGeneration() }
        if pendingSend == id { pendingSend = nil }
        let order = sortedChats
        let removed = conversations.remove(at: i)
        drafts[id] = nil
        if case .chat(let s) = selection, s == id {
            // The next chat down the list (or the one above when it was the last), as Mail and Notes do.
            let rest = order.filter { $0.id != id }
            let at = order.firstIndex { $0.id == id } ?? 0
            selection = rest.isEmpty ? .discover : .chat(rest[min(at, rest.count - 1)].id)
        }
        if let undoManager {
            undoManager.registerUndo(withTarget: self) { s in
                MainActor.assumeIsolated { s.restoreChat(removed, at: i, undoManager: undoManager) }
            }
            undoManager.setActionName("Delete Chat")
        }
        announce("Deleted \(removed.title)")
        scheduleSave()
    }

    private func restoreChat(_ c: Conversation, at i: Int, undoManager: UndoManager) {
        guard index(c.id) == nil else { return }
        conversations.insert(c, at: min(i, conversations.count))
        selection = .chat(c.id)
        undoManager.registerUndo(withTarget: self) { s in
            MainActor.assumeIsolated { s.deleteChat(c.id, undoManager: undoManager) }
        }
        undoManager.setActionName("Delete Chat")
        scheduleSave()
    }

    /// Selects the next (`offset` 1) or previous (-1) chat in sidebar order.
    func selectAdjacentChat(_ offset: Int) {
        let order = sortedChats
        guard !order.isEmpty else { return }
        guard case .chat(let id) = selection, let i = order.firstIndex(where: { $0.id == id }) else {
            selection = .chat(order[offset > 0 ? 0 : order.count - 1].id)
            return
        }
        let j = i + offset
        guard order.indices.contains(j) else { return }
        selection = .chat(order[j].id)
    }

    var selectedChatID: UUID? {
        if case .chat(let id) = selection { return id }
        return nil
    }

    func copyLastReply() {
        guard let id = selectedChatID, let i = index(id),
              let m = conversations[i].messages.last(where: { $0.role == .assistant && !$0.content.isEmpty }) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(m.content, forType: .string)
        announce("Copied")
    }

    /// Opens the model picker, switching to a chat first when another screen is showing.
    func openModelPicker() {
        if selectedChatID == nil {
            if let recent = sortedChats.first { selection = .chat(recent.id) } else { newChat() }
        }
        showModelPicker = true
    }

    func renameChat(_ id: UUID, _ title: String) {
        guard let i = index(id) else { return }
        conversations[i].title = title
        scheduleSave()
    }

    func clearAllChats() {
        stopGeneration()
        conversations.removeAll()
        drafts.removeAll()
        pendingSend = nil
        selection = .discover
        scheduleSave()
    }

    func index(_ id: UUID) -> Int? { conversations.firstIndex { $0.id == id } }

    func send(_ text: String, in id: UUID) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating, pendingSend == nil, let i = index(id) else { return }
        conversations[i].messages.append(StoredMessage(role: .user, content: trimmed))
        if conversations[i].title == "New Chat" {
            conversations[i].title = String(trimmed.split(separator: "\n").first ?? "").prefix(48).description
        }
        startReply(in: id)
        // Persist the user's message right away, not only when generation finishes.
        saveNow()
    }

    func regenerate(in id: UUID) {
        guard !isGenerating, pendingSend == nil, let i = index(id) else { return }
        while let last = conversations[i].messages.last, last.role == .assistant {
            conversations[i].messages.removeLast()
        }
        guard conversations[i].messages.last?.role == .user else { return }
        startReply(in: id)
    }

    /// Generates now, or once a model has loaded: the one loading, else the last one used.
    private func startReply(in id: UUID) {
        if loadedInfo == nil {
            if isLoading {
                pendingSend = id
                return
            }
            if let m = lastModel {
                pendingSend = id
                load(m)
                return
            }
        }
        generate(in: id)
    }

    func editAndResend(_ messageID: UUID, newText: String, in id: UUID) {
        guard !isGenerating, pendingSend == nil, let i = index(id),
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
            // What the previous update cost the main thread (state change through layout and
            // display): the next one waits twice that, so the UI keeps most of its time.
            var lastCost: TimeInterval = 0
            var thinkStart: Date?
            var thinkEnd: Date?
            var genStart: Date?
            var tokenCount = 0
            liveTokens = 0
            liveTokensPerSecond = 0
            liveReplyID = replyID
            liveContent = ""
            liveReasoning = nil
            liveReasoningSeconds = nil
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
                if final {
                    update(id, replyID) { m in
                        m.content = parsed.content.trimmingCharacters(in: .whitespacesAndNewlines)
                        m.reasoning = parsed.reasoning
                        if let s = thinkStart { m.reasoningSeconds = (thinkEnd ?? Date()).timeIntervalSince(s) }
                    }
                    liveReplyID = nil
                } else {
                    liveContent = parsed.content
                    liveReasoning = parsed.reasoning
                    if let s = thinkStart { liveReasoningSeconds = (thinkEnd ?? Date()).timeIntervalSince(s) }
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
                        // Each update re-lays out and redraws the chat (tens of ms for a long reply),
                        // so updates are paced by their measured cost: at most 20 a second, and
                        // never more often than twice what the last one took.
                        let interval = min(0.5, max(0.05, lastCost * 2))
                        if Date().timeIntervalSince(lastFlush) > interval {
                            let started = Date()
                            flush()
                            // Runs on the next run-loop turn, after this update's layout and display.
                            DispatchQueue.main.async { lastCost = Date().timeIntervalSince(started) }
                        }
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
            announce("Reply finished")
        }
    }

    /// Sampling settings from the loaded model's generation_config.json over the engine defaults (the reply
    /// limit is kept: it is not a sampling setting).
    func resetSamplingToModelDefaults() {
        var p = SamplingParameters()
        if let dir = loadedInfo?.directory {
            let d = GenerationDefaults.load(directory: dir)
            if let v = d.temperature { p.temperature = v }
            if let v = d.topP { p.topP = v }
            if let v = d.topK { p.topK = v }
            if let v = d.minP { p.minP = v }
            if let v = d.repetitionPenalty { p.repetitionPenalty = v }
        }
        p.maxTokens = sampling.maxTokens
        sampling = p
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

    /// Set while `loadSettings` runs: each assignment there fires `didSet`, and saving then would write
    /// the defaults of the settings not read yet over their stored values.
    @ObservationIgnored private var loadingSettings = false

    private func loadSettings() {
        loadingSettings = true
        defer { loadingSettings = false }
        let d = UserDefaults.standard
        if let data = d.data(forKey: "sampling"), let s = try? JSONDecoder().decode(SamplingParameters.self, from: data) { sampling = s }
        systemPrompt = d.string(forKey: "systemPrompt") ?? ""
        contextLength = d.object(forKey: "contextLength") as? Int ?? 8192
        draftModel = d.string(forKey: "draftModel") ?? ""
        thinkingChoices = d.dictionary(forKey: "thinkingChoices") as? [String: Bool] ?? [:]
        thinkingOffEverywhere = d.object(forKey: "enableThinking") as? Bool == false
        sparseAttention = d.object(forKey: "sparseAttention") as? Bool ?? true
        sparseKeys = d.object(forKey: "sparseKeys") as? Int ?? SparseAttention.standard.topK
        sparsePerHead = d.object(forKey: "sparsePerHead") as? Bool ?? true
        sieve = d.object(forKey: "sieve") as? Bool ?? false
        lastModelPath = d.string(forKey: "lastModelPath")
    }

    private func saveSettings() {
        guard !DevSnapshot.isActive, !loadingSettings else { return }
        let d = UserDefaults.standard
        if let data = try? JSONEncoder().encode(sampling) { d.set(data, forKey: "sampling") }
        d.set(systemPrompt, forKey: "systemPrompt")
        d.set(contextLength, forKey: "contextLength")
        d.set(draftModel, forKey: "draftModel")
        d.set(thinkingChoices, forKey: "thinkingChoices")
        d.set(sparseAttention, forKey: "sparseAttention")
        d.set(sparseKeys, forKey: "sparseKeys")
        d.set(sparsePerHead, forKey: "sparsePerHead")
        d.set(sieve, forKey: "sieve")
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
