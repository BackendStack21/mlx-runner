import AppKit
import MLXEngine
import SwiftUI

struct SidebarView: View {
    @Environment(AppState.self) private var state
    @Environment(\.undoManager) private var undoManager
    @State private var renaming: UUID?
    @State private var renameText = ""
    @FocusState private var renameFocused: Bool
    @State private var search = ""
    /// Start of today; moved on at midnight (`NSCalendarDayChanged`) so the day sections stay right.
    @State private var today = Calendar.current.startOfDay(for: Date())

    var body: some View {
        @Bindable var state = state
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                GradientIcon(systemName: "sparkles", size: 30)
                VStack(alignment: .leading, spacing: 0) {
                    Text("MLX Runner").font(.system(size: 15, weight: .bold, design: .rounded))
                    Text("Native · Metal · MLX").font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 14)
            .padding(.top, 8)
            .padding(.bottom, 10)
            .accessibilityElement(children: .combine)

            Button {
                state.newChat()
            } label: {
                Label("New Chat", systemImage: "square.and.pencil")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .foregroundStyle(.white)
                    .background(Theme.fillGradient, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .shadow(color: Theme.accent.opacity(0.35), radius: 8, y: 3)
            }
            .buttonStyle(.plain)
            .help("New Chat (⌘N)")
            .padding(.horizontal, 12)
            .padding(.bottom, 8)

            List(selection: $state.selection) {
                Section {
                    navRow("Discover", "safari", .discover)
                    navRow("My Models", "shippingbox", .models) { ModelsRowStatus() }
                    navRow("Local Server", "network", .server) { ServerRowStatus() }
                }
                ForEach(ChatGroups.group(filteredChats, today: today), id: \.title) { group in
                    Section(group.title) {
                        ForEach(group.chats) { c in
                            chatRow(c).tag(SidebarItem.chat(c.id))
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .searchable(text: $search, placement: .sidebar, prompt: "Search chats")
            .onReceive(NotificationCenter.default.publisher(for: .NSCalendarDayChanged)) { _ in
                today = Calendar.current.startOfDay(for: Date())
            }

            ModelStatusFooter()
        }
    }

    private var filteredChats: [Conversation] {
        let q = search.trimmingCharacters(in: .whitespaces).lowercased()
        let sorted = state.sortedChats
        guard !q.isEmpty else { return sorted }
        return sorted.filter { c in
            c.title.lowercased().contains(q) || c.messages.contains { $0.content.lowercased().contains(q) }
        }
    }

    private func navRow<Trailing: View>(_ title: String, _ icon: String, _ item: SidebarItem,
                                        @ViewBuilder trailing: () -> Trailing = { EmptyView() }) -> some View {
        Label {
            HStack {
                Text(title)
                Spacer()
                trailing()
            }
        } icon: {
            Image(systemName: icon).foregroundStyle(Theme.accent)
        }
        .tag(item)
    }

    @ViewBuilder
    private func chatRow(_ c: Conversation) -> some View {
        if renaming == c.id {
            TextField("Title", text: $renameText)
                .textFieldStyle(.plain)
                .focused($renameFocused)
                .onSubmit { commitRename(c.id) }
                .onExitCommand { renaming = nil }
                // Like Finder: clicking elsewhere keeps the new name.
                .onChange(of: renameFocused) { _, focused in if !focused { commitRename(c.id) } }
                .onAppear { DispatchQueue.main.async { renameFocused = true } }
                .accessibilityLabel("Chat title")
        } else {
            HStack(spacing: 8) {
                if state.generatingConversation == c.id || state.pendingSend == c.id {
                    ProgressView().controlSize(.mini).accessibilityLabel("Generating")
                } else {
                    Image(systemName: "bubble.left").foregroundStyle(.secondary).font(.system(size: 11))
                        .accessibilityHidden(true)
                }
                Text(c.title).lineLimit(1)
            }
            .help("\(c.title)\nUpdated \(c.updated.formatted(date: .abbreviated, time: .shortened))")
            .contextMenu {
                Button("Rename") { startRename(c) }
                Button("Duplicate") {
                    var copy = c
                    copy.id = UUID()
                    copy.title += " (copy)"
                    copy.updated = Date()
                    state.conversations.insert(copy, at: 0)
                    state.scheduleSave()
                }
                Divider()
                Button("Delete", role: .destructive) { state.deleteChat(c.id, undoManager: undoManager) }
            }
            .accessibilityActions {
                Button("Rename") { startRename(c) }
                Button("Delete") { state.deleteChat(c.id, undoManager: undoManager) }
            }
        }
    }

    private func startRename(_ c: Conversation) {
        renameText = c.title
        renaming = c.id
    }

    private func commitRename(_ id: UUID) {
        guard renaming == id else { return }
        let t = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { state.renameChat(id, t) }   // an empty title keeps the old one
        renaming = nil
    }
}

/// Chats grouped by when they were last updated, newest first.
enum ChatGroups {
    struct Group { let title: String; let chats: [Conversation] }

    static func group(_ chats: [Conversation], today: Date, calendar: Calendar = .current) -> [Group] {
        var order: [String] = []
        var buckets: [String: [Conversation]] = [:]
        for c in chats {
            let t = title(for: c.updated, today: today, calendar: calendar)
            if buckets[t] == nil { order.append(t) }
            buckets[t, default: []].append(c)
        }
        return order.map { Group(title: $0, chats: buckets[$0] ?? []) }
    }

    static func title(for date: Date, today: Date, calendar: Calendar = .current) -> String {
        let day = calendar.startOfDay(for: date)
        let days = calendar.dateComponents([.day], from: day, to: today).day ?? 0
        switch days {
        case ...0: return "Today"
        case 1: return "Yesterday"
        case 2...7: return "Previous 7 Days"
        case 8...30: return "Previous 30 Days"
        default: return date.formatted(.dateTime.month(.wide).year())
        }
    }
}

/// "My Models" row status: the model count, or an active-download count while downloads run. Reads only
/// counts that change when a download starts or ends (not the 10 Hz progress), so the sidebar doesn't re-render per tick.
private struct ModelsRowStatus: View {
    @Environment(AppState.self) private var state

    var body: some View {
        if state.activeDownloadCount > 0 {
            Label("\(state.activeDownloadCount)", systemImage: "arrow.down.circle.fill")
                .labelStyle(CompactLabel())
                .font(.caption2.monospacedDigit())
                .foregroundStyle(Theme.accent)
                .accessibilityLabel(state.activeDownloadCount == 1 ? "1 download in progress" : "\(state.activeDownloadCount) downloads in progress")
        } else if !state.localModels.isEmpty {
            Text("\(state.localModels.count)").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                .accessibilityLabel("\(state.localModels.count) models")
        }
    }
}

/// "Local Server" row status when the server runs: its port, and whether it has a model to serve.
private struct ServerRowStatus: View {
    @Environment(AppState.self) private var state
    @State private var server = ServerController.shared

    var body: some View {
        if server.running {
            let ready = state.loadedInfo != nil
            HStack(spacing: 4) {
                Circle().fill(ready ? Theme.good : .orange).frame(width: 6, height: 6)
                Text(ready ? String(server.port) : "No model").font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(ready ? "Running on port \(server.port)" : "Running on port \(server.port), no model loaded")
            .help(ready ? "Serving on port \(server.port)" : "Running on port \(server.port): requests fail until a model is loaded")
        }
    }
}

/// Bottom-of-sidebar status: loaded model and memory. Clicking it opens the model picker.
struct ModelStatusFooter: View {
    @Environment(AppState.self) private var state

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            UpdatePill()
            Button { state.openModelPicker() } label: {
                VStack(alignment: .leading, spacing: 6) {
                    switch state.loadState {
                    case .loaded(let info):
                        HStack(spacing: 6) {
                            Circle().fill(Theme.good).frame(width: 7, height: 7)
                                .shadow(color: Theme.good, radius: 4)
                            Text(info.name).font(.system(size: 11, weight: .semibold)).lineLimit(1)
                        }
                        MemoryBar(used: info.parameterBytes, total: Fmt.physicalMemory)
                    case .loading(let name):
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.mini)
                            Text("Loading \(name)…").font(.system(size: 11)).lineLimit(1)
                        }
                    default:
                        HStack(spacing: 6) {
                            Circle().fill(.secondary.opacity(0.5)).frame(width: 7, height: 7)
                            Text("No model loaded").font(.system(size: 11)).foregroundStyle(.secondary)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Choose a model (⌘L)")
            .accessibilityLabel(footerLabel)
            .accessibilityHint("Choose a model to load")
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(cornerRadius: 12)
        .padding(10)
    }

    private var footerLabel: String {
        switch state.loadState {
        case .loaded(let info):
            return "Loaded: \(info.name), \(Fmt.bytes(info.parameterBytes)) of weights, \(Fmt.bytes(Fmt.physicalMemory)) of memory"
        case .loading(let name): return "Loading \(name)"
        default: return "No model loaded"
        }
    }
}

struct MemoryBar: View {
    let used: Int
    let total: Int
    var body: some View {
        let f = min(1, Double(used) / Double(max(total, 1)))
        VStack(alignment: .leading, spacing: 3) {
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.secondary.opacity(0.18))
                    Capsule().fill(f > 0.75 ? AnyShapeStyle(Theme.warmGradient) : AnyShapeStyle(Theme.gradient))
                        .frame(width: max(4, g.size.width * f))
                }
            }
            .frame(height: 5)
            Text("\(Fmt.bytes(used)) weights · \(Fmt.bytes(total)) RAM")
                .font(.system(size: 9.5, weight: .medium).monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }
}
