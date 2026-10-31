import MLXEngine
import SwiftUI

struct SidebarView: View {
    @Environment(AppState.self) private var state
    @State private var renaming: UUID?
    @State private var renameText = ""
    @State private var search = ""

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

            Button {
                state.newChat()
            } label: {
                Label("New Chat", systemImage: "square.and.pencil")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .foregroundStyle(.white)
                    .background(Theme.gradient, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .shadow(color: Theme.accent.opacity(0.35), radius: 8, y: 3)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.bottom, 8)

            List(selection: $state.selection) {
                Section {
                    navRow("Discover", "safari", .discover)
                    navRow("My Models", "shippingbox", .models, badge: state.localModels.count)
                    navRow("Local Server", "network", .server)
                }
                Section("Chats") {
                    ForEach(filteredChats) { c in
                        chatRow(c).tag(SidebarItem.chat(c.id))
                    }
                }
            }
            .listStyle(.sidebar)
            .searchable(text: $search, placement: .sidebar, prompt: "Search chats")

            ModelStatusFooter()
        }
    }

    private var filteredChats: [Conversation] {
        let q = search.trimmingCharacters(in: .whitespaces).lowercased()
        let sorted = state.conversations.sorted { $0.updated > $1.updated }
        guard !q.isEmpty else { return sorted }
        return sorted.filter { c in
            c.title.lowercased().contains(q) || c.messages.contains { $0.content.lowercased().contains(q) }
        }
    }

    private func navRow(_ title: String, _ icon: String, _ item: SidebarItem, badge: Int = 0) -> some View {
        Label {
            HStack {
                Text(title)
                Spacer()
                if badge > 0 {
                    Text("\(badge)").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
        } icon: {
            Image(systemName: icon).foregroundStyle(Theme.accent)
        }
        .tag(item)
    }

    @ViewBuilder
    private func chatRow(_ c: Conversation) -> some View {
        if renaming == c.id {
            TextField("Title", text: $renameText, onCommit: {
                state.renameChat(c.id, renameText)
                renaming = nil
            })
            .textFieldStyle(.plain)
        } else {
            HStack(spacing: 8) {
                if state.generatingConversation == c.id {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: "bubble.left").foregroundStyle(.secondary).font(.system(size: 11))
                }
                VStack(alignment: .leading, spacing: 1) {
                    Text(c.title).lineLimit(1)
                    Text(c.updated, style: .relative).font(.system(size: 10)).foregroundStyle(.tertiary)
                }
            }
            .contextMenu {
                Button("Rename") { renameText = c.title; renaming = c.id }
                Button("Duplicate") {
                    var copy = c
                    copy.id = UUID()
                    copy.title += " (copy)"
                    copy.updated = Date()
                    state.conversations.insert(copy, at: 0)
                    state.scheduleSave()
                }
                Divider()
                Button("Delete", role: .destructive) { state.deleteChat(c.id) }
            }
        }
    }
}

/// Bottom-of-sidebar status: loaded model and memory.
struct ModelStatusFooter: View {
    @Environment(AppState.self) private var state

    var body: some View {
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
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(cornerRadius: 12)
        .padding(10)
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
