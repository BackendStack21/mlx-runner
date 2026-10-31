import MLXEngine
import SwiftUI

struct DiscoverView: View {
    @Environment(AppState.self) private var state
    @State private var query = ""
    @State private var sort: HubSort = .downloads
    @State private var results: [HubModel] = []
    @State private var selected: HubModel.ID?
    @State private var loading = false
    @State private var error: String?
    @State private var supportedOnly = true
    @State private var searchTask: Task<Void, Never>?

    var body: some View {
        HSplitView {
            VStack(spacing: 0) {
                header
                if query.isEmpty { RecommendedStrip(selected: $selected) }
                Divider()
                list
            }
            .frame(minWidth: 380, idealWidth: 440)

            Group {
                if let id = selected {
                    ModelDetailView(repo: id).id(id)
                } else {
                    ZStack {
                        AuroraBackground()
                        VStack(spacing: 12) {
                            GradientIcon(systemName: "safari", size: 60)
                            Text("Discover MLX models").font(.system(size: 24, weight: .bold, design: .rounded))
                            Text("Thousands of models converted for Apple Silicon on Hugging Face.\nPick one to see details and download it.")
                                .multilineTextAlignment(.center).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle("Discover")
        .task { await search() }
    }

    private var header: some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search Hugging Face (e.g. qwen3, llama 3.2, mistral)", text: $query)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .onSubmit { Task { await search() } }
                    .onChange(of: query) { _, _ in debounce() }
                if loading { ProgressView().controlSize(.small) }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .glassCard(cornerRadius: 12)
            HStack {
                Picker("Sort", selection: $sort) {
                    ForEach(HubSort.allCases, id: \.self) { Text($0.title).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
                .onChange(of: sort) { _, _ in Task { await search() } }
                Toggle("Runnable only", isOn: $supportedOnly)
                    .toggleStyle(.switch).controlSize(.mini)
                Spacer()
                Text("\(visible.count) models").font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12)
    }

    private var visible: [HubModel] {
        supportedOnly ? results.filter(\.isRunnable) : results
    }

    @ViewBuilder
    private var list: some View {
        if let error {
            ContentUnavailableView("Couldn't reach Hugging Face", systemImage: "wifi.exclamationmark", description: Text(error))
        } else if visible.isEmpty && !loading {
            ContentUnavailableView.search(text: query)
        } else {
            List(visible, selection: $selected) { m in
                HubRow(model: m).tag(m.id)
            }
            .listStyle(.inset)
            .scrollContentBackground(.hidden)
        }
    }

    private func debounce() {
        searchTask?.cancel()
        searchTask = Task {
            try? await Task.sleep(nanoseconds: 350_000_000)
            guard !Task.isCancelled else { return }
            await search()
        }
    }

    private func search() async {
        loading = true
        defer { loading = false }
        do {
            results = try await HubClient.search(query, sort: sort, limit: 100)
            error = nil
            if selected == nil {
                selected = query.isEmpty ? (RecommendedModel.forThisMac.first?.id ?? visible.first?.id) : visible.first?.id
            }
        } catch is CancellationError {
        } catch let e as URLError where e.code == .cancelled {
        } catch {
            self.error = error.localizedDescription
        }
    }
}

struct HubRow: View {
    @Environment(AppState.self) private var state
    let model: HubModel

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(LinearGradient(colors: [color.opacity(0.9), color.opacity(0.5)], startPoint: .topLeading, endPoint: .bottomTrailing))
                Text(String(model.name.prefix(1)).uppercased())
                    .font(.system(size: 14, weight: .bold, design: .rounded)).foregroundStyle(.white)
            }
            .frame(width: 32, height: 32)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(model.name).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                    if ModelStore.isInstalled(model.id) {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.good).font(.system(size: 11))
                    }
                    if state.downloads[model.id] != nil {
                        ProgressView().controlSize(.mini)
                    }
                }
                HStack(spacing: 8) {
                    Text(model.author).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    Label(Fmt.count(model.downloads), systemImage: "arrow.down.circle")
                    Label(Fmt.count(model.likes), systemImage: "heart")
                }
                .font(.system(size: 10.5))
                .foregroundStyle(.secondary)
                .labelStyle(CompactLabel())
                .lineLimit(1)
            }
            Spacer()
            if let q = model.quantizationHint { Badge(text: q, color: Theme.accent) }
            if let t = model.modelType { Badge(text: t, color: model.isRunnable ? Theme.accent2 : .orange) }
        }
        .padding(.vertical, 4)
    }

    private var color: Color {
        let colors: [Color] = [Theme.accent, Theme.accent2, Theme.pink, .orange, .teal, .indigo]
        return colors[abs(model.author.hashValue) % colors.count]
    }
}

struct CompactLabel: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 2) { configuration.icon; configuration.title }
    }
}

struct ModelDetailView: View {
    @Environment(AppState.self) private var state
    let repo: String
    @State private var details: HubModelDetails?
    @State private var readme: String?
    @State private var error: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(repo.split(separator: "/").first.map(String.init) ?? "")
                        .font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary)
                    Text(repo.split(separator: "/").last.map(String.init) ?? repo)
                        .font(.system(size: 26, weight: .bold, design: .rounded))
                        .textSelection(.enabled)
                }
                if let d = details {
                    HStack(spacing: 6) {
                        if let t = d.modelType { Badge(text: t, color: d.isSupported ? Theme.accent2 : .orange, systemImage: "cpu") }
                        if let b = d.bits { Badge(text: "\(b)-bit", color: Theme.accent, systemImage: "square.stack.3d.down.right") }
                        Badge(text: Fmt.bytes(d.downloadSize), color: .secondary, systemImage: "internaldrive")
                        if let l = d.license { Badge(text: l, color: .secondary, systemImage: "doc.text") }
                        Badge(text: Fmt.count(d.downloads), color: .secondary, systemImage: "arrow.down.circle")
                        Badge(text: Fmt.count(d.likes), color: .secondary, systemImage: "heart")
                    }
                    actionCard(d)
                } else if let error {
                    Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                } else {
                    ProgressView().frame(maxWidth: .infinity)
                }
                if let readme {
                    Divider()
                    MarkdownView(text: readme, baseSize: 13)
                }
            }
            .padding(28)
            .frame(maxWidth: 760, alignment: .leading)
        }
        .task {
            do {
                details = try await HubClient.details(repo)
            } catch { self.error = error.localizedDescription }
            if let r = await HubClient.readme(repo) { readme = Self.stripFrontMatter(r) }
        }
    }

    @ViewBuilder
    private func actionCard(_ d: HubModelDetails) -> some View {
        let installed = ModelStore.isInstalled(d.id) || state.localModels.contains { $0.repo == d.id }
        let ram = Fmt.physicalMemory
        let fit = Double(d.downloadSize) / Double(ram)
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: fit < 0.5 ? "checkmark.seal.fill" : (fit < 0.75 ? "exclamationmark.circle.fill" : "xmark.octagon.fill"))
                    .foregroundStyle(fit < 0.5 ? Theme.good : (fit < 0.75 ? .yellow : .red))
                VStack(alignment: .leading, spacing: 2) {
                    Text(fit < 0.5 ? "Fits comfortably in memory" : (fit < 0.75 ? "Tight fit — close other apps" : "Likely too large for this Mac"))
                        .font(.system(size: 13, weight: .semibold))
                    Text("\(Fmt.bytes(d.downloadSize)) of weights · \(Fmt.bytes(ram)) unified memory")
                        .font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }
            if !d.isSupported {
                Label("The \(d.modelType ?? "unknown") architecture isn't supported by the built-in engine yet. Supported: \(SupportedArchitectures.modelTypes.sorted().joined(separator: ", ")).",
                      systemImage: "info.circle")
                    .font(.system(size: 12)).foregroundStyle(.orange)
            }
            if let dl = state.downloads[d.id] {
                VStack(alignment: .leading, spacing: 6) {
                    if let err = dl.error {
                        Label(err, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange).font(.system(size: 12))
                        Button("Retry") { state.cancelDownload(d.id); state.download(d) }
                    } else {
                        ProgressView(value: dl.progress).tint(Theme.accent)
                        HStack {
                            Text("\(Fmt.bytes(dl.bytes)) of \(Fmt.bytes(dl.total))")
                            Spacer()
                            Text("\(Fmt.bytes(Int64(dl.speed)))/s")
                            Button("Cancel") { state.cancelDownload(d.id) }.controlSize(.small)
                        }
                        .font(.system(size: 11).monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
            } else if installed {
                HStack {
                    Button {
                        if let m = state.localModels.first(where: { $0.repo == d.id }) {
                            state.load(m)
                            state.newChat()
                        }
                    } label: {
                        Label("Load & Chat", systemImage: "bubble.left.and.text.bubble.right.fill").frame(minWidth: 140)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!d.isSupported)
                    Label("Downloaded", systemImage: "checkmark.circle.fill").foregroundStyle(Theme.good).font(.system(size: 12))
                }
            } else {
                Button { state.download(d) } label: {
                    Label("Download \(Fmt.bytes(d.downloadSize))", systemImage: "arrow.down.circle.fill").frame(minWidth: 160)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!d.isSupported)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(cornerRadius: 16)
    }

    static func stripFrontMatter(_ s: String) -> String {
        guard s.hasPrefix("---") else { return s }
        let lines = s.components(separatedBy: "\n")
        if let end = lines.dropFirst().firstIndex(where: { $0.trimmingCharacters(in: .whitespaces) == "---" }) {
            return lines[(end + 1)...].joined(separator: "\n")
        }
        return s
    }
}
