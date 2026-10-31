import AppKit
import MLXEngine
import SwiftUI

struct MyModelsView: View {
    @Environment(AppState.self) private var state
    @State private var confirmDelete: LocalModel?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("My Models").font(.system(size: 28, weight: .bold, design: .rounded))
                        Text("\(state.localModels.count) models · \(Fmt.bytes(state.localModels.reduce(0) { $0 + $1.sizeBytes })) on disk")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button { state.refreshModels() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                    Button {
                        try? FileManager.default.createDirectory(at: ModelStore.root, withIntermediateDirectories: true)
                        NSWorkspace.shared.open(ModelStore.root)
                    } label: { Label("Show in Finder", systemImage: "folder") }
                }

                if !state.downloads.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Downloading").font(.headline)
                        ForEach(state.downloads.keys.sorted(), id: \.self) { repo in
                            let d = state.downloads[repo]!
                            HStack {
                                Text(repo).font(.system(size: 12, weight: .medium)).lineLimit(1)
                                ProgressView(value: d.progress).tint(Theme.accent)
                                Text("\(Int(d.progress * 100))%").font(.system(size: 11).monospacedDigit()).frame(width: 36)
                                Button { state.cancelDownload(repo) } label: { Image(systemName: "xmark.circle.fill") }
                                    .buttonStyle(.borderless)
                            }
                        }
                    }
                    .padding(14)
                    .glassCard()
                }

                if state.localModels.isEmpty {
                    VStack(spacing: 12) {
                        GradientIcon(systemName: "shippingbox", size: 56)
                        Text("No models yet").font(.title3.bold())
                        Text("Download MLX models from Hugging Face to run them on this Mac.").foregroundStyle(.secondary)
                        Button("Discover Models") { state.selection = .discover }.buttonStyle(.borderedProminent)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(50)
                } else {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 14)], spacing: 14) {
                        ForEach(state.localModels) { m in
                            ModelCard(model: m, onDelete: { confirmDelete = m })
                        }
                    }
                }
            }
            .padding(28)
        }
        .navigationTitle("My Models")
        .onAppear { state.refreshModels() }
        .confirmationDialog("Delete \(confirmDelete?.name ?? "")?", isPresented: Binding(
            get: { confirmDelete != nil }, set: { if !$0 { confirmDelete = nil } }), titleVisibility: .visible) {
            Button("Delete \(Fmt.bytes(confirmDelete?.sizeBytes ?? 0))", role: .destructive) {
                if let m = confirmDelete { state.delete(m) }
                confirmDelete = nil
            }
        } message: {
            Text("The model files will be removed from disk.")
        }
    }
}

struct ModelCard: View {
    @Environment(AppState.self) private var state
    let model: LocalModel
    let onDelete: () -> Void
    @State private var hover = false

    var body: some View {
        let loaded = state.loadedInfo?.directory == model.directory
        let loading: Bool = { if case .loading(let n) = state.loadState { return n == model.name }; return false }()
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                GradientIcon(systemName: loaded ? "bolt.fill" : "cube.transparent", size: 36)
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.name).font(.system(size: 14, weight: .semibold)).lineLimit(1)
                    Text(model.author).font(.system(size: 11)).foregroundStyle(.secondary)
                }
                Spacer()
                Menu {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([model.directory]) }
                    Button("Open on Hugging Face") {
                        if let u = URL(string: "https://huggingface.co/\(model.repo)") { NSWorkspace.shared.open(u) }
                    }
                    Divider()
                    Button("Delete…", role: .destructive, action: onDelete)
                } label: { Image(systemName: "ellipsis.circle") }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
            }
            HStack(spacing: 6) {
                if let t = model.modelType { Badge(text: t, color: model.isSupported ? Theme.accent2 : .orange) }
                if let b = model.bits { Badge(text: "\(b)-bit", color: Theme.accent) }
                Badge(text: Fmt.bytes(model.sizeBytes), color: .secondary)
                if model.source == .huggingFaceCache { Badge(text: "HF cache", color: .secondary) }
            }
            HStack {
                if loaded {
                    Label("Loaded", systemImage: "circle.fill").font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Theme.good)
                    Spacer()
                    Button("Chat") { state.newChat() }.controlSize(.small)
                    Button("Eject") { state.eject() }.controlSize(.small)
                } else if loading {
                    ProgressView().controlSize(.small)
                    Text("Loading…").font(.system(size: 11)).foregroundStyle(.secondary)
                    Spacer()
                } else {
                    Spacer()
                    Button {
                        state.load(model)
                        state.newChat()
                    } label: { Label("Load", systemImage: "play.fill") }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(!model.isSupported)
                }
            }
        }
        .padding(16)
        .glassCard(cornerRadius: 18)
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous)
            .strokeBorder(loaded ? AnyShapeStyle(Theme.gradient) : AnyShapeStyle(Color.clear), lineWidth: 1.5))
        .scaleEffect(hover ? 1.01 : 1)
        .animation(.snappy(duration: 0.18), value: hover)
        .onHover { hover = $0 }
    }
}
