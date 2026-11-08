import SwiftUI
import MLXEngine

/// Hand-picked models that run well with the built-in engine.
struct RecommendedModel: Identifiable, Hashable {
    let id: String
    let title: String
    let blurb: String
    let gigabytes: Double
    let icon: String

    private static let icons: [String: String] = [
        "mlx-community/Qwen3-0.6B-4bit": "hare.fill",
        "mlx-community/Qwen3.5-0.8B-4bit": "bolt.horizontal.fill",
        "mlx-community/Llama-3.2-3B-Instruct-4bit": "bolt.fill",
        "mlx-community/Qwen3-4B-4bit": "brain.head.profile",
        "mlx-community/Phi-3.5-mini-instruct-4bit": "graduationcap.fill",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit": "wind",
        "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": "chevron.left.forwardslash.chevron.right",
        "mlx-community/Qwen3-8B-4bit": "sparkles",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": "flame.fill",
        "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit": "point.3.connected.trianglepath.dotted",
    ]

    /// The shared engine catalog plus this app's icons.
    static let all: [RecommendedModel] = Catalog.standard.models.map {
        RecommendedModel(id: $0.id, title: $0.title, blurb: $0.blurb, gigabytes: $0.gigabytes, icon: icons[$0.id] ?? "cube.fill")
    }

    /// Picks that fit comfortably in this Mac's unified memory.
    static var forThisMac: [RecommendedModel] {
        let ram = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return all.filter { $0.gigabytes < ram * 0.6 }
    }
}

struct RecommendedStrip: View {
    @Environment(AppState.self) private var state
    @Binding var selected: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Recommended for this Mac", systemImage: "star.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 14)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(RecommendedModel.forThisMac) { m in
                        Button { selected = m.id } label: { card(m) }
                            .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 2)
            }
        }
        .padding(.vertical, 10)
    }

    private func card(_ m: RecommendedModel) -> some View {
        let installed = state.localModels.contains { $0.repo == m.id }
        let isSelected = selected == m.id
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: m.icon).foregroundStyle(Theme.gradient).font(.system(size: 15, weight: .semibold))
                Spacer()
                if installed {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.good).font(.system(size: 12))
                } else if state.downloads[m.id] != nil {
                    ProgressView().controlSize(.mini)
                }
            }
            Text(m.title).font(.system(size: 12.5, weight: .semibold)).lineLimit(1)
            Text(m.blurb).font(.system(size: 10.5)).foregroundStyle(.secondary).lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Text(String(format: "%.1f GB", m.gigabytes)).font(.system(size: 10, weight: .medium).monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        .padding(10)
        .frame(width: 150, height: 112, alignment: .topLeading)
        .glassCard(cornerRadius: 12, interactive: true)
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
            .strokeBorder(isSelected ? AnyShapeStyle(Theme.gradient) : AnyShapeStyle(Color.clear), lineWidth: 1.5))
    }
}
