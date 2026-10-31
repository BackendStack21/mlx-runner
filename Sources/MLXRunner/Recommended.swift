import SwiftUI

/// Hand-picked models that run well with the built-in engine.
struct RecommendedModel: Identifiable, Hashable {
    let id: String
    let title: String
    let blurb: String
    let gigabytes: Double
    let icon: String

    static let all: [RecommendedModel] = [
        .init(id: "mlx-community/Qwen3-0.6B-4bit", title: "Qwen3 0.6B", blurb: "Tiny and blazing fast, with thinking mode", gigabytes: 0.35, icon: "hare.fill"),
        .init(id: "mlx-community/Llama-3.2-3B-Instruct-4bit", title: "Llama 3.2 3B", blurb: "Meta's compact all-rounder", gigabytes: 1.8, icon: "bolt.fill"),
        .init(id: "mlx-community/Qwen3-4B-4bit", title: "Qwen3 4B", blurb: "Strong reasoning in a small footprint", gigabytes: 2.3, icon: "brain.head.profile"),
        .init(id: "mlx-community/Phi-3.5-mini-instruct-4bit", title: "Phi-3.5 mini", blurb: "Microsoft's efficient 3.8B model", gigabytes: 2.2, icon: "graduationcap.fill"),
        .init(id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit", title: "Mistral 7B v0.3", blurb: "Classic, fluent 7B instruct model", gigabytes: 4.1, icon: "wind"),
        .init(id: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", title: "Qwen2.5 Coder 7B", blurb: "Code generation and review", gigabytes: 4.3, icon: "chevron.left.forwardslash.chevron.right"),
        .init(id: "mlx-community/Qwen3-8B-4bit", title: "Qwen3 8B", blurb: "Best quality under 5 GB", gigabytes: 4.6, icon: "sparkles"),
        .init(id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", title: "Llama 3.1 8B", blurb: "Popular 8B with 128K context", gigabytes: 4.5, icon: "flame.fill"),
        .init(id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", title: "DeepSeek R1 Distill 7B", blurb: "Step-by-step reasoning", gigabytes: 4.3, icon: "point.3.connected.trianglepath.dotted"),
    ]

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
