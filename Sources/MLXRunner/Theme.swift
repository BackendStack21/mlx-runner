import SwiftUI

enum Theme {
    static let accent = Color(red: 0.42, green: 0.36, blue: 1.0)
    static let accent2 = Color(red: 0.13, green: 0.73, blue: 0.98)
    static let pink = Color(red: 0.98, green: 0.35, blue: 0.62)
    static let gradient = LinearGradient(colors: [accent, accent2], startPoint: .topLeading, endPoint: .bottomTrailing)
    static let warmGradient = LinearGradient(colors: [pink, accent], startPoint: .topLeading, endPoint: .bottomTrailing)
    /// Fills behind white text (user bubble, New Chat, update pill, send button). The cyan end of `gradient`
    /// gives white text about 2.2:1; this deeper blue keeps it at or above 4.5:1 along the whole fill.
    static let fillEnd = Color(red: 0.0, green: 0.45, blue: 0.8)
    static let fillGradient = LinearGradient(colors: [accent, fillEnd], startPoint: .topLeading, endPoint: .bottomTrailing)
    static let good = Color(red: 0.2, green: 0.8, blue: 0.5)
}

enum GlassSupport {
    /// Liquid Glass when available; `MLXRUNNER_NO_GLASS=1` forces the material fallback.
    static let enabled: Bool = {
        guard ProcessInfo.processInfo.environment["MLXRUNNER_NO_GLASS"] == nil else { return false }
        if #available(macOS 26.0, *) { return true }
        return false
    }()
}

extension View {
    /// Liquid Glass on macOS 26+, material card on earlier systems.
    @ViewBuilder
    func glassCard(cornerRadius: CGFloat = 16, interactive: Bool = false) -> some View {
        if #available(macOS 26.0, *), GlassSupport.enabled {
            self.glassEffect(interactive ? .regular.interactive() : .regular, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        } else {
            self.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous).strokeBorder(.white.opacity(0.08)))
        }
    }

    @ViewBuilder
    func glassCapsule(interactive: Bool = false) -> some View {
        if #available(macOS 26.0, *), GlassSupport.enabled {
            self.glassEffect(interactive ? .regular.interactive() : .regular, in: Capsule())
        } else {
            self.background(.ultraThinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.08)))
        }
    }
}

/// Small rounded label used for architecture / quantization / status.
struct Badge: View {
    let text: String
    var color: Color = .secondary
    var systemImage: String?

    var body: some View {
        HStack(spacing: 3) {
            if let systemImage { Image(systemName: systemImage).font(.system(size: 9, weight: .bold)) }
            Text(text).font(.system(size: 10, weight: .semibold, design: .rounded)).lineLimit(1)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .foregroundStyle(color)
        .background(color.opacity(0.14), in: Capsule())
    }
}

/// Animated aurora background used behind empty states.
struct AuroraBackground: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase = false
    var body: some View {
        ZStack {
            Circle().fill(Theme.accent.opacity(0.35)).frame(width: 420).blur(radius: 120)
                .offset(x: phase ? -140 : -60, y: phase ? -80 : -140)
            Circle().fill(Theme.accent2.opacity(0.28)).frame(width: 380).blur(radius: 120)
                .offset(x: phase ? 160 : 90, y: phase ? 60 : 120)
            Circle().fill(Theme.pink.opacity(0.18)).frame(width: 300).blur(radius: 110)
                .offset(x: phase ? 20 : -40, y: phase ? 160 : 60)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 9).repeatForever(autoreverses: true)) { phase.toggle() }
        }
    }
}

struct GradientIcon: View {
    let systemName: String
    var size: CGFloat = 44
    var body: some View {
        Image(systemName: systemName)
            .font(.system(size: size * 0.48, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(Theme.gradient, in: RoundedRectangle(cornerRadius: size * 0.28, style: .continuous))
            .shadow(color: Theme.accent.opacity(0.45), radius: 12, y: 6)
            .accessibilityHidden(true)   // decorative: every use sits next to a text title
    }
}

/// A model's architecture. An unsupported one also gets a warning symbol and says so to VoiceOver,
/// so the state never depends on the orange colour alone.
struct ArchitectureBadge: View {
    let type: String
    let supported: Bool

    var body: some View {
        Badge(text: type, color: supported ? Theme.accent2 : .orange,
              systemImage: supported ? nil : "exclamationmark.triangle.fill")
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(supported ? "\(type) architecture" : "\(type) architecture, not supported")
            .help(supported ? "Architecture: \(type)" : "The \(type) architecture isn't supported by the built-in engine yet")
    }
}
