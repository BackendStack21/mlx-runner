import AppKit
import MLXEngine
import SwiftUI

/// Owns the app's update state: automatic checks, the "Check for Updates…" window, and the
/// verified install + relaunch (see `Updater` for the security checks).
@Observable @MainActor
final class UpdateController {
    static let autoCheckKey = "autoUpdateCheck"
    static let windowID = "updates"
    private static let interval: Duration = .seconds(6 * 3600)

    private(set) var info: UpdateInfo?
    private(set) var checking = false
    /// Non-nil while an install runs.
    private(set) var phase: UpdatePhase?
    private(set) var error: String?

    @ObservationIgnored private let updater = Updater()
    @ObservationIgnored private var loop: Task<Void, Never>?

    var current: String { MLXRunnerVersion.current }
    var updateAvailable: Bool { info?.available == true }
    var installing: Bool { phase != nil }

    /// Defaults to on: `bool(forKey:)` would read an unset key as false.
    var autoCheckEnabled: Bool { UserDefaults.standard.object(forKey: Self.autoCheckKey) as? Bool ?? true }

    /// Starts the launch check and the 6-hourly re-check (each only when enabled).
    func start() {
        guard !DevSnapshot.isActive, loop == nil else { return }
        loop = Task { [weak self] in
            while !Task.isCancelled {
                if let self, self.autoCheckEnabled { await self.check(force: false, quiet: true) }
                try? await Task.sleep(for: Self.interval)
            }
        }
    }

    /// Called when the Settings toggle changes.
    func autoCheckChanged() {
        guard autoCheckEnabled else { return }
        Task { await check(force: false, quiet: true) }
    }

    /// "Check for Updates…": always hits GitHub and shows errors.
    func checkNow() {
        Task { await check(force: true, quiet: false) }
    }

    private func check(force: Bool, quiet: Bool) async {
        guard !DevSnapshot.isActive, !checking, !installing else { return }
        checking = true
        if !quiet { error = nil }
        defer { checking = false }
        do {
            info = try await updater.check(force: force)
        } catch {
            if !quiet { self.error = error.localizedDescription }
        }
    }

    /// Downloads, verifies and installs the update, then relaunches into it.
    func install() {
        guard !DevSnapshot.isActive, !installing, info?.canInstall == true else { return }
        error = nil
        phase = .checking
        Task {
            do {
                let applied = try await updater.apply { p in
                    DispatchQueue.main.async { MainActor.assumeIsolated { self.phase = p } }
                }
                try Updater.relaunch(applied.target)
                NSApp.terminate(nil)
            } catch {
                // Let any queued phase updates land first, then clear.
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        self.phase = nil
                        self.error = error.localizedDescription
                    }
                }
            }
        }
    }

    var phaseText: String {
        switch phase {
        case .checking?: "Checking the release…"
        case .signature?: "Verifying the release signature…"
        case .download(let f)?: f > 0 ? "Downloading… \(Int(f * 100))%" : "Downloading…"
        case .checksum?: "Verifying the download…"
        case .install?: "Installing…"
        case .done(let v)?: "Installed \(v). Restarting…"
        case nil: ""
        }
    }
}

// MARK: - Views

/// "Check for Updates…" in the app menu.
struct CheckForUpdatesButton: View {
    let updates: UpdateController
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("Check for Updates…") {
            updates.checkNow()
            openWindow(id: UpdateController.windowID)
        }
        .disabled(DevSnapshot.isActive)
    }
}

/// The Software Update window: current vs. latest, release notes, install with progress.
struct UpdateView: View {
    @Environment(UpdateController.self) private var updates
    @Environment(AppState.self) private var state
    @State private var confirmStop = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                GradientIcon(systemName: "arrow.down.circle", size: 44)
                VStack(alignment: .leading, spacing: 2) {
                    Text(headline).font(.system(size: 16, weight: .bold, design: .rounded))
                    Text(subtitle).font(.system(size: 12)).foregroundStyle(.secondary)
                }
                Spacer()
            }

            if updates.checking && updates.info == nil {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Checking for updates…").foregroundStyle(.secondary)
                }
            }

            if let info = updates.info, info.available, !info.notes.isEmpty {
                ScrollView {
                    MarkdownView(text: info.notes, baseSize: 13).padding(12)
                }
                .frame(minHeight: 140, maxHeight: 280)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }

            if let reason = updates.info?.reason, updates.updateAvailable {
                Label(reason, systemImage: "info.circle").font(.system(size: 12)).foregroundStyle(.secondary)
            }

            if let error = updates.error {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 12)).foregroundStyle(.red).textSelection(.enabled)
            }

            if updates.installing {
                VStack(alignment: .leading, spacing: 6) {
                    if case .download(let f) = updates.phase, f > 0 {
                        ProgressView(value: f)
                    } else {
                        ProgressView().progressViewStyle(.linear)
                    }
                    Text(updates.phaseText).font(.system(size: 11).monospacedDigit()).foregroundStyle(.secondary)
                }
            }

            HStack {
                if let url = updates.info?.url, updates.updateAvailable {
                    Button("Release Page") { NSWorkspace.shared.open(url) }
                }
                Spacer()
                Button("Check Again") { updates.checkNow() }
                    .disabled(updates.checking || updates.installing)
                if updates.updateAvailable {
                    Button("Install and Restart") {
                        if state.isGenerating { confirmStop = true } else { updates.install() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(updates.info?.canInstall != true || updates.installing)
                }
            }
        }
        .padding(20)
        .frame(width: 520)
        .confirmationDialog("A reply is still being generated.", isPresented: $confirmStop) {
            Button("Stop and Install") {
                state.stopGeneration()
                updates.install()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Installing the update restarts MLX Runner.")
        }
    }

    private var headline: String {
        if let info = updates.info, info.available, let latest = info.latest { return "MLX Runner \(latest) is available" }
        if updates.info != nil { return "MLX Runner is up to date" }
        return "Software Update"
    }

    private var subtitle: String {
        if let info = updates.info {
            if info.available { return "You have \(info.current)." }
            if let latest = info.latest { return "You have \(info.current); the latest release is \(latest)." }
            return "You have \(info.current); no release was found."
        }
        return "You have \(updates.current)."
    }
}

/// A small pill shown in the sidebar footer when an update is available.
struct UpdatePill: View {
    @Environment(UpdateController.self) private var updates
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        if updates.updateAvailable, let latest = updates.info?.latest {
            Button {
                openWindow(id: UpdateController.windowID)
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "arrow.up.circle.fill")
                    Text("Update to \(latest)")
                }
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(Theme.gradient, in: Capsule())
            }
            .buttonStyle(.plain)
            .help("MLX Runner \(latest) is available (you have \(updates.current))")
        }
    }
}

/// The standard About panel, with the 21no.de credit as a link.
@MainActor
func showAboutPanel() {
    let para = NSMutableParagraphStyle()
    para.alignment = .center
    let base: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
        .foregroundColor: NSColor.labelColor,
        .paragraphStyle: para,
    ]
    let credits = NSMutableAttributedString(string: "Made by 21no.de — ", attributes: base)
    var link = base
    link[.link] = URL(string: "https://21no.de")!
    credits.append(NSAttributedString(string: "https://21no.de", attributes: link))
    NSApp.orderFrontStandardAboutPanel(options: [
        .credits: credits,
        .applicationVersion: MLXRunnerVersion.current,
        // Under `swift run` there's no CFBundleVersion; don't show "(null)".
        .version: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "",
        .applicationName: "MLX Runner",
    ])
    NSApp.activate(ignoringOtherApps: true)
}
