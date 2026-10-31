import AppKit
import MLXEngine
import SwiftUI

// When launched as a bare executable (`swift run MLXRunner`), become a regular foreground app.
NSApplication.shared.setActivationPolicy(.regular)
MLXRunnerApp.main()

struct MLXRunnerApp: App {
    @State private var state = AppState()
    @State private var updates = UpdateController()

    var body: some Scene {
        WindowGroup("MLX Runner", id: "main") {
            RootView()
                .environment(state)
                .environment(updates)
                .frame(minWidth: WindowLayout.windowMinWidth, minHeight: 640)
                .onAppear {
                    NSApp.activate(ignoringOtherApps: true)
                    DevSnapshot.runIfRequested(state)
                    updates.start()  // no-op in DevSnapshot mode or when already started
                }
        }
        // First launch has no saved frame: open roomy instead of at the minimum width.
        .defaultSize(width: 1280, height: 820)
        .windowResizability(.contentMinSize)
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About MLX Runner") { showAboutPanel() }
            }
            CommandGroup(after: .appInfo) {
                CheckForUpdatesButton(updates: updates)
            }
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { state.newChat() }.keyboardShortcut("n")
            }
            CommandMenu("Model") {
                Button("Discover Models") { state.selection = .discover }.keyboardShortcut("d", modifiers: [.command, .shift])
                Button("My Models") { state.selection = .models }.keyboardShortcut("m", modifiers: [.command, .shift])
                Divider()
                Button("Stop Generating") { state.stopGeneration() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!state.isGenerating)
                Button("Eject Model") { state.eject() }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
                    .disabled(state.loadedInfo == nil)
            }
        }

        Window("Software Update", id: UpdateController.windowID) {
            UpdateView()
                .environment(state)
                .environment(updates)
                // Release notes come from the network: their links go through LinkPolicy.
                .modifier(SafeLinkHandling())
        }
        .windowResizability(.contentSize)
        .defaultPosition(.center)

        Settings {
            SettingsView().environment(state).environment(updates)
        }
    }
}

/// Window and column minimums. The window minimum is the sum of what the columns need, so
/// shrinking the window can never squeeze the sidebar or a detail screen below its own minimum.
enum WindowLayout {
    static let sidebarMin: CGFloat = 240
    static let sidebarMax: CGFloat = 340
    /// The conversation column alone.
    static let chatMin: CGFloat = 560
    /// Chat with the inspector open (560 + 260); also fits Discover's two panes (340 + 400 + divider).
    static let detailMin: CGFloat = chatMin + 260
    static let windowMinWidth: CGFloat = sidebarMin + detailMin
}

struct RootView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        @Bindable var state = state
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: WindowLayout.sidebarMin, ideal: 260, max: WindowLayout.sidebarMax)
        } detail: {
            ZStack {
                switch state.selection {
                case .chat(let id):
                    ChatView(conversationID: id).id(id)
                case .discover:
                    DiscoverView()
                case .models:
                    MyModelsView()
                case .server:
                    ServerView()
                case nil:
                    WelcomeView()
                }
            }
            .frame(minWidth: WindowLayout.detailMin, maxWidth: .infinity, maxHeight: .infinity)
            .navigationSplitViewColumnWidth(min: WindowLayout.detailMin, ideal: 1020)
        }
        .modifier(SafeLinkHandling())
        .alert("Chat history", isPresented: Binding(
            get: { state.persistenceError != nil },
            set: { if !$0 { state.persistenceError = nil } })
        ) {
            Button("Show in Finder") { NSWorkspace.shared.open(AppState.supportDir) }
            Button("OK", role: .cancel) {}
        } message: {
            Text(state.persistenceError ?? "")
        }
    }
}

/// Routes every link opened from SwiftUI views (Markdown model cards, model output) through
/// `LinkPolicy`: https huggingface.co opens directly, other http(s)/mailto links ask first and
/// show the real URL, and every other scheme (file:, smb:, custom app schemes) is discarded.
struct SafeLinkHandling: ViewModifier {
    @State private var pending: URL?

    func body(content: Content) -> some View {
        content
            .environment(\.openURL, OpenURLAction { url in
                switch LinkPolicy.decide(url) {
                case .open: return .systemAction
                case .confirm:
                    pending = url
                    return .handled
                case .discard: return .discarded
                }
            })
            .alert("Open this link?", isPresented: Binding(get: { pending != nil }, set: { if !$0 { pending = nil } }),
                   presenting: pending) { url in
                Button("Open") { NSWorkspace.shared.open(url) }
                Button("Cancel", role: .cancel) {}
            } message: { url in
                Text(url.absoluteString)
            }
    }
}

struct WelcomeView: View {
    @Environment(AppState.self) private var state
    var body: some View {
        ZStack {
            AuroraBackground()
            VStack(spacing: 18) {
                GradientIcon(systemName: "cpu", size: 72)
                Text("MLX Runner").font(.system(size: 34, weight: .bold, design: .rounded))
                Text("Run open models from Hugging Face on Apple Silicon — natively, with Metal.")
                    .foregroundStyle(.secondary)
                HStack {
                    Button("Discover Models") { state.selection = .discover }.buttonStyle(.borderedProminent)
                    Button("New Chat") { state.newChat() }
                }
            }
        }
    }
}
