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
                .frame(minWidth: 980, minHeight: 640)
                .onAppear {
                    NSApp.activate(ignoringOtherApps: true)
                    DevSnapshot.runIfRequested(state)
                    updates.start()  // no-op in DevSnapshot mode or when already started
                }
        }
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

struct RootView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        @Bindable var state = state
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 320)
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
            .frame(maxWidth: .infinity, maxHeight: .infinity)
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
