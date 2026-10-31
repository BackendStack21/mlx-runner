import AppKit
import SwiftUI

/// Development aids for reviewing the UI. Both modes show sample chats (never the user's) and
/// never persist anything.
///
/// - `MLXRUNNER_SNAPSHOT=/dir`: renders each main screen offscreen to PNG (fast, headless, but
///   offscreen rendering skips Liquid Glass and vibrancy, so it is for layout review only).
/// - `MLXRUNNER_TOUR=/dir`: drives the real window through each screen for real screenshots.
///   At each step it writes `<dir>/ready-<name>` (holding the window number) and waits until
///   `<dir>/<name>.png` exists; `scripts/shots.sh` takes the shots with `screencapture -l`.
@MainActor
enum DevSnapshot {
    static let env = ProcessInfo.processInfo.environment
    static let isActive = env["MLXRUNNER_SNAPSHOT"] != nil || env["MLXRUNNER_TOUR"] != nil

    static func runIfRequested(_ state: AppState) {
        if let dir = env["MLXRUNNER_TOUR"] { tour(state, dir: URL(fileURLWithPath: dir)); return }
        guard let dir = env["MLXRUNNER_SNAPSHOT"] else { return }
        let screens: [(String, () -> Void)] = [
            ("discover", { state.selection = .discover }),
            ("models", { state.selection = .models }),
            ("chat", { if let c = state.conversations.first { state.selection = .chat(c.id) } else { state.newChat() } }),
            ("server", { state.selection = .server }),
        ]
        Task { @MainActor in
            loadShowcaseModel(state)
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            for (name, select) in screens {
                select()
                try? await Task.sleep(nanoseconds: 2_500_000_000)
                capture(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
            }
            if env["MLXRUNNER_SNAPSHOT_CHAT"] != nil {
                let id = state.newChat()
                try? await Task.sleep(nanoseconds: 500_000_000)
                state.enableThinking = false
                state.send(tourPrompt, in: id)
                while state.isGenerating { try? await Task.sleep(nanoseconds: 300_000_000) }
                try? await Task.sleep(nanoseconds: 800_000_000)
                capture(to: URL(fileURLWithPath: dir).appendingPathComponent("chat-reply.png"))
            }
            NSApp.terminate(nil)
        }
    }

    static let tourPrompt = "Give me a markdown table of 3 planets with their diameters, then a short Swift code example."

    private static func loadShowcaseModel(_ state: AppState) {
        let preferred = env["MLXRUNNER_TOUR_MODEL"] ?? "Qwen3-0.6B"
        if let m = state.localModels.first(where: { $0.name.hasPrefix(preferred) }) ?? state.localModels.first {
            state.load(m)
        }
    }

    private static func tour(_ state: AppState, dir: URL) {
        Task { @MainActor in
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try? await Task.sleep(nanoseconds: 800_000_000)
            guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }) else { NSApp.terminate(nil); return }
            // A fixed size so shots are consistent between runs.
            window.setContentSize(NSSize(width: 1280, height: 860))
            window.center()
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
            loadShowcaseModel(state)
            while state.loadedInfo == nil { try? await Task.sleep(nanoseconds: 200_000_000) }

            // Chat: a real reply, generated live.
            let id = state.newChat()
            state.enableThinking = false
            state.send(tourPrompt, in: id)
            while state.isGenerating { try? await Task.sleep(nanoseconds: 300_000_000) }
            await shot("chat", window: window, dir: dir, settle: 1.0)

            state.selection = .discover
            await shot("discover", window: window, dir: dir, settle: 5.0)  // Hub search + model card
            state.selection = .models
            await shot("models", window: window, dir: dir, settle: 1.5)
            state.selection = .server
            await shot("server", window: window, dir: dir, settle: 1.5)
            NSApp.terminate(nil)
        }
    }

    private static func shot(_ name: String, window: NSWindow, dir: URL, settle: Double) async {
        try? await Task.sleep(nanoseconds: UInt64(settle * 1e9))
        let png = dir.appendingPathComponent("\(name).png")
        try? FileManager.default.removeItem(at: png)
        try? String(window.windowNumber).write(to: dir.appendingPathComponent("ready-\(name)"), atomically: true, encoding: .utf8)
        let deadline = Date().addingTimeInterval(60)
        while !FileManager.default.fileExists(atPath: png.path), Date() < deadline {
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        try? await Task.sleep(nanoseconds: 300_000_000)  // let screencapture finish writing
    }

    static func capture(to url: URL) {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }),
              let view = window.contentView?.superview ?? window.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: url)
    }

    /// Sample sidebar history for screenshots.
    static func sampleConversations() -> [Conversation] {
        let now = Date()
        func chat(_ title: String, _ minutesAgo: Double, _ user: String, _ reply: String) -> Conversation {
            var c = Conversation()
            c.title = title
            c.created = now.addingTimeInterval(-minutesAgo * 60)
            c.updated = c.created
            c.messages = [StoredMessage(role: .user, content: user, date: c.created),
                          StoredMessage(role: .assistant, content: reply, date: c.created)]
            return c
        }
        return [
            chat("Explain rotary position embeddings", 12, "Explain rotary position embeddings in two sentences.",
                 "RoPE rotates each query and key pair by an angle proportional to its position, so attention scores depend on relative distance."),
            chat("Refactor a Swift actor", 95, "How do I make this cache an actor?", "Wrap the dictionary in an `actor` and make reads `async`."),
            chat("Haiku about unified memory", 60 * 26, "Write a haiku about unified memory.",
                 "One pool for us all,\nweights and tokens side by side,\nno copies to wait."),
        ]
    }
}
