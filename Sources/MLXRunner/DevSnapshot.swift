import AppKit
import SwiftUI

/// Development aid: `MLXRUNNER_SNAPSHOT=/dir` makes the app write PNG snapshots of its
/// window for each main screen, so UI can be reviewed in headless/CI environments.
@MainActor
enum DevSnapshot {
    static func runIfRequested(_ state: AppState) {
        guard let dir = ProcessInfo.processInfo.environment["MLXRUNNER_SNAPSHOT"] else { return }
        let screens: [(String, () -> Void)] = [
            ("discover", { state.selection = .discover }),
            ("models", { state.selection = .models }),
            ("chat", { if let c = state.conversations.first { state.selection = .chat(c.id) } else { state.newChat() } }),
            ("server", { state.selection = .server }),
        ]
        Task { @MainActor in
            if let m = state.localModels.first(where: { $0.name.hasPrefix("Qwen3") }) ?? state.localModels.first {
                state.load(m)
            }
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            for (name, select) in screens {
                select()
                try? await Task.sleep(nanoseconds: 2_500_000_000)
                capture(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
            }
            if ProcessInfo.processInfo.environment["MLXRUNNER_SNAPSHOT_CHAT"] != nil {
                let id = state.newChat()
                try? await Task.sleep(nanoseconds: 500_000_000)
                state.enableThinking = false
                state.send("Give me a markdown table of 3 planets with their diameters, then a short Swift code example.", in: id)
                while state.isGenerating { try? await Task.sleep(nanoseconds: 300_000_000) }
                try? await Task.sleep(nanoseconds: 800_000_000)
                capture(to: URL(fileURLWithPath: dir).appendingPathComponent("chat-reply.png"))
            }
            NSApp.terminate(nil)
        }
    }

    static func capture(to url: URL) {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }),
              let view = window.contentView?.superview ?? window.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: url)
    }
}
