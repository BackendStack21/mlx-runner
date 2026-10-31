import AppKit
import MLXEngine
import SwiftUI

/// Holds the OpenAI-compatible server across view changes.
@MainActor @Observable
final class ServerController {
    static let shared = ServerController()
    var running = false
    var port: Int = UserDefaults.standard.object(forKey: "serverPort") as? Int ?? 1234
    var allowLAN = false
    var log: [String] = []
    var error: String?
    @ObservationIgnored private var server: OpenAIServer?

    func start() {
        stop()
        let s = OpenAIServer(port: UInt16(clamping: port), allowLAN: allowLAN)
        s.onLog = { line in
            Task { @MainActor in
                let ts = Date().formatted(date: .omitted, time: .standard)
                ServerController.shared.log.append("\(ts)  \(line)")
                if ServerController.shared.log.count > 500 { ServerController.shared.log.removeFirst(100) }
            }
        }
        do {
            try s.start()
            server = s
            running = true
            error = nil
            UserDefaults.standard.set(port, forKey: "serverPort")
        } catch {
            self.error = error.localizedDescription
        }
    }

    func stop() {
        server?.stop()
        server = nil
        running = false
    }
}

struct ServerView: View {
    @Environment(AppState.self) private var state
    @State private var server = ServerController.shared

    var body: some View {
        @Bindable var server = server
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 14) {
                    GradientIcon(systemName: "network", size: 48)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Local Server").font(.system(size: 28, weight: .bold, design: .rounded))
                        Text("OpenAI-compatible API for your tools and apps, served from the loaded model.")
                            .foregroundStyle(.secondary)
                    }
                }

                HStack(spacing: 16) {
                    HStack(spacing: 8) {
                        Circle().fill(server.running ? Theme.good : Color.secondary.opacity(0.5)).frame(width: 10, height: 10)
                            .shadow(color: server.running ? Theme.good : .clear, radius: 5)
                        Text(server.running ? "Running" : "Stopped").font(.system(size: 14, weight: .semibold))
                    }
                    Divider().frame(height: 20)
                    LabeledContent("Port") {
                        TextField("", value: $server.port, format: .number.grouping(.never))
                            .frame(width: 70)
                            .disabled(server.running)
                    }
                    .fixedSize()
                    Toggle("Serve on local network", isOn: $server.allowLAN)
                        .disabled(server.running)
                    Spacer()
                    Button {
                        server.running ? server.stop() : server.start()
                    } label: {
                        Label(server.running ? "Stop" : "Start Server", systemImage: server.running ? "stop.fill" : "play.fill")
                            .frame(minWidth: 110)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(server.running ? .red : Theme.accent)
                    .controlSize(.large)
                }
                .padding(16)
                .glassCard()

                if let e = server.error {
                    Label(e, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                }
                if state.loadedInfo == nil {
                    Label("No model is loaded — requests will return 503 until you load one.", systemImage: "info.circle")
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Endpoints").font(.headline)
                    endpoint("GET", "/v1/models")
                    endpoint("POST", "/v1/chat/completions")
                    endpoint("POST", "/v1/completions")
                    Text("Example").font(.subheadline.weight(.semibold)).padding(.top, 6)
                    CodeBlock(language: "bash", code: """
                    curl http://localhost:\(server.port)/v1/chat/completions \\
                      -H "Content-Type: application/json" \\
                      -d '{"model": "\(state.loadedInfo?.name ?? "local")", "stream": true,
                           "messages": [{"role": "user", "content": "Hello!"}]}'
                    """)
                }
                .padding(16)
                .glassCard()

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Request log").font(.headline)
                        Spacer()
                        Button("Clear") { server.log.removeAll() }.controlSize(.small)
                    }
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 2) {
                                ForEach(Array(server.log.enumerated()), id: \.offset) { i, line in
                                    Text(line).font(.system(size: 11.5, design: .monospaced)).id(i)
                                        .textSelection(.enabled)
                                }
                                if server.log.isEmpty {
                                    Text("No requests yet.").font(.system(size: 12)).foregroundStyle(.tertiary)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                        }
                        .frame(height: 220)
                        .background(Color(nsColor: .textBackgroundColor).opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                        .onChange(of: server.log.count) { _, n in proxy.scrollTo(n - 1, anchor: .bottom) }
                    }
                }
                .padding(16)
                .glassCard()
            }
            .padding(28)
            .frame(maxWidth: 900, alignment: .leading)
        }
        .navigationTitle("Local Server")
    }

    private func endpoint(_ method: String, _ path: String) -> some View {
        HStack(spacing: 10) {
            Text(method).font(.system(size: 10.5, weight: .bold, design: .monospaced))
                .foregroundStyle(.white)
                .frame(width: 44)
                .padding(.vertical, 3)
                .background(method == "GET" ? Theme.accent2 : Theme.accent, in: RoundedRectangle(cornerRadius: 5))
            Text("http://localhost:\(server.port)\(path)").font(.system(size: 12.5, design: .monospaced)).textSelection(.enabled)
        }
    }
}

struct SettingsView: View {
    @Environment(AppState.self) private var state
    @AppStorage("hfToken") private var hfToken = ""

    var body: some View {
        Form {
            Section("Hugging Face") {
                SecureField("Access token (for gated models)", text: $hfToken)
                Text("Optional. Create one at huggingface.co/settings/tokens.").font(.caption).foregroundStyle(.secondary)
            }
            Section("Storage") {
                LabeledContent("Models folder") {
                    Button(ModelStore.root.path) { NSWorkspace.shared.open(ModelStore.root) }.buttonStyle(.link)
                }
                LabeledContent("Also scanning") { Text(ModelStore.huggingFaceCache.path).foregroundStyle(.secondary) }
            }
            Section("Chats") {
                Button("Delete all chats", role: .destructive) { state.clearAllChats() }
            }
            Section("About") {
                LabeledContent("Engine", value: "Native Swift + Metal (no third-party dependencies)")
                LabeledContent("GPU", value: GPU.shared.device.name)
                LabeledContent("Architectures", value: SupportedArchitectures.modelTypes.sorted().joined(separator: ", "))
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 460)
    }
}
