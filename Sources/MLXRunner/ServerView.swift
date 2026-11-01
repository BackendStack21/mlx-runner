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
    /// Require `Authorization: Bearer <apiKey>` even on localhost (always required in LAN mode).
    var requireAPIKey: Bool = UserDefaults.standard.bool(forKey: "serverRequireAPIKey") {
        didSet { UserDefaults.standard.set(requireAPIKey, forKey: "serverRequireAPIKey") }
    }
    /// Random key generated once with the system CSPRNG and kept in UserDefaults.
    private(set) var apiKey: String = ServerController.storedOrNewKey()
    /// Browser origins allowed to call the server, comma- or newline-separated.
    var allowedOrigins: String = UserDefaults.standard.string(forKey: "serverAllowedOrigins")
        ?? OpenAIServer.defaultAllowedOrigins.joined(separator: ", ") {
        didSet { UserDefaults.standard.set(allowedOrigins, forKey: "serverAllowedOrigins") }
    }
    var log: [String] = []
    var error: String?
    @ObservationIgnored private var server: OpenAIServer?

    /// The key clients must send, or nil when no key is enforced.
    var effectiveAPIKey: String? { (allowLAN || requireAPIKey) ? apiKey : nil }

    private static func storedOrNewKey() -> String {
        if DevSnapshot.isActive { return OpenAIServer.generateAPIKey() }  // screenshots never show the real key
        if let k = UserDefaults.standard.string(forKey: "serverAPIKey"), !k.isEmpty { return k }
        let k = OpenAIServer.generateAPIKey()
        UserDefaults.standard.set(k, forKey: "serverAPIKey")
        return k
    }

    /// Replaces the API key; a running server restarts so the old key stops working immediately.
    func regenerateAPIKey() {
        apiKey = OpenAIServer.generateAPIKey()
        UserDefaults.standard.set(apiKey, forKey: "serverAPIKey")
        if running { start() }
    }

    func copyAPIKey() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(apiKey, forType: .string)
    }

    func start() {
        stop()
        guard (1...65535).contains(port) else {
            error = "Port must be between 1 and 65535"
            return
        }
        let s = OpenAIServer(port: UInt16(port), allowLAN: allowLAN, apiKey: effectiveAPIKey)
        let origins = allowedOrigins.split(whereSeparator: { $0 == "," || $0.isNewline })
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        s.allowedOrigins = origins
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
                    Text("Security").font(.headline)
                    Toggle("Require API key", isOn: Binding(
                        get: { server.requireAPIKey || server.allowLAN },
                        set: { server.requireAPIKey = $0 }))
                        .disabled(server.running || server.allowLAN)
                    if server.allowLAN {
                        Text("An API key is always required when serving on the local network.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    HStack(spacing: 8) {
                        Text(server.apiKey)
                            .font(.system(size: 12, design: .monospaced))
                            .lineLimit(1).truncationMode(.middle)
                            .textSelection(.enabled)
                            .padding(.horizontal, 8).padding(.vertical, 5)
                            .background(Color(nsColor: .textBackgroundColor).opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                        Button { server.copyAPIKey() } label: { Label("Copy", systemImage: "doc.on.doc") }
                            .controlSize(.small)
                        Button { server.regenerateAPIKey() } label: { Label("Regenerate", systemImage: "arrow.clockwise") }
                            .controlSize(.small)
                            .help("Creates a new key; the old one stops working (the server restarts if running).")
                    }
                    .opacity(server.effectiveAPIKey == nil ? 0.5 : 1)
                    LabeledContent("Allowed browser origins") {
                        TextField("http://localhost:*, http://127.0.0.1:*", text: $server.allowedOrigins)
                            .font(.system(size: 12, design: .monospaced))
                            .disabled(server.running)
                    }
                    Text("Web pages from other origins cannot call the server. Use scheme://host:* for any port, or * to allow all.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(16)
                .glassCard()

                VStack(alignment: .leading, spacing: 10) {
                    Text("Endpoints").font(.headline)
                    endpoint("GET", "/v1/models")
                    endpoint("POST", "/v1/chat/completions")
                    endpoint("POST", "/v1/responses")
                    endpoint("POST", "/v1/completions")
                    Text("Example").font(.subheadline.weight(.semibold)).padding(.top, 6)
                    CodeBlock(language: "bash", code: """
                    curl http://localhost:\(server.port)/v1/chat/completions \\
                      -H "Content-Type: application/json" \\\(server.effectiveAPIKey.map { "\n  -H \"Authorization: Bearer \($0)\" \\" } ?? "")
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
            Text(verbatim: "http://localhost:\(String(server.port))\(path)").font(.system(size: 12.5, design: .monospaced)).textSelection(.enabled)
        }
    }
}

struct SettingsView: View {
    @Environment(AppState.self) private var state
    @Environment(UpdateController.self) private var updates
    @AppStorage(UpdateController.autoCheckKey) private var autoUpdateCheck = true
    /// Edited locally; persisted to the Keychain (never UserDefaults).
    @State private var hfToken = ""
    @State private var tokenSaveFailed = false

    var body: some View {
        Form {
            Section("Hugging Face") {
                SecureField("Access token (for gated models)", text: $hfToken)
                    .onSubmit(saveToken)
                    .onChange(of: hfToken) { saveToken() }
                Text("Optional. Create one at huggingface.co/settings/tokens. Stored in your Keychain.")
                    .font(.caption).foregroundStyle(.secondary)
                if HFToken.isFromEnvironment {
                    Text("HF_TOKEN is set in the environment and takes precedence.").font(.caption).foregroundStyle(.orange)
                }
                if tokenSaveFailed {
                    Text("Could not save the token to the Keychain.").font(.caption).foregroundStyle(.red)
                }
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
            Section("Updates") {
                Toggle("Check for updates automatically", isOn: $autoUpdateCheck)
                    .onChange(of: autoUpdateCheck) { updates.autoCheckChanged() }
                Text("Checks GitHub at launch and every 6 hours. Updates are only installed when you choose to, and only if their Ed25519 signature and checksum verify.")
                    .font(.caption).foregroundStyle(.secondary)
                LabeledContent("Version", value: MLXRunnerVersion.current)
            }
            Section("About") {
                LabeledContent("Engine", value: "Native Swift + Metal (no third-party dependencies)")
                LabeledContent("GPU", value: GPU.shared.device.name)
                LabeledContent("Architectures", value: SupportedArchitectures.modelTypes.sorted().joined(separator: ", "))
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 560)
        .onAppear { hfToken = HFToken.stored ?? "" }
    }

    private func saveToken() {
        let t = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard t != (HFToken.stored ?? "") else { return }
        tokenSaveFailed = !HFToken.save(t)
    }
}
