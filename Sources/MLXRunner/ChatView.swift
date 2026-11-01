import AppKit
import MLXEngine
import SwiftUI

struct ChatView: View {
    @Environment(AppState.self) private var state
    let conversationID: UUID

    var body: some View {
        @Bindable var state = state
        let conv = state.conversations.first { $0.id == conversationID }
        VStack(spacing: 0) {
            ModelLoaderBar()
                .padding(.top, 10)
                .padding(.bottom, 6)
            VStack(spacing: 0) {
                if let conv, !conv.messages.isEmpty {
                    MessageList(conversation: conv)
                } else {
                    EmptyChatView(conversationID: conversationID)
                }
                Composer(conversationID: conversationID)
            }
        }
        .frame(minWidth: WindowLayout.chatMin)
        .inspector(isPresented: $state.showInspector) {
            InspectorView(conversationID: conversationID)
                .inspectorColumnWidth(min: 260, ideal: 290, max: 360)
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    state.showInspector.toggle()
                } label: {
                    Label("Settings", systemImage: "slider.horizontal.3")
                }
                .help("Toggle generation settings")
            }
        }
        .navigationTitle(conv?.title ?? "Chat")
    }
}

// MARK: - Model loader (LM Studio–style top bar)

struct ModelLoaderBar: View {
    @Environment(AppState.self) private var state
    @State private var showPicker = false

    var body: some View {
        Button { showPicker.toggle() } label: {
            HStack(spacing: 10) {
                switch state.loadState {
                case .loaded(let info):
                    Image(systemName: "cpu.fill").foregroundStyle(Theme.gradient)
                    Text(info.name).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                    Badge(text: info.quantization, color: Theme.accent)
                    Badge(text: Fmt.bytes(info.parameterBytes), color: .secondary)
                case .loading(let name):
                    ProgressView().controlSize(.small)
                    Text("Loading \(name)…").font(.system(size: 13, weight: .medium))
                case .failed(let err):
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text(err).font(.system(size: 12)).lineLimit(1)
                case .idle:
                    Image(systemName: "sparkles").foregroundStyle(Theme.gradient)
                    Text("Select a model to load").font(.system(size: 13, weight: .medium))
                }
                Image(systemName: "chevron.down").font(.system(size: 10, weight: .bold)).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 9)
            .frame(maxWidth: 560)
            .contentShape(Capsule())
            .glassCapsule(interactive: true)
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showPicker, arrowEdge: .bottom) {
            ModelPicker { showPicker = false }
                .environment(state)
        }
    }
}

struct ModelPicker: View {
    @Environment(AppState.self) private var state
    var dismiss: () -> Void
    @State private var filter = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Filter local models", text: $filter).textFieldStyle(.plain)
            }
            .padding(10)
            Divider()
            ScrollView {
                VStack(spacing: 2) {
                    let models = state.localModels.filter { filter.isEmpty || $0.repo.lowercased().contains(filter.lowercased()) }
                    if models.isEmpty {
                        VStack(spacing: 8) {
                            Text("No models yet").font(.headline)
                            Button("Discover models on Hugging Face") {
                                state.selection = .discover
                                dismiss()
                            }
                        }
                        .padding(30)
                    }
                    ForEach(models) { m in
                        let loaded = state.loadedInfo?.directory == m.directory
                        Button {
                            if !loaded { state.load(m) }
                            dismiss()
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: loaded ? "checkmark.circle.fill" : "shippingbox")
                                    .foregroundStyle(loaded ? AnyShapeStyle(Theme.good) : AnyShapeStyle(Theme.accent))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(m.name).font(.system(size: 13, weight: .medium)).lineLimit(1)
                                    Text(m.author).font(.system(size: 10.5)).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if let t = m.modelType { Badge(text: t, color: m.isSupported ? Theme.accent2 : .orange) }
                                if let b = m.bits { Badge(text: "\(b)-bit", color: Theme.accent) }
                                Text(Fmt.bytes(m.sizeBytes)).font(.system(size: 11).monospacedDigit()).foregroundStyle(.secondary)
                                    .frame(width: 64, alignment: .trailing)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(HoverRowStyle())
                        .disabled(!m.isSupported)
                    }
                }
                .padding(6)
            }
            .frame(maxHeight: 380)
            if state.loadedInfo != nil {
                Divider()
                Button(role: .destructive) { state.eject(); dismiss() } label: {
                    Label("Eject model", systemImage: "eject.fill").frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(10)
            }
        }
        .frame(width: 520)
    }
}

struct HoverRowStyle: ButtonStyle {
    @State private var hover = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.accentColor.opacity(configuration.isPressed ? 0.25 : (hover ? 0.12 : 0))))
            .onHover { hover = $0 }
    }
}

// MARK: - Empty state

struct EmptyChatView: View {
    @Environment(AppState.self) private var state
    let conversationID: UUID
    private let suggestions = [
        ("lightbulb", "Explain quantum entanglement like I'm five"),
        ("chevron.left.forwardslash.chevron.right", "Write a Swift function that reverses a linked list"),
        ("pencil.and.outline", "Draft a friendly email declining a meeting"),
        ("globe.europe.africa", "Plan a 3-day trip to Lisbon"),
    ]

    var body: some View {
        ZStack {
            AuroraBackground()
            VStack(spacing: 22) {
                GradientIcon(systemName: "sparkles", size: 64)
                VStack(spacing: 6) {
                    Text("What can I help with?").font(.system(size: 28, weight: .bold, design: .rounded))
                    Text(state.loadedInfo.map { "Chatting with \($0.name) — fully on-device." }
                         ?? "Load a model from the bar above to start chatting.")
                        .foregroundStyle(.secondary)
                }
                LazyVGrid(columns: [GridItem(.fixed(260)), GridItem(.fixed(260))], spacing: 12) {
                    ForEach(suggestions, id: \.1) { icon, text in
                        Button { state.send(text, in: conversationID) } label: {
                            HStack(alignment: .top, spacing: 10) {
                                Image(systemName: icon).foregroundStyle(Theme.gradient).frame(width: 18)
                                Text(text).font(.system(size: 12.5)).multilineTextAlignment(.leading)
                                Spacer(minLength: 0)
                            }
                            .padding(12)
                            .frame(height: 64, alignment: .topLeading)
                            .glassCard(cornerRadius: 14, interactive: true)
                        }
                        .buttonStyle(.plain)
                        .disabled(state.loadedInfo == nil)
                    }
                }
            }
            .padding(.top, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Messages

struct MessageList: View {
    @Environment(AppState.self) private var state
    let conversation: Conversation

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 22) {
                    Color.clear.frame(height: 8)
                    ForEach(conversation.messages) { m in
                        MessageRow(message: m, conversationID: conversation.id,
                                   isLast: m.id == conversation.messages.last?.id)
                            .id(m.id)
                    }
                    Color.clear.frame(height: 8).id("bottom")
                }
                .frame(maxWidth: 820)
                .padding(.horizontal, 28)
                .frame(maxWidth: .infinity)
            }
            .scrollContentBackground(.hidden)
            .defaultScrollAnchor(.bottom)
            .onChange(of: conversation.messages.last?.content) { _, _ in
                if state.generatingConversation == conversation.id { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: conversation.messages.count) { _, _ in
                withAnimation(.snappy) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }
}

struct MessageRow: View {
    @Environment(AppState.self) private var state
    let message: StoredMessage
    let conversationID: UUID
    let isLast: Bool
    @State private var hovering = false
    @State private var editing = false
    @State private var editText = ""

    var generating: Bool { isLast && state.generatingConversation == conversationID }

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 80)
                VStack(alignment: .trailing, spacing: 6) {
                    if editing {
                        TextEditor(text: $editText)
                            .font(.system(size: 14))
                            .frame(minHeight: 60, maxHeight: 200)
                            .scrollContentBackground(.hidden)
                            .padding(8)
                            .glassCard(cornerRadius: 14)
                        HStack {
                            Button("Cancel") { editing = false }
                            Button("Send") {
                                editing = false
                                state.editAndResend(message.id, newText: editText, in: conversationID)
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    } else {
                        Text(message.content)
                            .font(.system(size: 14))
                            .textSelection(.enabled)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Theme.gradient, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .shadow(color: Theme.accent.opacity(0.25), radius: 10, y: 4)
                        HStack(spacing: 4) {
                            IconButton("pencil", "Edit") { editText = message.content; editing = true }
                            IconButton("doc.on.doc", "Copy") { copy(message.content) }
                        }
                        .opacity(hovering ? 1 : 0)
                    }
                }
            }
            .onHover { hovering = $0 }
        } else {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "sparkles")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 28, height: 28)
                    .background(Theme.gradient, in: Circle())
                    .shadow(color: Theme.accent.opacity(0.4), radius: 6, y: 2)
                VStack(alignment: .leading, spacing: 10) {
                    if let r = message.reasoning, !r.isEmpty {
                        ReasoningView(text: r, seconds: message.reasoningSeconds,
                                      inProgress: generating && message.content.isEmpty)
                    }
                    if let err = message.error {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.system(size: 13))
                            .foregroundStyle(.orange)
                            .padding(10)
                            .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                    }
                    if message.content.isEmpty && message.reasoning == nil && generating {
                        TypingIndicator()
                    } else if !message.content.isEmpty {
                        MarkdownView(text: message.content, selectable: !generating)
                    }
                    if !generating {
                        footer
                    } else if state.liveTokens > 0 {
                        LiveStats(tps: state.liveTokensPerSecond, tokens: state.liveTokens)
                    } else if let s = message.stats, s.promptTokens > 0, message.content.isEmpty, message.reasoning == nil {
                        Text("Processed \(s.promptTokens) prompt tokens in \(String(format: "%.2f", s.timeToFirstToken))s")
                            .font(.system(size: 11)).foregroundStyle(.tertiary)
                    }
                }
                Spacer(minLength: 40)
            }
            .onHover { hovering = $0 }
        }
    }

    @ViewBuilder
    private var footer: some View {
        HStack(spacing: 12) {
            if let s = message.stats, s.generatedTokens > 0 {
                Stat(icon: "bolt.fill", text: String(format: "%.1f tok/s", s.tokensPerSecond), highlight: true)
                Stat(icon: "number", text: "\(s.generatedTokens) tokens")
                Stat(icon: "timer", text: String(format: "%.2fs TTFT", s.timeToFirstToken))
                if s.stopReason != "stop" { Stat(icon: "stop.circle", text: s.stopReason) }
            }
            Spacer()
            HStack(spacing: 4) {
                IconButton("doc.on.doc", "Copy") { copy(message.content) }
                if isLast { IconButton("arrow.clockwise", "Regenerate") { state.regenerate(in: conversationID) } }
            }
            .opacity(hovering ? 1 : 0)
        }
    }

    private func copy(_ s: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
    }
}

struct Stat: View {
    let icon: String
    let text: String
    var highlight = false
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon).font(.system(size: 9.5, weight: .bold))
            Text(text).font(.system(size: 11, weight: .medium).monospacedDigit())
        }
        .lineLimit(1)
        .fixedSize()
        .foregroundStyle(highlight ? AnyShapeStyle(Theme.gradient) : AnyShapeStyle(.tertiary))
    }
}

struct IconButton: View {
    let icon: String
    let help: String
    let action: () -> Void
    init(_ icon: String, _ help: String, action: @escaping () -> Void) {
        self.icon = icon; self.help = help; self.action = action
    }
    var body: some View {
        Button(action: action) {
            Image(systemName: icon).font(.system(size: 11, weight: .medium))
                .frame(width: 24, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(help)
    }
}

struct ReasoningView: View {
    let text: String
    let seconds: Double?
    let inProgress: Bool
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button { withAnimation(.snappy) { expanded.toggle() } } label: {
                HStack(spacing: 6) {
                    Image(systemName: "brain").foregroundStyle(Theme.gradient)
                    if inProgress {
                        Text("Thinking…").shimmering()
                    } else {
                        Text(seconds.map { String(format: "Thought for %.1fs", $0) } ?? "Reasoning")
                    }
                    Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                        .rotationEffect(.degrees(expanded || inProgress ? 90 : 0))
                }
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            if expanded || inProgress {
                let body = Text(inProgress ? String(text.suffix(1200)) : text)
                    .font(.system(size: 12.5))
                    .foregroundStyle(.secondary)
                // Selectable once final (see MarkdownView.selectable).
                Group {
                    if inProgress { body } else { body.textSelection(.enabled) }
                }
                    .padding(.leading, 12)
                    .overlay(alignment: .leading) {
                        Rectangle().fill(Theme.gradient).frame(width: 2).opacity(0.6)
                    }
            }
        }
    }
}

struct TypingIndicator: View {
    @State private var phase = 0.0
    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3) { i in
                Circle().fill(Theme.gradient).frame(width: 7, height: 7)
                    .scaleEffect(0.6 + 0.4 * abs(sin(phase + Double(i) * 0.6)))
            }
        }
        .padding(.vertical, 6)
        .onAppear { withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) { phase = .pi } }
    }
}

struct Shimmer: ViewModifier {
    @State private var x: CGFloat = -1
    func body(content: Content) -> some View {
        content
            .overlay(
                LinearGradient(colors: [.clear, .white.opacity(0.7), .clear], startPoint: .leading, endPoint: .trailing)
                    .scaleEffect(x: 0.5)
                    .offset(x: x * 80)
                    .blendMode(.plusLighter)
                    .mask(content)
            )
            .onAppear { withAnimation(.linear(duration: 1.4).repeatForever(autoreverses: false)) { x = 1 } }
    }
}

extension View {
    func shimmering() -> some View { modifier(Shimmer()) }
}

// MARK: - Composer

struct Composer: View {
    @Environment(AppState.self) private var state
    let conversationID: UUID
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var state = state
        let generating = state.generatingConversation == conversationID
        VStack(spacing: 8) {
            ZStack(alignment: .topLeading) {
                if state.draft.isEmpty {
                    Text(state.loadedInfo == nil ? "Load a model to start chatting…" : "Message \(state.loadedInfo?.name ?? "")…")
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $state.draft)
                    .font(.system(size: 14))
                    .scrollContentBackground(.hidden)
                    .focused($focused)
                    .frame(minHeight: 22, maxHeight: 180)
                    .fixedSize(horizontal: false, vertical: true)
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) || press.modifiers.contains(.option) { return .ignored }
                        submit()
                        return .handled
                    }
            }
            HStack(spacing: 8) {
                if state.loadedInfo?.supportsThinking == true {
                    Toggle(isOn: $state.enableThinking) {
                        Label("Think", systemImage: "brain")
                    }
                    .toggleStyle(.button)
                    .controlSize(.small)
                    .help("Let the model reason before answering")
                }
                Menu {
                    ForEach([0.0, 0.3, 0.7, 1.0, 1.3], id: \.self) { t in
                        Button(String(format: "%.1f", t)) { state.sampling.temperature = Float(t) }
                    }
                } label: {
                    Label(String(format: "%.1f", state.sampling.temperature), systemImage: "thermometer.medium")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .controlSize(.small)
                .help("Temperature")
                Spacer()
                Text("⏎ send · ⇧⏎ newline").font(.system(size: 10.5)).foregroundStyle(.tertiary)
                Button {
                    generating ? state.stopGeneration() : submit()
                } label: {
                    Image(systemName: generating ? "stop.fill" : "arrow.up")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 30, height: 30)
                        .background(canSend || generating ? AnyShapeStyle(Theme.gradient) : AnyShapeStyle(Color.secondary.opacity(0.3)), in: Circle())
                        .shadow(color: Theme.accent.opacity(canSend ? 0.4 : 0), radius: 6, y: 2)
                }
                .buttonStyle(.plain)
                .disabled(!canSend && !generating)
                .keyboardShortcut(generating ? KeyboardShortcut(".", modifiers: .command) : KeyboardShortcut(.return, modifiers: .command))
            }
        }
        .padding(14)
        .glassCard(cornerRadius: 22)
        .frame(maxWidth: 820)
        .padding(.horizontal, 28)
        .padding(.bottom, 18)
        .padding(.top, 6)
        .onAppear { focused = true }
    }

    private var canSend: Bool {
        !state.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && state.loadedInfo != nil && !state.isGenerating
    }

    private func submit() {
        guard canSend else { return }
        let text = state.draft
        state.draft = ""
        state.send(text, in: conversationID)
    }
}

// MARK: - Inspector

struct InspectorView: View {
    @Environment(AppState.self) private var state
    let conversationID: UUID

    var body: some View {
        @Bindable var state = state
        Form {
            Section("System prompt") {
                TextEditor(text: Binding(
                    get: { state.conversations.first { $0.id == conversationID }?.systemPrompt ?? state.systemPrompt },
                    set: { v in
                        if let i = state.index(conversationID) { state.conversations[i].systemPrompt = v; state.scheduleSave() }
                    }))
                    .font(.system(size: 12))
                    .frame(minHeight: 80)
                Button("Use as default for new chats") {
                    if let p = state.conversations.first(where: { $0.id == conversationID })?.systemPrompt { state.systemPrompt = p }
                }
                .controlSize(.small)
            }
            Section("Sampling") {
                SliderRow(title: "Temperature", value: $state.sampling.temperature, range: 0...2, step: 0.05)
                SliderRow(title: "Top P", value: $state.sampling.topP, range: 0.05...1, step: 0.01)
                SliderRow(title: "Min P", value: $state.sampling.minP, range: 0...0.5, step: 0.01)
                Stepper(value: $state.sampling.topK, in: 0...200, step: 5) {
                    LabeledContent("Top K", value: state.sampling.topK == 0 ? "off" : "\(state.sampling.topK)")
                }
                SliderRow(title: "Repeat penalty", value: $state.sampling.repetitionPenalty, range: 1...1.5, step: 0.01)
            }
            Section("Length") {
                Stepper(value: $state.sampling.maxTokens, in: 64...32768, step: 256) {
                    LabeledContent("Max new tokens", value: "\(state.sampling.maxTokens)")
                }
                Picker("Context length", selection: $state.contextLength) {
                    ForEach([2048, 4096, 8192, 16384, 32768, 65536], id: \.self) { Text("\($0)").tag($0) }
                }
                Text("Context length applies the next time a model is loaded.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let info = state.loadedInfo {
                Section("Model") {
                    LabeledContent("Architecture", value: info.modelType)
                    LabeledContent("Quantization", value: info.quantization)
                    LabeledContent("Layers", value: "\(info.layers)")
                    LabeledContent("Hidden size", value: "\(info.hiddenSize)")
                    LabeledContent("Context", value: "\(info.contextLength)")
                    LabeledContent("Prompt format", value: info.chatFormat)
                    LabeledContent("Load time", value: String(format: "%.2fs", info.loadSeconds))
                }
            }
        }
        .formStyle(.grouped)
    }
}

struct SliderRow: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let step: Float
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: "%.2f", value)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range, step: step).controlSize(.small)
        }
    }
}

/// Pulsing live throughput readout shown while tokens stream in.
struct LiveStats: View {
    let tps: Double
    let tokens: Int
    @State private var pulse = false
    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(Theme.gradient).frame(width: 7, height: 7)
                .scaleEffect(pulse ? 1.25 : 0.8)
                .opacity(pulse ? 1 : 0.6)
                .animation(.easeInOut(duration: 0.7).repeatForever(), value: pulse)
            Text(String(format: "%.0f tok/s", tps))
                .font(.system(size: 11, weight: .semibold).monospacedDigit())
                .foregroundStyle(Theme.gradient)
                .contentTransition(.numericText())
            Text("\(tokens) tokens").font(.system(size: 11).monospacedDigit()).foregroundStyle(.tertiary)
        }
        .onAppear { pulse = true }
    }
}
