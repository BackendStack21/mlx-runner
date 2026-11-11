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
            LoadFailureBanner()
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
                    Label("Inspector", systemImage: "slider.horizontal.3")
                }
                .help("Show or hide the inspector (⌃⌘I): system prompt, sampling and model settings")
            }
        }
        .navigationTitle(conv?.title ?? "Chat")
    }
}

// MARK: - Model loader (LM Studio–style top bar)

struct ModelLoaderBar: View {
    @Environment(AppState.self) private var state

    var body: some View {
        @Bindable var state = state
        Button { state.showModelPicker.toggle() } label: {
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
                case .failed:
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text("Couldn't load the model").font(.system(size: 13, weight: .medium)).lineLimit(1)
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
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint("Choose a model to load")
        .help("Choose a model (⌘L)")
        .popover(isPresented: $state.showModelPicker, arrowEdge: .bottom) {
            ModelPicker { state.showModelPicker = false }
                .environment(state)
        }
        .onAppear {
            // Opened from another screen (⌘L, the sidebar footer): present once this bar is in the window.
            if state.showModelPicker {
                state.showModelPicker = false
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(80))
                    state.showModelPicker = true
                }
            }
        }
    }

    private var accessibilityLabel: String {
        switch state.loadState {
        case .loaded(let info): return "Model: \(info.name), \(info.quantization)"
        case .loading(let name): return "Loading \(name)"
        case .failed: return "Couldn't load the model"
        case .idle: return "No model loaded"
        }
    }
}

/// A load that failed while the previous model stayed loaded.
struct LoadFailureBanner: View {
    @Environment(AppState.self) private var state

    var body: some View {
        if let f = state.loadFailure, let info = state.loadedInfo {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Couldn't load \(f.model). Still using \(info.name).")
                        .font(.system(size: 12, weight: .semibold))
                    Text(f.message).font(.system(size: 11)).foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Button { state.loadFailure = nil } label: { Image(systemName: "xmark") }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Dismiss")
            }
            .padding(10)
            .frame(maxWidth: 560)
            .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .padding(.bottom, 6)
        }
    }
}

struct ModelPicker: View {
    @Environment(AppState.self) private var state
    var dismiss: () -> Void
    @State private var filter = ""
    @State private var highlighted = 0
    @FocusState private var filterFocused: Bool

    private var models: [LocalModel] {
        let q = filter.lowercased()
        return state.localModels.filter { q.isEmpty || $0.repo.lowercased().contains(q) }
    }

    var body: some View {
        let models = self.models
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary).accessibilityHidden(true)
                TextField("Filter local models", text: $filter)
                    .textFieldStyle(.plain)
                    .focused($filterFocused)
                    .onSubmit { choose(models, at: highlighted) }
                    .onKeyPress(.downArrow) {
                        highlighted = min(highlighted + 1, max(models.count - 1, 0))
                        return .handled
                    }
                    .onKeyPress(.upArrow) {
                        highlighted = max(highlighted - 1, 0)
                        return .handled
                    }
                    .onChange(of: filter) { highlighted = 0 }
            }
            .padding(10)
            if case .failed(let message) = state.loadState {
                failure(message)
            }
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 2) {
                        if models.isEmpty {
                            VStack(spacing: 8) {
                                Text(state.localModels.isEmpty ? "No models yet" : "No matches").font(.headline)
                                Button("Discover models on Hugging Face") {
                                    state.selection = .discover
                                    dismiss()
                                }
                            }
                            .padding(30)
                        }
                        ForEach(Array(models.enumerated()), id: \.element.id) { i, m in
                            row(m, highlighted: i == highlighted).id(i)
                        }
                    }
                    .padding(6)
                }
                .frame(maxHeight: 380)
                .onChange(of: highlighted) { _, i in proxy.scrollTo(i) }
            }
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
        .onAppear { filterFocused = true }
    }

    private func failure(_ message: String) -> some View {
        Label {
            Text(message).font(.system(size: 11.5)).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 10)
    }

    private func row(_ m: LocalModel, highlighted: Bool) -> some View {
        let loaded = state.loadedInfo?.directory == m.directory
        return Button {
            if !loaded { state.load(m) }
            dismiss()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: loaded ? "checkmark.circle.fill" : "shippingbox")
                    .foregroundStyle(loaded ? AnyShapeStyle(Theme.good) : AnyShapeStyle(Theme.accent))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(m.name).font(.system(size: 13, weight: .medium)).lineLimit(1)
                    Text(m.author).font(.system(size: 10.5)).foregroundStyle(.secondary)
                }
                Spacer()
                if let t = m.modelType { ArchitectureBadge(type: t, supported: m.isSupported) }
                if let b = m.bits { Badge(text: "\(b)-bit", color: Theme.accent) }
                Text(Fmt.bytes(m.sizeBytes)).font(.system(size: 11).monospacedDigit()).foregroundStyle(.secondary)
                    .frame(width: 64, alignment: .trailing)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .contentShape(Rectangle())
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.accentColor.opacity(highlighted ? 0.16 : 0)))
        }
        .buttonStyle(HoverRowStyle())
        .disabled(!m.isSupported)
        .accessibilityLabel("\(m.name), \(Fmt.bytes(m.sizeBytes))\(loaded ? ", loaded" : "")")
        .accessibilityAddTraits(highlighted ? .isSelected : [])
    }

    private func choose(_ models: [LocalModel], at i: Int) {
        guard models.indices.contains(i), models[i].isSupported else { return }
        if state.loadedInfo?.directory != models[i].directory { state.load(models[i]) }
        dismiss()
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
                VStack(spacing: 8) {
                    Text("What can I help with?").font(.system(size: 28, weight: .bold, design: .rounded))
                        .accessibilityAddTraits(.isHeader)
                    status
                }
                LazyVGrid(columns: [GridItem(.fixed(260)), GridItem(.fixed(260))], spacing: 12) {
                    ForEach(suggestions, id: \.1) { icon, text in
                        Button { state.send(text, in: conversationID) } label: {
                            HStack(alignment: .top, spacing: 10) {
                                Image(systemName: icon).foregroundStyle(Theme.gradient).frame(width: 18)
                                    .accessibilityHidden(true)
                                Text(text).font(.system(size: 12.5)).multilineTextAlignment(.leading)
                                Spacer(minLength: 0)
                            }
                            .padding(12)
                            .frame(height: 64, alignment: .topLeading)
                            .glassCard(cornerRadius: 14, interactive: true)
                        }
                        .buttonStyle(.plain)
                        .disabled(!state.canStartReply || state.pendingSend != nil)
                    }
                }
            }
            .padding(.top, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// What is loaded, and the one action that gets a model ready when nothing is.
    @ViewBuilder
    private var status: some View {
        switch state.loadState {
        case .loaded(let info):
            Text("Chatting with \(info.name) — fully on-device.").foregroundStyle(.secondary)
        case .loading(let name):
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Loading \(name)…").foregroundStyle(.secondary)
            }
        case .failed(let message):
            VStack(spacing: 8) {
                Text(message).font(.system(size: 12)).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center).textSelection(.enabled)
                    .frame(maxWidth: 520)
                Button("Choose a Model…") { state.openModelPicker() }
            }
        case .idle:
            if let m = state.lastModel {
                VStack(spacing: 8) {
                    Button { state.load(m) } label: {
                        Label("Load \(m.name) · \(Fmt.bytes(m.sizeBytes))", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    Text("or just send a message: it loads first. Choose another model above.")
                        .font(.system(size: 11.5)).foregroundStyle(.secondary)
                }
            } else if state.modelsScanned && state.localModels.isEmpty {
                VStack(spacing: 8) {
                    Text("Download a model to start chatting.").foregroundStyle(.secondary)
                    Button("Discover Models") { state.selection = .discover }.buttonStyle(.borderedProminent)
                }
            } else if state.modelsScanned {
                VStack(spacing: 8) {
                    Text("Load a model to start chatting.").foregroundStyle(.secondary)
                    Button("Choose a Model…") { state.openModelPicker() }.buttonStyle(.borderedProminent)
                }
            }
        }
    }
}

// MARK: - Messages

/// Scroll position facts used to tell a user scroll from content growth.
private struct ScrollMetrics: Equatable {
    var offset: CGFloat
    var contentHeight: CGFloat
    var visibleHeight: CGFloat
    var fromBottom: CGFloat { contentHeight - (offset + visibleHeight) }
}

struct MessageList: View {
    @Environment(AppState.self) private var state
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let conversation: Conversation
    /// Follow new text at the bottom. Only the user turns it off (by scrolling up), so a reader is never
    /// pulled back down while a reply streams; scrolling back to the bottom or sending turns it on again.
    @State private var following = true

    var body: some View {
        let firstInContext = Self.firstMessageInContext(conversation.messages)
        ScrollViewReader { proxy in
            ScrollView {
                // A plain stack: a lazy one could be left scrolled past its (stale) content when a
                // reply's layout changed at the end of generation, showing an empty transcript.
                VStack(spacing: 22) {
                    Color.clear.frame(height: 8)
                    ForEach(conversation.messages) { m in
                        if m.id == firstInContext {
                            ContextCutDivider()
                        }
                        MessageRow(message: m, conversationID: conversation.id,
                                   isLast: m.id == conversation.messages.last?.id)
                            .id(m.id)
                    }
                    if state.pendingSend == conversation.id {
                        PendingLoadRow()
                    }
                    Color.clear.frame(height: 8).id("bottom")
                }
                .frame(maxWidth: 820)
                .padding(.horizontal, 28)
                .frame(maxWidth: .infinity)
            }
            .scrollContentBackground(.hidden)
            // Opens at the bottom; while following, growth (the streaming reply) keeps the bottom in view
            // within the same layout pass, so no scroll is issued per streamed update. Not following, the top
            // anchor keeps what the reader is looking at in place.
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .defaultScrollAnchor(following ? .bottom : .top, for: .sizeChanges)
            .onScrollGeometryChange(for: ScrollMetrics.self) { g in
                ScrollMetrics(offset: g.contentOffset.y, contentHeight: g.contentSize.height, visibleHeight: g.containerSize.height)
            } action: { old, new in
                // Same content and viewport: the offset moved because the user scrolled (wheel, trackpad,
                // scroller, keys, VoiceOver). Growth and window resizes change a height and are ignored.
                guard abs(new.contentHeight - old.contentHeight) < 0.5, abs(new.visibleHeight - old.visibleHeight) < 0.5 else { return }
                if new.offset < old.offset - 0.5, new.fromBottom > 4 {
                    if following { following = false }
                } else if new.offset > old.offset, new.fromBottom < 40, !following {
                    following = true
                }
            }
            .onChange(of: conversation.messages.last?.content) { _, _ in
                if following { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: conversation.messages.count) { _, _ in
                following = true
                withAnimation(reduceMotion ? nil : .snappy) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .overlay(alignment: .bottom) {
                if !following {
                    JumpToLatestButton(conversationID: conversation.id) {
                        following = true
                        withAnimation(reduceMotion ? nil : .snappy) { proxy.scrollTo("bottom", anchor: .bottom) }
                    }
                    .padding(.bottom, 10)
                }
            }
        }
    }

    /// The first message the model still saw in the latest request, when older ones were left out to fit
    /// the context window (`GenerationStats.droppedMessages` counts the turns sent before it, errors excluded).
    static func firstMessageInContext(_ messages: [StoredMessage]) -> UUID? {
        guard let replyIndex = messages.lastIndex(where: { $0.role == .assistant && $0.stats != nil }),
              let dropped = messages[replyIndex].stats?.droppedMessages, dropped > 0 else { return nil }
        let sent = messages[..<replyIndex].filter { $0.error == nil }
        return sent.indices.contains(dropped) ? sent[dropped].id : nil
    }
}

/// Marks where the model's view of the chat begins.
struct ContextCutDivider: View {
    @Environment(AppState.self) private var state

    var body: some View {
        Button { state.showInspector = true } label: {
            HStack(spacing: 10) {
                Rectangle().fill(.orange.opacity(0.5)).frame(height: 1)
                Label("Earlier messages are outside the model's context", systemImage: "arrow.up.to.line")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.orange)
                    .fixedSize()
                Rectangle().fill(.orange.opacity(0.5)).frame(height: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("The model only sees the messages below this line. A longer context length (inspector) keeps more of the chat.")
        .accessibilityHint("Opens the inspector, where the context length is set")
    }
}

/// The reply slot while the model a message was sent to is still loading.
struct PendingLoadRow: View {
    @Environment(AppState.self) private var state

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            AssistantAvatar()
            VStack(alignment: .leading, spacing: 6) {
                TypingIndicator()
                if case .loading(let name) = state.loadState {
                    Text("Loading \(name)…").font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 40)
        }
    }
}

/// Shown when the reader has scrolled away from the bottom.
struct JumpToLatestButton: View {
    @Environment(AppState.self) private var state
    let conversationID: UUID
    let action: () -> Void

    var body: some View {
        let live = state.generatingConversation == conversationID
        Button(action: action) {
            HStack(spacing: 6) {
                if live { Circle().fill(Theme.accent).frame(width: 6, height: 6) }
                Image(systemName: "arrow.down").font(.system(size: 12, weight: .bold))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .contentShape(Capsule())
            // Not interactive glass: that adds hover tracking, and this button is on screen while text streams.
            .glassCapsule()
        }
        .buttonStyle(.plain)
        .help("Jump to latest")
        .accessibilityLabel(live ? "Jump to latest, reply in progress" : "Jump to latest")
    }
}

struct AssistantAvatar: View {
    var body: some View {
        Image(systemName: "sparkles")
            .font(.system(size: 13, weight: .bold))
            .foregroundStyle(.white)
            .frame(width: 28, height: 28)
            .background(Theme.gradient, in: Circle())
            .shadow(color: Theme.accent.opacity(0.4), radius: 6, y: 2)
    }
}

struct MessageRow: View {
    @Environment(AppState.self) private var state
    let message: StoredMessage
    let conversationID: UUID
    let isLast: Bool
    @State private var hovering = false
    /// The action buttons also track the pointer themselves, so they stay visible while it is on
    /// them even if the row reports that its hover ended.
    @State private var hoveringButtons = false
    @State private var editText = ""

    var generating: Bool { isLast && state.generatingConversation == conversationID }
    private var editing: Bool { state.editingMessage == message.id }
    /// Row actions: always on the last message, on hover elsewhere.
    private var actionsVisible: Bool { isLast || hovering || hoveringButtons }

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
                            .accessibilityLabel("Edit message")
                            .onKeyPress(.escape) {
                                state.editingMessage = nil
                                return .handled
                            }
                        HStack {
                            Button("Cancel") { state.editingMessage = nil }
                            Button("Send") {
                                state.editingMessage = nil
                                state.editAndResend(message.id, newText: editText, in: conversationID)
                            }
                            .buttonStyle(.borderedProminent)
                            .keyboardShortcut(.return, modifiers: .command)
                            .disabled(editText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    } else {
                        Text(message.content)
                            .font(.system(size: 14))
                            .textSelection(.enabled)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Theme.fillGradient, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .shadow(color: Theme.accent.opacity(0.25), radius: 10, y: 4)
                            .accessibilityLabel("You: \(message.content)")
                            .accessibilityAddTraits(.isHeader)
                            .accessibilityActions {
                                Button("Edit") { startEditing() }
                                Button("Copy") { copy(message.content) }
                            }
                        HStack(spacing: 4) {
                            IconButton("pencil", "Edit") { startEditing() }
                            CopyButton(text: message.content)
                        }
                        .onHover { hoveringButtons = $0 }
                        .onDisappear { hoveringButtons = false }
                        .opacity(actionsVisible ? 1 : 0)
                        .disabled(!actionsVisible || state.isGenerating)
                        .accessibilityHidden(!actionsVisible)
                    }
                }
            }
            .onHover { hovering = $0 }
        } else {
            // While this reply streams, its text lives in the state's live properties.
            let live = generating && state.liveReplyID == message.id
            let content = live ? state.liveContent : message.content
            let reasoning = live ? state.liveReasoning : message.reasoning
            let reasoningSeconds = live ? state.liveReasoningSeconds : message.reasoningSeconds
            HStack(alignment: .top, spacing: 12) {
                AssistantAvatar()
                    .accessibilityLabel("Assistant")
                    .accessibilityAddTraits(.isHeader)
                    .accessibilityActions {
                        if !generating {
                            Button("Copy") { copy(message.content) }
                            if isLast { Button("Regenerate") { state.regenerate(in: conversationID) } }
                        }
                    }
                VStack(alignment: .leading, spacing: 10) {
                    if let r = reasoning, !r.isEmpty {
                        ReasoningView(text: r, seconds: reasoningSeconds,
                                      inProgress: generating && content.isEmpty)
                    }
                    if let err = message.error {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.system(size: 13))
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                            .padding(10)
                            .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                    }
                    if content.isEmpty && reasoning == nil && generating {
                        TypingIndicator()
                    } else if !content.isEmpty {
                        if generating {
                            // Finished blocks keep their view; only the block being written re-renders.
                            let parts = MarkdownParser.splitStreaming(content)
                            if !parts.head.isEmpty { MarkdownView(text: parts.head, selectable: false) }
                            if !parts.tail.isEmpty { MarkdownView(text: parts.tail, selectable: false) }
                        } else {
                            MarkdownView(text: content)
                        }
                    }
                    if !generating {
                        if let s = message.stats {
                            StopNotice(stats: s, message: message, conversationID: conversationID, isLast: isLast)
                        }
                        footer
                    } else if state.liveTokens > 0 {
                        LiveStats(tps: state.liveTokensPerSecond, tokens: state.liveTokens)
                    } else if let s = message.stats, s.promptTokens > 0, content.isEmpty, reasoning == nil {
                        Text("Processed \(s.promptTokens) prompt tokens in \(String(format: "%.2f", s.timeToFirstToken))s")
                            .font(.system(size: 11)).foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 40)
            }
            // No hover tracking while the reply streams: a tracking area makes AppKit
            // re-hit-test the whole window on every update.
            .onHover { if !generating { hovering = $0 } }
        }
    }

    @ViewBuilder
    private var footer: some View {
        HStack(spacing: 12) {
            if let s = message.stats, s.generatedTokens > 0 {
                HStack(spacing: 12) {
                    Stat(icon: "bolt.fill", text: String(format: "%.1f tok/s", s.tokensPerSecond), highlight: true)
                    Stat(icon: "number", text: "\(s.generatedTokens) tokens")
                    Stat(icon: "timer", text: String(format: "%.2fs TTFT", s.timeToFirstToken))
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(String(format: "%.1f tokens per second, %d tokens, first token after %.2f seconds",
                                           s.tokensPerSecond, s.generatedTokens, s.timeToFirstToken))
            }
            Spacer()
            HStack(spacing: 4) {
                CopyButton(text: message.content)
                if isLast {
                    IconButton("arrow.clockwise", "Regenerate") { state.regenerate(in: conversationID) }
                        .disabled(state.isGenerating || state.pendingSend != nil)
                }
            }
            .onHover { hoveringButtons = $0 }
            .opacity(actionsVisible ? 1 : 0)
            .disabled(!actionsVisible)
            .accessibilityHidden(!actionsVisible)
        }
    }

    private func startEditing() {
        editText = message.content
        state.editingMessage = message.id
    }

    private func copy(_ s: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        state.announce("Copied")
    }
}

/// Why a reply ended early, in words, with the next step.
struct StopNotice: View {
    @Environment(AppState.self) private var state
    let stats: GenerationStats
    let message: StoredMessage
    let conversationID: UUID
    let isLast: Bool

    var body: some View {
        switch kind {
        case .replyLimit(let thinking):
            notice(thinking ? "Ran out of tokens while thinking" : "Reached the reply limit (\(stats.generatedTokens.formatted()) tokens)",
                   icon: "text.badge.xmark") {
                let limits = [8192, 16384, 32768].filter { $0 > state.sampling.maxTokens }
                if !limits.isEmpty {
                    Menu("Raise Limit") {
                        ForEach(limits, id: \.self) { n in
                            Button("\(n.formatted()) tokens") { state.sampling.maxTokens = n }
                        }
                    }
                    .fixedSize()
                    .help("Sets the reply limit (Max new tokens in the inspector)")
                }
                Button("Regenerate") { state.regenerate(in: conversationID) }
                    .disabled(state.isGenerating || state.pendingSend != nil)
            }
        case .contextFull:
            notice("The context window is full", icon: "rectangle.compress.vertical") {
                let current = state.loadedInfo?.contextLength ?? state.contextLength
                let larger = [16384, 32768, 65536].filter { $0 > current }
                if !larger.isEmpty, state.loadedInfo != nil {
                    Menu("Raise Context") {
                        ForEach(larger, id: \.self) { n in
                            Button("\(n.formatted()) tokens (reloads the model)") {
                                state.contextLength = n
                                state.reloadModel()
                            }
                        }
                    }
                    .fixedSize()
                }
                Button("New Chat") { state.newChat() }
            }
        case .stopped:
            notice("Stopped", icon: "stop.circle") { EmptyView() }
        case nil:
            EmptyView()
        }
    }

    private enum Kind { case replyLimit(thinking: Bool), contextFull, stopped }

    private var kind: Kind? {
        switch stats.stopReason {
        case "length":
            // The engine caps a reply at whatever room the context has left: that is a full context too.
            if let ctx = state.loadedInfo?.contextLength, stats.promptTokens + stats.generatedTokens >= ctx - 1 { return .contextFull }
            return .replyLimit(thinking: message.content.isEmpty && !(message.reasoning ?? "").isEmpty)
        case "context": return .contextFull
        case "cancelled": return .stopped
        default: return nil
        }
    }

    private func notice<Actions: View>(_ text: String, icon: String, @ViewBuilder actions: () -> Actions) -> some View {
        HStack(spacing: 10) {
            Label(text, systemImage: icon)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(.secondary)
            if isLast {
                actions()
                    .controlSize(.small)
            }
        }
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
        .foregroundStyle(highlight ? AnyShapeStyle(Theme.gradient) : AnyShapeStyle(.secondary))
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
        .accessibilityLabel(help)
    }
}

/// Copy with a brief checkmark (and a VoiceOver announcement) as confirmation.
struct CopyButton: View {
    @Environment(AppState.self) private var state
    let text: String
    @State private var copied = false

    var body: some View {
        IconButton(copied ? "checkmark" : "doc.on.doc", copied ? "Copied" : "Copy") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            state.announce("Copied")
            copied = true
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(1.5))
                copied = false
            }
        }
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
            .accessibilityValue(expanded || inProgress ? "expanded" : "collapsed")
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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase = 0.0
    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3) { i in
                Circle().fill(Theme.gradient).frame(width: 7, height: 7)
                    .scaleEffect(reduceMotion ? 0.8 : 0.6 + 0.4 * abs(sin(phase + Double(i) * 0.6)))
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Generating")
        // Stepped, see LiveStats. Not started at all with Reduce Motion.
        .task(id: reduceMotion) {
            guard !reduceMotion else { return }
            while !Task.isCancelled { try? await Task.sleep(nanoseconds: 120_000_000); phase += 0.45 }
        }
    }
}

/// A slow blink. (A moving-gradient shimmer was a repeat-forever animation: it rendered every
/// frame and, on this window, dragged an AppKit layout and hit-test pass with each one.)
struct Shimmer: ViewModifier {
    @State private var on = true
    func body(content: Content) -> some View {
        content
            .opacity(on ? 1 : 0.55)
            .task { while !Task.isCancelled { try? await Task.sleep(nanoseconds: 600_000_000); on.toggle() } }
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
        let draft = Binding(get: { state.drafts[conversationID] ?? "" },
                            set: { state.drafts[conversationID] = $0.isEmpty ? nil : $0 })
        VStack(spacing: 8) {
            ZStack(alignment: .topLeading) {
                if draft.wrappedValue.isEmpty {
                    Text(placeholder)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
                TextEditor(text: draft)
                    .font(.system(size: 14))
                    .scrollContentBackground(.hidden)
                    .focused($focused)
                    .frame(minHeight: 22, maxHeight: 180)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityLabel("Message")
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
                .accessibilityLabel("Temperature")
                .accessibilityValue(String(format: "%.1f", state.sampling.temperature))
                ContextMeter(conversationID: conversationID)
                Spacer()
                Text("⏎ send · ⇧⏎ newline").font(.system(size: 10.5)).foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
                Button {
                    generating ? state.stopGeneration() : submit()
                } label: {
                    Image(systemName: generating ? "stop.fill" : "arrow.up")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 30, height: 30)
                        .background(canSend || generating ? AnyShapeStyle(Theme.fillGradient) : AnyShapeStyle(Color.secondary.opacity(0.3)), in: Circle())
                        .shadow(color: Theme.accent.opacity(canSend ? 0.4 : 0), radius: 6, y: 2)
                }
                .buttonStyle(.plain)
                .disabled(!canSend && !generating)
                // ⌘. lives in the Model menu (Stop Generating); ⌘⏎ belongs to the message being edited, if any.
                .keyboardShortcut(generating || state.editingMessage != nil ? nil : KeyboardShortcut(.return, modifiers: .command))
                .accessibilityLabel(generating ? "Stop generating" : "Send")
                .help(generating ? "Stop generating (⌘.)" : "Send (⏎)")
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

    private var placeholder: String {
        switch state.loadState {
        case .loaded(let info): return "Message \(info.name)…"
        case .loading(let name): return "Message \(name) (loading)…"
        default:
            if let m = state.lastModel { return "Message \(m.name) (loads when you send)…" }
            if state.modelsScanned && state.localModels.isEmpty { return "Download a model to start chatting…" }
            return "Choose a model above to start chatting…"
        }
    }

    private var canSend: Bool {
        !(state.drafts[conversationID] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && state.canStartReply && !state.isGenerating && state.pendingSend == nil
    }

    private func submit() {
        guard canSend else { return }
        let text = state.drafts[conversationID] ?? ""
        state.drafts[conversationID] = nil
        state.send(text, in: conversationID)
    }
}

/// How full the model's context window was after the last reply. A leaf view: it reads `conversations`,
/// so it alone re-renders on a conversation write (the composer and its text view don't).
struct ContextMeter: View {
    @Environment(AppState.self) private var state
    let conversationID: UUID

    var body: some View {
        if let info = state.loadedInfo,
           let conv = state.conversations.first(where: { $0.id == conversationID }),
           let used = Self.tokensInContext(conv.messages, model: info.name) {
            let total = info.contextLength
            let f = min(1, Double(used) / Double(max(total, 1)))
            let color: Color = f >= 0.95 ? .red : (f >= 0.75 ? .orange : .secondary)
            HStack(spacing: 5) {
                if f >= 0.95 {
                    Image(systemName: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(.red)
                } else {
                    ZStack {
                        Circle().stroke(Color.secondary.opacity(0.25), lineWidth: 2)
                        Circle().trim(from: 0, to: f).stroke(color, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                    }
                    .frame(width: 11, height: 11)
                }
                Text("\(Self.short(used)) / \(Self.short(total))")
                    .font(.system(size: 10.5, weight: .medium).monospacedDigit())
                    .foregroundStyle(f >= 0.75 ? AnyShapeStyle(color) : AnyShapeStyle(.secondary))
            }
            .help("Context: \(used.formatted()) of \(total.formatted()) tokens after the last reply, system prompt included. When it fills, the oldest messages are left out.")
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Context")
            .accessibilityValue("\(used.formatted()) of \(total.formatted()) tokens")
        }
    }

    /// Prompt tokens of the latest request plus the part of its reply that is resent next turn (the answer,
    /// not the reasoning). Nil before any reply, or when the last reply came from another model.
    static func tokensInContext(_ messages: [StoredMessage], model: String) -> Int? {
        guard let m = messages.last(where: { $0.role == .assistant && ($0.stats?.promptTokens ?? 0) > 0 }),
              let s = m.stats, m.model == nil || m.model == model else { return nil }
        let answer = m.content.utf8.count
        let thought = m.reasoning?.utf8.count ?? 0
        let share = answer + thought > 0 ? Double(answer) / Double(answer + thought) : 0
        return s.promptTokens + Int((Double(s.generatedTokens) * share).rounded())
    }

    static func short(_ n: Int) -> String {
        n >= 1000 ? String(format: n >= 10_000 ? "%.0fK" : "%.1fK", Double(n) / 1000) : "\(n)"
    }
}

// MARK: - Inspector

struct InspectorView: View {
    @Environment(AppState.self) private var state
    let conversationID: UUID

    var body: some View {
        @Bindable var state = state
        Form {
            if state.needsReload, let info = state.loadedInfo {
                Section {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "arrow.clockwise.circle.fill").foregroundStyle(Theme.accent)
                            .accessibilityHidden(true)
                        Text("Reload \(info.name) to apply the new settings.")
                            .font(.system(size: 12))
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 4)
                        Button("Reload") { state.reloadModel() }
                            .controlSize(.small)
                            .disabled(state.isGenerating)
                    }
                }
            }
            Section("System prompt") {
                TextEditor(text: Binding(
                    get: { state.conversations.first { $0.id == conversationID }?.systemPrompt ?? state.systemPrompt },
                    set: { v in
                        if let i = state.index(conversationID) { state.conversations[i].systemPrompt = v; state.scheduleSave() }
                    }))
                    .font(.system(size: 12))
                    .frame(minHeight: 80)
                    .accessibilityLabel("System prompt")
                Button("Use as default for new chats") {
                    if let p = state.conversations.first(where: { $0.id == conversationID })?.systemPrompt { state.systemPrompt = p }
                }
                .controlSize(.small)
            }
            Section {
                SliderRow(title: "Temperature", value: $state.sampling.temperature, range: 0...2, step: 0.05)
                SliderRow(title: "Top P", value: $state.sampling.topP, range: 0.05...1, step: 0.01)
                SliderRow(title: "Min P", value: $state.sampling.minP, range: 0...0.5, step: 0.01)
                Stepper(value: $state.sampling.topK, in: 0...200, step: 5) {
                    LabeledContent("Top K", value: state.sampling.topK == 0 ? "off" : "\(state.sampling.topK)")
                }
                SliderRow(title: "Repeat penalty", value: $state.sampling.repetitionPenalty, range: 1...1.5, step: 0.01)
                Button(state.loadedInfo == nil ? "Reset to Defaults" : "Reset to Model Defaults") {
                    state.resetSamplingToModelDefaults()
                }
                .controlSize(.small)
                .help("The loaded model's own sampling settings (generation_config.json), else the engine defaults")
            } header: {
                Text("Sampling")
            }
            Section("Length") {
                Stepper(value: $state.sampling.maxTokens, in: 64...32768, step: 256) {
                    LabeledContent("Max new tokens", value: "\(state.sampling.maxTokens)")
                }
                Picker("Context length", selection: $state.contextLength) {
                    ForEach([2048, 4096, 8192, 16384, 32768, 65536], id: \.self) { Text("\($0)").tag($0) }
                }
                Picker("Draft model", selection: $state.draftModel) {
                    Text("Off").tag("")
                    Text("Auto").tag("auto")
                    ForEach(draftCandidates) { Text($0.name).tag($0.directory.path) }
                }
                Text("Context length and draft model apply when the model is (re)loaded. A small model of the same family drafts tokens for the loaded one to verify: the output is unchanged.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Long context") {
                Toggle("Sparse attention", isOn: $state.sparseAttention)
                Picker("Long-range keys", selection: $state.sparseKeys) {
                    ForEach(Self.sparseKeyChoices(including: state.sparseKeys), id: \.self) { Text("\($0)").tag($0) }
                }
                .disabled(!state.sparseAttention)
                Picker("Key selection", selection: $state.sparsePerHead) {
                    Text("Per KV head").tag(true)
                    Text("Shared").tag(false)
                }
                .disabled(!state.sparseAttention)
                Text(sparseCaption)
                    .font(.caption).foregroundStyle(.secondary)
                // Opened through SafeLinkHandling, which asks first for hosts other than huggingface.co.
                Link("Based on Latent Sparse Attention (arXiv 2610.42147)", destination: Self.sparsePaper)
                    .font(.caption)
            }
            Section("Sieve") {
                Toggle("Sieve", isOn: $state.sieve)
                if let status = state.sieveFitStatus {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Fitting: \(status)").font(.caption).foregroundStyle(.secondary)
                    }
                } else if let info = state.loadedInfo {
                    Button(info.sieveHead == nil ? "Fit for \(info.name)" : "Fit again") { state.fitSieve() }
                        .disabled(state.isGenerating)
                        .help("Unloads the model, measures it for a minute or more with the GPU to itself, then loads it again with Sieve on")
                }
                if let e = state.sieveFitError {
                    Text("Couldn't fit: \(e)").font(.caption).foregroundStyle(.red)
                }
                Text("Reads a small sketch of the output layer to find the likely next tokens, and computes only those exactly: faster generation, with a small change in output. Each model has to be fitted once. Applies when the model is (re)loaded.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let info = state.loadedInfo {
                Section("Model") {
                    LabeledContent("Architecture", value: info.modelType)
                    LabeledContent("Quantization", value: info.quantization)
                    LabeledContent("Layers", value: "\(info.layers)")
                    LabeledContent("Hidden size", value: "\(info.hiddenSize)")
                    LabeledContent("Context", value: "\(info.contextLength)")
                    if let d = info.draftName { LabeledContent("Draft model", value: d) }
                    LabeledContent("Sparse attention", value: Self.sparseStatus(info))
                    LabeledContent("Sieve", value: Self.sieveStatus(info))
                    LabeledContent("Prompt format", value: info.chatFormat)
                    LabeledContent("Load time", value: String(format: "%.2fs", info.loadSeconds))
                }
            }
        }
        .formStyle(.grouped)
    }

    /// Kubicek et al., "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention".
    static let sparsePaper = URL(string: "https://arxiv.org/abs/2610.42147")!

    /// Long-range key budgets offered (plus a saved value that is not one of them).
    static func sparseKeyChoices(including current: Int) -> [Int] {
        Array(Set([1024, 2048, 4096, 8192, current])).sorted()
    }

    private var sparseCaption: String {
        guard let s = state.sparseConfig else {
            return "Every token attends to the whole context. Applies when the model is (re)loaded."
        }
        return "From \(s.minContext) tokens of context on, each new token attends to the first and latest tokens and the \(s.topK) earlier ones that score highest, not the whole context: faster at long contexts, with a small change in output. Applies when the model is (re)loaded."
    }

    static func sieveStatus(_ info: LoadedModelInfo) -> String {
        if let h = info.sieveHead {
            return String(format: "On: %d output rows per token, argmax kept in %.1f%% of test states", h.rows, 100 * h.agreement)
        }
        return info.sieveNote.map { "Off: \($0)" } ?? "Off"
    }

    static func sparseStatus(_ info: LoadedModelInfo) -> String {
        if info.sparseLayers > 0 { return "\(info.sparseLayers) of \(info.layers) layers, from \(info.sparseMinContext) tokens" }
        return info.sparseNote.map { "Off: \($0)" } ?? "Off"
    }

    /// Installed models that can draft for the loaded one (same family, at most a quarter of its size;
    /// none for a model with a recurrent state, Qwen 3.5, which cannot drop rejected drafts).
    private var draftCandidates: [LocalModel] {
        let loaded = state.localModels.first { $0.directory == state.loadedInfo?.directory }
        return state.localModels.filter { m in
            guard m.isSupported else { return false }
            if m.directory.path == state.draftModel { return true }
            guard let loaded, !SupportedArchitectures.hasRecurrentState(loaded.modelType) else { return false }
            return m.modelType == loaded.modelType && m.sizeBytes * 4 <= loaded.sizeBytes
        }
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
            .accessibilityHidden(true)
            Slider(value: $value, in: range, step: step).controlSize(.small)
                .accessibilityLabel(title)
                .accessibilityValue(String(format: "%.2f", value))
        }
    }
}

/// Pulsing live throughput readout shown while tokens stream in.
struct LiveStats: View {
    let tps: Double
    let tokens: Int
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse = false
    var body: some View {
        // Stepped, not animated: a repeat-forever animation renders every frame, and on this
        // window each frame costs an AppKit layout and hit-test pass (the chat froze for it).
        HStack(spacing: 8) {
            Circle().fill(Theme.gradient).frame(width: 7, height: 7)
                .opacity(pulse ? 1 : 0.5)
            Text(String(format: "%.0f tok/s", tps))
                .font(.system(size: 11, weight: .semibold).monospacedDigit())
                .foregroundStyle(Theme.gradient)
                .contentTransition(.numericText())
            Text("\(tokens) tokens").font(.system(size: 11).monospacedDigit()).foregroundStyle(.secondary)
        }
        // Changes several times a second: VoiceOver hears "Reply finished" instead.
        .accessibilityHidden(true)
        .task(id: reduceMotion) {
            guard !reduceMotion else { return }
            while !Task.isCancelled { try? await Task.sleep(nanoseconds: 700_000_000); pulse.toggle() }
        }
    }
}
