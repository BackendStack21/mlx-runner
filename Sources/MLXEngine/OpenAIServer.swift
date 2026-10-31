import Foundation
import Network
import Security

/// A small OpenAI-compatible HTTP server backed by an `InferenceEngine` (by default `.shared`).
///
/// Endpoints:
///   GET  /                      health text
///   GET  /v1/models             loaded model (OpenAI list format)
///   GET  /v1/models/{id}        one model
///   POST /v1/chat/completions   Chat Completions (streaming via SSE or not)
///   POST /v1/responses          Responses API (typed SSE events or a `response` object)
///   POST /v1/completions        legacy text completions
///
/// Security model:
///   - Loopback only by default. The `Host` header must name localhost / 127.0.0.1 / [::1]
///     (DNS-rebinding defence) unless LAN mode is on.
///   - Requests carrying an `Origin` header are refused unless the origin matches
///     `allowedOrigins`; CORS headers are only sent back to allowed origins.
///   - POST bodies must be `application/json` (blocks CORS "simple" form/text posts).
///   - Optional bearer API key, mandatory in LAN mode. It guards every endpoint except
///     CORS preflights.
///   - Header/body/write deadlines, a connection cap, a generation-queue cap and body,
///     message-count and output-token limits.
///
/// Unsupported OpenAI features are rejected with 400 rather than silently ignored when ignoring
/// them would change the meaning of the request: tools/functions (unless `tool_choice` is
/// "none"), `tool`/`function` messages, non-text content parts, `response_format` /
/// `text.format` other than text, `n` > 1, `logprobs`, non-empty `logit_bias`, audio output,
/// `previous_response_id`, `conversation`, `background`, prompt templates and `echo`/`suffix`/
/// `best_of` on legacy completions. Accepted but ignored: `model` (the loaded model always
/// answers), `frequency_penalty`/`presence_penalty` (range-checked; the engine's multiplicative
/// repetition penalty is not equivalent), `store` (nothing is ever stored; responses report
/// `store: false`), `metadata` (echoed back), `user`, `safety_identifier`, `prompt_cache_key`,
/// `service_tier`, `parallel_tool_calls`, `include`, `truncation` (history is always trimmed
/// to fit), `stream_options` other than `include_usage`.
///
/// Reasoning: when a model's reply starts with `<think>…</think>` (Qwen3 and similar), Chat
/// Completions return it as `reasoning_content` (message and delta), and the Responses API
/// emits it as a `reasoning` output item with `reasoning_text` content (the format OpenAI uses
/// for open-weight models), followed by the clean assistant `message`. `reasoning_effort:
/// "none"` (Chat) or `reasoning: {effort: "none"}` (Responses) turns thinking off for models
/// whose template supports it, as do `enable_thinking: false` and
/// `chat_template_kwargs.enable_thinking: false`.
///
/// Generation requests run one at a time in FIFO order; at most `maxQueuedRequests` wait.
public final class OpenAIServer: @unchecked Sendable {
    /// Browser origins allowed by default: pages served from this machine.
    public static let defaultAllowedOrigins = ["http://localhost:*", "http://127.0.0.1:*", "http://[::1]:*"]

    public let port: UInt16
    // Configuration: set these before `start()`; they are snapshotted when the server starts.
    /// When false (default) the server only accepts connections on 127.0.0.1.
    public var allowLAN: Bool
    /// When set, every request (except CORS preflights) needs `Authorization: Bearer <key>`.
    /// Required in LAN mode; if absent there, the `MLXR_API_KEY` environment variable is used.
    public var apiKey: String?
    /// Allowed `Origin` values. Exact origins, or `scheme://host:*` for any port, or `*` for any.
    public var allowedOrigins: [String] = OpenAIServer.defaultAllowedOrigins
    public var maxConnections = 32
    public var maxQueuedRequests = 8
    public var maxBodyBytes = 4 * 1024 * 1024
    public var maxMessages = 1000
    /// Server-side ceiling for max_tokens / max_completion_tokens / max_output_tokens.
    public var maxOutputTokens = 32_768
    public var headerTimeout: TimeInterval = 10
    public var bodyTimeout: TimeInterval = 30
    public var writeTimeout: TimeInterval = 30
    /// Called once per request with a log line: method, path, status, duration.
    public var onLog: (@Sendable (String) -> Void)?

    let engine: InferenceEngine
    private static let queueKey = DispatchSpecificKey<Bool>()
    private let queue: DispatchQueue = {
        let q = DispatchQueue(label: "mlxrunner.openai-server")
        q.setSpecific(key: OpenAIServer.queueKey, value: true)
        return q
    }()
    private let stateLock = NSLock()
    private var listener: NWListener?
    private var connections: [ObjectIdentifier: HTTPConnection] = [:]
    private var inFlightGenerations = 0
    private var _config = Config()
    private var _boundPort: UInt16 = 0
    let generationLock = AsyncLock()

    struct Config {
        var allowLAN = false
        var apiKey: String?
        var allowedOrigins: [String] = OpenAIServer.defaultAllowedOrigins
        var maxConnections = 32
        var maxQueued = 8
        var maxMessages = 1000
        var maxOutputTokens = 32_768
        var limits = HTTPConnection.Limits()
    }

    public init(port: UInt16 = 1234, allowLAN: Bool = false, apiKey: String? = nil, engine: InferenceEngine = .shared) {
        self.port = port
        self.allowLAN = allowLAN
        self.apiKey = apiKey
        self.engine = engine
    }

    public var isRunning: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return listener != nil
    }

    /// The port actually listened on (differs from `port` when `port` is 0).
    /// Open client connections (for monitoring and tests).
    public var openConnections: Int { stateLock.lock(); defer { stateLock.unlock() }; return connections.count }

    public var boundPort: UInt16 {
        stateLock.lock(); defer { stateLock.unlock() }
        return _boundPort
    }

    var config: Config {
        stateLock.lock(); defer { stateLock.unlock() }
        return _config
    }

    /// Generates a random API key (`sk-mlx-` + 32 random bytes, hex) using the system CSPRNG.
    public static func generateAPIKey() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let rc = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        precondition(rc == errSecSuccess, "SecRandomCopyBytes failed")
        return "sk-mlx-" + bytes.map { String(format: "%02x", $0) }.joined()
    }

    /// Starts listening and waits (up to 5 s) until the socket is bound, so bind errors surface here.
    public func start() throws {
        stateLock.lock()
        guard listener == nil else { stateLock.unlock(); return }
        var key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        if key?.isEmpty == true { key = nil }
        if allowLAN && key == nil {
            if let env = ProcessInfo.processInfo.environment["MLXR_API_KEY"], !env.isEmpty {
                key = env
            } else {
                stateLock.unlock()
                throw OpenAIServerError.apiKeyRequired
            }
        }
        _config = Config(
            allowLAN: allowLAN, apiKey: key, allowedOrigins: allowedOrigins,
            maxConnections: max(1, maxConnections), maxQueued: max(0, maxQueuedRequests),
            maxMessages: max(1, maxMessages), maxOutputTokens: max(1, maxOutputTokens),
            limits: .init(maxBodyBytes: max(0, maxBodyBytes), headerTimeout: headerTimeout,
                          bodyTimeout: bodyTimeout, writeTimeout: writeTimeout))
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            stateLock.unlock()
            throw OpenAIServerError.invalidPort
        }
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        let l: NWListener
        do {
            if !allowLAN {
                params.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: nwPort)
                l = try NWListener(using: params)
            } else {
                l = port == 0 ? try NWListener(using: params) : try NWListener(using: params, on: nwPort)
            }
        } catch {
            stateLock.unlock()
            throw error
        }
        listener = l
        stateLock.unlock()

        let ready = DispatchSemaphore(value: 0)
        let failure = LockedBox<Error?>(nil)
        l.newConnectionHandler = { [weak self] conn in self?.accept(conn) }
        l.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                ready.signal()
            case .failed(let err), .waiting(let err):
                failure.value = err
                ready.signal()
                self?.log("server failed: \(err)")
                self?.stop(waitForRelease: false)
            default: break
            }
        }
        l.start(queue: queue)
        if ready.wait(timeout: .now() + 5) == .timedOut {
            stop()
            throw OpenAIServerError.startTimeout
        }
        if let err = failure.value { throw err }
        stateLock.lock(); _boundPort = l.port?.rawValue ?? port; stateLock.unlock()
    }

    /// Stops listening and closes open connections. Waits (up to 2 s) until the listening
    /// socket is released, so the port can be reused immediately.
    public func stop() { stop(waitForRelease: true) }

    private func stop(waitForRelease: Bool) {
        stateLock.lock()
        let l = listener
        listener = nil
        let conns = Array(connections.values)
        connections.removeAll()
        stateLock.unlock()
        if let l {
            let released = DispatchSemaphore(value: 0)
            l.stateUpdateHandler = { state in
                if case .cancelled = state { released.signal() }
            }
            l.newConnectionHandler = { $0.cancel() }
            l.cancel()
            if waitForRelease && DispatchQueue.getSpecific(key: Self.queueKey) == nil {
                _ = released.wait(timeout: .now() + 2)
            }
        }
        for c in conns { c.close() }
    }

    func log(_ line: String) { onLog?(line) }

    private func accept(_ nw: NWConnection) {
        let cfg = config
        let conn = HTTPConnection(connection: nw, queue: queue, limits: cfg.limits)
        let id = ObjectIdentifier(conn)
        stateLock.lock()
        let full = connections.count >= cfg.maxConnections
        if !full { connections[id] = conn }
        stateLock.unlock()
        if full {
            let e = APIError(status: 503, message: "Too many open connections; try again shortly",
                             type: "server_error", code: "server_busy", headers: ["Retry-After": "1"])
            conn.reject(status: e.status, json: e.body, headers: e.headers)
            log("- - 503 (connection limit)")
            return
        }
        conn.onClose = { [weak self] in
            guard let self else { return }
            self.stateLock.lock(); self.connections[id] = nil; self.stateLock.unlock()
        }
        conn.start { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let req):
                Task { await self.handle(req, on: conn) }
            case .failure(let err):
                let e = APIError(status: err.status, message: err.message, code: err.code)
                conn.sendResponse(status: e.status, json: e.body)
                self.log("- - \(err.status) (\(err.message))")
            }
        }
    }

    // MARK: - Routing

    private func handle(_ req: HTTPRequest, on conn: HTTPConnection) async {
        let start = Date()
        let status = await route(req, on: conn)
        let ms = Date().timeIntervalSince(start) * 1000
        log(String(format: "%@ %@ %d %.0fms", req.method, req.path, status, ms))
    }

    /// Method sets per route; used for 404 vs 405.
    private static func allowedMethods(_ path: String) -> [String]? {
        switch path {
        case "/", "/v1/models": return ["GET"]
        case "/v1/chat/completions", "/v1/completions", "/v1/responses": return ["POST"]
        default:
            if path.hasPrefix("/v1/models/") { return ["GET"] }
            if path.hasPrefix("/v1/responses/") { return ["GET", "DELETE"] }
            return nil
        }
    }

    private func route(_ req: HTTPRequest, on conn: HTTPConnection) async -> Int {
        do {
            return try await routeThrowing(req, on: conn)
        } catch let e as APIError {
            if e.status == 499 { conn.close(); return 499 }
            conn.sendResponse(status: e.status, json: e.body, headers: e.headers)
            return e.status
        } catch {
            let e = APIError(status: 500, message: "Internal error: \(error.localizedDescription)", type: "server_error")
            conn.sendResponse(status: 500, json: e.body)
            return 500
        }
    }

    private func routeThrowing(_ req: HTTPRequest, on conn: HTTPConnection) async throws -> Int {
        let cfg = config
        // 1. Host (DNS rebinding defence).
        if !cfg.allowLAN {
            guard let host = req.headers["host"], Self.isLoopbackHost(host) else {
                throw APIError(status: 403, message: "Invalid Host header; this server only answers to localhost",
                               code: "invalid_host")
            }
        }
        // 2. Origin (cross-site requests from browsers).
        if let origin = req.headers["origin"] {
            guard Self.originAllowed(origin, patterns: cfg.allowedOrigins) else {
                throw APIError(status: 403, message: "Origin '\(origin)' is not allowed", code: "origin_not_allowed")
            }
            conn.corsOrigin = origin
        }
        // 3. CORS preflight (browsers never send credentials on it).
        if req.method == "OPTIONS" {
            var headers: [String: String] = [:]
            if conn.corsOrigin != nil, let requested = req.headers["access-control-request-headers"],
               requested.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == "," || $0 == " " }) {
                headers["Access-Control-Allow-Headers"] = requested
            }
            conn.sendResponse(status: 204, contentType: nil, body: Data(), headers: headers)
            return 204
        }
        // 4. API key.
        if let key = cfg.apiKey {
            let auth = req.headers["authorization"] ?? ""
            let given = auth.lowercased().hasPrefix("bearer ") ? String(auth.dropFirst(7)).trimmingCharacters(in: .whitespaces) : ""
            guard Self.constantTimeEqual(given, key) else {
                throw APIError(status: 401, message: given.isEmpty
                               ? "Missing API key. Send 'Authorization: Bearer <key>'."
                               : "Incorrect API key provided.",
                               code: "invalid_api_key", headers: ["WWW-Authenticate": "Bearer"])
            }
        }
        // 5. Method / path.
        let path = req.path.split(separator: "?", maxSplits: 1).first.map(String.init) ?? req.path
        let normalized = path.count > 1 && path.hasSuffix("/") ? String(path.dropLast()) : path
        guard let methods = Self.allowedMethods(normalized) else {
            throw APIError(status: 404, message: "Unknown endpoint: \(req.method) \(path)", code: "not_found")
        }
        guard methods.contains(req.method) else {
            throw APIError(status: 405, message: "Method \(req.method) is not allowed for \(path)",
                           code: "method_not_allowed", headers: ["Allow": (methods + ["OPTIONS"]).joined(separator: ", ")])
        }
        // 6. JSON bodies only.
        if req.method == "POST" {
            let mediaType = (req.headers["content-type"] ?? "").split(separator: ";").first?
                .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
            guard mediaType == "application/json" else {
                throw APIError(status: 415, message: "Content-Type must be application/json", code: "unsupported_media_type")
            }
        }

        switch (req.method, normalized) {
        case ("GET", "/"):
            let name = engine.info?.name
            let text = "MLX Runner OpenAI-compatible server is running.\nModel: \(name ?? "(none loaded)")\n"
            conn.sendResponse(status: 200, contentType: "text/plain; charset=utf-8", body: Data(text.utf8))
            return 200
        case ("GET", "/v1/models"):
            let data = engine.info.map { [Self.modelObject($0)] } ?? []
            conn.sendResponse(status: 200, json: ["object": "list", "data": data])
            return 200
        case ("GET", let p) where p.hasPrefix("/v1/models/"):
            let id = String(p.dropFirst("/v1/models/".count)).removingPercentEncoding ?? ""
            guard let info = engine.info, info.name == id else {
                throw APIError(status: 404, message: "The model '\(id)' does not exist", param: "model", code: "model_not_found")
            }
            conn.sendResponse(status: 200, json: Self.modelObject(info))
            return 200
        case (_, let p) where p.hasPrefix("/v1/responses/"):
            let id = String(p.dropFirst("/v1/responses/".count))
            throw APIError(status: 404, message: "Response '\(id)' not found. This server does not store responses.",
                           code: "not_found")
        default:
            break
        }

        let body = try JSONBody(req.body)
        switch normalized {
        case "/v1/chat/completions": return try await chatCompletions(body, on: conn)
        case "/v1/completions": return try await legacyCompletions(body, on: conn)
        default: return try await responses(body, on: conn)
        }
    }

    static func modelObject(_ info: LoadedModelInfo) -> [String: Any] {
        let attrs = try? FileManager.default.attributesOfItem(atPath: info.directory.path)
        let created = (attrs?[.creationDate] as? Date) ?? Date()
        return ["id": info.name, "object": "model", "created": Int(created.timeIntervalSince1970), "owned_by": "mlx-runner"]
    }

    // MARK: - Request parsing

    /// Sampling parameters common to all generation endpoints. All numbers are range-checked as
    /// doubles before any integer/float conversion, so huge values cannot trap.
    private func sampling(_ b: JSONBody, maxKeys: [String], defaultMax: Int) throws -> SamplingParameters {
        let cfg = config
        var p = SamplingParameters()
        if let v = try b.number("temperature") {
            guard v >= 0 && v <= 2 else { throw APIError.invalid("'temperature' must be between 0 and 2", param: "temperature") }
            p.temperature = Float(v)
        }
        if let v = try b.number("top_p") {
            guard v > 0 && v <= 1 else { throw APIError.invalid("'top_p' must be greater than 0 and at most 1", param: "top_p") }
            p.topP = Float(v)
        }
        if let v = try b.number("top_k") {
            guard v >= -1 else { throw APIError.invalid("'top_k' must be >= 0 (or -1 to disable)", param: "top_k") }
            p.topK = v <= 0 ? 0 : Int(min(v, 100_000))
        }
        if let v = try b.number("min_p") {
            guard v >= 0 && v <= 1 else { throw APIError.invalid("'min_p' must be between 0 and 1", param: "min_p") }
            p.minP = Float(v)
        }
        let rpKey = b.has("repetition_penalty") ? "repetition_penalty" : "repeat_penalty"
        if let v = try b.number(rpKey) {
            guard v > 0 else { throw APIError.invalid("'\(rpKey)' must be greater than 0", param: rpKey) }
            p.repetitionPenalty = Float(min(v, 100))
        }
        for k in ["frequency_penalty", "presence_penalty"] {
            // Accepted for compatibility and range-checked, but not applied (see type docs).
            if let v = try b.number(k), !(v >= -2 && v <= 2) {
                throw APIError.invalid("'\(k)' must be between -2 and 2", param: k)
            }
        }
        if let k = maxKeys.first(where: { b.has($0) }), let v = try b.number(k) {
            guard v >= 1 else { throw APIError.invalid("'\(k)' must be at least 1", param: k) }
            p.maxTokens = Int(min(v, Double(cfg.maxOutputTokens)))
        } else {
            p.maxTokens = min(defaultMax, cfg.maxOutputTokens)
        }
        if let v = try b.number("seed") {
            let c = min(max(v.rounded(.towardZero), -9.0e18), 9.0e18)
            p.seed = UInt64(bitPattern: Int64(c))
        }
        if let v = try b.number("n"), v != 1 {
            throw APIError.invalid("Only n=1 is supported", param: "n")
        }
        return p
    }

    private func stops(_ b: JSONBody) throws -> [String] {
        guard b.has("stop") else { return [] }
        let v = b.raw["stop"]
        var out: [String]
        if let s = v as? String { out = [s] }
        else if let a = v as? [Any], a.allSatisfy({ $0 is String }) { out = a as! [String] }
        else { throw APIError.invalid("'stop' must be a string or an array of strings", param: "stop") }
        guard out.count <= 16 else { throw APIError.invalid("At most 16 stop sequences are supported", param: "stop") }
        out = out.filter { !$0.isEmpty }
        return out
    }

    /// `enable_thinking`, `chat_template_kwargs.enable_thinking`, and a reasoning effort of "none".
    private func enableThinking(_ b: JSONBody, effort: String?) throws -> Bool {
        var on = true
        if effort == "none" { on = false }
        if let v = try b.bool("enable_thinking") { on = v }
        if let kw = b.raw["chat_template_kwargs"] as? [String: Any], let v = kw["enable_thinking"] as? Bool { on = v }
        return on
    }

    /// Extracts text from string-or-parts content; non-text parts are a 400.
    static func textContent(_ v: Any?, param: String, textTypes: Set<String>) throws -> String {
        if v == nil || v is NSNull { return "" }
        if let s = v as? String { return s }
        guard let parts = v as? [Any] else {
            throw APIError.invalid("'\(param)' must be a string or an array of content parts", param: param)
        }
        var out: [String] = []
        for (i, p) in parts.enumerated() {
            let pp = "\(param)[\(i)]"
            guard let d = p as? [String: Any], let type = d["type"] as? String else {
                throw APIError.invalid("Content part '\(pp)' must be an object with a 'type'", param: pp)
            }
            if textTypes.contains(type) {
                guard let t = d["text"] as? String else { throw APIError.invalid("Content part '\(pp)' needs a 'text' string", param: pp) }
                out.append(t)
            } else if type == "refusal", let t = d["refusal"] as? String {
                out.append(t)
            } else {
                throw APIError.invalid("Content part type '\(type)' is not supported: this server only accepts text",
                                       param: pp, code: "unsupported_content_type")
            }
        }
        return out.joined(separator: "\n")
    }

    private static func role(_ s: String, param: String) throws -> ChatMessage.Role {
        switch s {
        case "system", "developer": return .system
        case "user": return .user
        case "assistant": return .assistant
        case "tool", "function":
            throw APIError.invalid("Role '\(s)' is not supported: this server does not support tool calling", param: param)
        default:
            throw APIError.invalid("Invalid role '\(s)'", param: param)
        }
    }

    /// Rejects tool definitions unless tool_choice is "none".
    private static func rejectTools(_ b: JSONBody) throws {
        let toolChoiceNone = (b.raw["tool_choice"] as? String) == "none"
        for k in ["tools", "functions"] {
            if let a = b.raw[k] as? [Any], !a.isEmpty, !toolChoiceNone {
                throw APIError.invalid("Tool calling is not supported by this server (set tool_choice to \"none\" or omit '\(k)')",
                                       param: k, code: "unsupported_parameter")
            }
        }
        if let tc = b.raw["tool_choice"], !(tc is NSNull), (tc as? String) != "none", (tc as? String) != "auto" {
            throw APIError.invalid("Tool calling is not supported by this server", param: "tool_choice", code: "unsupported_parameter")
        }
    }

    // MARK: - Chat Completions

    private func chatCompletions(_ b: JSONBody, on conn: HTTPConnection) async throws -> Int {
        let cfg = config
        guard let raw = b.raw["messages"] as? [Any] else {
            throw APIError.invalid("'messages' is required and must be an array", param: "messages")
        }
        guard !raw.isEmpty else { throw APIError.invalid("'messages' must not be empty", param: "messages") }
        guard raw.count <= cfg.maxMessages else {
            throw APIError.invalid("Too many messages (max \(cfg.maxMessages))", param: "messages")
        }
        var messages: [ChatMessage] = []
        for (i, item) in raw.enumerated() {
            let p = "messages[\(i)]"
            guard let m = item as? [String: Any], let roleStr = m["role"] as? String else {
                throw APIError.invalid("'\(p)' must be an object with a 'role'", param: "\(p).role")
            }
            let role = try Self.role(roleStr, param: "\(p).role")
            if role == .assistant, let tc = m["tool_calls"] as? [Any], !tc.isEmpty {
                throw APIError.invalid("Assistant tool_calls are not supported", param: "\(p).tool_calls")
            }
            let content = try Self.textContent(m["content"], param: "\(p).content", textTypes: ["text"])
            messages.append(ChatMessage(role: role, content: content))
        }
        try Self.rejectTools(b)
        if let rf = b.raw["response_format"] as? [String: Any], let t = rf["type"] as? String, t != "text" {
            throw APIError.invalid("response_format '\(t)' is not supported by this server", param: "response_format",
                                   code: "unsupported_parameter")
        }
        if try b.bool("logprobs") == true {
            throw APIError.invalid("logprobs are not supported by this server", param: "logprobs", code: "unsupported_parameter")
        }
        if let lb = b.raw["logit_bias"] as? [String: Any], !lb.isEmpty {
            throw APIError.invalid("logit_bias is not supported by this server", param: "logit_bias", code: "unsupported_parameter")
        }
        if let mods = b.raw["modalities"] as? [String], mods.contains("audio") || b.has("audio") {
            throw APIError.invalid("Audio output is not supported", param: "modalities", code: "unsupported_parameter")
        }
        let params = try sampling(b, maxKeys: ["max_completion_tokens", "max_tokens"], defaultMax: 4096)
        let stopList = try stops(b)
        let stream = try b.bool("stream") ?? false
        let includeUsage = ((b.raw["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false
        let thinking = try enableThinking(b, effort: b.raw["reasoning_effort"] as? String)
        let job = Job(chat: true, messages: messages, prompt: "", params: params, stops: stopList,
                      enableThinking: thinking, splitReasoning: true)

        let token = CancelToken()
        conn.onClientGone = { token.cancel() }
        conn.watchForDisconnect()
        return try await withGenerationSlot(conn) { info in
            let id = "chatcmpl-" + Self.randomID()
            let created = Int(Date().timeIntervalSince1970)
            let model = info.name
            let fp = Self.fingerprint(info)

            if !stream {
                let out = try await self.run(job, token: token) { _ in !conn.isClosed }
                if conn.isClosed { return 499 }
                var message: [String: Any] = ["role": "assistant", "content": out.content, "refusal": NSNull(), "annotations": []]
                if !out.reasoning.isEmpty { message["reasoning_content"] = out.reasoning }
                conn.sendResponse(status: 200, json: [
                    "id": id, "object": "chat.completion", "created": created, "model": model, "system_fingerprint": fp,
                    "choices": [["index": 0, "message": message, "logprobs": NSNull(), "finish_reason": out.finish]],
                    "usage": Self.chatUsage(out),
                ])
                return 200
            }

            func chunk(_ delta: [String: Any], finish: String?) -> [String: Any] {
                var o: [String: Any] = [
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "system_fingerprint": fp,
                    "choices": [["index": 0, "delta": delta, "logprobs": NSNull(), "finish_reason": finish.map { $0 as Any } ?? NSNull()]],
                ]
                if includeUsage { o["usage"] = NSNull() }
                return o
            }
            func send(_ obj: [String: Any]) async -> Bool {
                await conn.sendRaw(Data("data: ".utf8) + Self.json(obj) + Data("\n\n".utf8))
            }
            guard await conn.sendHead(status: 200, headers: Self.sseHeaders) else { token.cancel(); return 499 }
            let first = await send(chunk(["role": "assistant", "content": ""], finish: nil))
            if !first { token.cancel() }
            do {
                let out = try await self.run(job, token: token) { d in
                    switch d {
                    case .reasoning(let r): return await send(chunk(["reasoning_content": r], finish: nil))
                    case .content(let c): return await send(chunk(["content": c], finish: nil))
                    }
                }
                guard first, !out.aborted, !conn.isClosed else { conn.close(); return 499 }
                _ = await send(chunk([:], finish: out.finish))
                if includeUsage {
                    _ = await send(["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                                    "system_fingerprint": fp, "choices": [], "usage": Self.chatUsage(out)])
                }
                _ = await conn.sendRaw(Data("data: [DONE]\n\n".utf8))
                conn.close()
                return 200
            } catch let e as APIError {
                _ = await send(e.body)
                conn.close()
                return e.status
            }
        }
    }

    // MARK: - Legacy completions

    private func legacyCompletions(_ b: JSONBody, on conn: HTTPConnection) async throws -> Int {
        let prompt: String
        if let p = b.raw["prompt"] as? String { prompt = p }
        else if let ps = b.raw["prompt"] as? [Any], ps.count == 1, let p = ps[0] as? String { prompt = p }
        else { throw APIError.invalid("'prompt' must be a string", param: "prompt") }
        guard !prompt.isEmpty else { throw APIError.invalid("'prompt' must not be empty", param: "prompt") }
        if let s = b.raw["suffix"] as? String, !s.isEmpty {
            throw APIError.invalid("'suffix' is not supported", param: "suffix", code: "unsupported_parameter")
        }
        if try b.bool("echo") == true { throw APIError.invalid("'echo' is not supported", param: "echo", code: "unsupported_parameter") }
        if let v = try b.number("best_of"), v != 1 { throw APIError.invalid("Only best_of=1 is supported", param: "best_of") }
        if let v = try b.number("logprobs"), v > 0 {
            throw APIError.invalid("logprobs are not supported by this server", param: "logprobs", code: "unsupported_parameter")
        }
        let params = try sampling(b, maxKeys: ["max_tokens"], defaultMax: 256)
        let stopList = try stops(b)
        let stream = try b.bool("stream") ?? false
        let includeUsage = ((b.raw["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false
        let job = Job(chat: false, messages: [], prompt: prompt, params: params, stops: stopList,
                      enableThinking: true, splitReasoning: false)

        let token = CancelToken()
        conn.onClientGone = { token.cancel() }
        conn.watchForDisconnect()
        return try await withGenerationSlot(conn) { info in
            let id = "cmpl-" + Self.randomID()
            let created = Int(Date().timeIntervalSince1970)
            let model = info.name
            let fp = Self.fingerprint(info)
            func obj(_ text: String, finish: String?) -> [String: Any] {
                ["id": id, "object": "text_completion", "created": created, "model": model, "system_fingerprint": fp,
                 "choices": [["index": 0, "text": text, "logprobs": NSNull(), "finish_reason": finish.map { $0 as Any } ?? NSNull()]]]
            }
            if !stream {
                let out = try await self.run(job, token: token) { _ in !conn.isClosed }
                if conn.isClosed { return 499 }
                var o = obj(out.content, finish: out.finish)
                o["usage"] = Self.chatUsage(out)
                conn.sendResponse(status: 200, json: o)
                return 200
            }
            func send(_ o: [String: Any]) async -> Bool {
                await conn.sendRaw(Data("data: ".utf8) + Self.json(o) + Data("\n\n".utf8))
            }
            guard await conn.sendHead(status: 200, headers: Self.sseHeaders) else { token.cancel(); return 499 }
            do {
                let out = try await self.run(job, token: token) { d in
                    if case .content(let c) = d { return await send(obj(c, finish: nil)) }
                    return true
                }
                guard !out.aborted, !conn.isClosed else { conn.close(); return 499 }
                _ = await send(obj("", finish: out.finish))
                if includeUsage {
                    var u = obj("", finish: nil)
                    u["choices"] = []
                    u["usage"] = Self.chatUsage(out)
                    _ = await send(u)
                }
                _ = await conn.sendRaw(Data("data: [DONE]\n\n".utf8))
            } catch let e as APIError {
                _ = await send(e.body)
            }
            conn.close()
            return 200
        }
    }

    // MARK: - Responses API

    private func responses(_ b: JSONBody, on conn: HTTPConnection) async throws -> Int {
        let cfg = config
        for k in ["previous_response_id", "conversation", "prompt"] where b.has(k) {
            throw APIError.invalid("'\(k)' is not supported: this server does not store responses or prompts",
                                   param: k, code: "unsupported_parameter")
        }
        if try b.bool("background") == true {
            throw APIError.invalid("Background responses are not supported", param: "background", code: "unsupported_parameter")
        }
        try Self.rejectTools(b)
        if let text = b.raw["text"] as? [String: Any], let f = text["format"] as? [String: Any],
           let t = f["type"] as? String, t != "text" {
            throw APIError.invalid("text.format '\(t)' is not supported by this server", param: "text.format",
                                   code: "unsupported_parameter")
        }
        if let v = try b.number("top_logprobs"), v > 0 {
            throw APIError.invalid("top_logprobs is not supported", param: "top_logprobs", code: "unsupported_parameter")
        }
        var messages: [ChatMessage] = []
        let instructions = try b.string("instructions")
        if let instructions, !instructions.isEmpty { messages.append(ChatMessage(role: .system, content: instructions)) }
        if let s = b.raw["input"] as? String {
            guard !s.isEmpty else { throw APIError.invalid("'input' must not be empty", param: "input") }
            messages.append(ChatMessage(role: .user, content: s))
        } else if let items = b.raw["input"] as? [Any] {
            guard !items.isEmpty else { throw APIError.invalid("'input' must not be empty", param: "input") }
            guard items.count <= cfg.maxMessages else {
                throw APIError.invalid("Too many input items (max \(cfg.maxMessages))", param: "input")
            }
            for (i, item) in items.enumerated() {
                let p = "input[\(i)]"
                guard let d = item as? [String: Any] else { throw APIError.invalid("'\(p)' must be an object", param: p) }
                let type = d["type"] as? String ?? "message"
                if type == "reasoning" { continue }  // prior reasoning items are not re-fed to the model
                guard type == "message" else {
                    throw APIError.invalid("Input item type '\(type)' is not supported", param: "\(p).type", code: "unsupported_parameter")
                }
                guard let roleStr = d["role"] as? String else { throw APIError.invalid("'\(p).role' is required", param: "\(p).role") }
                let role = try Self.role(roleStr, param: "\(p).role")
                let content = try Self.textContent(d["content"], param: "\(p).content", textTypes: ["input_text", "output_text", "text"])
                messages.append(ChatMessage(role: role, content: content))
            }
        } else {
            throw APIError.invalid("'input' is required and must be a string or an array of input items", param: "input")
        }
        guard messages.contains(where: { $0.role != .system }) else {
            throw APIError.invalid("'input' must contain at least one user or assistant message", param: "input")
        }
        let params = try sampling(b, maxKeys: ["max_output_tokens"], defaultMax: 4096)
        let maxOutputTokens: Any = b.has("max_output_tokens") ? params.maxTokens : NSNull()
        let effort = (b.raw["reasoning"] as? [String: Any])?["effort"] as? String
        let thinking = try enableThinking(b, effort: effort)
        let stream = try b.bool("stream") ?? false
        let metadata = b.raw["metadata"] as? [String: Any] ?? [:]
        let job = Job(chat: true, messages: messages, prompt: "", params: params, stops: [],
                      enableThinking: thinking, splitReasoning: true)

        let token = CancelToken()
        conn.onClientGone = { token.cancel() }
        conn.watchForDisconnect()
        return try await withGenerationSlot(conn) { info in
            let respID = "resp_" + Self.randomID(32)
            let msgID = "msg_" + Self.randomID(32)
            let rsID = "rs_" + Self.randomID(32)
            let created = Int(Date().timeIntervalSince1970)
            func response(status: String, output: [[String: Any]], usage: Any, incomplete: Bool = false,
                          error: Any = NSNull()) -> [String: Any] {
                [
                    "id": respID, "object": "response", "created_at": created, "status": status,
                    "background": false, "error": error,
                    "incomplete_details": incomplete ? ["reason": "max_output_tokens"] : NSNull(),
                    "instructions": instructions.map { $0 as Any } ?? NSNull(), "max_output_tokens": maxOutputTokens,
                    "max_tool_calls": NSNull(), "model": info.name, "output": output,
                    "parallel_tool_calls": false, "previous_response_id": NSNull(),
                    "reasoning": ["effort": effort.map { $0 as Any } ?? NSNull(), "summary": NSNull()],
                    "store": false, "temperature": Double(String(params.temperature)) ?? 1, "top_p": Double(String(params.topP)) ?? 1,
                    "text": ["format": ["type": "text"]], "tool_choice": "none", "tools": [],
                    "truncation": "auto", "usage": usage, "user": NSNull(), "metadata": metadata,
                ]
            }
            func reasoningItem(_ text: String, status: String) -> [String: Any] {
                ["type": "reasoning", "id": rsID, "status": status, "summary": [],
                 "content": text.isEmpty && status == "in_progress" ? [] : [["type": "reasoning_text", "text": text]]]
            }
            func messageItem(_ text: String?, status: String) -> [String: Any] {
                ["type": "message", "id": msgID, "status": status, "role": "assistant",
                 "content": text.map { [["type": "output_text", "text": $0, "annotations": [], "logprobs": []]] } ?? []]
            }

            if !stream {
                let out = try await self.run(job, token: token) { _ in !conn.isClosed }
                if conn.isClosed { return 499 }
                let incomplete = out.finish == "length"
                var output: [[String: Any]] = []
                if !out.reasoning.isEmpty { output.append(reasoningItem(out.reasoning, status: "completed")) }
                output.append(messageItem(out.content, status: incomplete ? "incomplete" : "completed"))
                conn.sendResponse(status: 200, json: response(status: incomplete ? "incomplete" : "completed",
                                                              output: output, usage: Self.responsesUsage(out),
                                                              incomplete: incomplete))
                return 200
            }

            guard await conn.sendHead(status: 200, headers: Self.sseHeaders) else { token.cancel(); return 499 }
            var seq = 0
            func event(_ type: String, _ payload: [String: Any]) async -> Bool {
                var p = payload
                p["type"] = type
                p["sequence_number"] = seq
                seq += 1
                return await conn.sendRaw(Data("event: \(type)\ndata: ".utf8) + Self.json(p) + Data("\n\n".utf8))
            }
            var output: [[String: Any]] = []
            var reasoningIndex: Int?   // open reasoning item
            var reasoningText = ""
            var messageIndex: Int?     // open message item
            var messageText = ""
            var alive = await event("response.created", ["response": response(status: "in_progress", output: [], usage: NSNull())])
            if alive { alive = await event("response.in_progress", ["response": response(status: "in_progress", output: [], usage: NSNull())]) }

            func closeReasoning() async -> Bool {
                guard let idx = reasoningIndex else { return true }
                reasoningIndex = nil
                let part: [String: Any] = ["type": "reasoning_text", "text": reasoningText]
                let item = reasoningItem(reasoningText, status: "completed")
                output.append(item)
                guard await event("response.reasoning_text.done", ["item_id": rsID, "output_index": idx, "content_index": 0, "text": reasoningText]),
                      await event("response.content_part.done", ["item_id": rsID, "output_index": idx, "content_index": 0, "part": part])
                else { return false }
                return await event("response.output_item.done", ["output_index": idx, "item": item])
            }
            func openMessage() async -> Bool {
                guard messageIndex == nil else { return true }
                guard await closeReasoning() else { return false }
                let idx = output.count
                messageIndex = idx
                guard await event("response.output_item.added", ["output_index": idx, "item": messageItem(nil, status: "in_progress")]) else { return false }
                return await event("response.content_part.added", [
                    "item_id": msgID, "output_index": idx, "content_index": 0,
                    "part": ["type": "output_text", "text": "", "annotations": [], "logprobs": []] as [String: Any]])
            }

            do {
                let out = try await self.run(job, token: token) { d in
                    guard alive else { return false }
                    switch d {
                    case .reasoning(let r):
                        if reasoningIndex == nil && messageIndex == nil {
                            let idx = output.count
                            reasoningIndex = idx
                            alive = await event("response.output_item.added", ["output_index": idx, "item": reasoningItem("", status: "in_progress")])
                            if alive {
                                alive = await event("response.content_part.added", [
                                    "item_id": rsID, "output_index": idx, "content_index": 0,
                                    "part": ["type": "reasoning_text", "text": ""] as [String: Any]])
                            }
                        }
                        guard alive, let idx = reasoningIndex else { return alive }
                        reasoningText += r
                        alive = await event("response.reasoning_text.delta", ["item_id": rsID, "output_index": idx, "content_index": 0, "delta": r])
                    case .content(let c):
                        alive = await openMessage()
                        guard alive, let idx = messageIndex else { return alive }
                        messageText += c
                        alive = await event("response.output_text.delta", [
                            "item_id": msgID, "output_index": idx, "content_index": 0, "delta": c, "logprobs": []])
                    }
                    return alive
                }
                guard alive, !out.aborted, !conn.isClosed else { conn.close(); return 499 }
                let incomplete = out.finish == "length"
                if await openMessage(), let idx = messageIndex {
                    let part: [String: Any] = ["type": "output_text", "text": messageText, "annotations": [], "logprobs": []]
                    let item = messageItem(messageText, status: incomplete ? "incomplete" : "completed")
                    output.append(item)
                    _ = await event("response.output_text.done", ["item_id": msgID, "output_index": idx, "content_index": 0, "text": messageText, "logprobs": []])
                    _ = await event("response.content_part.done", ["item_id": msgID, "output_index": idx, "content_index": 0, "part": part])
                    _ = await event("response.output_item.done", ["output_index": idx, "item": item])
                }
                _ = await event(incomplete ? "response.incomplete" : "response.completed", [
                    "response": response(status: incomplete ? "incomplete" : "completed", output: output,
                                         usage: Self.responsesUsage(out), incomplete: incomplete)])
                conn.close()
                return 200
            } catch let e as APIError {
                let err: [String: Any] = ["code": e.code ?? "server_error", "message": e.message]
                _ = await event("error", ["code": e.code ?? "server_error", "message": e.message, "param": e.param.map { $0 as Any } ?? NSNull()])
                _ = await event("response.failed", ["response": response(status: "failed", output: output, usage: NSNull(), error: err)])
                conn.close()
                return e.status
            }
        }
    }

    // MARK: - Generation core

    struct Job {
        let chat: Bool
        let messages: [ChatMessage]
        let prompt: String
        let params: SamplingParameters
        let stops: [String]
        let enableThinking: Bool
        let splitReasoning: Bool
    }

    enum Delta { case reasoning(String), content(String) }

    struct Outcome {
        var reasoning = ""
        var content = ""
        var stats = GenerationStats()
        var finish = "stop"
        var reasoningEvents = 0
        var aborted = false
    }

    /// Takes a generation ticket: at most one running plus `maxQueued` waiting.
    private func takeTicket() -> Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        guard inFlightGenerations < _config.maxQueued + 1 else { return false }
        inFlightGenerations += 1
        return true
    }

    private func returnTicket() {
        stateLock.lock(); inFlightGenerations -= 1; stateLock.unlock()
    }

    /// Waits for the single generation slot (bounded queue), then runs `body` with it held.
    private func withGenerationSlot(_ conn: HTTPConnection, _ body: (LoadedModelInfo) async throws -> Int) async throws -> Int {
        guard engine.info != nil else { throw APIError.noModel }
        guard takeTicket() else {
            throw APIError(status: 429, message: "Too many queued requests; try again shortly",
                           type: "requests", code: "rate_limit_exceeded", headers: ["Retry-After": "2"])
        }
        await generationLock.lock()
        do {
            if conn.isClosed { throw APIError.clientGone }
            guard let info = engine.info else { throw APIError.noModel }
            let r = try await body(info)
            await generationLock.unlock()
            returnTicket()
            return r
        } catch {
            await generationLock.unlock()
            returnTicket()
            throw error
        }
    }

    /// Runs one generation, splitting `<think>` reasoning and applying stop sequences to content.
    /// `onDelta` returns false when the client is gone; generation is then cancelled.
    private func run(_ job: Job, token: CancelToken, onDelta: (Delta) async -> Bool) async throws -> Outcome {
        let events = job.chat
            ? engine.chat(job.messages, parameters: job.params, enableThinking: job.enableThinking, token: token)
            : engine.complete(job.prompt, parameters: job.params, token: token)
        var splitter = ReasoningSplitter(enabled: job.splitReasoning)
        var filter = StopFilter(stops: job.stops)
        var out = Outcome()
        var stopped = false
        var alive = true
        func emit(_ r: String, _ c: String, final: Bool) async -> Bool {
            var c = c
            var hit = false
            if final { let (x, h) = filter.feed(c); c = x + (h ? "" : filter.flush()); hit = h }
            else { (c, hit) = filter.feed(c) }
            if !r.isEmpty {
                out.reasoning += r
                out.reasoningEvents += 1
                if !(await onDelta(.reasoning(r))) { return false }
            }
            if !c.isEmpty {
                out.content += c
                if !(await onDelta(.content(c))) { return false }
            }
            if hit { stopped = true; token.cancel() }
            return true
        }
        do {
            for try await ev in events {
                switch ev {
                case .promptProcessed(let s), .finished(let s): out.stats = s
                case .text(let t):
                    guard alive, !stopped else { continue }
                    let (r, c) = splitter.feed(t)
                    alive = await emit(r, c, final: false)
                    if !alive { token.cancel() }
                }
            }
        } catch where stopped || !alive || token.isCancelled {
            // We cancelled (stop sequence, disconnect, write timeout): not an error whichever way
            // the engine reports cancellation.
        } catch let e as EngineError {
            if case .contextFull = e {
                throw APIError(status: 400, message: "The prompt does not fit in the model's context window",
                               param: "messages", code: "context_length_exceeded")
            }
            throw APIError(status: 500, message: "Generation failed: \(e.localizedDescription)", type: "server_error")
        } catch {
            throw APIError(status: 500, message: "Generation failed: \(error.localizedDescription)", type: "server_error")
        }
        if alive && !stopped {
            let (r, c) = splitter.flush()
            alive = await emit(r, c, final: true)
        }
        out.aborted = !alive
        out.finish = stopped ? "stop" : Self.finishReason(out.stats.stopReason)
        return out
    }

    // MARK: - Helpers

    static let sseHeaders = ["Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"]

    static func finishReason(_ r: String) -> String {
        switch r {
        case "length", "context": return "length"
        default: return "stop"
        }
    }

    /// Reasoning tokens are estimated from the number of decoded chunks routed to reasoning
    /// (the engine emits about one chunk per token).
    static func reasoningTokens(_ o: Outcome) -> Int { min(o.reasoningEvents, o.stats.generatedTokens) }

    static func chatUsage(_ o: Outcome) -> [String: Any] {
        let s = o.stats
        return ["prompt_tokens": s.promptTokens, "completion_tokens": s.generatedTokens,
                "total_tokens": s.promptTokens + s.generatedTokens,
                "prompt_tokens_details": ["cached_tokens": s.cachedPromptTokens, "audio_tokens": 0],
                "completion_tokens_details": ["reasoning_tokens": reasoningTokens(o), "audio_tokens": 0,
                                              "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0]]
    }

    static func responsesUsage(_ o: Outcome) -> [String: Any] {
        let s = o.stats
        return ["input_tokens": s.promptTokens, "input_tokens_details": ["cached_tokens": s.cachedPromptTokens],
                "output_tokens": s.generatedTokens, "output_tokens_details": ["reasoning_tokens": reasoningTokens(o)],
                "total_tokens": s.promptTokens + s.generatedTokens]
    }

    /// Stable per-model fingerprint (FNV-1a of name + quantization + context length).
    static func fingerprint(_ info: LoadedModelInfo) -> String {
        var h: UInt64 = 0xcbf29ce484222325
        for b in "\(info.name)|\(info.quantization)|\(info.contextLength)".utf8 { h = (h ^ UInt64(b)) &* 0x100000001b3 }
        return "fp_" + String(String(h, radix: 16).prefix(10))
    }

    static func randomID(_ n: Int = 24) -> String {
        var s = ""
        while s.count < n { s += UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased() }
        return String(s.prefix(n))
    }

    static func number(_ v: Any?) -> Double? {
        guard let n = v as? NSNumber else { return nil }
        // JSON booleans also bridge to NSNumber; reject them.
        if CFGetTypeID(n) == CFBooleanGetTypeID() { return nil }
        let d = n.doubleValue
        return d.isFinite ? d : nil
    }

    /// `localhost`, `127.0.0.1` or `[::1]`, optionally with a port.
    static func isLoopbackHost(_ host: String) -> Bool {
        var h = host.trimmingCharacters(in: .whitespaces).lowercased()
        var portPart: Substring = ""
        if h.hasPrefix("[") {
            guard let close = h.firstIndex(of: "]") else { return false }
            portPart = h[h.index(after: close)...]
            h = String(h[...close])
        } else if let colon = h.firstIndex(of: ":") {
            portPart = h[colon...]
            h = String(h[..<colon])
        }
        if !portPart.isEmpty {
            guard portPart.first == ":", portPart.count > 1, portPart.dropFirst().allSatisfy(\.isASCII),
                  portPart.dropFirst().allSatisfy(\.isNumber) else { return false }
        }
        return h == "localhost" || h == "127.0.0.1" || h == "[::1]"
    }

    /// Matches an `Origin` against allowlist patterns (`*`, exact, or `scheme://host:*`).
    static func originAllowed(_ origin: String, patterns: [String]) -> Bool {
        let o = origin.trimmingCharacters(in: .whitespaces).lowercased()
        for raw in patterns {
            let p = raw.trimmingCharacters(in: .whitespaces).lowercased()
            if p.isEmpty { continue }
            if p == "*" { return true }
            if p == o { return true }
            if p.hasSuffix(":*") {
                let base = String(p.dropLast(2))
                if o == base { return true }
                if o.hasPrefix(base + ":") {
                    let port = o.dropFirst(base.count + 1)
                    if !port.isEmpty, port.count <= 5, port.allSatisfy({ $0.isASCII && $0.isNumber }) { return true }
                }
            }
        }
        return false
    }

    static func constantTimeEqual(_ a: String, _ b: String) -> Bool {
        let x = Array(a.utf8), y = Array(b.utf8)
        var diff = UInt8(x.count == y.count ? 0 : 1)
        for i in 0..<max(x.count, y.count) {
            diff |= (i < x.count ? x[i] : 0) ^ (i < y.count ? y[i] : 0)
        }
        return diff == 0
    }

    static func errorBody(_ message: String, type: String, param: String? = nil, code: String? = nil) -> [String: Any] {
        ["error": ["message": message, "type": type, "param": param.map { $0 as Any } ?? NSNull(), "code": code.map { $0 as Any } ?? NSNull()] as [String: Any]]
    }

    static func json(_ obj: Any) -> Data {
        (try? JSONSerialization.data(withJSONObject: obj, options: [.withoutEscapingSlashes])) ?? Data("{}".utf8)
    }
}

public enum OpenAIServerError: Error, LocalizedError {
    case invalidPort
    case apiKeyRequired
    case startTimeout
    public var errorDescription: String? {
        switch self {
        case .invalidPort: return "Invalid port"
        case .apiKeyRequired: return "Serving on the local network requires an API key (set one, or the MLXR_API_KEY environment variable)"
        case .startTimeout: return "The server did not start listening in time"
        }
    }
}

/// An OpenAI-style error response.
struct APIError: Error {
    var status: Int
    var message: String
    var type = "invalid_request_error"
    var param: String?
    var code: String?
    var headers: [String: String] = [:]

    init(status: Int, message: String, type: String = "invalid_request_error", param: String? = nil,
         code: String? = nil, headers: [String: String] = [:]) {
        self.status = status
        self.message = message
        self.type = type
        self.param = param
        self.code = code
        self.headers = headers
    }

    static func invalid(_ message: String, param: String? = nil, code: String? = nil) -> APIError {
        APIError(status: 400, message: message, param: param, code: code)
    }
    static let noModel = APIError(status: 503, message: "No model is loaded", type: "server_error", code: "model_not_loaded")
    /// Internal marker: the client disconnected, nothing is sent.
    static let clientGone = APIError(status: 499, message: "Client closed the connection")

    var body: [String: Any] { OpenAIServer.errorBody(message, type: type, param: param, code: code) }
}

/// Typed accessors over a decoded JSON object body.
struct JSONBody {
    let raw: [String: Any]

    init(_ data: Data) throws {
        guard !data.isEmpty else { throw APIError.invalid("Request body is empty; expected a JSON object") }
        guard let obj = try? JSONSerialization.jsonObject(with: data), let d = obj as? [String: Any] else {
            throw APIError.invalid("Request body must be a valid JSON object")
        }
        raw = d
    }

    func has(_ k: String) -> Bool { raw[k] != nil && !(raw[k] is NSNull) }

    func number(_ k: String) throws -> Double? {
        guard has(k) else { return nil }
        guard let d = OpenAIServer.number(raw[k]) else { throw APIError.invalid("'\(k)' must be a number", param: k) }
        return d
    }

    func bool(_ k: String) throws -> Bool? {
        guard has(k) else { return nil }
        guard let n = raw[k] as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else {
            throw APIError.invalid("'\(k)' must be a boolean", param: k)
        }
        return n.boolValue
    }

    func string(_ k: String) throws -> String? {
        guard has(k) else { return nil }
        guard let s = raw[k] as? String else { throw APIError.invalid("'\(k)' must be a string", param: k) }
        return s
    }
}

final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: T
    init(_ v: T) { _value = v }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return _value }
        set { lock.lock(); _value = newValue; lock.unlock() }
    }
}

final class OnceFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    /// True exactly once.
    func claim() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if done { return false }
        done = true
        return true
    }
}

// MARK: - Reasoning split

/// Separates a leading `<think>…</think>` block from the reply while streaming, holding back
/// partial tags across chunks. Output that does not start with `<think>` is passed through.
struct ReasoningSplitter {
    private enum State { case start, thinking, content }
    static let open = "<think>"
    static let close = "</think>"
    private var state: State
    private var pending = ""
    private var trimContentLead = false
    private var trimReasoningLead = true

    init(enabled: Bool) { state = enabled ? .start : .content }

    mutating func feed(_ t: String) -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        pending += t
        if state == .start {
            let trimmed = pending.drop(while: { $0.isWhitespace })
            if trimmed.hasPrefix(Self.open) {
                state = .thinking
                pending = String(trimmed.dropFirst(Self.open.count))
            } else if Self.open.hasPrefix(trimmed) {
                return ("", "")  // could still become the opening tag
            } else {
                state = .content
            }
        }
        if state == .thinking {
            if let r = pending.range(of: Self.close) {
                reasoning = String(pending[..<r.lowerBound])
                pending = String(pending[r.upperBound...])
                state = .content
                trimContentLead = true
            } else {
                var hold = 0
                for len in stride(from: min(Self.close.count - 1, pending.count), to: 0, by: -1)
                where pending.hasSuffix(Self.close.prefix(len)) {
                    hold = len
                    break
                }
                let cut = pending.index(pending.endIndex, offsetBy: -hold)
                reasoning = String(pending[..<cut])
                pending = String(pending[cut...])
            }
            if trimReasoningLead {
                reasoning = String(reasoning.drop(while: { $0.isWhitespace }))
                if !reasoning.isEmpty { trimReasoningLead = false }
            }
        }
        if state == .content {
            content = pending
            pending = ""
            if trimContentLead {
                content = String(content.drop(while: { $0.isWhitespace }))
                if !content.isEmpty { trimContentLead = false }
            }
        }
        return (reasoning, content)
    }

    mutating func flush() -> (reasoning: String, content: String) {
        defer { pending = "" }
        switch state {
        case .thinking:
            var r = pending
            if trimReasoningLead { r = String(r.drop(while: { $0.isWhitespace })) }
            return (r, "")
        case .start, .content:
            return ("", trimContentLead ? String(pending.drop(while: { $0.isWhitespace })) : pending)
        }
    }
}

// MARK: - Stop sequences

/// Applies stop strings to streamed text, holding back any suffix that could begin a stop sequence.
struct StopFilter {
    let stops: [String]
    private var pending = ""

    init(stops: [String]) { self.stops = stops }

    /// Returns text safe to emit and whether a stop sequence was hit.
    mutating func feed(_ t: String) -> (String, Bool) {
        guard !stops.isEmpty else { return (t, false) }
        pending += t
        var earliest: String.Index?
        for s in stops {
            if let r = pending.range(of: s), earliest == nil || r.lowerBound < earliest! { earliest = r.lowerBound }
        }
        if let e = earliest {
            let out = String(pending[..<e])
            pending = ""
            return (out, true)
        }
        // Hold back the longest suffix of `pending` that is a prefix of some stop string.
        var hold = 0
        for s in stops {
            let maxLen = min(s.count - 1, pending.count)
            if maxLen <= hold { continue }
            for len in stride(from: maxLen, to: hold, by: -1) where pending.hasSuffix(String(s.prefix(len))) {
                hold = len
                break
            }
        }
        let cut = pending.index(pending.endIndex, offsetBy: -hold)
        let out = String(pending[..<cut])
        pending = String(pending[cut...])
        return (out, false)
    }

    mutating func flush() -> String {
        defer { pending = "" }
        return pending
    }
}

// MARK: - Async FIFO lock

actor AsyncLock {
    private var locked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func lock() async {
        if !locked { locked = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func unlock() {
        if waiters.isEmpty { locked = false } else { waiters.removeFirst().resume() }
    }
}

// MARK: - HTTP

struct HTTPRequest: Sendable {
    let method: String
    let path: String
    let headers: [String: String]  // lowercased names
    let body: Data
}

struct HTTPError: Error {
    let status: Int
    let message: String
    var code: String?
}

/// One HTTP/1.1 request per connection; responses always use `Connection: close`.
/// Reading/parsing state is confined to `queue`.
final class HTTPConnection: @unchecked Sendable {
    struct Limits {
        var maxHeaderBytes = 32 * 1024
        var maxBodyBytes = 4 * 1024 * 1024
        var headerTimeout: TimeInterval = 10
        var bodyTimeout: TimeInterval = 30
        var writeTimeout: TimeInterval = 30
    }

    private let connection: NWConnection
    private let queue: DispatchQueue
    private let limits: Limits
    private var buffer = Data()
    private let lock = NSLock()
    private var _closed = false
    private var goneFired = false
    private var _corsOrigin: String?
    // queue-confined
    private var parseDone = false
    private var deadline: DispatchWorkItem?
    private var headersSeen = false
    var onClose: (() -> Void)?
    var onClientGone: (@Sendable () -> Void)?

    init(connection: NWConnection, queue: DispatchQueue, limits: Limits) {
        self.connection = connection
        self.queue = queue
        self.limits = limits
    }

    var isClosed: Bool { lock.lock(); defer { lock.unlock() }; return _closed }

    /// The request's allowed `Origin`, echoed in CORS headers. nil = no CORS headers.
    var corsOrigin: String? {
        get { lock.lock(); defer { lock.unlock() }; return _corsOrigin }
        set { lock.lock(); _corsOrigin = newValue; lock.unlock() }
    }

    func start(_ completion: @escaping (Result<HTTPRequest, HTTPError>) -> Void) {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled: self?.markGone()
            default: break
            }
        }
        connection.start(queue: queue)
        queue.async { self.arm(self.limits.headerTimeout, completion) }
        readRequest(completion)
    }

    /// Answers immediately without reading the request (used when over the connection cap).
    func reject(status: Int, json: Any, headers: [String: String] = [:]) {
        connection.start(queue: queue)
        sendResponse(status: status, json: json, headers: headers)
    }

    private func arm(_ seconds: TimeInterval, _ completion: @escaping (Result<HTTPRequest, HTTPError>) -> Void) {
        deadline?.cancel()
        guard !parseDone else { return }
        let item = DispatchWorkItem { [weak self] in
            guard let self, !self.parseDone else { return }
            self.parseDone = true
            completion(.failure(HTTPError(status: 408, message: "Timed out waiting for the request", code: "request_timeout")))
        }
        deadline = item
        queue.asyncAfter(deadline: .now() + seconds, execute: item)
    }

    private func finish(_ r: Result<HTTPRequest, HTTPError>, _ completion: (Result<HTTPRequest, HTTPError>) -> Void) {
        guard !parseDone else { return }
        parseDone = true
        deadline?.cancel()
        completion(r)
    }

    private func readRequest(_ completion: @escaping (Result<HTTPRequest, HTTPError>) -> Void) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self, !self.parseDone else { return }
            if let data, !data.isEmpty { self.buffer.append(data) }
            switch self.tryParse() {
            case .done(let req): self.finish(.success(req), completion); return
            case .fail(let err): self.finish(.failure(err), completion); return
            case .needMore(let headersComplete, let expectContinue):
                if headersComplete && !self.headersSeen {
                    self.headersSeen = true
                    self.arm(self.limits.bodyTimeout, completion)
                    if expectContinue {
                        self.connection.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .idempotent)
                    }
                }
            }
            if error != nil || isComplete {
                self.parseDone = true
                self.deadline?.cancel()
                self.markGone()
                return
            }
            self.readRequest(completion)
        }
    }

    private enum Parse {
        case needMore(headersComplete: Bool, expectContinue: Bool)
        case done(HTTPRequest)
        case fail(HTTPError)
    }

    private func tryParse() -> Parse {
        let sep = Data("\r\n\r\n".utf8)
        guard let headerEnd = buffer.range(of: sep) else {
            if buffer.count > limits.maxHeaderBytes { return .fail(HTTPError(status: 431, message: "Request headers too large")) }
            return .needMore(headersComplete: false, expectContinue: false)
        }
        if headerEnd.lowerBound - buffer.startIndex > limits.maxHeaderBytes {
            return .fail(HTTPError(status: 431, message: "Request headers too large"))
        }
        guard let head = String(data: buffer[buffer.startIndex..<headerEnd.lowerBound], encoding: .utf8) else {
            return .fail(HTTPError(status: 400, message: "Malformed request head"))
        }
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines[0].split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count == 3, parts[2].hasPrefix("HTTP/1."), parts[1].hasPrefix("/") || parts[1] == "*" else {
            return .fail(HTTPError(status: 400, message: "Malformed request line"))
        }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":"), line.first != " ", line.first != "\t" else {
                return .fail(HTTPError(status: 400, message: "Malformed header line"))
            }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            if name == "content-length", let prev = headers[name], prev != value {
                return .fail(HTTPError(status: 400, message: "Conflicting Content-Length headers"))
            }
            if name == "host", headers[name] != nil {
                return .fail(HTTPError(status: 400, message: "Duplicate Host header"))
            }
            headers[name] = value
        }
        if let te = headers["transfer-encoding"], te.lowercased() != "identity" {
            return .fail(HTTPError(status: 501, message: "Transfer-Encoding is not supported; send Content-Length"))
        }
        var length = 0
        if let cl = headers["content-length"] {
            guard cl.count <= 12, cl.allSatisfy({ $0.isASCII && $0.isNumber }), let n = Int(cl) else {
                return .fail(HTTPError(status: 400, message: "Invalid Content-Length"))
            }
            length = n
        }
        if length > limits.maxBodyBytes {
            return .fail(HTTPError(status: 413, message: "Request body too large (max \(limits.maxBodyBytes) bytes)",
                                   code: "request_too_large"))
        }
        let bodyStart = headerEnd.upperBound
        guard buffer.count - (bodyStart - buffer.startIndex) >= length else {
            let expect = headers["expect"]?.lowercased() == "100-continue"
            return .needMore(headersComplete: true, expectContinue: expect)
        }
        let body = Data(buffer[bodyStart..<(bodyStart + length)])
        return .done(HTTPRequest(method: String(parts[0]).uppercased(), path: String(parts[1]), headers: headers, body: body))
    }

    /// After the request is read, keep a receive pending so a client hang-up is noticed promptly.
    func watchForDisconnect() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if error != nil || (isComplete && (data?.isEmpty ?? true)) {
                self.markGone()
            } else if !self.isClosed {
                self.watchForDisconnect()
            }
        }
    }

    private func markGone() {
        lock.lock()
        let fire = !goneFired
        goneFired = true
        let cb = onClientGone
        let wasClosed = _closed
        _closed = true
        lock.unlock()
        if fire && !wasClosed { cb?() }
        connection.cancel()
        if !wasClosed { onClose?() }
    }

    private static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 408: return "Request Timeout"
        case 413: return "Payload Too Large"
        case 415: return "Unsupported Media Type"
        case 429: return "Too Many Requests"
        case 431: return "Request Header Fields Too Large"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        case 503: return "Service Unavailable"
        default: return "Status"
        }
    }

    private func head(status: Int, headers: [String: String], contentLength: Int?) -> Data {
        var s = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        var h = headers
        if let origin = corsOrigin {
            h["Access-Control-Allow-Origin"] = origin
            h["Vary"] = "Origin"
            h["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            if h["Access-Control-Allow-Headers"] == nil {
                h["Access-Control-Allow-Headers"] = "Content-Type, Authorization, OpenAI-Organization, OpenAI-Project, OpenAI-Beta, X-Requested-With"
            }
            h["Access-Control-Max-Age"] = "600"
        } else {
            h["Access-Control-Allow-Headers"] = nil
        }
        h["X-Content-Type-Options"] = "nosniff"
        for (k, v) in h.sorted(by: { $0.key < $1.key }) { s += "\(k): \(v)\r\n" }
        if let contentLength { s += "Content-Length: \(contentLength)\r\n" }
        s += "Connection: close\r\n\r\n"
        return Data(s.utf8)
    }

    func sendResponse(status: Int, json: Any, headers: [String: String] = [:]) {
        sendResponse(status: status, contentType: "application/json", body: OpenAIServer.json(json), headers: headers)
    }

    func sendResponse(status: Int, contentType: String?, body: Data, headers: [String: String] = [:]) {
        guard !isClosed else { return }
        var h = headers
        if let contentType { h["Content-Type"] = contentType }
        let data = head(status: status, headers: h, contentLength: body.count) + body
        let timer = DispatchWorkItem { [weak self] in self?.markGone() }
        queue.asyncAfter(deadline: .now() + limits.writeTimeout, execute: timer)
        connection.send(content: data, completion: .contentProcessed { _ in  // strong: rejected connections are owned by nobody else
            timer.cancel()
            self.close()
        })
    }

    /// Sends a response head without Content-Length (body delimited by connection close).
    func sendHead(status: Int, headers: [String: String]) async -> Bool {
        await sendRaw(head(status: status, headers: headers, contentLength: nil))
    }

    /// Sends bytes; returns false if the client is gone or the write did not finish within the write timeout.
    func sendRaw(_ data: Data) async -> Bool {
        guard !isClosed else { return false }
        return await withCheckedContinuation { cont in
            let once = OnceFlag()
            func resume(_ ok: Bool) -> Bool {
                guard once.claim() else { return false }
                cont.resume(returning: ok)
                return true
            }
            let timer = DispatchWorkItem { [weak self] in
                if resume(false) { self?.markGone() }
            }
            queue.asyncAfter(deadline: .now() + limits.writeTimeout, execute: timer)
            connection.send(content: data, completion: .contentProcessed { [weak self] error in
                timer.cancel()
                if error != nil { self?.markGone() }
                _ = resume(error == nil)
            })
        }
    }

    func close() {
        lock.lock()
        let was = _closed
        _closed = true
        lock.unlock()
        guard !was else { return }
        let timer = DispatchWorkItem { self.connection.cancel() }
        queue.asyncAfter(deadline: .now() + limits.writeTimeout, execute: timer)
        connection.send(content: nil, contentContext: .finalMessage, isComplete: true, completion: .contentProcessed { _ in
            timer.cancel()
            self.connection.cancel()
        })
        onClose?()
    }
}
