import Foundation
import Network

/// A minimal OpenAI-compatible HTTP server backed by `InferenceEngine.shared`.
///
/// Endpoints:
///   GET  /                      health text
///   GET  /v1/models             loaded model (OpenAI list format)
///   POST /v1/chat/completions   chat completions (streaming via SSE or not)
///   POST /v1/completions        raw text completions (streaming via SSE or not)
///
/// Generation requests are served one at a time in FIFO order; concurrent requests wait their turn.
public final class OpenAIServer: @unchecked Sendable {
    public let port: UInt16
    /// When false (default) the server only accepts connections on 127.0.0.1.
    public var allowLAN: Bool
    /// Called once per request with a log line: method, path, status, duration.
    public var onLog: (@Sendable (String) -> Void)?

    private let queue = DispatchQueue(label: "mlxrunner.openai-server")
    private let stateLock = NSLock()
    private var listener: NWListener?
    private var connections: [ObjectIdentifier: HTTPConnection] = [:]
    let generationLock = AsyncLock()

    public init(port: UInt16 = 1234, allowLAN: Bool = false) {
        self.port = port
        self.allowLAN = allowLAN
    }

    public var isRunning: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return listener != nil
    }

    public func start() throws {
        stateLock.lock(); defer { stateLock.unlock() }
        guard listener == nil else { return }
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { throw OpenAIServerError.invalidPort }
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        if !allowLAN {
            params.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: nwPort)
        }
        let l = allowLAN ? try NWListener(using: params, on: nwPort) : try NWListener(using: params)
        l.newConnectionHandler = { [weak self] conn in self?.accept(conn) }
        l.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed(let err):
                self?.log("server failed: \(err)")
                self?.stop()
            default: break
            }
        }
        l.start(queue: queue)
        listener = l
    }

    public func stop() {
        stateLock.lock()
        let l = listener
        listener = nil
        let conns = Array(connections.values)
        connections.removeAll()
        stateLock.unlock()
        l?.cancel()
        for c in conns { c.close() }
    }

    func log(_ line: String) { onLog?(line) }

    private func accept(_ nw: NWConnection) {
        let conn = HTTPConnection(connection: nw, queue: queue)
        let id = ObjectIdentifier(conn)
        stateLock.lock(); connections[id] = conn; stateLock.unlock()
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
                conn.sendResponse(status: err.status, json: Self.errorBody(err.message, type: "invalid_request_error"))
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

    private func route(_ req: HTTPRequest, on conn: HTTPConnection) async -> Int {
        if req.method == "OPTIONS" {
            conn.sendResponse(status: 204, contentType: nil, body: Data())
            return 204
        }
        let path = req.path.split(separator: "?", maxSplits: 1).first.map(String.init) ?? req.path
        let normalized = path.count > 1 && path.hasSuffix("/") ? String(path.dropLast()) : path
        switch (req.method, normalized) {
        case ("GET", "/"):
            let name = InferenceEngine.shared.info?.name
            let text = "MLX Runner OpenAI-compatible server is running.\nModel: \(name ?? "(none loaded)")\n"
            conn.sendResponse(status: 200, contentType: "text/plain; charset=utf-8", body: Data(text.utf8))
            return 200
        case ("GET", "/v1/models"):
            var data: [[String: Any]] = []
            if let info = InferenceEngine.shared.info {
                data.append(["id": info.name, "object": "model", "created": Int(Date().timeIntervalSince1970), "owned_by": "mlx-runner"])
            }
            conn.sendResponse(status: 200, json: ["object": "list", "data": data])
            return 200
        case ("GET", let p) where p.hasPrefix("/v1/models/"):
            let id = String(p.dropFirst("/v1/models/".count)).removingPercentEncoding ?? ""
            if let info = InferenceEngine.shared.info, info.name == id {
                conn.sendResponse(status: 200, json: ["id": info.name, "object": "model", "created": Int(Date().timeIntervalSince1970), "owned_by": "mlx-runner"])
                return 200
            }
            conn.sendResponse(status: 404, json: Self.errorBody("The model '\(id)' does not exist", type: "invalid_request_error", code: "model_not_found"))
            return 404
        case ("POST", "/v1/chat/completions"):
            return await completion(req, on: conn, chat: true)
        case ("POST", "/v1/completions"):
            return await completion(req, on: conn, chat: false)
        default:
            conn.sendResponse(status: 404, json: Self.errorBody("Unknown endpoint: \(req.method) \(path)", type: "invalid_request_error"))
            return 404
        }
    }

    // MARK: - Completions

    private func completion(_ req: HTTPRequest, on conn: HTTPConnection, chat: Bool) async -> Int {
        func bad(_ msg: String, _ status: Int = 400) -> Int {
            conn.sendResponse(status: status, json: Self.errorBody(msg, type: "invalid_request_error"))
            return status
        }
        guard let obj = try? JSONSerialization.jsonObject(with: req.body), let body = obj as? [String: Any] else {
            return bad("Request body must be a valid JSON object")
        }
        var messages: [ChatMessage] = []
        var prompt = ""
        if chat {
            guard let raw = body["messages"] as? [[String: Any]], !raw.isEmpty else {
                return bad("'messages' must be a non-empty array")
            }
            for m in raw {
                guard let roleStr = m["role"] as? String else { return bad("Each message needs a 'role'") }
                let role: ChatMessage.Role
                switch roleStr {
                case "system", "developer": role = .system
                case "user": role = .user
                case "assistant": role = .assistant
                case "tool", "function": role = .user
                default: return bad("Unsupported role '\(roleStr)'")
                }
                guard let content = Self.textContent(m["content"]) else {
                    return bad("Message content must be a string or an array of text parts")
                }
                messages.append(ChatMessage(role: role, content: content))
            }
        } else {
            if let p = body["prompt"] as? String { prompt = p }
            else if let ps = body["prompt"] as? [String], ps.count == 1 { prompt = ps[0] }
            else { return bad("'prompt' must be a string") }
        }

        var params = SamplingParameters()
        if let v = Self.number(body["temperature"]) { params.temperature = Float(v) }
        if let v = Self.number(body["top_p"]) { params.topP = Float(v) }
        if let v = Self.number(body["top_k"]) { params.topK = Int(v) }
        if let v = Self.number(body["min_p"]) { params.minP = Float(v) }
        if let v = Self.number(body["repetition_penalty"] ?? body["repeat_penalty"]) { params.repetitionPenalty = Float(v) }
        if let v = Self.number(body["max_completion_tokens"] ?? body["max_tokens"]) {
            guard v >= 1 else { return bad("'max_tokens' must be >= 1") }
            params.maxTokens = Int(v)
        } else if !chat {
            params.maxTokens = 256
        }
        if let v = Self.number(body["seed"]) { params.seed = UInt64(truncatingIfNeeded: Int64(v)) }
        var stops: [String] = []
        if let s = body["stop"] as? String { stops = [s] }
        else if let s = body["stop"] as? [String] { stops = s }
        else if body["stop"] != nil, !(body["stop"] is NSNull) { return bad("'stop' must be a string or an array of strings") }
        stops = stops.filter { !$0.isEmpty }
        let stream = (body["stream"] as? Bool) ?? false
        let includeUsage = ((body["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false
        var enableThinking = true
        if let v = body["enable_thinking"] as? Bool { enableThinking = v }
        if let kw = body["chat_template_kwargs"] as? [String: Any], let v = kw["enable_thinking"] as? Bool { enableThinking = v }

        // Serialize generation: one request at a time.
        await generationLock.lock()
        defer { Task { await generationLock.unlock() } }

        let engine = InferenceEngine.shared
        guard let info = engine.info else {
            conn.sendResponse(status: 503, json: Self.errorBody("No model is loaded", type: "server_error", code: "model_not_loaded"))
            return 503
        }
        if conn.isClosed { return 499 }

        let id = (chat ? "chatcmpl-" : "cmpl-") + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24).lowercased()
        let created = Int(Date().timeIntervalSince1970)
        let model = info.name
        let cancelToken = CancelToken()
        let events = chat ? engine.chat(messages, parameters: params, enableThinking: enableThinking, token: cancelToken)
            : engine.complete(prompt, parameters: params, token: cancelToken)
        var filter = StopFilter(stops: stops)
        var stats = GenerationStats()
        var stoppedBySequence = false

        if stream {
            // Watch for client disconnect while we stream.
            conn.onClientGone = { cancelToken.cancel() }
            guard await conn.sendHead(status: 200, headers: [
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            ]) else { cancelToken.cancel(); await Self.drain(events); return 499 }
            conn.watchForDisconnect()

            func chunk(delta: [String: Any]?, text: String?, finish: String?) -> [String: Any] {
                var choice: [String: Any] = ["index": 0, "finish_reason": finish ?? NSNull()]
                if chat { choice["delta"] = delta ?? [:] } else { choice["text"] = text ?? "" }
                if chat { choice["logprobs"] = NSNull() }
                return ["id": id, "object": chat ? "chat.completion.chunk" : "text_completion",
                        "created": created, "model": model, "choices": [choice]]
            }
            func send(_ obj: [String: Any]) async -> Bool {
                await conn.sendRaw(Data("data: ".utf8) + Self.json(obj) + Data("\n\n".utf8))
            }
            var alive = true
            if chat { alive = await send(chunk(delta: ["role": "assistant", "content": ""], text: nil, finish: nil)) }
            do {
                for try await ev in events {
                    guard alive else { cancelToken.cancel(); continue }
                    switch ev {
                    case .promptProcessed(let s): stats = s
                    case .text(let t):
                        guard !stoppedBySequence else { continue }
                        let (out, hit) = filter.feed(t)
                        if !out.isEmpty {
                            alive = await send(chunk(delta: ["content": out], text: out, finish: nil))
                        }
                        if hit { stoppedBySequence = true; cancelToken.cancel() }
                    case .finished(let s): stats = s
                    }
                }
            } catch {
                if alive {
                    _ = await send(Self.errorBody("Generation failed: \(error)", type: "server_error"))
                }
                conn.close()
                return 500
            }
            guard alive, !conn.isClosed else { conn.close(); return 499 }
            if !stoppedBySequence {
                let rest = filter.flush()
                if !rest.isEmpty { alive = await send(chunk(delta: ["content": rest], text: rest, finish: nil)) }
            }
            let finish = stoppedBySequence ? "stop" : Self.finishReason(stats.stopReason)
            _ = await send(chunk(delta: [:], text: "", finish: finish))
            if includeUsage {
                _ = await send(["id": id, "object": chat ? "chat.completion.chunk" : "text_completion",
                                "created": created, "model": model, "choices": [], "usage": Self.usage(stats)])
            }
            _ = await conn.sendRaw(Data("data: [DONE]\n\n".utf8))
            conn.close()
            return 200
        }

        // Non-streaming.
        conn.onClientGone = { cancelToken.cancel() }
        conn.watchForDisconnect()
        var text = ""
        do {
            for try await ev in events {
                switch ev {
                case .promptProcessed(let s): stats = s
                case .text(let t):
                    guard !stoppedBySequence else { continue }
                    let (out, hit) = filter.feed(t)
                    text += out
                    if hit { stoppedBySequence = true; cancelToken.cancel() }
                case .finished(let s): stats = s
                }
            }
        } catch {
            conn.sendResponse(status: 500, json: Self.errorBody("Generation failed: \(error)", type: "server_error"))
            return 500
        }
        if conn.isClosed { return 499 }
        if !stoppedBySequence { text += filter.flush() }
        let finish = stoppedBySequence ? "stop" : Self.finishReason(stats.stopReason)
        let choice: [String: Any] = chat
            ? ["index": 0, "message": ["role": "assistant", "content": text], "logprobs": NSNull(), "finish_reason": finish]
            : ["index": 0, "text": text, "logprobs": NSNull(), "finish_reason": finish]
        conn.sendResponse(status: 200, json: [
            "id": id, "object": chat ? "chat.completion" : "text_completion", "created": created, "model": model,
            "choices": [choice], "usage": Self.usage(stats),
        ])
        return 200
    }

    // MARK: - Helpers

    private static func drain(_ s: AsyncThrowingStream<GenerationEvent, Error>) async {
        do { for try await _ in s {} } catch {}
    }

    static func finishReason(_ r: String) -> String {
        switch r {
        case "length", "context": return "length"
        default: return "stop"
        }
    }

    static func usage(_ s: GenerationStats) -> [String: Any] {
        ["prompt_tokens": s.promptTokens, "completion_tokens": s.generatedTokens, "total_tokens": s.promptTokens + s.generatedTokens]
    }

    static func number(_ v: Any?) -> Double? {
        guard let n = v as? NSNumber else { return nil }
        // JSON booleans also bridge to NSNumber; reject them.
        if CFGetTypeID(n) == CFBooleanGetTypeID() { return nil }
        let d = n.doubleValue
        return d.isFinite ? d : nil
    }

    static func textContent(_ v: Any?) -> String? {
        if v == nil || v is NSNull { return "" }
        if let s = v as? String { return s }
        if let parts = v as? [Any] {
            var out: [String] = []
            for p in parts {
                if let s = p as? String { out.append(s); continue }
                guard let d = p as? [String: Any] else { return nil }
                let type = d["type"] as? String ?? "text"
                if type == "text" || type == "input_text", let t = d["text"] as? String { out.append(t) }
                else if type == "text" || type == "input_text" { return nil }
                // Non-text parts (images etc.) are ignored.
            }
            return out.joined(separator: "\n")
        }
        return nil
    }

    static func errorBody(_ message: String, type: String, code: String? = nil) -> [String: Any] {
        ["error": ["message": message, "type": type, "param": NSNull(), "code": code.map { $0 as Any } ?? NSNull()]]
    }

    static func json(_ obj: Any) -> Data {
        (try? JSONSerialization.data(withJSONObject: obj, options: [.withoutEscapingSlashes])) ?? Data("{}".utf8)
    }
}

public enum OpenAIServerError: Error, LocalizedError {
    case invalidPort
    public var errorDescription: String? { "Invalid port" }
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
}

/// One HTTP/1.1 request per connection; responses always use `Connection: close`.
final class HTTPConnection: @unchecked Sendable {
    static let maxHeaderBytes = 64 * 1024
    static let maxBodyBytes = 32 * 1024 * 1024

    private let connection: NWConnection
    private let queue: DispatchQueue
    private var buffer = Data()
    private let lock = NSLock()
    private var _closed = false
    private var goneFired = false
    var onClose: (() -> Void)?
    var onClientGone: (@Sendable () -> Void)?

    init(connection: NWConnection, queue: DispatchQueue) {
        self.connection = connection
        self.queue = queue
    }

    var isClosed: Bool { lock.lock(); defer { lock.unlock() }; return _closed }

    func start(_ completion: @escaping (Result<HTTPRequest, HTTPError>) -> Void) {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled: self?.markGone()
            default: break
            }
        }
        connection.start(queue: queue)
        readRequest(completion)
    }

    private func readRequest(_ completion: @escaping (Result<HTTPRequest, HTTPError>) -> Void) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let data, !data.isEmpty { self.buffer.append(data) }
            switch self.tryParse() {
            case .some(.success(let req)): completion(.success(req)); return
            case .some(.failure(let err)): completion(.failure(err)); return
            case .none: break
            }
            if error != nil || isComplete {
                self.close()
                return
            }
            self.readRequest(completion)
        }
    }

    /// nil = need more data.
    private func tryParse() -> Result<HTTPRequest, HTTPError>? {
        let sep = Data("\r\n\r\n".utf8)
        guard let headerEnd = buffer.range(of: sep) else {
            if buffer.count > Self.maxHeaderBytes { return .failure(HTTPError(status: 431, message: "Request headers too large")) }
            return nil
        }
        guard let head = String(data: buffer[buffer.startIndex..<headerEnd.lowerBound], encoding: .utf8) else {
            return .failure(HTTPError(status: 400, message: "Malformed request head"))
        }
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines[0].split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count == 3, parts[2].hasPrefix("HTTP/1.") else {
            return .failure(HTTPError(status: 400, message: "Malformed request line"))
        }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else {
                return .failure(HTTPError(status: 400, message: "Malformed header line"))
            }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        if let te = headers["transfer-encoding"], te.lowercased() != "identity" {
            return .failure(HTTPError(status: 501, message: "Transfer-Encoding is not supported; send Content-Length"))
        }
        var length = 0
        if let cl = headers["content-length"] {
            guard let n = Int(cl), n >= 0 else { return .failure(HTTPError(status: 400, message: "Invalid Content-Length")) }
            length = n
        }
        if length > Self.maxBodyBytes { return .failure(HTTPError(status: 413, message: "Request body too large")) }
        let bodyStart = headerEnd.upperBound
        guard buffer.count - (bodyStart - buffer.startIndex) >= length else { return nil }
        let body = Data(buffer[bodyStart..<(bodyStart + length)])
        return .success(HTTPRequest(method: String(parts[0]).uppercased(), path: String(parts[1]), headers: headers, body: body))
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
        if !wasClosed {
            connection.cancel()
            onClose?()
        }
    }

    private static let corsHeaders: [(String, String)] = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Authorization, OpenAI-Organization, OpenAI-Project, X-Requested-With"),
        ("Access-Control-Max-Age", "86400"),
    ]

    private static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 413: return "Payload Too Large"
        case 431: return "Request Header Fields Too Large"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        case 503: return "Service Unavailable"
        default: return "Status"
        }
    }

    private func head(status: Int, headers: [String: String], contentLength: Int?) -> Data {
        var s = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        for (k, v) in headers { s += "\(k): \(v)\r\n" }
        for (k, v) in Self.corsHeaders { s += "\(k): \(v)\r\n" }
        if let contentLength { s += "Content-Length: \(contentLength)\r\n" }
        s += "Connection: close\r\n\r\n"
        return Data(s.utf8)
    }

    func sendResponse(status: Int, json: Any) {
        sendResponse(status: status, contentType: "application/json", body: OpenAIServer.json(json))
    }

    func sendResponse(status: Int, contentType: String?, body: Data) {
        guard !isClosed else { return }
        var headers: [String: String] = [:]
        if let contentType { headers["Content-Type"] = contentType }
        let data = head(status: status, headers: headers, contentLength: body.count) + body
        connection.send(content: data, completion: .contentProcessed { [weak self] _ in self?.close() })
    }

    /// Sends a response head without Content-Length (body delimited by connection close).
    func sendHead(status: Int, headers: [String: String]) async -> Bool {
        await sendRaw(head(status: status, headers: headers, contentLength: nil))
    }

    func sendRaw(_ data: Data) async -> Bool {
        guard !isClosed else { return false }
        return await withCheckedContinuation { cont in
            connection.send(content: data, completion: .contentProcessed { [weak self] error in
                if error != nil { self?.markGone() }
                cont.resume(returning: error == nil)
            })
        }
    }

    func close() {
        lock.lock()
        let was = _closed
        _closed = true
        lock.unlock()
        guard !was else { return }
        connection.send(content: nil, contentContext: .finalMessage, isComplete: true, completion: .contentProcessed { [weak self] _ in
            self?.connection.cancel()
        })
        onClose?()
    }
}
