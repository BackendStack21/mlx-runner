import Foundation
import Testing
@testable import MLXEngine

// MARK: - Raw HTTP client (POSIX sockets, so Host/Origin/framing can be anything)

struct RawResponse {
    let status: Int
    let headers: [String: String]  // lowercased names
    let body: String
    var json: [String: Any]? { (try? JSONSerialization.jsonObject(with: Data(body.utf8))) as? [String: Any] }
    var error: [String: Any]? { json?["error"] as? [String: Any] }
}

/// Opens a TCP connection to 127.0.0.1:port. Returns the fd or -1.
func rawConnect(_ port: UInt16, timeout: Int = 10) -> Int32 {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { return -1 }
    var tv = timeval(tv_sec: timeout, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    var one: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let rc = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    if rc != 0 { close(fd); return -1 }
    return fd
}

func rawWrite(_ fd: Int32, _ s: String) { rawWrite(fd, Data(s.utf8)) }
func rawWrite(_ fd: Int32, _ d: Data) {
    d.withUnsafeBytes { p in
        var off = 0
        while off < d.count {
            let n = send(fd, p.baseAddress! + off, d.count - off, 0)
            if n <= 0 { return }
            off += n
        }
    }
}

/// Reads until EOF or timeout.
func rawReadAll(_ fd: Int32) -> String {
    var out = Data()
    var buf = [UInt8](repeating: 0, count: 65536)
    while true {
        let n = recv(fd, &buf, buf.count, 0)
        if n <= 0 { break }
        out.append(buf, count: n)
    }
    return String(decoding: out, as: UTF8.self)
}

func parseResponse(_ text: String) -> RawResponse {
    guard let sep = text.range(of: "\r\n\r\n") else { return RawResponse(status: 0, headers: [:], body: text) }
    let lines = text[..<sep.lowerBound].components(separatedBy: "\r\n")
    let status = Int(lines.first?.split(separator: " ").dropFirst().first ?? "") ?? 0
    var headers: [String: String] = [:]
    for l in lines.dropFirst() {
        if let c = l.firstIndex(of: ":") {
            headers[l[..<c].lowercased()] = l[l.index(after: c)...].trimmingCharacters(in: .whitespaces)
        }
    }
    return RawResponse(status: status, headers: headers, body: String(text[sep.upperBound...]))
}

/// Sends one raw request and returns the parsed response (blocking; run off the cooperative pool).
func rawRequest(_ port: UInt16, _ raw: String, timeout: Int = 180) async -> RawResponse {
    await withCheckedContinuation { cont in
        DispatchQueue.global().async {
            let fd = rawConnect(port, timeout: timeout)
            guard fd >= 0 else { cont.resume(returning: RawResponse(status: -1, headers: [:], body: "")); return }
            rawWrite(fd, raw)
            let text = rawReadAll(fd)
            close(fd)
            cont.resume(returning: parseResponse(text))
        }
    }
}

func httpRequest(_ port: UInt16, _ method: String, _ path: String, json: Any? = nil, body: String? = nil,
                 host: String? = nil, headers: [String: String] = [:]) async -> RawResponse {
    var payload = body ?? ""
    if let json { payload = String(decoding: try! JSONSerialization.data(withJSONObject: json), as: UTF8.self) }
    var s = "\(method) \(path) HTTP/1.1\r\nHost: \(host ?? "127.0.0.1:\(port)")\r\n"
    var h = headers
    if method == "POST" && h["Content-Type"] == nil { h["Content-Type"] = "application/json" }
    for (k, v) in h { s += "\(k): \(v)\r\n" }
    if method == "POST" || !payload.isEmpty { s += "Content-Length: \(payload.utf8.count)\r\n" }
    s += "\r\n" + payload
    return await rawRequest(port, s)
}

func post(_ port: UInt16, _ path: String, _ json: Any, headers: [String: String] = [:]) async -> RawResponse {
    await httpRequest(port, "POST", path, json: json, headers: headers)
}

/// Starts a server on an ephemeral loopback port with a private (unloaded) engine.
func makeServer(engine: InferenceEngine = InferenceEngine(), _ configure: (OpenAIServer) -> Void = { _ in }) throws -> OpenAIServer {
    let s = OpenAIServer(port: 0, engine: engine)
    configure(s)
    try s.start()
    return s
}

/// Asserts the OpenAI error schema: {"error": {"message", "type", "param", "code"}}.
func expectErrorSchema(_ r: RawResponse, status: Int, param: String? = nil, code: String? = nil,
                       sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(r.status == status, "body: \(r.body)", sourceLocation: sourceLocation)
    guard let e = r.error else {
        Issue.record("missing error object: \(r.body)", sourceLocation: sourceLocation)
        return
    }
    #expect(e["message"] is String, sourceLocation: sourceLocation)
    #expect(e["type"] is String, sourceLocation: sourceLocation)
    #expect(e.keys.contains("param") && e.keys.contains("code"), sourceLocation: sourceLocation)
    if let param { #expect(e["param"] as? String == param, "error: \(e)", sourceLocation: sourceLocation) }
    if let code { #expect(e["code"] as? String == code, "error: \(e)", sourceLocation: sourceLocation) }
    #expect(r.headers["content-type"] == "application/json", sourceLocation: sourceLocation)
}

let userMsg: [[String: Any]] = [["role": "user", "content": "hi"]]

// MARK: - Unit helpers

@Suite("Server helpers")
struct ServerHelperTests {
    @Test func loopbackHosts() {
        for h in ["localhost", "localhost:1234", "127.0.0.1", "127.0.0.1:80", "[::1]", "[::1]:1234", "LOCALHOST:1"] {
            #expect(OpenAIServer.isLoopbackHost(h), "\(h)")
        }
        for h in ["evil.com", "localhost.evil.com", "127.0.0.1.nip.io", "evil.com:1234", "", "localhost:", "localhost:abc",
                  "[::2]:1", "127.0.0.2", "localhost:1234@evil"] {
            #expect(!OpenAIServer.isLoopbackHost(h), "\(h)")
        }
    }

    @Test func originMatching() {
        let d = OpenAIServer.defaultAllowedOrigins
        #expect(OpenAIServer.originAllowed("http://localhost:3000", patterns: d))
        #expect(OpenAIServer.originAllowed("http://localhost", patterns: d))
        #expect(OpenAIServer.originAllowed("http://127.0.0.1:8080", patterns: d))
        #expect(!OpenAIServer.originAllowed("https://evil.com", patterns: d))
        #expect(!OpenAIServer.originAllowed("http://localhost.evil.com", patterns: d))
        #expect(!OpenAIServer.originAllowed("http://localhost:3000.evil.com", patterns: d))
        #expect(!OpenAIServer.originAllowed("null", patterns: d))
        #expect(!OpenAIServer.originAllowed("https://localhost:3000", patterns: d))
        #expect(OpenAIServer.originAllowed("https://app.example.com", patterns: ["https://app.example.com"]))
        #expect(OpenAIServer.originAllowed("https://x.dev", patterns: ["*"]))
    }

    @Test func constantTimeCompare() {
        #expect(OpenAIServer.constantTimeEqual("abc", "abc"))
        #expect(!OpenAIServer.constantTimeEqual("abc", "abd"))
        #expect(!OpenAIServer.constantTimeEqual("abc", "abcd"))
        #expect(!OpenAIServer.constantTimeEqual("", "a"))
    }

    @Test func apiKeysAreRandom() {
        let a = OpenAIServer.generateAPIKey(), b = OpenAIServer.generateAPIKey()
        #expect(a != b)
        #expect(a.hasPrefix("sk-mlx-") && a.count == 7 + 64)
    }

    @Test func reasoningSplitterHandlesChunkedTags() {
        func run(_ chunks: [String]) -> (String, String) {
            var s = ReasoningSplitter(enabled: true)
            var r = "", c = ""
            for ch in chunks { let (a, b) = s.feed(ch); r += a; c += b }
            let (a, b) = s.flush(); r += a; c += b
            return (r, c)
        }
        let full = "<think>\nLet me think.\n</think>\n\nThe answer is 4."
        #expect(run([full]) == ("Let me think.\n", "The answer is 4."))
        #expect(run(full.map { String($0) }) == ("Let me think.\n", "The answer is 4."))
        #expect(run(["<th", "ink>", "abc</th", "ink>", "x"]) == ("abc", "x"))
        #expect(run(["Hello <think> not reasoning"]) == ("", "Hello <think> not reasoning"))
        #expect(run(["<", "b>bold"]) == ("", "<b>bold"))
        #expect(run(["<think>unterminated"]) == ("unterminated", ""))
        #expect(run(["\n\n", "<think>", "r", "</think>", "c"]) == ("r", "c"))
        var off = ReasoningSplitter(enabled: false)
        #expect(off.feed("<think>x</think>y") == ("", "<think>x</think>y"))
    }

    @Test func stopFilter() {
        var f = StopFilter(stops: ["END"])
        var out = ""
        var hit = false
        for ch in ["ab", "cE", "N", "Dzz"] { let (o, h) = f.feed(ch); out += o; if h { hit = true; break } }
        #expect(out == "abc" && hit)
    }
}

// MARK: - Server without a model

@Suite("OpenAI server (no model)", .serialized)
struct ServerNoModelTests {
    @Test func healthAndModelsList() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        #expect(port != 0)
        let r = await httpRequest(port, "GET", "/")
        #expect(r.status == 200)
        #expect(r.body.contains("(none loaded)"))
        let m = await httpRequest(port, "GET", "/v1/models")
        #expect(m.status == 200)
        #expect(m.json?["object"] as? String == "list")
        #expect((m.json?["data"] as? [Any])?.isEmpty == true)
        expectErrorSchema(await httpRequest(port, "GET", "/v1/models/nope"), status: 404, code: "model_not_found")
        // No CORS headers without an Origin.
        #expect(r.headers["access-control-allow-origin"] == nil)
    }

    @Test func hostHeaderValidation() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        expectErrorSchema(await httpRequest(port, "GET", "/v1/models", host: "evil.com"), status: 403, code: "invalid_host")
        expectErrorSchema(await httpRequest(port, "GET", "/v1/models", host: "attacker.example:\(port)"), status: 403)
        #expect(await httpRequest(port, "GET", "/v1/models", host: "localhost:\(port)").status == 200)
        #expect(await httpRequest(port, "GET", "/v1/models", host: "[::1]:\(port)").status == 200)
        // Missing Host.
        let raw = await rawRequest(port, "GET /v1/models HTTP/1.1\r\n\r\n")
        #expect(raw.status == 403)
        // Duplicate Host.
        let dup = await rawRequest(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nHost: evil.com\r\n\r\n")
        #expect(dup.status == 400)
    }

    @Test func originValidationAndCORS() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        let evil = await httpRequest(port, "GET", "/v1/models", headers: ["Origin": "https://evil.com"])
        expectErrorSchema(evil, status: 403, code: "origin_not_allowed")
        #expect(evil.headers["access-control-allow-origin"] == nil)
        let evilPost = await post(port, "/v1/chat/completions", ["messages": userMsg], headers: ["Origin": "https://evil.com"])
        #expect(evilPost.status == 403)
        let nullOrigin = await httpRequest(port, "GET", "/v1/models", headers: ["Origin": "null"])
        #expect(nullOrigin.status == 403)

        let ok = await httpRequest(port, "GET", "/v1/models", headers: ["Origin": "http://localhost:3000"])
        #expect(ok.status == 200)
        #expect(ok.headers["access-control-allow-origin"] == "http://localhost:3000")
        #expect(ok.headers["vary"] == "Origin")

        let preflightEvil = await httpRequest(port, "OPTIONS", "/v1/chat/completions",
                                              headers: ["Origin": "https://evil.com", "Access-Control-Request-Method": "POST"])
        #expect(preflightEvil.status == 403)
        #expect(preflightEvil.headers["access-control-allow-origin"] == nil)
        let preflight = await httpRequest(port, "OPTIONS", "/v1/chat/completions",
                                          headers: ["Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "content-type, authorization, x-stainless-os"])
        #expect(preflight.status == 204)
        #expect(preflight.headers["access-control-allow-origin"] == "http://127.0.0.1:5173")
        #expect(preflight.headers["access-control-allow-headers"] == "content-type, authorization, x-stainless-os")

        let custom = try makeServer { $0.allowedOrigins = ["https://app.example.com"] }
        defer { custom.stop() }
        #expect(await httpRequest(custom.boundPort, "GET", "/v1/models", headers: ["Origin": "https://app.example.com"]).status == 200)
        #expect(await httpRequest(custom.boundPort, "GET", "/v1/models", headers: ["Origin": "http://localhost:3000"]).status == 403)
    }

    @Test func apiKeyAuth() async throws {
        let s = try makeServer { $0.apiKey = "sk-test-123" }; defer { s.stop() }
        let port = s.boundPort
        expectErrorSchema(await httpRequest(port, "GET", "/v1/models"), status: 401, code: "invalid_api_key")
        expectErrorSchema(await httpRequest(port, "GET", "/v1/models", headers: ["Authorization": "Bearer wrong"]), status: 401)
        expectErrorSchema(await httpRequest(port, "GET", "/", headers: ["Authorization": "Basic sk-test-123"]), status: 401)
        expectErrorSchema(await post(port, "/v1/chat/completions", ["messages": userMsg]), status: 401)
        #expect(await httpRequest(port, "GET", "/v1/models", headers: ["Authorization": "Bearer sk-test-123"]).status == 200)
        // Preflight needs no credentials.
        let pre = await httpRequest(port, "OPTIONS", "/v1/models", headers: ["Origin": "http://localhost:1"])
        #expect(pre.status == 204)
    }

    @Test func lanModeRequiresAPIKey() throws {
        guard ProcessInfo.processInfo.environment["MLXR_API_KEY"] == nil else { return }
        let s = OpenAIServer(port: 0, allowLAN: true, engine: InferenceEngine())
        #expect(throws: OpenAIServerError.self) { try s.start() }
        #expect(!s.isRunning)
    }

    @Test func routingErrors() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        expectErrorSchema(await httpRequest(port, "GET", "/v1/nope"), status: 404)
        let r405 = await httpRequest(port, "GET", "/v1/chat/completions")
        expectErrorSchema(r405, status: 405, code: "method_not_allowed")
        #expect(r405.headers["allow"]?.contains("POST") == true)
        expectErrorSchema(await httpRequest(port, "DELETE", "/v1/models"), status: 405)
        expectErrorSchema(await httpRequest(port, "GET", "/v1/responses/resp_123"), status: 404)
        expectErrorSchema(await post(port, "/v1/chat/completions", ["messages": userMsg], headers: ["Content-Type": "text/plain"]),
                          status: 415, code: "unsupported_media_type")
        expectErrorSchema(await httpRequest(port, "POST", "/v1/chat/completions", body: "{\"messages\":[]}",
                                            headers: ["Content-Type": "application/x-www-form-urlencoded"]), status: 415)
        #expect(await post(port, "/v1/chat/completions", ["messages": userMsg],
                           headers: ["Content-Type": "application/json; charset=utf-8"]).status == 503)
        expectErrorSchema(await httpRequest(port, "POST", "/v1/chat/completions", body: "{not json"), status: 400)
        expectErrorSchema(await httpRequest(port, "POST", "/v1/chat/completions", body: "[1,2]"), status: 400)
        expectErrorSchema(await httpRequest(port, "POST", "/v1/chat/completions", body: ""), status: 400)
    }

    @Test func noModelIs503() async throws {
        let s = try makeServer(); defer { s.stop() }
        expectErrorSchema(await post(s.boundPort, "/v1/chat/completions", ["messages": userMsg]), status: 503, code: "model_not_loaded")
        expectErrorSchema(await post(s.boundPort, "/v1/responses", ["input": "hi"]), status: 503, code: "model_not_loaded")
        expectErrorSchema(await post(s.boundPort, "/v1/completions", ["prompt": "hi"]), status: 503, code: "model_not_loaded")
    }

    @Test func chatValidation() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        func check(_ extra: [String: Any], _ param: String?, messages: Any = userMsg, sourceLocation: SourceLocation = #_sourceLocation) async {
            var b: [String: Any] = ["model": "x", "messages": messages]
            for (k, v) in extra { b[k] = v }
            expectErrorSchema(await post(port, "/v1/chat/completions", b), status: 400, param: param, sourceLocation: sourceLocation)
        }
        await check([:], "messages", messages: [])
        await check([:], "messages", messages: "hello")
        await check([:], "messages[0].role", messages: [["role": "wizard", "content": "x"]])
        await check([:], "messages[0].role", messages: [["role": "tool", "content": "x", "tool_call_id": "1"]])
        await check([:], "messages[0].content[0]",
                    messages: [["role": "user", "content": [["type": "image_url", "image_url": ["url": "http://x/y.png"]]]]])
        await check([:], "messages[0].content[1]",
                    messages: [["role": "user", "content": [["type": "text", "text": "a"], ["type": "input_audio"]]]])
        await check(["temperature": 2.5], "temperature")
        await check(["temperature": -0.1], "temperature")
        await check(["temperature": "hot"], "temperature")
        await check(["top_p": 0], "top_p")
        await check(["top_p": 1.5], "top_p")
        await check(["repetition_penalty": 0], "repetition_penalty")
        await check(["max_tokens": 0], "max_tokens")
        await check(["max_completion_tokens": -5], "max_completion_tokens")
        await check(["presence_penalty": 3], "presence_penalty")
        await check(["n": 2], "n")
        await check(["n": 1e300], "n")
        await check(["response_format": ["type": "json_object"]], "response_format")
        await check(["response_format": ["type": "json_schema", "json_schema": ["name": "x", "schema": [:]]]], "response_format")
        await check(["tools": [["type": "function", "function": ["name": "f"]]]], "tools")
        await check(["tool_choice": "required"], "tool_choice")
        await check(["logprobs": true], "logprobs")
        await check(["stop": 5], "stop")
        await check(["stream": "yes"], "stream")
        await check(["min_p": 2], "min_p")
        // Too many messages.
        let many = Array(repeating: ["role": "user", "content": "x"], count: 1001)
        await check([:], "messages", messages: many)
    }

    @Test func acceptedChatShapesPassValidation() async throws {
        // Without a model, a request that passes validation gets 503 (not 400).
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        let bodies: [[String: Any]] = [
            ["messages": [["role": "developer", "content": "be brief"], ["role": "user", "content": [["type": "text", "text": "hi"]]]]],
            ["messages": userMsg, "tools": [["type": "function", "function": ["name": "f"]]], "tool_choice": "none"],
            ["messages": userMsg, "response_format": ["type": "text"], "stop": ["a", "b"], "seed": 7, "n": 1,
             "frequency_penalty": 0.5, "presence_penalty": -1, "stream": true, "stream_options": ["include_usage": true],
             "user": "u", "store": false, "metadata": ["a": "b"], "logprobs": false, "reasoning_effort": "none"],
            ["messages": [["role": "assistant", "content": NSNull()], ["role": "user", "content": "x"]]],
        ]
        for b in bodies {
            let r = await post(port, "/v1/chat/completions", b)
            #expect(r.status == 503, "\(b) -> \(r.body)")
        }
    }

    @Test func hugeNumbersDoNotCrash() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        let bodies: [[String: Any]] = [
            ["messages": userMsg, "max_tokens": 1e300],
            ["messages": userMsg, "max_completion_tokens": 1e19],
            ["messages": userMsg, "top_k": 1e20],
            ["messages": userMsg, "top_k": -1],
            ["messages": userMsg, "seed": 1e19],
            ["messages": userMsg, "seed": -1e300],
            ["messages": userMsg, "repetition_penalty": 1e300],
            ["messages": userMsg, "max_tokens": 2.5],
        ]
        for b in bodies {
            let r = await post(port, "/v1/chat/completions", b)
            #expect(r.status == 503, "\(b) -> \(r.body)")  // passed validation, no model
        }
        let raw = await httpRequest(port, "POST", "/v1/chat/completions",
                                    body: #"{"messages":[{"role":"user","content":"x"}],"max_tokens":1e400,"seed":99999999999999999999999}"#)
        #expect([400, 503].contains(raw.status))
        #expect(await post(port, "/v1/responses", ["input": "x", "max_output_tokens": 1e300, "top_k": 1e20]).status == 503)
        #expect(await post(port, "/v1/completions", ["prompt": "x", "max_tokens": 1e300, "seed": 1e30]).status == 503)
        // Still alive.
        #expect(await httpRequest(port, "GET", "/").status == 200)
    }

    @Test func completionsValidation() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        expectErrorSchema(await post(port, "/v1/completions", ["prompt": ""]), status: 400, param: "prompt")
        expectErrorSchema(await post(port, "/v1/completions", ["prompt": [1, 2, 3]]), status: 400, param: "prompt")
        expectErrorSchema(await post(port, "/v1/completions", ["prompt": "x", "echo": true]), status: 400, param: "echo")
        expectErrorSchema(await post(port, "/v1/completions", ["prompt": "x", "max_tokens": 0]), status: 400, param: "max_tokens")
    }

    @Test func responsesValidation() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        func check(_ b: [String: Any], _ param: String?, sourceLocation: SourceLocation = #_sourceLocation) async {
            expectErrorSchema(await post(port, "/v1/responses", b), status: 400, param: param, sourceLocation: sourceLocation)
        }
        await check([:], "input")
        await check(["input": ""], "input")
        await check(["input": []], "input")
        await check(["input": 5], "input")
        await check(["input": "x", "previous_response_id": "resp_1"], "previous_response_id")
        await check(["input": "x", "conversation": "conv_1"], "conversation")
        await check(["input": "x", "background": true], "background")
        await check(["input": "x", "tools": [["type": "web_search"]]], "tools")
        await check(["input": "x", "text": ["format": ["type": "json_schema", "name": "x", "schema": [:]]]], "text.format")
        await check(["input": "x", "max_output_tokens": 0], "max_output_tokens")
        await check(["input": "x", "temperature": 9], "temperature")
        await check(["input": [["role": "user", "content": [["type": "input_image", "image_url": "http://x"]]]]], "input[0].content[0]")
        await check(["input": [["type": "function_call_output", "call_id": "c", "output": "x"]]], "input[0].type")
        await check(["input": [["role": "system", "content": "only system"]]], "input")
        // Accepted shapes reach the model check.
        let ok: [[String: Any]] = [
            ["input": "hi", "instructions": "be brief", "store": true, "metadata": ["k": "v"], "include": ["reasoning.encrypted_content"]],
            ["input": [["role": "developer", "content": "x"], ["type": "message", "role": "user", "content": [["type": "input_text", "text": "hi"]]],
                       ["role": "assistant", "content": [["type": "output_text", "text": "hello"]]], ["role": "user", "content": "again"]]],
            ["input": "hi", "tools": [["type": "function", "name": "f"]], "tool_choice": "none", "reasoning": ["effort": "none"]],
        ]
        for b in ok { #expect(await post(port, "/v1/responses", b).status == 503, "\(b)") }
    }

    @Test func transferEncodingAndBodyLimits() async throws {
        let s = try makeServer(); defer { s.stop() }
        let port = s.boundPort
        let te = await rawRequest(port, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
        #expect(te.status == 501)
        let big = await rawRequest(port, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: \(5 * 1024 * 1024)\r\n\r\n")
        expectErrorSchema(big, status: 413, code: "request_too_large")
        let badCL = await rawRequest(port, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: -1\r\n\r\n")
        #expect(badCL.status == 400)
        let conflict = await rawRequest(port, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\n{}")
        #expect(conflict.status == 400)
        let hugeHeaders = await rawRequest(port, "GET / HTTP/1.1\r\nHost: localhost\r\nX-Pad: \(String(repeating: "a", count: 40_000))\r\n\r\n")
        #expect(hugeHeaders.status == 431)
        // A body just under the limit is read and parsed (then fails validation, not the size check).
        let pad = String(repeating: "a", count: 3 * 1024 * 1024)
        let nearLimit = await post(port, "/v1/chat/completions", ["messages": [["role": "user", "content": pad]], "temperature": 5])
        #expect(nearLimit.status == 400)
        #expect(nearLimit.error?["param"] as? String == "temperature")
    }

    @Test func headerAndBodyDeadlines() async throws {
        let s = try makeServer { $0.headerTimeout = 0.5; $0.bodyTimeout = 0.5 }; defer { s.stop() }
        let port = s.boundPort
        let start = Date()
        let slowHead = await rawRequest(port, "GET / HTTP/1.1\r\nHost: localhost\r\n")  // never finishes headers
        #expect(slowHead.status == 408)
        #expect(Date().timeIntervalSince(start) < 20)  // fires instead of hanging; generous for loaded CI runners
        let slowBody = await rawRequest(port, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"messages\"")
        expectErrorSchema(slowBody, status: 408, code: "request_timeout")
    }

    @Test func restartOnSamePort() async throws {
        let first = try makeServer(); let port = first.boundPort
        first.stop()
        for _ in 0..<3 {
            let s = OpenAIServer(port: port, engine: InferenceEngine())
            try s.start()
            #expect(s.boundPort == port)
            #expect(await httpRequest(port, "GET", "/").status == 200)
            s.stop()
        }
    }

    @Test func connectionCap() async throws {
        let s = try makeServer { $0.maxConnections = 2; $0.headerTimeout = 5 }; defer { s.stop() }
        let port = s.boundPort
        let a = rawConnect(port), b = rawConnect(port)
        // Close each socket exactly once: a second close() could hit a descriptor number that
        // Network.framework has reused (guarded fd → EXC_GUARD kills the process).
        var open = true
        defer { if open { close(a); close(b) } }
        #expect(a >= 0 && b >= 0)
        // Wait until the server has registered both (a fixed sleep races on slow CI runners).
        let deadline = Date().addingTimeInterval(20)
        while s.openConnections < 2, Date() < deadline { try await Task.sleep(nanoseconds: 50_000_000) }
        #expect(s.openConnections == 2)
        let third = await httpRequest(port, "GET", "/")
        expectErrorSchema(third, status: 503, code: "server_busy")
        close(a); close(b); open = false
        let freed = Date().addingTimeInterval(20)
        while s.openConnections > 0, Date() < freed { try await Task.sleep(nanoseconds: 50_000_000) }
        #expect(await httpRequest(port, "GET", "/").status == 200)
    }
}

// MARK: - Server with a real model

@Suite("OpenAI server (Qwen3 0.6B)", .serialized)
struct ServerModelTests {
    static let repo = "mlx-community/Qwen3-0.6B-4bit"
    static let engine = InferenceEngine()

    static func loadedEngine() async throws -> InferenceEngine {
        if engine.info == nil {
            _ = try await engine.load(directory: installedModel(repo)!, contextLength: 2048)
        }
        return engine
    }

    /// SSE `data:` payloads (excluding [DONE]) and event names.
    static func sse(_ body: String) -> (events: [String], data: [[String: Any]], done: Bool) {
        var events: [String] = [], data: [[String: Any]] = [], done = false
        for block in body.components(separatedBy: "\n\n") {
            for line in block.components(separatedBy: "\n") {
                if line.hasPrefix("event: ") { events.append(String(line.dropFirst(7))) }
                if line.hasPrefix("data: ") {
                    let payload = line.dropFirst(6)
                    if payload == "[DONE]" { done = true; continue }
                    if let o = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] { data.append(o) }
                }
            }
        }
        return (events, data, done)
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func chatNonStreamingSplitsReasoning() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let r = await post(s.boundPort, "/v1/chat/completions", [
            "model": "whatever", "messages": [["role": "developer", "content": "Answer with one word."],
                                              ["role": "user", "content": [["type": "text", "text": "What is the capital of France?"]]]],
            "max_completion_tokens": 400, "temperature": 0, "seed": 1,
        ])
        #expect(r.status == 200, "\(r.body)")
        let j = try #require(r.json)
        #expect(j["object"] as? String == "chat.completion")
        #expect((j["system_fingerprint"] as? String)?.hasPrefix("fp_") == true)
        let choice = try #require((j["choices"] as? [[String: Any]])?.first)
        let msg = try #require(choice["message"] as? [String: Any])
        let content = msg["content"] as? String ?? ""
        let reasoning = msg["reasoning_content"] as? String ?? ""
        #expect(!reasoning.isEmpty)
        #expect(!content.contains("<think>") && !content.contains("</think>"))
        #expect(content.contains("Paris"), "content: \(content)")
        #expect(choice["finish_reason"] as? String == "stop")
        let usage = try #require(j["usage"] as? [String: Any])
        #expect((usage["prompt_tokens"] as? Int ?? 0) > 0)
        #expect((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] is Int)
        let rt = (usage["completion_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int ?? -1
        #expect(rt > 0 && rt <= (usage["completion_tokens"] as? Int ?? 0))
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func chatStreamingWithUsage() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let r = await post(s.boundPort, "/v1/chat/completions", [
            "messages": [["role": "user", "content": "Say hello."]], "stream": true,
            "stream_options": ["include_usage": true], "max_tokens": 300, "temperature": 0,
        ])
        #expect(r.status == 200)
        #expect(r.headers["content-type"]?.hasPrefix("text/event-stream") == true)
        let (_, chunks, done) = Self.sse(r.body)
        #expect(done)
        let last = try #require(chunks.last)
        #expect((last["choices"] as? [Any])?.isEmpty == true)
        let usage = try #require(last["usage"] as? [String: Any])
        #expect((usage["completion_tokens"] as? Int ?? 0) > 0)
        for c in chunks.dropLast() {
            #expect(c["usage"] is NSNull)
            #expect(c["object"] as? String == "chat.completion.chunk")
        }
        let deltas = chunks.dropLast().compactMap { ($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any] }
        let content = deltas.compactMap { $0["content"] as? String }.joined()
        let reasoning = deltas.compactMap { $0["reasoning_content"] as? String }.joined()
        #expect(!reasoning.isEmpty)
        #expect(!content.isEmpty && !content.contains("<think>") && !content.contains("</think>"))
        let finishes = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }
        #expect(finishes == ["stop"])
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func chatLengthAndNoThinking() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let r = await post(s.boundPort, "/v1/chat/completions", [
            "messages": [["role": "user", "content": "Count from 1 to 100."]], "max_tokens": 5,
            "chat_template_kwargs": ["enable_thinking": false], "temperature": 0,
        ])
        let choice = try #require((r.json?["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "length")
        #expect((choice["message"] as? [String: Any])?["reasoning_content"] == nil)
        #expect((r.json?["usage"] as? [String: Any])?["completion_tokens"] as? Int == 5)
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func responsesNonStreaming() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let r = await post(s.boundPort, "/v1/responses", [
            "input": "What is 2+2? Reply with just the number.", "instructions": "Be terse.",
            "max_output_tokens": 400, "temperature": 0,
        ])
        #expect(r.status == 200, "\(r.body)")
        let j = try #require(r.json)
        #expect(j["object"] as? String == "response")
        #expect((j["id"] as? String)?.hasPrefix("resp_") == true)
        #expect(j["status"] as? String == "completed")
        #expect(j["created_at"] is Int)
        let output = try #require(j["output"] as? [[String: Any]])
        let message = try #require(output.last)
        #expect(message["type"] as? String == "message")
        #expect(message["role"] as? String == "assistant")
        let part = try #require((message["content"] as? [[String: Any]])?.first)
        #expect(part["type"] as? String == "output_text")
        #expect((part["text"] as? String)?.contains("4") == true, "\(part)")
        #expect(part["annotations"] is [Any])
        if output.count == 2 { #expect(output[0]["type"] as? String == "reasoning") }
        let usage = try #require(j["usage"] as? [String: Any])
        #expect(usage["input_tokens"] is Int && usage["output_tokens"] is Int && usage["total_tokens"] is Int)
        #expect((usage["input_tokens_details"] as? [String: Any])?["cached_tokens"] is Int)
        #expect((usage["output_tokens_details"] as? [String: Any])?["reasoning_tokens"] is Int)

        let short = await post(s.boundPort, "/v1/responses", ["input": "Write a long story.", "max_output_tokens": 4,
                                                             "reasoning": ["effort": "none"]])
        #expect(short.json?["status"] as? String == "incomplete")
        #expect((short.json?["incomplete_details"] as? [String: Any])?["reason"] as? String == "max_output_tokens")
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func responsesStreaming() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let r = await post(s.boundPort, "/v1/responses", [
            "input": [["role": "user", "content": [["type": "input_text", "text": "Say hi."]]]],
            "stream": true, "max_output_tokens": 300, "temperature": 0,
        ])
        #expect(r.status == 200)
        let (events, data, done) = Self.sse(r.body)
        #expect(!done)  // Responses streams do not end with [DONE]
        #expect(events.count == data.count)
        for (e, d) in zip(events, data) { #expect(d["type"] as? String == e) }
        #expect(data.compactMap { $0["sequence_number"] as? Int } == Array(0..<data.count))
        #expect(events.first == "response.created")
        #expect(events.dropFirst().first == "response.in_progress")
        #expect(events.last == "response.completed")
        for e in ["response.output_item.added", "response.content_part.added", "response.output_text.delta",
                  "response.output_text.done", "response.content_part.done", "response.output_item.done",
                  "response.reasoning_text.delta"] {
            #expect(events.contains(e), "missing \(e)")
        }
        let text = zip(events, data).filter { $0.0 == "response.output_text.delta" }.compactMap { $0.1["delta"] as? String }.joined()
        let doneText = zip(events, data).first { $0.0 == "response.output_text.done" }?.1["text"] as? String
        #expect(text == doneText)
        #expect(!text.contains("<think>"))
        let final = try #require(data.last?["response"] as? [String: Any])
        #expect(final["status"] as? String == "completed")
        #expect(((final["output"] as? [[String: Any]])?.last?["content"] as? [[String: Any]])?.first?["text"] as? String == text)
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func queueLimitReturns429() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()) { $0.maxQueuedRequests = 1 }; defer { s.stop() }
        let port = s.boundPort
        let long: [String: Any] = ["messages": [["role": "user", "content": "Write a long essay about the sea."]],
                                   "max_tokens": 400, "stream": true, "chat_template_kwargs": ["enable_thinking": false]]
        async let a = post(port, "/v1/chat/completions", long)
        try await Task.sleep(nanoseconds: 300_000_000)
        async let b = post(port, "/v1/chat/completions", ["messages": userMsg, "max_tokens": 2])
        try await Task.sleep(nanoseconds: 300_000_000)
        let c = await post(port, "/v1/chat/completions", ["messages": userMsg, "max_tokens": 2])
        expectErrorSchema(c, status: 429, code: "rate_limit_exceeded")
        #expect(c.headers["retry-after"] != nil)
        let (ra, rb) = await (a, b)
        #expect(ra.status == 200)
        #expect(rb.status == 200)
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func stopSequences() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let base: [String: Any] = ["messages": [["role": "user", "content": "Count from 1 to 10, separated by commas. Output only the numbers."]],
                                   "stop": ["5"], "max_tokens": 100, "temperature": 0, "chat_template_kwargs": ["enable_thinking": false]]
        let r = await post(s.boundPort, "/v1/chat/completions", base)
        #expect(r.status == 200, "\(r.body)")
        let choice = try #require((r.json?["choices"] as? [[String: Any]])?.first)
        let content = (choice["message"] as? [String: Any])?["content"] as? String ?? ""
        #expect(choice["finish_reason"] as? String == "stop")
        #expect(content.contains("4") && !content.contains("5"), "\(content)")

        var streamBody = base
        streamBody["stream"] = true
        let sr = await post(s.boundPort, "/v1/chat/completions", streamBody)
        #expect(sr.status == 200)
        let (_, chunks, done) = Self.sse(sr.body)
        #expect(done)
        #expect(!chunks.contains { $0["error"] != nil })
        let text = chunks.compactMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String }.joined()
        #expect(text == content)
        let finishes = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }
        #expect(finishes == ["stop"])
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func legacyCompletions() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let r = await post(s.boundPort, "/v1/completions", ["prompt": "The capital of France is", "max_tokens": 8, "temperature": 0])
        #expect(r.status == 200, "\(r.body)")
        #expect(r.json?["object"] as? String == "text_completion")
        let choice = try #require((r.json?["choices"] as? [[String: Any]])?.first)
        #expect((choice["text"] as? String)?.contains("Paris") == true, "\(choice)")
        #expect(choice["finish_reason"] as? String == "length")
        let sr = await post(s.boundPort, "/v1/completions", ["prompt": "The capital of France is", "max_tokens": 8, "temperature": 0,
                                                              "stream": true, "stream_options": ["include_usage": true]])
        let (_, chunks, done) = Self.sse(sr.body)
        #expect(done)
        let text = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first?["text"] as? String }.joined()
        #expect(text == choice["text"] as? String)
        #expect((chunks.last?["usage"] as? [String: Any])?["completion_tokens"] as? Int == 8)
    }

    @Test(.enabled(if: installedModel(repo) != nil))
    func clientDisconnectCancelsGeneration() async throws {
        let s = try makeServer(engine: try await Self.loadedEngine()); defer { s.stop() }
        let port = s.boundPort
        let body = String(decoding: try JSONSerialization.data(withJSONObject: [
            "messages": [["role": "user", "content": "Write a very long essay."]], "max_tokens": 2000, "stream": true,
            "chat_template_kwargs": ["enable_thinking": false]] as [String: Any]), as: UTF8.self)
        let fd = rawConnect(port)
        rawWrite(fd, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)")
        var buf = [UInt8](repeating: 0, count: 4096)
        _ = recv(fd, &buf, buf.count, 0)  // first bytes of the stream
        close(fd)
        // The slot must free up quickly (2000 tokens would take several seconds).
        let start = Date()
        let r = await post(port, "/v1/chat/completions", ["messages": userMsg, "max_tokens": 2])
        #expect(r.status == 200)
        #expect(Date().timeIntervalSince(start) < 3)
    }
}
