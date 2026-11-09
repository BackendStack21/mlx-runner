import Foundation
import Testing
@testable import MLXEngine

/// Qwen 3.5 tool calling. Prompts must match the model's own Jinja template; its XML calls
/// (`<function=…>` / `<parameter=…>`) must parse back into OpenAI calls. Expected strings were
/// rendered with `transformers` 5.22.1 `apply_chat_template(messages, tools=…, add_generation_prompt=True)`
/// (with and without `enable_thinking=True`) from mlx-community/Qwen3.5-2B-4bit's template. The 0.8B
/// and 4B templates have the same tool and message sections.
@Suite struct Qwen35ToolTests {
    /// A snippet that makes ChatTemplate detect what the full 2B template has: ChatML, the opt-in
    /// thinking toggle, the template-opened reasoning block and XML tool calls.
    static let template = ChatTemplate(template: #"<|im_start|> <tool_call> <function= enable_thinking is true {{- '<think>\n' }}"#,
                                       modelType: "qwen3_5", bosToken: nil)

    static var tools: [ToolDefinition] {
        guard case .array(let a) = try! JSONValue.parse(toolsJSON) else { return [] }
        return a.map { ToolDefinition(name: $0["function"]!["name"]!.stringValue!, spec: $0) }
    }

    static func cases() -> [String: [ChatMessage]] {
        let user = ChatMessage(role: .user, content: "Weather in Paris?")
        let weather = ToolCall(id: "call_1", name: "get_weather", arguments: #"{"city":"Paris"}"#)
        return [
            "user_only": [user],
            "sys_user": [ChatMessage(role: .system, content: "You are terse."), user],
            "blank_system": [ChatMessage(role: .system, content: "  "), user],
            "loop_one_call_followup": [
                ChatMessage(role: .system, content: "You are terse."),
                user,
                ChatMessage(role: .assistant, content: "", toolCalls: [weather]),
                ChatMessage(role: .tool, content: "18 C, clear", toolCallID: "call_1"),
                ChatMessage(role: .assistant, content: "It is 18 °C and clear in Paris."),
                ChatMessage(role: .user, content: "And tomorrow?"),
            ],
            "loop_two_calls": [
                ChatMessage(role: .user, content: "Weather and time in Paris?"),
                ChatMessage(role: .assistant, content: "", toolCalls: [
                    ToolCall(id: "call_1", name: "get_weather", arguments: #"{"city":"Paris","unit":"celsius"}"#),
                    ToolCall(id: "call_2", name: "get_time", arguments: #"{"timezone":"Europe/Paris"}"#),
                ]),
                ChatMessage(role: .tool, content: #"{"temp": 18, "sky": "clear"}"#, toolCallID: "call_1"),
                ChatMessage(role: .tool, content: "14:05", toolCallID: "call_2"),
            ],
            "typed_args_with_text": [
                ChatMessage(role: .user, content: "Set an alarm"),
                ChatMessage(role: .assistant, content: "Setting it now.", toolCalls: [ToolCall(
                    id: "call_1", name: "set_alarm",
                    arguments: #"{"hour":7,"minutes":30.5,"enabled":true,"tags":["work","daily"],"meta":{"snooze":false,"label":"Up"},"note":null}"#)]),
                ChatMessage(role: .tool, content: "ok", toolCallID: "call_1"),
            ],
            "reasoning_content_field": [
                user,
                ChatMessage(role: .assistant, content: "Let me check.", toolCalls: [weather], reasoning: "I should check the weather."),
                ChatMessage(role: .tool, content: "18 C", toolCallID: "call_1"),
            ],
            "reasoning_before_call": [
                user,
                ChatMessage(role: .assistant, content: "<think>\nI should check.\n</think>\n\nLet me check.", toolCalls: [weather]),
                ChatMessage(role: .tool, content: "18 C", toolCallID: "call_1"),
            ],
        ]
    }

    // MARK: Template

    @Test func detectsXMLToolCalls() {
        let t = Self.template
        #expect(t.qwen35 && t.supportsTools && t.toolCallSyntax == .xml && !t.thinksByDefault)
        // Qwen3's Hermes template keeps JSON calls.
        let q3 = ChatTemplate(template: "<|im_start|> <tool_call> enable_thinking", modelType: "qwen3", bosToken: nil)
        #expect(q3.supportsTools && q3.toolCallSyntax == .json)
        // XML calls outside a Qwen 3.5 template are not rendered, so such a model gets no tools.
        let other = ChatTemplate(template: "<|im_start|> <tool_call> <function=", modelType: "qwen3", bosToken: nil)
        #expect(!other.supportsTools)
    }

    @Test(arguments: [false, true])
    func matchesOfficialTemplate(thinking: Bool) {
        let cases = Self.cases()
        #expect(cases.count == 8)
        for (name, messages) in cases {
            let key = name + (thinking ? "+thinking" : "")
            guard let tail = Self.expectedTails[key] else { Issue.record("no reference for \(key)"); continue }
            #expect(Self.template.render(messages, enableThinking: thinking, tools: Self.tools) == Self.header + tail, "\(key)")
        }
    }

    /// A forced call closes the reasoning block (as with Hermes) and opens the call after it.
    @Test func forcedCallPrefill() throws {
        let required = try #require(ChatTemplate.toolCallPrefill(.chatML, .required, syntax: .xml))
        #expect(required == "<tool_call>\n<function=")
        let named = try #require(ChatTemplate.toolCallPrefill(.chatML, .function("get_weather"), syntax: .xml))
        #expect(named == "<tool_call>\n<function=get_weather>\n")
        let prompt = Self.template.render([ChatMessage(role: .user, content: "Weather?")], enableThinking: true,
                                          tools: Self.tools, prefill: named)
        #expect(prompt.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n"))
        // The reply continues the prefill; the gate parses both together.
        var g = ToolStreamGate(format: .chatML, prefill: named)
        #expect(g.feed("<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>") == "")
        let r = g.finish(tools: Self.tools)
        #expect(r.parsed && r.calls.map(\.name) == ["get_weather"] && r.calls.first?.arguments == #"{"city":"Paris"}"#)
    }

    // MARK: Parsing

    static func parse(_ s: String) -> ToolCallParser.Result? { ToolCallParser.parse(s, format: .chatML, tools: tools) }

    @Test func argumentsTakeTheirSchemaTypes() throws {
        let reply = """
        I'll set it.

        <tool_call>
        <function=set_alarm>
        <parameter=hour>
        7
        </parameter>
        <parameter=minutes>
        30.5
        </parameter>
        <parameter=enabled>
        True
        </parameter>
        <parameter=tags>
        ["work", "daily"]
        </parameter>
        <parameter=meta>
        {"snooze": false}
        </parameter>
        <parameter=note>
        null
        </parameter>
        </function>
        </tool_call>
        """
        let r = try #require(Self.parse(reply))
        #expect(r.content == "I'll set it.")
        #expect(r.calls.count == 1 && r.calls[0].name == "set_alarm")
        #expect(r.calls[0].arguments == #"{"hour":7,"minutes":30.5,"enabled":true,"tags":["work","daily"],"meta":{"snooze":false},"note":null}"#)
    }

    @Test func textStaysTextWhereTheSchemaSaysSoOrDoesNotParse() throws {
        let reply = "<tool_call>\n<function=set_alarm>\n<parameter=hour>\nseven\n</parameter>\n<parameter=note>\n007\nline two\n</parameter>\n"
            + "<parameter=extra>\n5\n</parameter>\n</function>\n</tool_call>"
        let r = try #require(Self.parse(reply))
        // Not an integer, a string in the schema (newlines kept), a parameter the tool does not define.
        #expect(r.calls.first?.arguments == #"{"hour":"seven","note":"007\nline two","extra":"5"}"#)
    }

    @Test func severalCallsAndHermesBodies() throws {
        let reply = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
            + "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}\n</tool_call>"
        let r = try #require(Self.parse(reply))
        #expect(r.calls.map(\.name) == ["get_weather", "get_time"])
        #expect(r.calls.map(\.arguments) == [#"{"city":"Paris"}"#, #"{"timezone":"UTC"}"#])
    }

    @Test func rejectsUnknownFunctionsAndUnfinishedCalls() {
        #expect(Self.parse("<tool_call>\n<function=launch_rockets>\n</function>\n</tool_call>") == nil)
        // Cut short (max_tokens, a stop sequence) inside a value, after a parameter, or right after the
        // function opener: not a call to run, even though what is there is well formed.
        #expect(Self.parse("<tool_call>\n<function=get_weather>\n<parameter=city>\nPar") == nil)
        #expect(Self.parse("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n") == nil)
        #expect(Self.parse("<tool_call>\n<function=get_weather>\n") == nil)
        #expect(Self.parse("<tool_call>\n<function=get_weather\n") == nil)
        // Text between the blocks is not the format.
        #expect(Self.parse("<tool_call>\n<function=get_time>\nnow\n</function>\n</tool_call>") == nil)
        // A call without arguments, and one missing only `</tool_call>`, are fine.
        #expect(Self.parse("<tool_call>\n<function=get_time>\n</function>\n</tool_call>")?.calls.first?.arguments == "{}")
        #expect(Self.parse("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>")?.calls.count == 1)
    }

    /// A value ends at its own `</parameter>`: the closing tags of the call may appear inside it.
    @Test func valuesMayHoldTheCallsClosingTags() throws {
        let note = "Ends with </function>\nthen </tool_call> and more"
        let reply = "<tool_call>\n<function=set_alarm>\n<parameter=hour>\n7\n</parameter>\n<parameter=note>\n\(note)\n</parameter>\n"
            + "</function>\n</tool_call>\nDone."
        let r = try #require(Self.parse(reply))
        #expect(r.calls.count == 1 && r.content == "Done.")
        let args = try JSONValue.parse(r.calls[0].arguments)
        #expect(args["note"]?.stringValue == note && args["hour"] == .number("7"))
    }

    @Test func gateStreamsTextBeforeTheCall() {
        var g = ToolStreamGate(format: .chatML, prefill: nil)
        var shown = ""
        for piece in ["Sure", ".\n<tool", "_call>\n<function=get_time>\n<parameter=timezone>\nUTC\n</parameter>\n</function>\n</tool_call>"] {
            shown += g.feed(piece)
        }
        #expect(shown == "Sure.\n")
        let r = g.finish(tools: Self.tools)
        #expect(r.parsed && r.calls.first?.arguments == #"{"timezone":"UTC"}"#)
    }

    /// A forced `required` call: the sampled name may only continue a defined one and ends at `>`.
    @Test func forcedNameEndsAtTheAngleBracket() {
        let vocab = ["get", "_weather", "_time", ">", ">\n", "\",", "hello", "x"]
        var by: [Character: [Int]] = [:]
        for (i, t) in vocab.enumerated() { if let c = t.first { by[c, default: []].append(i) } }
        func allowed(_ typed: String) -> Set<String> {
            Set(InferenceEngine.allowedNameTokens(["get_weather", "get_time"], typed: typed, texts: (vocab, by), end: ">").map { vocab[$0] })
        }
        #expect(allowed("") == ["get"])
        #expect(allowed("get") == ["_weather", "_time"])
        #expect(allowed("get_weather") == [">", ">\n"])
    }

    // MARK: References
    /// The tool block every case starts with (through `</IMPORTANT>`).
    static let header = [
        #"<|im_start|>system"#,
        #"# Tools"#,
        #""#,
        #"You have access to the following functions:"#,
        #""#,
        #"<tools>"#,
        #"{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}"#,
        #"{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}"#,
        #"{"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm", "parameters": {"type": "object", "properties": {"hour": {"type": "integer"}, "minutes": {"type": "number"}, "enabled": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}, "meta": {"type": "object"}, "note": {"type": ["string", "null"]}}, "required": ["hour"]}}}"#,
        #"</tools>"#,
        #""#,
        #"If you choose to call a function ONLY reply in the following format with NO suffix:"#,
        #""#,
        #"<tool_call>"#,
        #"<function=example_function_name>"#,
        #"<parameter=example_parameter_1>"#,
        #"value_1"#,
        #"</parameter>"#,
        #"<parameter=example_parameter_2>"#,
        #"This is the value for the second parameter"#,
        #"that can span"#,
        #"multiple lines"#,
        #"</parameter>"#,
        #"</function>"#,
        #"</tool_call>"#,
        #""#,
        #"<IMPORTANT>"#,
        #"Reminder:"#,
        #"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags"#,
        #"- Required parameters MUST be specified"#,
        #"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after"#,
        #"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls"#,
        #"</IMPORTANT>"#,
    ].joined(separator: "\n")

    /// What follows `header` in each reference rendering.
    static let expectedTails: [String: String] = [
        "user_only": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "user_only+thinking": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
        "sys_user": [
            #""#,
            #""#,
            #"You are terse.<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "sys_user+thinking": [
            #""#,
            #""#,
            #"You are terse.<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
        "blank_system": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "blank_system+thinking": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
        "loop_one_call_followup": [
            #""#,
            #""#,
            #"You are terse.<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"18 C, clear"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"It is 18 °C and clear in Paris.<|im_end|>"#,
            #"<|im_start|>user"#,
            #"And tomorrow?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "loop_one_call_followup+thinking": [
            #""#,
            #""#,
            #"You are terse.<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"18 C, clear"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"It is 18 °C and clear in Paris.<|im_end|>"#,
            #"<|im_start|>user"#,
            #"And tomorrow?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
        "loop_two_calls": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather and time in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"<parameter=unit>"#,
            #"celsius"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call>"#,
            #"<tool_call>"#,
            #"<function=get_time>"#,
            #"<parameter=timezone>"#,
            #"Europe/Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"{"temp": 18, "sky": "clear"}"#,
            #"</tool_response>"#,
            #"<tool_response>"#,
            #"14:05"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "loop_two_calls+thinking": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather and time in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"<parameter=unit>"#,
            #"celsius"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call>"#,
            #"<tool_call>"#,
            #"<function=get_time>"#,
            #"<parameter=timezone>"#,
            #"Europe/Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"{"temp": 18, "sky": "clear"}"#,
            #"</tool_response>"#,
            #"<tool_response>"#,
            #"14:05"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
        "typed_args_with_text": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Set an alarm<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #"Setting it now."#,
            #""#,
            #"<tool_call>"#,
            #"<function=set_alarm>"#,
            #"<parameter=hour>"#,
            #"7"#,
            #"</parameter>"#,
            #"<parameter=minutes>"#,
            #"30.5"#,
            #"</parameter>"#,
            #"<parameter=enabled>"#,
            #"True"#,
            #"</parameter>"#,
            #"<parameter=tags>"#,
            #"["work", "daily"]"#,
            #"</parameter>"#,
            #"<parameter=meta>"#,
            #"{"snooze": false, "label": "Up"}"#,
            #"</parameter>"#,
            #"<parameter=note>"#,
            #"None"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"ok"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "typed_args_with_text+thinking": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Set an alarm<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #"Setting it now."#,
            #""#,
            #"<tool_call>"#,
            #"<function=set_alarm>"#,
            #"<parameter=hour>"#,
            #"7"#,
            #"</parameter>"#,
            #"<parameter=minutes>"#,
            #"30.5"#,
            #"</parameter>"#,
            #"<parameter=enabled>"#,
            #"True"#,
            #"</parameter>"#,
            #"<parameter=tags>"#,
            #"["work", "daily"]"#,
            #"</parameter>"#,
            #"<parameter=meta>"#,
            #"{"snooze": false, "label": "Up"}"#,
            #"</parameter>"#,
            #"<parameter=note>"#,
            #"None"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"ok"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
        "reasoning_content_field": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #"I should check the weather."#,
            #"</think>"#,
            #""#,
            #"Let me check."#,
            #""#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"18 C"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "reasoning_content_field+thinking": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #"I should check the weather."#,
            #"</think>"#,
            #""#,
            #"Let me check."#,
            #""#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"18 C"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
        "reasoning_before_call": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #"I should check."#,
            #"</think>"#,
            #""#,
            #"Let me check."#,
            #""#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"18 C"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
            #"</think>"#,
            #""#,
            #""#,
        ].joined(separator: "\n"),
        "reasoning_before_call+thinking": [
            #"<|im_end|>"#,
            #"<|im_start|>user"#,
            #"Weather in Paris?<|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #"I should check."#,
            #"</think>"#,
            #""#,
            #"Let me check."#,
            #""#,
            #"<tool_call>"#,
            #"<function=get_weather>"#,
            #"<parameter=city>"#,
            #"Paris"#,
            #"</parameter>"#,
            #"</function>"#,
            #"</tool_call><|im_end|>"#,
            #"<|im_start|>user"#,
            #"<tool_response>"#,
            #"18 C"#,
            #"</tool_response><|im_end|>"#,
            #"<|im_start|>assistant"#,
            #"<think>"#,
            #""#,
        ].joined(separator: "\n"),
    ]

    static let toolsJSON = #"[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}, {"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}, {"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm", "parameters": {"type": "object", "properties": {"hour": {"type": "integer"}, "minutes": {"type": "number"}, "enabled": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}, "meta": {"type": "object"}, "note": {"type": ["string", "null"]}}, "required": ["hour"]}}}]"#
}
