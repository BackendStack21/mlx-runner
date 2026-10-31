import Foundation
import Testing
@testable import MLXEngine

/// Tool-calling prompts must match the models' own Jinja chat templates. Expected strings were
/// rendered with `transformers` `apply_chat_template(messages, tools=…, add_generation_prompt=True)`
/// from the official Qwen2.5-0.5B-Instruct, Qwen3-0.6B and Llama-3.2-1B-Instruct templates.
@Suite struct ToolTemplateTests {
    // Snippets that make ChatTemplate detect the same format and features as the full templates.
    static let qwen25 = ChatTemplate(template: #"<|im_start|> <tool_call> {{\"name\": <function-name> You are Qwen, created by Alibaba Cloud. You are a helpful assistant."#,
                                     modelType: "qwen2", bosToken: nil)
    static let qwen3 = ChatTemplate(template: "<|im_start|> <tool_call> enable_thinking", modelType: "qwen3", bosToken: nil)
    static let llama32 = ChatTemplate(template: "<|start_header_id|> Cutting Knowledge Date ipython", modelType: "llama",
                                      bosToken: "<|begin_of_text|>")
    static let mistral = ChatTemplate(template: "[INST] [AVAILABLE_TOOLS]", modelType: "mistral", bosToken: "<s>")

    static let toolsJSON = ##"""
[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}, {"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}]
"""##

    static var tools: [ToolDefinition] {
        guard case .array(let a) = try! JSONValue.parse(toolsJSON) else { return [] }
        return a.map { ToolDefinition(name: $0["function"]!["name"]!.stringValue!, spec: $0) }
    }

    static func cases(llama: Bool) -> [String: [ChatMessage]] {
        let weather = ToolCall(id: "call_1", name: "get_weather", arguments: #"{"city":"Paris","unit":"celsius"}"#)
        let time = ToolCall(id: "call_2", name: "get_time", arguments: #"{"timezone":"Europe/Paris"}"#)
        var c: [String: [ChatMessage]] = [
            "sys_user": [ChatMessage(role: .system, content: "You are terse."), ChatMessage(role: .user, content: "Weather in Paris?")],
            "user_only": [ChatMessage(role: .user, content: "Weather in Paris?")],
            "loop_one_call_followup": [
                ChatMessage(role: .system, content: "You are terse."),
                ChatMessage(role: .user, content: "Weather in Paris?"),
                ChatMessage(role: .assistant, content: "", toolCalls: [ToolCall(id: "call_1", name: "get_weather", arguments: #"{"city":"Paris"}"#)]),
                ChatMessage(role: .tool, content: "18 C, clear", toolCallID: "call_1"),
                ChatMessage(role: .assistant, content: "It is 18 °C and clear in Paris."),
                ChatMessage(role: .user, content: "And tomorrow?"),
            ],
        ]
        if !llama {
            c["loop_two_calls"] = [
                ChatMessage(role: .user, content: "Weather and time in Paris?"),
                ChatMessage(role: .assistant, content: "", toolCalls: [weather, time]),
                ChatMessage(role: .tool, content: #"{"temp": 18, "sky": "clear"}"#, toolCallID: "call_1"),
                ChatMessage(role: .tool, content: "14:05", toolCallID: "call_2"),
            ]
        }
        return c
    }

    static let expected: [String: String] = [
        "qwen25/loop_one_call_followup": ##"""
<|im_start|>system
You are terse.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
18 C, clear
</tool_response><|im_end|>
<|im_start|>assistant
It is 18 °C and clear in Paris.<|im_end|>
<|im_start|>user
And tomorrow?<|im_end|>
<|im_start|>assistant

"""##,
        "qwen25/loop_two_calls": ##"""
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
Weather and time in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"timezone": "Europe/Paris"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp": 18, "sky": "clear"}
</tool_response>
<tool_response>
14:05
</tool_response><|im_end|>
<|im_start|>assistant

"""##,
        "qwen25/sys_user": ##"""
<|im_start|>system
You are terse.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant

"""##,
        "qwen25/user_only": ##"""
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant

"""##,
        "qwen3/loop_one_call_followup": ##"""
<|im_start|>system
You are terse.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
18 C, clear
</tool_response><|im_end|>
<|im_start|>assistant
It is 18 °C and clear in Paris.<|im_end|>
<|im_start|>user
And tomorrow?<|im_end|>
<|im_start|>assistant

"""##,
        "qwen3/loop_two_calls": ##"""
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Weather and time in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"timezone": "Europe/Paris"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp": 18, "sky": "clear"}
</tool_response>
<tool_response>
14:05
</tool_response><|im_end|>
<|im_start|>assistant

"""##,
        "qwen3/sys_user": ##"""
<|im_start|>system
You are terse.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant

"""##,
        "qwen3/user_only": ##"""
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
{"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant

"""##,
        "llama32/loop_one_call_followup": ##"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: DATE

You are terse.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. Paris"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": [
                "city"
            ]
        }
    }
}

{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Current local time — ISO 8601",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string"
                }
            },
            "required": [
                "timezone"
            ]
        }
    }
}

Weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"name": "get_weather", "parameters": {"city": "Paris"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>

"18 C, clear"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

It is 18 °C and clear in Paris.<|eot_id|><|start_header_id|>user<|end_header_id|>

And tomorrow?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


"""##,
        "llama32/sys_user": ##"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: DATE

You are terse.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. Paris"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": [
                "city"
            ]
        }
    }
}

{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Current local time — ISO 8601",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string"
                }
            },
            "required": [
                "timezone"
            ]
        }
    }
}

Weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


"""##,
        "llama32/user_only": ##"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: DATE

<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. Paris"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": [
                "city"
            ]
        }
    }
}

{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Current local time — ISO 8601",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string"
                }
            },
            "required": [
                "timezone"
            ]
        }
    }
}

Weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


"""##,
    ]

    @Test(arguments: ["qwen25", "qwen3", "llama32"])
    func matchesOfficialTemplate(_ model: String) {
        let tpl = model == "qwen25" ? Self.qwen25 : model == "qwen3" ? Self.qwen3 : Self.llama32
        let cases = Self.cases(llama: model == "llama32")
        #expect(cases.count == (model == "llama32" ? 3 : 4))
        for (name, messages) in cases {
            guard let want = Self.expected["\(model)/\(name)"] else { Issue.record("no reference for \(model)/\(name)"); continue }
            let got = tpl.render(messages, tools: Self.tools)
            #expect(got == want.replacingOccurrences(of: "Today Date: DATE", with: "Today Date: \(ChatTemplate.todayString())"),
                    "\(model)/\(name)")
        }
    }

    /// Mistral 7B v0.3's bundled template is a Python-repr variant; the reference format is
    /// mistral-common's v3 tokenizer (`MistralTokenizer.v3().encode_chat_completion`, 1.13.0),
    /// whose text marks spaces as "▁" and newline bytes as "<0x0A>" (both converted here).
    static let mistralExpected: [String: String] = [
        "user_only": ##"""
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}, {"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}][/AVAILABLE_TOOLS][INST] Weather in Paris?[/INST]
"""##,
        "sys_user": ##"""
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}, {"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}][/AVAILABLE_TOOLS][INST] You are terse.

Weather in Paris?[/INST]
"""##,
        "loop_two_calls": ##"""
<s>[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}, {"type": "function", "function": {"name": "get_time", "description": "Current local time — ISO 8601", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}][/AVAILABLE_TOOLS][INST] Weather and time in Paris?[/INST][TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}, "id": "abcdefghi"}, {"name": "get_time", "arguments": {"timezone": "Europe/Paris"}, "id": "jklmnopqr"}]</s>[TOOL_RESULTS] {"content": {"temp": 18, "sky": "clear"}, "call_id": "abcdefghi"}[/TOOL_RESULTS][TOOL_RESULTS] {"content": "14:05", "call_id": "jklmnopqr"}[/TOOL_RESULTS]
"""##,
    ]

    @Test func mistralMatchesMistralCommon() {
        let weather = ToolCall(id: "abcdefghi", name: "get_weather", arguments: #"{"city": "Paris", "unit": "celsius"}"#)
        let time = ToolCall(id: "jklmnopqr", name: "get_time", arguments: #"{"timezone": "Europe/Paris"}"#)
        let cases: [String: [ChatMessage]] = [
            "user_only": [ChatMessage(role: .user, content: "Weather in Paris?")],
            "sys_user": [ChatMessage(role: .system, content: "You are terse."), ChatMessage(role: .user, content: "Weather in Paris?")],
            "loop_two_calls": [
                ChatMessage(role: .user, content: "Weather and time in Paris?"),
                ChatMessage(role: .assistant, content: "", toolCalls: [weather, time]),
                ChatMessage(role: .tool, content: #"{"temp": 18, "sky": "clear"}"#, toolCallID: "abcdefghi"),
                ChatMessage(role: .tool, content: "14:05", toolCallID: "jklmnopqr"),
            ],
        ]
        for (name, messages) in cases {
            #expect(Self.mistral.render(messages, tools: Self.tools) == Self.mistralExpected[name], "mistral/\(name)")
        }
    }

    @Test func mistralCallIDsAreNineAlphanumerics() {
        let id = ToolCall.mistralID("call_abc123XYZ456def789ghi0")
        #expect(id.count == 9 && id.allSatisfy { $0.isLetter || $0.isNumber })
        #expect(ToolCall.mistralID("call_abc123XYZ456def789ghi0") == id)  // stable, so calls and results match
        #expect(ToolCall.mistralID("abcdefghi") == "abcdefghi")
    }
}

@Suite struct ToolParsingTests {
    static let tools = [
        ToolDefinition(name: "get_weather", spec: .object([("type", .string("function"))])),
        ToolDefinition(name: "get_time", spec: .object([("type", .string("function"))])),
    ]

    @Test func hermesCalls() throws {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"celsius\"}}\n</tool_call>\n"
            + "<tool_call>\n{\"name\": \"get_time\", \"arguments\": \"{\\\"timezone\\\": \\\"Europe/Paris\\\"}\"}\n</tool_call>"
        let r = try #require(ToolCallParser.parse(text, format: .chatML, tools: Self.tools))
        #expect(r.content == "Let me check.")
        #expect(r.calls.map(\.name) == ["get_weather", "get_time"])
        #expect(r.calls[0].arguments == #"{"city":"Paris","unit":"celsius"}"#)  // key order kept, compact like OpenAI
        #expect(r.calls[1].arguments == #"{"timezone":"Europe/Paris"}"#)        // string arguments are decoded
        #expect(r.calls.allSatisfy { $0.id.hasPrefix("call_") && $0.id.count == 29 })
        // An unterminated final block (EOS right after the JSON) still parses.
        #expect(ToolCallParser.parse("<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}", format: .chatML, tools: Self.tools)?.calls.count == 1)
    }

    @Test func rejectsUnknownAndMalformedCalls() {
        #expect(ToolCallParser.parse("<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>", format: .chatML, tools: Self.tools) == nil)
        #expect(ToolCallParser.parse("<tool_call>{\"name\": \"get_time\", \"arguments\": {</tool_call>", format: .chatML, tools: Self.tools) == nil)
        #expect(ToolCallParser.parse("<tool_call>{\"name\": \"get_time\", \"arguments\": [1]}</tool_call>", format: .chatML, tools: Self.tools) == nil)
        #expect(ToolCallParser.parse("No tools needed.", format: .chatML, tools: Self.tools) == nil)
        #expect(ToolCallParser.parse("{\"name\": \"get_time\"}", format: .phi3, tools: Self.tools) == nil)
    }

    @Test func llamaCalls() throws {
        let one = try #require(ToolCallParser.parse(" {\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}", format: .llama3, tools: Self.tools))
        #expect(one.calls.map(\.name) == ["get_weather"] && one.calls[0].arguments == #"{"city":"Paris"}"#)
        let tagged = try #require(ToolCallParser.parse("<|python_tag|>{\"name\": \"get_time\", \"parameters\": {}}; {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}",
                                                       format: .llama3, tools: Self.tools))
        #expect(tagged.calls.map(\.name) == ["get_time", "get_weather"])
        #expect(ToolCallParser.parse("{\"answer\": 42}", format: .llama3, tools: Self.tools) == nil)
        #expect(ToolCallParser.parse("Sure! {\"name\": \"get_time\"}", format: .llama3, tools: Self.tools) == nil)
    }

    @Test func mistralCalls() throws {
        let r = try #require(ToolCallParser.parse("[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}, \"id\": \"abcdefghi\"}]",
                                                  format: .mistral, tools: Self.tools))
        #expect(r.calls.map(\.name) == ["get_weather"] && r.calls[0].arguments == #"{"city":"Paris"}"#)
        #expect(ToolCallParser.parse("[{\"name\": \"get_weather\"}]", format: .mistral, tools: Self.tools) == nil)  // no marker
    }

    @Test func gateStreamsTextAndHoldsCalls() {
        var g = ToolStreamGate(format: .chatML, prefill: nil)
        var shown = ""
        for chunk in ["Checking", " now <to", "ol", "_call>\n{\"name\": \"get_time\", ", "\"arguments\": {}}\n</tool_call>"] { shown += g.feed(chunk) }
        #expect(shown == "Checking now ")
        let end = g.finish(tools: Self.tools)
        #expect(end.parsed && end.calls.map(\.name) == ["get_time"] && end.content.isEmpty)

        // A partial marker that turns out to be ordinary text is released.
        var t = ToolStreamGate(format: .chatML, prefill: nil)
        let a = t.feed("Use <too"), b = t.feed("ls> wisely")
        #expect(a + b + t.finish(tools: Self.tools).content == "Use <tools> wisely")

        // A held block that is not a valid call comes back as text.
        var bad = ToolStreamGate(format: .chatML, prefill: nil)
        #expect(bad.feed("<tool_call>oops") == "")
        let r = bad.finish(tools: Self.tools)
        #expect(!r.parsed && r.content == "<tool_call>oops")
    }

    @Test func llamaGateDecidesOnFirstCharacter() {
        var json = ToolStreamGate(format: .llama3, prefill: nil)
        #expect(json.feed("  ") == "" && json.feed("{\"name\": \"get_time\", \"parameters\": {}}") == "")
        #expect(json.finish(tools: Self.tools).calls.count == 1)
        var text = ToolStreamGate(format: .llama3, prefill: nil)
        #expect(text.feed("Hello") == "Hello" && text.feed(" there") == " there")
        #expect(!text.finish(tools: Self.tools).parsed)
    }

    @Test func forcedCallParsesPrefillPlusReply() {
        let prefill = ChatTemplate.toolCallPrefill(.chatML, .function("get_weather"))
        var g = ToolStreamGate(format: .chatML, prefill: prefill)
        #expect(g.feed("{\"city\": \"Oslo\"}}\n</tool_call>") == "")
        let r = g.finish(tools: Self.tools)
        #expect(r.parsed && r.calls.first?.name == "get_weather" && r.calls.first?.arguments == #"{"city":"Oslo"}"#)
        #expect(ChatTemplate.toolCallPrefill(.chatML, .auto) == nil)
        #expect(ChatTemplate.toolCallPrefill(.phi3, .required) == nil)
    }
}

@Suite struct JSONValueTests {
    @Test func keepsOrderAndPrintsLikePython() throws {
        let v = try JSONValue.parse(#"{"b": 1, "a": [true, null, 2.50, -0.0, 1e-5, 12345678901234567890], "é": "x\"\n\u0001\u00e9\ud83d\ude00/"}"#)
        #expect(v.serialized(.python) == "{\"b\": 1, \"a\": [true, null, 2.5, -0.0, 1e-05, 12345678901234567890], \"é\": \"x\\\"\\n\\u0001é😀/\"}")
        #expect(v.serialized(.compact) == "{\"b\":1,\"a\":[true,null,2.5,-0.0,1e-05,12345678901234567890],\"é\":\"x\\\"\\n\\u0001é😀/\"}")
        #expect(try JSONValue.parse(#"{"a": {}, "b": []}"#).serialized(.pythonIndent(4)) == "{\n    \"a\": {},\n    \"b\": []\n}")
    }

    @Test func rejectsBadInput() {
        for bad in ["", "{", "{\"a\":}", "[1,]", "01", "1.", "\"\\x\"", "\"a\nb\"", "{} {}", "tru", "\"\\ud800\"x"] {
            #expect(throws: JSONValue.ParseError.self, "\(bad)") { try JSONValue.parse(bad) }
        }
        #expect(throws: JSONValue.ParseError.tooDeep) { try JSONValue.parse(String(repeating: "[", count: 70) + String(repeating: "]", count: 70)) }
        #expect((try? JSONValue.parse("\"\\ud800\"")) == .string("\u{FFFD}"))  // a lone surrogate becomes U+FFFD
    }
}

@Suite struct ToolNameConstraintTests {
    // A toy vocabulary: id -> text.
    static let vocab = ["get", "_weather", "_time", "\"", "\",", "hello", "g", "et_t", "ime\",", "x", "", "\"}", "\", ", ",", "\",\""]
    static var index: (texts: [String], byFirst: [Character: [Int]]) {
        var by: [Character: [Int]] = [:]
        for (i, t) in vocab.enumerated() { if let c = t.first { by[c, default: []].append(i) } }
        return (vocab, by)
    }
    static func allowed(_ typed: String) -> Set<String> {
        Set(InferenceEngine.allowedNameTokens(["get_weather", "get_time"], typed: typed, texts: index).map { vocab[$0] })
    }

    @Test func onlyContinuationsOfDefinedNames() {
        #expect(Self.allowed("") == ["get", "g"])                      // "hello" (an invented name) is masked
        #expect(Self.allowed("get") == ["_weather", "_time"])
        #expect(Self.allowed("g") == ["et_t"])
        #expect(Self.allowed("get_t") == ["ime\","])
        #expect(Self.allowed("get_weather") == ["\"", "\",", "\", "])  // never `"}`: the call cannot close without arguments
        #expect(Self.allowed("get_weather\"") == [","])
        #expect(Self.allowed("got").isEmpty)                            // off every name: constraint gives up
    }
}

@Suite struct ToolLenientParsingTests {
    static let tools = ToolParsingTests.tools

    @Test func toleratesSmallModelSlips() throws {
        // Qwen2.5-0.5B copies its prompt's doubled-brace example.
        let r = try #require(ToolCallParser.parse("<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}{}\n</tool_call>",
                                                  format: .chatML, tools: Self.tools))
        #expect(r.calls.first?.arguments == #"{"city":"Paris"}"#)
        // Llama 3.2 1B sometimes echoes the definition shape.
        let n = try #require(ToolCallParser.parse("{\"type\": \"function\", \"function\": {\"name\": \"get_time\", \"parameters\": {}}}",
                                                  format: .llama3, tools: Self.tools))
        #expect(n.calls.first?.name == "get_time")
        // Still strict about anything else after the object.
        #expect(ToolCallParser.parse("<tool_call>{\"name\": \"get_time\", \"arguments\": {}} and more</tool_call>", format: .chatML, tools: Self.tools) == nil)
    }

    @Test func revealedMarkersAreHiddenWhenNotACall() {
        var g = ToolStreamGate(format: .llama3, prefill: nil)
        _ = g.feed("<|python_tag|>print('hi')")
        let r = g.finish(tools: Self.tools)
        #expect(!r.parsed && r.content == "print('hi')")
    }

    @Test func mistralToolTemplateListEnablesTools() {
        let t = ChatTemplate(template: "{{bos_token}}[INST]", modelType: "mistral", bosToken: "<s>",
                             toolTemplate: "[AVAILABLE_TOOLS] [TOOL_CALLS] [TOOL_RESULTS]")
        #expect(t.supportsTools)
        #expect(!ChatTemplate(template: "{{bos_token}}[INST]", modelType: "mistral", bosToken: "<s>").supportsTools)
    }
}
