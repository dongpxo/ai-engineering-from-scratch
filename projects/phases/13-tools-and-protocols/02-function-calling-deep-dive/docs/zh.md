# 函数调用深潜：OpenAI、Anthropic、Gemini

> 三家 frontier provider 在 2024 年收敛到了同一个 tool-call loop，然后又在其它所有地方分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` blocks。Gemini 使用 `functionDeclarations` 和 unique-id correlation。本课把三者并排 diff，让你在一个 provider 上能跑的代码，迁移到另一个 provider 时不会因为管道形状不同而崩掉。

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini function-calling payload 的三类形状差异：declaration、call、result。
- 把同一个 tool declaration 翻译成三个 provider format，并预测 strict-mode constraints 会在哪里不同。
- 在每个 provider 中使用 `tool_choice`，强制、禁止或自动选择 tool call。
- 了解每个 provider 的硬限制（tool count、schema depth、argument length），以及违反限制时各自会发出的 error signature。

## 问题

function-calling request 的形状因 provider 而异。下面是 2026 年生产栈中的三个具体例子：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是必须由你解析的 JSON string。Strict mode（`strict: true`）通过 constrained decoding 强制 schema compliance。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应返回为 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input` 已经被解析好了（是 object，不是 string）。你再用一条新的 `user` message 回复，其中包含 `{type: "tool_result", tool_use_id, content}` block。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌在 `functionDeclarations` 下）。响应到达时是 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于 parallel-call correlation。你用 `{functionResponse: {name, id, response}}` 回复。

同一个 loop。不同的 field names，不同的 nesting，不同的 string-vs-object 约定，不同的 correlation mechanisms。一个团队如果在 OpenAI 上写了 weather agent，只是为了 plumbing，移植到 Anthropic 就要花两天，再移植到 Gemini 又要花一天。

本课会构建一个 translator，把三种格式统一成一个 canonical tool declaration，并在边缘处路由。Phase 13 · 17 会把同样的模式泛化成一个 LLM gateway。

## 概念

### 共同结构

每个 provider 都需要五件事：

1. **Tool list。** 每个 tool 的 name、description 和 input schema。
2. **Tool choice。** 强制指定某个 tool、禁止 tools，或者让模型自行决定。
3. **Call emission。** 结构化输出，说明 tool 名称和参数。
4. **Call id。** 把 response 关联回正确的 call（parallel 时很重要）。
5. **Result injection。** 一条 message 或 block，把 result 绑定回对应的 call。

### 逐字段形状差异

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | assistant message 上的 `tool_calls[]` | `content[]`，类型为 `tool_use` | `parts[]`，类型为 `functionCall` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| Result block | role `tool`、`tool_call_id` | 带 `tool_result`、`tool_use_id` 的 `user` | 带 matching `id` 的 `functionResponse` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema（always enforced） | request-level `responseSchema` |

### 你真正会撞到的限制

- **OpenAI。** 每个 request 最多 128 个 tools。Schema depth 5。Argument string <= 8192 bytes。Strict mode 要求没有 `$ref`，没有重叠的 `oneOf`/`anyOf`/`allOf`，每个 property 都列在 `required` 中。
- **Anthropic。** 每个 request 最多 64 个 tools。Schema depth 实际上没有硬上限，但 practical limit 是 10。没有 strict-mode flag；schema 是 contract，模型通常会遵守。
- **Gemini。** 每个 request 最多 64 个 functions。Schema types 是 OpenAPI 3.0 subset（与 JSON Schema 2020-12 略有差异）。从 Gemini 3 开始，parallel calls 有 unique-id。

### `tool_choice` 行为

三种模式大家都支持，只是命名不同。

- **Auto。** 模型选择 tool 或 text。默认模式。
- **Required / Any。** 模型必须至少调用一个 tool。
- **None。** 模型不得调用 tools。

此外每个 provider 还有一个独有模式：

- **OpenAI。** 按 name 强制指定某个 tool。
- **Anthropic。** 按 name 强制指定某个 tool；`disable_parallel_tool_use` flag 把 single 和 multi 分开。
- **Gemini。** `mode: "VALIDATED"` 会让每个 response 都经过 schema validator，不管模型意图是什么。

### Parallel calls

OpenAI 的 `parallel_tool_calls: true`（默认）会在一条 assistant message 中发出多个 calls。你运行全部 calls，然后用一批 tool-role message 回复，每个 entry 对应一个 `tool_call_id`。Anthropic 历史上是 single-call；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用 multi。Gemini 2 允许 parallel calls，但没有稳定 id；Gemini 3 增加 UUID，因此 out-of-order responses 可以干净地 correlate。

### Streaming

三者都支持 streamed tool calls。Wire format 不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的 delta chunks 增量到达。你持续累积，直到 `finish_reason: "tool_calls"`。
- **Anthropic。** Block-start / block-delta / block-stop events。`input_json_delta` chunks 携带 partial arguments。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）会发出带 `functionCallId` 的 chunks，因此多个 parallel calls 可以 interleave。

Phase 13 · 03 会深入 parallel + streaming reassembly。本课聚焦 declaration 和 single-call shapes。

### Errors and repair

Invalid-argument errors 的外观也不一样。

- **OpenAI（non-strict）。** 模型返回 `arguments: "{bad json}"`，你的 JSON parse 失败，你注入一条 error message 并重新调用。
- **OpenAI（strict）。** Validation 在 decoding 时发生；invalid JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是 advisory。需要 server-side validate。
- **Gemini。** OpenAPI 3.0 quirk：object fields 上的 `enum` 可能被静默忽略；你仍然要自己 validate。

### Translator pattern

你代码里的 canonical tool declaration 可以长这样（形状由你决定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它翻译成三种 provider shape。`code/main.py` 里的 harness 正是这样做的，然后把每个 provider response shape 中的假 tool call round-trip 成统一格式。不需要网络；本课教的是形状，不是 HTTP。

Production teams 会把这个 translator 包在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）里。Phase 13 · 17 会交付一个 gateway，在三者任意一个之上暴露 OpenAI-shaped API。

## Use It

`code/main.py` 定义了一个 canonical `Tool` dataclass，以及三个 translators，用于输出 OpenAI、Anthropic 和 Gemini declaration JSON。它随后把每种 shape 的手写 provider response 解析成同一个 canonical call object，证明语义在表皮之下是相同的。运行它，并把三个 declarations 并排 diff。

观察重点：

- 三个 declaration blocks 只在 envelope 和 field names 上不同。
- 三个 response blocks 的差异在于 call 所在位置（top-level `tool_calls`、`content[]` block、`parts[]` entry）。
- 一个 `canonical_call()` function 会从三种 response shapes 中提取 `{id, name, args}`。

## Ship It

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某个 provider 的 function-calling integration，这个 skill 会产出 portability audit：它依赖了哪些 provider limits、哪些 fields 需要重命名，以及迁移到其它 provider 时会坏在哪里。

## 练习

1. 运行 `code/main.py`，确认三个 provider declaration JSON 都序列化自同一个底层 `Tool` object。修改 canonical tool，添加一个 enum parameter，并确认只有 Gemini translator 需要处理 OpenAPI quirk。

2. 为每个 provider 添加一个 `ListToolsResponse` parser，从模型在 `list_tools` 或 discovery call 后返回的内容中提取 tool list。OpenAI 没有原生对应物；记下这个不对称点。

3. 实现 `tool_choice` conversion：把 canonical `ToolChoice(mode="force", tool_name="x")` 映射到三种 provider shapes。然后映射 `mode="any"` 和 `mode="none"`。核对本课的 diff table。

4. 从三个 provider 中选择一个，完整阅读它的 function-calling guide。找出它的 schema spec 里其它两家不支持的一个 field。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 写一个 test vector：其 tool call arguments 违反 declared schema。把它跑过每个 provider 的 validator（Lesson 01 里的 stdlib validator 可作为 proxy），并记录触发了哪些 errors。写下你会在生产中选择哪个 provider 来获得 strictness。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | “Tool use” | 用于结构化 tool-call emission 的 provider-level API |
| Tool declaration | “Tool spec” | Name + description + JSON Schema input payload |
| `tool_choice` | “Force / forbid” | Auto / required / none / specific-name modes |
| Strict mode | “Schema enforcement” | OpenAI flag，把 decoding 约束到匹配 schema |
| `tool_use` block | “Anthropic's call shape” | 带 id、name、input 的 inline content block |
| `functionCall` part | “Gemini's call shape” | 包含 name、args 和 id 的 `parts[]` entry |
| Arguments-as-string | “Stringified JSON” | OpenAI 以 JSON string 而非 object 返回 args |
| Parallel tool calls | “Fan-out in one turn” | 一条 assistant message 里的多个 tool calls |
| Refusal | “Model declines” | strict-mode-only refusal block，而不是 call |
| OpenAPI 3.0 subset | “Gemini schema quirk” | Gemini 使用的类 JSON Schema dialect，存在少量差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — canonical reference，包括 strict mode 和 parallel calls
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` block 语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — parallel calls、unique ids 和 OpenAPI subset
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的 enterprise surface
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict-mode schema enforcement details
