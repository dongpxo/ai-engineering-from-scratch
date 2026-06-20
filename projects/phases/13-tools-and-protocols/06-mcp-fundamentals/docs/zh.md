# MCP 基础：Primitives、Lifecycle、JSON-RPC Base

> MCP 之前的每个集成都是一次性方案。Model Context Protocol 最早由 Anthropic 在 2024 年 11 月发布，现在由 Linux Foundation 的 Agentic AI Foundation 托管，它标准化了 discovery 和 invocation，让任意 client 都能和任意 server 对话。2025-11-25 spec 命名了六个 primitives（三个 server primitive，三个 client primitive）、三阶段 lifecycle，以及 JSON-RPC 2.0 wire format。理解这些之后，本 phase 剩下的 MCP 章节基本就是顺着读。

**Type:** Learn
**Languages:** Python (stdlib, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01 through 05 (the tool interface and function calling)
**Time:** ~45 minutes

## 学习目标

- 说出全部六个 MCP primitives（server 侧的 tools、resources、prompts；client 侧的 roots、sampling、elicitation），并分别给出一个 use case。
- 走通三阶段 lifecycle（initialize、operation、shutdown），并说明每个阶段由谁发送哪条 message。
- 解析并生成 JSON-RPC 2.0 request、response 和 notification envelopes。
- 解释 `initialize` 中的 capability negotiation 是什么，以及没有它会破坏什么。

## 问题

MCP 之前，每个使用工具的 agent 都有自己的协议。Cursor 有一个形似 MCP 但不兼容的工具系统。Claude Desktop 发货时带着另一套。VS Code 的 Copilot extension 又有第三套。一个团队如果构建了一个 “Postgres query” 工具，就要把同一个工具写三遍，每次适配不同 host 的 API。想复用，就只能复制代码。

结果是一场一次性集成的寒武纪大爆发，也给生态速度加上了天花板。

MCP 通过标准化 wire format 修复这个问题。一个 MCP server 可以在每个 MCP client 中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf；到 2026 年 4 月已有 300+ clients。每月 SDK 下载量 110M。公开 servers 超过 10,000 个。Linux Foundation 在 2025 年 12 月通过新的 Agentic AI Foundation 接手托管。

本 phase 使用的 spec revision 是 **2025-11-25**。它加入了 async Tasks（SEP-1686）、URL-mode elicitation（SEP-1036）、sampling with tools（SEP-1577）、incremental scope consent（SEP-835），以及 OAuth 2.1 resource-indicator semantics。Phase 13 · 09 到 16 会覆盖这些扩展。本课只讲 base。

## 概念

### 三个 server primitives

1. **Tools。** 可调用动作。也就是 Phase 13 · 01 中的同一个四步循环。
2. **Resources。** 暴露的数据。通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 schemes。
3. **Prompts。** 可复用模板。Host UI 中的 slash commands；server 提供模板，client 填入参数。

### 三个 client primitives

4. **Roots。** Server 被允许触碰的一组 URI。Client 声明它们；server 尊重它们。
5. **Sampling。** Server 请求 client 的模型执行一次 completion。这样 server-hosted agent loops 不需要 server-side API keys。
6. **Elicitation。** Server 在执行中途向 client 的用户请求结构化输入。可以是 forms，也可以是 URLs（SEP-1036）。

MCP 中的每个 capability 都恰好属于这六个 primitives 之一。Phase 13 · 10 到 14 会逐个深入讲解。

### Wire format：JSON-RPC 2.0

每条 message 都是一个带这些字段的 JSON object：

- Requests: `{jsonrpc: "2.0", id, method, params}`。
- Responses: `{jsonrpc: "2.0", id, result | error}`。
- Notifications: `{jsonrpc: "2.0", method, params}`——没有 `id`，也不期待 response。

Base spec 大约有 15 个 methods，按 primitive 分组。重要的有：

- `initialize` / `initialized`（handshake）
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage`（server-to-client）
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### 三阶段 lifecycle

**Phase 1: initialize。**

Client 发送带有自身 `capabilities` 和 `clientInfo` 的 `initialize`。Server 回复自己的 `capabilities`、`serverInfo`，以及它使用的 spec version。Client 在消化完 response 后发送 `notifications/initialized`。从这里开始，双方都可以按已协商的 capabilities 发送 requests。

**Phase 2: operation。**

双向。Client 调用 `tools/list` 做 discovery，然后调用 `tools/call` 做 invocation。如果 server 声明了 sampling capability，它可以发送 `sampling/createMessage`。当 server 的工具集变化时，它可以发送 `notifications/tools/list_changed`。当用户修改 root scope 时，client 可以发送 `notifications/roots/list_changed`。

**Phase 3: shutdown。**

任意一方关闭 transport。MCP 中没有结构化 shutdown method；transport（stdio 或 Streamable HTTP，见 Phase 13 · 09）承载连接结束信号。

### Capability negotiation

`initialize` handshake 中的 `capabilities` 就是合约。一个 server 的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

这个 server 声明它可以发送 `tools/list_changed` notifications，并支持 `resources/subscribe`。Client 通过声明自己的 capabilities 来配合：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果 client 没有声明 `sampling`，server 就不能调用 `sampling/createMessage`。反过来也一样：如果 server 没有声明 `resources.subscribe`，client 就不能尝试 subscribe。

这就是防止生态漂移的机制。不支持 sampling 的 client 仍然是合法 MCP client；不调用 `sampling` 的 server 仍然是合法 MCP server。它们只是不会一起使用这个 feature。

### Structured content 和 error shapes

`tools/call` 返回一个由 typed blocks 组成的 `content` array：`text`、`image`、`resource`。Phase 13 · 14 会把 MCP Apps（`ui://` interactive UI）加入这个列表。

Errors 使用 JSON-RPC error codes。Spec 定义的扩展包括：`-32002` “Resource not found”、`-32603` “Internal error”，以及作为 `error.data` 的 MCP-specific error data。

### Client capabilities vs tool call details

一个常见混淆点是：`capabilities.tools` 表示 client 是否支持 tool-list-changed notifications。Client 是否会调用某个具体工具，是由其模型在运行时做出的选择，不是 capability flag。Capability flag 是 spec-level contract。模型的选择与它正交。

### 为什么用 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量级双向协议。REST 是 client-initiated。MCP 需要 server-initiated messages（sampling、notifications），所以带有对称 request/response 形状的 JSON-RPC 非常合适。JSON-RPC 也能干净地组合到 stdio 和 WebSocket/Streamable HTTP 上，不需要重新发明 HTTP 的 request shape。

```figure
mcp-tool-call
```

## Use It

`code/main.py` 提供一个最小 JSON-RPC 2.0 parser 和 emitter，然后手动走过 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，并打印每条 message。这里没有真实 transport，只有 message shapes。你可以和 Further Reading 中链接的 spec 对照，确认每个 envelope。

值得关注的地方：

- `initialize` 会双向声明 capabilities；response 包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` array；每个 entry 都有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- Response 的 `content` 是一个 `{type, text}` blocks array。

## Ship It

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一段 pcap-style 的 MCP client-server interaction transcript，这个 skill 会标注每条 message 属于哪个 primitive、哪个 lifecycle phase，以及依赖哪个 capability。

## Exercises

1. 运行 `code/main.py`。找出 capability negotiation 发生在哪一行，并描述如果 server 没有声明 `tools.listChanged`，会有什么变化。

2. 扩展 parser，让它能处理 `notifications/progress`。Message shape 是：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个 long-running `tools/call` 执行期间 emit 它，并确认 client handler 会显示 progress bar。

3. 从头到尾阅读 MCP 2025-11-25 spec——整份文档大约 80 页。找出大多数 servers 都不需要的那个 capability flag。提示：它和 resource subscription 有关。

4. 在纸上画出一个假想的 “cron job” feature 应该属于哪个 primitive。（提示：server 希望 client 在计划时间调用它。今天的六个 primitives 都不适合。）MCP 的 2026 roadmap 有一个关于这点的 draft SEP。

5. 从 GitHub 上的一个 open MCP server 解析一段 session log。统计 request、response、notification message 的数量。计算 lifecycle traffic 和 operation traffic 各占多少比例。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP | “Model Context Protocol” | 用于 model-to-tool discovery 和 invocation 的开放协议 |
| Server primitive | “Server 暴露了什么” | tools（actions）、resources（data）、prompts（templates） |
| Client primitive | “Client 允许 server 使用什么” | roots（scope）、sampling（LLM callbacks）、elicitation（user input） |
| JSON-RPC 2.0 | “Wire format” | 对称的 request/response/notification envelopes |
| `initialize` handshake | “Capability negotiation” | 第一组 message pair；server 和 client 声明自己支持的 features |
| `tools/list` | “Discovery” | Client 向 server 请求当前 tool set |
| `tools/call` | “Invocation” | Client 要求 server 带 arguments 执行一个工具 |
| `notifications/*_changed` | “Mutation events” | Server 告诉 client 它的 primitive list 已经变化 |
| Content block | “Typed result” | Tool result 中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | “Spec Evolution Proposal” | 命名的 draft proposal（例如用于 async Tasks 的 SEP-1686） |

## Further Reading

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — canonical spec document
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — six-primitive mental model
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月 launch post
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — one-year retrospective 和 2025-11-25 spec changes
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的摘要
