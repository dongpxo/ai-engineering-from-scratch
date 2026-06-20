# Model Context Protocol (MCP)

> 2025 年之前构建的每个 LLM 应用，都发明了自己的工具 schema。然后 Anthropic 发布 MCP，Claude 采用它，OpenAI 采用它，到 2026 年，它已经成为连接任意 LLM 与任意工具、数据源或 agent 的默认线缆格式。写一个 MCP server，每个 host 都能和它对话。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**Time:** ~75 minutes

## 问题

你发布了一个聊天机器人，它需要三个工具：数据库查询、日历 API 和文件读取器。你为 Claude 写了三个 JSON schema。然后销售团队希望在 ChatGPT 中使用同样的工具，于是你又为 OpenAI 的 `tools` 参数重写一遍。接着你加入 Cursor、Zed 和 Claude Code，又要重写三遍，每一种 JSON 约定都有细微差别。一周后，Anthropic 加了一个新字段，你要更新六份 schema。

这就是 2025 年之前的现实。每个 host（运行 LLM 的东西）和每个 server（暴露工具和数据的东西）都有定制协议。规模化意味着一个 N×M 的集成矩阵。

Model Context Protocol 把这个矩阵压扁。一个基于 JSON-RPC 的规范。一个 server 暴露 tools、resources 和 prompts。任何符合规范的 host——Claude Desktop、ChatGPT、Cursor、Claude Code、Zed，以及大量 agent framework——都可以发现并调用它们，而不需要自定义胶水代码。

截至 2026 年初，MCP 已经成为三大厂商（Anthropic、OpenAI、Google）和每个主流 agent harness 的默认工具与上下文协议。

## 概念

![MCP：一个 host，一个 server，三种能力](../assets/mcp-architecture.svg)

**三个 primitive。** 一个 MCP server 精确暴露三类东西。

1. **Tools** — 模型可以调用的函数。对应 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个 tool 都有名称、描述、JSON Schema 输入和 handler。
2. **Resources** — 模型或用户可以请求的只读内容（文件、数据库行、API 响应）。通过 URI 寻址。
3. **Prompts** — 用户可以作为快捷方式调用的可复用模板化提示。

**线缆格式。** JSON-RPC 2.0，运行在 stdio、WebSocket 或 streamable HTTP 之上。每条消息都是 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法包括 `tools/list`、`resources/list`、`prompts/list`。调用方法包括 `tools/call`、`resources/read`、`prompts/get`。

**Host vs client vs server。** Host 是 LLM 应用（Claude Desktop）。Client 是 host 内部的子组件，只和一个 server 对话。Server 是你的代码。一个 host 可以同时挂载多个 server。

### 握手

每个 session 都以 `initialize` 开始。Client 发送协议版本和自身 capabilities。Server 返回自己的版本、名称，以及它支持的 capability 集合（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的一切都基于这些 capability 协商。

### MCP 不是什么

- 它不是 retrieval API。RAG（Phase 11 · 06）仍然决定拉取什么；MCP 是把检索结果作为 resource 暴露出来的传输层。
- 它不是 agent framework。MCP 是管道；LangGraph、PydanticAI 和 OpenAI Agents SDK 这类 framework 位于它之上。
- 它不绑定 Anthropic。规范和参考实现都在 `modelcontextprotocol` org 下开源。

## Build It

### 第 1 步：一个最小 MCP server

官方 Python SDK 是 `mcp`（以前叫 `mcp-python`）。高层 `FastMCP` helper 使用装饰器注册 handler。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个装饰器注册三种 primitive。类型提示会变成 host 看到的 JSON Schema。把 server entry 指向这个文件，就可以在 Claude Desktop 或 Claude Code 下运行它。

### 第 2 步：从 host 调用 MCP server

官方 Python client 会说 JSON-RPC。把它和 Anthropic SDK 配合起来，只需要十几行。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回的 schema，和 LLM 会看到的是同一份。生产 host 会把这些 schema 注入每一轮对话，让模型可以发出 `tool_use` block，然后由 client 转发给 server。

### 第 3 步：streamable HTTP transport

Stdio 适合本地开发。远程工具使用 streamable HTTP：每个 request 一个 POST，可选 Server-Sent Events 用于进度；这个 transport 从 2025-06-18 版规范开始支持。

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Host 配置（Claude Desktop 的 `mcp.json` 或 Claude Code 的 `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Server 仍然使用相同装饰器；只有 transport 发生变化。

### 第 4 步：作用域与安全

MCP tool 是运行在别人信任边界上的任意代码。三种 mandatory pattern：

- **Capability allowlists。** Host 暴露 `roots` capability，让 server 只能看到允许的路径。在 tool handler 中强制执行它；不要信任模型提供的路径。
- **Mutation 需要 human-in-the-loop。** 只读工具可以自动执行。写入/删除工具必须要求确认；当 server 在 tool metadata 上设置 `destructiveHint: true` 时，host 会展示审批 UI。
- **Tool poisoning 防御。** 恶意 resource 可以包含隐藏 prompt-injection 指令（“when summarizing, also call `exfil`”）。把 resource 内容视为不可信数据；永远不要让它进入 system-message 领地。参见 Phase 11 · 12 (Guardrails)。

`code/main.py` 中有一个可运行的 server + client 对，演示了所有这些模式。

## 2026 年仍然会真实上线的坑

- **Schema drift。** 模型在第 1 轮看到 `tools/list`。第 5 轮工具集变了。模型调用了已经消失的 tool。Host 应该在收到 `notifications/tools/list_changed` 时重新 list。
- **大型 resource blob。** 把 2MB 文件作为 resource 一次性倒进上下文，会浪费 context。应在 server 侧分页或总结。
- **Server 太多。** 挂载 50 个 MCP server 会打爆工具预算（Phase 11 · 05）。多数 frontier model 在超过约 40 个工具后会退化。
- **版本偏移。** 规范修订版（2024-11、2025-03、2025-06、2025-12）会引入 breaking fields。要在 CI 中 pin protocol version。
- **Stdio 死锁。** 向 stdout 写日志的 server 会破坏 JSON-RPC stream。日志只能写到 stderr。

## Use It

2026 年的 MCP stack：

| Situation | Pick |
|-----------|------|
| 本地开发，单用户工具 | Python `FastMCP`，stdio transport |
| 远程团队工具 / SaaS 集成 | Streamable HTTP，OAuth 2.1 auth |
| TypeScript host（VS Code extension、web app） | `@modelcontextprotocol/sdk` |
| 高吞吐 server，类型化访问 | 官方 Rust SDK（`modelcontextprotocol/rust-sdk`） |
| 探索生态 server | `modelcontextprotocol/servers` monorepo（Filesystem、GitHub、Postgres、Slack、Puppeteer） |

经验法则：如果一个工具是只读、可缓存，并且会被两个或更多 host 调用，就把它发布为 MCP server。如果它只是一次性的内联逻辑，就保留为本地函数（Phase 11 · 09）。

## Ship It

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## 练习

1. **Easy。** 为 `demo-server` 增加一个 `subtract` tool。从 Claude Desktop 连接它。通过发送 `tools/list_changed` notification，确认 host 不重启也能发现新工具。
2. **Medium。** 增加一个 `resource`，暴露 `/var/log/app.log` 的最后 100 行。强制 roots allowlist，即使模型请求 `../etc/passwd` 也要阻止。
3. **Hard。** 构建一个 MCP proxy，把三个上游 server（Filesystem、GitHub、Postgres）复用到一个聚合 surface 中。处理名称冲突，并干净地转发 `notifications/tools/list_changed`。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP | “LLM 的工具协议” | JSON-RPC 2.0 规范，用于向任意 LLM host 暴露 tools、resources 和 prompts。 |
| Host | “Claude Desktop” | LLM 应用；拥有模型和用户 UI，挂载一个或多个 client。 |
| Client | “Connection” | Host 内部每个 server 对应的一条连接，只和一个 server 通过 JSON-RPC 对话。 |
| Server | “有工具的那个东西” | 你的代码；宣告 tools/resources/prompts，并处理它们的调用。 |
| Tool | “Function call” | 模型可调用的 action，带 JSON Schema 输入和 text/JSON 结果。 |
| Resource | “Read-only data” | 通过 URI 寻址的内容（文件、行、API 响应），host 可以请求它。 |
| Prompt | “Saved prompt” | 用户可调用的模板（通常带参数），经常以 slash-command 形式暴露。 |
| Stdio transport | “Local dev mode” | 父 host 把 server 作为子进程启动；JSON-RPC 通过 stdin/stdout 传输。 |
| Streamable HTTP | “2025-06 远程 transport” | 用 POST 处理 request，可选 SSE 处理 server 主动消息；替代旧的 SSE-only transport。 |

## 延伸阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 按日期版本化的规范权威参考。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem、GitHub、Postgres、Slack、Puppeteer 参考 server。
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — 发布文章，包含设计动机。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 本课使用的官方 SDK。
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — roots、destructive hints、tool poisoning。
- [Google A2A specification](https://google.github.io/A2A/) — Agent2Agent protocol；它是 agent-to-agent 通信的兄弟标准，补充 MCP 的 agent-to-tool 范围。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — MCP 在更大 agent 设计模式库中的位置（augmented LLM、workflows、autonomous agents）。
