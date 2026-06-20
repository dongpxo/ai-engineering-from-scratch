# 构建 MCP Server：Python + TypeScript SDKs

> 大多数 MCP tutorials 只展示 stdio hello-world。真实 server 会同时暴露 tools、resources 和 prompts，处理 capability negotiation，发出 structured errors，并且在不同 SDK 中保持同样行为。本课会端到端构建一个 notes server：stdlib stdio transport、JSON-RPC dispatch、三种 server primitives，以及一种 pure-function 风格，让你进阶时可以直接迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**Type:** Build
**Languages:** Python (stdlib, stdio MCP server)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals)
**Time:** ~75 minutes

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` methods。
- 编写一个 dispatch loop，从 stdin 读取 JSON-RPC messages，并向 stdout 写入 responses。
- 按照 JSON-RPC 2.0 spec 和 MCP 的额外 codes 发出 structured error responses。
- 在不重写 tool logic 的前提下，把 stdlib implementation 升级到 FastMCP（Python SDK）或 TypeScript SDK。

## 问题

在你使用 remote transport（Phase 13 · 09）或 auth layer（Phase 13 · 16）之前，需要先拥有一个干净的 local server。Local 指的是 stdio：client 把 server 作为 child process 启动，messages 通过 stdin/stdout 以 newline-delimited 形式流动。

2025-11-25 spec 规定，stdio messages 被编码为 JSON objects，并用显式的 `\n` separator 分隔。这里没有 SSE；SSE 是旧的 remote mode，并会在 2026 年中被移除（Atlassian 的 Rovo MCP server 将于 2026 年 6 月 30 日弃用它；Keboola 已于 2026 年 4 月 1 日弃用）。对于 stdio 来说，每行一个 JSON object 就是完整 wire format。

Notes server 是一个很好的形状，因为它会练到全部三种 server primitives。Tools 执行 mutations（`notes_create`）。Resources 暴露 data（`notes://{id}`）。Prompts 提供 templates（`review_note`）。本课的形状可以泛化到任何 domain。

## 概念

### Dispatch loop

```text
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何不是 JSON-RPC envelope 的内容。Debug logs 写到 stderr。
- 每个 request 都必须匹配一个带有相同 `id` 的 response。
- Notifications 一定不能被 response。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你实际支持的能力。Client 会依赖 capability set 来 gate features。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个 entry 都有 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}`，并返回 `{content: [blocks], isError: bool}`。

Content blocks 是 typed。最常见的几种：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Tool errors 有两种形状。Protocol-level errors（unknown method、bad params）是 JSON-RPC errors。Tool-level errors（调用合法，但 tool 自身失败）会以 `{content: [...], isError: true}` 返回。这样模型能在自己的 context 中看到失败信息。

### 实现 resources

Resources 按设计是 read-only。`resources/list` 返回 manifest；`resources/read` 返回 content。URIs 可以是 `file://...`、`http://...`，也可以是像 `notes://` 这样的 custom scheme。

当你把 data 暴露成 resource 而不是 tool 时：

- 模型不会“调用”它；client 可以在用户请求时把它注入 context。
- Subscriptions 让 server 能在 resource 变化时推送 updates（Phase 13 · 10）。
- Phase 13 · 14 会用 `ui://` 把它扩展到 interactive resources。

### 实现 prompts

Prompts 是带 named arguments 的 templates。Host 会把它们显示为 slash-commands。一个 `review_note` prompt 可能接收 `note_id` argument，并生成一个 multi-message prompt template，由 client 喂给自己的 model。

### Stdio transport subtleties

- Newline-delimited JSON。没有 length-prefixed framing。
- 不要 buffer。每次 write 后调用 `sys.stdout.flush()`。
- Client 控制 lifetime。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### Annotations

每个 tool 都可以携带 `annotations`，描述安全属性：

- `readOnlyHint: true` — 纯读取，safe to retry。
- `destructiveHint: true` — 不可逆副作用；client 应该确认。
- `idempotentHint: true` — 相同 inputs 产生相同 outputs。
- `openWorldHint: true` — 与 external systems 交互。

Client 使用这些信息决定 UX（confirmation dialogs、status indicators）和 routing（Phase 13 · 17）。

### Graduation path

`code/main.py` 中的 stdlib server 大约 180 行。FastMCP（Python）把同样逻辑压缩成 decorator-style：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价形状。准备好后，graduation path 可以直接替换；concepts（capabilities、dispatch、content blocks）是一样的。

## Use It

`code/main.py` 是一个完整的 notes MCP server，使用 stdio，且只依赖 stdlib。它处理 `initialize`、`tools/list`、`tools/call`（包含三个 tools：`notes_list`、`notes_search`、`notes_create`）、每条 note 的 `resources/list` 与 `resources/read`，以及一个 `review_note` prompt。你可以通过 pipe JSON-RPC messages 来驱动它：

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

观察重点：

- Dispatcher 是一个 `dict[str, Callable]`，以 method name 为 key。
- 每个 tool executor 都返回 content blocks list，而不是 bare string。
- 当 executor 抛出异常时，会设置 `isError: true`。

## Ship It

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个 domain（notes、tickets、files、database），这个 skill 会 scaffold 一个 MCP server，并正确拆分 tools / resources / prompts，同时给出 SDK graduation path。

## 练习

1. 运行 `code/main.py`，并用手写 JSON-RPC messages 驱动它。练习 `notes_create`，再用 `resources/read` 取回新 note。

2. 添加一个带 `annotations: {destructiveHint: true}` 的 `notes_delete` tool。确认 client 会显示 confirmation dialog（这需要真实 host；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，让 server 在 note 被修改时推送 `notifications/resources/updated`。添加 keepalive task。

4. 把 server 移植到 FastMCP。Python 文件应该缩到 80 行以内。Wire behavior 必须保持一致；用同一套 JSON-RPC test harness 验证。

5. 阅读 spec 的 `server/tools` section，找出本课 server 尚未实现的一个 tool definition field。（提示：有好几个；选一个并加上。）

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP server | “The thing that exposes tools” | 通过 stdio 或 HTTP 讲 MCP JSON-RPC 的 process |
| stdio transport | “Child process model” | Server 由 client 启动；通过 stdin/stdout 通信 |
| Dispatcher | “Method router” | JSON-RPC method name 到 handler function 的映射 |
| Content block | “Tool result chunk” | Tool response 的 `content` array 中的 typed element |
| `isError` | “Tool-level failure” | 表示 tool 失败；区别于 JSON-RPC error |
| Annotations | “Safety hints” | readOnly / destructive / idempotent / openWorld flags |
| FastMCP | “Python SDK” | 构建在 MCP protocol 之上的 decorator-based higher-level framework |
| Resource URI | “Addressable data” | 标识 resource 的 `file://`、`db://` 或 custom scheme |
| Prompt template | “Slash-command brief” | Server 提供的、带 argument slots 的 template，供 host UIs 使用 |
| Capability declaration | “Feature toggle” | 在 `initialize` 中声明的 per-primitive flags |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — reference Python implementation
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — parallel TS implementation
- [FastMCP — server framework](https://gofastmcp.com/) — decorator-style Python API for MCP servers
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的 end-to-end tutorial
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — `tools/*` messages 的完整 reference
