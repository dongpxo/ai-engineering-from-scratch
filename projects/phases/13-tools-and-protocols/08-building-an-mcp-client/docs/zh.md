# 构建 MCP Client：Discovery、Invocation 与 Session Management

> 大多数 MCP 内容都会教你写 server，然后轻描淡写地带过 client。但真正困难的 orchestration 都在 client 代码里：进程启动、capability negotiation、跨多个 server 合并工具列表、sampling callbacks、重连，以及命名空间冲突处理。本课会构建一个 multi-server client，把三个不同的 MCP servers 提升到同一个扁平工具命名空间中，供模型调用。

**Type:** Build
**Languages:** Python (stdlib, multi-server MCP client)
**Prerequisites:** Phase 13 · 07 (building an MCP server)
**Time:** ~75 minutes

## 学习目标

- 把 MCP server 作为子进程启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个 server 的 session state（capabilities、tool list、last-seen notification ids）。
- 把多个 server 的 tool lists 合并到一个命名空间，并处理冲突。
- 将一次 tool call 路由到拥有该工具的 server，并重新组装 response。

## 问题

真实的 agent host（Claude Desktop、Cursor、Goose、Gemini CLI）会一次加载多个 MCP servers。用户可能同时运行一个 filesystem server、一个 Postgres server 和一个 GitHub server。Client 的职责是：

1. 启动每个 server。
2. 分别和每个 server handshake。
3. 对每个 server 调用 `tools/list`，并把结果压平成一个列表。
4. 当模型发出 `notes_search` 时，在合并后的命名空间中查找它，并路由到正确的 server。
5. 在不阻塞的情况下处理任意 server 发来的 notifications（`tools/list_changed`）。
6. 在 transport failure 后重新连接。

把这些全部手写出来，正是 “toy” 和 “serviceable” 的分界线。官方 SDK 会封装这部分，但 mental model 必须属于你自己。

## 概念

### 启动子进程

使用 `subprocess.Popen`，并设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1`，并使用 text mode，以便逐行读取。每个 server 都是一个进程；client 为每个 server 持有一个 `Popen` handle。

### 每个 server 的 session state

每个 server 都有一个 `Session` object，保存：

- `process` — Popen handle。
- `capabilities` — server 在 `initialize` 时声明的能力。
- `tools` — 最近一次 `tools/list` 的结果。
- `pending` — request id 到 promise/future 的映射，用来等待 response。

Request 天生是 async 的；发给 server A 的 `tools/call`，不应该因为 server B 正在执行一次调用而阻塞。可以使用带 queues 的 threads，也可以使用 asyncio。

### 合并命名空间

Client 看到聚合后的 tool list 时，名称可能冲突。两个 server 可能都暴露 `search`。Client 有三种选择：

1. **按 server name 加前缀。** `notes/search`、`files/search`。清楚，但不够美观。
2. **静默 first-come。** 后加载的 server 的 `search` 覆盖更早的。风险很高，因为它会隐藏冲突。
3. **拒绝冲突。** 拒绝加载第二个 server，并通知用户。对安全敏感的 hosts 来说最安全。

Claude Desktop 使用按 server 加前缀。Cursor 使用 collision rejection，并给出清晰错误。VS Code MCP 也采用按 server 加前缀。

### 路由

合并之后，dispatch table 会映射 `tool_name -> session`。模型按名称发出调用；client 找到对应 session，把一条 `tools/call` message 写入该 server 的 stdin，然后等待 response。

### Sampling callback

如果 server 在 `initialize` 中声明了 `sampling` capability，它可能会发送 `sampling/createMessage`，要求 client 运行自己的 LLM。Client 必须：

1. 阻塞发往该 server 的后续 requests，直到 sample resolved；如果实现支持并发，也可以 pipeline。
2. 调用自己的 LLM provider。
3. 把 response 发送回 server。

Lesson 11 会端到端覆盖 sampling。本课会为了完整性 stub 掉它。

### Notification handling

`notifications/tools/list_changed` 表示需要重新调用 `tools/list`。`notifications/resources/updated` 表示如果该 resource 正在使用中，就需要重新读取它。Notifications 不应该产生 responses——不要试图 ack 它们。

一个常见的 client bug：在等待 `tools/call` 时阻塞 read loop，而 stream 中已经堆着一个 notification。使用一个 background reader thread，把每条 message 推进 queue；main thread 从 queue 中取出并 dispatch。

### 重连

Transport 可能失败：server 崩溃、进程被 OS kill、stdio pipe 断开。Client 检测到 stdout 上的 EOF 后，就把 session 视为 dead。可选策略：

- 静默重启 server 并重新 handshake。适合 pure read-only servers。
- 把 failure 暴露给用户。适合带用户可见 session 的 stateful servers。

Phase 13 · 09 会覆盖 Streamable HTTP 的重连语义；stdio 更简单。

### Keepalive 和 session id

Streamable HTTP 使用 `Mcp-Session-Id` header。Stdio 没有 session id——进程 identity 就是 session。Keepalive pings 是可选的；stdio pipes 不会因为 inactivity 而断开。

## Use It

`code/main.py` 会把三个模拟 MCP servers 作为 subprocesses 启动，分别 handshake，合并它们的 tool lists，并把 tool calls 路由到正确的 server。这些 “servers” 实际上是运行 toy responders 的其他 Python 进程（没有真实 LLM）。运行它可以看到：

- 三次 initialization，每个都有自己的 capability set。
- 三份 `tools/list` 结果合并成一个包含 7 个工具的命名空间。
- 基于 tool name 的一次 routing decision。
- 通过 namespace prefixing 防止一次 collision。

值得关注的地方：

- `Session` dataclass 清晰地保存每个 server 的 state。
- Background reader thread 会读取 stdout 上的每一行，不阻塞 main thread。
- Dispatch table 是一个简单的 `dict[str, Session]`。
- Collision handling 是显式的：当两个 servers 声明同名工具时，后加载的工具会被加上前缀重命名。

## Ship It

本课产出 `outputs/skill-mcp-client-harness.md`。给定一份 MCP servers 的声明式列表（name、command、args），这个 skill 会生成一个 harness：启动它们、合并 tool lists，并交付一个带 collision resolution 的 routing function。

## 练习

1. 运行 `code/main.py`，观察 server spawn log。用 SIGTERM kill 掉其中一个模拟 server 进程，观察 client 如何检测 EOF 并把该 session 标记为 dead。

2. 实现 namespace prefixing。当两个 servers 都暴露 `search` 时，把第二个重命名为 `<server>/search`。更新 dispatch table，并确认 tool calls 会正确路由。

3. 为 server restart 添加 connection-pool-style backoff：连续失败时 exponential backoff，上限 30 秒，三次失败后向用户发出 notification。

4. 设计一个支持 100 个并发 MCP servers 的 client。什么数据结构可以替代简单的 dispatch dict？（提示：用于 prefix namespacing 的 trie，再加上一个 tool-count-per-server metric。）

5. 把 client 移植到官方 MCP Python SDK。SDK 会封装 `stdio_client` 和 `ClientSession`。代码应该从约 200 行缩小到约 40 行，同时保留 multi-server routing。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP client | “Agent host” | 启动 servers 并编排 tool calls 的进程 |
| Session | “每个 server 的状态” | Capabilities、tool list 和 pending-request bookkeeping |
| Merged namespace | “一个工具列表” | 跨所有 active servers 的扁平 tool names 集合 |
| Namespace collision | “两个 servers 同名工具” | Client 必须对 duplicate 选择 prefix、reject 或 first-come |
| Routing | “这次调用给谁？” | 从 tool name dispatch 到拥有它的 server |
| Background reader | “非阻塞 stdout” | 把 server stdout drain 到 queue 的 thread 或 task |
| Sampling callback | “LLM-as-a-service” | Client 处理 server 发来的 `sampling/createMessage` |
| `notifications/*_changed` | “Primitive 变了” | Client 必须重新 discover 或重新读取的信号 |
| Reconnection policy | “Server 死了怎么办” | Transport failure 后的 restart semantics |
| Stdio session | “Process = session” | 没有 session id；child process lifetime 就是 session |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — canonical client behavior
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world client tutorial
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — reference `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS parallel
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个 editor host 中 multiplex 多个 MCP servers
