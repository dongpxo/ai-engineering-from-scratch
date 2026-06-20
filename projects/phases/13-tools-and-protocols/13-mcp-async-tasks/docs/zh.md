# 异步 Tasks（SEP-1686）：长时间任务的 Call-Now、Fetch-Later

> 真实的 agent 工作会持续几分钟到几小时：CI 运行、深度研究综合、批量导出。同步 tool calls 会断连接、超时，或阻塞 UI。SEP-1686 已在 2025-11-25 合并，它加入了 Tasks primitive：任何 request 都可以被增强成一个 task，结果之后再 fetch，或通过 state notifications 进行 stream。漂移风险提示：Tasks 在 2026 年上半年仍属实验性功能；SDK surface 仍在围绕 spec 设计。

**Type:** Build
**Languages:** Python (stdlib, async task state machine)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**Time:** ~75 minutes

## 学习目标

- 判断什么时候应该把一个工具从同步调用提升为 task-augmented（server-side work 超过 30 秒）。
- 走通 task lifecycle：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化 task state，确保崩溃不会丢失正在执行的工作。
- 正确轮询 `tasks/status`，并获取 `tasks/result`。

## 问题

一个 `generate_report` 工具会运行多分钟的抽取 pipeline。在同步模型下有几个选项：

1. 保持连接打开三分钟。Remote transports 会断开；clients 会超时；UI 会冻结。
2. 立即返回一个 placeholder；要求 client 轮询自定义 endpoint。这会破坏 MCP 的统一性。
3. Fire-and-forget；没有结果。

这些都不理想。SEP-1686 加入了第四种方案：task augmentation。任何 request（通常是 `tools/call`）都可以被标记为 task。Server 立即返回 task id。Client 轮询 `tasks/status`，完成后获取 `tasks/result`。Server-side state 会跨重启存活。

## 概念

### Task augmentation

通过设置 `params._meta.task.required: true`（或 `optional: true`，由 server 决定），一个 request 就会变成 task。Server 会立即返回：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是 server 对保留 state 的承诺；超过 ttl 后，task result 会被丢弃。

### 每个工具单独 opt-in

Tool annotations 可以声明 task support：

- `taskSupport: "forbidden"` — 该工具始终同步运行。适合快速工具。
- `taskSupport: "optional"` — client 可以请求 task-augmentation。
- `taskSupport: "required"` — client 必须使用 task augmentation。

`generate_report` 工具应该是 `required`。`notes_search` 工具应该是 `forbidden`。

### States

```text
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

State machine 是 append-only：一旦进入 `completed`、`failed` 或 `cancelled`，task 就进入 terminal state。

### Methods

- `tasks/status {taskId}` — 返回当前 state 和 progress hint。
- `tasks/result {taskId}` — 阻塞，或在尚未完成时返回 404。
- `tasks/cancel {taskId}` — 幂等；terminal states 会忽略它。
- `tasks/list` — 可选；枚举 active 和 recently-completed tasks。

### Streaming state changes

当 server 支持时，client 可以订阅 state notifications：

```text
server -> notifications/tasks/updated {taskId, state, progress?}
```

使用 stream 而不是 poll 的 clients 会获得更好的 UX。Polling 始终是最小可用 surface。

### Durable state

Spec 要求声明 task support 的 servers 持久化 state。崩溃不应该丢失 ttl 内的 completed results。Store 可以从 SQLite、Redis 到 filesystem 不等。Lesson 13 harness 使用 filesystem。

### Cancellation semantics

`tasks/cancel` 是幂等的。如果 task 正在执行，server 会尝试停止它（这要求 executor-cooperative cancellation）。如果 task 已经 terminal，请求就是 no-op。

### Crash recovery

Server process 重启时：

1. 加载所有持久化的 task states。
2. 把任何进程死亡时仍为 `working` 的 tasks 标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled`。

### Async tasks plus sampling

一个 task 自身也可以调用 `sampling/createMessage`。长时间运行的 research tasks 就是这样工作的：server 的 task thread 按需 sample client 的模型，而 client UI 则显示 task 处于 `working`，并周期性更新 progress。

### 为什么它仍是实验性功能

SEP-1686 已在 2025-11-25 发布，但更大的 roadmap 指出了三个开放问题：durable subscription primitives、subtasks（parent-child task relationships），以及 result-TTL standardization。预计 spec 会在 2026 年继续演化。生产代码应该只把 Tasks 的 common case 视为稳定，并为 subtasks 相关的未来 SDK 变化加防护。

## Use It

`code/main.py` 实现了一个 durable task store（filesystem-backed），以及一个在 background thread 中运行的 `generate_report` 工具。Clients 调用该工具后会立即得到 task id；worker 更新 progress 时，client 轮询 `tasks/status`；完成后 client 获取 `tasks/result`。Cancellation 可用；通过 kill worker thread 并 reload state 来模拟 crash recovery。

值得关注的地方：

- Task state JSON 会持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- Worker thread 更新 `progress` 字段；poll 会看到它前进。
- Client-side cancellation 会设置一个 event；worker 会检查它并提前退出。
- “Crash” 后 reload state，会把 in-flight task 标记为 `failed`，错误为 `CRASH_RECOVERY`。

## Ship It

本课产出 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（research、build、export），这个 skill 会设计 task store（state shape、ttl、durability），选择正确的 taskSupport flag，并草拟 progress notifications。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` task，轮询 status，然后获取 result。

2. 在运行中途加入一次 `tasks/cancel` 调用。确认 worker 会遵守它，并且 state 变为 `cancelled`。

3. 模拟 crash recovery：kill 掉 worker thread，重启 loader，并观察 `CRASH_RECOVERY` failure mode。

4. 把 store 扩展到 SQLite。Durability 收益相同，但 query options 会打开（例如列出 session X 的所有 tasks）。

5. 阅读 MCP 2026 roadmap post。找出最可能在下一年影响 SDK API design 的那个 Tasks-related open issue。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Task | “长时间运行的工具调用” | 通过 `_meta.task` 增强、用于异步执行的 request |
| SEP-1686 | “Tasks spec” | 在 2025-11-25 加入 Tasks 的 Spec Evolution Proposal |
| `_meta.task` | “Task envelope” | 包含 id、state、ttl 的 per-request metadata |
| taskSupport | “Tool flag” | 每个工具可设为 `forbidden` / `optional` / `required` |
| `tasks/status` | “Poll method” | 获取当前 state 和可选 progress hint |
| `tasks/result` | “Fetch result” | 返回 completed payload；尚未完成时返回 404 |
| `tasks/cancel` | “停止它” | 幂等 cancellation request |
| ttl | “Retention budget” | Server 承诺保留 task state 的毫秒数 |
| `notifications/tasks/updated` | “State push” | Server 发起的 state-change event |
| Durable store | “Crash-safe state” | Filesystem / SQLite / Redis persistence layer |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — originating proposal 和完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带 rationale 的 design walkthrough
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanics 和 state machine
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK-level task implementation patterns
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — open issues 和 2026 priorities，包括 subtasks
