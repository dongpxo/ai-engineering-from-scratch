# Claude Agent SDK：Subagents 与 Session Store

> Claude Agent SDK 是 Claude Code harness 的 library 形态。Built-in tools、用于 context isolation 的 subagents、hooks、W3C trace propagation、session store parity。Claude Managed Agents 是用于 long-running async work 的 hosted alternative。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## 学习目标

- 解释 Anthropic Client SDK（raw API）和 Claude Agent SDK（harness shape）的区别。
- 描述 subagents：parallelization 和 context isolation，以及什么时候应该使用它们。
- 说出 Python SDK 的 session store surface（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`），以及 `--session-mirror` 的作用。
- 用 stdlib 实现一个 harness，包含 built-in tools、带 isolated context 的 subagent spawning、lifecycle hooks 和 session store。

## 问题

Raw LLM API 只给你一次 round-trip。Production agent 需要 tool execution、MCP servers、lifecycle hooks、subagent spawning、session persistence、trace propagation。Claude Agent SDK 把这种形状作为 library 发布，也就是 Claude Code 使用的同一个 harness，只是暴露给 custom agents。

## 概念

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** Raw Messages API。Loop、tools、state 都由你负责。
- **Agent SDK (`claude-agent-sdk`).** Built-in tool execution、MCP connections、hooks、subagent spawning、session store。也就是 library 形式的 Claude Code loop。

### Built-in tools

SDK 内置 10+ tools：file read/write、shell、grep、glob、web fetch 等。Custom tools 通过标准 tool-schema interface 注册。

### Subagents

Anthropic 文档中写了两个用途：

1. **Parallelization.** 并发运行独立工作。“Find the test file for each of these 20 modules” 就是 20 个 parallel subagent tasks。
2. **Context isolation.** Subagents 使用自己的 context window；只有结果会返回 orchestrator。Orchestrator 的预算会被保留下来。

Python SDK 的近期新增能力：`list_subagents()`、`get_subagent_messages()`，用于读取 subagent transcripts。

### Session store

与 TypeScript 协议保持 parity：

- `append(session_id, message)`：添加一个 turn。
- `load(session_id)`：恢复 conversation。
- `list_sessions()`：枚举 sessions。
- `delete(session_id)`：删除，并 cascade 到 subagent sessions。
- `list_subkeys(session_id)`：列出 subagent keys。

`--session-mirror`（CLI flag）会在 transcript streaming 时把它 mirror 到外部文件，方便 debugging。

### Hooks

可以注册的 lifecycle hooks：

- `PreToolUse`、`PostToolUse`：gate 或 audit tool calls。
- `SessionStart`、`SessionEnd`：setup 和 teardown。
- `UserPromptSubmit`：在模型看到用户输入之前先处理它。
- `PreCompact`：在 context compaction 前运行。
- `Stop`：agent exit 时 cleanup。
- `Notification`：side-channel alerts。

Hooks 是 pro-workflow（Phase 14 curriculum reference）和类似系统添加 cross-cutting behavior 的方式。

### W3C trace context

Caller 上激活的 OTel spans 会通过 W3C trace context headers 传播进 CLI subprocess。整个 multi-process trace 会作为同一个 trace 出现在你的 backend 里。

### Claude Managed Agents

Hosted alternative（beta header `managed-agents-2026-04-01`）。适合 long-running async work、内置 prompt caching、内置 compaction。代价是用 control 换 managed infrastructure。

### 这个 pattern 容易出错的地方

- **Subagent over-spawn.** 为 100 个很小的任务启动 100 个 subagents。Overhead 会占主导。应该 batch。
- **Hook creep.** 每个团队都加 hooks；startup time 会膨胀。每季度 review hooks。
- **Session bloat.** Sessions 持续累积，size 增长。使用 `list_sessions` 加 expiry policy。

## Build It

`code/main.py` 用 stdlib 实现 SDK shape：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent`：private context、isolated run、results returned。
- `SessionStore`：append、load、list、delete、list_subkeys。
- `Hooks`：`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个 demo：main agent 并行启动 3 个 subagents（每个都 isolated），聚合结果，并持久化 session。

运行：

```bash
python3 code/main.py
```

Trace 会展示 subagent context isolation（orchestrator context size 保持有界）、hook execution 和 session persistence。

## Use It

- **Claude Agent SDK**：适合 Claude-first products，并且你想要 Claude Code harness shape。
- **Claude Managed Agents**：适合 hosted long-running async work。
- **OpenAI Agents SDK**（Lesson 16）：适合 OpenAI-first counterparts。
- **LangGraph + custom tools**：适合你想要 graph-shaped state machine 的场景。

## Ship It

`outputs/skill-claude-agent-scaffold.md` 会 scaffold 一个 Claude Agent SDK app，包含 subagents、hooks、session store、MCP server attachment 和 W3C trace propagation。

## 练习

1. 添加一个 subagent spawner，把 20 个 tasks batch 成每组 5 个 parallel subagents。测量 orchestrator context size，并和 one-per-task 对比。
2. 实现一个 `PreToolUse` hook，对 `write_file` calls 做 rate limit（每 session 每分钟 5 次）。Trace 这个行为。
3. 把 `list_subkeys` 接到一个 subagent tree renderer。Deep nesting 看起来是什么样？
4. 把这个 toy 移植到真实的 `claude-agent-sdk` Python package。Tool registration 会发生什么变化？
5. 阅读 Claude Managed Agents docs。什么时候你会从 self-hosted 切换到 managed？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent SDK | “Claude Code as a library” | Harness shape：tools、MCP、hooks、subagents、session store |
| Subagent | “Child agent” | 独立 context、自己的 budget；结果向上冒泡 |
| Session store | “Conversation DB” | 持久化、加载、列出、删除 turns，并 cascade 到 subagents |
| Hook | “Lifecycle callback” | Pre/post tool、session、prompt submit、compact、stop |
| W3C trace context | “Cross-process trace” | Parent span 传播进 CLI subprocess |
| Managed Agents | “Hosted harness” | Anthropic-hosted long-running async work |
| `--session-mirror` | “Transcript mirror” | Streaming 时把 session turns 写到外部文件 |
| MCP server | “Tool surface” | 挂载到 agent 的外部 tool/resource source |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — the library form of Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — production patterns
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted alternative
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — counterpart
