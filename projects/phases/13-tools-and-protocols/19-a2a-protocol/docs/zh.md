# A2A：Agent-to-Agent Protocol

> MCP 是 agent-to-tool。A2A（Agent2Agent）是 agent-to-agent——一个开放协议，用来让基于不同 frameworks 构建的不透明 agents 协作。它由 Google 在 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，并在 2026 年 4 月达到 v1.0，支持者超过 150 家，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP，并加入了 AP2 payments extension。本课会走过 Agent Card、Task lifecycle，以及两种 transport bindings。

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 区分 agent-to-tool（MCP）和 agent-to-agent（A2A）的 use cases。
- 在 `/.well-known/agent.json` 发布一个带 skills 和 endpoint metadata 的 Agent Card。
- 走通 Task lifecycle（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（text、file、data）的 Messages，并把 Artifacts 作为输出。

## 问题

一个 customer-service agent 需要把 report-writing 委托给一个专门的 writer agent。A2A 之前有这些选择：

- Custom REST API。可行，但每一组 pairing 都是一次性方案。
- Shared codebase。要求两个 agents 运行在同一个 framework 上。
- MCP。不适合：MCP 是用来调用工具的，不是用来让两个 agents 在保留各自不透明内部 reasoning 的同时协作。

A2A 补上了这个缺口。它把交互建模为一个 agent 向另一个 agent 发送 Task，并带有 lifecycle、messages 和 artifacts。被调用 agent 的内部状态保持不透明——调用方只能看到 task state transitions 和最终 outputs。

A2A 是“让跨 frameworks 的 agents 彼此对话”的协议。它不会替代 MCP；两者互补。

## 概念

### Agent Card

每个 A2A-compliant agent 都会在 `/.well-known/agent.json` 发布一张 card：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Discovery 是 URL-based：fetch card，学习 A2A endpoint 的 URL，并枚举 skills。

### Signed Agent Cards（AP2）

AP2 extension（2025 年 9 月）给 Agent Cards 加入了 cryptographic signatures。Publisher 用 JWT 签署自己的 card；consumers 进行验证。这可以防止 impersonation。

### Task lifecycle

```text
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Clients 通过 `tasks/send` 发起。被调用 agent 会在 states 之间转换；clients 通过 SSE 订阅 state updates，或进行 poll。

### Messages and Parts

一条 message 携带一个或多个 Parts：

- `text`——plain content。
- `file`——带 mimeType 的 base64 blob。
- `data`——typed JSON payload（给被调用 agent 的 structured input）。

Example：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

Outputs 是 Artifacts，而不是 raw strings。Artifact 是一个具名、带类型的 output：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为 chunks 流式传输。调用方负责累积。

### 两种 transport bindings

1. **JSON-RPC over HTTP。** `/a2a` endpoint，POST 用于 requests，可选 SSE 用于 streaming。默认 binding。
2. **gRPC。** 适用于 gRPC 原生的 enterprise environments。

两种 bindings 承载相同的 logical message shape。

### Opacity preservation

一个关键设计原则：被调用 agent 的内部状态是不透明的。调用方看到的是 task state 和 artifacts。被调用 agent 的 chain-of-thought、tool calls、sub-agent delegation——全部不可见。这不同于 MCP，MCP 中的 tool calls 是透明的。

Rationale：A2A 让竞争对手也能协作，而不暴露内部实现。A2A 可以是“call this customer-service agent”，而调用方不会知道那个 agent 是如何实现服务的。

### Timeline

- **2025-04-09。** Google announces A2A。
- **2025-06-23。** Donated to Linux Foundation。
- **2025-08。** Absorbs IBM's ACP。
- **2025-09。** AP2 extension（Agent Payments）ships。
- **2026-04。** v1.0 released with 150+ supporting organizations。

### Relationship to MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用某个具体工具时，用 MCP。当你想把一个完整任务委托给另一个 agent 时，用 A2A。许多生产系统会同时使用两者：一个 agent 用 MCP 作为自己的 tool layer，用 A2A 作为自己的 collaboration layer。

## Use It

`code/main.py` 实现了一个 minimal A2A harness：research agent 发布自己的 card，writer agent 接收一个 `tasks/send`，其中 parts 包含 PDF 和文本 instruction；随后它经历 working → input_required → working → completed，并返回一个 text artifact。全部使用 stdlib；用 in-memory transport 聚焦 message shapes。

值得关注的地方：

- Agent Card JSON shape。
- Task id assignment 和 state transitions。
- 带 mixed-type parts 的 Messages。
- Mid-task 的 input-required branch。
- Completion 时返回 Artifact。

## Ship It

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应被其他 agents 调用的新 agent，这个 skill 会生成 Agent Card JSON、skills schema 和 endpoint blueprint。

## Exercises

1. 运行 `code/main.py`。追踪完整 Task lifecycle，包括被调用 agent 请求 clarification 时的 input-required pause。

2. 添加 signed Agent Card。对 card 的 canonical JSON 做 HMAC 签名。编写 verifier，并确认 mutated card 会验证失败。

3. 实现 task streaming：writer agent 通过 SSE emit 三个 incremental artifact chunks，caller 负责累积它们。

4. 设计一个包装 MCP server 的 A2A agent。把每个 MCP tool 映射成 A2A skill。注明 trade-offs——失去了什么 opacity？

5. 阅读 A2A v1.0 announcement，找出截至 2026 年 4 月仍未被任何 framework 实现的一个 feature。（提示：它和 multi-hop task delegation 有关。）

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | “Agent-to-Agent protocol” | 用于不透明 agent collaboration 的开放协议 |
| Agent Card | “`.well-known/agent.json`” | 描述 agent skills 和 endpoint 的公开 metadata |
| Skill | “A callable unit” | Agent 支持的具名 operation（类似 MCP tool） |
| Task | “Unit of delegation” | 带 lifecycle 和 final artifact 的 work item |
| Message | “Task input” | 携带 Parts（text、file、data） |
| Part | “Typed chunk” | Message 中的 `text` / `file` / `data` element |
| Artifact | “Task output” | Completion 时返回的具名、带类型 output |
| AP2 | “Agent Payments Protocol” | 用于 trust 和 payments 的 Signed Agent Cards extension |
| Opacity | “Black-box collaboration” | 被调用 agent 的内部细节对调用方隐藏 |
| Input-required | “Task pause” | Agent 需要更多信息时的 lifecycle state |

## Further Reading

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — canonical A2A specification
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月 governance transfer
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — roadmap and partner momentum
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 release notes and backward-compat guidance
