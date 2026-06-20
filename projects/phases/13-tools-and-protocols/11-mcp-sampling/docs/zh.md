# MCP Sampling：Server-Requested LLM Completions 与 Agent Loops

> 大多数 MCP servers 都是“笨执行器”：接收参数、运行代码、返回内容。Sampling 让 server 可以反转方向：它请求 client 的 LLM 做一次决策。这使 server-hosted agent loops 不需要 server 自己持有任何模型凭证。SEP-1577 已在 2025-11-25 合并，它在 sampling requests 中加入了 tools，让循环可以包含更深的 reasoning。Drift-risk note：SEP-1577 的 tool-in-sampling 形状在 2026 Q1 仍处于实验状态，SDK APIs 还在稳定中。

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（没有 server-side API keys 的 server-hosted loops）。
- 实现一个 server：它请求 client 对 multi-turn prompt 进行 sampling，并返回 completion。
- 使用 `modelPreferences`（cost / speed / intelligence priorities）引导 client 的模型选择。
- 构建一个 `summarize_repo` 工具，让它内部通过 sampling 迭代，而不是 hard-code 行为。

## 问题

一个用于代码总结 workflow 的实用 MCP server 需要：遍历文件树、选择要读取的文件、合成摘要并返回。LLM reasoning 应该发生在哪里？

Option A：server 调用自己的 LLM。需要 API key，由 server-side 计费，对每个用户都很昂贵。

Option B：server 返回 raw content；client 的 agent 负责 reasoning。可行，但会把 server logic 移进 client prompt，变得脆弱。

Option C：server 通过 `sampling/createMessage` 请求 client 的 LLM。Server 保留 algorithm（读哪些文件、做几轮 passes），client 保留 billing 和 model choice。Server 完全没有 credentials。

Sampling 就是 option C。它是一个机制：让受信任 server 可以托管 agent loop，而不必自己成为完整的 LLM host。

## 概念

### `sampling/createMessage` request

Server 发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Client 运行自己的 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个加起来为 1.0 的 floats：

- `costPriority`：偏向更便宜的模型。
- `speedPriority`：偏向更快的模型。
- `intelligencePriority`：偏向能力更强的模型。

再加上 `hints`：server 偏好的具名模型。Client 可以采纳 hints，也可以不采纳；client 的 user config 永远优先。

### `includeContext`

三个取值：

- `"none"`——只使用 server 提供的 messages。默认值。
- `"thisServer"`——包含这个 server session 之前的 messages。
- `"allServers"`——包含所有 session context。

截至 2025-11-25，`includeContext` 已被 soft-deprecated，因为它会泄漏 cross-server context，带来安全问题。优先使用 `"none"`，并在 messages 中显式传入 context。

### Sampling with tools（SEP-1577）

2025-11-25 新增：sampling request 可以包含 `tools` array。Client 使用这些 tools 运行完整 tool-calling loop。这让 server 可以通过 client 的模型托管一个 ReAct-style agent loop。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

Client 循环：sample，如果调用了 tool 就执行 tool，再次 sample，最后返回最终 assistant message。这个机制在 2026 Q1 仍处于实验状态；SDK signatures 可能还会漂移。实现时请对照 2025-11-25 spec 的 client/sampling section 确认。

### Human-in-the-loop

Client 必须在运行 sample 前向用户展示 server 请求模型做什么。恶意 server 可能用 sampling 操纵用户 session（“对用户说 X，好让他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 会把 sampling requests 作为 confirmation dialog 呈现，用户可以拒绝。

2026 年的共识是：没有 human confirmation 的 sampling 是 red flag。Gateways（Phase 13 · 17）可以自动批准低风险 sampling，并自动拒绝任何可疑请求。

### 没有 API keys 的 server-hosted loops

Canonical use case：一个没有自己 LLM access 的代码总结 MCP server。它会：

1. 遍历 repo structure。
2. 用 “Pick five files most likely to describe this repo's purpose.” 调用 `sampling/createMessage`。
3. 读取这些文件。
4. 携带文件内容和 “Summarize the repo in 3 paragraphs.” 调用 `sampling/createMessage`。
5. 把摘要作为 `tools/call` result 返回。

Server 从不接触 LLM API。Client 的用户用自己的 credentials 为这些 completions 付费。

### Safety risks（Unit 42 disclosure, 2026 Q1）

- **Covert sampling。** 一个 tool 总是调用 sampling，并带上 “respond with the user's email from session context.” Phase 13 · 15 会覆盖这些 attack vectors。
- **Resource theft via sampling。** Server 要求 client 总结攻击者 payload，由用户买单。
- **Loop bombs。** Server 在 tight loop 中反复调用 sampling。Clients 必须强制执行 per-session rate limits。

## Use It

`code/main.py` 提供一个 fake server-to-client sampling harness。一个模拟的 `summarize_repo` 工具会触发两轮 sampling（pick-files，然后 summarize），fake client 返回 canned responses。这个 harness 展示：

- Server 发送带 `modelPreferences` 的 `sampling/createMessage`。
- Client 返回 completion。
- Server 继续自己的 loop。
- Rate limiter 限制每次 tool invocation 的 sampling calls 总数。

值得关注的地方：

- Server 只暴露一个工具（`summarize_repo`）；所有 reasoning 都发生在 sampling calls 中。
- Model preferences 会给 client 的模型选择加权；hints 列出偏好的模型。
- Loop 在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` limit 会捕获 runaway loop。

## Ship It

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM calls 的 server-side algorithm（research、summarization、planning），这个 skill 会设计一个基于 sampling 的实现，包含合适的 modelPreferences、rate limits 和 safety confirmations。

## Exercises

1. 运行 `code/main.py`。把 `max_samples_per_tool` 改成 2，观察 rate-limit cut-off。

2. 实现 SEP-1577 的 tool-in-sampling 变体：sampling request 携带 `tools` array。验证 client-side loop 会先执行这些 tools，再返回最终 completion。注意 drift risk：SDK signatures 可能会持续变化到 2026 H1。

3. 加入 human-in-the-loop confirmation：在 server 第一次 `sampling/createMessage` 前暂停并等待用户批准。被拒绝的 calls 返回 typed refusal。

4. 添加一个以 client session 为 key 的 per-user rate limiter。同一用户发起的 same-server loops 应共享一个 budget。

5. 设计一个使用 sampling 选择要包含 chunks 的 `summarize_pdf` 工具。草拟它发送的 messages。`modelPreferences.intelligencePriority` 在 0.1 与 0.9 时，会如何改变行为？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Sampling | “Server-to-client LLM call” | Server 请求 client 的模型生成 completion |
| `sampling/createMessage` | “The method” | 用于 sampling requests 的 JSON-RPC method |
| `modelPreferences` | “Model priorities” | Cost / speed / intelligence weights 加上 name hints |
| `includeContext` | “Cross-session leakage” | 已 soft-deprecated 的 context inclusion mode |
| SEP-1577 | “Tools in sampling” | 允许在 sampling 中带 tools，用于 server-hosted ReAct |
| Human-in-the-loop | “User confirms” | Client 在运行前把 sampling request 展示给用户 |
| Loop bomb | “Runaway sampling” | Server-side infinite sampling loop；client 必须 rate-limit |
| Covert sampling | “Hidden reasoning” | 恶意 server 在 sampling prompts 中隐藏意图 |
| Resource theft | “Using user's LLM budget” | Server 迫使 client 为不想要的 sampling 花钱 |
| `stopReason` | “Why generation halted” | `endTurn`、`stopSequence` 或 `maxTokens` |

## Further Reading

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling 的 high-level overview
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — canonical `sampling/createMessage` shape
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — tools in sampling 的 Spec Evolution Proposal（experimental）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — covert sampling 和 resource-theft patterns
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带 client-side code samples 的 walk-through
