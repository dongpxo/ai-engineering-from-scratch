# A2A：Agent-to-Agent 协议

> Google 在 2025 年 4 月宣布 A2A；到 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，并获得 150+ 个组织支持。A2A 是 MCP（Lesson 13）的横向补充：MCP 是垂直的（agent ↔ tools），A2A 是 peer-to-peer 的（agent ↔ agent）。它定义了 Agent Cards（发现）、带 artifacts 的 tasks（text、structured data、video）、不透明 task lifecycles，以及 auth。生产系统越来越多地把 MCP 和 A2A 搭配使用。Google Cloud 在 2025-2026 年把 A2A 支持加入 Vertex AI Agent Builder。

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题

你的 agent 需要调用另一个系统上的另一个 agent。怎么调用？你可以暴露一个 HTTP endpoint，定义一套 bespoke JSON schema，然后希望对方也按这套说话。这样每一对 agents 都会变成一次 custom integration。

A2A 是这类调用的通用 wire protocol。标准发现、标准 task model、标准 transport、标准 artifacts。它像 HTTP+REST，但把 agents 当作一等公民。

## 概念

### 四个元素

**Agent Card。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述 agent：name、skills、endpoints、supported modalities、auth requirements。Discovery 通过读取 card 完成。

```text
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task。** 工作单元。一个异步、有状态的对象，拥有生命周期：`submitted → working → completed / failed / canceled`。Client 发送 task，然后轮询或订阅 updates。

**Artifact。** Task 产生的结果类型。Text、structured JSON、image、video、audio。Artifacts 有类型，因此不同 modalities 都是一等公民。

**Opaque lifecycle。** A2A 不规定 remote agent *如何* 解决任务。Client 看到 state transitions 和 artifacts；实现方可以自由使用任何 framework。

### MCP/A2A 分工

- **MCP**（Lesson 13）：agent ↔ tool。Agent 通过 JSON-RPC 读写 tool server。默认无状态。
- **A2A**：agent ↔ agent。Peer protocol；双方都是拥有自己推理能力的 agents。

生产 multi-agent systems 会同时使用两者。一个 A2A peer 会调用自己一侧的 MCP tools。这个分工让两类关注点保持清晰。

### Discovery flow

```text
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用 streaming：通过 SSE 订阅 `/tasks/{id}/events`，获得 push updates。

### Auth

A2A 支持三种常见模式：

- **Bearer token**：OAuth2 或 opaque。
- **mTLS**：mutual TLS；组织之间彼此证明身份。
- **Signed requests**：对 payload 做 HMAC。

Auth 会在 Agent Card 中声明；clients 负责发现并遵守。

### 到 2026 年 4 月已有 150+ 个组织

Enterprise adoption 推动了 A2A 的规模。重点是：A2A 成为企业 agent systems 跨越 trust boundaries 的方式。Google Cloud 发布了 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持它；大多数主要 frameworks（LangGraph、CrewAI、AutoGen）都提供 A2A adapters。

### A2A 胜出的场景

- **跨组织调用。** 公司 A 的 agent 调用公司 B 的 agent。没有 A2A 时，每一对都是 bespoke contract。
- **异构 frameworks。** LangGraph agent 调用 CrewAI agent，再调用 custom Python agent。A2A 负责标准化。
- **Typed artifacts。** Video result、structured JSON、audio 都是一等公民。
- **长时间运行任务。** Opaque lifecycle + polling 让持续数小时的任务变得直接。

### A2A 吃力的场景

- **对延迟敏感的 micro-calls。** A2A 的生命周期是异步的。Sub-millisecond agent-to-agent 不适合；使用 direct RPC。
- **紧耦合的进程内 agents。** 如果两个 agents 都在同一个 Python process 中运行，A2A 的 HTTP round-trip 就过度了。
- **小团队。** 规范开销是真实存在的；仅供内部使用的 agents 可能不需要这种正式性。

### A2A vs ACP、ANP、NLIP

2024-2026 年出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）：A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）：更重视 peer discovery，decentralized-first。
- **NLIP**（Ecma Natural Language Interaction Protocol，2025 年 12 月标准化）：natural-language content type。

截至 2026 年 4 月，A2A 是采用最广的 peer protocol。比较可见 arXiv:2505.02279（Liu 等，“A Survey of Agent Interoperability Protocols”）。

## Build It

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小 A2A server 和 client。Server：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理 task state，
- 在 `GET /tasks/{id}` 上返回 artifacts。

Client：

- 获取 Agent Card，
- 提交 task，
- 轮询直到完成，
- 读取 artifact。

运行：

```bash
python3 code/main.py
```

脚本会在后台线程中启动 server，然后运行 client 访问它。你会看到完整流程：discovery、submit、poll、artifact。

## Use It

`outputs/skill-a2a-integrator.md` 会设计一个 A2A integration：Agent Card contents、task schemas、auth choice、streaming vs polling。

## Ship It

Checklist：

- **Pin the spec version。** A2A 仍在演进；Agent Card 应声明 protocol version。
- **Idempotent task creation。** 重复提交（网络 retry）应该只产生一个 task。
- **Artifact schemas。** 声明 agent 返回的形状；consumers 应该验证。
- **Rate limits + auth。** A2A 面向公网；应用标准 web security。
- **Dead-letter for failed tasks。** 随时间检查模式，找出反复出现的失败类型。

## 练习

1. 运行 `code/main.py`。确认 client 能发现 server，并收到正确 artifact。
2. 给 server 添加第二个 skill（例如 “summarize”）。更新 Agent Card。写一个 client，根据 task type 选择 skill。
3. 实现一个 SSE streaming endpoint：`/tasks/{id}/events`，用于发出 state changes。Client 需要做什么不同的事？
4. 阅读 A2A spec（https://a2a-protocol.org/latest/specification/）。找出三件 spec 要求但这个 demo 没实现的事情。
5. 比较 A2A（Agent Card discovery）和 MCP（通过 `listTools` 做 server-side capability listing）。Self-describing agents 和 capability-probing 之间的取舍是什么？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | “Agent-to-agent” | Agents 跨系统调用其他 agents 的 peer protocol；Google 2025 |
| Agent Card | “Agent 的名片” | 位于 `/.well-known/agent.json` 的 JSON，描述 skills、endpoints、auth |
| Task | “工作单元” | 异步有状态对象，带 lifecycle；完成时产出 artifacts |
| Artifact | “结果” | Typed output：text、structured JSON、image、video、audio；media 是一等公民 |
| Opaque lifecycle | “如何解决是 agent 自己的事” | Client 看到 state transitions；server 可自由选择 framework/tools |
| Discovery | “找到 agent” | `GET /.well-known/agent.json` 返回 card |
| MCP vs A2A | “Tools vs peers” | MCP：垂直 agent ↔ tool。A2A：横向 agent ↔ agent |
| ACP / ANP / NLIP | “Sibling protocols” | 相邻规范；A2A 是 2026 年采用最广的协议 |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/)：canonical spec。
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)：2025 年 4 月发布文章。
- [A2A GitHub repo](https://github.com/a2aproject/A2A)：reference implementations and SDKs。
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1)：MCP、ACP、A2A、ANP 比较。
