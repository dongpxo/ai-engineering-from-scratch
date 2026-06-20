# Handoffs 与 Routines：无状态编排

> OpenAI 的 Swarm（2024 年 10 月）把 multi-agent orchestration 提炼成两个 primitives：**routines**（作为 system prompt 的 instructions + tools）和 **handoffs**（返回另一个 Agent 的 tool）。没有 state machine，没有 branching DSL，LLM 通过调用正确的 handoff tool 来路由。OpenAI Agents SDK（2025 年 3 月）是 production successor。Swarm 本身仍然是最干净的概念参考，它的完整源码只有几百行。这个 pattern 会流行，是因为 API surface 大致就是 “agent = prompt + tools; handoff = function returning agent”。限制：stateless，所以 memory 是 caller 的问题。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题

每个 multi-agent framework 都想让你学习它的 DSL：LangGraph 的 nodes 和 edges，CrewAI 的 crews 和 tasks，AutoGen 的 GroupChat 和 managers。这些 DSL 确实是抽象，但它们会让事情显得比必要的更重。

Swarm 走相反方向：使用模型本来就有的 tool-calling capability。Handoffs 变成 tool calls。Orchestrator 就是当前持有 conversation 的 agent。State machine 隐含在各个 agents 的 system prompts 里。

## 概念

### 两个 primitives

**Routine。** 定义 agent role 和 available tools 的 system prompt。可以把它想成一组 scoped instructions：“you are a triage agent; if the user asks about refunds, hand off to the refund agent.”

**Handoff。** Agent 可以调用的 tool，返回一个新的 Agent object。Swarm runtime 检测到 Agent return value 后，会在下一轮切换 active agent。

整个抽象就是这些。

```python
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return -> switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

Triage agent 的 system prompt 会让它基于 user message 选择正确的 handoff。LLM 的 tool-calling 完成 routing。

### 为什么它会流行

- **Small API。** 只需要学习两个概念。
- **使用模型已经会做的事。** Tool calling 已经在多个 providers 中 production-grade。
- **没有 state-machine burden。** 你不描述 graph；agents 的 prompts 描述它们会 hand off 给谁。

### Stateless trade

Swarm 明确地在 runs 之间 stateless。Framework 会在一次 run 中保留 message history，但不会持久化任何东西。Memory、continuity、long-running tasks 全都是 caller 的问题。

在 production（OpenAI Agents SDK，2025 年 3 月）中，这是主要变化之一：SDK 在保留 handoff primitive 的同时，加入了 built-in session management、guardrails 和 tracing。

### 什么时候 Swarm / handoffs 合适

- **Triage patterns。** Front-line agent 把 user 路由到 specialist。
- **Skill-based handoffs。** “如果 task 需要 code，就调用 coder；如果需要 research，就调用 researcher。”
- **短而有界的 conversations。** Customer support、FAQ-to-ticket、simple workflows。

### 什么时候 Swarm 吃力

- **带 shared memory 的 long sessions。** Handoffs 会把 conversation state 重置为新 agent 的 prompt 加 history。没有 caller-managed memory，就没有跨 agents 的 persistent state。
- **Parallel execution。** Handoff 是 one-at-a-time，active agent 会切换。Parallelism 需要 caller 编排多个 Swarm runs。
- **Audit and replay。** Stateless runs 很难精确 replay；LLM 的 handoff choice 不是 deterministic。

### OpenAI Agents SDK（2025 年 3 月）

Production successor 加入了：

- **Session state。** Runs 之间持久化 thread。
- **Guardrails。** Input/output validation hooks。
- **Tracing。** 每个 tool call 和 handoff 都被记录。
- **Handoff filters。** 控制 handoff 时转移哪些 context。

Handoff primitive 保留下来；production ergonomics 被加在它周围。

### Swarm vs GroupChat

二者都使用 LLM-driven routing，但差别在于 **谁选择 next**：

- GroupChat：一个 selector（function 或 LLM）从外部选择 next speaker。
- Swarm：current agent 通过调用 handoff tool 选择自己的 successor。

Swarm 是 “agent decides what's next”；GroupChat 是 “manager decides what's next”。Swarm 的 decision 存在于 active agent 的 tool call 中；GroupChat 的 decision 存在于 `GroupChatManager` 中。

## Build It

`code/main.py` 从零实现了 Swarm：一个 Agent dataclass、一个 handoff mechanism（tool returns Agent），以及一个检测 agent switches 的 run loop。

Demo：一个 triage agent 路由到 refund、sales 或 support specialists。每个 specialist 都有自己的 tools。Run loop 会打印每次 handoff。

运行：

```bash
python3 code/main.py
```

## Use It

`outputs/skill-handoff-designer.md` 会为给定 task 设计 handoff topology：有哪些 agents，它们可以调用哪些 handoffs，哪些 context 会转移。

## Ship It

Checklist：

- **Handoff logging。** 每次 handoff 都写一条 trace event，包含 from-agent、to-agent、context snapshot。
- **Context transfer rules。** 决定 handoff 时移动什么：full history（昂贵）、last N messages，或 summary。
- **Guardrail on handoff。** Handoff 到具有不同 tool permissions 的 specialist 必须 authenticated，否则 prompt injection 可以强制 unwanted handoffs。
- **Loop detection。** 两个 agents 来回 hand off 是常见 failure；用简单的 last-K ring check 检测。
- **Fallback agent。** 如果 handoff target 不存在，就 fallback 到 safe default。

## 练习

1. 运行 `code/main.py`，triage 到 refund agent。确认第二轮的 active agent 是 refund。
2. 添加 loop-detection rule：如果同两个 agents 连续 hand off 了 3 次，就强制退出。设计 fallback。
3. 阅读 OpenAI Agents SDK docs 中关于 handoff filters 的内容。实现一个 “summarize-on-handoff” 版本：outgoing agent 在 incoming agent 接手前，把 context 压缩成 bullet summary。
4. 比较 Swarm handoff 和 GroupChatManager selector。哪种 pattern 会让 prompt injection 更糟，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 做出的一个 explicit design decision，并说明 OpenAI Agents SDK 改变了它还是保留了它。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routine | “The agent prompt” | System prompt + tool list。定义 role 和 available handoffs。 |
| Handoff | “Transfer to another agent” | Active agent 可以调用的 tool，返回一个新 Agent。Runtime 切换 active agent。 |
| Stateless | “No memory between runs” | Swarm 不持久化任何东西；memory 是 caller 的责任。 |
| Active agent | “Who's speaking now” | 当前持有 conversation 的 agent。Handoff 会改变它。 |
| Context transfer | “What moves on handoff” | Incoming agent 能看到哪些 history 的策略：full、last N 或 summarized。 |
| Handoff loop | “Agents ping-pong” | 两个 agents 不断把任务交回给对方的 failure mode。 |
| OpenAI Agents SDK | “Production Swarm” | 2025 年 3 月 successor；在 handoff primitive 上加入 sessions、guardrails、tracing。 |
| Handoff filter | “Gate on transfer” | SDK feature，用于在 handoff boundary 检查和修改 context。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — reference articulation
- [OpenAI Swarm repo](https://github.com/openai/swarm) — original implementation, kept as conceptual reference
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — production successor with sessions and tracing
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code subagents 如何通过 `Task` 使用 handoff-like pattern
