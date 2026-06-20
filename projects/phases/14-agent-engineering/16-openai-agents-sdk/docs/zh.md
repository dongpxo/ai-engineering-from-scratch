# OpenAI Agents SDK：Handoffs、Guardrails、Tracing

> OpenAI Agents SDK 是建立在 Responses API 上的轻量多 agent 框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoffs 是名为 `transfer_to_<agent>` 的工具。Guardrails 会在输入或输出上触发。Tracing 默认开启。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## 学习目标

- 命名 OpenAI Agents SDK 的五个原语。
- 解释 handoffs：为什么它们被建模为工具，模型看到的名称形状是什么，以及 context 如何转移。
- 区分 input guardrails、output guardrails 和 tool guardrails；解释 `run_in_parallel` 与 blocking mode 的区别。
- 用 stdlib 实现一个带 handoffs、guardrails 和 span-style tracing 的 runtime。

## 问题

不能干净 delegation 的 agents，最后会把所有东西都塞进一个 prompt。没有 guardrails 的 agents 会输出 PII、违反 policy 的内容，或者无限循环。OpenAI 的 SDK 把让 multi-agent 工作可控的三个原语标准化了。

## 概念

### 五个原语

1. **Agent.** LLM + instructions + tools + handoffs。
2. **Handoff.** 委派给另一个 agent。对模型来说，它表现为一个名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail.** 对 input（只在第一个 agent 上）、output（只在最后一个 agent 上）或 tool invocation（每个 function tool 上）做 validation。
4. **Session.** 跨 turns 自动保存 conversation history。
5. **Tracing.** 为 LLM generations、tool calls、handoffs、guardrails 提供内置 spans。

### Handoffs as tools

模型会在工具列表里看到 `transfer_to_billing_agent`。调用它会通知 runtime：

1. 复制 conversation context（或通过 beta 版 `nest_handoff_history` 把它压缩）。
2. 用目标 agent 的 instructions 初始化目标 agent。
3. 用目标 agent 继续这次 run。

这就是 supervisor pattern（Lesson 13 / Lesson 28）的产品化版本。

### Guardrails

三种类型：

- **Input guardrails.** 在第一个 agent 的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **Output guardrails.** 在最后一个 agent 的输出上运行。捕获 PII 泄漏、policy violations、malformed responses。
- **Tool guardrails.** 对每个 function tool 运行。校验 arguments、检查权限、审计执行。

模式：

- **Parallel**（默认）。Guardrail LLM 和 main LLM 并行运行。Tail latency 更低。如果 guardrail 触发，main LLM 的工作会被丢弃（浪费 tokens）。
- **Blocking**（`run_in_parallel=False`）。Guardrail LLM 先运行。如果触发，就不会在 main call 上浪费 tokens。

Tripwires 会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

Tracing 默认开启。每次 LLM generation、tool call、handoff 和 guardrail 都会发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可以关闭。`add_trace_processor(processor)` 会把 spans 同时 fan out 到你自己的 backend 和 OpenAI 的 backend。

### Sessions

`Session` 会把 conversation history 存储在 backend 中（SQLite、Redis、自定义 backend）。`Runner.run(agent, input, session=session)` 会自动加载并追加历史。

### 这个模式容易在哪里出错

- **Handoff drift.** Agent A hand off 到 Agent B，Agent B 又 hand back 到 Agent A。添加 hop counter。
- **Guardrail bypass.** Tool guardrails 只会在 function tools 上触发；built-in tools（file reader、web fetch）需要单独 policy。
- **Over-tracing.** Spans 中包含 sensitive content。和 OTel GenAI content-capture rules（Lesson 23）配套使用——内容存到外部，只在 span 里引用 ID。

## Build It

`code/main.py` 用 stdlib 实现了 SDK 的形状：

- `Agent`、`FunctionTool`、`Handoff`（作为带 transfer semantics 的 function tool）。
- `Runner`，包含 input/output/tool guardrails、handoff dispatch 和 hop counter。
- 一个简单 span emitter，用来展示 trace shape。
- 一个 triage agent，根据用户查询 hand off 到 billing 或 support；其中一个 input 会触发 guardrail。

运行它：

```bash
python3 code/main.py
```

Trace 会展示两次成功 handoffs、一次 input guardrail trip，以及一棵和真实 SDK 输出相似的 span tree。

## Use It

- **OpenAI Agents SDK**：用于 OpenAI-first products。
- **Claude Agent SDK**（Lesson 17）：用于 Claude-first products。
- **LangGraph**（Lesson 13）：当你需要显式 state 和 durable resume 时使用。
- **Custom**：当你需要精确控制（voice、multi-provider、federated deployments）时使用。

## Ship It

`outputs/skill-agents-sdk-scaffold.md` 会 scaffold 一个 Agents SDK app，包含 triage agent、handoffs、input/output/tool guardrails、session store 和 trace processor。

## 练习

1. 添加 handoff hop counter：超过 N 次 transfer 后拒绝。Trace 这个行为。
2. 把 `nest_handoff_history` 实现成一个选项——transfer 前把 prior messages 压缩成一个 summary。
3. 写一个 blocking output guardrail。比较会触发它的 prompts 和会通过它的 prompts 的 latency。
4. 把 `add_trace_processor` 接到 JSON logger。每个 span 会输出什么形状？
5. 阅读 SDK docs。把你的 stdlib toy 移植到 `openai-agents-python`。你之前建模错了什么？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | “LLM + instructions” | SDK 中的 Agent 类型；拥有 tools 和 handoffs |
| Handoff | “Transfer” | 模型调用它来委派给另一个 agent 的工具 |
| Guardrail | “Policy check” | 对 input / output / tool invocation 的 validation |
| Tripwire | “Guardrail trip” | Guardrail 拒绝时抛出的 exception |
| Session | “History store” | 在 runs 之间持久化的 conversation memory |
| Tracing | “Spans” | 对 LLM + tool + handoff + guardrail 的内置 observability |
| Blocking guardrail | “Sequential check” | Guardrail 先运行；触发时不浪费 tokens |
| Parallel guardrail | “Concurrent check” | Guardrail 并行运行；latency 更低，触发时会浪费 tokens |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：primitives、handoffs、guardrails、tracing。
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：Claude-flavored counterpart。
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：什么时候应该使用 handoffs。
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：Agents SDK spans 映射到的标准。
