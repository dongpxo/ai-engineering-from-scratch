# 长时间运行的后台 Agents：持久化执行

> 生产级 long-horizon agents 不应该跑在 `while True` 里。每一次 LLM 调用都应该成为一个带 checkpoint、retry 和 replay 的 activity。Temporal 的 OpenAI Agents SDK 集成已在 2026 年 3 月 GA。Claude Code Routines（Anthropic）可以运行定时的 Claude Code 调用，而不需要一个持久存在的本地进程。Sessions 会在等待人工输入时暂停，能跨部署存活，并从由 `thread_id` 标识的最新 checkpoint 恢复。这些新 ergonomics 背后是一个旧模式：workflow orchestration。只是现在多了一个新输入：LLM 调用是非确定性的 activities，恢复时必须能被确定性 replay。

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## 问题

想象一个运行四小时的 agent。它调用了三个工具，两次向用户发起 prompt，并完成了四十次 LLM 调用。运行到一半时，所在主机重启了。会发生什么？

- 在朴素的 `while True` 循环里：一切都会丢失。运行从头开始。三个已经产生真实副作用的工具调用会再次执行。用户会再次被要求批准已经批准过的内容。四十次 LLM 调用会再次计费。
- 使用持久化执行：运行会从最近的 checkpoint 恢复。已经完成的 activities 不会重新执行；它们的结果会从 durable log 中 replay。用户不需要重新批准已经批准过的内容。已经发生的 LLM 调用也不会再次计费。

这是 workflow engine 十年来一直在交付的同一个模式（Temporal、Cadence、Uber 的 Cherami）。新的地方在于，LLM 调用现在也是一种 activity：非确定性、昂贵、带副作用，而且能很好地放进这个模式里。

本课贯穿的主题是：long-horizon reliability 会衰减。METR 观察到一种 “35-minute degradation”：成功率会随着 horizon 大致按二次方下降。持久化执行能让运行时间超过可靠性曲线本身所支持的范围。如果设计正确，这是一种新的安全失败方式；如果设计错误，它会以不安全的方式失败。

## 概念

### Activities、workflows 和 replay

- **Workflow**：确定性的编排代码。定义 activities 的顺序、分支和等待点。它必须是确定性的，这样才能从 event log replay，而不会出现意外分歧。
- **Activity**：一个非确定性、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求都属于这一类。每个 activity 都会记录 inputs；完成后也会记录 outputs。
- **Event log**：持久化的后端存储。每一次 activity start、complete、fail、retry，以及每一次 workflow decision 都会被记录。
- **Replay**：恢复时，workflow code 会从头重新运行；所有已经完成的 activities 会直接返回日志里的结果，不重新执行。只有尚未完成的 activities 才会真正运行。

它的形状和 React 基于 virtual DOM 重新渲染、Git 从 commits 重建 working tree 很像。Orchestrator 的确定性，正是让 durability 变便宜的原因。

### 为什么 LLM 调用适合这个模式

LLM 调用具有这些特征：

- 非确定性（temperature > 0；即使 temperature 0，也会随着模型版本漂移）。
- 昂贵（成本和延迟）。
- 可能失败（rate limits、timeouts）。
- 带副作用（如果它们调用工具）。

这正是 activity 的画像。把每个 LLM 调用都包装成 activity，就能得到带 exponential backoff 的 retry、跨重启的 checkpointing，以及可 replay 的调试 trace。

### 用 `thread_id` 标识 checkpoints

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到了相同的 API 形状：一个 `thread_id`（或等价物）标识 session；每次状态转移都会持久化到后端（PostgreSQL 默认、SQLite 用于开发、Redis 用于缓存）；恢复时读取最新 checkpoint。

后端选择很重要：

- **PostgreSQL**：持久、可查询、能跨部署存活。LangGraph 默认使用它。
- **SQLite**：只适合本地开发；跨主机会丢数据。
- **Redis**：速度快，但除非配置 AOF/snapshot，否则是临时的。
- **Cloudflare Durable Objects**：透明分布式；由唯一 key 限定作用域；可存活数小时到数周。

### 把人工输入作为一等状态

Propose-then-commit（Lesson 15）需要一个持久化的 “waiting on human” 状态。Workflow 暂停，外部队列保存 pending request，审批到达后从准确的位置恢复。没有 durability 时，这只是 best-effort；有了 durability，隔夜审批第二天早上到达，workflow 也能继续执行。

### 35-minute degradation

METR 观察到，所有被测 agent class 在连续运行超过约 35 分钟后都会出现可靠性衰减。任务时长翻倍，失败率大致变为四倍。持久化执行不能修复这个问题；它只是让你能运行超过可靠性画像支持的时长。安全模式是把 durability 与两件事结合起来：恢复进入时要求 fresh HITL 的 checkpoint，以及 budget kill switches（Lesson 13），无论 wall-clock time 多长都限制总 compute。

### 什么时候持久化执行不是正确答案

- 运行时间短于几分钟，且没有人工输入。开销大于收益。
- 严格只读的信息检索。
- 正确性要求在一个 context window 内端到端完成的任务（某些推理任务，某些一次性生成任务）。

```figure
memory-consolidation
```

## Use It

`code/main.py` 用 stdlib Python 实现了一个最小的 durable-execution engine。它支持：

- `@activity` decorator，会把 inputs 和 outputs 记录到 JSON event log。
- 一个负责串联 activities 的 workflow function。
- 一个 `run_or_replay(workflow, event_log)` function，用来 replay 已完成 activities，而不是重新执行它们。

Driver 会模拟一个三 activity workflow，在中途 crash，并展示：（a）朴素 retry 会重新执行所有内容；（b）replay 只运行缺失的 activity。

## Ship It

`outputs/skill-durable-execution-review.md` 会审查一个 proposed long-running agent deployment 是否具备正确的 durable-execution 形状：activities、determinism、checkpoint backend、human-input state，以及 HITL-on-resume policy。

## 练习

1. 运行 `code/main.py`。观察 naive retry 和 replay 之间 activity 执行次数的差异。修改 crash point，并展示 replay count 会如何相应变化。

2. 把玩具 engine 改成显式使用 `thread_id`。模拟两个并发 sessions 共享同一个 engine，并确认它们的 event logs 不会相互冲突。

3. 选取玩具 engine 里的一个 activity。引入一个非确定性因素（例如 workflow decision 中的 wall-clock timestamp）。演示 replay 时的分歧。解释真实 engine 如何处理这个问题（side-effect registration、`Workflow.now()` APIs）。

4. 阅读 LangChain 的 “Runtime behind production deep agents” 文章。列出 runtime 持久化的每一种 state，并说明每一种覆盖了哪个 failure mode。

5. 为一个 6 小时的 autonomous coding task 设计 checkpoint policy。在哪里 checkpoint？crash 后恢复是什么样子？哪些地方需要 fresh HITL？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Workflow | “Agent 的脚本” | 确定性的编排代码；可从 event log replay |
| Activity | “一个步骤” | 非确定性单元（LLM 调用、工具调用）；执行前后都被记录 |
| Event log | “后端存储” | 每一次状态转移的持久记录 |
| Replay | “Resume” | 重新运行 workflow；已完成 activities 返回日志结果，不重新执行 |
| Checkpoint | “保存点” | 由 `thread_id` 标识的持久状态；恢复时采用最新状态 |
| thread_id | “Session key” | 限定 durable state 作用域的标识符 |
| 35-minute degradation | “可靠性衰减” | METR：成功率随 horizon 约按二次方下降 |
| Non-determinism | “Replay 时漂移” | Wall clock、random、LLM output；必须注册为 side effect |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop)：budget、turns 和 resume 语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop)：RequestInfoEvent 形状。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents)：具体 runtime requirements。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev)：LLM 调用的 activity 形状。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)：35-minute degradation 的参考来源。
