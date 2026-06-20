# AutoGen v0.4：Actor Model 与 Agent Framework

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 actor model 重新设计了 agent orchestration。异步消息交换、事件驱动 agents、故障隔离、天然并发。这个框架现在处于 maintenance mode，而 Microsoft Agent Framework（2025 年 10 月 public preview）正在成为后继者。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01（Agent Loop），Phase 14 · 12（Workflow Patterns）
**Time:** ~75 minutes

## 学习目标

- 描述 actor model：agents 是 actors，messages 是唯一 IPC，每个 actor 都有故障隔离。
- 命名 AutoGen v0.4 的三层 API——Core、AgentChat、Extensions——并说明每层用途。
- 解释为什么把 message delivery 和 handling 解耦，会带来故障隔离和天然并发。
- 用 Python stdlib 实现一个 actor runtime，并把一个双 agent code-review flow 移植到它上面。

## 问题

大多数 agent frameworks 都是同步的：一个 agent 产出，一个 agent 消费，运行在同一个 call stack 里。失败会打崩整个 stack。并发是后面加上去的。分布式需要重写。

AutoGen v0.4 的回答是：actor model。每个 agent 都是一个拥有私有 inbox 的 actor。Messages 是唯一交互方式。Runtime 把 delivery 和 handling 解耦。失败被隔离在单个 actor 内。并发是原生能力。分布式只是换一种 transport。

## 概念

### Actors

一个 actor 有：

- 私有 state（外部永远不能直接触碰）。
- 一个 inbox（message queue）。
- 一个 handler：`receive(message) -> effects`，其中 effects 可以是 “reply”、“send to other actor”、“spawn new actor”、“update state”、“stop self”。

两个 actors 不能共享内存。它们只能发送 messages。

### AutoGen v0.4 的三层 API

1. **Core.** 低层 actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat.** 面向任务的高层 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions.** 集成层——OpenAI、Anthropic、Azure、tools、memory。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 会把 message 放进 agent_b 的 inbox，然后立即返回。Runtime 稍后再 delivery。它带来三个结果：

- **Fault isolation.** Agent B 崩溃不会打崩 Agent A——runtime 会捕获 B handler 内的失败，并决定该怎么处理（log、retry、dead-letter）。
- **Natural concurrency.** 许多 messages 可以同时在路上；actors 并发处理自己的 inbox。
- **Distribution-ready.** 无论 actor 在进程内还是另一台 host 上，inbox + transport 都是同一个抽象。

### Topologies

- **RoundRobinGroupChat.** Agents 按固定轮转顺序发言。
- **SelectorGroupChat.** Selector agent 根据 conversation context 选择下一个发言者。
- **Magentic-One.** 用于 web browsing、code execution、file handling 的 reference multi-agent team。构建在 AgentChat 之上。

### Observability

OpenTelemetry support 是内建的。每条 message 都会发出一个 span；tool calls 会按照 2026 OTel GenAI semantic conventions（第 23 课）携带 `gen_ai.*` attributes。

### 状态：maintenance mode

2026 年初：AutoGen v0.7.x 对 research 和 prototyping 来说是稳定的。Microsoft 已经把活跃开发转向 Microsoft Agent Framework（2025 年 10 月 1 日 public preview；目标是在 2026 年 Q1 末 1.0 GA）。AutoGen patterns 可以干净地向前迁移——actor model 才是耐用的思想。

## Build It

`code/main.py` 实现了一个 stdlib actor runtime：

- `Message`：typed payload，包含 `sender`、`recipient`、`topic`、`body`。
- `Actor`：抽象类，带 `receive(message, runtime)`。
- `Runtime`：event loop，包含 shared queue、delivery 和 failure isolation。
- 一个双 actor demo：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行 checklist；它们交换 messages，直到达成 consensus。

运行它：

```bash
python3 code/main.py
```

trace 会展示 message delivery、某个 actor 中的模拟失败不会打崩另一个 actor，以及系统如何收敛到共同 verdict。

## Use It

- **AutoGen v0.4/v0.7**（maintenance）：适合 research、prototyping、multi-agent patterns。
- **Microsoft Agent Framework**（public preview）：向前路径；在刷新后的 API 中保留同样的 actor-model ideas。
- **LangGraph swarm topology**（第 13 课）：通过 shared-tool handoffs 实现类似模式。
- **Custom actor runtime**：当你需要特定 transport（NATS、RabbitMQ、gRPC）时。

## Ship It

`outputs/skill-actor-runtime.md` 会为给定 multi-agent task 生成一个最小 actor runtime，以及一个 team template（RoundRobin 或 Selector）。

## 练习

1. 添加一个 dead-letter queue：handler 抛错时，把失败 message 放到队列里供人工检查。在你的 toy 中，DLQ 会命中多少次？
2. 实现 `SelectorGroupChat`：selector actor 根据 conversation state 选择谁来处理下一条 message。
3. 添加 distributed transport：把进程内 queue 换成 JSON-over-HTTP server，让 actors 可以跑在不同 processes 中。
4. 为每条 message 接一个 OTel span（或 no-op stand-in）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的 architecture post。把你的 toy 移植到真实 `autogen_core` API。你跳过了哪些生产中重要的东西？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Actor | “Agent” | 私有 state + inbox + handler；没有共享内存 |
| Message | “Event” | Typed payload；actors 交互的唯一方式 |
| Inbox | “Mailbox” | 每个 actor 自己的 pending messages 队列 |
| Runtime | “Agent host” | 路由 messages 并隔离 failures 的 event loop |
| Topic | “Channel” | Actors 之间具名的 publish-subscribe route |
| Fault isolation | “Let it crash” | 一个 actor 失败不会打崩其它 actors |
| RoundRobinGroupChat | “Fixed-rotation team” | Agents 按顺序轮流发言 |
| SelectorGroupChat | “Context-routed team” | Selector 选择下一个发言者 |
| Magentic-One | “Reference team” | 用于 web + code + files 的 multi-agent squad |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：redesign post。
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：graph-shaped alternative。
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：AutoGen 默认发出的 spans。
