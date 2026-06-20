# 多 Agent Primitive Model

> 2026 年发货的每个 multi-agent framework——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。只有四个 primitives，没有更多：agent、handoff、shared state、orchestrator。本课从零构建它们，在一个 toy system 上运行全部四个 primitive，然后把每个主流框架映射到同样的轴上，这样你就能用一段话读懂任何新 release。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

每六个月就会有一个新的 multi-agent framework 发货。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每个 press release 都宣称自己是 “the right abstraction”。

如果你试图逐个学习它们，会很快耗尽耐心。APIs 看起来不同。Docs 对 “agent” 是什么意见不一。一个框架把 shared memory 叫 “blackboard”，另一个叫 “message pool”，第三个叫 “StateGraph”。你会开始怀疑这个领域只是不断 churn。

并不是。marketing 之下，四个 primitives 是稳定的。学一次，就能用一段话读懂每个新框架。

## 概念

### 四个 primitives

1. **Agent**：一个 system prompt 加一组 tools。无状态；每次 run 都从 system prompt 和当前 message history 开始。
2. **Handoff**：从一个 agent 到另一个 agent 的结构化控制权转移。机械上，它可以是返回新 agent 的 tool call，也可以是根据 condition 走向下一个节点的 graph edge。
3. **Shared state**：任何可以被多个 agent 读取（有时写入）的数据结构。Message pool、blackboard、key-value store、vector memory。
4. **Orchestrator**：决定谁下一个发言的人或机制。选项包括：显式 graph（deterministic）、LLM speaker-selector（soft）、last speaker 的 handoff call（OpenAI Swarm），或 shared queue 上的 scheduler（swarm architecture）。

这就是完整设计空间。每个框架只是为每条轴选择 defaults；其余都是 surface syntax。

### 每个 2026 framework 如何映射到它

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | caller's problem | LLM 的下一个 handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的 speaker-selector | message pool | selector function（LLM 或 round-robin） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task outputs chained | manager LLM 或 static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | graph，deterministic |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host decides |

Surface differences 看起来很大。底下是同样的四个 knobs。

### 为什么这很重要

一旦你看见这些 primitives，framework comparison 就会变成一个短 checklist：

- Orchestrator 是信任 LLM 来 route（Swarm），还是把 routing 固定在代码里（LangGraph）？
- Shared state 是 full-history（GroupChat），还是 projected（StateGraph reducer）？
- Agents 能修改彼此的 prompts（CrewAI manager），还是只能 hand off（Swarm）？

这三个问题能回答 80% 的“哪个框架适合这个问题”。你会停止寻找 “the best multi-agent framework”，转而围绕真正关心的轴做设计。

### Stateless insight

除了 shared state，其他每个 primitive 都是 stateless。Agent 是 `(prompt, tools)` 的函数。Handoff 是一个 function call。Orchestrator 是一个 scheduler。**系统中唯一 stateful 的东西是 shared state。** 有趣的 bugs 全都住在那里：memory poisoning（Lesson 15）、message ordering、versioning、write contention。

隐藏 shared state 的框架（Swarm）会把问题推给 caller。集中 shared state 的框架（LangGraph checkpoint、AutoGen pool）让它可检查，但也把 coordination cost 转移给 shared-state implementation。

### 单个 primitive 的解剖

#### Agent

```text
Agent = (system_prompt, tools, model, optional_name)
```

没有 memory。没有 state。两个拥有相同 system prompt 和 tools 的 agents 可以互换。所有看起来像 per-agent state 的东西，实际上都在 shared state 或 handoff protocol 里。

#### Handoff

```text
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现最常见：

- **Function return**：tool 返回 next agent。这是 OpenAI Swarm pattern。Agents 把 routing 放在自己的 tool schemas 里。
- **Graph edge**：LangGraph。Edges 是 declarative。LLM 产出一个值；condition 选择下一个 node。
- **Speaker selection**：AutoGen GroupChat。selector function（有时本身也是一次 LLM call）读取 pool，然后选择谁下一个发言。

#### Shared state

```text
SharedState = { messages: [], artifacts: {}, context: {} }
```

最小形态是一组 messages。通常还会更多：structured artifacts（CrewAI Task outputs）、typed context（LangGraph reducers）、external memory（MCP、vector DB）。

两种拓扑：**full pool**（每个 agent 都看见每条 message）和 **projected**（agents 看见 role-scoped view）。Full pools 简单，但 scale 很差。Projected pools 可扩展，但需要提前做 schema design。

#### Orchestrator

```text
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种 flavors：

- **Static**：graph 在 build time 固定（LangGraph deterministic、CrewAI Sequential）。
- **LLM-selected**：LLM 读取 pool 并选择 next speaker（AutoGen、CrewAI Hierarchical）。
- **Handoff-driven**：当前 agent 通过调用 handoff tool 来决定（Swarm）。
- **Queue-driven**：workers 从 shared queue 拉取任务；没有显式 next-speaker（swarm architectures、Matrix）。

### 框架之间真正改变的是什么

Primitives 固定以后，剩下的设计决策是：

- **Memory strategy**：ephemeral vs durable checkpointing（LangGraph checkpointer）。
- **Safety boundary**：谁能批准 handoff（human-in-the-loop）。
- **Cost accounting**：per-agent token budgets。
- **Observability**：tracing handoffs、persisting state for replay。

这些都可以构建在 primitives 之上。它们都不是新的 primitives。

## Build It

`code/main.py` 用约 150 行 stdlib Python 实现了四个 primitives。没有真实 LLM——每个 agent 都是 scripted policy，这样注意力能停留在 coordination structure 上。

这个文件导出：

- `Agent`：一个 dataclass，包含 name、system prompt、tools、policy function。
- `Handoff`：返回新 agent 的 function。
- `SharedState`：thread-safe message pool。
- `Orchestrator`：三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（simulated）。

Demo 会把同一个 three-agent pipeline（research → write → review）放进三种 orchestrator types 里运行，并在最后打印 message pool。你会看到 outputs 只在 *who picks next* 上不同；agents 和 shared state 在每次 run 中完全相同。

运行：

```bash
python3 code/main.py
```

Expected output：三次 orchestrator runs，每种 pattern 一次。每次都会打印最终 message pool。如果 researcher 决定任务已经完成，handoff-driven run 会触达更少 agents——这就是 LLM-routing tradeoff 的微缩版。

## Use It

`outputs/skill-primitive-mapper.md` 是一个 skill，会读取任意 multi-agent codebase 或 framework doc，并返回 four-primitive mapping。遇到新 framework release 时先运行它，得到一段话理解，再深入读 docs。

## Ship It

采用新框架之前，先为它写 primitive mapping。如果你写不出来，说明 docs 不完整，或者这个框架真的发明了第五个 primitive（罕见——先检查是不是你没见过的一种 shared-state flavor）。

把 mapping 固定在 architecture doc 里。新成员加入团队时，先给他们 mapping，再给 API docs。框架版本变化时，diff mapping，而不是 diff changelog。

## 练习

1. 用不同 agent policies 运行 `code/main.py` 三次。观察 orchestrator choice 如何改变哪些 agents 会运行。
2. 实现第四种 orchestrator type：queue-driven，让 agents 从 shared state 中 poll work。可能发生什么 deadlock，你如何检测？
3. 取 LangGraph quickstart（https://docs.langchain.com/oss/python/langgraph/workflows-agents），把它改写成这四个 primitives。LangGraph 的哪些 abstractions 是 1:1 mapping，哪些只是 convenience wrappers？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 让哪一个 primitive 最 ergonomic，又把哪一个 primitive 推给 caller。
5. 在这张表里找一个完全隐藏 shared state 的框架。解释当 agents 需要跨 handoffs 协调、但又不重新读取 history 时，会坏掉什么。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | “An LLM with tools” | 一个 `(system_prompt, tools, model)` triple。Stateless。 |
| Handoff | “Transfer of control” | 命名 next agent 和 optional payload 的结构化 call。三种实现：function return、graph edge、speaker selection。 |
| Shared state | “Memory” / “context” | Multi-agent system 中唯一 stateful 的部分。Message pool 或 blackboard。 |
| Orchestrator | “Coordinator” | 决定谁下一个运行的机制。Static graph、LLM selector、handoff-driven 或 queue-driven。 |
| Primitive | “Abstraction” | 每个 framework 都参数化的四条轴之一。不是 framework feature。 |
| Message pool | “Shared chat history” | Full-history shared state。容易推理，但 scale 很差。 |
| Projected state | “Scoped view” | 面向角色的 shared state view。可扩展，但需要 schema design。 |
| Speaker selection | “Who talks next” | Orchestrator pattern：一个 function（通常是 LLM）从 group 中选择 next agent。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — handoff-driven orchestration 最清晰的表述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection 是 LLM-selected orchestration 的参考实现
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration 和 reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agents，Sequential / Hierarchical processes
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — Microsoft 将 v0.4 进入 maintenance 后，仍在活跃的 AutoGen v0.2 line
