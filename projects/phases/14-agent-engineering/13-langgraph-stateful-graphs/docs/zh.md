# LangGraph：有状态图与持久执行

> LangGraph 是 2026 年低层有状态编排的参考实现。Agent 是状态机；nodes 是函数；edges 是 transitions；state 是不可变的，并在每一步之后 checkpoint。任何失败都可以从中断处精确恢复。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述 LangGraph 的核心模型：带不可变 state、函数 nodes、conditional edges 和 step 后 checkpoints 的状态机。
- 命名文档强调的四个能力：durable execution、streaming、human-in-the-loop、comprehensive memory。
- 解释 LangGraph 支持的三种编排拓扑：supervisor、peer-to-peer（swarm）、hierarchical（nested subgraphs）。
- 实现一个 stdlib state graph，包含不可变 state、conditional edges，以及 checkpoint/resume cycle。

## 问题

Agents 和 workflows 共享同一个问题：当一个 40 步运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二等公民式的 state model 会迫使 operators 在一个假设每次都是 fresh run 的库外面硬套 retries。

LangGraph 的设计回答是：state 是一等 typed object，mutations 是显式的，并且每个 node 后都会持久化 checkpoints。Resume 就是一次 `load_state(session_id)` 调用。

## 概念

### 图

一个 graph 由这些部分定义：

- **State type.** 一个 typed dict（或 Pydantic model），每个 node 都会读取并修改它。
- **Nodes.** 纯函数 `(state) -> state_update`。函数返回后，updates 会 merge 进 state。
- **Edges.** Nodes 之间的 conditional 或 direct transitions。
- **Entry and exit.** `START` 和 `END` sentinel nodes 标记边界。

示例：一个 agent 有 `classify`、`refund`、`bug`、`sales`、`done` nodes。它就是一个以 graph 表示的 routing workflow。

### Durable execution

每个 node 返回后，runtime 会序列化 state，并写入 checkpointer（SQLite、Postgres、Redis、自定义后端）。如果在第 N 步失败，runtime 可以 `resume(session_id)`，带着精确 state 从第 N+1 步继续。

LangGraph docs 明确强调了一些这件事很重要的生产用户：Klarna、Uber、J.P. Morgan。重点不只是 graph shape，而是 graph shape 加 checkpointing 让恢复成本变低。

### Streaming

每个 node 都可以 yield partial output。Graph 会把 per-node-delta events stream 给调用方，让 UI 随着 graph 运行而更新。

### Human-in-the-loop

在 nodes 之间检查并修改 state。实现方式：在关键 node 前暂停，把 state 展示给人类，接受修改，然后 resume。Checkpointer 让这件事变容易，因为 state 本来就已经被序列化了。

### Memory

短期记忆（一次 run 内，也就是 state 中的 conversation history）和长期记忆（跨 runs，通过 checkpointer 加单独 long-term store 持久化）。LangGraph 通过 tools 和外部记忆系统（Mem0、自定义系统）集成。

### 三种拓扑

1. **Supervisor.** 中央 router LLM 分派给 specialist subagents。`langgraph-supervisor` 中有 `create_supervisor()`（不过 LangChain 团队在 2026 年建议直接通过 tool calls 做这件事，以便获得更多 context control）。
2. **Swarm / peer-to-peer.** Agents 通过共享 tool surface 直接 hand off。没有中央 router。
3. **Hierarchical.** Supervisors 管理 sub-supervisors，以 nested subgraphs 实现。

### 这个模式会在哪里出错

- **Checkpoints too small.** 只 checkpoint conversation turns，会让 tool state 和 memory writes 无法恢复。完整 state 必须可序列化。
- **Non-deterministic nodes.** Resume 假设相同 node inputs 会产生相同 state update。Random seeds、wall-clock、external APIs 都必须被捕获。
- **Over-use of conditional edges.** 如果每条 edge 都是 conditional，这个状态机就无法推理。优先使用线性 chains，只在必要处偶尔分支。

## Build It

`code/main.py` 实现了一个 stdlib stateful graph：

- `State`：一个 typed dict，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node`：接收 state 并返回 update dict 的 callable。
- `StateGraph`：nodes + edges + conditional edges + run + resume。
- `SQLiteCheckpointer`（in-memory fake）：每个 node 后序列化 state；`load(session_id)` 会 restore。
- 一个 demo graph：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```bash
python3 code/main.py
```

trace 会展示第一次运行在 human gate 失败、state 持久化，然后 resume 并生成最终输出。

## Use It

- **LangGraph**：参考实现，production-ready。使用 `create_react_agent`、`create_supervisor`，或构建自己的 graph。
- **AutoGen v0.4**（Lesson 14）：适合高并发场景的 actor model alternative。
- **Claude Agent SDK**（Lesson 17）：带内置 session store 的 managed harness。
- **Custom**：当你需要精确控制 state shape 或 checkpointer backend 时使用。

## Ship It

`outputs/skill-state-graph.md` 会在任意目标 runtime 中生成一个 LangGraph-shaped state graph，并接好 checkpointing 和 resume。

## 练习

1. 当 classification confidence 低于阈值时，从 `classify` 添加一条 conditional edge 到 `end`。在人类手动设置 `route` 后 resume 这次 run。
2. 把 SQLite-like fake 换成真正的 SQLite checkpointer。测量每一步 serialization overhead。
3. 实现 parallel edges：两个 nodes 并发运行，并通过自定义 reducer merge。不可变 state 在这里带来什么好处？
4. 阅读 `langgraph-supervisor` reference。把 toy port 到 `create_supervisor`。比较 trace shapes。
5. 添加 streaming：每个 node 在运行时 yield partial state。打印到达的 deltas。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | 每个 node 后序列化 state；支持 resume |
| Reducer | "State merger" | 把当前 state 与 node update 组合起来的函数 |
| Conditional edge | "Branch" | 由 state 函数选择的 edge |
| Subgraph | "Nested graph" | 作为另一个 graph 中 node 使用的 graph |
| Durable execution | "Resume from failure" | 用精确 state 从最后一个成功 node 重新开始 |
| Supervisor | "Router LLM" | 面向 specialist subagents 的中央 dispatcher |
| Swarm | "P2P agents" | Agents 通过共享 tools hand off；没有中央 router |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：reference docs。
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/)：supervisor pattern API。
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：actor-model alternative。
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：session store 和 subagents。
