# Orchestration Patterns：Supervisor、Swarm、Hierarchical

> 2026 年各类 frameworks 中反复出现四种 orchestration patterns：supervisor-worker、swarm / peer-to-peer、hierarchical、debate。Anthropic 的建议是：“It's about building the right system for your needs.” 从简单开始；只有当单个 agent 加五种 workflow patterns 仍然不够时，才增加 topology。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12（Workflow Patterns），Phase 14 · 25（Multi-Agent Debate）
**Time:** ~60 minutes

## 学习目标

- 命名四种反复出现的 orchestration patterns，并说明各自适合什么时候。
- 描述 2026 年 LangChain 的建议：tool-call-based supervision vs supervisor libraries。
- 解释 Anthropic 的 “build the right system” 规则，以及它如何约束 topology 选择。
- 用 stdlib 基于同一个 scripted LLM 实现四种模式。

## 问题

团队常常在真正需要之前就伸手去拿 “multi-agent”。Frameworks 中反复出现四种模式；一旦你能叫出它们的名字，就能选对一种——或者干脆跳过 topology。

## 概念

### Supervisor-worker

- 一个 central routing LLM 分派任务给 specialist agents。
- 决策：回到自己、handoff 给 specialist，或 terminate。
- Specialists 彼此不直接对话；所有 routing 都通过 supervisor。

Frameworks：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 LangChain recommendation：** 通过 direct tool calls 做 supervision，而不是使用 `create_supervisor`。这样能获得更细的 context engineering control——你可以精确决定每个 specialist 看见什么。

### Swarm / peer-to-peer

- Agents 通过 shared tool surface 直接 hand off。
- 没有 central router。
- 比 supervisor latency 更低（hop 更少）。
- 更难推理（没有单一控制点）。

Frameworks：LangGraph swarm topology、OpenAI Agents SDK handoffs（当所有 agents 都能 hand off 给所有其它 agents 时）。

### Hierarchical

- Supervisors 管理 sub-supervisors，sub-supervisors 再管理 workers。
- 在 LangGraph 中实现为 nested subgraphs；在 CrewAI 中实现为 nested crews。
- 能扩展到大量 agents，但代价是 operational complexity。

什么时候需要它：当单个 supervisor 的 context budget 放不下所有 specialists 的描述时。

### Debate

- Parallel proposers + iterative cross-critique（第 25 课）。
- 严格来说不是真正的 orchestration——更像 verification——但在 frameworks 中经常作为一种 topology choice 出现。

### CrewAI Crew vs Flow

CrewAI 形式化了两种 deployment modes：

- **Flow** 用于 deterministic event-driven automation（生产起点推荐）。
- **Crew** 用于 autonomous role-based collaboration。

这和上面四种 patterns 正交，但会映射到 topology：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带 LLM router 的 supervisor。

### Anthropic 的建议

“Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs.”

决策顺序：

1. Single agent + workflow patterns（第 12 课）——从这里开始。
2. Supervisor-worker——当你有 2-4 个 specialists。
3. Swarm——当 latency 比 reasoning clarity 更重要。
4. Hierarchical——只有当 supervisor context budget 失败时。
5. Debate——当 accuracy 比 cost 更重要时。

### 这个模式哪里会出错

- **Topology-first thinking.** 在识别 multi-agent 到底解决什么问题之前，就说 “我们需要 multi-agent”。
- **Bouncing handoffs in swarm.** A -> B -> A -> B。使用 hop counters。
- **Fake hierarchy.** 因为 “enterprise” 搞三层，但实际上只有两个真实 teams。压平它。

## Build It

`code/main.py` 基于同一个 scripted LLM，用 stdlib 实现了四种模式：

- `Supervisor`：central router。
- `Swarm`：peer-to-peer with direct handoffs。
- `Hierarchical`：supervisors of supervisors。
- `Debate`：parallel proposers + critique。

每种模式都处理同一个三意图任务（refund / bug / sales）。Trace shapes 不同。

运行它：

```bash
python3 code/main.py
```

输出：每种 pattern 的 trace + op count。Supervisor 最清晰；swarm 最短；hierarchical 最深；debate 最昂贵。

## Use It

- **LangGraph** 用于 supervisor 和 hierarchical（nested subgraphs）。
- **OpenAI Agents SDK** 用于 handoffs-as-tools（supervisor-shaped）。
- **CrewAI Flow** 用于生产中的 deterministic workflow。
- **Custom** 用于 debate，或当你想要精确控制时。

## Ship It

`outputs/skill-orchestration-picker.md` 会选择一种 topology 并实现它。

## 练习

1. 通过移除 router，把 supervisor-worker 转成 swarm。什么会坏？什么会变好？
2. 给 swarm 添加 hop counter：超过 3 次 handoffs 就拒绝。它能抓住 A->B->A bouncing 吗？
3. 为一个 12-specialist domain 构建两层 hierarchical system。不嵌套时，context budget 会在哪里失败？
4. 在 production-shaped workload 上 profile 四种模式。哪一种在哪个指标上胜出（latency、cost、accuracy、debuggability）？
5. 阅读 Anthropic 的 “Building Effective Agents” 文章。把你的每个 production flow 映射到四种模式之一。有不容易映射的吗？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor-worker | “Router + specialists” | Central LLM 分派给 specialists；specialists 彼此不直接对话 |
| Swarm | “Peer-to-peer” | 通过 shared tools 直接 handoffs；没有 central router |
| Hierarchical | “Supervisors of supervisors” | 用 nested subgraphs 管理大型 agent populations |
| Debate | “Proposer + critique” | Parallel proposers，cross-critique（第 25 课） |
| Tool-call-based supervision | “Supervisor without a library” | 通过 direct tool calls 实现 supervisor，以控制 context |
| Crew | “Autonomous team” | CrewAI 的 role-based collaboration mode |
| Flow | “Deterministic workflow” | CrewAI 的 event-driven production mode |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：five patterns + agent vs workflow。
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：supervisor、swarm、hierarchical。
- [CrewAI docs](https://docs.crewai.com/en/introduction)：Crew vs Flow。
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325)：debate pattern。
