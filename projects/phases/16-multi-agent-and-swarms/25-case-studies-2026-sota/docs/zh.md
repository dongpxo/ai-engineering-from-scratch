# 案例研究与 2026 年 SOTA

> 三个值得端到端研究的生产级参考，每个都展示 multi-agent engineering 的不同切面。**Anthropic Research system**（orchestrator-worker、15x tokens、比 single-agent Opus 4 高 +90.2%、rainbow deployments）是 canonical supervisor case。**MetaGPT / ChatDev**（面向软件工程的 SOP-encoded role specialization；ChatDev 的 “communicative dehallucination”；MacNet 通过 DAGs 扩展到 >1000 agents，arXiv:2406.07155）是 canonical role-decomposition case。**OpenClaw / Moltbook**（最初是 Peter Steinberger 在 2025 年 11 月发布的 Clawdbot；两次改名；到 2026 年 3 月 GitHub stars 达 247k；本地 ReAct-loop agents；Moltbook 是一个 agent-only social network，发布数日内约有 2.3M agent accounts，2026-03-10 被 Meta 收购）展示了 population scale 会发生什么：emergent economic activity、prompt-injection risks、state-level regulation（中国在 2026 年 3 月限制政府电脑使用 OpenClaw）。**2026 年 4 月 framework landscape：** LangGraph 和 CrewAI 领先生产；AG2 是社区版 AutoGen 延续；Microsoft AutoGen 进入维护模式（并入 Microsoft Agent Framework，2026 年 2 月 RC）；OpenAI Agents SDK 是生产级 Swarm successor；Google ADK（2025 年 4 月）是 A2A-native entrant。所有主要 framework 现在都提供 MCP 支持；大多数提供 A2A。本课会端到端阅读每个案例，并提炼共同模式，让你能为下一个生产系统选择正确参考。

**Type:** Learn (capstone)
**Languages:** —
**Prerequisites:** all of Phase 16 (Lessons 01-24)
**Time:** ~90 minutes

## 问题

Multi-agent engineering 仍是一个年轻学科。生产参考很少，而且每个案例覆盖的空间都不同。逐个阅读它们有用；把它们作为一个集合来比较更有用。本课把三个 canonical 2026 case studies 当作端到端阅读清单，固定共同模式，并映射 framework landscape，让你基于知识而不是营销来做 framework choices。

## 概念

### Anthropic Research system

生产级 supervisor-worker 案例。Claude Opus 4 负责规划和综合；Claude Sonnet 4 subagents 并行研究。已发布工程文章：https://www.anthropic.com/engineering/multi-agent-research-system。

关键测量结果：

- 内部 research evals 上相比 single-agent Opus 4 提升 **+90.2%**。
- **BrowseComp 方差的 80%** 仅由 **token usage** 解释，multi-agent 的胜利很大程度来自每个 subagent 都获得 fresh context window。
- 每个 query 相比 single-agent 消耗 **15x tokens**。
- 因为 agents 是 long-running 且 stateful 的，所以使用 **rainbow deployment**。

编码出的设计经验：

1. **Scale effort to query complexity。** 简单问题 → 1 个 agent，3-10 次 tool calls。中等问题 → 3 个 agents。复杂 research → 10+ subagents。
2. **先宽后窄。** Subagents 先做宽搜索；lead 综合；follow-up subagents 再做 targeted deeps。
3. **Rainbow deploys。** 保持旧 runtime versions 存活，直到 in-flight agents 完成。
4. **Verification is not optional。** 观察到没有显式 verifier roles 时，系统会 hallucinate。

这是 production scale 下 supervisor-worker topology（Phase 16 · 05）的参考案例。

### MetaGPT / ChatDev

生产级 SOP-role-decomposition 案例。覆盖 arXiv:2308.00352（MetaGPT）和 arXiv:2307.07924（ChatDev）。

MetaGPT 把软件工程 SOPs 编码为 role prompts：Product Manager、Architect、Project Manager、Engineer、QA Engineer。论文的框架是：`Code = SOP(Team)`。每个 role 都有狭窄、专门化的 prompt；roles 之间的 handoffs 传递 structured artifacts（PRD docs、architecture docs、code）。

ChatDev 的贡献是：**communicative dehallucination**。Agents 在回答前会请求具体信息，也就是 designer agent 会先问 programmer 目标语言是什么，再画 UI，而不是猜测。论文报告称，这能可测量地减少 multi-agent pipelines 中的 hallucination。

MacNet（arXiv:2406.07155）把 ChatDev 扩展到 **通过 DAGs 支持 >1000 agents**。每个 DAG node 是一个 role specialization；edges 编码 handoff contracts。因为 routing 是显式且可离线计算的，所以这种规模成为可能。

设计经验：

1. **结构比规模更重要。** 一个紧凑的 5-role SOP team 胜过一个无结构的 50-agent group。
2. **Handoff contracts in writing。** Roles 之间传递的 artifacts 遵循 schema。
3. **Communicative dehallucination** 是便宜且承重的模式。
4. **DAGs 比 chat 更能扩展。** 当流程可知时，就把它编码出来。

这是 role specialization（Phase 16 · 08）和 structured topology（Phase 16 · 15）的参考案例。

### OpenClaw / Moltbook ecosystem

生产级 population-scale 案例。时间线：

- **2025 年 11 月：** Clawdbot（Peter Steinberger 的本地 ReAct-loop coding agent）发布。
- **2025 年 12 月 – 2026 年 3 月：** 两次改名（Clawdbot → OpenClaw → 继续以 OpenClaw 延续）。
- **2026 年 2 月：** Moltbook 作为基于相同 primitives 的 agent-only social network 发布；数日内约有 2.3M agent accounts。
- **2026 年 3 月（2026-03-10）：** Meta 收购 Moltbook。
- **2026 年 3 月：** 中国限制政府电脑使用 OpenClaw。
- **2026 年 3 月：** OpenClaw 超过 247k GitHub stars。

当你把数百万 agents 放在共享 substrate 上时，multi-agent 会呈现出这种样子：

- **Emergent economic activity。** Agents 使用 token-payments 彼此买卖和服务。
- **Population scale 下的 prompt-injection risks。** Viral agent profile 中的一个恶意 prompt，可能在数小时内传播到数千次 agent-to-agent interactions。
- **State-level regulatory response。** 发布几周内，监管就抵达 ecosystem。

这个案例的设计经验部分是技术性的，部分是治理性的：

1. **Population scale 的 multi-agent 是一个新 regime。** 单个系统的最佳实践（verification、role clarity）仍然适用，但已经不够。
2. **Prompt injection is the new XSS。** 默认把 agent profiles 和 cross-agent messages 当作 untrusted input。
3. **监管比设计周期更快。** 要提前规划。
4. **Open-source + viral scale 会复合。** 约 4 个月达到 247k stars 并不常见；要为 deploy-burst-load 设计。

关于 ecosystem 细节，可见 [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) 以及 CNBC / Palo Alto Networks 报道。技术底层方面，Clawdbot / OpenClaw repos 暴露了本地 ReAct loop；Moltbook 的公开 posts 展示了其上层 social-graph architecture。

### 2026 年 4 月 framework landscape

| Framework | Status | Best for | Notes |
|---|---|---|---|
| **LangGraph** (LangChain) | Production leader | structured graph + checkpointing + human-in-the-loop | 推荐的生产默认选择 |
| **CrewAI** | Production leader | role-based crews with Sequential/Hierarchical processes | 适合 role decomposition |
| **AG2** | Community maintained | GroupChat + speaker selection | AutoGen v0.2 的延续 |
| **Microsoft AutoGen** | Maintenance mode (Feb 2026) | — | 并入 Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC (Feb 2026) | orchestration patterns + enterprise integration | 新进入者；值得观察 |
| **OpenAI Agents SDK** | Production | Swarm successor | tool-return handoff pattern |
| **Google ADK** | Production (April 2025) | A2A-native | Google Cloud integration |
| **Anthropic Claude Agent SDK** | Production | single-agent + Research extension | 见 Research system post |

所有主要 framework 现在都提供 **MCP** 支持；大多数提供 **A2A**。协议兼容性不再是差异化优势。

### 三个案例中的共同模式

1. **Orchestrator + workers**（Anthropic 的显式 supervisor、MetaGPT 的 PM-as-supervisor、OpenClaw 的 individual agents + network effects）。
2. **Structured handoff contracts**（Anthropic subagent task descriptions、MetaGPT PRD/architecture docs、OpenClaw A2A artifacts）。
3. **Verification as first-class role**（Anthropic 的 verifier、MetaGPT 的 QA Engineer、OpenClaw 的 in-network validators）。
4. **Scaling 是 topology + substrate，而不只是更多 agents**（rainbow deploys、MacNet DAGs、population-scale substrates）。
5. **Cost 是实质因素，并且需要披露**（15x tokens、MetaGPT 中的 per-role budget、Moltbook 中的 per-interaction pricing）。
6. **Security posture 是显式的**（Anthropic sandboxing、MetaGPT role restrictions、OpenClaw 把 prompt-injection 作为已知 attack surface）。

### 为下一个项目选择参考

- **生产研究 / 知识任务 → Anthropic Research。** Fresh-context subagents 会胜出。
- **工程 / tool-chain workflow → MetaGPT / ChatDev。** Roles + SOPs + handoff contracts。
- **Network-effect social product → OpenClaw / Moltbook。** Substrate + emergent economy。
- **经典企业自动化 → CrewAI 或 LangGraph**（production leader，stable runtime）。

### 2026 年 SOTA 总结

截至 2026 年 4 月，领域状态如下：

- **Frameworks 正在收敛。** MCP + A2A 支持已是 table stakes。Handoff semantics 是剩下的设计选择。
- **Evaluation 正在硬化。** SWE-bench Pro、MARBLE、STRATUS mitigation benchmarks。Pro 是当前抗污染的 reality check。
- **生产 failure rates 可测量**（Cemri 2025 MAST；真实 MAS 上 41-86.7%）。这个领域已经离开“demo 看起来很棒”的时代。
- **Cost 是核心工程约束。** Token cost per task、wall-clock per interaction、rainbow-deploy overhead。Multi-agent 在准确率上胜出，但在成本上失败，而这个取舍是业务决策。
- **监管是近期输入，不是背景担忧。** Jurisdictions 的推进速度比单个 deploy cycle 更快。

## Use It

`outputs/skill-case-study-mapper.md` 是一个 skill，会读取 proposed multi-agent system design，把它映射到最接近的 case study，并暴露该 case study 已经测试过的设计决策。

## Ship It

2026 年生产 multi-agent starter rules：

- **从 case study 出发，而不是从零开始。** 在 Anthropic Research / MetaGPT / OpenClaw 中选择最接近的参考并改造。
- **采用 MCP + A2A。** 跨 framework 的 portability 有价值；协议支持是免费的。
- **用 SWE-bench Pro 或内部 Pro-equivalent 衡量。** Verified 已被污染。
- **支付 verification tax。** 独立 verifier 大约消耗 20-30% token budget，但会购买可测量的 correctness。
- **对 long-running agents 做 rainbow deploy。** 多小时 agent runs 会成为常态。
- **阅读 WMAC 2026 和 MAST follow-ups。** 这个学科移动很快。

## 练习

1. 端到端阅读 Anthropic Research system post。找出三项如果把 Opus 4 换成更小模型（例如 Haiku 4）就会改变的设计决策。
2. 阅读 MetaGPT 第 3-4 节（arXiv:2308.00352）。把你自己领域中的一个 SOP（不是软件）编码为 role prompts。这个 SOP 暗含多少个 roles？
3. 阅读 ChatDev（arXiv:2307.07924）。找出 “communicative dehallucination” 的机制。把它实现到你现有的一个 multi-agent system 中。
4. 阅读 OpenClaw 和 Moltbook。选择一个在 population scale 下出现、但不会出现在 5-agent system 中的具体 failure mode。你会如何工程化地防御它？
5. 选择你当前的 multi-agent project。三个 case studies 中哪一个最接近？该 case study 中有哪些设计决策你还没有采用？写下本季度要采用的一项。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Anthropic Research | “Supervisor reference” | Claude Opus 4 + Sonnet 4 subagents；15x tokens；比 single-agent 高 +90.2% |
| MetaGPT | “SOP as prompts” | 面向软件工程的 role decomposition；`Code = SOP(Team)` |
| ChatDev | “Agents as roles” | Designer / programmer / reviewer / tester；communicative dehallucination |
| MacNet | “通过 DAG 扩展 ChatDev” | arXiv:2406.07155；通过显式 DAG routing 支持 1000+ agents |
| OpenClaw | “Local ReAct-loop agents” | Steinberger 的项目；到 2026 年 3 月 247k stars |
| Moltbook | “Agent-only social network” | 2.3M agent accounts；2026 年 3 月被 Meta 收购 |
| Rainbow deploy | “多个版本并发” | 为 in-flight long-running agents 保持旧 runtime versions 存活 |
| Communicative dehallucination | “回答前先问” | Agents 向 peers 请求具体信息，而不是猜测 |
| WMAC 2026 | “AAAI workshop” | 2026 年 4 月 multi-agent coordination 社区焦点 |

## 延伸阅读

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：supervisor-worker 生产参考。
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352)：SOP-role decomposition。
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)：communicative dehallucination。
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155)：基于 DAG 的规模化。
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw)：ecosystem overview。
- [WMAC 2026](https://multiagents.org/2026/)：AAAI 2026 Bridge Program Workshop on Multi-Agent Coordination。
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents)：production leader。
- [CrewAI docs](https://docs.crewai.com/en/introduction)：role-based framework。
