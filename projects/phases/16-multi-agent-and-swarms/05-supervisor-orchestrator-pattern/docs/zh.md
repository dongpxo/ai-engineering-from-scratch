# Supervisor / Orchestrator-Worker 模式

> 一个 lead agent 负责规划和委派；专门化 workers 在并行 context 中执行并回报结果。这就是 Anthropic Research system 背后的模式（Claude Opus 4 作为 lead，Sonnet 4 作为 subagents），在内部 research evals 上相比单 agent Opus 4 提升 +90.2%。Anthropic 的工程文章报告，BrowseComp 上 80% 的方差仅由 token usage 解释——multi-agent 的胜利很大程度上来自每个 subagent 都获得一个全新的 context window。本课会从原语构建 supervisor 模式，并覆盖 2026 年生产部署中的工程经验。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题

Research 是 single-agent systems 最容易失败的典型任务。你问：“2023 到 2026 年 multi-agent systems 发生了什么变化？”一个 single agent 会顺序阅读五篇论文，把一半 context 填满论文文本，然后还得一起推理它们。等它读到第五篇时，第一篇已经被遗忘。它也无法并行。

Supervisor pattern 修复了这个问题：一个 lead agent 规划搜索，把每个子问题委派给 worker，然后综合结果。每个 worker 都为一个狭窄问题获得自己的 200k-token window。Lead 不看原始论文，只看 worker summaries。

Anthropic 的生产 Research system 报告称，在内部 research evals 上相比单个 Opus 4 提升 +90.2%。同一篇文章还指出，BrowseComp 方差的 80% 仅由 *token usage* 解释。每个 subagent 拥有 fresh context 是主要机制。

## 概念

### 模式

```text
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Lead 永远不读取原始材料。Workers 在 lead 综合之前也不会看到彼此的工作。每条箭头都是一次 handoff，并携带一个狭窄 artifact。

### 为什么它会赢

三个机制：

1. **每个 subagent 都有 fresh context。** 探索 “FIPA-ACL heritage” 的 worker 不需要携带 lead 规划时消耗的 40k tokens。它能把一个 200k window 用在一个问题上。
2. **通过 prompt 实现专门化。** Lead 的 prompt 是“分解并综合”，不是“研究”。每个 worker 的 prompt 都很窄：“找出 X 中发生了什么变化”。聚焦的 prompts 产生聚焦的 outputs。
3. **并行。** Workers 并发运行。Wall-clock time 近似为 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程经验（Anthropic 2025）

Anthropic 的文章列出了几条到 2026 年仍然相关的生产经验：

- **按 query complexity 调整 effort。** 简单查询：一个 agent，3-10 次 tool calls。复杂查询：10+ 个 agents。这个判断应该由 lead 做，而不是调用方。
- **先宽后窄。** 先分解成宽泛子问题；如果答案值得深入，再为每个子问题生成更多 workers。
- **Rainbow deployments。** Agents 是 long-running 且 stateful 的。传统 blue-green 不适用。Anthropic 使用 rainbow：逐步推出新版本，同时让旧版本自然 drain。
- **Token usage 主导。** Multi-agent 约消耗 single-agent 15 倍 tokens。只有当任务价值能证明成本合理时才运行它。

### LangGraph 的转向

LangGraph 最初提供了一个 `langgraph-supervisor` library，里面有高层 `create_supervisor` helper。2025 年，LangChain 把推荐方式转为通过 tool-calling 直接实现 supervisor pattern，因为 tool calls 能更好地控制 *supervisor 看见什么*（context engineering）。Library 仍然可用；但 docs 现在推荐 tool-calling 形式。

### 失败模式

- **Lead 幻觉出错误计划。** 如果 lead 生成的子问题没有分解真实问题，workers 会在错误目标上做精确研究。
- **Workers 过度探索。** 如果没有明确 scope boundaries，workers 会漂移到 assigned sub-question 之外，并污染 synthesis step。
- **Synthesis conflicts。** 两个 workers 返回矛盾事实。Lead 必须重新询问（增加一轮），或明确标注分歧。静默选择一边是最糟糕的失败：用户永远不知道分歧发生过。

### 什么时候不该用 supervisor

- **顺序任务。** 如果 step 2 真的需要 step 1 的输出，并行没有收益。使用 pipeline（CrewAI Sequential、LangGraph linear graph）。
- **简单查询。** Single-agent 更快、更便宜。生成 workers 前先让 lead 做 “scale effort” 检查。
- **严格确定性。** Supervisor 使用 LLM-selected delegation。当 audit/replay 比适应性更重要时，static graphs 更好。

```figure
supervisor-hierarchy
```

## Build It

`code/main.py` 用 `threading` 实现了一个包含三个并行 workers 的 supervisor。Lead 把 query 分解为 sub-questions，workers 在每个 sub-question 上并发运行，lead 最后综合。没有真实 LLM；workers 是脚本化的，用来模拟 fetch-and-summarize。

关键结构：

- `Lead.plan(query)` 把 query 拆成 3 个 sub-questions。
- `Worker.run(sub_q)` 返回一个假 summary（生产中可以替换为任何 tool-using agent）。
- `Lead.run(query)` 在线程中启动 workers，join，并综合。

运行：

```bash
python3 code/main.py
```

输出会展示 plan、带 start/end timestamps 的并行 worker traces，以及最终 synthesis。你可以看到 wall-clock 收益：三个 0.3 秒 workers 会在约 0.35 秒内跑完，而不是 0.9 秒。

## Use It

`outputs/skill-supervisor-designer.md` 会接收一个用户 query，并产出一个 supervisor-pattern design：lead system prompt、worker roles、sub-question decomposition rules 和 synthesis template。在构建新的 research-style agent system 前使用它。

## Ship It

部署 supervisor pattern 前的 checklist：

- **Model pairing。** Lead 使用 reasoning-tier model（Opus class、`o3` class）。Workers 使用更快、更便宜的 model（Sonnet、`o4-mini`）。
- **Worker timeout。** 任何超过 2× median runtime 的 worker 都会被 kill；lead 要么用更窄 scope 重新 spawn，要么在缺少它的情况下继续。
- **Token cap per worker。** 硬限制（例如 expected synthesis input 的 10×）防止 runaway worker 吹爆预算。
- **Observability。** Trace lead 的 plan、每个 worker 的 tool calls 和 synthesis。这是一切事后 debugging 的基础。
- **Rainbow rollout。** Stateful long-running agents 需要渐进式版本迁移，而不是 hot swap。

## 练习

1. 运行 `code/main.py`，然后把 lead 改成生成 5 个 workers，而不是 3 个。观察 wall-clock effect。在这个 demo 里，worker 数量到多少时 spawn overhead 会超过并行收益？
2. 实现 worker timeout：kill 任何运行超过 0.5 秒的 worker，并让 lead 综合剩余结果。你需要哪些 observability 才能知道某个 worker 被切掉了？
3. 给 lead 的 synthesis 加一个 conflict-detection step：如果两个 workers 返回矛盾答案，lead 标注分歧，而不是选一边。不调用 LLM 时，你如何检测 contradiction？
4. 阅读 Anthropic 的 Research-system engineering post。列出这个 toy demo 若要进入生产，需要采用的三项实践。
5. 比较 LangGraph 的 `create_supervisor`（legacy）和新的 tool-calling 推荐方式。哪一种让你更好地控制 supervisor 能看见什么？为什么 Anthropic 明确只把 sub-answers 传入 synthesis，而不把 raw worker context 传入？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor | “Lead agent” | 负责规划、委派和综合的 orchestrator agent；不亲自完成工作 |
| Worker | “Subagent” | 由 supervisor 调用的聚焦 agent，scope 狭窄且拥有自己的 context window |
| Orchestrator-worker | “Supervisor pattern” | 同一件事的不同名称；2026 文献中两者都会出现 |
| Fresh context | “Clean window” | Worker 的 context 从 system prompt 和 assigned question 开始，而不是从 lead history 开始 |
| Rainbow deployment | “Gradual rollout” | Long-running stateful agents 需要 versioned drain-and-replace，而不是 blue-green |
| Token dominance | “Context is the variable” | Anthropic 报告称 research-eval 方差的 80% 来自 total tokens used，而不是 model choice |
| Scale effort | “Match agent count to complexity” | Lead 估计 query 难度，并相应生成 1 个或 10+ 个 workers |
| Synthesis conflict | “Workers disagree” | 两个 workers 返回矛盾事实；lead 必须暴露分歧，而不是静默选择一边 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：supervisor pattern 的生产参考。
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)：tool-calling supervisor 现在是推荐形式。
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor)：legacy helper，2026 年生产中仍有人使用。
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)：基于 handoff 的 supervisor 变体。
