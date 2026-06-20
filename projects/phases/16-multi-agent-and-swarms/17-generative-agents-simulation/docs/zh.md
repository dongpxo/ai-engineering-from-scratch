# Generative Agents 与涌现式仿真

> Park 等 2023 年（UIST '23，arXiv:2304.03442）在 **Smallville** 这个由 25 个 agents 构成的 sandbox 中，使用了三部分架构：**memory stream**（自然语言日志）、**reflection**（agent 基于自身 stream 生成的更高层综合）、以及 **plan**（日级行为，再细分为 sub-plans）。标志性结果是情人节派对的涌现：一个 agent 被 seed 为“想办一场情人节派对”，没有额外脚本，邀请就在群体中传播，日期被协调，派对最终发生——其余 24 个 agents 一开始对此毫不知情。Ablations 显示，三个组件都对 believability 必不可少。文档记录的失败包括 spatial-norm errors（进入关门商店、共用单人卫生间）。这是 2026 年 agent simulations 和 multi-agent social evaluation 的参考架构。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## 问题

大多数 multi-agent systems 都是紧密脚本化的团队：planner 规划，coder 写代码，reviewer 审查。这对定义清晰的任务有效。但它无法捕捉 agents 拥有 memory、priorities 和 open world 时产生的涌现式、非脚本行为。研究、社会仿真，以及越来越多的 game AI，需要的是第二种系统。

Smallville 架构就是它的 benchmark。在 Park 2023 之前，最好的 agent simulations 大多是浅层 script-followers；之后，这个模式成了 open worlds 中 generative agents 的默认架构。如果你在 2026 年构建 agent simulation，要么使用 Smallville 的三个组件，要么明确说明为什么不用。

## 概念

### 三个组件

**Memory stream。** 一个 append-only log，记录 observations、actions、reflections 和 plans。每条 entry 都有 timestamp、type、description（自然语言），以及派生 metadata：**recency**、**importance**（agent 自评 1-10）和 **relevance**（与当前 query 的 cosine similarity）。

```text
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Memory retrieval 会组合三个分数：`score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`。Top-k entries 会进入当前 prompt。

**Reflection。** 周期性地（每 N 条 memories，或在重要事件发生时），agent 会从最近 memories 中生成高阶综合。Reflection entries 会被写回 stream，并像其他 memory 一样可检索。这就是 agents 建立“理解”的方式，也是该架构中的长期 beliefs 等价物。

**Plan。** 自顶向下分解。首先是粗粒度日级计划（“去上班，和 Klaus 吃晚饭”）。然后是小时级计划。再到 action-level plans。Plans 可以修改：当某个 observation 与计划矛盾时，agent 会重新规划受影响的 segment。

### 为什么三个组件都重要（ablation）

Park 等做过 ablations，分别去掉 observation、reflection 和 plan。每一种 ablation 都会伤害 believability：

- 没有 **observation**，agent 会错过上下文，并基于过时 beliefs 行动。
- 没有 **reflection**，agent 无法形成高阶 beliefs；互动会停留在浅层。
- 没有 **plan**，行为会变成反应式噪声；goals 会消散。

人类评分者给出的 believability scores 在三个组件齐全时最高；去掉任意一个都会产生可测量的退化。

### 情人节派对的涌现

一个 agent，Isabella Rodriguez，被 seed 了目标：“想在 2 月 14 日下午 5 点在 Hobbs Cafe 办一场情人节派对。”其余 24 个 agents 没有收到这个 seed。在模拟的几天里：

1. Isabella 的 plan 包含邀请别人。
2. 每次邀请都会变成邻居 memory stream 中的一条 observation。
3. 邻居的 reflection 生成 beliefs：“Isabella 正在办派对。”
4. 邻居的 plan 纳入“2 月 14 日参加派对”。
5. 邻居告诉其他邻居。邀请在没有中心协调的情况下传播。
6. 2 月 14 日下午 5 点，多个 agents 聚集到 Hobbs Cafe。

这是技术意义上的 emergence：系统级行为（一场派对）来自局部互动（双边邀请 + 个体规划），没有中心 orchestrator。

### 文档记录的失败模式

Park 等明确记录了：

- **Spatial norm errors。** Agents 走进关门的商店。Agents 试图使用同一个单人卫生间。Agents 在不适合吃饭的房间里吃饭。模型不会仅从环境中推断 social-physical norms。
- **Memory overflow。** 深度仿真运行会让 memory-retrieval 成本增长。实用补救方法：周期性 memory compaction（summarize-and-prune），以及对低 importance entries 做 decay。
- **Reflection hallucination。** Reflections 可能发明 memory stream 中不存在的关系。缓解方法：在 reflection prompts 中包含 source memory ids，并在 retrieval 时验证。

这些都是生产相关的失败模式：任何 2026 年的 agent simulation 都会继承它们。

### 三组件实现规则

1. **Memory is append-only。** 永远不要修改 memory entry。Corrections 是新的 entries。
2. **Importance scores should be cheap。** 写入时调用 LLM 给 importance 打 1-10 分。缓存这个分数。
3. **Retrieval is ranked, not filtered。** 按组合分数取 top-k；不要使用 hard filters，因为它们会丢失上下文。
4. **Reflection runs periodically。** 当未处理 memories 的 importance 总和超过阈值（例如 150）时触发。
5. **Plans are revisable。** 当新 observation 与 plan 矛盾时，只重新生成受影响的 segment，而不是整个 plan。

### Smallville 之外的 generative agents

2024-2026 年的后续文献扩展了这个架构：

- **用于政策 / 市场研究的 multi-agent social simulation。** 类 Smallville populations 可以模拟用户对 features 的行为反应。它比 A/B tests 更快；准确性仍有争议。
- **游戏中的 NPC AI。** 带 Smallville agents 的 RPG 会产生涌现故事线，而不是脚本化 quests。
- **Generative-agent evaluation benchmarks。** Metric 不再是 task accuracy，而是长时间运行中的 believability + coherence of behavior。

这个架构是参考点。扩展工作会替换组件（用 vector store 存 memory、retrieval-augmented reflection、neurosymbolic plan），但仍保留三部分结构。

### 为什么这对 multi-agent engineering 重要

Smallville 证明了，只要组件正确，multi-agent emergence 可以很便宜。这个架构现在已经在 open-source models 上复现过（更小 LLMs 的 believability 会平滑下降，而不是突然崩塌）。任何需要 **emergent social behavior** 的生产系统都会使用这个形状。任何需要 **tight task execution** 的系统，则会使用本阶段前面讲过的 supervisor / roles / primitives patterns。

## Build It

`code/main.py` 用 stdlib Python 和脚本化 agent policies（没有真实 LLM）实现了三个组件。Demo 用微型版本复现情人节派对涌现：

- `MemoryStream`：带 recency/importance/relevance retrieval 的 append-only log。
- `reflect(stream)`：基于近期高 importance memories 的脚本化 reflection。
- `plan(agent_state)`：基于当前 beliefs 的日级和小时级 plans。
- Scenario：5 个 agents。Agent 1 一开始带有“下午 5 点办派对”的 seed。经过多个 simulated ticks，邀请会传播，agents 会聚集。

运行：

```bash
python3 code/main.py
```

预期输出：逐 tick trace。到最后一个 tick 时，5 个 agents 中至少 3 个会在 plan 中显示派对，并聚集到派对地点。单个 seed 在没有任何 orchestrator 的情况下产生了 coordinated arrival。

## Use It

`outputs/skill-simulation-designer.md` 会设计一个 generative-agent simulation：agent 数量、memory schema、reflection cadence、plan horizon 和 evaluation metric。

## Ship It

生产仿真的规则：

- **Memory is the database。** 规模化时选择真实 store（vector DB、Postgres）。In-memory stdlib 只适合 prototypes。
- **Log the retrieval trace。** 对每个 action，记录驱动它的 top-k memories。这就是你的 debug 能力。
- **Budget per-agent tokens。** 每个 agent 每个 tick 的 retrieve + reflect + plan 都是 O(k) LLM calls。N agents × T ticks × calls-per-tick 可能吞掉预算。
- **Compact memory periodically。** Summarize-and-prune 低 importance entries。Retention policy 是设计决策，不是实现细节。
- **显式检测 spatial / social norm violations。** 这个架构不会自己学会它们。

## 练习

1. 运行 `code/main.py`。确认 3+ 个 agents 聚集到派对。把 agents 增加到 10 个——emergence 还会发生吗？
2. 移除 reflection step。行为看起来如何？把它对应到 Park 2023 的 ablation finding。
3. 引入一个竞争性 seeded goal（“Klaus 想在下午 5 点做 research talk”）。Agents 会分流，还是某个 goal 占主导？是什么决定了结果？
4. 添加空间约束：Hobbs Cafe 最多容纳 4 个 agents。Simulation 会优雅处理 overflow，还是会撞上“single-person bathroom”式失败模式？
5. 阅读 Park 等（arXiv:2304.03442）第 6 节（emergent behavior experiments）。找出一个你的微型版本无法复现的行为。你需要增强架构的哪个组件？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory stream | “Agent 的日记” | Observations、actions、reflections、plans 的 append-only log |
| Recency | “这条 memory 有多新” | 按 age 做 exponential-decay 的分数 |
| Importance | “Agent 有多在意” | 写入时自评 1-10；会被缓存 |
| Relevance | “与当前 query 有多相关” | Cosine similarity（基于 embedding） |
| Reflection | “高阶 belief” | 从近期 memories 生成的综合，再作为新 memory 写回 |
| Plan | “日 / 小时 / action 分解” | 自顶向下的 plan tree；当 observations 矛盾时可修订 |
| Smallville | “Park 2023 的 sandbox” | 产生情人节派对涌现的 25-agent simulation |
| Believability | “质量指标” | 人类评分者判断行为是否像合理 agent 的分数 |

## 延伸阅读

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)：参考架构。
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763)：发表会议。
- [Smallville code release](https://github.com/joonspk-research/generative_agents)：参考 Python 实现。
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639)：structured-memory agents 的前置工作。
