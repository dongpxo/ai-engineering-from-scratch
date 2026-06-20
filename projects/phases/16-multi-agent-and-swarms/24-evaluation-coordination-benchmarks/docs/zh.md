# Evaluation 与协调 Benchmarks

> 五个 2025-2026 benchmarks 覆盖了 multi-agent evaluation space。**MultiAgentBench / MARBLE**（ACL 2025, arXiv:2503.01935）用 milestone KPIs 评估 star/chain/tree/graph topologies；**graph 最适合 research**，cognitive planning 增加约 3% milestone achievement。**COMMA** 评估 multimodal asymmetric-information coordination；包括 GPT-4o 在内的 state-of-the-art models 很难超过 random baseline。**MedAgentBoard**（arXiv:2505.12371）覆盖四类 medical tasks，并且经常发现 multi-agent 并不优于 single-LLM。**AgentArch**（arXiv:2509.10769）评测结合 tool-use + memory + orchestration 的 enterprise agent architectures。**SWE-bench Pro**（[arXiv:2509.16941](https://arxiv.org/abs/2509.16941)）包含 41 个 repos 的 1865 个 problems，覆盖 business apps、B2B services 和 developer tools；frontier models 在 Pro 上约 23%，而在 Verified 上超过 70%。这是对 contamination 的 reality check。Claude Opus 4.7（2026 年 4 月）据称在 Pro 上达到 **64.3%**，并显式使用 agent-teams coordination（尚无 Anthropic primary source published，按 preliminary 对待）；Verdent（agent scaffold）在 Verified 上达到 **76.1% pass@1**（[Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。**AAAI 2026 Bridge Program WMAC**（https://multiagents.org/2026/）是 2026 年社区焦点。本课基于 MARBLE 的 metrics，运行 topology-vs-metric sweep，并固定一条规则：“仅通过 SWE-bench Verified 不是 generalization 的证据。”

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**Time:** ~75 minutes

## 问题

当一篇论文声称“我们的 multi-agent system 更好”时，问题是：比什么更好？在哪些任务上？如何测量？2023-2024 年的 multi-agent evaluation 很混乱，每个人都选择自己的 metrics、自己的 baselines、自己的 task sets。2025-2026 年的 benchmarks 开始强加结构。

没有 shared benchmarks，就无法有意义地比较两个 multi-agent systems。更糟的是，没有 hold-out benchmarks，frontier models 可能被 contamination。到 2025 年中，SWE-bench Verified 已经部分污染进 training corpora；frontier scores 膨胀；Pro 被设计成 uncontaminated reality check。

本课枚举 2026 年五个 canonical benchmarks，说明每个 benchmark 测量什么，并教你带着怀疑阅读 benchmark claims。

## 概念

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935。在 research、coding 和 planning tasks 上评估四种 coordination topologies（star、chain、tree、graph）。Milestone-based KPIs 跟踪 partial progress，而不只看 final success。

测得结果：

- **Graph** topology 最适合 research scenarios；支持 any-to-any critique。
- **Chain** 最适合 stepwise-refinement coding。
- **Star** 最适合 fast-factual consolidation。
- 在 graph 上超过约 4 个 agents 后会出现 **coordination tax**。
- **Cognitive planning** 在各种 topologies 上增加约 3% milestone achievement。

适用场景：你想 apples-to-apples 地比较 coordination topologies。MARBLE repo（https://github.com/ulab-uiuc/MARBLE）提供 evaluator。

### COMMA — multimodal asymmetric information

覆盖 agents 拥有不同 observation modalities、且必须在不完全共享信息的情况下协调的任务。报告结果令人不舒服：包括 GPT-4o 在内的 frontier models，在 COMMA 的 agent-agent collaboration 中很难超过 **random baseline**。信号是：multi-agent modalities 训练不足、评估不足。LLM 能相对合理地处理 single-modality cooperation；multi-modality coordination 会坍塌。

适用场景：你的系统包含 multimodal 或 asymmetric-information coordination。COMMA 的 null result 是一个提醒：先测量，再声称。

### MedAgentBoard — domain stress test

arXiv:2505.12371。四类 medical tasks：diagnosis、treatment planning、report generation、patient communication。比较 multi-agent、single-LLM 和 conventional rule-based systems。

发现：multi-agent 在大多数 categories 上并不优于 single-LLM。Multi-agent advantage 很窄：当 subtasks 清晰可分时（diagnosis + treatment），task decomposition 有帮助；当 coordination overhead 超过 specialization gain 时（report generation），它会伤害结果。

适用场景：你的领域有清晰的 single-LLM baselines。如果 MedAgentBoard 的经验可以泛化，那么许多 proposed multi-agent systems 都是 over-engineered。

### AgentArch — enterprise architectures

arXiv:2509.10769。Enterprise settings 中 tool use、memory 和 orchestration 叠加在一起。Benchmark 会隔离每一层的贡献：adding tools 帮助多少？adding memory 呢？adding multi-agent orchestration 呢？

适用场景：你正在设计 enterprise agent stack，并且需要证明每一层的价值。AgentArch 能帮助避免购买无法测量价值的 features。

### SWE-bench Pro — reality check

arXiv:2509.16941。1865 个 problems，覆盖 41 个 repositories，范围包括 business apps、B2B services 和 developer tools。它被设计为对较晚 training cutoffs **uncontaminated**。Frontier models 在 Pro 上约 23%，而在 Verified 上超过 70%。这个 gap 就是 contamination signal。

2026 年 4 月 scores：

- Claude Opus 4.7 on Pro：**64.3%**（据称显式使用 agent-teams coordination；尚无 Anthropic primary source published，按 preliminary 对待）。
- Verdent（agent scaffold）on Verified：**76.1% pass@1**（[technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。
- 没有 agent scaffolding 的 frontier raw scores on Pro：约 23-35%（[SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)）。

要点：“we beat SWE-bench Verified” 已经不再是能力证据。Pro 是当前 gating test。Agent-team scaffolding 在 Pro 上产生可测量收益（约 30-40 个百分点 delta），这是 2026 年支持 multi-agent coordination 的最强 empirical arguments 之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination（https://multiagents.org/2026/）。这是 2026 年 multi-agent AI research 的社区焦点。Accepted papers 和 workshop proceedings 是评估新方法的 canonical venue；做 production decisions 时，优先参考 WMAC-accepted claims，而不是 arXiv preprints。

### 带怀疑阅读 benchmark claims：2026 checklist

当有人声称一个 multi-agent result：

1. **哪个 benchmark，哪个 split？** SWE-bench Verified vs Pro 差别很大。报告在错误 split 上的数字没有价值。
2. **Contamination check。** Benchmark 是否在 model training cutoff 之后发布？如果不是，要谨慎对待。
3. **Baseline comparison。** 对比 single-LLM baseline、random、prior multi-agent work。不是“对比同一个系统的 untuned version”。
4. **Statistical significance。** N trials、p-value、confidence interval。Frontier models 高方差；single runs 会误导。
5. **Task diversity。** 一个 task 还是多个？Production 需要 generalization。
6. **Cost disclosure。** 每个 task 的 tokens、wall-clock。20x 成本下的 90% solution 是商业决策，不是能力声明。

### 当前 benchmarks 都没测好的东西

- **Long-horizon coordination。** 持续几天 wall-clock interaction。当前 benchmarks 都很短。
- **Adversarial resilience。** 如果一个 agent 是 malicious 或 compromised，会发生什么？
- **Drift under deployment。** Benchmarks 是静态的；production distributions 会变化。
- **Cost-normalized performance。** 大多数 benchmarks 报 raw accuracy，而不是 accuracy-per-dollar。

为你真正关心的轴构建 internal benchmark，通常是正确选择。

## Build It

`code/main.py` 是一个 non-interactive walk-through：

- 在 toy task 上模拟 3 个 multi-agent systems。
- 为每个系统计算 MARBLE-style milestone metrics。
- 通过从 “training” set 中 withholding tasks 来运行 contamination check。
- 显式对比 random baseline。
- 打印 benchmark-claims scorecard。

运行：

```bash
python3 code/main.py
```

Expected output：system scorecard，包含 raw accuracy、milestone achievement、cost-per-task、vs-random baseline delta，以及 contamination-check note。

## Use It

`outputs/skill-benchmark-reader.md` 会读取任意 multi-agent benchmark claim，并应用 scrutiny checklist。Output：grade 和 caveats。

## Ship It

Production evaluation discipline：

- **Build an internal benchmark**，反映你的真实 production distribution。Public benchmarks 提供参考，但不能替代。
- **Include a random baseline** in every comparison。如果你在 coordination task 上不能大幅超过 random，这个 task 可能定义得不好。
- **Report cost alongside accuracy。** Token cost 和 wall-clock。Ops teams 两者都需要。
- **Rebuild the benchmark quarterly。** Production distribution 会漂移；stale benchmarks 会误导。
- **Avoid published-benchmark overfitting。** 如果你的团队专门优化 SWE-bench Pro numbers，production 上会 regression。

## 练习

1. 运行 `code/main.py`。识别三个 simulated systems 中哪个 cost-per-milestone 最好。它是否也是 raw-accuracy 最高的系统？
2. 阅读 MultiAgentBench（arXiv:2503.01935）。针对你自己的 task domain，决定 MARBLE 会推荐四种 topologies 中的哪一种。用论文结果来说明理由。
3. 阅读 SWE-bench Pro paper。具体是什么让它 contamination-resistant？同样技术能否应用到你关心的其他 benchmarks？
4. 阅读 COMMA 关于 multimodal coordination 的发现。设计一个可以加入 internal benchmark 的简单 multimodal coordination task。什么结果算有用信号？
5. 将 benchmark-claims checklist 应用到最近一篇 multi-agent paper 的 headline result。你会给这个 claim 什么 grade？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARBLE | “MultiAgentBench” | ACL 2025；带 milestone KPIs 的 star/chain/tree/graph topologies。 |
| COMMA | “Multimodal benchmark” | Multimodal asymmetric-info coordination；frontier models 难以超过 random。 |
| MedAgentBoard | “Domain stress test” | 四个 medical categories；经常发现 multi-agent 不优于 single-LLM。 |
| AgentArch | “Enterprise benchmark” | Tools + memory + orchestration layered。 |
| SWE-bench Pro | “Contamination-resistant” | 1865 problems，41 repos；Pro 上约 23%，Verified 上超过 70%（contamination signal）。 |
| Milestone achievement | “Partial credit” | Reward progress，而不只 reward final success 的 benchmarks。 |
| Contamination | “Benchmark leaked into training” | Benchmark 发布后进入 training corpora，导致 scores inflate。 |
| WMAC | “AAAI 2026 Bridge Program” | Workshop on Multi-Agent Coordination；社区焦点。 |

## 延伸阅读

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark with milestone KPIs
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — reference implementation
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — domain stress test；multi-agent often does not dominate
- [AgentArch](https://arxiv.org/abs/2509.10769) — enterprise agent architectures
- [SWE-bench leaderboards](https://www.swebench.com/) — Verified and Pro scores for frontier models
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — the 2026 community focal point
