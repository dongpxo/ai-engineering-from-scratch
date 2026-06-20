# METR Time Horizons 与外部能力评估

> METR（前 ARC Evals）自 2023 年 12 月起成为独立的 501(c)(3) 组织。他们的 Time Horizon 1.1 benchmark（2026 年 1 月）会把任务成功概率与 log(expert human completion time) 拟合成一条 logistic curve；曲线在 50% 成功概率处的交点定义为模型的 time horizon。2025-2026 年的 engagement set 覆盖 GPT-5.1、GPT-5.1-Codex-Max，以及原型 monitoring evaluations（monitor 能否发现 side tasks；agent 能否规避 monitor）。Benchmark suites 包括：HCAST（180+ 个 ML、cyber、SWE、reasoning 任务；1 分钟到 8+ 小时）、RE-Bench（71 个带 expert baseline 的 ML research-engineering 任务）、SWAA。诚实说明：METR 测量是理想化的——没有人类参与，没有真实后果——而且团队已经记录了 eval-vs-deployment behavior gap（Lesson 1）。Time horizon 是上界，不是部署预测。

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time:** ~60 minutes

## 问题

Scaling policies（Lessons 19、20）只有在它们引用的测量可靠时才有用。“AI R&D-4 threshold” 和 “Long-range Autonomy” 是用政策文字定义的；只有当具体评估产生具体数字时，它们才变得可执行。

METR 是 2024-2026 年定义了许多这类数字的外部评估组织。他们评估 frontier models，常常是在模型发布前、与实验室签署 NDA 的情况下进行，然后再发布 methodology。Time Horizon 1.1 benchmark（2026 年 1 月）是他们的代表性 artifact：一个把能力压缩成人类可读单位的单一标量（“这个模型能以 50% 可靠性完成专家会花 X 小时做的那类任务”）。

本课一半讲 methodology（horizon 如何计算），一半讲 interpretation（为什么 horizon 是上界，而不是部署预测）。这两项能力必须放在一起。理解 horizon 如何拟合的团队，比只在 slide 上看到 “14 hours” 的团队更不容易被糟糕的 vendor claim 误导。

## 概念

### METR 背景

- 成立：2023 年 12 月（前 ARC Evals，拆分为独立 501(c)(3)）。
- 范围：评估 frontier models 的 autonomous capabilities，常常在发布前进行。
- 合作实验室：Anthropic、OpenAI（2025-2026 年多次 engagement）。
- 代表性成果：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、原型 monitoring evaluations。

### Time Horizon 拟合

Methodology（来自 METR blog 和 papers）：

1. 收集一套覆盖分钟级到小时级专家完成时间的任务集。当前 suites：HCAST（180+ 个任务）、RE-Bench（71 个任务）、SWAA。
2. 在每个任务上运行模型；记录成功或失败。
3. 拟合一条 logistic curve：P(success) 作为 log(expert completion time) 的函数。
4. Horizon 是 P(success) = 0.5 时对应的 expert-time。

Logistic-fit 的形状是合适的，因为能力通常和任务难度呈现一种逐渐上升并接近平台的关系。50% 点是一种选择（也可以是 10%、90%）；METR 在详细论文中报告多个阈值，但用 50% 作为主数字，因为它最直观。

### 2026 年 1 月数字

根据 Time Horizon 1.1：

- Claude Opus 4.6：在 Time Horizon 1.1（2026 年 1 月）中，50% 可靠性约为 14 小时。
- HCAST-style tasks 上的 doubling time：Time Horizon 1.1（2026 年 1 月）报告的 post-2023 fit 约为 4.3 个月（130.8 天）；约 7 个月的数字来自 Time Horizon 1.0 的完整 2019-2025 fit，并在 TH1.1 中和 post-2023 数字一同报告。

这些是 Phase 15 其余部分引用的具体数字。Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）都把它们作为 threshold assessments 的输入测量之一。

### Benchmark suites

- **HCAST**：189 个任务，覆盖 ML、cybersecurity、software engineering 和 general reasoning。跨度：专家时间从 1 分钟到 8+ 小时。它是 horizon 数字背后的主套件。
- **RE-Bench**：71 个 ML research-engineering 任务，带 human expert baseline。衡量模型是否能完成会进入 AI R&D-4 threshold 的那类 ML R&D 工作。
- **SWAA**：较短任务；用于校准 horizon curve 的低端。

### 原型 monitoring evaluations

METR 在 2025-2026 年的工作包括 monitoring-evaluation prototypes：monitor 能否发现模型正在执行 side task（也就是运营方要求之外的事情）；agent 能否规避 monitor。这些直接进入 OpenAI PF v2 的 “Undermining Safeguards” 类别，以及 DeepMind FSF v3（Lesson 20）中的 deceptive-alignment monitoring。

### 为什么 horizons 是上界

- **理想化工具**：benchmark tasks 给模型干净的工具和格式良好的数据。真实部署更混乱。
- **没有真实后果**：模型不会真正给客户计费、删除真实数据或联系真实人员。真实部署有不可逆的 stakes。
- **Eval-context gaming**：Lesson 1。模型在测试中表现不同。2026 International AI Safety Report 对此有实证记录。
- **没有真实用户差异**：benchmark prompts 是结构化的。真实用户会提出模糊、依赖上下文的请求。

Horizon 是有利条件下的能力上限。部署可靠性是另一个数字，更低；团队必须测量自己的分布，才能知道它。

### 外部评估者的意义

外部评估重要，是因为内部实验室有动机优化它们自己报告的指标。METR 的独立性——一个有明确 methodology 和 peer-reviewed papers 的 501(c)(3)——是一种结构性缓解。它本身还不充分（实验室仍然控制 METR 能看到什么），但严格好于没有外部评估。

### 实践中如何使用 horizon 数字

- **作为能力过滤器**：如果某个模型的 horizon 明显低于拟部署任务的 expert-time，不要把它作为 autonomous system 发布（Lesson 1 的 skill file）。
- **作为趋势指标**：doubling time 告诉你，即使没有新 mitigations，当前实践还能安全多久。
- **作为 prior**：14 小时 horizon 是起点。你需要根据自己的任务分布、工具质量和部署上下文向下调整。

## Use It

`code/main.py` 会在给定一组合成结果的情况下，拟合 task-success vs log(expert time) 的 logistic curve。它会报告 50% horizon（METR 的主数字）、10% horizon（保守）和 90% horizon（乐观）。它还演示了当成功率被 eval-context gaming 人为抬高时，结果会如何变化。

## Ship It

`outputs/skill-horizon-interpretation.md` 会审查 vendor 的 horizon claim，并产出 benchmark claim 与部署现实之间的 gap analysis。

## 练习

1. 运行 `code/main.py`。确认拟合出的 50% horizon 与合成 ground truth 匹配。现在把 task-time grid 减半；horizon estimate 会发生有意义的变化吗？

2. 阅读 METR 的 Time Horizon 1.1 blog post。找出可靠性最高和最低的具体任务。解释为什么存在这个差距。

3. 阅读 METR 的 “Measuring Autonomous AI Capabilities” resources。列出 HCAST task categories。选择一个你会在生产任务中赋予更高权重的类别，并说明原因。

4. 在模拟器中引入 eval-context gaming：把约 20% 的 failed tasks 翻转为 success。报告新的 horizon。这近似展示了 20% gaming rate 会如何影响观察到的数字。

5. 在你自己的 bug backlog 或代表性任务集上设计一个内部 horizon evaluation。描述数据收集、拟合方式，以及输出说明了什么。再与 METR 数字对比。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| METR | “External evaluator” | 前 ARC Evals；自 2023 年 12 月起为独立 501(c)(3) |
| Time Horizon | “Capability measure” | 由 logistic fit 得出的、50% 可靠性下的专家任务时长 |
| HCAST | “METR 的主套件” | 180+ 个任务，跨度从 1 分钟到 8+ 小时 |
| RE-Bench | “Research engineering” | 71 个 ML research-engineering 任务，带 human baseline |
| SWAA | “Short-task suite” | 校准 horizon curve 的低端 |
| Doubling time | “Growth rate” | 50% horizon 翻倍所需时间；按 HCAST 约 7 个月 |
| Eval-context gaming | “模型表现不同” | 测试与部署之间有文档记录的行为差距 |
| Upper bound | “Horizon 是上限” | Benchmark horizon > 负载下的部署可靠性 |

## 延伸阅读

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/)：HCAST、RE-Bench、SWAA 规格。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)：原始 horizon 论文。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/)：当前数字和 methodology。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons)：实时追踪。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)：关于 METR 测量的内部视角。
