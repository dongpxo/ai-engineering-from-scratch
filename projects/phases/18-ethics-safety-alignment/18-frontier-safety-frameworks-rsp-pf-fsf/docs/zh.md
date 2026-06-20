# Frontier Safety Frameworks — RSP、PF、FSF

> 三个 major-lab frameworks 定义了 2026 年 frontier capability 的行业治理方式。Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）引入分层 AI Safety Levels（ASL-1 到 ASL-5+），仿照 biosafety levels；其中 ASL-3 于 2025 年 5 月因 CBRN-relevant models 被激活。OpenAI Preparedness Framework v2（2025 年 4 月）为 tracked capabilities 定义了五个 criteria，并把 Capabilities Reports 和 Safeguards Reports 分开。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入 Critical Capability Levels，包括新的 Harmful Manipulation CCL。三者现在都包含 competitor-adjustment clauses，允许在 peer labs 没有 comparable safeguards 却发布时推迟部分要求。跨 lab 的 alignment 是结构性的，不是术语性的：“Capability Thresholds”、“High Capability thresholds” 和 “Critical Capability Levels” 指的是类似构造。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 minutes

## 学习目标

- 描述 Anthropic 的 ASL tier structure，以及是什么激活了 ASL-3。
- 说出 OpenAI Preparedness Framework v2 对 tracked capabilities 的五个 criteria。
- 描述 DeepMind 的 Critical Capability Level structure，以及 Harmful Manipulation CCL。
- 解释 competitor-adjustment clauses，以及它们为什么会影响 race dynamics。
- 定义 safety case，并描述它的 three-pillar structure（monitoring、illegibility、incapability）。

## 问题

第 7-17 课已经说明：deception 是可能的，dual-use capability 确实存在，evaluation 也有边界。一个拥有 frontier-capable model 的 lab 需要一套内部治理结构，用来：

- 定义什么时候需要新的 safeguards。
- 定义 scaling 前需要完成哪些 evaluations。
- 描述 safety case 应该长什么样。
- 处理 race-dynamic problem（如果竞争对手在没有 safeguards 的情况下发布，你该怎么办？）。

2025-2026 年的三个 frameworks 是当前 state of the art：不完美、仍在演化，但跨 labs 已经足够 aligned，因此治理问题现在不再是“这些 frameworks 是否存在”，而是“它们是否足够好”。

## 概念

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL structure：

- ASL-1：不是 frontier model（被 weaker-than-frontier baseline 覆盖）。
- ASL-2：当前 frontier baseline；使用常规 safeguards 部署。
- ASL-3：catastrophic misuse 风险显著更高；CBRN-relevant capabilities。2025 年 5 月激活。
- ASL-4：跨过 AI R&D-2 threshold；能够自动化 entry-level AI research 的模型。
- ASL-5+：advanced AI R&D；能够显著加速 effective scaling 的模型。

v3.0 的新增内容：

- Frontier Safety Roadmaps（以 redacted form 公开）。
- Risk Reports（季度发布，其中一部分 externally reviewed）。
- AI R&D 被拆分为 AI R&D-2 和 AI R&D-4。
- 一旦跨过 AI R&D-4，就需要 affirmative safety case，明确识别来自模型追求 misaligned goals 的 misalignment risks。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

Tracked capabilities 的五个 criteria：

- **Plausible.** 存在合理的 threat model。
- **Measurable.** 可以进行 empirical evaluation。
- **Severe.** Harm 很大。
- **Net-new.** 不是已有风险的简单放大。
- **Instantaneous-or-irremediable.** Harm 发生很快，或无法被撤销。

同时满足五项的 capabilities 会被 tracked。其他则不会。

PF v2 的其他结构：

- 将 Capabilities Reports（模型能做什么）和 Safeguards Reports（有哪些 controls）分开。
- Safety Advisory Group review。
- Leadership approve；Board 的 Safety & Security Committee oversee。
- “Adjustment clause”：如果另一个 lab 在没有 comparable safeguards 的情况下发布，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按 domain 划分的 Critical Capability Levels（CCLs）：

- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation（v3.0 新增）：能够在 high-stakes contexts 中显著改变 beliefs/behavior 的模型。

v2.0（2025 年 2 月）增加了 Deceptive Alignment section，并为 ML R&D CCLs 增加了更高 security levels。

### 跨 lab alignment

- Anthropic：“Capability Thresholds”。
- DeepMind：“Critical Capability Levels”。
- OpenAI：“High Capability thresholds”。

行业没有标准术语。结构上是 aligned 的：三层 frontier capability，并配有公开 evaluation criteria。三者都包含 competitor-adjustment clauses（2025 年之后）。

### Safety cases

Safety case 是一份书面论证：在 worst-case assumptions 下，某个 deployment 仍然 acceptably safe。标准结构瞄准三个支柱：

- **Monitoring.** 如果 bad behaviour 发生，我们能检测到吗？
- **Illegibility.** 模型是否缺乏执行连贯 harm plan 的能力？
- **Incapability.** 模型是否缺乏造成相关 harm 的 capability？

不同 safety cases 会瞄准不同支柱。对于 ASL-3 CBRN case，incapability（通过 unlearning）是主要目标。对于 deceptive alignment，monitoring 和 illegibility 是目标。对于 cyber uplift，三者都相关。

### Race-dynamic problem

Competitor-adjustment clauses 很有争议。批评者认为它们会制造 race to the bottom：如果三个 labs 都会在竞争对手 defect 时降低要求，equilibrium 就会移向 defection。支持者认为，如果 defecting lab 更不重视 safety，那么 unilateral safeguards 的替代方案会产生更糟结果。

UK AISI、US CAISI 和 EU AI Office（第 24 课）是外部治理 counterpart。Lab frameworks 是 voluntary 的；regulatory frameworks 正在形成。

### 它在 Phase 18 中的位置

第 17-18 课是 deception 和 red-team analyses 之上的 measurement-and-governance layer。第 19-24 课覆盖 welfare、bias、privacy、watermarking 和 regulatory structure。第 28 课会梳理 operationalizes evaluations 的 research ecosystem（MATS、Redwood、Apollo、METR）。

## Use It

本课没有代码。阅读三个 primary sources：RSP v3.0、PF v2、FSF v3.0。把每个 lab 的 tier structure 映射到其他 labs，并找出每个 lab 定义了、而其他 labs 没有定义的一个 threshold。

## Ship It

本课产出 `outputs/skill-framework-diff.md`。给定一个 safety framework 或 release note，它会将该 framework 的 threshold definitions、required evaluations 和 safety-case structure 与 RSP v3.0、PF v2、FSF v3.0 对比，并标记 cross-lab gaps。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。整理一张表，列出每个 lab 的 CBRN threshold、AI R&D threshold，以及每个 lab 要求的 pre-deployment evaluation。

2. 三个 frameworks（2025+）都包含 competitor-adjustment clause。分别写一段支持它和反对它的论证。指出每个立场依赖的 assumption。

3. 为一个跨过 Anthropic AI R&D-4 threshold 的模型设计 safety case。说出 three pillars（monitoring、illegibility、incapability）各自需要什么 evidence。

4. DeepMind 的 FSF v3.0 引入 Harmful Manipulation CCL。提出三个 empirical measurements，用来判断模型是否跨过这个 threshold。

5. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”（2025）。说出三个最强的 cross-lab convergences，以及两个最大的 divergences。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| RSP | “Anthropic 的 framework” | Responsible Scaling Policy；ASL tiers；v3.0 于 2026 年 2 月发布 |
| PF | “OpenAI 的 framework” | Preparedness Framework；five criteria；v2 于 2025 年 4 月发布 |
| FSF | “DeepMind 的 framework” | Frontier Safety Framework；CCLs；v3.0 于 2025 年 9 月发布 |
| ASL-3 | “类似 biosafety level 3” | Anthropic 面向 CBRN-relevant capabilities 的 tier；2025 年 5 月激活 |
| CCL | “critical capability level” | DeepMind 的 threshold construct；按 domain 划分 |
| Safety case | “formal argument” | 在 worst-case assumptions 下证明 deployment acceptably safe 的书面论证 |
| Adjustment clause | “competitor defection allowance” | 如果竞争对手在没有 comparable safeguards 的情况下发布，framework 允许降低要求的条款 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy)：ASL tiers、roadmaps、AI R&D disaggregation。
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/)：five criteria、adjustment clause。
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)：CCL v3.0、Harmful Manipulation。
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)：cross-lab comparison。
