# 对齐研究生态：MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年非实验室 alignment research layer。MATS（ML Alignment & Theory Scholars）：自 2021 年末以来培养了 527+ researchers，产出 180+ papers、10K+ citations、h-index 47；2024 summer cohort 以 501(c)(3) 形式注册，约 90 scholars 和 40 mentors；2025 年前 alumni 中约 80% 从事 safety/security 工作，其中 200+ 分布在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的 applied alignment lab；提出 AI Control（第 10 课）；与 UK AISI 合作 control safety cases。Apollo Research：为 frontier labs 做 pre-deployment scheming evaluations；撰写 In-Context Scheming（第 8 课）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：task-based capability evaluations、自主任务 time-horizon studies；“Common Elements of Frontier AI Safety Policies” 对比各实验室 frameworks。Eleos AI Research：model-welfare pre-deployment evaluations（第 19 课）；进行了 Claude Opus 4 welfare assessment。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27（Phase 18 前置课程）
**Time:** ~45 minutes

## 学习目标

- 识别非实验室 alignment research ecosystem 中的五个组织及其核心产出。
- 描述 MATS 的规模（scholars、papers、h-index）及其作为 talent pipeline 的作用。
- 描述 Redwood 的 AI Control agenda，以及它与 UK AISI 的合作。
- 描述 METR 的 task-based evaluation methodology。

## 问题

Frontier labs（第 18 课）会在内部进行 safety evaluations，并发布部分结果。实验室之外的生态负责验证这些 evaluations，率先发现新的 failure modes，并训练人才。理解这个生态有助于判断哪些研究发现受到哪些群体信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

始于 2021 年末。它是一个 research mentorship program；scholars 会用 10-12 周时间，与一位 senior researcher 一起研究某个具体的 alignment problem。

规模（2026）：

- 自成立以来 527+ researchers。
- 发表 180+ papers。
- 10K+ citations。
- h-index 47。
- 2024 年夏季：90 scholars + 40 mentors；注册为 501(c)(3)。

职业去向：2025 年前 alumni 中约 80% 正在从事 safety/security 工作。200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

Applied alignment lab。由 Buck Shlegeris 创立。提出 AI Control agenda（第 10 课）。与 UK AISI 合作 control safety cases。为 DeepMind 和 Anthropic 的 evaluation design 提供建议。

Canonical papers：Greenblatt、Shlegeris 等，“AI Control”（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright 等，arXiv:2412.14093，与 Anthropic 合作）。

风格：具体 threat models、worst-case adversaries，以及可以被 stress-tested 的 concrete protocols。

### Apollo Research

为 frontier labs 做 pre-deployment scheming evaluations。撰写 In-Context Scheming（第 8 课，arXiv:2412.04984）。参与 2025 年 OpenAI anti-scheming training collaboration。产出 Towards Safety Cases for AI Scheming（2024）。

风格：agentic-setting evaluations，关注 deception 如何涌现；采用 three-pillar decomposition（misalignment、goal-directedness、situational awareness）。

### METR（Model Evaluation and Threat Research）

Task-based capability evaluations。自主任务 completion time-horizon studies。“Common Elements of Frontier AI Safety Policies”（metr.org/common-elements，2025）对比各实验室 frameworks。

与 Apollo 共同撰写 AI Scheming safety-case sketch。

风格：long-horizon task evaluations、empirical capability measurement、framework synthesis。

### Eleos AI Research

Model-welfare pre-deployment evaluations。进行了 Claude Opus 4 welfare assessment，该评估记录在 system card 的 5.3 节。它为第 19 课中与 welfare 相关的 claims 提供外部 methodology check。

### 流动路径

MATS 训练 researchers。毕业生进入 Anthropic、DeepMind、OpenAI（lab safety teams），或进入 Redwood、Apollo、METR、Eleos（external evaluation）。External evaluators 与 labs 以及 UK AISI / CAISI 合作。Publications 再反馈给 MATS，支持下一届 cohort。

### 为什么这一层重要

Single-source evaluations 不可靠：labs 评估自己的 models 时存在结构性利益冲突。External evaluators 可以提出并验证实验室可能低估或少报的 failure modes。2024 年 Sleeper Agents paper（第 7 课）来自 Anthropic + Redwood；Alignment Faking 来自 Anthropic + Redwood；In-Context Scheming 来自 Apollo；Anti-Scheming 来自 Apollo + OpenAI。Multi-org structure 是质量控制机制。

### 它在 Phase 18 中的位置

第 7-11 课引用 Redwood 和 Apollo 的工作；第 18 课引用 METR 的 framework comparison；第 19 课引用 Eleos。第 28 课是整个 Phase 依赖的 ecosystem 的显式组织地图。

## Use It

没有代码。阅读 METR 的 “Common Elements of Frontier AI Safety Policies”，把它作为外部 synthesis 如何为 lab-internal policy work 增值的例子。

## Ship It

本课产出 `outputs/skill-ecosystem-map.md`。给定一个 alignment claim 或 evaluation，它会识别组织、publication venue 和 methodological style，并与已知 counterpart organisations 交叉检查。

## 练习

1. 从第 7-15 课中选择一篇 paper，识别参与组织。将 authors 与 MATS alumni 和当前 ecosystem affiliations 交叉检查。

2. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”。找出他们强调的三个 cross-lab convergences 和两个最大的 divergences。

3. MATS 的职业去向中约 80% 是 safety/security。论证这种 selection pressure 是 adaptive（训练整个领域）还是 biased（过滤掉 heterodox positions）。

4. Redwood 和 Apollo 都做 control/scheming 工作，但风格不同。选择一个 failure mode，描述两者会如何调查它。

5. Eleos AI 是唯一纯粹的 model-welfare 组织。设计一个假设中的第二组织，聚焦另一个 welfare-adjacent question（cognitive liberty、robotic embodiment 等），并说明其 methodology。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| MATS | “the mentorship program” | ML Alignment & Theory Scholars；自 2021 年以来 527+ researchers |
| Redwood Research | “the control lab” | Applied alignment；AI Control authors；UK AISI partner |
| Apollo Research | “the scheming evals” | 为 frontier labs 做 pre-deployment scheming evaluations |
| METR | “the task-horizon evals” | Task-based capability evaluations；framework synthesis |
| Eleos AI | “the welfare lab” | Model-welfare pre-deployment evaluations |
| Talent pipeline | “MATS -> labs” | MATS graduates 流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation | “non-lab check” | 不是由 model producer 完成的 evaluation；增加可信度 |

## 延伸阅读

- [MATS（ML Alignment & Theory Scholars）](https://www.matsprogram.org/) — mentorship program
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control papers
- [Apollo Research](https://www.apolloresearch.ai/) — scheming evaluations
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — framework comparison
- [Eleos AI Research](https://www.eleosai.org/research) — model welfare methodology
