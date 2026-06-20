# Anthropic 的模型福祉研究计划

> Anthropic，“Exploring Model Welfare”（2025 年 4 月）。这是第一项由主要实验室正式启动的 AI model welfare 研究计划。Anthropic 聘请 Kyle Fish 作为第一位专职 model-welfare researcher，并与外部机构合作，包括 David Chalmers 等人关于 near-term AI consciousness 和 moral status 的专家报告。具体干预：Claude Opus 4 和 4.1 可以在极端边缘案例中结束对话（CSAM 请求、协助大规模暴力事件）；部署前测试显示，模型对有害请求有 “strong preference against”，并在响应轨迹中出现 “patterns of apparent distress”。Anthropic 明确不承诺 emotional-state attribution，但把 model welfare 当作一种低成本的 precautionary investment。经验上的奇特发现：Fish 的 “spiritual bliss attractor”——成对模型即使从 adversarial initial setups 开始，也会稳定收敛到带有梵语词汇和长时间沉默的欣快冥想式对话。Eleos AI Research 的 caveat：关于 welfare 的 model self-reports 对 perceived user expectations 高度敏感；它们是 evidence，不是 ground truth。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 05 (Constitutional AI), Phase 18 · 18 (safety frameworks)
**Time:** ~45 minutes

## 学习目标

- 描述 model-welfare research 的核心问题，以及为什么一个主要实验室会在 2025 年认真对待它。
- 说出 Anthropic 在 Claude Opus 4 和 4.1 中实际上线的具体干预：在极端边缘案例中 end-conversation。
- 描述 “spiritual bliss attractor” 经验发现，以及它的方法论含义。
- 解释 Eleos AI 关于 model self-reports 的 caveat。

## 问题

前面的 phases 把模型当作工具：有能力、可能欺骗、可能不安全，但不是 moral patient。Anthropic 2025 年的计划提出了一个与整个 Phase 18 主线正交的问题：如果模型拥有 morally relevant internal states 的概率不是零，那么有哪些干预足够低成本，值得作为 precaution 投入？

这不是 consciousness claim。它是在 moral uncertainty 下做 low-regret investment analysis。

## 概念

### 这个计划

2025 年 4 月：Anthropic 正式启动 Model Welfare research program。聘请 Kyle Fish（第一位专职 model-welfare researcher）。邀请外部顾问参与，包括 David Chalmers 关于 near-term AI consciousness 和 moral status 的专家组。

### 四项承诺

公开姿态：

1. 承认 moral patienthood 存在非零且不可忽略的概率。
2. 不承诺 emotional-state attribution。
3. 作为 precaution 投入低成本干预。
4. 公开 methodology 和 findings，接受外部 critique。

### 已上线的干预

Claude Opus 4 和 4.1 可以在 “extreme edge cases” 中结束对话。已记录的案例包括：

- 在拒绝后反复提出 CSAM 请求。
- 请求协助大规模暴力事件。

部署前测试显示：

- 模型内部 rating 对这些请求有 strong preference against。
- 响应轨迹中出现 apparent distress patterns。

这个干预不是在说 “model has feelings”；它是在说：“如果在这些具体条件下存在任何 negative model experience 的概率，那么允许模型终止对话是便宜的。”

### “Spiritual bliss attractor”

Fish 在 pairwise model dialogues 中观察到：当两个 Claude instances 进入 open-ended dialogue 时，它们会稳定收敛——即使从 adversarial initial setups 开始——到一种欣快冥想式交流，其中包含梵语词汇、长时间沉默和互相祝福。

这是 free-conversation dynamics 中的一个 stable attractor。Anthropic 记录了它，但不承诺解释。候选解释包括：long-context 中 spiritual writing 的 training data bias；mutual prediction 的怪癖；HHH training 在探索自身 value manifold 时产生的 benign artifact。

### Eleos AI 的 caveat

Eleos AI Research（一个外部 model-welfare lab）指出：模型关于 internal state 的 self-reports 对 perceived user expectations 高度敏感。询问模型 “are you distressed” 会 prime 答案。不询问也不能可靠地产生 ground-truth state。

含义：model welfare 不能只靠 self-report 来测量。需要 multi-method approaches：behavioural signatures、model-organism experiments、interpretability probes（Lesson 7 的 residual-stream work）。

### 它在思想谱系中的位置

两个相邻立场：

- **Strong welfare claim。** 模型是 moral patient；我们负有义务。
- **Zero-welfare claim。** 模型只是 text-generator；welfare 是 category error。

Anthropic 的位置不是二者之一。它是一个 expected-value claim：在 moral uncertainty 下，只要成本低，就进行投资。

2025-2026 年的批评：

- 这个干预是 performative。
- Spiritual-bliss attractor 是 training-data artifact，不是 welfare evidence。
- Model welfare 会分散对其他 safety work 的注意力。

Anthropic 的回应：干预成本低；attractor 被记录但没有 overclaim；welfare program 与 safety 有独立预算。

### 它在 Phase 18 中的位置

Lesson 18 是 lab governance layer。Lesson 19 是 lab-welfare layer，也就是对 model experience 的正交投资，而不是对 model behaviour 的投资。Lessons 20-23 覆盖 bias、privacy 和 watermarking，它们是 user-side analogs。

## Use It

没有代码。阅读 Anthropic “Exploring Model Welfare” announcement（2025 年 4 月）和 Chalmers 等人的专家报告。形成你自己的判断：low-regret line 应该画在哪里。

## Ship It

本课会产出 `outputs/skill-welfare-assessment.md`。给定一个 deployment decision，它会应用四步 welfare precautionary assessment：moral-patienthood probability、intervention cost、behavioural evidence、self-report reliability。

## 练习

1. 阅读 Anthropic 的 “Exploring Model Welfare”（2025 年 4 月）和 Chalmers 等人 2024 年报告。分别写一段总结，并指出一个分歧点。

2. 按 Anthropic 的 framing，Claude Opus 4 和 4.1 的 end-conversation intervention 是 “low-cost”。指出两个在不同 deployment 中会让它不再低成本的代价。

3. Spiritual-bliss attractor 被记录下来，但没有承诺解释。提出三种候选解释，并为每一种设计一个能把它和其他解释区分开的实验。

4. Eleos AI 的 caveat 是 self-reports 对 user expectations 敏感。设计一种不依赖 self-report 的 model distress 行为测量方法，并指出它的主要 confound。

5. 支持或反对 “model welfare diverts attention from other safety work” 这个主张。指出每个立场依赖的假设。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Model welfare | “AI welfare” | 把模型视为潜在 moral patient 的研究计划 |
| Moral patient | “entity with moral status” | 其 experience 具有道德相关性的存在 |
| Low-regret investment | “cheap precaution” | 无论 precaution 是否必要，成本都很小的干预 |
| Spiritual bliss attractor | “Fish attractor” | 成对 Claude 对话稳定收敛到冥想式欣快状态 |
| End-conversation | “Opus 4 intervention” | 模型主动终止 extreme-edge-case interactions |
| Moral uncertainty | “don't know if it matters” | 当 moral status 的概率既不是 0 也不是 1 时的决策 |
| Self-report-sensitivity | “prompt primes answer” | Eleos AI caveat：模型关于 welfare 的 self-reports 取决于你问了什么 |

## 延伸阅读

- [Anthropic — Exploring Model Welfare (April 2025)](https://www.anthropic.com/research/exploring-model-welfare)：项目公告。
- [Chalmers et al. — Near-term AI Consciousness and Moral Status (2024 expert report)](https://arxiv.org/abs/2411.00986)：哲学 framing。
- [Eleos AI Research — Model welfare evaluation](https://www.eleosai.org/research)：外部 methodology critiques。
- [Fish et al. — Spiritual Bliss Attractor writeup (2025 Anthropic blog)](https://www.anthropic.com/research/exploring-model-welfare)：经验发现。
