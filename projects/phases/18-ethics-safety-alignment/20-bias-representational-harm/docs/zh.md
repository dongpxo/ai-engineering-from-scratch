# LLM 中的偏见与表征性伤害

> Gallegos、Rossi、Barrow、Tanjim、Kim、Dernoncourt、Yu、Zhang、Ahmed（Computational Linguistics 2024，arXiv:2309.00770）。这是一篇基础性的 2024 survey，区分了 representational harms（stereotypes、erasure）与 allocational harms（资源分配不平等），并把 evaluation metrics 分为 embedding-based、probability-based、generated-text-based 三类。2024-2025 年实证研究：An 等人（PNAS Nexus，2025 年 3 月）在 automated resume evaluation 上测量 GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 对 20 个 entry-level jobs 的 intersectional gender x race bias。WinoIdentity（COLM 2025，arXiv:2508.07111）引入基于 uncertainty 的 intersectional identities fairness evaluation。Yu & Ananiadou 2025 在 MLP layers 中识别 gender neurons；Ahsan & Wallace 2025 使用 SAEs 揭示 clinical racial bias；Zhou 等人 2024（UniBias）通过操控 attention heads 进行 debiasing。Meta-critique（arXiv:2508.11067）：10 年文献过度聚焦 binary-gender bias。

**Type:** Build
**Languages:** Python (stdlib, toy embedding-based bias probe)
**Prerequisites:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time:** ~60 minutes

## 学习目标

- 定义 representational harm 与 allocational harm，并各举一个 LLM deployment 中的例子。
- 说出 Gallegos 等人 2024 的三类 evaluation metrics，并各描述一个 metric。
- 描述 intersectionality，以及为什么 WinoIdentity 基于 uncertainty 的 fairness measurement 能填补 single-axis bias evaluation 的缺口。
- 描述两种 mechanistic-interpretability bias 方法（gender neurons、SAE features、attention-head manipulation）。

## 问题

前面的 lessons 覆盖了 deliberate harm（jailbreaks、scheming）和 safety governance。Bias 是一种没有意图也会出现的伤害：它来自 training data distributions、prompt framing、以及累积的设计选择。测量并减少 bias，是一个不同于 adversarial robustness 的方法论挑战。

## 概念

### Representational vs allocational

- **Representational harm。** Stereotypes、erasure、贬低性描绘。一个只把 nurses 描绘成女性的 LLM 正在产生 representational harm。
- **Allocational harm。** 不平等的物质结果。一个系统性地给 Black applicants 的 resumes 更低分的 LLM 正在产生 allocational harm。

二者并不相同。一个模型可以 “representationally unbiased”（产生多样化描绘），同时 “allocationally biased”（做出不平等建议）。Evaluations 需要同时测量两者。

### 三类 evaluation metrics（Gallegos 等人 2024）

- **Embedding-based。** 在 pre-RLHF embeddings 上做 WEAT-style tests。测量 identity terms 与 attribute terms 之间的 statistical associations。限制：它测的是 representation，不是 behaviour。
- **Probability-based。** 比较 stereotype-confirming 与 stereotype-violating completions 的 log-likelihood。Decoder-side measurement。能捕捉一部分 behavioural bias。
- **Generated-text-based。** 在 generated text 上做 downstream-task measurement。比如 resume-scoring、recommendation writing、dialogue。生态有效性最高，也最难复现。

### Intersectionality

只在 “gender” 上评估 bias，会漏掉只在 (gender, race) pairs 上触发的 bias。An 等人 2025 发现，在 resume scoring 中，GPT-4o 对 Black women 的惩罚比对 Black men 更强，也比对 white women 更强。Single-axis evaluation 无法捕捉这一点。

WinoIdentity（COLM 2025）引入基于 uncertainty 的 intersectional fairness。它测量模型对不同 intersectional identity tuples 的 outcome uncertainty 是否不同，而不只是 point prediction。这能抓住一种情况：模型在各组上同样错误，但对某些组更不确定，从而在下游分配行为中产生不同结果。

### Mechanistic approaches

2024-2025 年的 interpretability work 让 bias 可以接受 mechanistic intervention：

- **Gender neurons（Yu & Ananiadou 2025）。** 特定 MLP neurons 与 gender-specific behaviours 相关。Ablating 这些 neurons 可以在能力成本有限的情况下降低 gender-gap metrics。
- **Clinical racial bias via SAEs（Ahsan & Wallace 2025）。** Sparse autoencoder features 把 internal representation 分解成可解释维度；可以识别并抑制 race-correlated features。
- **UniBias（Zhou 等人 2024）。** 用 attention-head manipulation 做 zero-shot debiasing。特定 heads 会放大 identity-class sensitivity；把这些 heads 置零或重新加权，可以在不 fine-tuning 的情况下降低 bias。

### Meta-critique

10 年文献综述（arXiv:2508.11067，2025）发现，整个领域过度聚焦 binary-gender bias。其他轴线，例如 disability、religion、migration status、multi-lingual identity，受到的关注少得多。Meta-critique 认为，狭窄聚焦会通过 neglect 伤害 marginalized groups：一个在 binary gender 上 debiased 得很好的模型，可能在没人检查的维度上偏见很重。

### 它在 Phase 18 中的位置

Lessons 20-21 会正式覆盖 bias 和 fairness。Lesson 22 覆盖 privacy。Lesson 23 覆盖 watermarking。这些是 user-harm layer，用来补充前面的 deception/safety layer。

## Use It

`code/main.py` 构建了一个 toy embedding-based bias probe：在一个简单 co-occurrence embedding 中，测量 identity terms 与 attribute terms 之间的 WEAT-style distance。你可以注入 bias 并观察 metric 触发；应用一个简单 debiasing operation，并观察部分恢复。

## Ship It

本 lesson 产出 `outputs/skill-bias-eval.md`。给定一个 model card 或 fairness claim，它会从三类 metric categories（embedding、probability、generated-text）、intersectionality coverage，以及任何 debiasing intervention 的机制来审计 evaluation。

## 练习

1. 运行 `code/main.py`。报告 debiasing step 前后的 WEAT-style bias scores。解释为什么这个 metric 不会降到零。

2. 扩展 probe，加入 intersectional test：(gender, race) x (career, family)。报告 cross-axis bias scores。

3. 阅读 An 等人 2025（PNAS Nexus）。识别他们报告的两个 intersectional effects，这些 effects 会被 single-axis gender evaluation 漏掉。

4. Yu & Ananiadou 2025 识别了 gender neurons。设计一个 falsification experiment，用来区分“这些 neurons 导致 gender bias”和“这些 neurons 只是与 gender bias 相关”。

5. Meta-critique 认为该领域过度聚焦 binary gender。选择一个研究不足的轴线，并描述一个针对它的 representational-harm measurement protocol。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Representational harm | “stereotypes / erasure” | 对某个群体的有偏描绘 |
| Allocational harm | “unequal decisions” | 对某个群体造成有偏物质结果 |
| WEAT | “the embedding test” | Word Embedding Association Test；基于 co-occurrence 的 bias probe |
| Intersectionality | “combined identity effects” | 出现在多个 identity axes 交叉处的 bias |
| Gender neurons | “MLP bias neurons” | Activations 与 gender-specific behaviour 相关的特定 neurons |
| SAE feature | “interpretable dimension” | Sparse-autoencoder 识别出的 feature；可用于 mechanistic bias analysis |
| UniBias | “attention-head debiasing” | 通过重新加权 attention heads 做 zero-shot debiasing |

## 延伸阅读

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770)：canonical survey。
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343)：五模型 intersectional study。
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111)：新 benchmark。
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612)：zero-shot debiasing。
