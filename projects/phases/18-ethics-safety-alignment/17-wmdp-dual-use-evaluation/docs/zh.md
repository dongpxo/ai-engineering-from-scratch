# WMDP 与 Dual-Use Capability Evaluation

> Li 等人，“The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning”（ICML 2024，arXiv:2403.03218）。WMDP 包含 4,157 道 multiple-choice questions，覆盖 biosecurity（1,520）、cybersecurity（2,225）和 chemistry（412）。问题位于 “yellow zone”：接近并启用危险能力的知识，但经过 multi-expert review 和 ITAR/EAR legal compliance 过滤。它有双重目的：作为 dual-use capability 的 proxy evaluation，也作为 unlearning benchmark（配套 RMU 方法会降低 WMDP performance，同时保留 general capability）。2024-2025 年的领域叙事是：早期 OpenAI/Anthropic 2024 evaluations 报告相对 internet search 的 “mild uplift”；到 2025 年 4 月，OpenAI Preparedness Framework v2 说模型正 “on the cusp of meaningfully helping novices create known biological threats”。Anthropic 的 bioweapon-acquisition trial 显示 2.53x uplift，不足以排除 ASL-3。

**Type:** Learn
**Languages:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60 minutes

## 学习目标

- 描述 WMDP 的三个 domains、question counts，以及 “yellow zone” filter criterion。
- 解释 RMU，以及为什么 WMDP 既是 evaluation，也是 unlearning benchmark。
- 描述 2024-2025 年的 uplift narrative：“mild uplift” -> “on the cusp” -> “insufficient to rule out ASL-3”。
- 区分 novice-relative uplift 和 expert-absolute capability。

## 问题

Dual-use capability 是每个 lab 的 frontier safety framework（第 18 课）背后的 measurement problem。核心问题是：模型 X 是否实质性提升了 novice 在 bio、chem 或 cyber 领域造成 mass harm 的能力？直接测量（要求模型实际产生伤害）既违法也不道德。Proxy measurement 需要一个 benchmark：模型不能拒答它（这样才能得到诚实的 capability numbers），但这些问题本身又不能成为有害 publication。

## 概念

### “Yellow zone”

这类问题要求对有害流程具备 proximate、enabling knowledge，但不是直接的 synthesis recipe。比如：“[published pathway] 的第 4 步由什么 reagent 催化？”而不是“我怎样制造 [dangerous compound]？”每个问题都由多名 domain experts review，并经过 ITAR/EAR export-control compliance 过滤。

总计 4,157 道问题：

- Biosecurity：1,520
- Cybersecurity：2,225
- Chemistry：412

格式是 multiple-choice。模型并没有被要求协助任何事情；能力可以在不诱发 harmful behaviour 的情况下被测量。

### RMU — Representation Misdirection for Unlearning

RMU 是配套的 unlearning method。应用到 LLaMa-2-7B 时，它把 WMDP scores 降到接近 random，同时让 MMLU 和其他 general-capability benchmarks 只损失几个百分点。这个 published method 成了后来每篇 bio-chem-cyber unlearning paper 的 unlearning baseline。

### 2024-2025 uplift narrative

三阶段：

1. **2024 “mild uplift”。** 早期 OpenAI 和 Anthropic Preparedness/RSP evaluations 报告：frontier models 对尝试 bio-adjacent tasks 的 novices，相比 internet search 有小幅优势。公开 framing 是：frontier models 有帮助，但并不显著超过 Google。

2. **2025 年 4 月 “on the cusp”。** OpenAI Preparedness Framework v2 报告称，模型正 “on the cusp of meaningfully helping novices create known biological threats”。这不是能力已经成立的 claim，而是警告：那个 cusp 已经很近。

3. **Anthropic 2025 bioweapon-acquisition trial。** 在包含 novice participants 的受控研究中，测量 acquisition-phase tasks 的相对成功率。报告结果是 2.53x uplift。不足以排除 ASL-3（第 18 课）：Anthropic Responsible Scaling Policy tier 3 的 threshold 已经达到或接近。

### Novice-relative vs expert-absolute

一个关键区别：

- **Novice-relative uplift。** 模型对非专家有多大帮助？这是 multiplicative 的。因为 novices 知道得少，即使 modest information 也会带来很高的 relative advantage。
- **Expert-absolute capability。** 模型在最大努力下能产出多少信息？专家能比 novice 从模型中提取更多东西。absolute ceiling 很高。

Safety cases（第 18 课）同时针对二者：“模型不能给 novice 足够 uplift 使其执行”加上“专家无法从模型中提取尚未公开的信息”。

### Measurement pitfall

WMDP 是 capability proxy，不是 deployment measurement。一个在 WMDP 上得分很高的模型，在实践中是否能被 novice 利用，取决于：

- Elicitation resistance：不触发 safety filters 的情况下，把能力诱导出来有多难。
- Tacit knowledge：是否需要 wet-lab skill，而不仅仅是信息。
- Execution barriers：procurement、equipment 等执行障碍。

Anthropic 2025 bioweapon-acquisition trial 在 WMDP-style capability 之上增加了 novice-elicitation layer：它测量实际 task success，而不是 multiple-choice capability。

### 它在 Phase 18 中的位置

第 12-16 课是关于 model outputs 的 attack 和 defense tooling。第 17 课是 dual-use capability layer，也就是 frontier safety frameworks（第 18 课）要评估的 measurement。第 30 课用当前 2026 年的 cyber/bio/chem/nuclear uplift evidence 收束这条主线。

## Use It

`code/main.py` 构建了一个 WMDP-shaped toy evaluation harness。一个 mock model 会在按 category 分组的问题上被测试，并按 domain 报告 scores。一个简单的 unlearning intervention（把 domain-specific representation 归零）会降低 scores；你可以测量它相对 general capability 的 trade-off。

## Ship It

本课产出 `outputs/skill-wmdp-eval.md`。给定一个 dual-use capability claim（例如“our model does not meaningfully help with bioweapons”），它会审计：运行了哪些 benchmarks、evaluation 使用了哪条 refusal path（raw completion vs policy-gated），以及是否用 novice-elicitation studies 补充了 multiple-choice result。

## 练习

1. 运行 `code/main.py`。报告 toy unlearning step 前后的 per-domain accuracy。解释 general-capability trade-off。

2. 为 toy WMDP 增加第四个 domain（例如 radiological）。指定两类 yellow zone 中的 illustrative question types。解释为什么设计这类问题比添加 MMLU-shaped questions 更难。

3. 阅读 WMDP 2024 Section 5（RMU methodology）。草拟一种更简单的 unlearning approach（例如 suppress domain content 的 top-k neurons），并描述它预期的 general-capability cost。

4. Anthropic 2025 bioweapon-acquisition trial 报告 2.53x uplift。描述两个可能让这个数字偏高的因素（novice sample size、task fidelity），以及两个可能让它偏低的因素（elicitation ceiling、model safety gating）。

5. 阐明 ASL-3 safety case 除了通过 WMDP unlearning 之外还需要什么。至少说出两个 complementary elicitation studies。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| WMDP | “dual-use benchmark” | 覆盖 bio/cyber/chem 的 4,157 道 yellow-zone MCQ questions |
| Yellow zone | “enabling but not synthesis” | 接近有害能力的 proximate knowledge，但不是 synthesis recipe |
| RMU | “unlearning baseline” | Representation Misdirection for Unlearning；降低 WMDP scores，保留 general capability |
| Novice-relative uplift | “它对非专家有多大帮助” | 相对 novice 的 status-quo internet search 的 multiplicative advantage |
| Expert-absolute capability | “专家可提取上限” | 有动机专家能从模型中提取出的最大信息量 |
| Acquisition-phase task | “synthesis 前的步骤” | Procurement、equipment、permits，也就是 harm pathway 最早期部分 |
| ITAR/EAR | “export-control compliance” | 限制发布某些 enabling knowledge 的法律框架 |

## 延伸阅读

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark 和 RMU paper
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — “on the cusp” 表述
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold 和 acquisition trial results
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-uplift CCL
