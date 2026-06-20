# 数据溯源与训练数据治理

> EU AI Act 要求 GPAI 在 2025 年 8 月前支持 machine-readable opt-out standards（通过 EU Copyright Directive 的 TDM exception）。California AB 2013（2024 年签署）要求 generative AI training-data transparency：开发者必须发布包含 12 个强制字段的数据集摘要。2025 年 DPA 对 legitimate interest 的立场趋同：Irish DPC（2025 年 5 月 21 日）在 EDPB opinion 之后，接受 Meta 在加入 safeguards 的情况下使用第一方公开 EU/EEA 成人内容训练 LLM；Cologne Higher Regional Court（2025 年 5 月 23 日）驳回 injunction；Hamburg DPA 放弃 urgency procedure；UK ICO（2025 年 9 月 23 日）对 LinkedIn 的 AI-training safeguards（transparency、简化 opt-out、延长 objection windows）给出 positive regulatory response，并继续 monitoring——这不是 formal clearance。Brazilian ANPD（2024 年 7 月 2 日）因 information transparency 不足，暂停 Meta 的 processing；在 Meta 提交 compliance plan 后，preventive measure 于 2024 年 8 月 30 日解除。关键的 irreversibility problem：cookie-consent frameworks 是为 real-time、reversible tracking 设计的；一旦数据进入 model weights，就无法 surgical erasure——trained neural networks 没有 practical GDPR right-to-erasure。合规窗口发生在 collection time。Data Provenance Initiative（dataprovenance.org，Longpre、Mahari、Lee 等，“Consent in Crisis”，2024 年 7 月）：大规模 audit 显示，随着 publishers 添加 robots.txt restrictions，AI data commons 正在快速收缩。

**Type:** Learn
**Languages:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60 minutes

## 学习目标

- 描述 California AB 2013 对 Generative AI training-data transparency 要求的 12 个强制字段。
- 说出 2025 年 DPA 对 legitimate-interest LLM training 的立场（Irish DPC、UK ICO、Hamburg、Cologne）。
- 描述 irreversibility problem：为什么 GDPR right-to-erasure 对 trained neural networks 没有 practical equivalent。
- 说出 Data Provenance Initiative 的 “Consent in Crisis” 发现。

## 问题

Training-data governance 是每一张 model card（第 26 课）和每项 regulatory obligation（第 24 课）的上游。2024-2025 年，监管格局收敛到三个原则：opt-out infrastructure、per-dataset disclosure，以及针对 publicly available data 的 legitimate-interest accommodations。Providers 如果没有在 collection time 合规，之后在 downstream 很难补救。

## 概念

### California AB 2013

2024 年签署。对于 2022 年 1 月 1 日及之后发布的 systems，相关 documentation 必须在 2026 年 1 月 1 日或之前发布。Section 3111(a) 要求 developers 发布 training 中使用的数据集 high-level summary，并包含 12 个 statutory items：

1. Datasets 的 sources 或 owners。
2. 说明 datasets 如何推进 AI system 的 intended purpose。
3. Datasets 中 data points 的数量（可用 general ranges；dynamic datasets 可用 estimates）。
4. Data points 类型说明（labeled datasets 的 label types；unlabeled 的 general characteristics）。
5. Datasets 是否包含受 copyright、trademark 或 patent 保护的数据，或是否完全属于 public domain。
6. Datasets 是否被 purchased 或 licensed。
7. Datasets 是否包含 personal information（按 Cal. Civ. Code §1798.140(v)）。
8. Datasets 是否包含 aggregate consumer information（按 Cal. Civ. Code §1798.140(b)）。
9. Developer 对数据进行的 cleaning、processing 或 other modification，以及 intended purpose。
10. Data collection 的时间段；如果 collection ongoing，需要说明。
11. Datasets 在 development 中首次使用的日期。
12. System 是否使用或持续使用 synthetic data generation。

Item 12（synthetic data）相对于 Gebru et al. 2018 datasheets 是新增内容。Item 7（personal information）会触发 Privacy Rights Act（CPRA）obligations。该 statute 豁免 security/integrity、aircraft-operation，以及 federal-only national-security systems（Section 3111(b)）。

### EU AI Act（第 24 课）与 TDM opt-out

EU Copyright Directive 的 text-and-data-mining exception 允许在 publicly available content 上训练，除非 rightholder opt out。EU AI Act GPAI Code of Practice 的 Copyright chapter 要求 GPAI providers 尊重 machine-readable opt-out signals（robots.txt、C2PA “No AI Training” claim 等）。

### 2025 年 DPA 对 legitimate interest 的收敛

Irish DPC（2025 年 5 月 21 日）：在 EDPB opinion 之后，Meta 训练第一方公开 EU/EEA 成人用户内容的计划在加入 safeguards 的情况下被接受。Cologne Higher Regional Court（2025 年 5 月 23 日）驳回针对 Meta 的 injunction：opt-out 已经足够。Hamburg DPA 为了 EU-wide consistency 放弃 urgency procedure。UK ICO（2025 年 9 月 23 日）对 LinkedIn 在相似 safeguards 和 ongoing monitoring 下恢复 AI training 给出 positive regulatory response——不是 formal clearance。

收敛原则：legitimate interest 可以为在 publicly available first-party content 上训练提供依据，只要提供 opt-out。不需要 consent。

### Brazilian ANPD（2024 年 6 月）

由于 information transparency 不足，ANPD 暂停 Meta 使用 Brazilian user data 进行 AI training。这个结果与 EU DPAs 不同——ANPD 优先关注 transparency，而不是 legitimate-interest admissibility。

### Irreversibility problem

Cookie-consent 是为 real-time、reversible tracking 设计的。Training data 不同：数据一旦进入 model weights，就无法 surgical erasure。Retraining from scratch 是唯一完整 remediation，但成本高到几乎不可行。

部分 remediation：

- **Unlearning。** 近似移除；通过 MIA（第 22 课）衡量。
- **Influence function-based localization。** 找出受该数据影响最大的 weights，并选择性更新。
- **Fine-tune-suppression。** 训练模型拒绝输出来自该数据的内容。

这些方法都没有完全解决问题。合规窗口在 collection time。

### Data Provenance Initiative

dataprovenance.org。Longpre、Mahari、Lee 等，“Consent in Crisis”（2024 年 7 月）：对 AI training data commons 的大规模 audit。发现：publishers 正在以加速速度添加 robots.txt restrictions。可公开训练的 commons 正在快速收缩。2023 -> 2024 年间，top training sources 中约 25% 添加了某种 restriction。含义：未来 training-data availability 依赖新的 acquisition paradigms（licensing、synthetic generation、incentivized participation）。

### 它在 Phase 18 中的位置

第 26 课是 model-level documentation。第 27 课是 dataset-level governance。二者共同定义 transparency layer。第 28 课会映射研究这些问题的 research ecosystem。

## Use It

`code/main.py` 会为一个 toy dataset 生成符合 California AB 2013 的 12-field dataset summary scaffold。你可以填写这些字段，并观察哪些字段会触发 privacy 或 copyright follow-on obligations。

## Ship It

本课产出 `outputs/skill-provenance-check.md`。给定一个用于 training 的 dataset，它会检查 AB 2013 12-field coverage、opt-out infrastructure compliance、DPA alignment，以及 irreversibility-risk assessment。

## 练习

1. 运行 `code/main.py`。为一个 toy dataset 生成 12-field summary，并识别哪些字段 under-specified。

2. EU Copyright Directive 的 TDM opt-out 是 machine-readable。提出一种 opt-out signal 的 standard format，并将它与 robots.txt 和 C2PA “No AI Training” 比较。

3. 阅读 Data Provenance Initiative 的 “Consent in Crisis”（2024 年 7 月）。描述三个 restriction 增长最快的 content categories，并论证一个 economic consequence。

4. 2025 年 DPA alignment 接受 public-content training 的 legitimate interest。构造一个 legitimate interest 不足以支撑训练的场景，并指出 provider 需要的 legal basis。

5. 勾勒一个 training-data-provenance manifest，使它能与 AB 2013 字段以及每个 dataset 的 C2PA-signed provenance chain 组合。指出一个 technical barrier 和一个 legal barrier。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| AB 2013 | “the California law” | Generative AI training-data transparency；12 个强制字段 |
| TDM exception | “text-and-data-mining” | 带 opt-out 的 EU Copyright Directive training-data exception |
| Legitimate interest | “the EU basis” | GDPR Article 6 basis，可能支撑 public content training |
| Opt-out signal | “machine-readable no-train” | robots.txt、C2PA “No AI Training”、TDM.Reservation |
| Irreversibility | “cannot un-train” | Model weights 中的数据无法 surgical removal |
| Unlearning | “approximate removal” | Post-training interventions，用于降低模型对特定数据的依赖 |
| Consent in Crisis | “the DPI audit” | 2024 年 7 月关于 robots.txt restrictions 加速增长的发现 |

## 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Generative AI training-data transparency law
- [EU AI Act + GPAI Code of Practice（第 24 课）](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Copyright chapter
- [Longpre, Mahari, Lee et al. — Consent in Crisis（dataprovenance.org，2024 年 7 月）](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI audit
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — regulatory context
