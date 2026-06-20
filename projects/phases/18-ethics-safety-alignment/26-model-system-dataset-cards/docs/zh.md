# 模型卡、系统卡与数据集卡

> 三种文档格式为 AI transparency 提供结构。Model Cards（Mitchell et al. 2019）是模型的 nutrition labels：training data、quantitative disaggregated analyses、ethical considerations、caveats；Oreamuno et al. 2023 发现 Hugging Face model cards 中只有 0.3% 记录 ethical considerations。Datasheets for Datasets（Gebru et al. 2018，CACM）覆盖 motivation、composition、collection process、labeling、distribution、maintenance；类比电子元件 datasheet。Data Cards（Pushkarna et al., Google 2022）使用 modular layered detail（telescopic、periscopic、microscopic），作为面向多类读者的 boundary objects。2024-2025 年的新进展：通过 LLMs 自动生成（CardGen，Liu et al. 2024）；model-card detail 与 HF 上最高 29% 的下载增长相关（Liang et al. 2024）；verifiable attestations（Laminator，Duddu et al. 2024）；面向 carbon/water 的 sustainability reporting additions（Jouneaux et al. July 2025）；EU/ISO regulatory cards 正在出现。System Cards（Sidhpurwala 2024；Meta system-level transparency；“Blueprints of Trust” arXiv:2509.20394）是 end-to-end AI system documentation，覆盖 security capabilities、prompt-injection protection、data-exfiltration detection，以及与 human values 的 alignment。

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 minutes

## 学习目标

- 描述 Mitchell et al. 2019 model card 和 Gebru et al. 2018 datasheet 的原始定义。
- 描述 Data Cards 的 telescopic/periscopic/microscopic 分层方式。
- 描述 System Cards 及其 end-to-end coverage。
- 说出三个 2024-2025 年进展（automated generation、verifiable attestations、sustainability reporting）。

## 问题

Regulatory frameworks（第 24 课）和 lab safety policies（第 18 课）都需要 documentation。Documentation formats 从 model-specific（model cards）演化到 dataset-specific（datasheets），再到 system-specific（system cards）。每一种都对应不同范围的 transparency。2024-2025 年的 automation 和 verifiable-attestation 工作，正是在解决长期存在的 adoption problem。

## 概念

### Model Cards（Mitchell et al. 2019）

Sections：

- Model details。
- Intended use。
- Factors（用于 evaluation 的相关 demographic 或 environmental factors）。
- Metrics。
- Evaluation data。
- Training data。
- Quantitative analyses（按 factors disaggregated）。
- Ethical considerations。
- Caveats and recommendations。

Adoption problem：Oreamuno et al. 2023 对 Hugging Face model cards 的 audit 发现，只有 0.3% 记录 ethical considerations。

### Datasheets for Datasets（Gebru et al. 2018）

电子元件 datasheet 类比。Sections：

- Motivation（为什么创建这个 dataset）。
- Composition（里面包含什么）。
- Collection process（如何组装）。
- Labeling（如适用）。
- Uses（intended、prohibited、risks）。
- Distribution。
- Maintenance。

发表于 CACM 2021。Datasheet 是 upstream documentation；model card 依赖 datasheet 的准确性。

### Data Cards（Pushkarna et al., Google 2022）

Modular layered detail。三个 zoom levels：

- **Telescopic。** 面向 non-experts 的 high-level summary。
- **Periscopic。** 面向 ML practitioners 的 middle-level overview。
- **Microscopic。** 面向 auditors 的 detailed feature-level documentation。

Boundary-object framing：不同读者从同一份文档中提取不同信息。

### System Cards

Scope：end-to-end AI system，包括 model + safety stack + deployment context。典型 sections 包括：

- Security capabilities。
- Prompt-injection protection。
- Data-exfiltration detection。
- 与 stated human values 的 alignment。
- Incident response。

Sidhpurwala 2024 和 Meta system-level transparency work。“Blueprints of Trust”（arXiv:2509.20394）把 System Card 形式化为 Model Cards 的 deployment-layer complement。

### 2024-2025 年进展

- **CardGen（Liu et al. 2024）。** 通过 LLMs 自动生成 model cards；在标准化的 Mitchell 2019 fields 上，相比许多人写的 cards，报告出更高的 objectivity。
- **Download correlation（Liang et al. 2024）。** 详细的 model cards 与 HF 上最高 29% 的更高下载率相关；adoption pressure 现在不仅来自 compliance，也来自市场。
- **Laminator（Duddu et al. 2024）。** 通过 hardware TEE / cryptographic signatures 实现 verifiable attestations；让 model card 携带 proof-of-claim，而不仅是 claim。
- **Sustainability（Jouneaux et al. July 2025）。** 增加 carbon、water 和 compute-energy footprint 字段；emerging ISO standards。
- **Regulatory cards。** EU AI Act（第 24 课）GPAI Code of Practice 的 Transparency chapter 要求 model cards 作为 compliance artifact。

### 它在 Phase 18 中的位置

第 24-25 课是 regulatory 和 CVE layers。第 26 课是 documentation layer。第 27 课是 training-data governance，也就是 datasheet 的 upstream。第 28 课是 research ecosystem，产出 cards 中引用的 evaluations。

## Use It

`code/main.py` 为一个 toy deployment 生成 minimal model card、datasheet 和 system card。每份文档都遵循 canonical section structure。你可以检查格式，并比较三种 scope。

## Ship It

本课产出 `outputs/skill-card-audit.md`。给定一个 model card、datasheet 或 system card，它会 audit section coverage、numerical disaggregation，以及是否存在 verifiable attestations。

## 练习

1. 运行 `code/main.py`。检查生成的 cards。识别哪些 sections 很弱（只有 placeholder），并说明什么 evidence 可以强化它们。

2. 用两个 demographic groups 之间的 quantitative disaggregated analysis 扩展 model card（第 20 课）。

3. 阅读 Oreamuno et al. 2023 关于 0.3% adoption rate 的内容。提出一个对 model card specification 的 structural change，以提高 ethical-considerations adoption。

4. Laminator（Duddu et al. 2024）使用 TEEs 做 verifiable attestations。设计一个 model-card field，用它承载 evaluation result 的 cryptographic attestation，并描述 verifier 的角色。

5. 为你过去的一个项目或一个 hypothetical deployment 写一份 System Card（System Card，不是 Model Card）。识别对 third-party auditors 价值最高的 section。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Model Card | “the Mitchell card” | Mitchell et al. 2019 面向 ML models 的标准 documentation |
| Datasheet | “the Gebru datasheet” | Gebru et al. 2018 面向 datasets 的标准 documentation |
| Data Card | “the Pushkarna card” | Google 2022 modular layered data documentation |
| System Card | “the deployment card” | 包含 safety stack 的 end-to-end AI system documentation |
| Boundary object | “different readers, one doc” | Data Cards framing：同一份文档服务不同 audiences |
| Verifiable attestation | “the Laminator attestation” | 附加在 documentation claim 上的 cryptographic 或 TEE proof |
| Sustainability field | “carbon / water footprint” | 2025 年出现的 environmental accounting additions |

## 延伸阅读

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — canonical model card
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — datasheet paper
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — layered data documentation
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card formalization
