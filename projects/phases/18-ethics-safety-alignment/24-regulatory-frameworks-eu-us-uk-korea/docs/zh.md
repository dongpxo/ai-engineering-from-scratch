# 监管框架：EU、US、UK、Korea

> 四个主要监管制度定义了 2026 年的 AI governance landscape。EU AI Act（2024 年 8 月 1 日生效）：prohibited practices 和 AI literacy 自 2025 年 2 月 2 日起适用；GPAI obligations 自 2025 年 8 月 2 日起适用；full applicability 和 Article 50 transparency 于 2026 年 8 月 2 日适用；legacy GPAI 和 embedded high-risk systems 于 2027 年 8 月 2 日适用；罚款最高可达 1500 万欧元或全球营业额的 3%。GPAI Code of Practice（2025 年 7 月 10 日）：三章，Transparency、Copyright、Safety and Security，共 12 项 commitments；enforcement 从 2026 年 8 月开始。UK AISI -> AI Security Institute（2025 年 2 月）：更名表示 scope 收窄。US AISI -> CAISI（2025 年 6 月）：NIST 下的 Center for AI Standards and Innovation；转向更 pro-growth 的姿态。Korean AI Framework Act（2024 年 12 月通过，2026 年 1 月生效）：Article 12 在 MSIT 下设立 AISI；要求 foreign AI companies 设置 local representatives，并要求 high-impact 和 generative AI 进行 risk assessment 与 safety measures。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time:** ~75 minutes

## 学习目标

- 描述 EU AI Act 的 risk tiers（prohibited、high-risk、general-purpose、limited-risk），以及 2025 年 8 月 / 2026 年 8 月 / 2027 年 8 月时间线。
- 描述 GPAI Code of Practice 的三章，以及每章约束哪些 providers。
- 描述 2025 年的两个 rebrands：UK AISI -> AI Security Institute；US AISI -> CAISI；以及每次更名暗示的 policy direction。
- 说出 Korea AI Framework Act 的核心条款。

## 问题

Lab frameworks（第 18 课）是 voluntary。Regulatory frameworks 是 compulsory。2024-2026 年，第一波 comprehensive AI regulation 开始生效。Deployers 必须把 technical controls 映射到 regulatory obligations；不同 jurisdiction 的映射方式不同。

## 概念

### EU AI Act

**2024 年 8 月 1 日生效。** Risk-tier structure：

- **Prohibited practices**（Article 5）。Social scoring、公共场所 real-time remote biometric identification（有 law-enforcement exceptions）、对 vulnerable groups 的 exploitative manipulation。2025 年 2 月 2 日适用。
- **High-risk systems**（Annex III）。Employment、education、credit、law enforcement、justice、migration。要求 conformity assessment、risk management、logging、transparency。
- **General-Purpose AI（GPAI）models**。2025 年 8 月 2 日适用。所有 GPAI providers 都有 obligations；systemic-risk GPAI（训练 compute >1e25 FLOP）有额外 obligations。
- **Limited-risk systems**。Article 50 下的 transparency obligations（AI-generated content labelling）。2026 年 8 月 2 日适用。

Timeline：

- 2025 年 2 月 2 日：prohibited practices + AI literacy。
- 2025 年 8 月 2 日：GPAI + governance。
- 2026 年 8 月 2 日：full applicability + Article 50 transparency + 罚款最高 1500 万欧元 / 全球营业额 3%。
- 2027 年 8 月 2 日：legacy GPAI + embedded high-risk。

Commission 在 2025 年末提议把 high-risk timeline 调整为 16 个月。

### GPAI Code of Practice

2025 年 7 月 10 日发布。三章：

- **Transparency。** 约束所有 GPAI providers。
- **Copyright。** 约束所有 GPAI providers。
- **Safety and Security。** 约束 systemic-risk GPAI providers（估计 5-15 家公司）。

共 12 项 commitments。由 AI Office 主持的 Signatory Taskforce 管理 implementation。Enforcement 从 2026 年 8 月 2 日开始；在此之前，good-faith compliance 会被接受。

### Article 50 Transparency Code

第一稿：2025 年 12 月 17 日。第二稿：2026 年 3 月。最终版：2026 年 6 月。覆盖 AI-generated content labelling，包括 deepfakes，也就是需要第 23 课 watermarking technology 的 regulatory layer。

### UK AI Security Institute（2025 年 2 月）

由 AI Safety Institute 更名而来。这个 rebrand 收窄了 scope：淡化 algorithmic bias 和 free-speech framings，聚焦 frontier capability security。它开源了 Inspect evaluation tool（2024 年 5 月），并与 Redwood（第 10 课）合作 control safety cases。

### US CAISI（2025 年 6 月）

Trump administration 将 NIST 的 AI Safety Institute 转换为 Center for AI Standards and Innovation。根据 VP Vance 在 Paris AI Action Summit 的表述，政策转向 “pro-growth AI policies”。Pre-deployment evaluation 的权重下降；standards 和 innovation support 的权重上升。它成为 EU AI Act regulatory posture 的 domestic counterweight。

### Korean AI Framework Act

2024 年 12 月通过。2025 年 1 月 enact。2026 年 1 月生效。整合了 19 个 separate AI bills。

Article 12 在 Ministry of Science and ICT（MSIT）下设立 AISI。要求：

- 在韩国运营的 foreign AI companies 必须设置 local representatives。
- 对 “high-impact” AI systems 进行 risk assessment。
- 对 generative AI 和 high-impact AI 采取 safety measures。

这是亚洲第一个 comprehensive horizontal AI regulation。

### Cross-jurisdiction dynamics

- EU：strict、risk-tiered、heavy penalties。Privacy-adjacent regulation 的 benchmark。
- US：innovation-favouring、decentralized；states（例如 California AB 2013，第 27 课）填补 federal gaps。
- UK：narrow security focus，strong evaluation infrastructure。
- Korea：MSIT-led，foreign-provider-focused。

这些是相互竞争的 regulatory philosophies。跨多个 jurisdictions 部署的 deployers 必须满足最严格的规则；在 2026 年，这通常是 EU AI Act。

### 它在 Phase 18 中的位置

第 18 课是 lab-voluntary governance；第 24 课是 regulatory；第 25 课是 AI systems 的 emerging CVE 类别；第 26-27 课覆盖 documentation（cards）和 training-data governance。

## Use It

没有代码。阅读 EU AI Act primary sources：regulation text、GPAI Code of Practice、UK AISI Inspect framework。把你的 deployment 映射到每个 jurisdiction 中适用的 obligations。

## Ship It

本课产出 `outputs/skill-regulatory-map.md`。给定一个 deployment description，它会映射 applicable jurisdictions、每个 jurisdiction 中的 tier classifications、per-jurisdiction obligations，以及 deadline structure。

## 练习

1. 阅读 EU AI Act（regulation 2024/1689）和 GPAI Code of Practice（2025 年 7 月 10 日）。找出三项适用于每个 GPAI provider 的 obligations，以及三项只适用于 systemic-risk GPAI 的 obligations。

2. 一个 deployment 由美国公司提供，运行在 EU infrastructure 上，并服务 Korean users。哪三个 jurisdictions 的规则适用？每个 substantive question 上是哪条规则约束？

3. UK AI Security Institute 的更名收窄了 scope。分别为这个更窄 framing 做支持和反对论证。识别每种立场依赖的 policy assumption。

4. CAISI 的 “pro-growth” framing 偏离了 2022-2024 年 AI safety institute model。指出两个可以从这种 framing 推出的 measurable policy shifts。

5. Korea AI Framework Act 要求 foreign providers 设置 local representatives。描述一家服务 Korean users 的 Bay Area 公司会遇到哪些 operational implications。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| EU AI Act | “the regulation” | 基于 risk tiers 的 horizontal AI regulation；2024 年 8 月生效 |
| GPAI | “general-purpose AI” | 大型 foundation models；systemic-risk subset 有额外 obligations |
| Article 50 | “transparency obligations” | AI-generated content labelling；2026 年 8 月适用 |
| UK AISI | “AI Security Institute” | 2025 年 2 月更名；更窄的 frontier-security focus |
| CAISI | “US center for AI standards” | 2025 年 6 月由 AI Safety Institute 更名而来；pro-growth posture |
| Korean AI Framework Act | “MSIT horizontal regulation” | 亚洲第一部 comprehensive AI law；2026 年 1 月生效 |
| Systemic-risk GPAI | “1e25 FLOP threshold” | 额外 obligations tier；估计约束 5-15 家公司 |

## 延伸阅读

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — regulation 和 timeline
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — three-chapter code
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — official page
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — Korean framework analysis
