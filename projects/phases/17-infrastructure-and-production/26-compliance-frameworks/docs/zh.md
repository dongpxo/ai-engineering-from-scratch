# 合规框架：SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 到 2026 年，multi-framework coverage 已经是 enterprise deals 的基本门槛。**EU AI Act**：已于 2024 年 8 月 1 日生效。大多数 high-risk requirements 从 2026 年 8 月 2 日开始执行。High-risk-system obligations（Art. 99(4)）罚款最高 €15M 或全球年营业额 3%；prohibited AI practices（Art. 99(3)）罚款最高 €35M 或 7%。只要服务 EU users，就具有全球适用性。**Colorado AI Act**：2026 年 6 月 30 日生效（由 SB25B-004 从 2026 年 2 月推迟）——high-risk systems 需要 impact assessments，并提供 appeal AI decisions 的权利。Virginia 对 credit/employment/housing/education 有类似规则。**SOC 2 Type II**：事实上的 B2B AI 要求（fintech 需要 Type II，不是 Type I）。**GDPR**：目前有记录的最大 AI-specific fine 是 Dutch DPA 于 2024 年 9 月对 Clearview AI 的 €30.5M；Italy's Garante 在 2024 年 12 月对 OpenAI 罚款 €15M（随后在 2026 年 3 月 appeal 中被推翻）。Inference 时的 real-time PII redaction 是可辩护标准；post-processing cleanup 不够。**HIPAA**：医疗场景有约束——没有 BAA，不能把 PHI 发送给外部 AI services。**PCI-DSS**：AI-interaction-layer coverage 需要 configuration + contractual agreements，不会自动覆盖。**ISO 42001**：新兴 AI governance standard，正和 ISO 27001 一起成为越来越常见的 procurement requirement。参考 profile：OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment components 的 PCI-DSS。Cross-framework mapping 会降低 audit fatigue：access controls 可以映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)。

**Type:** Learn
**Languages:** （Python optional——compliance 是 policy + process，不是 code）
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举与 LLM products 相关的七个 2026 frameworks，并把每个 framework 匹配到对应 customer segment。
- 引用 EU AI Act enforcement timeline（2024 年 8 月生效；2026 年 8 月开始 high-risk enforcement）和两级 fine ceiling（high-risk obligations 为 €15M / 3%，prohibited practices 为 €35M / 7%）。
- 解释为什么 post-processing PII cleanup 对 GDPR 来说不够，并说出 defensible standard 是 real-time inference-layer redaction。
- 描述 cross-framework control mapping（例如 access control 映射到 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)）。

## 问题

一个 enterprise customer 的 procurement 要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001，以及 “EU AI Act compliance statement”。你的团队只有 SOC 2 Type I。距离 Type II 还差六个月，而且还没有开始 GDPR Article 30 records。

Multi-framework coverage 不是 LLM 问题——它是 enterprise-SaaS 问题，只是叠加了 LLM-specific overlays。2026 年的 procurement teams 想要的是一个 matrix：每个 framework 一行，每个 control 一列，而不是一个 PDF。

## 概念

### 七个 frameworks

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls 需要经过 6-12 个月的运行审计。 |
| HIPAA | US healthcare | 需要 BAA；没有签署 agreement，PHI 不能离开 infrastructure。 |
| GDPR | EU users | Real-time PII redaction；data subject rights；Article 30 records。 |
| PCI-DSS | Payment data | 接触 payment 的 AI 需要 configuration + contracts。 |
| EU AI Act | Serving EU users | Risk tier classification；high-risk systems 需要 conformity assessment、documentation、logging。 |
| Colorado AI Act | Serving CO residents | Impact assessments；right to appeal。 |
| ISO 42001 | AI governance | Emerging；与 ISO 27001 配套。 |

### EU AI Act timeline

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：prohibited-AI practices 开始执行。
- 2026 年 8 月 2 日：high-risk systems 开始执行（conformity assessment、documentation、logging）。
- 2027 年 8 月：harmonized legislation 下产品中的 high-risk systems。

Risk tiers：Unacceptable（banned）、High-risk（conformity + logging）、Limited-risk（transparency）、Minimal-risk（no constraint）。大多数 B2B LLM SaaS 属于 limited-risk；employment、credit、education、law enforcement、migration、essential services 会触发 high-risk。

Fines（Article 99）：违反 high-risk-system obligations（Art. 99(4)）最高 €15M 或全球年营业额 3%；prohibited AI practices（Art. 99(3)）最高 €35M 或 7%；适用较高者。

### GDPR：real-time redaction 是标准

Post-processing cleanup（LLM 看完之后再 redact PII）不是可辩护姿态——model 已经看到了数据。2026 年的标准是 real-time inference-layer redaction：

- LLM call 之前做 entity recognition。
- Consistent tokenization（Mesh approach）保留 semantics。
- 只存储 redacted prompts + 已 consented opt-in 的 raw。

近期执法：Dutch DPA 在 2024 年 9 月对 Clearview AI 的 €30.5M 罚款，是目前有记录的最大 AI-specific GDPR fine；Italy's Garante 在 2024 年 12 月对 OpenAI 的 €15M 罚款，是最大的 LLM-specific fine，尽管它在 2026 年 3 月 appeal 中被推翻，并且 ruling 仍处在 further review 中。Post-processing claims 在 audit 中站不住脚。

### HIPAA：BAA 不是可选项

没有签署 Business Associate Agreement，就不能把 PHI 发送给外部 AI services。三大 hyperscaler LLM platforms（Bedrock、Azure OpenAI、Vertex）都提供 BAAs。OpenAI direct API 提供 BAA。Anthropic direct API 提供 BAA。发送 PHI 前必须确认。

### SOC 2 Type II

Type I：controls 已设计并记录。

Type II：controls 在 6-12 个月内有效运行。

2026 年，B2B procurement 默认要求 Type II。Type I 是起点；Type II 才是 gate。

常见 audit drivers：access logs（谁看了什么）、change management（如何部署）、risk assessments（季度）、incident response（是否测试过）。Phase 17 · 25 的 audit log 可以直接复用。

### Cross-framework mapping

一个 access control policy 可以满足多个 framework controls：

| Control | Frameworks |
|---------|------------|
| Access logging | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| Encryption in transit | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Compliance tools（Drata、Vanta、Secureframe）会自动化这种 mapping。规模上来后很值。

### ISO 42001：新兴标准

2023 年末发布。正在和 ISO 27001 一起成为越来越常见的 procurement requirement。它是 AI governance framework，覆盖 risk management、data quality、transparency、human oversight。

### OpenAI 的 reference profile

OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment components 的 PCI-DSS。这大致就是 2026 年 enterprise table stakes。

### 你应该记住的数字

- EU AI Act fines：high-risk obligations（Art. 99(4)）最高 €15M / 3%；prohibited practices（Art. 99(3)）最高 €35M / 7%。
- EU AI Act high-risk enforcement：2026 年 8 月 2 日。
- 最大有记录的 AI-specific GDPR fine：€30.5M，Clearview AI（Dutch DPA，2024 年 9 月）。
- 最大 LLM-specific GDPR fine：€15M，OpenAI（Italy's Garante，2024 年 12 月；2026 年 3 月 appeal 中被推翻）。
- SOC 2 Type II window：controls 运行 6-12 个月。
- Colorado AI Act effective date：2026 年 6 月 30 日（由 SB25B-004 从 2026 年 2 月推迟）。

## Use It

`code/main.py` 是一个用 Python 写的 compliance-mapping spreadsheet——给定一个 control，它会列出该 control 满足哪些 frameworks。

## Ship It

本课产出 `outputs/skill-compliance-matrix.md`。给定 customer segment 和 geography，它会指定 required frameworks 和 controls。

## 练习

1. 你的第一个 enterprise customer 要求 SOC 2 Type II、HIPAA BAA、EU AI Act statement。为了赢下这单，minimum viable compliance posture 是什么？
2. 把三个假设 LLM products 按 EU AI Act risk tiers 分类。进入 high-risk 后会发生什么变化？
3. 你不小心把 PHI 发送给了没有 BAA 的 provider。走一遍 incident response。
4. 论证 ISO 42001 对 mid-market AI vendor 来说，在 2026 年是否 “necessary”。
5. 把你的 LLM audit log fields（Phase 17 · 25）映射到至少三个 framework controls。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| SOC 2 Type II | “audited controls” | Controls 在 6-12 个月内运行，并经过独立证明。 |
| HIPAA BAA | “healthcare contract” | Business Associate Agreement；处理 PHI 时必需。 |
| GDPR | “EU privacy” | Real-time PII redaction 是 2026 年可辩护标准。 |
| EU AI Act | “EU AI rules” | 2026 年 8 月 high-risk enforcement；high-risk obligations 为 €15M / 3%，prohibited practices 为 €35M / 7%。 |
| Colorado AI Act | “US AI state law” | 2026 年 6 月 30 日生效（由 SB25B-004 推迟）；impact assessments。 |
| ISO 42001 | “AI governance” | 面向 AI risk + transparency 的新兴 framework。 |
| ISO 27001 | “security ISMS” | Information Security Management System baseline。 |
| Conformity assessment | “EU AI doc package” | High-risk requirement：docs、testing、logging。 |
| Cross-framework mapping | “one control, many frames” | 单个 policy 满足多个 framework controls。 |

## 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard。
