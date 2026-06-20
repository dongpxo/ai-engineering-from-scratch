# Agent Observability：Langfuse、Phoenix、Opik

> 2026 年有三个开源 agent observability platforms 占据主导。Langfuse（MIT）：每月 600 万+ installs，tracing + prompt management + evals + session replay。Arize Phoenix（Elastic 2.0）：深入的 agent-specific evals、RAG relevancy、OpenInference auto-instrumentation。Comet Opik（Apache 2.0）：自动 prompt optimization、guardrails、LLM-judge hallucination detection。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## 学习目标

- 命名三个顶级开源 agent observability platforms，并说明它们的 licenses。
- 区分每个平台最擅长的方向：Langfuse（prompt mgmt + sessions）、Phoenix（RAG + auto-instrumentation）、Opik（optimization + guardrails）。
- 解释为什么到 2026 年，89% 的组织都报告已经部署 agent observability。
- 实现一个 stdlib trace-to-dashboard pipeline，并带 LLM-judge evaluation。

## 问题

OTel GenAI（Lesson 23）给了你 schema。你仍然需要一个平台来 ingest spans、运行 evaluations、存储 prompt versions，并暴露 regressions。三个主要候选者分别强调 lifecycle 的不同部分。

## 概念

### Langfuse（MIT）

- 每月 600 万+ SDK installs，19k+ GitHub stars。
- 功能：tracing、带 versioning + playground 的 prompt management、evaluations（LLM-as-judge、user feedback、custom）、session replays。
- 2025 年 6 月：原先商业化的模块（LLM-as-a-judge、annotation queues、prompt experiments、Playground）以 MIT 协议开源。
- 最强项：带紧密 prompt-management loop 的 end-to-end observability。

### Arize Phoenix（Elastic License 2.0）

- 更深入的 agent-specific evaluation：trace clustering、anomaly detection、面向 RAG 的 retrieval relevancy。
- 原生 OpenInference auto-instrumentation。
- 与 managed Arize AX 配合用于 production。
- 没有 prompt versioning：定位是与更广平台配合使用的 drift/behavioral-regression tool。
- 最强项：RAG relevancy、behavioral drift、anomaly detection。

### Comet Opik（Apache 2.0）

- 通过 A/B experiments 做 automated prompt optimization。
- Guardrails（PII redaction、topical constraints）。
- LLM-judge hallucination detection。
- 来自 Comet 自己测量的 benchmark：Opik logs + evals 用时 23.44s，Langfuse 用时 327.15s（约 14x 差距）——vendor benchmarks 只能当作方向性信号。
- 最强项：optimization loop、automated experimentation、guardrail enforcement。

### 行业数据

根据 Maxim（2026 field analysis）：89% 的组织已经部署 agent observability；质量问题是最大的生产障碍（32% 的受访者提到它）。

### 如何选择

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open licensing, no ELv2 | Langfuse (MIT) or Opik (Apache 2.0) |
| Datadog / New Relic integration | Any — they all export OTel |

### 这个模式会在哪里出错

- **No eval strategy.** 只有 tracing、没有 evaluation，只是昂贵的 logging。
- **Self-rolled LLM-judge without grounding.** CRITIC pattern（Lesson 05）适用：judges 需要外部工具来做事实验证。
- **Prompt versions not tied to traces.** 当 production regress 时，你无法 bisect 到导致问题的 prompt。

## Build It

`code/main.py` 实现了一个 stdlib trace collector + LLM-judge evaluator：

- Ingest GenAI-shaped spans。
- 按 session 分组，标记 failed runs（guardrail trips、low-confidence evals）。
- 一个脚本化 LLM-judge，根据 rubric 给 agent responses 打分。
- 一个 dashboard-like summary：failure rate、top failure reasons、eval score distribution。

运行：

```bash
python3 code/main.py
```

输出：per-session eval scores 和 failure categorization，形状与 Langfuse/Phoenix/Opik 会展示的内容一致。

## Use It

- **Langfuse**：self-hosted 或 cloud；通过 OTel 或它们的 SDK 接入。
- **Arize Phoenix**：self-hosted；auto-instrument OpenInference。
- **Comet Opik**：self-hosted 或 cloud；automated optimization loop。
- **Datadog LLM Observability**：适合已经运行 Datadog 的 mixed ops+ML teams。

## Ship It

`outputs/skill-obs-platform-wiring.md` 会选择一个平台，并把 traces + evals + prompt versions 接入现有 agent。

## 练习

1. 把一周的 OTel traces 导出到 Langfuse cloud（free tier）。哪些 sessions 失败了？为什么？
2. 为你的领域写一个 LLM-judge rubric（factual correctness、tone、scope adherence）。在 50 条 traces 上测试。
3. 比较 Langfuse prompt versioning 与 Phoenix trace clustering。哪一个能更快告诉你哪里坏了？
4. 阅读 Opik guardrail docs。给你的一次 agent run 接入 PII redaction guardrail。
5. 在你自己的 corpus 上 benchmark 三个平台。忽略 vendors 发布的数据；测量你自己的。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tracing | "Spans collector" | Ingest OTel / SDK spans；按 session 建索引 |
| Prompt management | "Prompt CMS" | 与 traces 绑定的 versioned prompts |
| LLM-as-judge | "Automated eval" | 独立 LLM 根据 rubric 给 agent output 打分 |
| Session replay | "Trace playback" | 逐步回放过去 runs，用于 debugging |
| RAG relevancy | "Retrieval quality" | Retrieved context 是否匹配 query |
| Trace clustering | "Behavioral grouping" | 聚类相似 runs，用于 drift detection |
| Guardrail enforcement | "Policy at log time" | 对 logged content 做 PII/toxicity/scope checks |

## 延伸阅读

- [Langfuse docs](https://langfuse.com/)：tracing、evals、prompt mgmt。
- [Arize Phoenix docs](https://docs.arize.com/phoenix)：auto-instrumentation、drift。
- [Comet Opik](https://www.comet.com/site/products/opik/)：optimization + guardrails。
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：三者都会消费的 schema。
