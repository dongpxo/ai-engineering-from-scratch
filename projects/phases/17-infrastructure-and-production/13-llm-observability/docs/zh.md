# LLM 可观测性栈选择

> 2026 年 observability 市场分成两类。Development platforms（LangSmith、Langfuse、Comet Opik）把 monitoring 和 evals、prompt management、session replays 打包在一起。Gateway/instrumentation tools（Helicone、SigNoz、OpenLLMetry、Phoenix）专注 telemetry。Langfuse 是 MIT-licensed core，OSS 平衡很好（cloud free tier 每月 50K events）。Phoenix 是 OpenTelemetry-native，采用 Elastic License 2.0；非常适合 drift/RAG visualization，但不是持久化 production backend。Arize AX 使用 zero-copy Iceberg/Parquet integration，声称比 monolithic observability 便宜 100x。LangSmith 在 LangChain/LangGraph 场景领先，$39/user/mo，只有 Enterprise 才能 self-host。Helicone 是 proxy-based，15-30 分钟即可接入，free tier 每月 100K req，但 agent traces 深度较弱。常见生产模式是：Gateway（Helicone/Portkey）+ eval platform（Phoenix/TruLens），用 OpenTelemetry 粘起来。

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 区分 development platforms（bundled: evals + prompts + sessions）和 gateway/telemetry tools（traces + metrics only）。
- 把六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到它们的 license、pricing 和 sweet-spot use cases。
- 解释 OpenTelemetry-glue pattern：它如何让你把 gateway tool 和单独的 eval platform 组合起来。
- 命名 2026 年的 cost differentiator（Arize AX 的 zero-copy approach vs monolithic ingest），并说出粗略的 100x multiplier。

## 问题

你已经发布了一个 LLM feature。它能工作。但你看不到 prompt failures、tool loops、latency regressions、cost spikes，也看不到 prompt-cache hit rate。你搜索 “LLM observability”，会看到八个工具都声称自己能解决同一个问题，而且价格点分成三档。

它们解决的并不是同一个问题。LangSmith 回答的是：“为什么这次 LangGraph run 失败了？”Phoenix 回答的是：“我的 RAG pipeline 是否在 drifting？”Helicone 回答的是：“哪个 app 正在烧 tokens？”Langfuse 回答的是：“我能不能把整套东西 self-host？”不同工具，面向不同受众。

选择时有四条轴：stack（LangChain？raw SDK？multi-vendor？）、license tolerance（只接受 MIT？Elastic 可以？commercial 也行？）、budget（free tier？$100/mo？$1000/mo？）和 self-host（必须？nice-to-have？绝不？）。

## 概念

### 两大类别

**Development platforms** 把 observability 和 evals、prompt management、dataset versioning、session replay 打包在一起。你运行 experiments，查看哪个 prompt 有效，把新 prompt 放到 dataset regression 里对比旧 winners。代表是 LangSmith、Langfuse、Comet Opik。

**Gateway/telemetry tools** instrument inference calls：prompt、response、tokens、latency、model、cost。代表是 Helicone、SigNoz、OpenLLMetry、Phoenix。它们更 minimalist，可以通过 OpenTelemetry 和单独的 eval tool 组合。

### Langfuse — OSS 平衡

- Core Apache / MIT licensed；通过 Docker self-host。
- Cloud free tier：50K events/month。Paid：team $29/mo。
- Evals、prompt management、traces、datasets。对四个 dev-platform features 都有合理覆盖。
- Sweet spot：你想要 LangSmith-class features，但必须 self-host 或保持 OSS license。

### Phoenix (Arize) — telemetry-first, OpenTelemetry-native

- Elastic License 2.0；self-host 很容易。
- 擅长 RAG 和 drift visualization。Embedding-space scatter plots 是 first-class 功能。
- 不是为持久化 production backend 设计的；主要是 development-time observability。
- Sweet spot：RAG pipeline development、drift debugging，并与单独的 production gateway 搭配。

### Arize AX — scale play

- Commercial。通过 Iceberg/Parquet 做 zero-copy data lake integration。
- 声称在 scale 下比 monolithic observability（Datadog-class）便宜约 100x。数学是：你把 traces 存在自己的 S3 Parquet 中；Arize 直接读取。
- Sweet spot：>10M traces/day，已有 data lake，希望获得 LLM-specific dashboards，但不想付 Datadog pricing。

### LangSmith — LangChain/LangGraph first

- Commercial，$39/user/month。Self-host 仅限 Enterprise。
- 对 LangChain 和 LangGraph stacks 是 best-in-class。如果你不在二者之一上，它的吸引力会下降。
- Sweet spot：团队已经 committed to LangChain，并且愿意付费。

### Helicone — proxy-based minimum viable

- 通过把 `OPENAI_API_BASE` 切到 Helicone proxy，15-30 分钟即可 setup。
- MIT licensed；100K req/mo free，paid $20/mo+。
- 包含 failover、caching、rate limits；也可以作为 gateway。
- 对 agent / multi-step traces 的深度较弱。
- Sweet spot：quick start、single-stack app，需要 gateway + observability in one。

### Opik (Comet) — OSS dev platform

- Apache 2.0，fully OSS。
- Feature set 类似 Langfuse，并继承 Comet 生态。
- Sweet spot：ML teams 已经使用 Comet，希望在同一个 pane 中做 LLM observability。

### SigNoz — OpenTelemetry-first full APM

- Apache 2.0。通过 OpenTelemetry 同时处理 general APM 和 LLM。
- Sweet spot：统一观察 services 和 LLM calls。

### Glue：OpenTelemetry + GenAI semantic conventions

OpenTelemetry 在 2025 年末发布了 GenAI semantic conventions（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在形成的生产模式：

1. 从每个 LLM call 发出带 GenAI conventions 的 OTel。
2. Route 到 gateway（Helicone / Portkey）处理 day-to-day。
3. Dual-ship 到 eval platform（Phoenix / Langfuse）处理 regressions。
4. 归档到 data lake（Iceberg），用于通过 Arize AX 或 DuckDB 做长期分析。

### 陷阱：instrumenting at the wrong layer

在 agent framework 内部做 instrumentation（例如添加 LangSmith traces）会把你耦合到该 framework。在 HTTP/OpenAI-SDK 层做 instrumentation（通过 OpenLLMetry 或你的 gateway）更可移植。

### Sampling — 你不能保留一切

当请求量超过 >1M requests/day，保留 full-trace 的成本会高于 LLM calls 本身。按规则采样：100% errors、100% high-cost、5% success。Aggregates 始终保留；raw traces 只保留 long tail。

### 你应该记住的数字

- Langfuse free cloud：50K events/month。
- LangSmith：$39/user/month。
- Helicone free：100K req/month。
- Arize AX claim：scale 下比 monolithic 便宜约 100x。
- OpenTelemetry GenAI conventions：2025 年发布，2026 年广泛采用。

## Use It

`code/main.py` 会模拟一天 1M traces，在多种 retention strategies（100% ingest、sampling、sampling + errors）之间比较，报告 storage cost 以及每种策略会丢失什么。

## Ship It

本课产出 `outputs/skill-observability-stack.md`。给定 stack、scale、budget、license posture，它会选择合适的 tool(s)。

## 练习

1. 你的团队使用 LangChain，并希望 OSS self-hosted observability。选择 Langfuse 或 Opik，并说明理由。
2. 当每天 5M traces 且 Datadog quote 为 $150K/month 时，计算 Arize AX 的 break-even。
3. 设计一组 OpenTelemetry GenAI attributes，作为你组织的 guideline，要求每个 LLM call 都必须携带。
4. 论证 Phoenix alone 是否足够用于 production。什么时候它不够？
5. Helicone 有 20ms proxy overhead。当 P99 TTFT 为 300 ms 时，这是否可接受？如果 SLA 是 100 ms 呢？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | “OTel for LLMs” | 面向 LLMs 的 open-source OpenTelemetry instrumentation |
| GenAI conventions | “OTel attributes” | LLM calls 的标准 OTel attribute names |
| LangSmith | “LangChain observability” | 与 LangChain ecosystem 打包在一起的 commercial platform |
| Langfuse | “OSS LangSmith” | MIT OSS，feature set 类似 LangSmith |
| Phoenix | “Arize dev tool” | OpenTelemetry-native dev/eval platform |
| Arize AX | “scale observability” | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | “proxy observability” | 收集 LLM telemetry 并提供 gateway features 的 HTTP proxy |
| Opik | “Comet LLM” | Comet 出品的 Apache 2.0 OSS dev platform |
| Session replay | “trace rerun” | 重放包含 tool calls 的完整 agent session |
| Eval | “offline test” | 在 labeled dataset 上运行 candidate model/prompt |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
