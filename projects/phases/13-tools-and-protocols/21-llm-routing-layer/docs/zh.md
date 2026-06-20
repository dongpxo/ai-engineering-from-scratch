# LLM Routing Layer：LiteLLM、OpenRouter、Portkey

> Provider lock-in 很昂贵。不同 tool-calling workloads 适合不同 models。Routing gateways 提供统一 API surface、retries、failover、cost tracking 和 guardrails。2026 年有三种 archetypes 占主导：LiteLLM（open-source self-hosted）、OpenRouter（managed SaaS）、Portkey（production-grade，2026 年 3 月 open-sourced）。本课会命名 decision criteria，并走读一个 stdlib routing gateway。

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## 学习目标

- 区分 self-hosted、managed 和 production-grade routing options。
- 实现一条 fallback chain，在 provider failures 时按定义好的 priority order retry。
- 跨 providers 跟踪 per-request cost 和 token usage。
- 根据给定 production constraint，在 LiteLLM、OpenRouter 和 Portkey 之间做选择。

## 问题

Provider routing 会在这些场景里变得重要：

1. **Cost。** Claude Sonnet 的成本是 Haiku 的 3 倍。对 triage task 来说，Haiku 已经足够；对 synthesis task 来说，Sonnet 值得。按 request route。

2. **Failover。** OpenAI 状态糟糕一小时。每个 request 都失败。你希望自动 fallback 到 Anthropic，而不是重新部署。

3. **Latency。** Live chat UI 需要快速 time-to-first-token。Batch summarizer 不需要。按 latency SLA route。

4. **Compliance。** EU users 必须留在 EU regions。按 region route。

5. **Experimentation。** 在同一 workload 上 A/B 两个 models。按 test bucket route。

如果给每个 integration 手写这些逻辑，会非常重复。Routing gateway 提供一个 OpenAI-compatible API，并处理其余部分。

## 概念

### OpenAI-compatible proxy shape

大家都讲 OpenAI-shape。Routing gateway 暴露 `/v1/chat/completions`，接收 OpenAI schema，并在内部 proxy 到 Anthropic / Gemini / Cohere / Ollama / 任意 backend。Client 不需要关心。

### Model aliases

你的代码不写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。Gateway 把 aliases 映射到真实 models。Anthropic 发布 Claude 4 时，你在 server-side 改 alias；代码完全不用动。

### Fallback chains

```text
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateways 会在 config 中定义它。Retries 会计入 budget，避免 fallback cascades 把成本炸开。

### Semantic caching

完全相同或近似相同的 prompts 命中 cache，而不是再次打 provider。重复 agent loops 上的节省可以达到 30% 到 60%。Keys 基于 embedding；near-identical prompts 共享同一个 cache slot。

### Guardrails

Gateway-level：

- **PII redaction。** 发送 prompts 前执行 regex 或 ML-based pass。
- **Policy violations。** 拒绝包含 prohibited content 的 prompts。
- **Output filters。** 清理 completions 中的 leaks。

Portkey 和 Kong 都提供 opinionated guardrails。LiteLLM 把它们作为 optional。

### Per-key rate limits

一个 API key = 一个 team。Per-key budgets 防止某个 team 吃掉 shared quota。多数 gateways 支持这一点。

### Self-hosted vs managed trade-offs

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Managed SaaS | Open source (Mar 2026) + managed |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your own keys | OpenRouter credits | Your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Teams that want full control | Rapid prototyping | Production with compliance |

当你有 SRE team 并希望 data sovereignty 时，LiteLLM 胜出。当你想要 single subscription 且不想管 infra 时，OpenRouter 胜出。当你需要开箱即用的 guardrails 和 compliance 时，Portkey 胜出。

### Cost tracking

每个 request 都携带 `provider`、`model`、`input_tokens`、`output_tokens`。用 per-model per-token prices 相乘（从 gateway 维护的 pricing sheet 拉取）。按 per-user / per-team / per-project 聚合。

### MCP plus routing

Gateway 可以同时 route LLM calls 和 MCP sampling requests。当 sampling request 的 modelPreferences 偏好某个 model 时，gateway 会把它翻译到正确 backend。这也是 Phase 13 · 17（MCP gateway）和本课 routing gateway 有时会合并成同一个 service 的地方。

### Routing strategies

- **Static priority。** 列表中第一个；error 时 fallback。
- **Load balancing。** Round-robin 或 weighted。
- **Cost-aware。** 选择满足 latency / quality 的最便宜 model。
- **Latency-aware。** 选择最近 N 分钟内最快的 model。
- **Task-aware。** Prompt classifier 把 coding route 到一个 model，把 summarization route 到另一个 model。

## Use It

`code/main.py` 用约 150 行实现了一个 routing gateway：接收 OpenAI-shaped requests，翻译到 per-provider stubs，运行 priority fallback chain，跟踪 per-request cost，并对 inputs 应用 PII redaction pass。运行三个 scenarios：normal request、primary-provider outage triggering fallback、PII leakage caught by redaction。

观察重点：

- `ROUTES` dict：alias -> 按 priority 排序的 concrete providers list。
- Fallback loop 会在 5xx 上 retry。
- Cost tracker 用 token usage 乘以 per-model rates。
- PII redactor 会在 forwarding 前 scrub SSN-shaped patterns。

## Ship It

本课产出 `outputs/skill-routing-config-designer.md`。给定 workload profile（latency、cost、compliance），这个 skill 会选择 LiteLLM / OpenRouter / Portkey，并生成 routing config。

## 练习

1. 运行 `code/main.py`。触发 outage scenario；确认 fallback 落到第二个 provider，并且 cost attribution 正确。

2. 添加 semantic caching：prompt 的 SHA256 作为 lookup key；cache hits 立即返回。测量 repeated call 上节省的 cost。

3. 添加 prompt classifier，把 “code ...” prompts route 到偏 intelligence 的 alias，把 “summarize ...” prompts route 到偏 speed 的 alias。

4. 设计 per-team budgets：每个 team 有 monthly spend cap；gateway 在 cap 命中后拒绝 requests。选择 enforcement granularity（per-request 或 windowed）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey docs。说出每个产品提供、另外两个没有的一个 feature。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routing gateway | “LLM proxy” | 位于 many providers 前方的一层 one-API-surface |
| OpenAI-compatible | “Speaks the OpenAI schema” | 接收 `/v1/chat/completions` shape，并翻译到任意 backend |
| Model alias | “our_smart_model” | 代码里的名称，由 gateway 映射到具体 model |
| Fallback chain | “Retry list” | 失败时按顺序尝试的 providers list |
| Semantic caching | “Prompt-embedding cache” | Key 是 prompt embedding；near-duplicates 共享 cache hit |
| Guardrails | “Input/output filters” | Redact PII，reject policy violations |
| Per-key rate limit | “Team budget” | 以 API key 为 scope 的 quota |
| Cost tracking | “Per-request spend” | 聚合 token usage x price per model |
| LiteLLM | “The open proxy” | 可 self-host 的 OSS routing gateway |
| OpenRouter | “The managed SaaS” | 使用 credit-based billing 的 hosted gateway |
| Portkey | “The production option” | Open-source + managed，并内置 guardrails |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — self-hosted routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — managed routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — production routing with guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — decision guide
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — vendor survey
