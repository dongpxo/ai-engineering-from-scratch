# AI 网关：LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway 位于你的应用和模型 providers 之间。核心能力包括 provider routing、fallback、retries、rate limiting、secret references、observability、guardrails。2026 年的市场格局：**LiteLLM** 是 MIT OSS，支持 100+ providers，兼容 OpenAI API，但公开 benchmarks 中在约 2000 RPS 附近崩溃（8 GB memory、级联失败）；最适合 Python、<500 RPS、dev/prototyping。**Portkey** 定位为 control plane（guardrails、PII redaction、jailbreak detection、audit trails），2026 年 3 月以 Apache 2.0 开源，latency overhead 为 20-40 ms，production tier 为 $49/mo。**Kong AI Gateway** 构建在 Kong Gateway 之上；Kong 自家同 12 CPUs benchmark 显示：比 Portkey 快 228%，比 LiteLLM 快 859%；价格为 $100/model/month（Plus tier 最多 5 个）；如果你已经在用 Kong，它很适合 enterprise。**Bifrost**（Maxim AI）：自动 retries，支持 configurable backoff，OpenAI 429 时 fallback 到 Anthropic。**Cloudflare / Vercel AI Gateways**：managed、zero-ops、basic retry。Data residency 会推动 self-host 决策；Portkey 和 Kong 处在 OSS + optional managed 的中间地带。

**Type:** Learn
**Languages:** Python（stdlib，toy gateway-routing simulator）
**Prerequisites:** Phase 17 · 01（Managed LLM Platforms），Phase 17 · 16（Model Routing）
**Time:** ~60 minutes

## 学习目标

- 枚举 gateway 的核心能力：routing、fallback、retries、rate limits、secrets、observability、guardrails。
- 将四个 2026 gateways（LiteLLM、Portkey、Kong AI、Bifrost）映射到 scale ceilings 和 use cases。
- 引用 Kong benchmark（相对 Portkey 快 228%，相对 LiteLLM 快 859%），并解释它为何影响 >500 RPS 的选择。
- 在给定 data residency 和 ops budget 时，选择 self-hosted 或 managed。

## 问题

你的产品会调用 OpenAI、Anthropic 和一个 self-hosted Llama。每个 provider 都有不同 SDK、error model、rate limit 和 auth scheme。你想要 failover（如果 OpenAI 返回 429，就尝试 Anthropic）、单一 credential store、统一 observability，以及 per-tenant rate limits。

如果在 app layer 重新实现这些能力，每个 service 都会和每个 provider 绑定。Gateway layer 会把它们收敛到一个 process 和一个 API（通常兼容 OpenAI），再 fan out 到各个 providers。

## 概念

### 核心能力

1. **Provider routing**：把 OpenAI、Anthropic、Gemini、self-hosted 等藏在一个 API 后面。
2. **Fallback**：遇到 429、5xx 或 quality failure 时，切到别处重试。
3. **Retries**：exponential backoff，bounded attempts。
4. **Rate limits**：per-tenant、per-key、per-model。
5. **Secret references**：运行时从 vault 拉 credentials（永远不要放在 app 里）。
6. **Observability**：OTel + GenAI attributes（Phase 17 · 13）+ cost attribution。
7. **Guardrails**：PII redaction、jailbreak detection、allowed-topics filters。

### LiteLLM：MIT OSS，Python

- 100+ providers，兼容 OpenAI API，支持 router config、fallback、basic observability。
- 在 Kong benchmark 中约 2000 RPS 附近崩溃；8 GB memory footprint，sustained load 下出现 cascading failures。
- 最适合：Python app、<500 RPS、dev/staging gateways、experimental routing。
- 成本：OSS 为 $0；cloud 有 free tier。

### Portkey：control plane 定位

- 2026 年 3 月起以 Apache 2.0 OSS 开源。提供 guardrails、PII redaction、jailbreak detection、audit trails。
- 每个 request 增加 20-40 ms latency overhead。
- Production tier 为 $49/mo，包含 retention + SLA。
- 最适合：需要 bundled guardrails + observability 的 regulated industries。

### Kong AI Gateway：规模路线

- 构建在 Kong Gateway 上（成熟的 API gateway product，lua + OpenResty）。
- Kong 自家 12-CPU equivalent benchmark：相对 Portkey 快 228%，相对 LiteLLM 快 859%。
- 价格：$100/model/month，Plus tier 最多 5 个。
- 最适合：已经在用 Kong；>1000 RPS；愿意 license。

### Bifrost（Maxim AI）

- 自动 retries，支持 configurable backoff。
- OpenAI 429 时 fallback 到 Anthropic 是典型 recipe。
- 较新的 entrant；commercial。

### Cloudflare AI Gateway / Vercel AI Gateway

- Managed、zero-ops。提供 basic retry 和 observability。
- 最适合：运行在 Cloudflare/Vercel 上的 edge-serving JavaScript apps。
- 在 guardrails 和 rate limits 上比 Kong/Portkey 有限。

### Self-hosted vs managed

Data residency 是 forcing function。Healthcare 和 finance 默认 self-host（LiteLLM、Portkey OSS 或 Kong）。Consumer products 默认 managed（Cloudflare AI Gateway）或 middle-tier（Portkey managed）。Hybrid：regulated tenant 走 self-hosted，其余走 managed。

### Latency budget

- LiteLLM：typical overhead 为 5-15 ms。
- Portkey：overhead 为 20-40 ms。
- Kong：overhead 为 3-8 ms。
- Cloudflare/Vercel：overhead 为 1-3 ms（edge advantage）。

Gateway latency 会直接加到 TTFT 上。如果 TTFT P99 < 100 ms SLA，选 Kong 或 Cloudflare。若 P99 < 500 ms，任何方案都可以。

### Rate-limit semantics matter

Simple token-bucket 在中等规模之前都有效。Multi-tenant 需要 sliding-window + burst allowance + per-tenant tiering。LiteLLM 提供 token-bucket；Kong 提供 sliding-window；Portkey 提供 tiered。

### Gateway + observability + routing 会组合在一起

Phase 17 · 13（observability）+ 16（model routing）+ 19（gateways）在生产里通常是同一层。要么选择一个工具覆盖三者，要么仔细接线：多数 2026 deployments 会把 Helicone（observability）或 Portkey（guardrails）和 Kong（scale）组合起来，形成分工。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 崩溃，8 GB memory。
- Portkey：20-40 ms overhead；2026 年 3 月起 Apache 2.0。
- Kong：相对 Portkey 快 228%，相对 LiteLLM 快 859%。
- Kong pricing：$100/model/month，Plus tier 最多 5 个。
- Cloudflare/Vercel：edge 上 1-3 ms overhead。

## Use It

`code/main.py` 会模拟 3 个 providers 之间的 gateway routing，并在注入 429/5xx 时执行 fallback。它会报告 latency、retry rate 和 fallback hit rate。

## Ship It

本课产出 `outputs/skill-gateway-picker.md`。给定 scale、ops posture、compliance 和 latency budget，它会选择一个 gateway。

## 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→self-hosted fallback。在 provider error rate 为 5% 时，预期 hit rate 是多少？
2. 你的 SLA 是 300 ms baseline 上的 TTFT P99 < 200 ms。哪些 gateways 仍在 budget 内？
3. 一个 healthcare customer 要求 self-hosted + PII redaction + audit。选择 Portkey OSS 还是 Kong？
4. 比较 LiteLLM 和 Kong：团队应该在什么 RPS ceiling 迁移？
5. 为 multi-tenant SaaS 设计 rate-limit policy：free tier、trial tier、paid tier。用 token-bucket 还是 sliding-window？

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Gateway | “API broker” | 位于 apps 和 providers 之间的 process。 |
| LiteLLM | “the MIT one” | Python OSS，100+ providers，在 2K RPS 附近崩溃。 |
| Portkey | “guardrails gateway” | Control plane + observability，Apache 2.0。 |
| Kong AI Gateway | “the scale one” | 构建在 Kong Gateway 上，benchmark leader。 |
| Bifrost | “Maxim's gateway” | Retries + Anthropic fallback recipe。 |
| Cloudflare AI Gateway | “edge managed” | Edge-deployed managed gateway，zero-ops。 |
| PII redaction | “data scrub” | 发送给 model 之前用 regex + NER mask。 |
| Jailbreak detection | “prompt injection guard” | 对 user input 跑 classifier。 |
| Audit trail | “regulated log” | 每一次 LLM call 的 immutable record。 |
| Token-bucket | “simple rate limit” | 基于 refill 的 rate limiter。 |
| Sliding-window | “precise rate limit” | Time-windowed rate limiter；fairness 更好。 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
