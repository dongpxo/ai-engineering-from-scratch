# LLM FinOps：Unit Economics 与 Multi-Tenant Attribution

> 传统 FinOps 在 LLM spend 上会失效。成本是 token-transactions，不是 resource-uptime。Tags 对不上——一次 API call 是 transaction，不是 asset。Engineering decisions（prompt design、context window、output length）就是 financial decisions。2026 年 playbook 要在第一天就 instrument 三个 attribution dimensions：per-user（`user_id`）用于 seat pricing 和 expansion；per-task（`task_id` + `route`）用于 product surface cost 和优先级；per-tenant（`tenant_id`）用于 unit economics 和 renewal。四个 token layers——prompt、tool、memory、response——放进一个 bucket 会隐藏 spend。Multi-tenant products 的 enforcement ladder：按 tenant 做 rate limits（预期峰值的 2-3x，清晰的 429 + retry-after）；daily spend cap（contracted ceiling 的 1.5-3x；触发 rate tightening + alert）；spend z-score > 4 时启动 kill switches（auto-pause + page on-call）。Attribution patterns：tag-and-aggregate、telemetry-joiner（trace-ID → billing；最高准确度）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。Unit metric 应该是 cost per resolved query、cost per generated artifact，而不是 $/M tokens。Retroactive tagging 总会漏；要在 request creation 时 instrument。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 minutes

## 学习目标

- 解释为什么传统 FinOps（tags + tiers）会在 LLM spend 上失效，并说出三个新的 attribution dimensions。
- 列举四个 token layers（prompt、tool、memory、response），并解释为什么 single-bucket billing 会隐藏成本。
- 为 multi-tenant product 设计 enforcement ladder（rate → spend cap → kill switch）。
- 选择 unit metric（cost per resolved query / artifact），而不是 $/M tokens。

## 问题

账单显示 $40,000。你不知道：
- 是哪个 tenant 花掉的。
- 是哪个 product feature 驱动的。
- 是否有 individual user 在滥用。
- 真正的问题是 prompt bloat、tool calls，还是 memory amplification。

Provider-side 的 tag-and-aggregate 适合 cloud resources（EC2、S3），因为 tags 会传播到 line items。LLM API calls 不会自动带 tag——你必须在 call site stamp user/task/tenant，并一路传下去。Retroactive attribution 总会漏掉 edge cases。

## 概念

### 三个 attribution dimensions

**Per-user**（`user_id`）：谁在花多少钱。驱动 seat pricing、expansion conversations，并识别 power users。

**Per-task**（`task_id` + `route`）：哪个 product surface 花多少钱。驱动 feature prioritization，以及是否 kill expensive features 的决策。

**Per-tenant**（`tenant_id`）：哪个 customer 是 profitable。驱动 unit economics、renewal pricing 和 tier thresholds。

第一天就在 call site instrument 三者。事后补永远更差。

### 四个 token layers

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40%（agent workloads） |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

把四层全塞进一个 bucket，会让 optimization 变成盲飞。要在 attribution schema 中拆出来。

### Enforcement ladder

1. **Rate limit** per tenant。预期峰值的 2-3x。返回带 `Retry-After` 的 429。Tenant 会感到 friction，但不会收到 surprise bill。

2. **Daily spend cap** per tenant。Contracted ceiling 的 1.5-3x。Trigger：tighten rate limit + alert customer-success。

3. **Kill switch**：当 spend z-score 相对 tenant baseline > 4 时触发。Auto-pause tenant；page on-call；升级给 ops + CS。

### Attribution patterns

- **Tag-and-aggregate**：stamp metadata headers；之后 aggregate。简单；粗糙。
- **Telemetry joiner**：通过 trace IDs 把 traces join 到 billing。准确度最高。成熟团队会这么做。
- **Sampling + extrapolation**：采样 5-10%，再放大。适合粗略 spend 的 cost-effective 方案；会漏掉 tails。
- **Model-based allocation**：用 regression 推断 cost driver。适合没有 tags 的 legacy data。
- **Event-sourced**：把 cost 作为 events 写入 stream（Kafka / Kinesis）。Real-time。
- **Real-time streaming**：dashboard sub-second 更新。

### Cost per X 才是 unit metric

$/M tokens 是 vendor speak。Product metrics：

- Cost per resolved support ticket。
- Cost per generated article。
- Cost per successful agent task。
- Cost per user-session-minute。

把 cost 绑定到 product outcome。否则 optimization 没有锚点。

### Cost attribution trace shape

```text
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次调用都 emit。存入 data lake。按 dimension aggregate。Phase 17 · 13 observability stack 就是它的归宿。

### Compounded-savings stack

Stack：cache + batch + route + gateway。四者全部启用时：
- Cache L2（Phase 17 · 14）：input 约便宜 10x。
- Batch（Phase 17 · 15）：50% off。
- Route to cheap model（Phase 17 · 16）：60% cost reduction。
- Gateway efficiency（Phase 17 · 19）：redundancy + retries。

Best-case stacked：约为 naive baseline 的 5-10%。大多数团队启用了 2-3 个 levers；很少把四个都叠起来。

### 你应该记住的数字

- Attribution dimensions：per-user、per-task、per-tenant。
- Four token layers：prompt、tool、memory、response。
- Kill switch：spend z-score > 4。
- Unit metric：cost per resolved query，不是 $/M tokens。
- Stacked optimizations：可能降到 baseline 的约 5-10%。

## Use It

`code/main.py` 会模拟一个 multi-tenant LLM service，包含三层 enforcement ladder。它会注入一个 abusive tenant，并展示 kill switch 被触发。

## Ship It

本课产出 `outputs/skill-finops-plan.md`。给定 product 和 scale，它会设计 attribution schema 与 enforcement ladder。

## 练习

1. 运行 `code/main.py`。Kill switch 在什么 z-score 触发？你如何选择 threshold？
2. 设计一个 per-tenant、per-task cost dashboard。最先构建的 5 个 views 是什么？
3. 你的最大 tenant 在 unit economics 上为负。按 customer impact 从低到高，提出三个 interventions。
4. 为一个 support product 计算 cost per resolved ticket：3M tokens/ticket，约 800 tickets/day，GPT-5 cached rate。
5. 论证 retroactive tagging 是否可能有效。什么时候可以接受？

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Per-user attribution | “user-level cost” | 每次调用都 stamp `user_id` |
| Per-task attribution | “feature cost” | `task_id` + `route` 标识 product surface |
| Per-tenant attribution | “customer cost” | `tenant_id`；驱动 unit economics |
| Four token layers | “cost layers” | prompt + tool + memory + response |
| Rate limit | “429 guard” | 在 gateway enforce 的 per-tenant ceiling |
| Daily spend cap | “daily ceiling” | Tenant-scoped budget with alert |
| Kill switch | “auto-pause” | Spend z-score > 4 触发 auto-suspension |
| Cost per resolved | “product unit metric” | Cost 绑定到 product outcome，而不是 tokens |
| Telemetry joiner | “trace-to-billing” | 准确度最高的 attribution pattern |
| Stacked optimization | “cache+batch+route+gateway” | 叠加 savings 到 baseline 的约 5-10% |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
