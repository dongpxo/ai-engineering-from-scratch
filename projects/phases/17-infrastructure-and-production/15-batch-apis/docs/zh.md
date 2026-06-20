# Batch APIs：50% 折扣成为行业标准

> 每个主要 provider 都提供 async batch API，通常带 50% discount 和约 24-hour turnaround。OpenAI、Anthropic、Google，以及大多数 inference platforms（Fireworks batch tier、Together batch）都实现了同一种模式。把 batch 和 prompt caching 叠加起来，overnight pipelines 的成本会降到 synchronous-uncached cost 的约 10%。规则非常简单：只要不是 interactive，就应该放到 batch。Content generation pipelines、document classification、data extraction、report generation、bulk labeling、catalog tagging——任何能容忍 24 小时 latency 的 workload，在迁移到 batch 前都在浪费钱。2026 年的生产模式是把每个新的 LLM workload triage 到三条 lane：interactive（synchronous with caching）、semi-interactive（async queue with fallback）、batch（overnight, cached input stacked）。那些伪装成 interactive、但其实能容忍分钟级 latency 的 workloads 浪费最多。

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## 学习目标

- 命名三个 provider batch APIs（OpenAI、Anthropic、Google），并说出共同的 50% discount + 24h turnaround guarantees。
- 计算 overnight classification workload 中 batch + cached-input 叠加后的成本，并与 synchronous-uncached baseline 比较。
- 把 workload triage 到 interactive / semi-interactive / batch，并说明选择理由。
- 命名两个陷阱：partial interactivity（用户期望快于 24h）和 output-schema drift（不同 provider 的 batch file format 不同）。

## 问题

你的团队上线了一个 nightly report generation pipeline。50,000 份 documents，每份都要 summarize，再把 summaries cluster，最后起草 executive brief。同步运行需要 4 小时，成本是 $2,000/night。然后你听说了 batch APIs。

Batch 直接给你 50% off。你还在 system prompt（所有 50k calls 共享）上启用了 prompt caching。两者叠加后，账单降到 $180/night——约为 baseline 的 9%。同一条 pipeline，只改了三个 config。

Batch 是 LLM cost toolkit 中最便宜、却最少被使用的杠杆。原因主要是组织性的：团队以为自己需要 “real-time”，但 SLA 实际上是 “by morning”。本课的目标是别把 90% 的账单留在桌上。

## 概念

### 三个 batch APIs

**OpenAI Batch API**：上传 JSONL file，其中包含一组 requests。承诺 24-hour turnaround（实践中通常约 2-8 小时）。Input 和 output tokens 均享受 50% discount。端点是 `/v1/batches`。符合 cache 条件的 inputs 还能在此基础上获得 cached-input pricing。

**Anthropic Message Batches**：JSONL upload。24-hour turnaround。50% discount。支持 `cache_control`——cache writes 是显式的，batch 内 reads 会自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS input。Gemini 有类似的 50% discount。可与 Vertex pipelines 集成。

### 语义：asynchronous，不是 slow

Batch 的含义是 “我承诺在 24 小时内返回”，不是 “这一定要跑 24 小时”。典型 P50 是 2-6 小时。Provider 会在 off-peak windows 调度你的 batch，因为那时 GPU inventory 利用不足。

### 与 caching 叠加

一个 50k-document summarization workload，使用相同的 4K-token system prompt：

- Synchronous uncached：50000 × ($input × 4000 + $output × 200)，按 full rates 计价。
- Synchronous cached：system prompt 在第一次 write 后被缓存；剩余 49999 次获得便宜 10x 的 input。
- Batch cached：上述全部成立，并且 read 和 write 再享受 50% discount。

叠加结果：batch + cache = sync uncached bill 的约 10%。任何 overnight 运行、并且有 shared system prompt 的 workload 都应该使用它。

### Workload triage

**Interactive** — 用户等着响应。TTFT matters。使用带 prompt caching 的 synchronous call。不能 batch。

**Semi-interactive** — 用户提交任务，几分钟后回来查看。使用 async queue；如果 batch 不可用，则 fallback 到 sync。可以想象 moderate-volume RAG indexing。

**Batch** — 用户期待 “by morning” 或 “next hour” 拿到结果。Content pipelines、scale classification、offline analysis。总是 batch，总是叠加 caching。

常见错误：因为 pipeline 在 production，就把所有东西都归为 interactive。Production 不是 latency spec；SLA 才是。

### Partial-interactivity trap

有些 features 看起来是 interactive，但其实可以容忍 5-10 分钟。例如：带 “refresh” button 的 nightly customer health report。用户点击 refresh；等 10 分钟没问题。团队却把它做成 synchronous。50 个 concurrent refreshes 的成本会是 batched-and-delivered-via-email 方案的 10x。

要问的问题是：“24-hour 对这个用户意味着什么？”如果答案是 “他们根本不会注意到”，就 batch it。

### Output-schema trap

Batch file formats 因 provider 而异：

- OpenAI：JSONL，一行一个 request。
- Anthropic：JSONL，一行一条 message；response format 嵌在里面。
- Vertex：BigQuery table 或 GCS prefix with TFRecord。

跨 providers 写 “one batch client” 意味着每个 provider 都需要 adapter code。那些宣称 multi-provider batch 的 gateways（Portkey、LiteLLM 某些 tiers）仍然只是 thin-wrap raw format。

### 你应该记住的数字

- Providers 间 batch discount：input + output 统一 50% flat。
- Turnaround SLA：保证 24 小时，典型 P50 是 2-6 小时。
- Batch + cached input 叠加：约为 sync uncached cost 的 10%。
- Workload triage rule：如果 24h latency 可接受，就 always batch。

## Use It

`code/main.py` 会为 50k-document workload 计算 sync、sync+cache、batch、batch+cache 四种方式的成本，并报告以美元和百分比计的 savings。

## Ship It

本课产出 `outputs/skill-batch-triager.md`。给定 workload characteristics，它会 triage 到 interactive/semi/batch，并估算 savings。

## 练习

1. 运行 `code/main.py`。对于一个 100k-doc pipeline，system prompt 为 3K tokens、output 为 500 tokens，计算 full stack（batch + cache）相对 sync baseline 的 savings。
2. 选择一个你熟悉的真实产品中的三个 features。把每个 triage 到 interactive/semi/batch。
3. 用户抱怨他们的 report 花了 3 小时。这是 batch mis-triage，还是合法的 interactive？写下 decision criterion。
4. 你的 batch API return SLA 是 24h，但 P99 是 20 hours。你如何向用户沟通？在 edge case 下 downstream system behavior 是什么？
5. 计算 break-even：当 shared-prefix length 达到多少时，batch + cache 会比在自己的 reserved GPU 上 overnight 运行更便宜？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | “async discount” | 50% off，24h turnaround |
| JSONL | “batch format” | 每行一个 JSON request；OpenAI/Anthropic standard |
| Message Batches | “Anthropic batch” | Anthropic 的 batch API product name |
| Batch prediction | “Vertex batch” | Vertex AI 的 batch API product |
| Turnaround SLA | “24h promise” | 保证值，不是典型值；典型值是 2-6h |
| Workload triage | “interactivity decision” | Interactive / semi / batch routing decision |
| Output schema | “response format” | 每个 provider 的 JSONL layout；不可移植 |
| Stacked discount | “batch + cache” | 两者都适用时，约为 uncached sync bill 的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format 和 `/v1/batches` semantics。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format 和 `cache_control` interaction。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
