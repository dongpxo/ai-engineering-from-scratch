# 托管 LLM 平台：Bedrock、Vertex AI、Azure OpenAI

> 三大云厂商，三种不同策略。AWS Bedrock 是模型市场：Claude、Llama、Titan、Stability、Cohere 都在同一个 API 后面。Azure OpenAI 是 OpenAI 独家合作加上 Provisioned Throughput Units (PTUs)，用于购买专用容量。Vertex AI 则是 Gemini-first，并且拥有最强的长上下文和多模态叙事。2026 年 Artificial Analysis 测得 Azure OpenAI 在 Llama 3.1 405B 等价负载上的 median 约为 50 ms，Bedrock 约为 75 ms；PTUs 解释了这个差距，因为专用容量会优于共享 on-demand。决策规则不是“哪个最快”，而是“哪个模型目录和 FinOps surface 匹配我的产品”。本课教你把 tradeoffs 写下来，而不是凭感觉选择。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## 学习目标

- 命名三种平台策略（marketplace vs exclusive vs Gemini-first），并把每种策略匹配到一个产品用例。
- 解释 Azure OpenAI 的 Provisioned Throughput Units (PTUs) 买到的是什么，以及为什么 on-demand Bedrock 在 405B 规模上通常看起来慢约 25 ms。
- 画出每个平台的 FinOps attribution surface（Bedrock Application Inference Profiles vs Vertex project-per-team vs Azure scopes + PTU reservations）。
- 写下一条 “two-provider minimum” policy，并解释为什么 single-vendor lock-in 是 2026 年昂贵的错误。

## 问题

你为产品选择了 Claude 3.7 Sonnet。现在你需要把它服务出去。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者走一个 gateway。Direct API 最简单；Bedrock 增加了 BAAs、VPC endpoints、IAM 和 CloudWatch attribution。Gateway 增加了跨 providers 的 failover、统一账单和 rate limits。

更深层的问题是 catalog。如果你需要在同一个产品里同时使用 Claude、Llama 和 Gemini，除非同时购买 Bedrock、Vertex 和 Azure OpenAI，否则无法从单一地方买齐。Hyperscalers 并不是可互换的；它们各自押注了谁拥有 model layer。

本课会映射这三种押注、latency gap、FinOps gap 和 lock-in risk。

## 概念

### 三种策略

**AWS Bedrock** — marketplace。Claude (Anthropic)、Llama (Meta)、Titan (AWS first-party)、Stability (image)、Cohere (embeddings)、Mistral，以及 image 和 embedding 子目录。一个 API、一个 IAM surface、一个 CloudWatch export。Bedrock 的押注是：客户更想要 optionality，而不是单一模型。

**Azure OpenAI** — exclusive partnership。你可以在 Azure datacenters 中获得 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，以及 OpenAI models 的 fine-tuning。“Azure OpenAI Service” catalog 中没有非 OpenAI 模型；这些模型属于 Azure AI Foundry（单独产品）。Azure 的押注是：OpenAI 仍然处在 frontier，并且客户想要针对这段关系的 enterprise controls。

**Vertex AI** — Gemini first, everything else second。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（third-party）。Vertex 的押注是 multimodal long-context；1M-token Gemini context 是差异化点。

### 大规模下的 latency gap

Artificial Analysis 会持续运行 benchmarks。在等价的 Llama 3.1 405B deployments（shared on-demand）上，Azure OpenAI 的 median first-token latency 约为 50 ms；Bedrock 约为 75 ms。这个差距不是 AWS 失败，而是 capacity model 的差异。Azure 出售 PTUs (Provisioned Throughput Units)，为你的 tenant 预留 GPU capacity。Bedrock 的等价产品（Provisioned Throughput）也存在，但每 unit 大约从 $21/hour 起，大多数客户仍停留在 shared on-demand。

On-demand shared capacity 会和其他所有客户的 traffic 竞争。Dedicated capacity 不会。如果你的产品 SLA 是 TTFT < 100 ms at P99，你要么购买 Azure PTUs，要么购买 Bedrock Provisioned Throughput，要么接受默认方差。

### Provisioned Throughput economics

Azure PTUs：一块预留的 inference compute。对于可预测 workload，相比 on-demand 最多可节省约 70%。成本按小时固定，不管有没有流量；即使 idle 也要为 reservation 付费。Break-even 通常在 40-60% sustained utilization 左右。

Bedrock Provisioned Throughput：根据 model 和 region 不同，每小时 $21-$50。数学类似；break-even 约为 half peak utilization。需要 monthly commitment。

Vertex provisioned capacity 按 Gemini SKU 出售；pricing 随 model 和 region 变化，公开程度较低。

### FinOps surface — 真正的差异化点

**Bedrock Application Inference Profiles** 是 marketplace 里最干净的 attribution。给 profile 打上 `team`、`product`、`feature` 标签；所有 model invocations 都通过它 route；CloudWatch 会按 profile 拆出 cost，不需要 post-processing。该功能 2025 年加入，仍是 hyperscaler native 中最细粒度的方案。

**Vertex** attribution 是 project-per-team 加 labels-everywhere。你把每个 team 建模为一个 GCP project，在每个 resource 上放 labels，然后用 BigQuery Billing Export + DataStudio 做 rollups。工作量更大，但 BigQuery 让你可以对 cost data 写任意 SQL。

**Azure** 依赖 subscription/resource-group scopes 加 tags，PTU reservations 是 first-class cost object。Tags 从 resource groups 继承，而不是从 requests 继承，所以 per-request attribution 需要 Application Insights custom metrics，或一个会 stamp headers 的 gateway。

模式是：Bedrock native 最干净，Vertex 通过 BigQuery 最灵活，Azure 除非你自己 instrumentation，否则最不透明。

### Lock-in 是 2026 年的风险

当一个模型占主导时，single-hyperscaler commitment 还可以接受。2026 年 frontier 每个月都在移动：一个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。锁到一个平台，就是把自己锁在 frontier 的另外三分之二之外。

有效团队采用的模式：任何 product-critical LLM call 都至少使用两个 providers。Bedrock plus Azure OpenAI 是常见组合：Claude 来自一个平台，GPT 来自另一个平台，通过同一个 gateway 在它们之间 failover。Cost uplift 很小，因为 gateway 会 route 到最优路径；outage 期间的 availability uplift（例如 Azure OpenAI 2025 年 1 月 incident、AWS us-east-1 outage）是决定性的。

### Data residency、BAAs 和 regulated industries

Bedrock：大多数 region 提供 BAAs；VPC endpoints；guardrails。常见 fintech default。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；EU data residency；enterprise-regulated default。
Vertex：HIPAA、GDPR、按 region 的 data residency；Google Cloud compliance stack。

三者都满足基本 checkbox。差异在于 data retention policies、logs 如何处理，以及 abuse-monitoring 是否读取你的 traffic（多数默认 opt-in；enterprise 可 opt-out）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等价负载上的 median TTFT：约 50 ms（with PTUs）。
- Bedrock on-demand median TTFT：约 75 ms。
- Bedrock Provisioned Throughput：每 unit $21-$50/hr。
- Azure PTU break-even：约 40-60% sustained utilization。
- 高利用率下 PTU 相比 on-demand 的节省：最高 70%。

## Use It

`code/main.py` 会在 synthetic workload 上比较三个平台：它建模 on-demand vs PTU economics、TTFT variance 和 cost attribution fidelity。运行它，观察 PTUs 在哪里回本，以及 marketplace 的模型广度在哪里压过 TTFT gap。

## Ship It

本课产出 `outputs/skill-managed-platform-picker.md`。给定 workload profile（models needed、TTFT SLA、daily volume、compliance requirements），它会推荐 primary platform、fallback，以及 FinOps instrumentation plan。

## 练习

1. 运行 `code/main.py`。对于 70B class model，Azure PTU 在什么 sustained utilization 下会击败 on-demand？计算 break-even，并与 advertised 40-60% band 比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个 two-provider deployment：哪个 model 放到哪个 hyperscaler，前面放什么 gateway，failover policy 是什么？
3. 一个 regulated healthcare customer 要求 BAAs、US-East data residency 和 sub-100ms P99 TTFT。选择一个平台，并用三个具体 features 证明选择合理。
4. 你发现 Bedrock bill 本月上涨 4x，但 traffic 没变。没有 Application Inference Profiles 时，你会如何找到 culprit？有 profiles 时，需要多久？
5. 阅读 Azure OpenAI 和 Bedrock pricing pages。对于 100M-token/month Claude workload，哪个更便宜：direct Anthropic API、Bedrock on-demand，还是 Bedrock Provisioned Throughput？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Bedrock | “AWS LLM service” | 横跨 Claude、Llama、Titan、Mistral、Cohere 的 model marketplace |
| Azure OpenAI | “Azure's ChatGPT” | 位于 Azure datacenters 中、带 enterprise controls 的 OpenAI 独家模型 |
| Vertex AI | “Google's LLM” | Gemini-first platform，并通过 Model Garden 提供 third-party models |
| PTU | “dedicated capacity” | Provisioned Throughput Unit；预留 inference GPUs，按小时计价 |
| Application Inference Profile | “Bedrock tagging” | 带 tags 的 per-product cost/usage profile，CloudWatch-native |
| Model Garden | “Vertex catalog” | Vertex AI 的 third-party model section，与 Gemini 分开 |
| Two-provider minimum | “LLM redundancy” | 让每条 critical LLM path 横跨至少 2 个 hyperscalers 的 policy |
| BAA | “HIPAA paperwork” | Business Associate Agreement；处理 PHI 时必需；三家都提供 |
| Abuse monitoring | “the log watcher” | Provider-side safety scan，扫描 prompts/outputs；enterprise 可 opt-out |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — authoritative rate card 和 Provisioned Throughput pricing。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU economics 和 rate cards。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini tiers 和 Model Garden surcharges。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — providers 间持续 latency 和 throughput benchmarks。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — enterprise decision framework。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — attribution mechanics side-by-side。
