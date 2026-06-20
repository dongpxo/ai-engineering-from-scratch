# 多区域 LLM Serving 与 KV Cache Locality

> Round-robin load balancing 对带缓存的 LLM inference 是主动有害的。一个 request 如果没有落到持有其 prefix 的节点上，就要支付完整 prefill cost：长 prompt 上 P50 大约 800 ms，而 cache hit 时约 80 ms。到 2026 年，production pattern 是 cache-aware router（Rust 写的 vLLM Router、llm-d router），它消费 KV-cache events，并按 prefix-hash match 路由。近期研究（GORGO）把 cross-region network latency 显式纳入 routing objective。商业 “cross-region inference” 产品（Bedrock cross-region inference、GKE multi-cluster gateways）把 inference 当作 opaque service，它们处理 availability，不处理 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月演练 us-east-1 failover，约 22 分钟恢复。DR 的现实是：32% 的 LLM DR failures 源自团队备份了 weights，却忘了 tokenizer files 或 quantization configs。

**Type:** Learn
**Languages:** Python（stdlib，玩具版 prefix-cache-aware router simulator）
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 解释为什么 round-robin load balancing 会破坏 cached inference，并量化 TTFT penalty。
- 画出 cache-aware router：输入（KV-cache events）、算法（prefix-hash match）、tie-breaker（GPU utilization）。
- 说出 LLM 中 32% DR failure driver（missing tokenizer files / quantization configs），并给出三文件 DR checklist。
- 区分商业 cross-region offerings（Bedrock CRI、GKE Multi-Cluster Gateway）和 KV-aware routing。

## 问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB，并使用 round-robin。生产中的 prefix cache hit rate 掉到 8%。TTFT P50 变成三倍。vLLM logs 显示每个 request 都在支付完整 prefill cost。

Round-robin 对 stateless services 是最优的。LLM inference 天生是 stateful 的：KV cache 编码了模型已经看过的一切。盲目路由就是路由到错误的 cache。

另外，你的团队有一份 DR plan。你把 model weights 跨区域备份到 S3。区域故障发生；你尝试 failover；replica 拒绝启动。你忘了 `tokenizer.json`、quantization config 和 RoPE scaling config 位于另一个没有同步的 bucket。

Multi-region LLM serving 是 cache 问题、routing 问题和 DR hygiene 问题，不是 load-balancer 问题。

## 概念

### Cache-aware routing

Request 带着 prompt 到达。Router 对 prefix 做 hash（比如前 512 tokens）；它询问每个 replica：“你是否已经缓存了这个 prefix？” Replicas 在分配和驱逐 blocks 时，通过 pub/sub channel 发布 KV-cache events。Router 选择有 match 的 replica；如果没有任何 match，则回落到基于 GPU utilization 的 tie-breaker。

**vLLM Router**（Rust，2026 production-stack）：订阅 `kv.cache.block_added` events，维护 prefix-hash → replica index，用 O(1) lookup 路由。没有 match 时回落到 least-queue-depth。

**llm-d router**：同样模式，Kubernetes-native。通过 ControlPlane API 发布 events。

**SGLang RadixAttention**（Phase 17 · 06）是 intra-replica 等价物。Cross-replica routing 严格位于它的上游。

### 数字

2K-token prompt、Llama 3.3 70B FP8、H100 上的 TTFT P50：

- Cache hit（同一 replica，prefix resident）：约 80 ms。
- Cache miss（cold prefill）：约 800 ms。

10x 差距。如果你的 router 在 replicas 之间达到 60-80% 的 prefix cache hit，就能在 N-replica capacity 下接近 single-replica performance。如果只有 10%，就接近 naive scaling。

### Cross-region 有新的约束：network latency

Inter-region RTT：

- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果 routing 把一个 us-east-1 的 request 发到 ap-southeast-1 的 hot prefix，省下的 prefill（800 → 80 ms）会被 440 ms round-trip 吞掉。GORGO（2026 research）把这一点显式化：联合最小化 `prefill_time + network_latency`，而不是只最小化 prefill。通常答案是保持 regional routing，除非 prefix 是巨大的 multi-MB，prefill 才会占主导。

### 商业 “cross-region inference” 在这里帮不上忙

AWS Bedrock cross-region inference 会在 capacity pressure 时自动把 requests 路由到其他 regions。它优化 availability，不优化 TTFT，并且把 inference 当作 opaque。GKE Multi-Cluster Gateway 也是一样：service-level failover，不感知 KV cache。

即使用了这些产品，你仍然需要 app-layer cache-aware router。它们处理 “us-east-1 着火了” 的场景。Cache-aware routing 处理 TTFT 场景。

### DR hygiene：32% missing-files problem

广为引用的 2026 年统计：32% 的 LLM DR failures 发生在团队备份了 weights，却忘了：

- `tokenizer.json` 或 `tokenizer.model`
- Quantization configs（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- Model-specific configs（RoPE scaling、attention masks、chat templates）
- Engine config（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复方式是一个三文件最小 DR manifest：

1. HF model repo 下的所有文件（weights + configs + tokenizer）。
2. Engine-specific serving config。
3. Deployment manifest（K8s YAML、Dockerfile、dependency lock）。

另外：每季度跑一次 DR drill。JPMorgan 的 us-east-1 drill 在 2024 年 11 月达到 22 分钟恢复，只是因为 playbook 已经演练过。

### Data residency 是正交约束

EU customer PHI 不能离开 EU。如果你的 cache-aware router 因为 prefix match 把巴黎来源的 request 发到 us-east-1，你就违反了 GDPR，不管 TTFT 收益有多大。先按 residency boundary 切分 routers，再优化 cache。

### 你应该记住的数字

- Cache hit vs miss TTFT gap：约 10x（2K prompt 上 80 ms vs 800 ms）。
- Inter-region RTT US-EU：约 75 ms。
- DR failure：32% 缺 tokenizer / quant configs。
- JPMorgan us-east-1 failover，2024 年 11 月：22 分钟（30-min SLA）。

## Use It

`code/main.py` 会在一个 multi-region workload 上模拟三种 routing strategies：round-robin、cache-aware regional、cache-aware global。它报告 cache hit rate、TTFT P50/P99 和 cross-region bill。

## Ship It

本课产出 `outputs/skill-multi-region-router.md`。给定 regions、residency constraints 和 SLA，它会设计 routing plan。

## 练习

1. 运行 `code/main.py`。在 75 ms RTT 下，prompt length 到多少时 cross-region routing 会胜过 local-only routing？
2. 你的 cache hit rate 从 70% 掉到 12%。诊断三个可能原因，以及能确认每个原因的 observables。
3. 为一个用 vLLM serving、带 5 个 LoRA adapters 的 70B AWQ-quantized model 设计 DR manifest。列出每个 file 和 config。
4. 论证 Bedrock cross-region inference 对一个有严格 TTFT SLOs 的 fintech 是否 “enough”。引用具体行为。
5. 一个巴黎来源 request 在 us-east-1 命中了 prefix。你会路由过去吗？写出 policy。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Cache-aware routing | “smart LB” | 按 prefix-hash match 路由到持有 KV cache 的 replica。 |
| KV-cache events | “cache pub-sub” | Replicas 发布 block add/evict；router 建索引。 |
| Prefix hash | “cache key” | 前 N 个 tokens 的 hash，用作 router lookup。 |
| GORGO | “cross-region routing research” | arXiv 2602.11688；把 network latency 作为显式项。 |
| Cross-region inference | “Bedrock CRI” | AWS 产品；availability failover，而不是 TTFT awareness。 |
| DR manifest | “the backup list” | 恢复所需的每个文件，不只是 weights。 |
| Data residency | “GDPR boundary” | 限制哪些 region 可以看到 user data 的法律约束。 |
| RTT | “round-trip time” | Network latency；US-EU 75 ms，US-APAC 220 ms。 |
| LLM-aware LB | “cache-hit LB” | 作为产品类别的 cache-aware router。 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带 network latency term 的 cross-region KV-cache reuse。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — availability failover documentation。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware router source。
