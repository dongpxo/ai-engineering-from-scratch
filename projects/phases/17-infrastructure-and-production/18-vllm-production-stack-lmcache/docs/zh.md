# 搭配 LMCache KV Offloading 的 vLLM Production Stack

> vLLM 的 production-stack 是 Kubernetes 部署参考实现：router、engines 和 observability 已经连在一起。LMCache 是 KV-offloading layer，它把 KV cache 从 GPU memory 中抽离出来，并在 queries 和 engines 之间复用（先放 CPU DRAM，再放 disk/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）让这条路径变成 asynchronous 和 pluggable。Offload latency 不直接暴露给用户。即使没有 shared prefixes，LMCache 也有价值：当 GPU 的 KV slots 不够用时，被 preempted 的 requests 可以从 CPU 恢复，而不是重新计算 prefill。在 4 台 a3-highgpu-4g、共 16x H100（80GB HBM）的公开 benchmark 中：当 KV cache 超过 HBM 时，native CPU offload 和 LMCache 都显著提升 throughput；在低 KV footprint 下，所有配置都接近 baseline，只带来小 overhead。

**Type:** Learn
**Languages:** Python（stdlib，玩具版 KV-spill simulator）
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 画出 vLLM production-stack layers：router、engines、KV offload、observability。
- 解释 KV Offloading Connector API（v0.9.0+），以及 0.11.0 asynchronous path 如何隐藏 offload latency。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM），何时只是增加 overhead（KV 足够小，可以放进 HBM）。
- 根据 deployment constraints，在 native vLLM CPU offload 和 LMCache connector 之间做选择。

## 问题

你的 vLLM serving 在 concurrency 上升时显示 GPUs 处于 100% HBM，并持续出现 preemption events。Requests 被 evict、requeue，同一个 2K-token prompt 在一分钟内被重新 prefill 四次。GPU compute 花在冗余 prefill 上；goodput 远低于 raw throughput。

增加 GPUs 的成本线性上升。增加 HBM 不现实。但 CPU DRAM 很便宜：单个 socket 可以有 512 GB+，latency 比 HBM 高几个数量级，但对“临时温热”的 KV cache 来说足够。

LMCache 把 KV cache 抽取到 CPU DRAM，这样 preempted requests 可以快速恢复，并且 engines 之间的 repeated prefixes 可以共享 cache，不需要每个 engine 各自重新 prefill。

## 概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是 Kubernetes 部署参考实现：

- **Router**：cache-aware（Phase 17 · 11）。消费 KV events。
- **Engines**：vLLM workers。每个 GPU 一个，或每个 TP/PP group 一个。
- **KV cache offload**：LMCache deployment 或 native connector。
- **Observability**：Prometheus scrape、Grafana dashboards、OTel traces。
- **Control plane**：service discovery、config、rolling updates。

以 Helm chart + operator 形式发布。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了 Connector API，用于 pluggable KV cache backends。Engine 把 blocks offload 给 connector；connector 存储它们（RAM、disk、object storage、LMCache）。Request 需要某个 block 时，connector 再把它加载回来。

vLLM 0.11.0（2026 年 1 月）加入 asynchronous offload path：常见情况下，offload 可以在后台发生，engine 不必阻塞在它上面。End-to-end latency 和 throughput 仍然取决于 workload shape、KV cache hit rate 和 system pressure；vLLM 自己的 notes 也指出，custom-kernel offload 在低 hit rate 下可能降低 throughput，并且 async scheduling 与 speculative decoding 存在已知 interaction issues。

### Native CPU offload vs LMCache

**Native vLLM CPU offload**：engine-local。把 KV blocks 存到 host RAM。实现快，没有 network hop。不能跨 engines。

**LMCache connector**：cluster-scale。把 blocks 存到 shared LMCache server（CPU DRAM + Ceph/S3 tier）。任何 engine 都可以访问这些 blocks。已有 16x H100 benchmarks。

当单个 engine 有 HBM pressure 时，选 native。当多个 engines 共享 prefixes（带 common system prompts 的 RAG、shared templates 的 multi-tenant）时，选 LMCache。

### Benchmark behavior

4 台 a3-highgpu-4g、共 16x H100（80 GB HBM）的测试：

- Low KV footprint（短 prompts、低 concurrency）：所有配置都接近 baseline，LMCache 增加约 3-5% overhead。
- Moderate footprint：LMCache 开始在 engines 之间的 prefix reuse 上产生帮助。
- KV exceeds HBM：native CPU offload 和 LMCache 都显著提升 throughput；LMCache 收益更大，因为能 cross-engine sharing。

### LMCache 何时 decisive

- Multi-tenant serving，其中 system prompts 在 tenants 之间共享。
- RAG，其中 document chunks 在 queries 之间重复。
- Fine-tuned variants（LoRA）运行在同一个 base 上，base-model KV reuse 可以减少重复工作。
- Preemption-heavy workloads：从 CPU 恢复比重新 prefill 更便宜。

### 什么时候不要启用

- HBM pressure 很小：你只付 overhead，没有收益。
- Short contexts（<1K tokens）：transfer time > re-prefill。
- Single-tenant single-prompt workload：没有可捕获的 reuse。

### 与 disaggregated serving 的集成

Phase 17 · 17 的 disaggregated serving + LMCache 会叠加收益：prefill pool 到 decode pool 的 KV transfers 如果暂时没用，会落入 LMCache；后续 queries 再从 LMCache 取回。Phase 17 · 11 的 cache-aware router 可以路由到 local cache 或 LMCache-shared cache 匹配的 engine。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：asynchronous offload path；end-to-end latency impact 取决于 workload、KV hit rate 和 system pressure（不是绝对保证）。
- 16x H100 benchmark：KV footprint 超过 HBM 时，LMCache 有帮助。
- HBM pressure 很小时：3-5% overhead 且没有收益。

```figure
zero-sharding
```

## Use It

`code/main.py` 会模拟一个 preemption-heavy workload，分别在启用和不启用 LMCache 的情况下运行。它报告 avoided re-prefills、throughput gain 和 break-even HBM utilization。

## Ship It

本课产出 `outputs/skill-vllm-stack-decider.md`。给定 workload shape 和 vLLM deployment，它会决定 native、LMCache 或 neither。

## 练习

1. 运行 `code/main.py`。LMCache 从什么 HBM utilization 开始回本？
2. 一个 tenant 每小时 200 个 queries，共享一个 6K-token system prompt。计算每个 tenant 的预期 LMCache savings。
3. LMCache server 是 single point of failure。设计 HA strategy（replicas、fallback to native）。
4. LMCache 把数据存到 spinning disk 上的 Ceph。对于一个 70B FP8 的 4K-token KV（500 MB），read time 和 re-prefill 相比如何？
5. 论证 vLLM 0.11.0 asynchronous path 是否 “free”：overhead 藏在哪里？

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Production-stack | “the reference deployment” | vLLM 的 Kubernetes Helm chart + operator。 |
| Connector API | “KV backend interface” | vLLM 0.9.0+ 的 pluggable KV store interface。 |
| Native CPU offload | “engine-local spill” | 把 KV 存到同一 engine 的 host RAM。 |
| LMCache | “cluster KV cache” | 运行在 CPU DRAM + disk 上的 cross-engine KV cache server。 |
| 0.11.0 async | “non-blocking offload” | 隐藏在 engine stream 后面的 offload。 |
| Preemption | “evict to make room” | HBM 满时发生的 KV cache shuffle。 |
| Prefix reuse | “same system prompt” | 多个 queries 共享开头；cache hit。 |
| Ceph tier | “disk tier” | Cache hierarchy 中位于 DRAM 下方的 durable storage。 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details。
