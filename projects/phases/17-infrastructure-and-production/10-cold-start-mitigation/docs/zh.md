# Serverless LLMs 的冷启动缓解

> 一个 20 GB model image 从 cold 到 serving，需要 5-10 分钟（7B）到 20+ 分钟（70B）。在真正的 serverless 世界里，这不是 warm-up，而是 outage。Mitigations 主要工作在五层：pre-seeded node images（AWS 上的 Bottlerocket，dual-volume arch）、model streaming（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU memory snapshots（Modal checkpoints，restart 最多快 10x）、warm pools（`min_workers=1`）、tiered loading（ServerlessLLM 的 NVMe→DRAM→HBM pipeline，latency reduction 10-200x），以及 live migration 这个 bonus pattern：移动 input tokens（KB）而不是 KV cache（GB）。Modal 公布的 cold starts 下限是 2-4s；Baseten default 为 5-10s，pre-warming 后可达 sub-second。本课教你测量、预算并叠加这些 layers。

**Type:** Learn
**Languages:** Python（stdlib，toy cold-start path simulator）
**Prerequisites:** Phase 17 · 02（Inference Platform Economics），Phase 17 · 03（GPU Autoscaling）
**Time:** ~60 minutes

## 学习目标

- 枚举 cold-start mitigation 的五层，并为每层说出一个 tool 或 pattern。
- 对一个 70B model，把总 cold-start time 拆成（node provision）+（weights download）+（weights load into HBM）+（engine init）并计算。
- 解释为什么 live migration 传输 input tokens（KB）而不是 KV cache（GB），以及代价是什么（recomputation）。
- 说出 warm-pool trade-off（为空闲 GPU 付费，或接受 cold-start tail），以及 `min_workers > 0` 变成 mandatory 的 SLA threshold。

## 问题

你的 serverless LLM endpoint 在夜间 scale to zero。早上 8 点 traffic spikes。第一个 request 会等待：

1. Karpenter provision 一个 GPU node：45-60s。
2. Container pull 一个带 weights 的 30 GB image：120-300s。
3. Engine 把 weights load into HBM：45-120s，取决于 model size 和 storage speed。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache pool、tokenizer：10-30s。

总计：220-510s（约 3-8 分钟）后才返回第一个 token。你的 SLA 是 2s。你发货一个 warm-pool（`min_workers=1`），问题看似消失——但现在你要为一张 idle GPU 24x7 付费。如果 service 有 5 个 products，每个都有一个 warm replica，那就是 5 × 24 × 30 = 3,600 GPU-hours/month，无论有没有一个用户调用。

Cold-start mitigation 的目标，是保留 serverless economics，同时尽量接近 always-on 的 latency。

## 概念

### Layer 1 — pre-seeded node images（Bottlerocket）

在 AWS 上，Bottlerocket 的 dual-volume architecture 会把 OS 和 data 分离。你可以 snapshot 已经 pre-pulled container image 的 data volume，并在 `EC2NodeClass` 中引用这个 snapshot ID。新 nodes 启动时，weights 已经在 local NVMe 上——step 2 和部分 step 3 消失。它和 Karpenter 原生配合。对大模型来说，典型节省是每次 cold start 2-4 分钟。

GCP 上的等价模式是带 pre-baked container layers 的 custom VM images。Azure 上则是同样模式的 managed disk snapshots。

### Layer 2 — model streaming（Run:ai Model Streamer）

不要等完整文件加载完再回答第一个 request，而是 layer-by-layer 把 weights stream into GPU memory；只要第一个 transformer block resident，就开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生提供。它支持 S3、GCS 和 local NVMe。通过把 I/O 和 compute setup 重叠，大模型的 weight-load time 大约减半。

### Layer 3 — GPU memory snapshots（Modal）

Modal 在第一次 load 后，对 GPU state（weights、CUDA graphs、KV cache region）做 checkpoint。后续 restarts 直接 deserialize into HBM——比重新初始化快 10x。这是最接近“2 秒 boot 一个 warm GPU”的东西。Trade-off：snapshots 是 per-GPU-topology 的，所以如果 Karpenter 把你迁移到不同 SKU，就要重新 checkpoint。

### Layer 4 — warm pools（min_workers=1）

最简单的 mitigation：保持一个 replica 永远 ready。成本是一张 GPU 的 hourly rate，24x7。对小模型来说，这个算术很残酷（你每小时支付 $0.85-$1.50 来避免 30s cold start）；对大模型则相对友好（每小时支付 $4 来避免 5-minute cold start）。Warm pools 变成 mandatory 的 SLA threshold 通常是：70B+ model 上 TTFT P99 < 60s。

### Layer 5 — tiered loading（ServerlessLLM）

ServerlessLLM 把 storage 看成一个 hierarchy：NVMe（快且容量大）、DRAM（中等但可分层）、HBM（小但即时）。Weights 会预加载到 DRAM，再按需 load into HBM。论文报告，相比 naive disk-to-HBM，cold loads latency reduction 为 10-200x。Production adoption 还早，但已经有 vLLM integrations。

### Layer 6 — live migration（bonus pattern）

当一个 node 变得不可用（spot eviction、node drain），传统模式是 cold-start 另一个 replica 并 drain request queue。Live migration 会把 input tokens（kilobytes）移动到已经加载模型的 destination，并在 destination 上 recompute KV cache。Recomputation 比通过网络传输 GB 级 KV cache 便宜。它适用于 disaggregated deployments。

### Warm-pool math

对 P99 TTFT SLA 为 2s 的 service，问题不是“要不要 warm pool”，而是“要多少 warm replicas，以及哪些 paths 获得它们”。

- High-value interactive paths（live chat、voice agent）：`min_workers=1-2`。
- Background batch paths（nightly classification）：接受 scale-to-zero，能忍受 5-10 分钟 cold start。
- Premium tier：每个 tenant 使用 `min_workers` 和 dedicated capacity。

### 先测量，再优化

Fresh node 上 70B model 的 cold-start anatomy（示例）：

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume（eliminate） |
| Weights to HBM | 75s | Model streamer（halve）；GPU snapshot（eliminate） |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total with mitigations** | **~15s** | 22x reduction |

### 你应该记住的数字

- Modal cold start：2-4s（with GPU snapshots）。
- Baseten default cold start：5-10s；pre-warming 后 sub-second。
- Raw 70B cold start：3-8 minutes。
- Run:ai Model Streamer：约 2x weight-load speedup。
- ServerlessLLM tiered loading：latency reduction 10-200x（paper numbers）。

## Use It

`code/main.py` 会建模一条 cold-start path，并分别打开 / 关闭每个 mitigation。它会报告 total cold-start time、warm-pool cost，以及 warm pool 相比 cold-start tax 划算的 break-even request rate。

## Ship It

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、model size 和 traffic shape，它会选择该叠加哪些 mitigations。

## 练习

1. 运行 `code/main.py`。计算 warm replica 比“用额外 request drops 支付 cold-start tax”更便宜的 break-even request rate。
2. 你部署一个 13B model，P99 TTFT SLA 为 3s。选择能达到目标的 minimum mitigation stack（最少 layers）。
3. Bottlerocket pre-seeding 消除了 image pull，但 weights 仍然要从 snapshot load 到 HBM。如果 snapshot-backed NVMe read 为 7 GB/s，计算一个 70B model 的 wall-clock。
4. 你的 serverless provider 提供 GPU snapshots（Modal），但团队拒绝，理由是“snapshots leak PII”。分别为两边论证：真实风险是什么？mitigation 是什么（ephemeral snapshots、encryption、namespace isolation）？
5. 设计一个 tiered warm-pool policy：paid users、trial users 和 batch workloads 分别需要多少 warm replicas？展示数学推导。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cold start | “the big pause” | Fresh replica 上从 request 到 first token 的时间 |
| Warm pool | “always-on minimum” | `min_workers >= 1`，保持至少一个 replica ready |
| Pre-seeded image | “baked AMI” | Node image 中 container weights 已经 pre-resident |
| Bottlerocket | “AWS node OS” | AWS container-optimized OS，支持 dual-volume snapshot |
| Model streamer | “streaming load” | 把 weights I/O 与 compute setup 重叠 |
| GPU snapshot | “checkpoint to HBM” | 序列化 post-load GPU state；restart 时反序列化 |
| Tiered loading | “NVMe + DRAM + HBM” | 分层 storage tiers；按需加载 |
| Live migration | “move tokens” | 传输 input（KB），在 destination 上 recompute KV |
| `min_workers` | “warm replicas” | Serverless minimum keep-alive count |
| Scale-to-zero | “full serverless” | Idle 时无成本；接受完整 cold-start tax |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal published benchmarks 和 checkpoint architecture。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot pattern。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — overlap weights load with compute setup。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — pre-warming playbook。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — tiered loading design。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — disaggregated deployments 的 live migration。
