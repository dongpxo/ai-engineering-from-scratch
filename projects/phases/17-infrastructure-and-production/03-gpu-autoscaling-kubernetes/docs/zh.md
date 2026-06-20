# Kubernetes 上的 GPU 自动扩缩容：Karpenter、KAI Scheduler、Gang Scheduling

> 这是三层，不是一层。Karpenter 动态 provision nodes（低于一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、topology awareness 和 hierarchical queues——它会避免 7-of-8 partial allocation trap：七个节点在等最后一张 GPU，白白燃烧成本。应用层 autoscalers（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于 inference-specific signals 扩缩容：queue depth、KV cache utilization，而不是 CPU/DCGM duty cycle。经典 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle measurement：100% 可能是 10 个请求，也可能是 100 个。vLLM 会预分配 KV cache memory，所以 memory 永远不会触发 scale-down。本课教你组合这三层，并避开默认 Karpenter `WhenEmptyOrUnderutilized` policy，因为它会在 inference 进行中终止正在运行的 GPU jobs。

**Type:** Learn
**Languages:** Python（stdlib，toy queue-depth autoscaler simulator）
**Prerequisites:** Phase 17 · 02（Inference Platform Economics），Phase 17 · 04（vLLM Serving Internals）
**Time:** ~75 minutes

## 学习目标

- 画出三层 autoscaling（node provisioning、gang scheduling、application-level），并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 不是 vLLM 的正确 HPA signal，并说出两个替代信号（queue depth、KV cache utilization）。
- 描述 gang scheduling，以及 KAI Scheduler 防止的 partial-allocation failure mode（8 张 GPU 中 7 张 idle）。
- 说出会终止 running GPU jobs 的 Karpenter consolidation policy（`WhenEmptyOrUnderutilized`），并说明 2026 年的安全替代方案。

## 问题

你的团队在 Kubernetes 上发货了一个 LLM-serving service。你把 HPA 的 signal 设成 `DCGM_FI_DEV_GPU_UTIL`。业务时段里，service 固定显示 100% utilization。HPA 从不 scale up——它已经认为你满了。你手动加一个 replica；TTFT 下降。HPA 仍然不扩容。这个 signal 在骗你。

另一方面，你用 Cluster Autoscaler 管 nodes。凌晨 2 点来了一个 1M-token prompt；cluster 花了 3 分钟 provision 一个 node，请求超时。

再另一方面，你部署了一个需要跨 2 个 nodes 使用 8 张 GPUs 的 70B model。cluster 有 7 张 GPUs 空闲，另有 1 张分散在 3 个 nodes 上。Cluster Autoscaler 为缺失的那 1 张 GPU provision 一个 node。七个 nodes 等了 4 分钟，一边烧钱，一边等 Kubernetes 把最后一张 GPU 拉起来。

三层，三种不同 failure modes。2026 年的 GPU-aware autoscaling 不是“打开 HPA”。它是组合 node provisioning、gang scheduling 和 application-signal autoscaling。

## 概念

### Layer 1 — node provisioning（Karpenter）

Karpenter 监听 pending pods，并在约 45-60 秒内 provision nodes（GPU nodes 上，Cluster Autoscaler 通常需要 90-120 秒）。它会根据 `NodePool` constraint 动态选择 instance types——如果你的 pod 需要 8 张 H100，而 cluster 没有匹配 node，Karpenter 会直接 provision 一个，而不是去 scale 一个已有 group。

**Consolidation trap**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU pools 很危险。它会终止一个正在运行的 GPU node，把 pods 迁移到更便宜、尺寸更合适的 instance。对 inference workloads 来说，这意味着驱逐正在运行的请求，并在新 node 上重新加载一个 70B model。损失是数分钟 capacity 加上请求失败。

GPU pools 的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

这允许 Karpenter 在一小时后 consolidate 真正空闲的 nodes，但永远不会 evict 一个 running job。

### Layer 2 — gang scheduling（KAI Scheduler）

KAI Scheduler（项目曾名为 “Karp”，后来更名）处理 default kube-scheduler 不处理的事情：

**Gang scheduling**——all-or-nothing scheduling。一个需要 8 张 GPUs 的 distributed inference pod，要么 8 张一起启动，要么一张都不启动。没有它，你会遇到 partial-allocation trap：8 个 pods 中 7 个启动，随后无限等待，同时持续烧钱。

**Topology awareness**——知道哪些 GPUs 共享 NVLink，哪些在同一个 rack，哪些之间有 InfiniBand。然后据此放置 pods。DeepSeek-V3 67B tensor-parallel workload 必须留在一个 NVLink domain 内；KAI Scheduler 会尊重这一点。

**Hierarchical queues**——多个 teams 用 priority 和 quota 竞争同一个 GPU pool。Team A 的 production pinch 是否会被 Team B 的 training job 抢占，只能由 priority rules 决定。

KAI 会作为 secondary scheduler 和 kube-scheduler 一起部署；你通过 annotation 指定 workloads 使用它。Ray 和 vLLM production-stack 都有集成。

### Layer 3 — application-level signals

**HPA trap**：`DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle metric——它测量 GPU 在每个 sampling interval 是否正在做 work。100% utilization 可能代表 10 个 concurrent requests，也可能代表 100 个；反正 GPU 都是 busy。基于 duty cycle 扩缩容，就是盲目扩缩容。

更糟的是，vLLM 和类似 engines 会预分配 KV cache memory（最高到 `--gpu-memory-utilization`）。即使只有一个 request，memory usage 也会停在接近 90%。基于 memory 的 HPA 永远不会 scale down。

**2026 replacement signals**：

- Queue depth（等待 prefill 的 requests 数）。
- KV cache utilization（active sequences 已分配 block 的比例）。
- Per-replica P99 TTFT（你的 SLA signal）。
- Goodput（每秒满足全部 SLOs 的 requests）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 会消费这些 signals 并 scale replicas。对于 LLM serving，它们会完全替代 HPA。

### 什么时候用什么

| Scale decision | Tool |
|----------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA（或基于 queue depth 的 custom HPA） |
| Choose GPU type | Karpenter NodePool |
| Preempt low-priority | KAI Scheduler queues |

### Disaggregated prefill/decode 会让一切更复杂

如果你运行 disaggregated prefill/decode（Phase 17 · 17），你会有两类 pod，它们的 scaling triggers 不同：prefill pods 根据 queue depth 扩缩容，decode pods 根据 KV cache pressure 扩缩容。llm-d 会把它们暴露为带有 per-role HPA 的 separate `Services`。不要尝试在两者前面放一个单一 HPA。

### Cold start 在这里也重要

Cold-start mitigation（Phase 17 · 10）是 node provisioning time 变成用户可见延迟的地方。Karpenter 的 45-60 秒 warm-up，加上 20GB model load，再加 engine init，意味着 from-zero request 要花 2-5 分钟。对 SLO-critical paths 保留一个 warm pool（`min_workers=1`），或者在 application layer 使用 Modal-style checkpointing。

### 你应该记住的数字

- Karpenter node provisioning：约 45-60s，而 Cluster Autoscaler 约 90-120s（GPU nodes）。
- KAI Scheduler 防止 partial-allocation waste——7-of-8 trap。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA signal：坏的；使用 queue depth 或 KV utilization。
- Karpenter `WhenEmptyOrUnderutilized`：会终止 running GPU jobs。对 inference 使用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## Use It

`code/main.py` 会在 bursty GPU workload 上模拟一个三层 autoscaler。它对比 naive HPA（duty cycle）、queue-depth HPA 和 KAI-gang-scheduled scaling，并报告 unmet requests、idle-GPU minutes 和 composite score。

## Ship It

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定 cluster topology、workload shape 和 SLO，它会设计一套三层 autoscaling plan。

## 练习

1. 运行 `code/main.py`。在 bursty workload 下，naive duty-cycle HPA 会 drop 多少个 queue-depth HPA 能接住的 requests？差异来自哪里？
2. 为一个服务 Llama 3.3 70B FP8 on H100 SXM5 的 cluster 设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个让 non-GPU workloads 远离这些 nodes 的 taint。
3. 你的团队报告 deployments 卡在 Pending，因为“GPUs available but pod won't schedule”。诊断一下——这是 Karpenter、kube-scheduler，还是 KAI Scheduler？哪些 metrics 可以确认？
4. 为 disaggregated prefill pods 选择一个 autoscale signal，再为 decode pods 选择另一个 signal。分别说明理由。
5. 计算 `WhenEmptyOrUnderutilized` consolidation trap 的成本：一个 24x7 production service 平均每天有 60 次 request-dropping events，且 P99 TTFT > 10s。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Karpenter | “node provisioner” | Kubernetes node autoscaler；sub-minute provisioning |
| Cluster Autoscaler | “old scaler” | Kubernetes node autoscaler 的前身；更慢，基于 group |
| KAI Scheduler | “GPU scheduler” | 用于 gang + topology + queues 的 secondary scheduler |
| Gang scheduling | “all or nothing” | 原子化 schedule N 个 pods，或把它们全部 defer |
| Topology awareness | “rack-aware” | 根据 NVLink/IB/rack placement 放置 pods |
| `DCGM_FI_DEV_GPU_UTIL` | “GPU utilization” | Duty-cycle metric；不是 LLMs 的 scaling signal |
| Queue depth | “waiting requests” | 对 prefill-bound scaling 正确的 HPA signal |
| KV cache utilization | “memory pressure” | 对 decode-bound scaling 正确的 HPA signal |
| Consolidation | “Karpenter consolidation” | 为了换到更便宜 instance type 而终止 node |
| `WhenEmpty + 1h` | “safe consolidation” | 不驱逐 running GPU jobs 的 policy |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — design docs 和 configuration examples。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — consolidation policy semantics 和 GPU-safe defaults。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner scaling signals。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray integration pattern。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — managed-Kubernetes-specific guidance。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler design。
