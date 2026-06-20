# Blackwell 上的 TensorRT-LLM：FP8 与 NVFP4

> TensorRT-LLM 只属于 NVIDIA 生态，但它在 Blackwell 上赢得很明显。在带 Dynamo orchestration 的 GB200 NVL72 上，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得一个 120B model 的成本为每百万 tokens $0.012；对比 H100 + vLLM 的 $0.09/M，经济差距达到 7x。这个 stack 是三个 floating-point regimes 的复合：FP8 对 KV cache 和 attention kernels 仍然关键，因为它们需要足够的 dynamic range；NVFP4（4-bit microscaling）处理 weights 和 activations；multi-token prediction（MTP）和 disaggregated prefill/decode 又在上面叠加 2-3x。Day-0 model support 可以直接加载 FP4 weights，不需要 post-training conversion。2026 年工程团队要面对的代价是：TRT-LLM 是闭源 NVIDIA stack，所以采用它意味着用 portability 换 throughput。在承诺迁移前，先根据你的 models 和 hardware mix 把账算清楚。

**Type:** Learn
**Languages:** Python（stdlib，toy FP8/NVFP4 memory and cost calculator）
**Prerequisites:** Phase 17 · 04（vLLM Serving Internals），Phase 10 · 13（Quantization）
**Time:** ~75 minutes

## 学习目标

- 解释为什么即使 weights 使用 NVFP4，FP8 对 KV cache 和 attention 仍然关键。
- 计算 frontier model 在 BF16、FP8 和 NVFP4 下的 HBM footprint，并推理节省来自哪里。
- 说出 TRT-LLM 利用的 Blackwell-specific features（day-0 FP4、MTP、disaggregated serving、all-to-all primitives）。
- 判断什么时候 TRT-LLM 的 NVIDIA-lock 值得用来换取相对 Hopper 上 vLLM 的 7x cost gap。

## 问题

2026 年 inference economics 的前沿问题是“每一美元能产出多少 tokens”。答案取决于四个叠加选择：hardware generation（Hopper H100/H200 vs Blackwell B200/GB200）、precision（BF16 → FP8 → NVFP4）、serving engine（vLLM vs SGLang vs TRT-LLM）和 orchestration（plain vs disaggregated vs Dynamo）。

在 Hopper + vLLM 上，一个 120B MoE 的运行成本约为每百万 tokens $0.09。在 Blackwell + TRT-LLM + Dynamo 上，同一模型约为 $0.012——便宜 7x。这部分差距来自 hardware（Blackwell 的 per-GPU LLM throughput 是 Hopper 的 11-15x），也来自 stack：FP4 weights、MTP draft、disaggregated prefill/decode，以及用于 MoE expert communication 的 NVLink 5 all-to-all。

你无法在 NVIDIA stack 之外复制这一点。这就是 tradeoff：用 portability 换 economics。理解哪个 stack choice 贡献了哪部分差距，就是本课的重点。

## 概念

### 为什么 FP8 仍然是 KV cache 的下限

2026 年的一个常见错误是：假设 NVFP4 可以用在所有地方。它不行。KV cache 需要 FP8（8-bit floating point），因为它存储的 attention keys 和 values 横跨很宽的 dynamic range。把 KV 量化到 FP4 会造成灾难性精度损失——分布尾部掉落，attention scores 崩塌。FP8 的 exponent bits 给了 KV cache 所需的 range。

NVFP4（2025-2026）适用于 weights 和 activations。Microscaling 的意思是：每个 weights block 有自己的 scale factor，所以小 blocks 可以覆盖不同 dynamic ranges，而不会遭遇 per-tensor scale loss。对 activations 来说，FP4 能撑住，是因为一层内部的 activations range 较小。

典型 Blackwell 配置：

- Weights：NVFP4（4-bit microscaling）。
- Activations：NVFP4。
- KV cache：FP8。
- Attention accumulator：FP32（softmax stability）。

### TRT-LLM 使用的 Blackwell-specific primitives

- **Day-0 FP4 weights**：model providers 直接发布 FP4 weights；TRT-LLM 无需 post-training conversion 就能加载。不需要为了 FP4 再跑 AWQ / GPTQ step。
- **Multi-token prediction（MTP）**：思想和 EAGLE（Phase 17 · 05）相同，但集成在 TRT-LLM build 中。
- **Disaggregated serving**：prefill 和 decode 位于不同 GPU pools，KV cache 通过 NVLink 或 InfiniBand 传输。思想和 Dynamo（Phase 17 · 20）相同。
- **All-to-all communication primitives**：NVLink 5 把 MoE expert communication latency 相比 Hopper 降低 3x。TRT-LLM 的 MoE kernels 为此调优。
- **NVFP4 + MXFP8 microscaling**：Blackwell Tensor Cores 对 scale-factor handling 做了 hardware acceleration。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上达到 $0.02/M tokens。
- GB200 NVL72 通过 Dynamo（orchestrating TRT-LLM）达到 $0.012/M tokens。
- H100 + vLLM 在可比 workload 上约为 $0.09/M tokens。
- TRT-LLM updates 在三个月内带来 2.8x throughput gain（2026）。
- Blackwell vs Hopper：per-GPU LLM throughput 达到 11-15x。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 统治所有 submitted tasks。

### FP4 在 quality 上到底付出什么代价

NVFP4 很激进。在 reasoning-heavy workloads（chain-of-thought、math、long-context code-gen）上，FP4 weights 会产生可见退化。Per-block calibration 可以缓解，但不能消除。发货 reasoning models 的团队常用 FP8 weights + FP4 activations 作为折中，或者继续在 H200 上全程使用 FP8。

规则：在承诺使用 NVFP4 weights 前，永远先在自己的 eval set 上验证 task quality。

### 为什么这是一个 NVIDIA-lock 决策

TRT-LLM 是 C++ + CUDA + closed-source kernels。Models 需要为具体 GPU SKU 编译。没有 AMD，没有 Intel，没有 ARM。如果你的 infra strategy 是 multi-vendor，那么 TRT-LLM-served tier 无法采用 TRT-LLM——你仍然可以在 mixed hardware 上用 vLLM serving。如果你已经是 NVIDIA-only，7x gap 就足以支付 lock-in 的代价。

### 2026 practical recipe

如果年度 inference bill 超过 $100M，继续运行 Hopper + vLLM 等于把 7-10x 留在桌上。把 cost-dominant workloads 迁移到 Blackwell + TRT-LLM + Dynamo。把 experimentation tier 留在 H100 + vLLM 上，以获得 model iteration speed。每个 NVFP4-converted model 进 production 前都要验证 quality。

### Disaggregation bonus

TRT-LLM 的 disaggregated serving（分离 prefill 和 decode pools）会在 Phase 17 · 20 深入讲解。在 Blackwell 上，multiplier 会叠加：FP4 weights × MTP speedup × disaggregated placement × cache-aware routing。7x 数字假设使用的是这个 full stack。

```figure
pipeline-parallel
```

## Use It

`code/main.py` 会为一个 model 在三种 stacks 下计算 HBM footprint、decode throughput（memory-bound regime）和 $/M-tokens：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它，观察 compounding effect，以及每个变化贡献了 gap 的哪一部分。

## Ship It

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定 workload、model size 和 annual token volume，它会判断 Blackwell + TRT-LLM stack 是否值得 NVIDIA-lock。

## 练习

1. 运行 `code/main.py`。对一个 active parameters 为 30% 的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上的 memory-bandwidth-limited decode throughput。最大跃迁来自哪里？
2. 一个客户每年在 H100 + vLLM 上花 $2M。考虑 7x economic gap，如果要在 12 个月内摊销迁移到 TRT-LLM 的成本，他们需要购买多少 Blackwell GPUs 才 break even？
3. NVFP4 weight conversion 后，你在 MATH 上看到 accuracy 下降 3 points。说出两条 recovery paths：一条 quality-first（保留 FP8 weights），一条 cost-first（用 in-domain data calibration）。
4. 阅读 MLPerf v6.0 inference results。哪项 task 的 Blackwell-over-Hopper gap 最小？为什么？
5. 计算一个 405B model 在 NVFP4 weights + FP8 KV cache、128k context 下所需的 HBM。它能放进单个 GB200 NVL72 node 吗？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| FP8 | “eight-bit float” | 8-bit floating point；由于 dynamic range，用于 KV cache 和 attention |
| NVFP4 | “four-bit micro” | NVIDIA 的 4-bit microscaling FP format；Blackwell 上用于 weights 和 activations |
| MXFP8 | “MX eight” | Microscaling FP8 variant；Blackwell Tensor Cores 上有 hardware acceleration |
| Day-0 FP4 | “ship FP4 weights” | Model providers 直接发布 FP4 weights；不需要 post-train conversion step |
| MTP | “multi-token prediction” | TRT-LLM 集成的 speculative-decoding draft（Phase 17 · 05） |
| Disaggregated serving | “split prefill/decode” | Prefill 和 decode 位于不同 GPU pools；KV 通过 NVLink/IB 传输 |
| All-to-all | “MoE expert comm” | 把 tokens route 到 expert GPUs 的 communication pattern；NVLink 5 降低 3x |
| InferenceX | “SemiAnalysis inference bench” | 2026 年行业接受的 cost-per-token benchmark |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf results。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 和 MoE kernels。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — official engine documentation。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的 disaggregated orchestration。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell numbers 的 benchmark suite。
