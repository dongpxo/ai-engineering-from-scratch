# Capstone 14 — Speculative-Decoding Inference Server

> vLLM 0.7 中的 EAGLE-3 在真实流量上提供 2.5-3x throughput。P-EAGLE（AWS 2026）进一步推进了 parallel speculation。SGLang 的 SpecForge 可以规模化训练 draft heads。Red Hat 的 Speculators hub 为常见 open models 发布了 aligned drafts。TensorRT-LLM 让 speculative decoding 成为 NVIDIA 上的一等能力。2026 年 production serving stack 是 vLLM 或 SGLang，配 EAGLE-family drafts、FP8 或 INT4 quantization，并基于 queue-wait 做 HPA。本 capstone 的目标是让两个 open models 在 baseline throughput 之上达到 2.5x+，并产出完整 tail-latency report。

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 hours

## 问题

Speculative decoding 在 2026 年已经 commodity 化。EAGLE-3 draft heads 在 target model 的 hidden states 上训练，并提前预测 N 个 tokens；target model 在一次 pass 中验证这些 tokens。60-80% 的 acceptance rates 会转化为 2-3x end-to-end throughput。vLLM 0.7 原生集成了这一能力。SGLang + SpecForge 提供训练 pipeline。Red Hat 的 Speculators 发布了 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的 aligned drafts。

Craft 在 serving operations，而不是 model 本身。Acceptance rate 会随 traffic distribution 漂移（ShareGPT vs code vs domain data）。Rejection 下的 tail latency 比不使用 speculation 更差——你必须报告多个 batch sizes 下的 p99，而不只是 steady-state tokens/sec。与 Anthropic / OpenAI API 相比的 cost per 1M tokens 是可信度杠杆。

## 概念

Speculative decoding 有两层。**Draft** model（EAGLE-3 head、ngram 或更小的 target-aligned model）每一步提出 k 个 candidate tokens。**Target** model 在一次 pass 中验证全部 k 个 tokens；任何 accepted prefix 都会替代 greedy path。Acceptance rate 取决于 draft-target alignment 和 input distribution。

EAGLE-3 在大多数 traffic 上优于 ngram drafts。P-EAGLE 使用 parallel speculation 构建更深的 draft trees。Trade-off：rejection 时 P99 latency 更高，因为 verify pass 更大。Serving config 必须报告按 batch size 分桶的 latency，才能暴露这个问题。

Deployment 使用 Kubernetes。vLLM 0.7 每个 GPU 或 tensor-parallel shard 运行一个 replica。HPA 根据 queue-wait 而不是 CPU autoscale。FP8（Marlin）和 INT4（AWQ）quants 让 GPU memory 保持在 H100 / H200 envelope 内。End-to-end report 包含 throughput、acceptance rate、batch 1/8/32 的 p50/p99，以及 $/1M tokens。

## 架构

```text
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- Serving：vLLM 0.7 或 SGLang 0.4
- Speculative methods：EAGLE-3 draft heads、P-EAGLE parallel speculation、ngram fallback
- Draft training：SpecForge（SGLang）或 Red Hat Speculators
- Target models：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- Quantization：FP8（Marlin）、INT4 AWQ
- Deployment：Kubernetes + NVIDIA device plugin；基于 queue-wait metric 的 HPA
- Eval：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于 domain-spread acceptance measurement
- Reference：TensorRT-LLM speculative decoding，作为 vendor baseline

## Build It

1. **Target model prep。** 选择 Llama 3.3 70B。通过 Marlin quantize 到 FP8。在 vLLM 0.7 下部署到 1xH100（或 2x tensor-parallel）。

2. **Draft source。** 从 Red Hat Speculators 拉取 aligned EAGLE-3 draft head（或通过 SpecForge 训练一个）。加载到 vLLM 的 speculative-decoding config。

3. **Baseline numbers。** 在 speculation 之前测量：batch 1/8/32 下的 tokens/s、p50/p99 latency、GPU utilization。发布结果。

4. **Enable EAGLE-3。** 切换 config；重新运行同一 benchmark。报告 speedup、acceptance rate、p99 tail-latency delta。

5. **P-EAGLE。** 启用 parallel speculation；比较 deeper draft tree 与 serial EAGLE-3。报告 P-EAGLE 从有帮助到有害的 inflection point。

6. **Domain traffic。** 用同一 server 跑 ShareGPT vs HumanEval vs domain-specific traffic。测量每种 distribution 的 acceptance rate。识别 drafts 何时 drift。

7. **Second target model。** 在 Qwen3-Coder-30B MoE 上跑同一 pipeline。Draft 更棘手（MoE routing noise）。报告结果。

8. **K8s HPA。** 在 K8s 下部署，并让 HPA 追踪 `queue_wait_ms`。演示 load 增至三倍时的 scale-out。

9. **Cost comparison。** 在同一 eval 上计算 $/1M tokens，并与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 对比。发布结果。

## Use It

```console
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Ship It

`outputs/skill-inference-server.md` 描述 deliverable：一个 measured serving stack，包含 speculative decoding、完整 benchmark report 和 K8s deployment。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Measured speedup vs baseline | 在两个 models 上以 matched quality 达到 2.5x+ throughput |
| 20 | Acceptance rate on realistic traffic | Per-distribution acceptance-rate report |
| 20 | P99 tail-latency discipline | batch 1/8/32 下，开启和关闭 speculation 的 p99 |
| 20 | Ops | K8s deploy、基于 queue-wait 的 HPA、rollout smooth |
| 15 | Write-up and methodology | 清楚解释 changed what and why |
| **100** | | |

## 练习

1. 当 draft 比 target 落后一个版本时（例如 Llama 3.3 -> 3.4 drift），测量 acceptance-rate degradation。构建 monitoring alert。

2. 实现 ngram-fallback：如果 EAGLE-3 acceptance 低于某个 threshold，则切换到 ngram drafts。报告 reliability improvement。

3. 运行一个受控 MoE experiment：同一个 Qwen3-Coder-30B，分别注入 routing noise 和不注入。测量 draft acceptance sensitivity。

4. 扩展到 H200（141 GB）。报告 model-size-per-replica headroom 增加多少，以及是否可以 serve 未 quantized 的 Llama 3.3 70B。

5. 在同一 H100 hardware 上 benchmark TensorRT-LLM speculative decoding。报告它在哪些方面胜过 vLLM。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Draft model | “Speculator” | 提出 N 个 tokens 供 target 验证的小模型 |
| EAGLE-3 | “2026 draft architecture” | 在 target hidden states 上训练的 draft head；约 75% acceptance |
| P-EAGLE | “Parallel speculation” | 在一次 target pass 中验证的 draft branches tree |
| Acceptance rate | “Hit rate” | Drafted tokens 中无需 resampling 即被接受的比例 |
| Quantization | “FP8 / INT4” | 低精度 weights，用于在 GPU memory 中容纳更多模型 |
| Queue wait | “HPA metric” | Request 在 inference 开始前排队等待的时间 |
| Speculators hub | “Aligned drafts” | Red Hat Neural Magic 为常见 open models 提供的 EAGLE drafts hub |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — reference serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — parallel speculative decoding paper + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head training pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — method paper
- [vLLM repository](https://github.com/vllm-project/vllm) — code and benchmarks
