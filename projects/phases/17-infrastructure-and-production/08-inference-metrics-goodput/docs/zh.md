# 推理指标：TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一个 inference deployment 是否真正可用。TTFT 是 prefill 加 queue 加 network。TPOT（等价于 ITL）是 memory-bound decode 每个 token 的成本。End-to-end latency 是 TTFT 加上 TPOT 乘以输出长度。Throughput 是整个 fleet 聚合后的 tokens per second。但对产品真正重要的是 goodput——同时满足所有 SLO 的请求比例。低 goodput 下的高 throughput，意味着你正在处理最终没有及时抵达用户的 tokens。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数字：mean TTFT 162 ms、mean TPOT 7.33 ms、mean E2E 1,093 ms。永远报告 P50、P90、P99，不要只报告 mean。还要注意 measurement trap：GenAI-Perf 在计算 ITL 时排除 TTFT，LLMPerf 会包含它；同一次 run，两个工具会给出不同 TPOT。

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、throughput 和 goodput，并说出每个指标测量的组件。
- 解释为什么 mean 是 LLM serving 的错误统计量，以及如何阅读 P50/P90/P99。
- 构造一个 SLO multi-constraint（例如 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s），并计算对应 goodput。
- 说出两个会在同一次 run 上给出不同 TPOT 的 benchmark tools，并解释原因。

## 问题

“我们的 throughput 是 15,000 tokens per second。”那又怎样？如果 40% 请求都超过了 2 秒 end-to-end，用户已经离开 session。Throughput alone 不能说明产品是否可用。

Inference 有多条 latency axes，而且每条失败方式都不同。Prefill 是 compute-bound，随 prompt length 扩展。Decode 是 memory-bound，随 batch size 扩展。Queuing delay 是运营问题。Network 是物理距离问题。你需要为每个部分设置不同指标，需要 percentiles，还需要一个单一 composite 来回答“用户是否得到了预期体验”——这就是 goodput。

## 概念

### TTFT — time to first token

`TTFT = queue_time + network_request + prefill_time`

Prompt 很长时，prefill 会主导 TTFT。在 H100 上运行 Llama-3.3-70B FP8 时，32k prompt 需要约 800 ms 纯 prefill 时间。Queue time 是 scheduler 在负载下的行为。Network request 是包括 TLS 在内的 wire time。TTFT 是用户看到任何 streaming response 之前感知到的 latency。

### TPOT / ITL — inter-token latency

一个量有很多名字。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token`——它们本质上相同。它表示第一个 token 之后，相邻 streamed tokens 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一套 Llama-3.3-70B H100 stack 上，启用 chunked prefill 时 TPOT mean 约 7 ms。没有 chunked prefill 时，如果 neighboring sequence 正在做长 prefill，TPOT 可能 spike 到 50 ms。看 P99，不要只看 mean。

### E2E latency

`E2E = TTFT + TPOT * output_tokens + network_response`

对长输出（>500 tokens），E2E 由 TPOT 主导。对长 prompt、短输出，E2E 由 TTFT 主导。报告 E2E 时要按 output length 条件化。

### Throughput

`throughput = total_output_tokens / elapsed_time`

这是聚合指标。它告诉你 fleet efficiency，但不能告诉你单个 request 的健康状况。

### Goodput — 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是一个 multi-constraint。只有每个 constraint 都满足时，一个 request 才算 “good”。Goodput 就是这类请求的比例。60% goodput 下的高 throughput 是失败。99% goodput 下的较低 throughput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 submissions 和 AI platform providers 内部 SLA tracking 中使用的指标。

### 为什么 mean 是错误统计量

LLM latency distributions 是右偏的。一个 decode batch 中，如果有一个 long-prefill neighbor，可以有 500 个 tokens 的 TPOT 约 7 ms，同时有 20 个 tokens 的 TPOT 约 60 ms。Mean TPOT 是 9 ms。P99 TPOT 是 65 ms。用户会经常撞到 P99，这就是他们离开的原因。

永远报告三元组（P50、P90、P99）。对用户体验来说，P99 才是你要优化的指标。

### 参考数字 — TRT-LLM 上的 Llama-3.1-8B-Instruct，2026

- mean TTFT：162 ms
- mean TPOT：7.33 ms
- mean E2E：1,093 ms
- P99 TPOT：取决于 chunked-prefill configuration，约 10-25 ms。

这些是 NVIDIA 发布的 reference points。它们会随 model size（70B 会高 3-5 倍）、hardware（H100 vs B200 约 3 倍）和 load 改变。

### Measurement trap

2026 年两个最常用的 benchmark tools，会在同一次 run 上给出不同 TPOT：

- **NVIDIA GenAI-Perf**：在 ITL calculation 中排除 TTFT。ITL 从 token 2 开始。
- **LLMPerf**：包含 TTFT。ITL 从 token 1 开始。

对于一个 TTFT 500 ms、100 output tokens、total decode 700 ms 的请求，GenAI-Perf 会报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 会报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变数字。

永远说明使用哪个工具。永远公布定义。

### 构造 SLO

一个 2026 年面向消费者的 70B chat model，可以使用这样的 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- <300-token outputs 的 E2E P99 <= 3 s。
- Goodput target >= 99%。

Enterprise SLOs 会收紧 TTFT（200-400 ms）并放宽 E2E。关键是写下来，三个指标都测，并把 goodput 作为单一 composite 来追踪。

### 如何测量

- 运行真实流量，或 realistic synthetic（例如 LLMPerf with `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- Benchmark run 目标设为 2x peak concurrency。
- 运行 30-50 iterations，对合并样本取 percentiles。
- 发布时写明 tool name、tool version、model、hardware、concurrency、prompt distribution。

```figure
throughput-latency
```

## Use It

`code/main.py` 是一个 toy goodput calculator。它会生成 synthetic latency distribution，应用 SLO，并计算 goodput。它也会在同一条 trace 上展示 GenAI-Perf 与 LLMPerf 的 TPOT 差异。

## Ship It

本课产出 `outputs/skill-slo-goodput-gate.md`。给定 workload 和 SLO，它会生成一个可用于 CI/CD 的 benchmark recipe，以 goodput 而不是 throughput 来 gate deploys。

## 练习

1. 运行 `code/main.py`。生成一个带 1% tail spike 的 distribution。当你把 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 某供应商报价：“Llama 3.3 70B H100 上 15,000 tok/s。”信任它之前，你要问哪三个问题？
3. 为什么 chunked prefill 会保护 P99 TPOT，但不保护 mean TPOT？
4. 为 voice assistant 构造一个 consumer SLO（first token 是被听见，而不是被读到）。哪个指标最能被用户感知？
5. 阅读 LLMPerf README 和 GenAI-Perf docs。找出还有哪三个 metrics 上这些工具定义不同。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| TTFT | “time to first token” | Queue + network + prefill；长 prompts 下由 prefill 主导 |
| TPOT | “time per output token” | 第一个 token 之后，每个 token 的 memory-bound decode 成本 |
| ITL | “inter-token latency” | 多数工具中与 TPOT 相同（并非全部，见 GenAI-Perf） |
| E2E | “end to end” | TTFT + TPOT * output_len；再叠加 response-side network |
| Throughput | “tok/s” | Fleet efficiency；没有 latency percentiles 就没什么用 |
| Goodput | “SLO-met rate” | 同时满足所有 SLO constraints 的请求比例 |
| P99 | “tail” | 1-in-100 worst-case latency；用户体验指标 |
| SLO multi-constraint | “the joint” | 三个 latency bounds 的 AND；任意一个违反，请求就失败 |
| GenAI-Perf vs LLMPerf | “the tool trap” | 工具对 ITL 是否包含 TTFT 意见不同 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)：TTFT、ITL、TPOT 的 canonical definition。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)：alternative definitions 和 measurement recipe。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)：真实 deployments 上的 applied measurement。
- [LLMPerf](https://github.com/ray-project/llmperf)：基于 Ray 的 open-source benchmark。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html)：NVIDIA benchmark tool。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)：行业认可的 goodput-based benchmark。
