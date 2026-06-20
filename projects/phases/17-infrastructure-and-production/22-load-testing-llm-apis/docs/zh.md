# LLM API 负载测试：为什么 k6 和 Locust 会骗人

> 传统 load testers 不是为 streaming responses、可变 output lengths、token-level metrics 或 GPU saturation 设计的。大多数团队会踩两个坑。GIL trap：Locust 的 token-level measurement 会在 Python GIL 下运行 tokenization；高并发时，它会和 request generation 争抢执行权，tokenization backlog 随后会抬高报告出来的 inter-token latency——瓶颈在 client，不在 server。Prompt-uniformity trap：循环里使用相同 prompt 只测试 token distribution 上的一个点；真实 traffic 有可变长度和多样 prefix matches。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复这个问题。2026 年工具映射：LLM-specialized（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）适合 token-level accuracy；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** 支持 streaming-aware、Kubernetes-native distributed（通过 TestRun/PrivateLoadZone CRDs），最适合 CI/CD gates；Vegeta 适合 Go constant-rate saturation；Locust 2.43.3 只有搭配 LLM-Locust extension 才适合 streaming。Load patterns：steady-state、ramp、spike（autoscaling test）、soak（memory leaks）。

**Type:** Build
**Languages:** Python（stdlib，玩具版 realistic-prompt generator + latency collector）
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## 学习目标

- 解释两个会让通用 load testers 在 LLM APIs 上说谎的 anti-patterns：GIL trap 和 prompt-uniformity trap。
- 根据目的选择工具：LLMPerf（benchmark run）、k6 + streaming extension（CI gate）、guidellm（large-scale synthetic）、GenAI-Perf（NVIDIA reference）。
- 设计四种 load patterns（steady、ramp、spike、soak），并说出每种 pattern 捕获的 failure mode。
- 使用 input tokens 的 mean + stddev 构造 realistic prompt distribution，而不是固定长度。

## 问题

你用 k6 对 LLM endpoint 做了 500 concurrent users 的测试。它撑住了。你发布上线。结果生产环境里只有 200 个真实 users，服务就倒了——P99 TTFT 爆炸，GPUs 打满。

发生了两件事。第一，k6 发送了 500 个完全相同的 prompts——你的 request coalescing 和 prefix caching 让它看起来像是在处理 500 个 concurrent decodes，但实际上只是在处理一个。第二，k6 不会按照人眼感知 streaming response 的方式追踪 inter-token latency；它看到的是一条 HTTP connection，而不是 500 个 tokens 以不同间隔到达。

LLM 负载测试是一门独立的学科。

## 概念

### GIL trap（Locust）

Locust 使用 Python，并在 client-side、GIL 下运行 tokenization。在高并发下，tokenizer 会排在 request generation 后面。报告出来的 inter-token latency 包含了 client-side tokenization backlog。你以为是 server 慢；其实是 test harness 慢。

修复方式：LLM-Locust extension 把 tokenization 移到独立 processes，或者使用 compiled-language harness（k6、使用 tokenizers.rs 的 LLMPerf）。

### Prompt-uniformity trap

所有已知 load testers 都允许你配置一个 prompt。在 10,000 次 iterations 的 loop test 中，每次发送的都是同一个 prompt。Server 每次看到同一个 prefix——prefix cache hits 接近 100%，throughput 看起来非常好。

修复方式：从 prompt distribution 中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——长度多样，内容也多样。

### 四种 load patterns

1. **Steady-state**：以固定 RPS 运行 30-60 分钟。捕获：baseline performance regressions。
2. **Ramp**：在 15 分钟内把 RPS 从 0 线性提升到 target。捕获：capacity breakpoint、warm-up anomalies。
3. **Spike**：RPS 突然升高 3-10 倍，持续 2 分钟后回落。捕获：autoscaling latency、queue saturation、cold-start impact。
4. **Soak**：steady-state 运行 4-8 小时。捕获：memory leaks、connection-pool drift、observability overflow。

### 2026 工具映射

**LLMPerf**（Anyscale）：Python，但 tokenization 由 Rust 支撑。支持 mean/stddev prompts。Streaming-aware。性能测试的最佳默认选择。

**NVIDIA GenAI-Perf**：NVIDIA reference。使用 Triton client；metrics coverage 很完整。注意它的 ITL 不包含 TTFT；LLMPerf 的 ITL 包含 TTFT。同一个 server 上，两个工具会产生不同的 TPOT。

**LLM-Locust**（TrueFoundry）：修复 GIL trap 的 Locust extension。熟悉的 Locust DSL + streaming metrics。

**guidellm**：large-scale synthetic benchmarking。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：

- k6 本身是 Go、compiled、没有 GIL，并且加入了 streaming-aware metrics。
- k6 Operator 使用 TestRun / PrivateLoadZone CRDs 实现 Kubernetes-native distributed testing。
- 最适合 CI/CD gates 和 SLA testing。

**Vegeta**：Go，比 k6 更简单。Constant-rate HTTP saturation。不懂 LLM，但很适合 gateway / rate-limit testing。

**Locust 2.43.3 stock**：对 LLM 有 GIL trap。只有搭配 LLM-Locust extension 才适合。

### CI 中的 SLA gate

在 PR 上运行 k6：

- 每个 baseline RPS 下运行 30-50 iterations。
- Gate：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 超过阈值就 break build。

### Realistic prompt distribution

从真实 traffic samples 构造（如果你有），或从公开 distributions 构造（例如用于 chat 的 ShareGPT prompts、用于 code 的 HumanEval）。把 mean + stddev 传给 LLMPerf。无论如何都要避免 loop-with-one-prompt。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：streaming-aware metrics。
- 典型 LLMPerf run：concurrency X 下 100-1000 requests。
- 典型 CI gate：每个 PR 30-50 iterations。
- 四种 patterns：steady、ramp、spike、soak。

## Use It

`code/main.py` 会模拟一个带 realistic prompt distribution 的 load test，测量 effective TPOT，并展示 uniform-prompt trap。

## Ship It

本课产出 `outputs/skill-load-test-plan.md`。给定 workload 和 SLA，它会选择工具并设计四种 load patterns。

## 练习

1. 运行 `code/main.py`。比较 uniform 和 realistic distribution——差距在哪里？
2. 为 CI gate 写一个 k6 script：在 100 concurrent、运行 5 分钟时，TTFT P95 < 800 ms。
3. 你的 soak test 显示 memory 每小时增长 50 MB。列出三个原因，以及用于区分它们的 instrumentation。
4. Spike test 从 10 RPS 跳到 100 RPS。如果已经部署 Karpenter + vLLM production-stack（Phase 17 · 03 + 18），预期 recovery time 是多少？
5. 同一个 server 上，GenAI-Perf 报告 TPOT=6ms；LLMPerf 报告 TPOT=11ms。解释原因。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| LLMPerf | “the LLM harness” | Anyscale benchmark tool，streaming-aware。 |
| GenAI-Perf | “NVIDIA tool” | NVIDIA reference harness。 |
| LLM-Locust | “Locust for LLMs” | 修复 GIL trap 的 Locust extension。 |
| guidellm | “synthetic benchmark” | Large-scale synthetic tool。 |
| k6 Operator | “K8s k6” | 基于 CRD 的 distributed k6。 |
| GIL trap | “Python client overhead” | Tokenization backlog 抬高报告出来的 latency。 |
| Prompt-uniformity trap | “single-prompt lie” | 使用同一个 prompt 循环，命中 cache 并抬高 throughput。 |
| Steady-state | “constant load” | 持续 N 分钟的平坦 RPS。 |
| Ramp | “linear up” | 在一段时间内从 0 提升到 target。 |
| Spike | “burst test” | 突然乘倍放大，然后恢复。 |
| Soak | “long test” | 用数小时检测 leaks。 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
