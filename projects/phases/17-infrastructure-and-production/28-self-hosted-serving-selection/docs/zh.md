# 自托管 Serving 选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年，自托管 inference 主要由四类 engines 主导。选择时看 hardware、scale 和 ecosystem。**llama.cpp** 在 CPU 上最快 — model support 最广，对 quantization 和 threading 有完全控制。**Ollama** 是 dev laptop 上的一条命令安装方案，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP serialization），在类似 production 的负载下 throughput gap 可到 3x。**TGI 在 2025 年 12 月 11 日进入 maintenance mode** — 只做 bug fixes，raw throughput 比 vLLM 慢约 10%，但历史上有顶级 observability 和 HF-ecosystem integration。这个 maintenance status 让它成为 risky long-term bet — 新项目默认选 SGLang 或 vLLM 更安全。**vLLM** 是 general-purpose production default — v0.15.1（2026 年 2 月）加入 PyTorch 2.10、RTX Blackwell SM120、H200 optimization。**SGLang** 是 agentic multi-turn / prefix-heavy specialist — 生产中已有 400,000+ GPUs（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。Hardware constraints：CPU-only → 只能选 llama.cpp。AMD / non-NVIDIA → 只能选 vLLM（TRT-LLM 被 NVIDIA lock）。2026 pipeline pattern：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。全程使用同一份 GGUF/HF weights。

**Type:** Learn
**Languages:** Python（stdlib，engine-decision tree walker）
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## 学习目标

- 根据 hardware（CPU / AMD / NVIDIA Hopper / Blackwell）、scale（1 user / 100 / 10,000）和 workload（general chat / agent / long-context）选择 engine。
- 说出 2026 年 TGI maintenance-mode status（2025 年 12 月 11 日），以及它为什么会让新项目偏向 vLLM 或 SGLang。
- 描述 dev/staging/prod pipeline：全程使用同一份 GGUF 或 HF weights。
- 解释为什么 “CPU only” 会强制选择 llama.cpp，而 “AMD” 会排除 TRT-LLM。

## 问题

你的团队开始一个新的 self-hosted LLM project。一个工程师说用 Ollama，另一个说用 vLLM，第三个说 “TGI 不是 out of the box 就能跑吗？” 三个人在各自语境下都对。没有一个答案适用于所有情况。

2026 年，choice tree 很重要：hardware first，scale second，workload third。还有一个具体的 2025 年事件 — TGI 在 12 月 11 日进入 maintenance mode — 改变了新项目的默认选型。

## 概念

### 五个 engines

| Engine | Best for | Notes |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / minimal deps / 最广 model support | CPU 上最快，完全控制 |
| **Ollama** | Dev laptops、single user、一条命令安装 | 比 llama.cpp 慢 15-30%；prod throughput gap 可到 3x |
| **TGI** | HF ecosystem、regulated industries | **2025 年 12 月 11 日进入 maintenance mode** |
| **vLLM** | General-purpose production、100+ users | 广泛的 production default；v0.15.1 2026 年 2 月 |
| **SGLang** | Agentic multi-turn、prefix-heavy workloads | 生产中 400,000+ GPUs |

### Hardware-first decision

**CPU only** → llama.cpp。Ollama 也能跑，但更慢。其他 engines 在 CPU 上都不具备竞争力。

**AMD GPU** → vLLM（AMD ROCm support）。SGLang 也能跑。TRT-LLM 是 NVIDIA-locked，所以排除。

**NVIDIA Hopper (H100 / H200)** → vLLM、SGLang 或 TRT-LLM。三者都是 top-tier。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM 是 throughput leader（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon (M-series)** → llama.cpp（Metal）。Ollama 封装了这条路径。

### Scale-second decision

**1 user / local dev** → Ollama。一条命令，几秒内 first-token。

**10-100 users / small team** → vLLM single-GPU。

**100-10k users / production** → vLLM production-stack（Phase 17 · 18）或 SGLang。

**10k+ users / enterprise** → vLLM production-stack + disaggregated（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### Workload-third decision

**General chat / Q&A** → vLLM 是最稳妥的 broad default。

**Agentic multi-turn（tools、planning、memory）** → SGLang 的 RadixAttention（Phase 17 · 06）占优。

**RAG with heavy prefix reuse** → SGLang。

**Code generation** → vLLM 可以；SGLang 在 cache 上略好。

**Long context (128K+)** → vLLM + chunked prefill；SGLang + tiered KV。

### TGI maintenance trap

Hugging Face TGI 在 2025 年 12 月 11 日进入 maintenance mode — 之后只做 bug fixes。历史上：顶级 observability、best-in-class HF-ecosystem integration（model cards、safety tools），raw throughput 略落后 vLLM。

2026 年的新项目：默认避开 TGI。已有 TGI deployments 可以继续运行，但最终应该迁移。SGLang 和 vLLM 是更安全的 defaults。

### Pipeline pattern

Dev（Ollama）→ staging（llama.cpp）→ prod（vLLM）。全程使用同一份 GGUF 或 HF weights。工程师在 laptops 上快速迭代；staging mirror production quantization；prod 是 serving target。

### Ollama caveat

Ollama 很适合 dev。它不适合 shared production：Go HTTP serialization 会增加 overhead，concurrency management 比 vLLM 简单，OpenTelemetry support 落后。把 Ollama 用在它擅长的地方 — one user、one command — 然后 shared 场景切到 vLLM。

### Self-hosted vs managed 是另一个决策

Phase 17 · 01（managed hyperscalers）、· 02（inference platforms）覆盖 managed。本课假设你已经决定 self-host。选择 self-host 的理由包括：data residency、custom fine-tune、scale 下的 total cost ownership、hosted 平台没有的 domain model。

### 你应该记住的数字

- TGI maintenance mode：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 support。
- SGLang production footprint：400,000+ GPUs。
- Ollama throughput gap vs llama.cpp：慢 15-30%；prod load 下 gap 可到 3x。

```figure
data-parallel
```

## Use It

`code/main.py` 是 decision-tree walker：给定 hardware + scale + workload，它会选择一个 engine 并解释原因。

## Ship It

本课产出 `outputs/skill-engine-picker.md`。给定 constraints，它会选择 engine 并写出 migration plan。

## 练习

1. 用你的 hardware / scale / workload 运行 `code/main.py`。输出是否符合你的直觉？
2. 你的 infra 是 12 张 H100 和 8 张 MI300X AMD。选什么 engine？为什么 TRT-LLM 不在候选里？
3. 一个团队想在 2026 年使用 TGI，因为 “it's what we know”。论证 migration case。
4. Ollama dev 到 vLLM prod：quantization、configuration 和 observability 会发生什么变化？
5. RAG product 的 P99 prefix length 是 8K，并且 tenants 之间有高 reuse。选择一个 engine，并和 Phase 17 · 11 + 18 组合成 stack。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| llama.cpp | “the CPU one” | Model support 最广，CPU 上最快。 |
| Ollama | “the laptop one” | 一条命令安装，dev-grade throughput。 |
| TGI | “HF's serving” | 自 2025 年 12 月起处于 maintenance mode。 |
| vLLM | “the default” | 2026 年的广泛 production baseline。 |
| SGLang | “the agentic one” | Prefix-heavy，RadixAttention。 |
| TRT-LLM | “NVIDIA-locked” | Blackwell throughput leader，只支持 NVIDIA。 |
| GGUF | “llama.cpp format” | Bundled K-quant variants。 |
| Production-stack | “vLLM K8s” | Phase 17 · 18 的 reference deployment。 |
| Pipeline pattern | “dev→stage→prod” | 在同一份 weights 上使用 Ollama → llama.cpp → vLLM。 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
