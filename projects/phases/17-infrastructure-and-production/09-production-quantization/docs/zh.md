# 生产量化：AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> Quantization format 不是一个通用选择，它取决于 hardware、serving engine 和 workload。GGUF Q4_K_M 或 Q5_K_M 通过 llama.cpp 和 Ollama 占据 CPU 与 edge。需要在同一个 base 上跑 multi-LoRA 时，GPTQ 在 vLLM 中胜出。AWQ 配合 Marlin-AWQ kernels，在 7B class model 上达到约 741 tok/s，并在 INT4 中有最好的 Pass@1——这是 2026 年 datacenter production 的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间路线：near-lossless 且广泛支持。NVFP4 和 MXFP4（Blackwell microscaling）更激进，需要 per-block validation。团队常踩两个坑：calibration dataset 必须匹配 deployment domain；KV cache 与 weight quantization 是两回事——AWQ lesson 中“我的模型现在只有 4 GB”常常忘了 production batch sizes 下还有 10-30 GB KV cache。

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 说出六种 production quantization formats，以及它们在 2026 年各自的 sweet spots。
- 给定 hardware（CPU vs GPU、Hopper vs Blackwell）、engine（vLLM、TRT-LLM、llama.cpp）和 workload（routine chat、reasoning、multi-LoRA）后选择合适格式。
- 计算某个格式节省的 weight memory，以及没有被触碰的 KV cache。
- 说出会让 quantized models 在 domain traffic 上退化的 calibration-dataset pitfall。

## 问题

Quantization 会降低 memory 和 HBM bandwidth，这正是 decode 需要的东西。一个 FP16 70B model 有 140 GB weights。把 weights 量化到 INT4（AWQ 或 GPTQ）后，model 变成 35 GB，可以放进一张 H100，并给 KV cache 留出空间；这一点很重要，因为 128 concurrent sequences、2k context 时，仅 KV cache 就有 20-30 GB。

但 quantization 不是免费的。激进量化会降低 quality，尤其是在 reasoning-heavy tasks 上。不同 formats 适配不同 engines。不同 hardware 原生支持不同 precisions。2026 年的 format zoo 是真实存在的，你不能照抄别人的选择，必须基于自己的 stack 做决定。

## 概念

### 六种格式

| Format | Bits | Sweet spot | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、edge、laptops | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的 multi-LoRA | vLLM、TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ)、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### GGUF — CPU/edge 默认选择

GGUF 是 file format，严格说不是一种 quantization scheme；它会把 K-quant variants（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包到一个 container。Q4_K_M 和 Q5_K_M 是 production defaults，在 4-5 bits 下接近 BF16 quality。对于 CPU 或 edge serving，这是最佳选择，因为 llama.cpp 远远是最快的 CPU inference engine。

它在 vLLM 中有 throughput penalty：7B 上约 93 tok/s，这个格式没有针对 GPU kernels 优化。只有 deployment target 是 CPU/edge 时才用 GGUF。除此之外不要用。

### GPTQ — vLLM 中的 multi-LoRA

GPTQ 是一种带 calibration pass 的 post-training quantization algorithm。Marlin kernels 让它在 GPU 上变快（相比 non-Marlin GPTQ 有 2.6x speedup）。7B 上约 712 tok/s。

它的独特胜点是：GPTQ-Int4 在 vLLM 中支持 LoRA adapters。如果你要服务一个 base model 加 10-50 个 fine-tuned variants（每个都是一个 LoRA），GPTQ 就是你的路径。截至 2026 年初，NVFP4 还不支持 LoRA。

### AWQ — datacenter GPU 默认选择

Activation-aware Weight Quantization。它会在量化过程中保护约 1% 最显著 weights。Marlin-AWQ kernels：相比 naive 有 10.9x speedup。7B 上约 741 tok/s，在 INT4 formats 中 Pass@1 最好。

新的 GPU serving 默认选 AWQ，除非你需要 multi-LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4）。

### FP8 — 可靠的中间路线

8-bit floating point。Near-lossless，广泛支持。Hopper Tensor Cores 原生加速 FP8，Blackwell 继承这一点。当 quality 不能妥协时（reasoning、medical、code-gen），FP8 是 2026 年安全默认选择。它的 memory savings 是 INT4 的一半，但 quality risk 低得多。

### MXFP4 / NVFP4 — Blackwell aggressive

Microscaling FP4。每个 weights block 都有自己的 scale factor。它很激进，但在 Blackwell Tensor Cores 上有 hardware acceleration。相比 FP8，它可以把每 token bytes 减半——这就是 Phase 17 · 07 中的经济收益。

Caveats：

- 还没有 LoRA 支持（2026 年初）。
- Reasoning-heavy workloads 上会看到 quality drop。
- 每个 model 都要在自己的 eval set 上验证。

### Calibration trap

AWQ 和 GPTQ 需要 calibration dataset，通常是 C4 或 WikiText。对于 domain models（code、medical、legal），如果用 generic web text 做 calibration，算法会错误判断哪些 weights 应该被保护。HumanEval 上的 Pass@1 可能下降数个百分点。

修复方法：使用 in-domain data 做 calibration。几百个 domain samples 通常就足够。上线前在 eval set 上测试。

### KV cache trap

AWQ 会把 weights 缩到 4 bits。KV cache 是独立的，仍保持 FP16/FP8。对于一个使用 AWQ 的 70B model：

- Weights：约 35 GB（从 140 GB INT4 化）。
- KV cache at 128 concurrent × 2k context：约 20 GB。
- Activations：约 5 GB。
- Total：约 60 GB，可以放进 H100 80GB。

天真地说“我把模型量化到 4 GB 了”，会忘记另外 30-50 GB。要整体预算 HBM。

另外，KV cache quantization（FP8 KV 或 INT8 KV）是另一个独立选择，有自己的 tradeoffs——它直接影响 attention accuracy，不是免费收益。

### AWQ INT4 对 reasoning 有风险

Chain-of-thought、math、long-context code-gen——这些任务会明显受激进量化影响。AWQ INT4 在 MATH 上会损失约 3-5 points。对于 reasoning-heavy workloads，上 FP8 或 BF16，接受 memory cost。

### 2026 picking guide

- CPU/edge serve：GGUF Q4_K_M。结束。
- GPU serve、routine chat、无 LoRA：AWQ。
- GPU serve、multi-LoRA：GPTQ with Marlin。
- Reasoning workload：FP8。
- Blackwell datacenter、quality 已验证：NVFP4 + FP8 KV。
- 不确定：对每个候选 format 跑 1,000-sample eval。

```figure
gpu-memory-breakdown
```

## Use It

`code/main.py` 会在不同 model sizes 上，计算六种 formats 的 memory footprint（weights + KV + activations）和相对 throughput。它展示 KV cache 何时主导、weight compression 何时值得，以及 FP8 何时是安全选择。

## Ship It

本课产出 `outputs/skill-quantization-picker.md`。给定 hardware、model size、workload type 和 quality tolerance，它会选择一种 format，并生成 calibration/validation plan。

## 练习

1. 运行 `code/main.py`。对于 128 concurrent、2k context 的 70B model，计算每种 format 的 total HBM。哪种 format 能让你放进一张 H100 80GB？
2. 你有一个 7B coding model。选择一种 format 并说明理由。如果你判断错了 quality tolerance，恢复路径是什么？
3. 计算为 medical domain model 校准 AWQ 所需的 calibration-dataset size。为什么更多数据不总是更好？
4. 阅读 Marlin-AWQ kernel paper 或 release notes。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而 raw GPTQ 约为 712。
5. 什么时候组合 AWQ weights + FP8 KV cache 比保持 KV 为 BF16 更有意义？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GGUF | “llama.cpp format” | 打包 K-quant variants 的 file format；CPU/edge 默认选择 |
| Q4_K_M | “Q4 K M” | 4-bit K-quant medium；production GGUF default |
| GPTQ | “gee pee tee q” | 带 calibration 的 post-train INT4；在 vLLM 中支持 LoRA |
| AWQ | “a w q” | Activation-aware INT4；Marlin kernels；INT4 下最佳 Pass@1 |
| Marlin kernels | “fast INT4 kernels” | Hopper 上用于 INT4 的 custom CUDA kernels；10x speedup |
| FP8 | “eight-bit float” | Hopper/Ada/Blackwell 上的安全 precision default |
| MXFP4 / NVFP4 | “microscaling four” | 带 per-block scale factors 的 Blackwell 4-bit FP |
| Calibration dataset | “cal data” | 用于选择 quantization parameters 的输入文本；必须匹配 domain |
| KV cache quantization | “KV INT8” | 与 weights 独立的选择；会影响 attention accuracy |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)：comparative benchmarks。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)：按 format 展示 throughput numbers。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/)：format-by-format picking。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html)：supported formats 和 flags。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978)：AWQ 原始 formulation。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323)：GPTQ 原始 formulation。
