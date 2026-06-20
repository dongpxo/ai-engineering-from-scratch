# 边缘推理：Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘侧的核心约束是内存带宽，不是算力。移动 DRAM 大约是 50-90 GB/s；数据中心 HBM3 能达到 2-3 TB/s，两者相差 30-50 倍。Decode 是 memory-bound，因此这个差距是决定性的。到 2026 年，格局分成四条路线。Apple M4/A18 Neural Engine 峰值 38 TOPS，使用统一内存（没有 CPU↔NPU copy）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 可以在 M3 Max 上以约 41 tok/s 运行 Llama 3.1 8B（Q4）（约为 native 的 70-80%）；WebLLM 有 17.6k GitHub stars，提供 OpenAI-compatible API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）能放下 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill，Bosch、ThunderSoft、MediaTek 已在 CES 2026 展示过。

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## 学习目标

- 解释为什么移动端 LLM 推理是内存带宽受限，而算力只是次要因素。
- 枚举四类边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并把每类目标匹配到合适用例。
- 说出 2026 年 WebGPU 的覆盖缺口（Firefox Android 仍在追赶）以及 Safari iOS 26 的落地情况。
- 为每个目标选择量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题

客户想要一个端侧聊天机器人：voice-first、private-by-default、可离线工作。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 可以跑到约 55 tok/s，足够好。在 iPhone 16 Pro 上，同一个模型只有 3 tok/s，不够好。在搭载 Snapdragon 8 Gen 3 的中端 Android 上约为 7 tok/s。通过 Chrome Android v121+ 的 WebGPU 在浏览器中运行时，会根据设备落在 4-8 tok/s。

吞吐差异不是移植问题。它来自带宽差距、量化格式，以及 NPU 是否能从 user-space 访问。2026 年的边缘推理其实是四个不同问题，对应四种不同解法。

## 概念

### 带宽才是真正天花板

Decode 每生成一个 token 都要读取完整权重。一个 7B Q4 模型是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms，理论上限约为 14 tok/s。到了 90 GB/s（高端移动 DRAM），上限会移动到约 25 tok/s。在这个数字以下，增加 compute 没有帮助。

数据中心 HBM3 以 3 TB/s 读取同样的 3.5 GB 只需要 1.2 ms，上限是 830 tok/s。同一个模型，同一组权重。差别只在内存子系统。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一个池）意味着没有 copy overhead。
- 可通过 Core ML + 编译后的 `.mlmodel` 模型访问，或者通过 PyTorch 的 Metal Performance Shaders（MPS）间接使用。
- Llama.cpp 的 Metal backend 使用 MPS，不是直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS app 最实际的路线：Core ML + INT4 weights + FP16 activations。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。与 CPU 和 GPU 集成在同一个 SoC 中，但有独立 memory domain。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 转换的能力。
- Chat templates、Llama 3.2、Phi-3 都作为 AI Hub 的一等 artifacts 发布。

### Intel / AMD NPUs（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件生态落后于 Apple/Qualcomm；OpenVINO 正在改善，但仍偏小众。
- 最适合 Windows ARM copilot apps；也适合 AMD/Intel desktop 上的 local-first native 应用。

### WebGPU + WebLLM

- 通过 WebGPU compute shaders 在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s，约为同 backend native 的 70-80%。
- WebLLM 有 17.6k GitHub stars；提供 OpenAI-compatible JS API；Apache 2.0。
- 2026 年覆盖：Chrome Android v121+、Safari iOS 26 GA，Firefox Android 仍在追赶。整体移动端覆盖约 70-75%。

### NVIDIA Jetson family

- Orin Nano Super（8GB）：能放下 Llama 3.2 3B、Phi-3，并达到不错的 tok/s。
- AGX Orin：通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b。
- Thor / T4000（JetPack 7.1）：性能是 AGX Orin 的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 speculative decoding、NVFP4 weights、chunked prefill，也就是把数据中心优化带到了边缘侧。

### 每个目标的量化选择

| Target | Format | Notes |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML conversion path |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub converters |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译后的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 | Memory-bound |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM path |

### 边缘侧的长上下文陷阱

Llama 3.1 的 128K context 是数据中心特性。在一台 8 GB RAM 的手机上，4 GB 模型 + 32K tokens 的 2 GB KV cache + OS overhead = OOM。边缘部署通常把 context 控制在 4K-8K，除非你接受激进的 KV quantization（Q4 KV）。

### Voice 是杀手级应用

Voice agents 对 latency 非常敏感（first token < 500 ms）。本地推理可以完全消除网络延迟。把它与 speech-to-text（Whisper Turbo 变体可以在边缘侧运行）组合起来，edge inference 就能形成生产质量的语音闭环。

### 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 上约 41 tok/s。
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上约 40 tok/s。
- 数据中心到边缘侧的带宽差距：30-50x。
- WebGPU 移动端覆盖率：约 70-75%（Firefox Android 落后）。

## Use It

`code/main.py` 会用 bandwidth-bound 数学计算不同边缘目标的理论 decode throughput 上限。它会与观察到的 benchmarks 对比，并指出瓶颈在哪里来自带宽，而不是算力。

## Ship It

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/browser/Jetson）、模型，以及 latency/memory budget，它会选择量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上的 7B Q4 模型，计算 decode ceiling。与观察到的 6-8 tok/s 对比：runtime 是否高效？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧浏览器设计 fallback：通过同一个 OpenAI-compatible API 走 server-side。
3. 你的 iOS app 需要 4K-context streaming。哪种 model/format 组合可以让 iPhone 16 上的 active memory 保持在 4 GB 以下？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能放下 3B。如果产品同时面向两者，你如何统一 inference stack？
5. 论证“WebLLM 在 2026 年是否已经 production-ready”。引用覆盖率、性能和 Firefox Android 缺口。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| ANE | “Apple neural engine” | M-series 和 A-series 中的端侧 NPU；统一内存 |
| Hexagon | “Qualcomm NPU” | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | “browser GPU” | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | “browser LLM runtime” | MLC-LLM 项目；Apache 2.0；OpenAI-compatible JS |
| Jetson | “NVIDIA edge” | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | “edge TensorRT” | TensorRT-LLM 的 2026 边缘移植；EAGLE-3 + NVFP4 |
| Unified memory | “shared pool” | CPU 和 NPU 看到同一块 RAM；没有 copy overhead |
| Bandwidth-bound | “memory limited” | Decode 受每秒读取权重的字节数限制 |
| Core ML | “Apple conversion” | Apple 的 ANE-native model framework |
| QNN | “Qualcomm stack” | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — landscape and benchmarks。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge port announcement。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — design and benchmarks。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native conversion。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — pre-converted models for Hexagon。
