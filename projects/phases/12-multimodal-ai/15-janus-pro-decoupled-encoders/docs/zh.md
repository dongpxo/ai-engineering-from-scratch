# Janus-Pro：用于统一多模态模型的解耦编码器

> Unified multimodal models 有一个不可避免的张力。Understanding 需要 semantic features，也就是 SigLIP 或 DINOv2 输出的、富含 concept-level information 的 vectors。Generation 需要 reconstruction-friendly codes，也就是能重新组合成清晰 pixels 的 VQ tokens。这两个目标在单个 encoder 中并不兼容。Janus（DeepSeek, October 2024）和 Janus-Pro（DeepSeek, January 2025）认为修复方法是不再强行兼容：把两个 encoders 解耦。任务之间共享 transformer body，但 understanding 走 SigLIP，generation 走 VQ tokenizer。7B 规模下，Janus-Pro 在 GenEval 上超过 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课阅读为什么 two encoders 能解决 one encoder 失败的问题。

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## Learning Objectives

- 解释为什么单个 shared encoder 会牺牲 understanding 或 generation 质量。
- 描述 Janus-Pro 的 routing：understanding 在 input side 使用 SigLIP features，generation 在 input 和 output 两侧使用 VQ tokens。
- 追踪让 Janus-Pro 成功而 Janus 未能成功的 data-mix scaling。
- 比较 decoupled（Janus-Pro）、coupled-continuous（Transfusion）和 coupled-discrete（Show-o）架构。

## 问题

Unified models 在 understanding 和 generation 之间共享 transformer body。以往尝试（Chameleon、Show-o、Transfusion）都为两个方向使用同一个 visual tokenizer。这个 tokenizer 必然是妥协：

- 为 reconstruction 优化（generation）：VQ-VAE 捕捉 fine-grained pixel detail，但产生的 tokens 语义一致性较弱。
- 为 semantics 优化（understanding）：SigLIP embeddings 会把 "cat" images 聚到 "cat" tokens 附近，但无法提供良好 reconstruction。

Show-o 和 Transfusion 都为此付出代价：某个方向上会出现明显 quality tax。Janus-Pro 反问：既然任务需求不同，为什么必须要求一个 tokenizer？

## 概念

### 解耦视觉编码

Janus-Pro 的架构把两个 encoders 分开：

- Understanding path。Input image → SigLIP-SO400m → 2-layer MLP → transformer body。
- Generation path。Input image（如果基于已有 image 做 conditioning）→ VQ tokenizer → token IDs → transformer body。
- Output generation。Transformer 预测 image tokens → VQ decoder → pixels。

Transformer body 是共享的。Body 上游和下游的一切都是 task-specific。

Inputs 通过 prompt format 消歧：`<understand>` tag 走 SigLIP；`<generate>` 走 VQ。或者 routing 可从 task 隐式推断。

### 为什么这能工作

Understanding loss 得到 SigLIP features，而 CLIP-style pretraining 已经把它们调成适合 semantic similarity 的表示。因此相较 Show-o / Transfusion，模型的 perception benchmarks 会提升，因为 input features 更适合该任务。

Generation loss 得到 VQ tokens，而 tokenizer 已经被调成适合 reconstruction。相较 Show-o，image quality 会提升，因为 VQ codes 能干净地组合回 pixels。

Shared transformer body 会看到两种 input distributions（SigLIP 和 VQ），并学习同时处理二者。核心主张是：只要数据足够、参数足够，body 可以吸收这种 switching。

### Data scaling：Janus vs Janus-Pro

Janus（original, arXiv 2410.13848）引入了解耦，但规模较小（1.3B params，有限数据）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 7B params（vs 1.3B）。
- Stage 1（alignment）使用 90M image-text pairs，从 72M 提升。
- Stage 2（unified）使用 72M，从 26M 提升。
- Stage 3 增加 200k image-gen instruction samples。

结果是：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），并在 GenEval 上超过 DALL-E 3（0.80 vs 0.67）。一个 open model，在 unified spectrum 的两个方向上都有竞争力。

### JanusFlow：rectified flow 变体

JanusFlow（arXiv 2411.07975）把 VQ generation path 换成 rectified-flow generation path（continuous）。拆分变成 SigLIP-for-understanding + rectified-flow-for-generation。Quality ceilings 进一步提高。架构仍然是 decoupled-encoders-shared-body。

### Shared body 的工作

Transformer body 处理 unified sequence，但面对两种 input distributions。它的工作是：

- 对 understanding：消费 SigLIP features + text tokens → autoregressively 输出 text。
- 对 generation：消费 text tokens +（可选 image VQ tokens）→ autoregressively 输出 image VQ tokens。

Body 在每个 block 中没有 modality-specific weights。它就是你会在 Qwen 或 Llama 中看到的 text-style transformer，再加上两个 input adapters。

有趣的是，这意味着 Janus-Pro 的 body 可以从 pretrained LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 提供了 reasoning ability，而 pure-from-scratch unified models 很难达到这一点。

### 与 InternVL-U 对比

InternVL-U（Lesson 12.10）是 2026 年的后续路线。它结合了：

- Native multimodal pretraining（InternVL3 backbone）。
- Decoupled-encoder routing（SigLIP in, VQ + diffusion heads out）。
- Unified understanding + generation + editing。

InternVL-U 把 Janus-Pro 的架构选择纳入一个更大的 framework。Decoupled-encoder idea 现在已经是 scale 上 unified models 的默认方案。

### 局限性

Decoupled encoders 会增加架构复杂度。两个 tokenizers 要训练，两个 input paths 要维护，两组 failure modes 要处理。对于不需要 generation 的产品，Janus-Pro 是 over-engineered：选择 LLaVA-family understanding model。

对于不需要 understanding 的产品，Janus-Pro 又是 overqualified：选择 Stable Diffusion 3 / Flux model。

对于二者都需要的产品，Janus-Pro 现在是 reference open architecture。

## Use It

`code/main.py` 模拟 Janus-Pro routing：

- 两个 mock encoders：SigLIP-like（产生 256-dim semantic vectors）和 VQ-like（产生 integer codes）。
- 一个 prompt router，根据 task tag 选择 encoder。
- 一个 shared body（stand-in），不管 tokens 来自哪个 encoder，都处理 token sequences。
- 一个从 stage 1（alignment）切换到 stage 3（instruction tune）的 weighted-sample schedule。

运行后会打印 3 个 examples 的 routed paths：image QA、T2I、image editing。

## Ship It

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望同时拥有 frontier-ish quality 的 generation + understanding 的产品，它会选择 Janus-Pro、JanusFlow 或 InternVL-U，并给出具体 data-scale recommendation。

## Exercises

1. Janus-Pro-7B 在 GenEval 上超过 DALL-E 3。解释为什么一个 7B open model 可以在 generation 上匹配 frontier proprietary model，但不能在 understanding 上做到同样水平。

2. 实现一个 router function：给定 prompt text，分类为 `understand` 或 `generate`。你如何处理 "describe and then sketch" 这类 ambiguous prompts？

3. JanusFlow 用 rectified flow 替换 VQ path。Transformer body 现在输出什么？Loss 中发生了什么变化？

4. 提出 Janus-Pro 架构可以通过再加一个 decoupled encoder 处理的第四种任务。例子：image segmentation（DINO-style）、depth（MiDaS-style）。

5. 阅读 Janus-Pro Section 4.2 关于 data scaling 的内容。哪个 data stage 对 T2I quality gain 相比 Janus 贡献最大？

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Decoupled encoding | "Two visual encoders" | 每个方向使用独立 tokenizer 或 encoder：understanding 用 semantic，generation 用 reconstruction |
| Shared body | "One transformer" | 单个 transformer 处理任一 encoder 的输出；没有 modality-specific weights |
| SigLIP for understanding | "Semantic features" | CLIP-family vision tower，提供丰富 conceptual features，但 reconstruction 差 |
| VQ for generation | "Reconstruction codes" | Vector-quantized tokens，可以干净地 decode 回 pixels |
| JanusFlow | "Rectified-flow variant" | 使用 continuous flow-matching generation head 而不是 VQ 的 Janus-Pro |
| Routing tag | "Task tag" | Prompt marker（`<understand>` / `<generate>`），用于选择 input encoder |

## Further Reading

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
