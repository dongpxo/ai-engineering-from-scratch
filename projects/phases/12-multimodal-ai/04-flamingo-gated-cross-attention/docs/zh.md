# Flamingo 与用于 Few-Shot VLMs 的门控交叉注意力

> DeepMind 的 Flamingo（2022）比所有人都更早做了两件事。第一，它证明单个模型可以处理任意交错的 images、videos 和 text sequences。第二，它证明 VLMs 可以做 in-context learning：给一个包含三个 (image, caption) pairs 的 few-shot prompt，模型无需任何 gradient step，就能为新 image 生成 caption。机制是 gated cross-attention layers：把它们插入 frozen LLM 的已有 layers 之间，并使用一个 learned tanh gate，gate 从 zero 开始，所以初始化时 LLM 的 text capability 会被保留。本课会走过 Flamingo 的 Perceiver resampler 和 gated cross-attention architecture，也就是 Gemini interleaved inputs 与 Idefics2 visual tokens 的祖先。

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 minutes

## 学习目标

- 解释 gated cross-attention 如何通过 tanh(gate) = 0，在初始化时保留 frozen LLM 的 text capability。
- 走过一个 Perceiver resampler：N 个 image patches → K 个固定 "latent" queries，通过 cross-attention 完成。
- 描述 Flamingo 如何用 respect image placement 的 causal masking 处理 interleaved image-text sequences。
- 复现 few-shot multimodal prompt structure（3 个 image-caption examples，然后一个 query image）。

## 问题

BLIP-2 把 32 个 visual tokens 喂进 frozen LLM 的 input layer。对于每个 prompt 一张 image，这能工作。但如果你想喂入 *many* images，并让它们和 text 交错出现呢？比如："here is image A, caption it; here is image B, caption it; now here is image C, caption it"。LLM 的 self-attention 需要在单一 stream 中同时处理 image tokens 和 text tokens，而且哪些 position 可以 attend 到哪些 images 会变得很麻烦。

Flamingo 的答案是：完全不要改变 LLM 的 input stream。把额外的 cross-attention layers 插入已有 LLM blocks 之间。Text tokens 仍然像往常一样流过 LLM 的 causal self-attention。每隔几个 LLM blocks，text tokens 还会通过一个新的 gated layer cross-attend 到 image features。Gate（初始化为 zero）意味着在 step zero，新 layers 是 no-op：模型行为和 pretrained LLM 完全一样。随着训练推进，gate 打开，visual information 开始流入。

Flamingo 回答的第二个问题是：如何处理每个 prompt 中数量可变的 images（0、1 或 many）？答案是 Perceiver resampler：一个小型 cross-attention module，接收任意数量的 patches，并输出固定数量的 visual latent tokens。无论 prompt 中有多少 images，LLM cross-attention layer 看到的 shape 都一样。

## 核心概念

### Frozen LLM

Flamingo 从一个 frozen Chinchilla 70B LLM 开始。全部 70B weights 都不动。已有 text self-attention 和 FFN 正常运行。

### Perceiver Resampler

对于 prompt 中的每张 image，ViT 会产生 N 个 patch tokens。Perceiver resampler 有 K 个固定 learnable latents（Flamingo 使用 K=64）。每个 resampler block 有两个子步骤：

1. Cross-attention：K 个 latents attend over N 个 patch tokens（Q 来自 latents，K/V 来自 patches）。
2. Latents 内部的 self-attention + FFN。

经过 6 个 resampler blocks 后，输出是 K=64 个 dim 1024 的 visual tokens，不管 ViT 产生了多少 patches。224x224 image（196 patches）和 480x480 image（900 patches）都会输出 64 个 resampler tokens。

对于 video，resampler 会在时间维度上应用：每一帧的 patches 产生 64 个 latents，temporal positional encoding 让模型区分 t=0 和 t=N。完整 video 会变成 T * 64 个 visual tokens。

### Gated Cross-Attention

在 frozen LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的 gated cross-attention block：

```text
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是 learnable scalar，初始化为 zero。
- `tanh(0) = 0`，所以初始化时 gated branch 的贡献为 zero。
- 当 `alpha` 离开 zero，cross-attention contribution 会平滑增长。
- Residual connection 意味着即使 gate 完全打开，也不会覆盖 LLM 的 text representation；它只是把 visual information 加在上面。

这是 Flamingo 中最重要的设计选择：visual conditioning 是 additive、gated，并且在初始化时为 zero。Step 0 的 Flamingo 在 text-only inputs 上就是一个完美的 Chinchilla 70B。

### Interleaved Inputs 的 Masked Cross-Attention

在像 "<image A> caption A <image B> caption B <image C> ?" 这样的 prompt 中，每个 text token 应该只看见 sequence 中位于它之前的 images。Cross-attention mask 会强制：position `t` 的 text token 只 attend 到 image index `i < i_t` 的 image resampler tokens，其中 `i_t` 是 position `t` 之前最近的 image。"只看最近的前置 image" 或 "看所有前置 images" 都是合理选择；Flamingo 选择了前者。

### In-Context Few-Shot Learning

Flamingo prompt 看起来像这样：

```text
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到 completion pattern，然后输出 "bird"（或 image3 实际展示的内容）。没有 gradient steps。Frozen LLM 的 in-context learning capability 会通过 gated cross-attention 延续下来：这是论文的 punchline，也是它重要的原因。

### Training Data

Flamingo 使用三个 datasets 训练：

1. MultiModal MassiveWeb (M3W)：43M web pages，带 interleaved images and text，并重建 reading order。
2. Image-Text Pairs (ALIGN + LTIP)：4.4B pairs。
3. Video-Text Pairs (VTP)：27M short video clips。

OBELICS（2023）是 interleaved web corpus 的 open reproduction，Idefics、Idefics2 以及多数 open "Flamingo-like" models 都在它上面训练。

### OpenFlamingo and Otter

OpenFlamingo（2023）是 open reproduction。Architecture 相同（Perceiver resampler + gated cross-attention on frozen LLaMA or MPT）。Checkpoints 有 3B、4B、9B。由于 base LLM 更小、数据更少，质量落后 Flamingo。

Otter（2023）基于 OpenFlamingo，在 MIMIC-IT（一个 multimodal instructions dataset）上做 instruction tuning，证明 gated cross-attention 也适用于 instruction following。

### 后代模型

- Idefics / Idefics2 / Idefics3：Hugging Face 的 gated cross-attention lineage，逐步变得更简单（Idefics2 放弃 resampler，改为 direct patch tokens with adaptive pooling）。
- Flamingo-to-Chameleon transition：到 2024 年，许多团队转向 early-fusion（Lesson 12.11）；但在必须冻结 backbone 的生产场景中，Flamingo-style gated cross-attention 仍然存在。
- Gemini 的 interleaved input：概念上继承了 Flamingo 的 interleaved-format flexibility，尽管确切机制是 proprietary。

### 与 BLIP-2 对比

| | BLIP-2 | Flamingo |
|---|---|---|
| Visual bridge | Q-Former once at input | Gated cross-attention at every M layers |
| Visual tokens | 32 per image | 64 per image per cross-attn layer |
| Frozen LLM | Yes | Yes |
| Few-shot in-context | Weak | Strong — 论文的核心 |
| Interleaved inputs | No native support | Yes，设计目标 |
| Training data | 130M pairs | 1.3B pairs + 43M interleaved pages |
| Parameter count | 188M trained | ~10B trained (cross-attn layers) |
| Compute | Days on 8 A100s | Weeks on thousands of TPUv4 |

如果预算有限，并且目标是 single-image VQA，选 BLIP-2。如果需要 interleaved、few-shot 或 multi-image reasoning，选 Flamingo/Idefics2。

## 实战使用

`code/main.py` 演示：

1. 在 36 个 fake patch tokens 上运行 Perceiver resampler，使用 8 个 learnable latents（pure Python cross-attention）。
2. 一个 gated cross-attention step：`alpha = 0` → output 等于 input（LLM unchanged），然后 `alpha = 2.0` → visual contribution 被混入。
3. 一个 interleaved-mask builder，为 "(image 1) (text 1) (image 2) (text 2)" sequence 产生 2D attention mask。

## 交付

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个 open VLM 的 config（resampler Y/N、cross-attn frequency、gate scheme），它会识别 Flamingo lineage elements，并解释 freezing strategy。这对调试 fine-tune 为什么降低 text performance 很有用（答案通常是：gate opening 太快、太宽）。

## 练习

1. 计算 Flamingo-9B 的 visual parameter count：9B LLM + 1.4B gated cross-attention layers + 64M resampler。训练参数占 total params 的多少比例？

2. 在 PyTorch 中实现 gated residual `y = tanh(alpha) * cross + x`。用实验展示当 `alpha=0` 时，初始化时 `y==x` 完全成立。

3. 阅读 OpenFlamingo Section 3.2（arXiv:2308.01390），了解当 batch 中每个 prompt 的 image count 不同时，它们如何处理 multiple images。描述 padding strategy。

4. 为什么 Flamingo 的 cross-attention mask 让 text token 只能 attend 到 *最近的* 前置 image，而不是所有前置 images？阅读 Flamingo paper Section 2.4，并解释 tradeoff。

5. In-context few-shot：为一个新的 Flamingo variant 构造一个 prompt，包含 4 个 "image → color of main object" examples。描述当 examples 数量从 0 变到 8 时，expected accuracy pattern 如何变化。

## 关键术语

| Term | 人们通常怎么说 | 它实际是什么意思 |
|------|----------------|------------------|
| Perceiver resampler | "Fixed-latent cross-attention" | 从可变数量 input patches 中产生 K 个固定 tokens 的 module |
| Gated cross-attention | "Tanh-gated bridge" | Residual layer `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| Interleaved input | "Mixed sequence" | Images 和 text 按 reading order 自由混合的 prompt format |
| Frozen LLM | "No LLM gradients" | Text LLM 的 weights 不更新；只训练 resampler + cross-attn layers |
| Few-shot | "In-context examples" | 在 prompt 中给几个 (image, answer) pairs；模型无需 finetuning 就能泛化 |
| OBELICS | "Interleaved web corpus" | 141M web pages 的 open dataset，包含按 reading order 排列的 images 和 text |
| Chinchilla | "70B frozen base" | Flamingo 的 frozen text LLM，来自 DeepMind 的 Chinchilla paper |
| Gate schedule | "How alpha moves" | Cross-attention gate 在训练期间打开的速度 |
| Cross-attn frequency | "Every M layers" | 插入 gated cross-attention block 的频率；Flamingo 使用 M=4 |
| OpenFlamingo | "Open reproduction" | MosaicML/LAION 的 3-9B open checkpoint；architecture 与 Flamingo 相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — open reproduction。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — interleaved web corpus。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — general Perceiver architecture。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — instruction-tuned Flamingo descendant。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo approach 的现代简化版。
