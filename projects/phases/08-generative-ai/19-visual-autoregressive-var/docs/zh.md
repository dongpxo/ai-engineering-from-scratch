# 视觉自回归建模（VAR）：下一尺度预测

> Diffusion models 在时间上迭代采样（denoising steps）。VAR 在尺度上迭代采样：先预测一个 1x1 token，再预测 2x2、4x4，直到最终分辨率，每个尺度都条件化在前一个尺度上。2024 年的论文证明，VAR 在图像生成上符合 GPT-style scaling laws，并且在相同 compute budget 下超过 DiT。本课会构建它的核心机制。

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## 问题

Autoregressive generation 统治了 language modeling，因为它能可预测地 scale：更多 compute、更多参数、更低 perplexity、更好输出。2024 年之前，图像生成里主要有两类 AR 尝试：PixelRNN/PixelCNN（pixel-by-pixel）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE codes 上 token-by-token）。

二者都卡在 generation-order problem 上。Pixels 和 tokens 排成 2D grid，但 AR model 必须按 1D raster order 访问它们。左上角的早期 pixel 并不知道图像最终会变成什么。生成质量的 scaling 比 GPT-on-text 差，并且在 matched compute 下从未达到 diffusion-model quality。

VAR 通过改变“正在生成的东西”来修复 generation-order problem。它不是在空间上逐个预测 image tokens，而是在逐步提高的分辨率上预测整张图像。Step 1：预测一个 1x1 token（整体图像 “summary”）。Step 2：预测一个 2x2 token grid（更粗的 features）。Step 3：预测一个 4x4 grid。Step K：预测最终的 `(H/8)x(W/8)` grid。

每个尺度都会 attend 到所有先前尺度（在 “scale order” 上 causal），并且在自己的尺度内部 parallel。Order problem 消失了：scale k 上的整张图像由一次 transformer pass 生成。

## 概念

### VQ-VAE Multi-Scale Tokenizer

VAR 需要一个 **multi-scale discrete tokenizer**。对一张图像 x，它会产生一串分辨率逐步升高的 token grids：

```text
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 `z_k` 使用同一个 codebook（典型大小 4096-16384）。各尺度的 tokenization 不是彼此独立的，它会被训练成：把每个尺度的 residual 加起来就能重建 `f`：

```text
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一个 **residual VQ** 变体。Scale k 捕捉 scales 1..k-1 没有解释掉的部分。Decoder 接收所有 scale embeddings 的和，然后生成图像。

Multi-scale VQ tokenizer 只训练一次（像 VQGAN 一样），然后冻结。所有生成工作都由上层 autoregressive model 完成。

### Next-Scale Prediction

Generative model 是一个 transformer：它看到所有 previous scales 的 tokens，并预测 next scale 的 tokens。

Input sequence structure：

```text
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embeddings 同时编码 scale index 和该尺度内部的 spatial position。Attention 在 scale order 上 causal：scale k、position `(i, j)` 的 token 可以 attend 到 scales 1..k 的所有 tokens，也可以 attend 到 scale k 内部在某种 intra-scale order 下更早出现的 tokens（VAR 使用 fixed positional attention，没有 intra-scale causality，同一尺度内的所有 positions 并行预测）。

Training loss：在每个 scale k，给定所有 prior-scale tokens，预测 tokens `z_k`。对离散 VQ codes 做 cross-entropy loss。结构和 GPT 相同，只是这里的 “sequence” 变成了 scale-structured。

### Generation

推理时：

```text
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

如果 K = 10 scales，generation 就是 10 次 transformer forward passes。每次 pass 都会并行生成整个 scale，而不是在 scale 内做 per-token autoregression。对 256x256 图像来说，这大约是 10 passes，而 DiT 通常是 28-50 passes。

### 为什么 Next-Scale 胜过 Next-Token

三个结构性优势：

1. **Coarse-to-fine 符合自然图像统计。** 人类视觉感知和图像数据集都有 scale-dependent regularities：低频结构稳定、可预测；高频细节条件化在低频内容上。Next-scale prediction 利用了这一点。
2. **Scale 内部并行生成。** 不同于 GPT-style token AR，VAR 一步生成某个尺度上的所有 tokens。有效生成长度是 log-scale，而不是 linear。
3. **没有 generation order bias。** Scale k 的 tokens 可以看到完整的 scale k-1；不存在 “left-of” 或 “above” bias，早期 tokens 不必在缺少晚期上下文时提前承诺。

### Scaling Law

Tian 等人证明，VAR 在 ImageNet 上的 FID 遵循 power-law scaling curve，就像 GPT 的 perplexity 一样。参数或 compute 翻倍，error 会可靠下降。这是第一个像语言模型那样干净呈现这种 scaling behavior 的 image-generative model。结果是，VAR-scale predictions 可以从 compute 预测，而不再是每个架构都靠经验猜。

### 与 Diffusion 的关系

VAR 和 diffusion 共享同一种 data-compression story：二者都把生成问题拆成一串更容易的子问题。

- Diffusion：逐步加入 noise，学习撤销其中一步。
- VAR：逐步加入 resolution，学习预测下一个尺度。

它们是在同一个问题上切过的两条不同轴线。二者都产生 tractable conditional distributions。经验上，VAR inference 更快（passes 更少、每个 scale 内部全并行），并且在 class-conditional ImageNet 上匹配或超过 DiT。Text-conditional VAR（VARclip、HART）仍是活跃研究方向。

## 动手构建

在 `code/main.py` 中，你会：

1. 在 synthetic “image” data（2D Gaussian rings）上构建一个小型 **multi-scale VQ tokenizer**。
2. 训练一个 **VAR-style transformer** 来 next-scale-predict tokens。
3. 通过调用 transformer 4 次（4 scales）进行 sample 并 decode。
4. 验证 scale-ordered training 使 generation 能在 scale 内部并行。

这是一个 toy implementation。重点是看见 scale-structured attention mask 和 parallel-within-scale generation 真的在工作。

## 交付它

本课产出 `outputs/skill-var-tokenizer-designer.md`，这是一个用于设计 multi-scale tokenizer 的 skill：number of scales、scale ratios、codebook size、residual sharing、decoder architecture。

## 练习

1. **Scale count ablation.** 用 4、6、8、10 scales 训练 VAR。测量 reconstruction quality 与 autoregressive passes 数量的关系。更多 scales = 更细的 residuals = 更好质量，但 passes 更多。

2. **Codebook size.** 用 codebook sizes 512、4096、16384 训练 tokenizers。更大的 codebook 带来更好 reconstruction，但 prediction 更难。找出 knee。

3. **Parallel-within-scale check.** 对一个训练好的 VAR，显式测量 attention pattern。在 scale k 内部，模型是否 attend 到 cross-scale positions 而不是 intra-scale？验证 mask implementation。

4. **VAR vs DiT scaling.** 对相同的 ImageNet class-conditional 任务，在 matched param budgets（例如 33M、130M、458M）下训练 VAR 和 DiT。画出 FID vs compute。VAR 应该在每个 size 上领先 DiT，在小规模复现论文结果。

5. **Text conditioning.** 扩展 VAR，让它通过 adaLN 接收一个 text embedding（CLIP pooled）作为额外 conditioning input。这是 HART recipe。它会让 text-aligned sampling 的 FID 改善多少？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAR | “Visual AutoRegressive” | 通过 VQ token grids 金字塔上的 next-scale prediction 做图像生成 |
| Next-scale prediction | “先预测粗，再预测细” | 模型按逐步升高的 resolution scales 预测 tokens，并条件化在所有 previous scales 上 |
| Multi-scale VQ tokenizer | “Residual VQ” | 产生 K 个分辨率递增 token grids 的 VQ-VAE，decoder 会求和所有 scales |
| Scale k | “Pyramid level k” | K 个 resolution levels 中的一个，从 k=1 的 1x1 到 k=K 的 `(H/p)x(W/p)` |
| Parallel-within-scale | “One forward per scale” | Scale k 的所有 tokens 由一次 transformer pass 预测，不做 autoregressive |
| Causal-across-scales | “Scale-ordered attention” | Scale k 的 token 可以 attend 到 scales 1..k，但不能 attend 到 scales k+1..K |
| Residual VQ | “Additive tokenization” | 每个尺度的 tokens 编码低尺度留下的 residual；decoder 求和所有 scale embeddings |
| VAR scaling law | “Image GPT scaling” | FID 随 compute 遵循可预测 power law，类似语言模型 perplexity |
| HART | “Hybrid VAR + text” | Text-conditional VAR 变体，把 MaskGIT-style iterative decoding 与 VAR 的 scale structure 结合 |
| Scale position embedding | “(scale, row, col) triple” | Positional encoding 同时携带 scale index 和尺度内部 spatial coordinates |

## 延伸阅读

- [Tian et al., 2024 — “Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction”](https://arxiv.org/abs/2404.02905) — VAR 论文，canonical reference。
- [Peebles and Xie, 2022 — “Scalable Diffusion Models with Transformers”](https://arxiv.org/abs/2212.09748) — DiT，diffusion comparison baseline。
- [Esser et al., 2021 — “Taming Transformers for High-Resolution Image Synthesis”](https://arxiv.org/abs/2012.09841) — VQGAN，VAR multi-scale tokenizer 扩展的 tokenizer family。
- [van den Oord et al., 2017 — “Neural Discrete Representation Learning”](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像 tokenization 的基础。
- [Tang et al., 2024 — “HART: Efficient Visual Generation with Hybrid Autoregressive Transformer”](https://arxiv.org/abs/2410.10812) — text-conditional VAR。
