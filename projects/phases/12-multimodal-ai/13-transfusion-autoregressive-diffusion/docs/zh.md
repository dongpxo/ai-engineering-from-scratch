# Transfusion：一个 Transformer 中的自回归文本 + Diffusion 图像

> Chameleon 和 Emu3 把赌注全押在 discrete tokens 上。它们能工作，但 quantization bottleneck 很明显：图像质量会停在 continuous-space diffusion models 之下。Transfusion（Meta，Zhou et al.，2024 年 8 月）押了相反方向：保留 continuous images，完全丢掉 VQ-VAE，并用两个 losses 训练一个 transformer。Text tokens 使用 next-token-prediction。Image patches 使用 flow-matching / diffusion loss。两个 objectives 优化同一组 weights。Stable Diffusion 3 背后的架构（MMDiT）是它的近亲。本课阅读 Transfusion thesis，构建一个 toy two-loss trainer，并追踪让一个 transformer 同时完成两项工作的 attention mask。

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 连接一个在同一 backbone 上运行两个 losses 的 transformer：text tokens 上的 NTP，以及 image patches 上的 diffusion MSE
- 解释为什么 image patches 内部使用 bidirectional attention，而 text tokens 上使用 causal attention，是正确的 mask 选择
- 从 compute、quality 和 code complexity 角度比较 Transfusion-style（continuous images，diffusion loss）与 Chameleon-style（discrete images，NTP）
- 说出 MMDiT 的贡献：每个 block 使用 modality-specific weights，但在 residual stream 上做 joint attention

## 问题

Discrete vs continuous image tokens 的争论比 LLM 更早。Continuous representations（raw pixels、VAE latents）能保留细节。Discrete tokens（VQ indices）适合 transformer 原生 vocabulary，但会在 quantization step 损失细节。

Chameleon / Emu3 选择了 discrete：一个 loss、一个 architecture，但 image fidelity 被 tokenizer quality 限制住。

Diffusion models 选择了 continuous：图像质量极高，但它们是与 LLM 分离的模型，需要复杂的 noise-schedule engineering，也没有与 text generation 干净集成的方式。

Transfusion 问的是：能不能两者兼得？保留 continuous images，仍然训练一个模型，并把两个 losses 缝进同一个 gradient step。

## 概念

### Two-loss architecture

一个单一 decoder-only transformer 处理一条包含以下内容的 sequence：

- Text tokens（discrete，来自 BPE vocab）。
- Image patches（continuous，16x16 pixel blocks，通过 linear embedding 投影到 hidden dim；和 ViT encoder 的输入相同）。
- `<image>` 和 `</image>` tags，用来标记 continuous patches 所在位置。

Forward pass 只运行一次。Loss 会按 token 类型选择两个 heads 中的一个：

- 对 text tokens：在 vocab-logits head 上使用标准 cross-entropy。
- 对 image patches：在 continuous patches 上使用 diffusion loss，即预测加到每个 patch 上的 noise。

Gradient 流过共享 transformer body。两个 losses 同时改进共享 weights。

### Attention mask：causal text + bidirectional image

Text tokens 必须 causal：你不能让 text token attend 到未来 text，否则 teacher forcing 会破坏。Image patches 则表示同一个快照；它们应该在同一 image block 内彼此 bidirectionally attend。

Mask：

```text
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

训练和推理时，它实现为 block-triangular mask。

### Transformer 内部的 diffusion loss

Diffusion loss 是标准做法：给 image patch 加噪声，让模型预测噪声（或者等价地预测 clean patch）。Transfusion 的版本使用 flow matching，即预测从 noisy 到 clean 的 velocity field。

训练时：

1. 对每个 image patch x0，采样一个随机 timestep t。
2. 采样 noise ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. Transformer 预测 v_theta(xt, t)；loss = MSE(v_theta(xt, t), ε - x0)。
4. 与同一 sequence 中的 text NTP losses 一起 backprop。

推理时，generation 是：

- Text tokens：标准 autoregressive sampling。
- Image patches：在 prior text tokens 条件下运行 diffusion sampling loop（通常 10-30 steps）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser et al.，2024 年 3 月）在 Transfusion 前后发布了 MMDiT（Multimodal Diffusion Transformer）。两个架构像兄弟。

MMDiT 的关键差异：

- 每个 block 使用 modality-specific weights。每个 transformer block 对 text tokens 和 image patches 有单独的 Q、K、V 和 MLP weights。Attention 是 joint（cross-modality），其他部分是 modality-specific。
- Rectified flow training。这是一种具体的 flow-matching 变体，相比 DDPM 有已知的 sampling 方式和更简单的数学。
- Scale。MMDiT 是 SD3（2B 和 8B 参数版本）的 backbone。Transfusion 论文扩展到 7B。

两者收敛到同一个核心想法：一个 transformer 对 text 运行 NTP，对 continuous image representations 运行 diffusion。

### 为什么它胜过 Chameleon-style

Continuous-diffusion 和 discrete-NTP 在 image generation 上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B params 下，它在 FID 上比同尺寸 Chameleon-style model 好 3-5 分。
- 不需要 tokenizer training；image encoder 更简单（linear projection 到 hidden，和 ViT 的 input layer 相同）。
- Image patch denoising 可以在推理时并行化，不像 autoregressive image tokens。

缺点是：Transfusion 是 dual-loss model，训练动态更棘手。Loss weights 需要调参。NTP 和 diffusion 的 schedule mismatch 可能导致某个 head 占主导。

### 下游分支

Janus-Pro（第 12.15 课）通过解耦用于 understanding 和 generation 的 vision encoder，细化了 Transfusion 的思路：一个用 SigLIP，一个用 VQ，同时共享 transformer body。Show-o（第 12.14 课）把 diffusion 换成 discrete-diffusion（masked prediction）。Unified-generation 家族在 Transfusion 之后迅速分叉。

2026 年能生成图像的 production VLMs，比如 Gemini 3 Pro、GPT-5、Claude Opus 4.7 的 image generation path，几乎肯定使用了这个家族的某个后代。细节是 proprietary。

## 投入使用

`code/main.py` 在一个很小的 MNIST-like 问题上构建 toy Transfusion：

- Text captions 是描述数字（0-9）的短整数序列。
- Images 是 4x4 byte grids。
- 一对共享权重的 linear projections 充当 transformer stand-in；text 上用 NTP loss，noisy patches 上用 MSE loss。
- Training loop 交替使用两个 losses，attention mask 是显式的。
- Generation 在一次 forward pass 中产出 text caption 和 4x4 image。

Transformer 是 toy。真正的 artifacts 是 two-loss plumbing、attention mask construction 和 inference loop。

## 交付成果

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的 multimodal training task（text + image、text + audio、text + video），它会设计 two-loss schedule（loss weights、mask shape、shared vs modality-specific blocks），并标记 implementation risks。

## 练习

1. 一个 Transfusion-style model 的训练数据中 70% 是 text tokens，30% 是 image patches。Image diffusion loss 的量级约为 text NTP loss 的 10 倍。什么 loss weights 能平衡它们？

2. 为这个 sequence 实现 block-triangular mask：`[T, T, <image>, P, P, P, P, </image>, T]`。把每个 entry 标成 0 或 1。

3. MMDiT 有 modality-specific QKV weights。相比 Transfusion 的 fully-shared transformer，这会增加多少参数量？在 7B params 下，值得吗？

4. Generation：给定一个 text prompt，模型先对 50 tokens 运行 NTP，然后遇到 `<image>`，接着在 256 个 patches 上运行 20 个 denoise steps。总共需要多少次 forward passes？

5. 阅读 SD3 paper 第 3 节。描述 rectified flow，以及为什么它比 DDPM 用更少 inference steps 收敛。

## 关键术语

| Term | 人们常说 | 它真正的意思 |
|------|----------|-------------|
| Two-loss training | "NTP + diffusion" | 一个 transformer 在同一个 gradient step 中同时优化 text tokens 上的 cross-entropy 和 continuous image patches 上的 MSE |
| Flow matching | "Rectified flow" | Diffusion 变体，预测从 noise 到 clean data 的 velocity field；数学比 DDPM 更简单 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 的架构：joint attention、modality-specific MLPs 和 norms |
| Block-triangular mask | "Causal text + bidirectional image" | Attention mask：text 之间 causal，但 image regions 内 bidirectional |
| Continuous image representation | "No VQ" | Image patches 是实值向量，而不是 integer codebook indices |
| Velocity prediction | "v-parameterization" | Network output 是 noise 和 data 之间的 velocity field，而不是 noise 本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
