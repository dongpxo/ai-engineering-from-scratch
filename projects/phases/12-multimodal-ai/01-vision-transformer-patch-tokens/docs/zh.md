# Vision Transformers 与 Patch-Token 原语

> 在任何 multimodal 之前，一张 image 都必须变成 transformer 能吃下的 token sequence。2020 年的 ViT paper 用 16x16 pixel patches、linear projection 和 position embedding 回答了这个问题。五年后，每个 2026 frontier model（Claude Opus 4.7 at 2576px native、Gemini 3.1 Pro、Qwen3.5-Omni）仍然从这里开始：encoder 从 ViT 变成 DINOv2 再到 SigLIP 2，加入了 register tokens，position scheme 变成 2D-RoPE，但这个 primitive 没有变。本课端到端读取 patch-token pipeline，并用 stdlib Python 构建它，让 Phase 12 后续课程对 “visual tokens” 有具体 mental model。

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## 学习目标

- 把一张 HxWx3 image 转换成带正确 positional encoding 的 patch token sequence。
- 计算给定 ViT（patch size、resolution、hidden dim、depth）的 sequence length、parameter count 和 FLOPs。
- 说出让 ViT 从 2020 research 走到 2026 production 的三个升级：self-supervised pretraining（DINO / MAE）、register tokens 和 native-resolution packing。
- 为 downstream task 在 CLS pooling、mean pooling 和 register tokens 之间做选择。

## 问题

Transformers 处理的是 vector sequences。Text 天生就是 sequence（bytes 或 tokens）。Image 是带三个 color channels 的 2D pixel grid，不是 sequence。如果你 flatten 每个 pixel，一张 224x224 RGB image 会变成 150,528 tokens，而这个长度下的 self-attention 完全不可行，因为它对 sequence length 是 quadratic。

2020 年以前的方法会在前面接一个 CNN feature extractor：ResNet 生成一个 7x7 的 2048-dim vector feature map，然后把这 49 个 tokens 送进 transformer。这能工作，但继承了 CNN 的 biases（translation equivariance、local receptive fields），也损失了 transformer 对 scale 的胃口。

Dosovitskiy et al. (2020) 问了一个直接的问题：如果跳过 CNN 呢？把 image 切成 fixed-size patches（比如 16x16 pixels），把每个 patch linearly project 成一个 vector，加上 positional embedding，然后把 sequence 送进 vanilla transformer。当时这像异端：不用 convolutions 做 vision。可是一旦数据足够多（JFT-300M，随后 LAION），它就在 ImageNet 上击败 ResNet，并且继续变强。

到 2026 年，ViT primitive 已经是无可争议的基础。每个 open-weights VLM 的 vision tower 都是它的某个后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是 “should we use patches?”，而是 “what patch size, what resolution schedule, what pretraining objective, what positional encoding.”

## 概念

### Patches as tokens

给定一张 shape 为 `(H, W, 3)` 的 image `x` 和 patch size `P`，你把 image 切成 `(H/P) x (W/P)` 的 non-overlapping patch grid。每个 patch 是一个 `P x P x 3` 的 pixel cube。把每个 cube flatten 成一个 `3 P^2` vector。应用一个 shape 为 `(3 P^2, D)` 的 shared linear projection `W_E`，把每个 patch 映射到 model hidden dimension `D`。

对于 canonical ViT-B/16 config：

- Resolution 224，patch size 16 → grid 14x14 → 196 patch tokens。
- 每个 patch 是 `16 x 16 x 3 = 768` 个 pixel values，project 到 `D = 768`。
- 加上一个 learnable `[CLS]` token → sequence length 197。

Patch projection 在数学上等价于一个 kernel size `P`、stride `P`、`D` output channels 的 2D convolution。这也是 production code 实际的实现方式：`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“Linear projection” 是概念框架；kernel 框架更高效。

### Positional embeddings

Patches 没有内在顺序，transformer 看到的是一个 bag。早期 ViTs 加一个 learnable 1D positional embedding（每个 position 一个 768-dim vector，总共 197 个）。这能工作，但会把模型绑定到 training resolution：inference 时如果改变 grid，就必须 interpolate position table。

现代 vision backbones 使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或 factorized 2D positions。2D-RoPE 会根据 patch 的 `(row, column)` index 旋转 query 和 key vectors，让模型从 rotation angle 推断 relative 2D position。没有 position table。模型在 inference 时可以处理任意 grid sizes。

### CLS token、pooled output 和 register tokens

Image-level representation 是什么？三种选择并存：

1. `[CLS]` token。把一个 learnable vector prepend 到 patch sequence。经过所有 transformer blocks 后，CLS token 的 hidden state 就是 image representation。继承自 BERT。Original ViT、CLIP 使用它。
2. Mean pool。对 patch tokens 的 output hidden states 取平均。SigLIP、DINOv2 和大多数现代 VLMs 使用它。
3. Register tokens。Darcet et al. (2023) 观察到，没有 explicit sink token 的 ViTs 会发展出 high-norm “artifact” patches，劫持 self-attention。加入 4-16 个 learnable register tokens 可以吸收这个负载，并提升 dense-prediction quality（segmentation、depth）。DINOv2 和 SigLIP 2 都带 registers。

这个选择会影响 downstream tasks。CLS 对 classification 没问题。对于把 patch tokens 送进 LLM 的 VLMs，你完全跳过 pooling：每个 patch 都成为 LLM input token。Registers 在 handoff 前会被丢弃，因为它们是 scaffolding，不是 content。

### Pretraining: supervised, contrastive, masked, self-distilled

2020 年的 ViT 使用 JFT-300M 上的 supervised classification 预训练。它很快被下面的方法取代：

- CLIP (2021)：在 400M image-text pairs 上做 contrastive learning。见 Lesson 12.02。
- MAE (2021, He et al.)：mask 75% patches，reconstruct pixels。Self-supervised，适用于 pure images。
- DINO (2021) / DINOv2 (2023)：student-teacher self-distillation，无 labels、无 captions。2023 年的 DINOv2 ViT-g/14 是最强的 purely-visual backbone，也是 “dense features” 用例的默认选择。
- SigLIP / SigLIP 2 (2023, 2025)：使用 sigmoid loss 和 NaFlex native aspect ratio 的 CLIP。它是 2026 open VLMs（Qwen、Idefics2、LLaVA-OneVision）中的主流 vision tower。

你的 pretraining 选择决定 backbone 擅长什么：CLIP/SigLIP 适合和 text 做 semantic matching，DINOv2 适合 dense visual features，MAE 适合作为 downstream finetuning 起点。

### Scaling laws

ViT scaling（Zhai et al. 2022）确立了 ViT quality 在 model size、data size 和 compute 上遵循可预测规律。在 fixed compute 下：

- Bigger model + more data → better quality。
- Patch size 是 sequence length vs fidelity 的 lever。Patch 14（DINOv2/SigLIP SO400m 的典型选择）比 patch 16 产生更多 tokens；对 OCR 和 dense tasks 更好，但速度更慢。
- Resolution 是另一个大 lever。从 224 到 384 再到 512 几乎总有帮助，代价是 FLOPs quadratic 增长。

ViT-g/14（1B params，patch 14，resolution 224 → 256 tokens）和 SigLIP SO400m/14（400M params，patch 14）是 2026 open VLMs 的两个 workhorse encoders。

### ViT 的 parameter count

完整计算在 `code/main.py` 里。对于 224 分辨率下的 ViT-B/16：

```text
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 前，先用这种方式粗算每个 ViT。Backbone size 会决定任何 downstream VLM 的 VRAM floor。

### 2026 production config

2026 年大多数 open VLMs 使用的 encoder 是 native resolution（NaFlex）下的 SigLIP 2 SO400m/14。它有：

- 400M parameters。
- Patch size 14，default resolution 384 → 每张 image 729 patch tokens。
- Image-level tasks 使用 mean pool；VQA 中全部 729 patches 流入 LLM。
- 4 个 register tokens，在 LLM handoff 前丢弃。
- 使用带 image-level scaling 的 2D-RoPE，支持 native aspect ratio。

这个 config 中每个 decision 都能追溯到一篇你可以阅读的 paper。

```figure
image-patch-tokens
```

## Use It

`code/main.py` 是一个 patch tokenizer 和 geometry calculator。它接收 `(image H, W, patch P, hidden D, depth L)`，并报告：

- Patching 后的 grid shape 和 sequence length。
- 一张 synthetic 8x8 pixel toy image 的 token sequence（逐步走过 flatten + project path）。
- 按 patch embed、position embed、transformer blocks 和 head 拆解的 parameter count。
- 目标 resolution 下单次 forward pass 的 FLOPs。
- 横向比较 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384。

运行它。把 parameter counts 和 published numbers 对上。调整 patch size 和 resolution，感受 token-count cost。

## Ship It

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT config（patch size、resolution、hidden dim、depth），它会生成 token-count、parameter-count 和 VRAM estimate，并附上理由。每当你为 VLM 选择 vision backbone 时，都使用这个 skill，它可以避免 “tokens exploded and my LLM context filled up” 这类意外。

## 练习

1. 计算 Qwen2.5-VL 在 native 1280x720 input、patch size 14 下的 patch-token sequence length。它和 CLS-only representation 相比如何？

2. 一个 1080p frame（1920x1080）在 patch 14 下会产生多少 tokens？30 FPS、5 分钟 video 会有多少 total visual tokens？哪种 cost saving 最大：pooling、frame sampling，还是 token merging？

3. 用 pure Python 实现 patch tokens 的 mean pooling。验证对 DINOv2 output 的 196 tokens 做 mean-pool，是否匹配模型 `forward` 在请求 pooled embedding 时返回的结果。

4. 阅读 “Vision Transformers Need Registers” (arXiv:2309.16588) 的 Section 3。用两句话描述 registers 吸收了什么 artifact，以及它为什么影响 downstream dense prediction。

5. 修改 `code/main.py` 支持 patch-n'-pack：给定一组不同 resolutions 的 images，生成一个 packed sequence 和 block-diagonal attention mask。学到 Lesson 12.06 时再对照验证。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Patch | “16x16 pixel square” | 输入 image 的 fixed-size non-overlapping region；会变成一个 token |
| Patch embedding | “Linear projection” | 一个 shared learned matrix（或 stride=P 的 Conv2d），把 flattened patch pixels 映射到 D-dim vectors |
| CLS token | “Class token” | Prepended learnable vector，它的 final hidden state 表示整张 image；2026 年已经是可选项 |
| Register token | “Sink token” | 额外 learnable tokens，用来吸收 ViTs 在 pretraining 中形成的 high-norm attention artifacts |
| Position embedding | “Positional info” | 让 sequence 感知 order 的 per-position vector 或 rotation；2D-RoPE 是现代默认方案 |
| Grid | “Patch grid” | 给定 resolution 和 patch size 后得到的 `(H/P) x (W/P)` 2D patch array |
| NaFlex | “Native flexible resolution” | SigLIP 2 feature：单个模型无需 retraining 即可服务多种 aspect ratios 和 resolutions |
| Backbone | “Vision tower” | 预训练 image encoder，它的 patch-token outputs 会送入 VLM 中的 LLM |
| Pooling | “Image-level summary” | 把 patch tokens 变成一个 vector 的策略：CLS、mean、attention pool 或 register-based |
| Patch 14 vs 16 | “Finer vs coarser grid” | Patch 14 每张 image 产生更多 tokens，对 OCR fidelity 更好但更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929)：original ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377)：MAE，self-supervised pretraining。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193)：大规模 self-distillation，无 labels。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588)：register tokens 和 artifact analysis。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)：2026 默认 vision tower。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560)：empirical scaling laws。
