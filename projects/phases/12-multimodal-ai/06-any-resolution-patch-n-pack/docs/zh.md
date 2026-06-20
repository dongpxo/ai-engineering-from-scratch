# Any-Resolution Vision：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年以前的 VLM 答案是把所有东西 resize 成固定正方形，但这会丢掉让 OCR、document understanding 和 high-resolution scene parsing 真正起作用的信号。NaViT（Google, 2023）证明了你可以用 block-diagonal masking，把 variable-resolution patches pack 进单个 transformer batch。Qwen2-VL 的 M-RoPE（2024）完全抛弃了 absolute positional tables。LLaVA-NeXT 的 AnyRes 把高分辨率图像切成 base + sub-images。SigLIP 2 的 NaFlex variant（2025）现在是 open VLMs 的默认 encoder，用一个 checkpoint 服务所有 aspect ratios。本课从头到尾实现 patch-n'-pack。

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## Learning Objectives

- 把一批 variable-resolution images 的 patches pack 成一个 sequence，并构建 block-diagonal attention mask。
- 针对给定任务，在 AnyRes tiling（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做选择。
- 在不 resize 的情况下计算 OCR、charts 和 photography 的 token budgets。
- 说出 square-resize 的三种 failure modes：squished text、cropped content、wasted tokens on padding。

## 问题

Transformers 期望输入是 sequence。Batch 是一组长度相同的 sequences。如果你的图像都是 224x224，每次都会得到 196 个 patch tokens，不需要 padding，任务完成。Train on 224，infer on 224，从此不用再思考 resolution。

现实不会配合。Documents 是 portrait（8.5x11 inches，大约 2:3）。Chart screenshots 是 landscape（16:9）。Receipts 又高又窄（1:3）。Medical imaging 通常是 2048x2048 或更大。Mobile device screenshots 是 1170x2532（0.46:1）。

2024 年以前有三种方案，每种都会失败：

1. Resize 到固定正方形（224x224 或 336x336）。挤压会扭曲文字和人脸。Downscale 会毁掉 chart labels 和 OCR content。这是 LLaVA-1.5 之前的标准做法。
2. Crop 到固定 aspect ratio。你会丢掉大部分图像，而选择 crop location 本身又是一个 vision problem。
3. Pad 到最长边。解决了 distortion，但对 portrait images 来说，50%+ 的 tokens 会浪费在 padding 上。所有这些 pad tokens 都要付出 quadratic attention cost。

2024-2025 年的答案是：让 transformer 吃图像 native resolution 下的 patches，然后解决如何把 heterogeneous batch pack 成一个 sequence，同时不浪费 compute。

## 概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani et al., 2023）是证明这件事能 scale 的论文。想法很机械：

1. 对 batch 中每张 image，在选定 patch size（比如 14）下计算它的 native patch grid。
2. 把每张 image 的 patches flatten 成自己的 variable-length sequence。
3. 把所有 images 的 patches concatenate 成 batch 的一个 long sequence。
4. 构建 block-diagonal attention mask，让 image A 的 patches 只 attend 到 image A 内部。
5. 携带 per-patch position information（2D RoPE 或 fractional position embeddings）。

三张图像的 batch：336x336（576 tokens）、224x224（256 tokens）和 448x336（768 tokens），会变成一个 1600-token sequence，配一个 1600x1600 block-diagonal mask。没有 padding。没有 wasted compute。Transformer 可以处理任意 aspect ratios。

NaViT 还在训练中引入 fractional patch dropping：在 batch 中随机 drop 50% patches。这既能 regularize，也能加速训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实替代方案。给定一张 high-resolution image 和一个 fixed encoder（CLIP 或 SigLIP at 336），把图像 tile：

1. 从预定义集合中选择最适合 image aspect ratio 的 grid layout，如 (1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等。
2. 把完整 image 切进这个 grid；每个 tile 都变成 336x336 crop。
3. 同时生成 thumbnail：把整张 image resize 到 336x336，作为 global-context token。
4. 通过 frozen 336-encoder encode 每个 tile。Concatenate tile tokens + thumbnail tokens。

对于 672x672 image，2x2 grid 加 thumbnail：4 * 576 + 576 = 2880 visual tokens。昂贵但有效，因为 LLM 同时看到 local detail 和 global context。

当你的 encoder 是 frozen 且只支持一种 resolution 时，AnyRes 是首选路径。它会让大图 token count 爆炸（1344x1344 image 用 4x4 grid 是 9216 + 576 ≈ 9800 tokens，几乎填满大部分 8k LLM context）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了 Multimodal Rotary Position Embedding。它不用 NaViT 的 fractional positions，也不用 AnyRes 的 tile-and-thumbnail，而是让每个 patch 携带一个 3D position（temporal、height、width）。Query/key rotations 会处理任意 H、W 和 temporal length。

M-RoPE 不需要重新训练就能提供 native dynamic resolution。Inference 时输入任意 HxW image，patch embedder 产生 H/14 x W/14 tokens，每个 token 得到自己的 (t=0, r=row, c=col) position，RoPE 用正确 frequencies 旋转 attention，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这个方案。InternVL3 的 V2PE 是同一个想法，只是为不同 modality 使用 variable encoding。

不同于 AnyRes，M-RoPE 在 native resolution 下是 O(H x W / P^2) tokens，没有 multiplicative tile overhead。不同于 NaViT，它仍然期望每次 forward 处理单张 image。跨 resolutions batching 仍然需要在上层使用 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的 native-flex mode。单个模型在 inference 时服务多个 sequence lengths（256、729、1024 tokens）。内部在训练中使用 NaViT-style patch-n'-pack，并为每个 patch 使用 absolute fractional positions。卖点是：一个 checkpoint，在 inference 时根据任务选择 token budget。

Semantic task（classification、retrieval）用 256 tokens。OCR 或 chart understanding 用 1024 tokens。不需要重新训练。

### Packing mask

Block-diagonal mask 是多数实现容易踩坑的地方。对一个 packed sequence，长度为 `N_total`，覆盖 images `i=0..B-1`，每张 image 长度为 `n_i`，mask `M` 的 shape 是 `(N_total, N_total)`。如果两个 indices 落在同一张 image 的 block 中，`M` 为 1，否则为 0。可以用 cumulative length list 构建它：

```text
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中，这可以用 `torch.block_diag` 或 explicit gather 一行实现。FlashAttention 的 variable-length path（`cu_seqlens`）会完全跳过 dense mask，直接用 cumulative-length tensor 在 sequences 内部 attend。对典型 batches 来说，它比 dense mask 快约 10x。

### Token budgets

按任务选择策略：

- OCR / documents：1024-4096 tokens。SigLIP 2 NaFlex at 1024，或 AnyRes 3x3 + thumbnail。
- Charts and UI：384-448 native 下 729-1024 tokens。Qwen2.5-VL dynamic resolution，配 max pixels cap。
- Natural photos：256-576 tokens 就够了。Downstream LLM 已经看到足够信息。只在 content density 高的地方为 tokens 付费。
- Video：spatial pooling 后每帧 64-128 tokens，2-8 FPS。Lesson 12.17 会覆盖这个主题。

2026 年 production rule：选择 per-task max-pixels cap，在该 cap 内按 native aspect ratio encode，pack batch，并跳过 padding。Qwen2.5-VL 暴露的 `min_pixels` 和 `max_pixels` 正是这个 knob。

## Use It

`code/main.py` 为 heterogeneous image batch 实现 patch-n'-pack，使用 integer pixel coordinates。它会：

- 接收一组 (H, W) image sizes。
- 按 patch size 14 计算每张 image 的 patch sequence length。
- 把它们 pack 成一个总长度为 `sum(n_i)` 的 sequence。
- 构建 block-diagonal attention mask（dense 版本，便于理解）。
- 比较 packed cost、square-resize 和 AnyRes tiling。
- 为 mixed batch（receipt、chart、screenshot、photo）打印 token budget table。

运行它。输出的数字就是每个 2026 open VLM 都使用 patch-n'-pack 的原因。

## Ship It

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个 mixed-aspect-ratio workload（OCR、charts、photos、video frames）和 total-token budget，它会选择合适策略（NaFlex、AnyRes、M-RoPE 或 fixed-square），并输出 per-request configuration。在为产品 sizing VLM 时使用这个 skill，它能防止悄悄出现的 10x token blowup 吃掉 latency budgets。

## Exercises

1. 一张 receipt 是 600x1500（1:2.5）。Patch size 为 14 时，native-resolution tokens 有多少？Square-resize 到 336 后有多少？在实践中哪种更会损失 OCR accuracy？

2. 为四张图像的 batch 构建 block-diagonal mask，长度分别为 256、576、729、1024。验证 attention matrix 是 2585x2585，并且正好有 `256^2 + 576^2 + 729^2 + 1024^2` 个 non-zero entries。

3. 对一张 1792x896 image，patch 14，比较：(a) square-resize 到 336 后 encode，(b) AnyRes 2x1 + thumbnail，(c) M-RoPE at native。哪个使用最少 tokens？哪个保留最多 detail？

4. 实现 fractional patch dropping：给定 packed sequence，uniformly at random drop 50% tokens，并相应更新 block-diagonal mask。测量 mask sparsity 的变化。

5. 阅读 Qwen2-VL paper（arXiv:2409.12191）Section 3.2。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么上下界都重要。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT-style packing" | 把不同 images 的 variable-length patch sequences concatenate 到同一个 batch dimension 中 |
| Block-diagonal mask | "Packing mask" | Attention mask，限制每张 image 的 patches 只 attend 自己，而不是 pack 中的邻居 |
| AnyRes | "LLaVA-NeXT tiling" | 把 high-res image 切成 fixed-size tiles 的 grid，再加一个 global thumbnail；用 fixed encoder encode 每个 tile |
| NaFlex | "SigLIP 2 native-flex" | 单个 SigLIP 2 checkpoint 在 inference 时支持 256/729/1024-token budgets，不需要 retraining |
| M-RoPE | "Multimodal RoPE" | 3D rotary position encoding（time、row、column），能在没有 position tables 的情况下处理任意 H、W、T |
| cu_seqlens | "FlashAttention packing" | FlashAttention varlen path 使用的 cumulative-length tensor，用来替代 dense block-diagonal mask |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 的 per-request knobs，用于限制特别小或特别大输入的 token count |
| Visual token budget | "How many tokens per image" | 每张 image 大致产生的 patch tokens 数量；决定 LLM 的 prompt budget 和 attention cost |

## Further Reading

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
