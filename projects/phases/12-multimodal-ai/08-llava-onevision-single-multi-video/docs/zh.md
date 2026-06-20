# LLaVA-OneVision：一个模型处理单图、多图与视频

> 在 LLaVA-OneVision（Li et al., 2024 年 8 月）之前，open-VLM 世界有几条彼此分离的 lineage：LLaVA-1.5 处理 single images，Mantis 和 VILA 这类模型处理 multi-image，Video-LLaVA 和 Video-LLaMA 处理 video。每条路线都赢过自己的 benchmark，也都在其他场景上失败。LLaVA-OneVision 认为，一个 single curriculum 可以训练出同时主导三种场景的模型，而且 emergent task-transfer effects（single-image skills exported to video、multi-image reasoning exported to single-image）胜过一组 specialists 的总和。Recipe 看起来很简单：一个在不同场景中保持恒定的 visual-token budget，加上一条从 single-image 到 OneVision（multi-image）再到 video 的显式 curriculum。本课读取这个 budget、curriculum 和 emergent behaviors。

**Type:** Build
**Languages:** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 minutes

## 学习目标

- 设计一个在 single-image、multi-image 和 video inputs 之间保持恒定的 visual-token budget。
- 安排一条 training curriculum，让 skills 从 single-image 迁移到 video，同时避免 catastrophic forgetting。
- 解释为什么在相同 parameter count 下，curriculum 做对时 single model 会胜过 specialists。
- 说出 LLaVA-OneVision 报告的三种 emergent capabilities：multi-camera reasoning、set-of-mark prompting、iPhone-screenshot agent。

## 问题

Image、multi-image 和 video 会以不同方式压迫模型。

Single-image 需要 high-resolution tokens（AnyRes，约 2880 visual tokens）来捕捉 OCR 和 fine detail。每个 sample 的 budget：1 张 image，2880 tokens。

Multi-image 需要若干张 moderate resolution images（每张约 576 tokens），这样跨图 reasoning 才能放进 context。每个 sample 的 budget：4-8 张 images，每张 576 tokens，总共 2300-4600 tokens。

Video 需要许多低分辨率 frames（pooling 后每 frame 约 196 tokens）来捕捉 temporal dynamics。每个 sample 的 budget：8-32 frames，每帧 196 tokens，总共 1600-6200 tokens。

如果训练 separate models，你可以只选一个 budget。如果训练一个 model，你就需要 budget 在不同场景之间合理缩放，同时不炸掉 context。

在 OneVision 之前，默认答案是 “train one scenario, ignore the others.” Video-LLaVA 用额外 training stages 把 video retrofit 到 image model 上。LLaVA-NeXT 用 tiling 添加 multi-image 支持。它们都没有干净地同时处理三种场景。

## 概念

### OneVision token budget

LLaVA-OneVision 选择一个统一的 visual-token budget：每个 sample 约 3000-4000 tokens，并按场景做不同分配：

- Single image：AnyRes-9（3x3 tiles + thumbnail），每个 tile 384 分辨率、729 patches， aggressive 2x2 bilinear pooling → 每 tile 182 tokens。总计：9 * 182 + 182 = 1820 tokens。或者 AnyRes-4，每 tile 729 → 2916 + 729。
- Multi-image：每张 image 使用 moderate resolution（384，无 tiling），729 tokens，不 pooling。6 张 images 的 budget → 4374 tokens。
- Video：32 frames，每帧 384 resolution，使用 aggressive 3x3 bilinear pool → 每 frame 81 tokens。总计：32 * 81 = 2592 tokens。

这个分配保持总 tokens 大致恒定。LLM 不会看到炸掉 context 的 batch。Encoder 在不同场景中产生不同 geometry，但 LLM 消耗的是同一个 budget。

### 三阶段 curriculum

LLaVA-OneVision 分三阶段训练：

1. Single-image SFT（stage SI）。所有数据都是 single-image-plus-text。在 high-resolution AnyRes input 上训练。这一阶段教 perception、OCR 和 fine-grained understanding。使用 LLaVA-NeXT 数据以及 OneVision-specific single-image data。
2. OneVision SFT（stage OV）。混合 single-image + multi-image + video（uniformly sampled frames）。在 unified token budget 上训练。这一阶段教模型处理 heterogeneous batch shapes。不 reset weights，而是从 stage SI 继续。
3. Task transfer（stage TT）。继续使用 target task mix，通常根据产品目标加重 multi-image 或 video。Deployment 前可选 fine-tune。

关键点：curriculum order 很重要。即使用同样数据，先训练 video 或先训练 multi-image，也会比 single-image-first 产生更差的 image performance。论文显式做了 ablation。

### 为什么 curriculum 有效

Single-image training 构建 perceptual base。Patch tokens 携带 fine-grained visual features；LLM 学会把它们和 text 整合。Multi-image 和 video 引入 structural challenges（哪张图是哪张、什么先发生），如果没有强 perceptual base，这些很难学。

如果从零开始把所有场景混在一起训练，模型会 underfit perception（每个 batch 中 single-image data 有限），并 overfit structure（大量 multi-image / video data）。结果是：模型会遵循 cross-image reasoning patterns，但视觉理解很浅。

Curriculum ordering 先通过 stage SI 获得 perception strength，再通过 stage OV 获得 compositional/temporal reasoning，同时不丢掉任何一边。

### 跨场景 emergent skills

LLaVA-OneVision paper 报告了三种 emergent capabilities：

1. Multi-camera reasoning。模型分别在 multi-image + video 上训练；inference 时，被要求推理一个 multi-camera driving scene。即便训练中没见过这个精确格式，模型也能正确整合多个 views。
2. Set-of-mark prompting。用户用 numbered marks 标注 image 中的 objects；模型推理 “mark 3 相对 mark 7 在做什么”。训练中既没有 marks，也没有 annotation；这个能力来自 spatial grounding + multi-image reference 的组合。
3. iPhone-screenshot agent。用户提供 iPhone screen screenshot，并要求模型规划下一次点击。模型训练过 UI screenshots、用户 workflow videos 和 multi-image before/after pairs，于是泛化到 agent 用例。

这些不是被训练出来的 tasks；它们从 curriculum 的 compositional structure 中涌现。

### Visual-token pooling

Token budget 需要 pooling。OneVision 在 2D patch grid 上使用 bilinear interpolation：24x24 = 576 patches 变成 12x12 = 144（2x factor）或 8x8 = 64（3x factor）。Pooling 在 patch-grid space 中完成，而不是在 token space 中完成，这样可以保留 locality。

每个场景的 pooling factor 本身也是 hyperparameter。更少 pooling = 更多 tokens = 更丰富 representation。更多 pooling = 更少 tokens = 能放进更多 frames / images。

### LLaVA-OneVision-1.5

2025 年的 follow-up（LLaVA-OneVision-1.5, arXiv 2509.23661）在 training data、model weights 和 code 上 “fully open”。它在一些 benchmarks 上追平 proprietary gap，并把 recipe 民主化。同样的 curriculum，更多数据，更好的 base LLM。没有 architecture change。

### 与 Qwen2.5-VL 对比

Qwen2.5-VL（Lesson 12.09）做了不同选择。它使用 M-RoPE 和 dynamic FPS，而不是 fixed pooling。它的 budget 会随 input 缩放：1-minute video 比 5-second video 使用更多 tokens。LLaVA-OneVision 固定 budget 并缩放 pooling。两者都有效；它们是在 configurability 和 predictability 之间做取舍。

## Use It

`code/main.py` 是一个 OneVision-style VLM 的 curriculum 和 budget planner。给定每个 sample 的 token budget 和目标 scenario mix（比如 40% single-image、30% multi-image、30% video），它会：

- 为每个 scenario 分配 resolution、pooling factor 和 frames。
- 检查每个 scenario 是否都放得进 shared budget。
- 报告 expected token count、LLM FLOPs，以及哪些 scenarios under-tokenized。
- 打印 stage-by-stage training schedule。

用它规划 OneVision fine-tune，或者 sanity-check 一个 VLM deployment 的 per-request cost。

## Ship It

本课产出 `outputs/skill-onevision-budget-planner.md`。给定 target task distribution 和 per-sample budget，它会输出 AnyRes factor、per-frame pooling、video frame count 和 curriculum stage weights。每当你训练或 fine-tune 一个 unified-scenario VLM 时，都使用它。

## 练习

1. 你的产品支持 80% single-image、10% multi-image（2-4 images）、10% video（8-16 frames）。设计 token budget。因为不做 heavy multi-image 而省下的 extra budget，你会放到哪里？

2. 阅读 LLaVA-OneVision Section 4.3（emergent capabilities）。提出一个 curriculum 很可能解锁、但论文没有报告的第四种 emergent skill。

3. 交换 curriculum order：先训练 multi-image，再训练 single-image，最后训练 video。预测哪些 benchmarks 会退化，以及为什么。

4. 论文报告的 video benchmarks 只用每个 sample 8 frames 训练。这能泛化到 inference 时的 30-second videos 吗？最先坏掉的是 token budget，还是 temporal reasoning？

5. 把 24x24 patches bilinear pooling 到 12x12，按每个维度看是 4x reduction。用 stdlib Python 实现 pooling，并验证每个 2x2 block 的 mean 是否匹配 bilinear output。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OneVision scenario | “Single-image, multi-image, or video” | Unified VLM 处理的三种 input shapes 之一；budget 在它们之间保持恒定 |
| Token budget | “How many tokens per sample” | 每个 training / inference sample 中 LLM 看到的 total visual tokens，通常 3000-4000 |
| Curriculum | “Training order” | 为 emergent transfer 选择的 stage ordering（single-image → multi-image → video） |
| Bilinear pooling | “Token shrink” | 对 patch grid（2D）应用 bilinear interpolation，在保留 locality 的同时减少 token count |
| Emergent skill | “Not trained, still works” | Inference 时出现的能力，训练数据中没有匹配任务，来自 curriculum composition |
| AnyRes-k | “k-tile setup” | k 个固定 resolution sub-tiles 加一个 thumbnail，典型 k ∈ {4, 9} |
| Task transfer | “Cross-scenario generalization” | Single-image 学到的 skills 通过 shared backbone 应用到 video，反之亦然 |

## 延伸阅读

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
