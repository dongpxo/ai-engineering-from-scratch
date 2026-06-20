# Video-Language Models：Temporal Tokens 与 Grounding

> Video 不是一堆照片。一个 5 秒 clip 有 causal ordering、action verbs 和 event timing，这些都是 image model 无法表示的。Video-LLaMA（Zhang et al., 2023 年 6 月）发布了第一个带 audio-visual grounding 的 open video-LLM。VideoChat 和 Video-LLaVA 扩展了这个模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与 frontier proprietary models 的差距。每个系统处理 temporal tokens 的方式都不同：Q-former per clip、concat-pool per frame、TMRoPE per token。本课读取这些 patterns，构建 uniform-vs-dynamic frame sampler，并在 temporal grounding tasks 上评估。

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## 学习目标

- 解释为什么 temporal positional encoding 会独立于 vision encoder 改变 video VLM performance。
- 比较 uniform、dynamic-FPS 和 event-driven frame sampling 在 tokens-per-second 与 grounding accuracy 上的取舍。
- 描述 Q-former-per-clip（Video-LLaMA）、pooled-per-frame（Video-LLaVA）和 M-RoPE-per-token（Qwen2.5-VL）设计。
- 说出四个 video benchmarks：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

一段 1-minute video，在 30 FPS 下有 1800 frames。每 frame 196 visual tokens（224 分辨率的 ViT-B），总共就是 352k tokens，比任何 2024-era LLM context 都大。

有三种 reduction strategies：

1. Subsample frames（根据 content 取 1-8 FPS）。
2. Aggressively pool 每一帧的 patch tokens（3x3 或 4x4 bilinear pool）。
3. 用 Q-former 压缩：接收一个 16-frame clip，输出 64 tokens。

每种 trade-off 都不同。Subsampling 会丢 temporal detail。Pooling 会丢 spatial detail。Q-former 两者都丢一点，但节省 tokens。

Temporal position encoding 是另一条轴：模型怎么知道 frame 5 发生在 frame 6 之前？选项包括简单 1D temporal RoPE（Video-LLaMA）、learned temporal embeddings（Video-LLaVA），以及 TMRoPE（Qwen2.5-VL，完整 3D）。

## 概念

### Video-LLaMA：Q-former per clip + audio branch

Video-LLaMA（2023）是第一个 open video-LLM。Architecture：

- 16-frame clips at 2 FPS（所以是 8 秒）。
- Per-frame ViT features -> Video Q-former 对全部 16 frames 做 cross-attention -> 32 learned queries -> LLM。
- 并行 audio branch：waveform -> ImageBind audio encoder -> Audio Q-former -> 32 queries -> LLM。

优势：audio-visual joint reasoning。弱点：fixed clip length，无法做 arbitrary time grounding。

### VideoChat 和 Video-LLaVA

VideoChat 保留 Video-LLaMA 思路，但去掉 audio 并简化。Video-LLaVA（Lin et al., 2023）在 images 和 video frames 上训练同一个 visual encoder（“alignment before projection”），得到 unified representation。两者都是 frozen-CLIP-encoder + MLP + LLM。

二者都不能处理 long video。它们都是 8-16 frame systems。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE：Temporal-Modality Rotary Position Embedding。每个 patch token 都携带一个 `(t, h, w)` position，其中 `t` 是实际 timestamp，而不是 frame index。

它和简单 temporal embedding 的关键区别：

- Absolute time，而不是 index。模型看到的是 “at 4.2 seconds”，不是 “at frame 15”。
- Per-token rotation，而不是 per-clip。每个 visual token 都按自己的 timestamp 独立旋转。
- 兼容 dynamic FPS。如果这里用 2 FPS 采样、那里用 4 FPS 采样，TMRoPE 可以原生处理 uneven spacing。

TMRoPE 支持 “at what second does the cat jump?” 这类问题。模型可以输出 “at 4.2 seconds”。Video-LLaMA 只能说 “early in the clip.”

### Frame sampling strategies

Uniform：在 duration 上均匀采样 N frames。简单，但会错过 motion peaks。

Dynamic FPS：根据 motion intensity 自适应采样。Optical flow 或 frame differencing 会为 high-motion segments 选择更密集采样。Qwen2.5-VL 使用这种训练方式。

Event-driven：运行 lightweight detector，在 action 发生的位置采样更多 frames。VideoAgent 使用这种方式。

Keyframe + context：在 shot boundaries 采样，再加上一些 adjacent frames。适合 cinematic content。

### Pooling per frame

1 FPS 且每 frame 576 tokens 时，一个 5-minute clip 是 172,800 tokens。Qwen2.5-VL-72B 的 128k context 勉强接近，但成本很高。

3x3 bilinear pool 会把每 frame 降到 64 tokens，5 分钟是 19,200 tokens。对大多数 tasks 来说，这是 sweet spot。

Agent workflows 中 spatial detail 没那么重要，可以更激进地 pool（6x6 -> 每 frame 16 tokens）。

### 四个 video benchmarks

- VideoMME：综合 video understanding，覆盖 short + medium + long。
- TempCompass：fine-grained temporal reasoning，包含 “before” / “after” questions。
- EgoSchema：long-horizon first-person video。
- Video-MMMU：multimodal multi-discipline video questions。

完整 video-VLM evaluation 应该覆盖这四个。它们压测的轴不同：TempCompass 完全关注 ordering，EgoSchema 关注 3+ minute reasoning，VideoMME 覆盖多种 durations。

### Grounding output formats

Temporal grounding 的输出格式：

- Free text：“The cat jumps around the 4-second mark.” 易解析但不精确。
- Structured JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这种格式。
- Token-based：特殊 `<time>4.1</time>` tokens 与 answer 交织。Qwen2.5-VL 的内部格式。

Token-based 对 downstream use 最准确。Qwen2.5-VL 的 JSON output format 可以直接 parse。

### 2026 best practice

2026 年的 video VLMs 最佳实践：

- Encoder：SigLIP 2 with M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- Frame sampling：dynamic FPS（根据 motion 取 1-4），带 max-frame cap。
- Per-frame pooling：3x3 bilinear。
- Output：包含 time + event fields 的 structured JSON。
- Benchmarks：general 用 VideoMME + TempCompass；long-horizon 用 EgoSchema。

## Use It

`code/main.py` 包含：

- Uniform 和 dynamic-FPS frame samplers。
- 一个 toy temporal-grounding evaluator：给定 time T 的 “ground truth” event 和 model output，用 tolerance 评估 accuracy。
- 横向比较 Video-LLaMA（16 frames, Q-former）、Video-LLaVA（8 frames, MLP）、Qwen2.5-VL（dynamic FPS + TMRoPE）。

## Ship It

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个 video task（monitoring、action recognition、temporal grounding、summarization），它会选择 frame sampler、pooling factor、output format 和 expected accuracy tier。

## 练习

1. 对于一段 3-minute cooking demo，选择 uniform 还是 dynamic FPS。用 token count 证明你的选择。

2. TMRoPE 具体增加了什么，是 simple temporal embedding table 做不到的？

3. 写一个 VLM 可以学会输出的 temporal grounding JSON schema。包含 error cases。

4. 阅读 Video-LLaVA Section 3 中的 “Alignment Before Projection”。为什么它比训练 separate image and video encoders 更好？

5. 根据 VideoMME leaderboard，截至 2026 年，top open model 和 top proprietary model 之间差距多大？其中有多少来自 temporal encoding，有多少来自 base LLM scale？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | “Time-localized answers” | VLM 输出某个 event 发生的具体 timestamp range |
| TMRoPE | “Time-Multimodal RoPE” | 带 absolute timestamps 的 3D rotary position，由 Qwen2.5-VL 使用 |
| Dynamic FPS | “Motion-aware sampling” | 在 high-motion segments 采更多 frames，在 static segments 采更少 frames |
| Frame pooling | “Spatial compress per frame” | 在进入 LLM 前，用 bilinear interpolation 减少每 frame 的 patches |
| Video Q-former | “Clip compressor” | Cross-attention bottleneck，把 N frames 映射成 K learned queries |
| VideoMME | “Video bench” | 综合 short/medium/long video benchmark，2500+ samples |

## 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
