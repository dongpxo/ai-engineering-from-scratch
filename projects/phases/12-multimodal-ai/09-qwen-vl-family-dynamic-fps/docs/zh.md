# Qwen-VL 家族与 Dynamic-FPS Video

> Qwen-VL 家族——Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025）——是 2026 年最有影响力的开放视觉语言模型谱系。每一代都做了一个决定性架构押注，并在 12 个月内被开放生态复制：通过 M-RoPE 实现原生动态分辨率、带绝对时间对齐的 dynamic-FPS 采样、ViT 中的 window attention，以及结构化 agent 输出格式。到 Qwen3-VL，recipe 已经稳定下来：2D-RoPE-ViT encoder 接收原生宽高比输入，MLP projector 接入大型 Qwen3 language base，训练阶段把 OCR、grounding 和 agent behavior 当作一等目标。本课按时间顺序阅读这个家族，让你理解每个旋钮为什么在那里。

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## 学习目标

- 计算 M-RoPE 的三轴旋转（temporal、height、width），并解释为什么三轴都需要。
- 为视频选择 dynamic-FPS 采样策略，并推理 tokens-per-second 与 event-detection accuracy 之间的取舍。
- 按顺序说出 Qwen-VL 四代升级，以及每一代解锁了什么能力。
- 接入 Qwen2.5-VL 风格的 JSON agent 输出格式，并从 VLM 响应中解析结构化 tool calls。

## 问题

Qwen-VL 于 2023 年 8 月发布，是对 LLaVA-1.5 和 BLIP-2 的直接回应。Qwen 团队瞄准的差距有三类：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 运行在 336x336。对照片还行，但对中文发票或密集电子表格截图没什么用。Qwen-VL 的第一个创新是 448x448 和 grounded bounding-box output，让模型能指向东西。

视频：Video-LLaMA 堆叠每帧 encoder 并把它们喂给 LLM。短 clip 可以，面对时间轴才是信号的多分钟视频就不够了。Qwen 团队想要一个理解时间的单一 encoder。

结构化输出：LLaVA 输出自由文本。Agent 需要 JSON。Qwen-VL 在显式 JSON 输出格式上训练，包括把 bounding-box coordinates 作为文本输出。

每一代 Qwen-VL 都在延展这三条轴线之一。

## 概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为 encoder（2.5B 参数）、LLama-compatible Q-Former（1-step with 256 queries）、Qwen-7B base。贡献：

- 448x448 分辨率（当时开放 VLM 的 SOTA）。
- Grounding：在带显式 coordinate-token 输出的 image-text pairs 上训练。“The cat is at <box>(112, 204), (280, 344)</box>”。
- 从一开始就做中文 + 英文多语言训练。

当时的 benchmark：英文上接近 GPT-4V，中文上占优。Grounding supervision 才是真正的 headline。

### Qwen2-VL（2024 年 9 月）— M-RoPE 与原生分辨率

Qwen2-VL 用原生动态分辨率 ViT encoder 替换了固定分辨率 + Q-Former stack。关键变化：

- 原生动态分辨率。ViT 接受任意 HxW，只要能被 28 整除（patch 14，再做 2x spatial merge）。一张 1120x672 图像（40x24 merged patches）会产生 960 个 visual tokens。不 resize，不 tiling，不 thumbnail。
- M-RoPE（Multimodal RoPE）。每个 token 携带 3D position（t, h, w），而不是 1D。对图像 t=0，对视频 t=frame_index。RoPE 会按每个轴的频率旋转 query/key vectors。没有 positional embedding table。
- MLP projector。去掉 Q-Former；在 merged patch tokens 上使用 2 层 MLP。
- 支持 dynamic FPS 的视频。默认以 1-2 FPS 采样视频，但模型接受任意 frame counts。

结果：Qwen2-VL-7B 在多个多模态 benchmark 上追平 GPT-4o，并在 DocVQA 上击败它（94.5 vs 88.4）。架构变化是决定性动作。

### Qwen2.5-VL（2025 年 2 月）— dynamic FPS + absolute time

Qwen2.5-VL 的最大变化是视频。Dynamic FPS 不只是“需要时采样更多帧”。论文形式化了：

- 绝对时间 token。不是使用位置索引（frame 0, 1, 2...），而是使用真实时间戳。“At 0:04, the cat jumps.” 模型会看到与 frame tokens 交错的 `<time>0.04</time>` tokens。
- Dynamic FPS。慢速视频用 1 FPS，动作视频用 4+ FPS。用户或训练者选择；M-RoPE 会适配。
- ViT 中的 window attention。Spatial attention 使用 windowed 形式（block 内局部 attention）提升吞吐，每隔几层加入 global attention。
- 显式 JSON 输出格式。在 tool-call data 上训练："{\"tool\": \"click\", \"coords\": [380, 220]}"。开箱即 agent-ready。
- MRoPE-v2 scaling。Position 会随最大输入大小缩放，因此 10 分钟视频不会耗尽频率范围。

Benchmarks：Qwen2.5-VL-72B 在多数视频 benchmark 上超过 GPT-4o，在文档上追平 Gemini 2.0，并为 GUI grounding 创下开放模型 SOTA（ScreenSpot：84% accuracy vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是一次巩固式的增量升级，而不是重造：更大的 LLM backbone（Qwen3-72B）、扩展训练数据、改进 OCR、通过 Qwen3 “thinking mode”增强推理。ViT 和 M-RoPE 保持不变。论文重点放在数据和训练改进，而不是架构。

谱系结论：到 2025 年，Qwen-VL 架构已经稳定。后续 generation 扩展的是 compute 和 data，而不是 primitives。

### M-RoPE 的数学形式

经典 RoPE 会按 position `m` 旋转维度为 `d` 的 query `q`，使用成对坐标：

```text
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 把 hidden dim 分成三个 band。假设 `d = 96`，把 32 维分给 temporal，32 维分给 height，32 维分给 width。每个 band 按自己的轴位置旋转。一个位于 (t=5, h=10, w=20) 的 patch，会把 `R_t(5)`、`R_h(10)`、`R_w(20)` 分别应用到三个 band。

Text tokens 使用 `t = text_index, h = 0, w = 0`（或某种归一化选择），保持兼容性。Video frames 使用 `t = frame_time, h = row, w = col`。单图使用 `t = 0`。

好处是：一个 position encoding 就能处理文本、图像和视频，不需要分支代码，也不需要不同 position tables。

### Dynamic-FPS 采样逻辑

给定一个时长为 `T` 秒的视频和目标 token budget `B`：

1. 计算你能负担的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择满足 `fps <= fps_max` 的目标 FPS。
3. 如果 motion 很高（optical-flow heuristic 或显式用户请求），选择更高 FPS。如果 motion 很低，选择更低 FPS。
4. 按所选 FPS 均匀采样；在 frames 之间插入 `<time>t</time>` tokens。

Qwen2.5-VL 会隐式训练这种逻辑；推理时用户通过 `fps` 参数控制。一个 60 秒 action sequence，以 4 FPS、每帧 81 tokens 计算，总计 19440 tokens，在 32k context 内可管理。

### 结构化 agent 输出

Qwen2.5-VL 的 agent training 显式目标是结构化 tool calls：

```json
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出做 JSON.parse。对比自由文本“click at (1024, 512)”，后者需要正则和歧义处理。这个转变解释了为什么 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跳到 84%。

## Use It

`code/main.py` 实现：

- 针对混合 text、image patches 和 video frames 的 packed sequence，计算 M-RoPE position。
- Dynamic-FPS sampler：给定 (duration, budget, motion_level)，选择 FPS 并输出 frame timestamps。
- 一个 toy Qwen2.5-VL JSON-output parser，处理带 coordinate fields 的 tool-call responses。

运行它，然后在 5 分钟视频上把 fixed-FPS 换成 dynamic-FPS，感受两者差异。

## Ship It

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定一个视频任务（monitoring、agent、action recognition、accessibility），它会输出 Qwen2.5-VL 配置（frame budget、FPS strategy、window-attention flag、agent-output mode）和延迟估计。每当你为视频产品部署 Qwen-VL-family model 时，都可以使用它。

## 练习

1. 对 hidden 48（每个 band 16，base theta 10000）下位于 (t=3, h=5, w=7) 的 patch 计算 M-RoPE rotations。展示每个 band 前三对坐标的 rotation angles。

2. 一段 10 分钟的安防摄像头录像在 1 FPS 下会产生多少帧？在 384 分辨率、3x pool 下，总 token 数是多少？Qwen2.5-VL 默认 32k context 能否处理？

3. 为 30 秒网球回合、30 秒食谱 demo、30 秒 UI-agent 录屏分别选择 FPS。用 dynamic-FPS 逻辑解释理由。

4. Qwen2.5-VL 完全去掉 Q-Former。为什么简单 MLP 在 2025 年可行，而在 2023 年不可行？（提示：data scale 和 encoder quality。）

5. 把三个 Qwen2.5-VL JSON tool-call outputs 解析成 Python dict。Malformed JSON 会在哪里失败？Qwen cookbook 推荐什么恢复策略？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| M-RoPE | “Multimodal RoPE” | 在 hidden dim 中带 temporal、height、width bands 的 3D rotary position embedding |
| Dynamic FPS | “Smart sampling” | 根据 motion、duration 和 token budget 为每个视频选择 frame sampling rate |
| Absolute time token | “Timestamp token” | 序列中交错插入的 `<time>t</time>`，让模型看到真实秒数而不是 frame index |
| Window attention | “Local attention” | 为速度把 spatial self-attention 限制在小窗口内，并周期性加入 global attention |
| Structured agent output | “JSON mode” | 训练数据监督 VLM 输出可解析 JSON，包含 coords 和 tool names |
| min_pixels / max_pixels | “Resolution bounds” | Qwen2.5-VL 的 per-request 控制项，用来限制总像素数，从而限制 token 数 |
| Grounding | “Point-at-it” | 以文本 token 输出 bounding-box coordinates；从 Qwen-VL v1 开始使用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
