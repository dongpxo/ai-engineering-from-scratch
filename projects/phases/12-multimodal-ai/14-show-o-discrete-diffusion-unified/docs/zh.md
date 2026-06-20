# Show-o 与离散扩散统一模型

> Transfusion 混合连续和离散表示。Show-o（Xie et al., 2024 年 8 月）走向另一边：文本 token 使用 causal next-token prediction，图像 token 使用 MaskGIT 风格的 masked discrete diffusion。两者都放在一个带 hybrid attention mask 的 transformer 里。结果是在一个 backbone、每个模态一个 tokenizer、一个 loss formulation（从 next-token 扩展到 masked prediction）下统一 VQA、text-to-image、inpainting 和混合模态生成。本课会走读 Show-o 设计——为什么 masked discrete diffusion 是一个并行、少步数图像生成器——并与 Transfusion 和 Emu3 对比。

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## 学习目标

- 解释 masked discrete diffusion：均匀 mask tokens，然后让 transformer 恢复它们的 schedule。
- 在速度和质量上比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 在一个 checkpoint 中处理的三类任务：T2I、VQA、image inpainting。
- 选择一种 masking schedule（cosine、linear、truncated），并推理它对 sample quality 的影响。

## 问题

Transfusion 的双 loss 训练可行，但动态更棘手——continuous diffusion loss 与 discrete NTP loss 位于不同数值尺度。平衡 loss weights 变成超参数搜索。架构有效，但复杂。

Show-o 的回答是：让两个模态都保持离散（像 Chameleon 一样），但用 masked discrete diffusion 并行生成图像，而不是顺序生成。训练目标变成单一 masked-token-prediction，它自然泛化了 next-token-prediction。

## 概念

### Masked discrete diffusion（MaskGIT）

Chang 等人（2022）的原始 MaskGIT 技巧很优雅。从一张完全被 mask 的图像开始（每个 token 都是特殊 `<MASK>` id）。每一步并行预测所有 masked tokens，然后保留置信度最高的 top-K 预测，并重新 mask 其余部分。经过约 8-16 次迭代后，所有 token 都被填满。每一步 unmask 多少 token 的 schedule 需要调优；cosine schedule 表现很好。

训练很简单：从 [0, 1] 中均匀采样一个 masking ratio，将它应用到图像的 VQ tokens 上，训练 transformer 恢复被 mask 的 token。这正是 BERT 对文本做过的事，只是扩展到了图像生成。

### Show-o：一个 transformer，hybrid mask

Show-o 把 MaskGIT 放进 causal-language-model transformer。Attention mask 是：

- Text tokens：causal（标准 LLM）。
- Image tokens：在 image block 内 full bidirectional（因此 masked tokens 在预测时能看到所有其他 image tokens）。
- Text-to-image：text attends to prior images，image attends to prior text。

训练在以下任务之间交替：

1. 文本序列上的标准 NTP。
2. T2I samples：text → image，其中 image tokens 被 mask，使用 masked-token-prediction loss。
3. VQA samples：image → text，其中 text tokens 被 mask（本质上就是 NTP）。

统一 loss 是 `<MASK>` tokens 上的 cross-entropy，覆盖文本 NTP（只有最后一个 token 被“mask”）和图像 masked-diffusion（随机子集被 mask）。

### 并行采样

Show-o 生成一张图像大约需要 16 步，而不是约 1000 步（按 token 自回归）或约 20 步（扩散）。每一步并行预测所有 masked tokens；提交 top-K 高置信度 token；重复。

对比：

- Chameleon / Emu3（tokens 上的 autoregressive）：N_tokens 次 forward pass，每张图通常 1024-4096 次。
- Transfusion（continuous diffusion）：约 20 步，每步一次完整 transformer pass。
- Show-o（masked discrete diffusion）：约 16 步，每步一次完整 transformer pass。

在同等规模模型下，Show-o 比 Chameleon 快，步数大致接近 Transfusion，但每步成本更低（discrete vocab logits vs continuous MSE loss）。

### 一个 checkpoint 中的任务

Show-o 推理时通过 prompt format 选择任务，支持四类任务：

- Text generation：标准自回归文本输出。
- VQA：image in, text out。
- T2I：text in，通过 masked discrete diffusion 输出 image。
- Inpainting：输入部分 tokens 被 mask 的图像，并填充它们。

Inpainting 能力来自 masked-prediction training，可以算是免费获得。Mask 掉 VQ-token grid 的一个区域，输入其余部分和文本 prompt，预测 masked tokens。

### Masking schedule

每一步 unmask 多少 token 的 schedule 会塑造质量。Show-o 推荐 cosine：

```text
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Step 0 时所有 token 都被 mask（ratio 1.0）。Step T 时没有 token 被 mask。Cosine 会把质量集中在中间 ratio 区域，因为那里预测信息量最大。Linear schedule 也能工作，但更快进入 plateau。

### Show-o2

Show-o2（2025 后续，arXiv 2506.15564）扩展 Show-o：更大的 LLM base、更好的 tokenizer、改进的 mask schedule。架构模式保持不变。

### Show-o 的位置

在 2026 年 taxonomy 中：

- Discrete tokens + NTP：Chameleon、Emu3。简单，但推理慢。
- Discrete tokens + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。并行采样，但仍受 tokenizer 有损压缩限制。
- Continuous + diffusion：Transfusion、MMDiT、DiT。质量最高，训练更复杂。
- VLM 中的 continuous + flow matching：JanusFlow、InternVL-U。最新方向。

按任务选择：当你想要一个开放模型同时处理 T2I + inpainting + VQA，并且速度合理时，选 Show-o；当质量最重要且你能承受双 loss 管线时，选 Transfusion。

## Use It

`code/main.py` 模拟 Show-o sampling：

- 一个 16 个 VQ tokens 的 toy grid。
- 一个 mock “transformer”，根据 prompt 和当前已 unmask tokens 预测 logits。
- 使用 cosine schedule，在 8 步内进行并行 masked sampling。
- 打印中间状态（mask pattern evolution）和最终 tokens。

运行它，观察 mask 一步步消融。

## Ship It

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个同时需要理解（VQA、captioning）和生成（T2I、inpainting）、且有 open-weights 约束的产品，它会在 Show-o family、Transfusion/MMDiT family、Emu3 / Chameleon family 之间做选择，并给出具体取舍。

## 练习

1. Masked discrete diffusion 大约 16 步采样。为什么不是 1 步？如果在 step 0 就 unmask 所有 token，会坏在哪里？

2. Inpainting 对 masked diffusion 来说是免费的。提出一个真实或假设产品用例，其中 Show-o 的 inpainting 胜过专用模型。

3. Cosine schedule vs linear schedule：追踪 T=8 时每一步 unmasked tokens 数量。哪一个更平衡？

4. 一张 512x512 的 Show-o 图像是 1024 tokens。若 vocab K=16384，模型输出 `1024 * log2(16384) = 14,336` bits（约 1.75 KiB）数据。Stable Diffusion 输出 `512*512*24` bits = 6,291,456 bits（约 768 KiB）原始像素。压缩比是多少？它换来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的 class-conditional autoregressive image model 与 Show-o 的 masked approach 有什么不同？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Masked discrete diffusion | “MaskGIT-style” | 训练模型预测 masked tokens；推理时迭代 unmask 最高置信度预测 |
| Cosine schedule | “Unmask schedule” | 推理步骤中 mask ratio 的衰减；把置信度增长集中在中间区域 |
| Parallel decoding | “All tokens at once” | 每一步用一次 forward pass 预测完整 masked token 序列，然后提交 top-K |
| Hybrid attention | “Causal + bidirectional” | 对 text tokens 使用 causal，对 image blocks 内使用 bidirectional 的 mask |
| Inpainting | “Fill-in generation” | 以部分 tokens 被 mask 的图像为条件，预测缺失部分；由训练目标免费带来 |
| Commitment rate | “Top-K per step” | 每次迭代声明“完成”的 token 数量；控制推理与质量之间的取舍 |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
