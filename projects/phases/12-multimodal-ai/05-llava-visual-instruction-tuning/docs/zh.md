# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素 token 拼接替换了 Flamingo 的 gated cross-attention，并在 158k 条由 GPT-4 根据纯文本 caption 生成的视觉指令轮次上训练。2023 到 2026 年间，任何构建过 VLM 的实践者，几乎都构建过某种 LLaVA 变体。LLaVA-1.5 加入 AnyRes。LLaVA-NeXT 提高分辨率。LLaVA-OneVision 用同一套 recipe 统一图像、多图和视频。本课会读懂这份 recipe，实现 projector，并解释为什么“更简单赢了”。

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## 学习目标

- 构建一个 2 层 MLP projector，把 ViT patch embeddings（dim 1024）映射到 LLM embedding dim（dim 4096）。
- 走通 LLaVA 两阶段 recipe：(1) 在 558k caption pairs 上做 projector alignment，(2) 在 158k 条 GPT-4 生成的视觉指令轮次上做 visual instruction tuning。
- 构造 LLaVA 格式 prompt，包含 image token placeholder、system prompt，以及 user/assistant turns。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 更省 token budget。

## 问题

BLIP-2 的 Q-Former（Lesson 12.03）把一张图像压缩为 32 个 token。干净、高效，benchmark 表现也不错。但它有两个问题。

第一，Q-Former 是可训练的，但它的 loss 不是最终任务。Stage 1 训练 ITC+ITM+ITG。Stage 2 训练 LM loss。Queries 学到的是某种中间表示，随后 LLM 还要把它解码。瓶颈中会丢失信息。

第二，Q-Former 有 188M 参数，并且在 LLaVA 2023 年的规模下，你必须把它和目标 LLM 共同设计。换 LLM，就要重训 Q-Former。换 vision encoder，也要重训。每个组合都是一个独立研发项目。

LLaVA 的回答简单到有点尴尬：取 ViT 的 576 个 patch token，让每个 token 通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后把全部 576 个 token 倒进 LLM 的输入序列。没有瓶颈。没有 stage 1 上奇怪 objective 的预训练。只用直接的 LM loss 训练 MLP。

数据从哪里来？LLaVA 的第二个洞察是：用 GPT-4（text-only）生成 instruction data。把一张图像的 COCO caption 和 bounding-box data 喂给 GPT-4，让它生成对话、描述和复杂推理问题。免费获得 158k 条 instruction-response turns。不需要人工标注。

结果是：一个 VLM，在 8 张 A100 上训练一天，MMMU 上超过 Flamingo，并发布了社区可以扩展的开放 checkpoint。到 2023 年底，它已经催生了 50+ 个 fork。

## 概念

### 架构

LLaVA-1.5 13B：

- Vision encoder：CLIP ViT-L/14 @ 336（stage 1 冻结，stage 2 可选解冻）。
- Projector：带 GELU activation 的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来也用 Llama-3.1-8B）。

图像 + 文本 prompt 的 forward pass：

```text
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM context 中的 576 个 token。在 2048 context 下，还剩 1472 个 token 给文本。在 32k context 下，这几乎只是舍入误差。

### Stage 1：projector alignment

冻结 ViT。冻结 LLM。只训练 2 层 MLP。数据集：558k image-caption pairs（LAION-CC-SBU）。Loss：以 projected image tokens 为条件，对 caption 做 language modeling。

在 batch 128 下跑一个 epoch，几个小时就能完成。Projector 学会把 ViT-space 映射到 LLM-space。不需要 task-specific supervision。

### Stage 2：visual instruction tuning

解冻 projector（仍然可训练）。解冻 LLM（通常全量，有时用 LoRA）。在 158k 条 visual-instruction turns 上训练。

Instruction data 是关键技巧。Liu 等人这样生成它：

1. 取一张 COCO image。
2. 提取文本描述（5 条人工 caption + bounding-box list）。
3. 用三个 prompt template 发给 GPT-4：
   - Conversation：“Generate a back-and-forth dialogue between a user and assistant about this image.”
   - Detailed description：“Give a rich, detailed description of the image.”
   - Complex reasoning：“Ask a question that requires reasoning about the image, then answer it.”
4. 把 GPT-4 输出解析成 (instruction, response) pairs。

这一切都没有直接接触图像，只接触文本描述。GPT-4 会幻觉出合理的图像内容。有一些噪声，但它确实有效：158k turns 足以解锁对话能力。

### 为什么社区复制了它

- 没有 stage-1-specific losses 需要调。全程都是 LM loss。
- Projector 以小时为单位训练，而不是以天为单位。
- LLM 可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需要重训 projector。
- Visual-instruction data pipeline 使用 GPT-4，便宜且可为新领域重新生成。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）加入：

- Academic-task data（VQA、OKVQA、RefCOCO）混入 instruction tuning。
- 更好的 system prompt。
- 2048 → 32k context。

LLaVA-NeXT（2024 年 1 月）加入：

- AnyRes：把高分辨率图像切成 2x2 或 1x3 的 336x336 crop 网格，再加一个全局低分辨率 thumbnail。每个 crop 变成 576 个 token；每张图总计大约 2880 个 visual tokens。OCR 和 chart tasks 明显提升。
- 使用 ShareGPT4V（高质量 GPT-4V captions）改进 instruction data mixture。
- 更强的 base LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

Lesson 12.08 会深入覆盖 OneVision。简短版本：它使用同样的 projector，但通过一个 curriculum 训练，在一个模型中用共享 visual-token budget 覆盖单图、多图和视频。

### 与 Q-Former 的对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图 visual tokens | 32 | 576（base）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | 只有 LM |
| LLM drop-in | 需要重训 | 最小重训即可替换 |
| Multi-image | 别扭 | 自然（concat） |
| Video | 别扭 | 自然（per-frame concat） |
| Token budget | 小 | 大 |

MLP 赢在简单性和 token 灵活性。Q-Former 赢在 token budget。到 2023 年底，token budget 已经不再是绑定约束（LLM context 增长到 32k-128k+），于是简单性占了上风。

### Prompt 格式

```text
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个 placeholder token。Tokenization 之前，它会被替换成 576 个 visual tokens（AnyRes 下是 2880 个）。Tokenizer 看到的序列会比训练时稍长，但 LLM 能处理这种新输入，因为 stage 1 已经教会它如何使用这些 token。

### 参数经济性

LLaVA-1.5-7B 参数拆解：

- CLIP ViT-L/14 @ 336：303M（stage 1 冻结，stage 2 经常解冻）。
- Projector（2x linear）：约 22M 可训练参数。
- Llama-7B：7B。
- 总计：7.3B 参数。Stage 2 可训练：完整 7B + 22M projector。

Stage 2 训练成本：8xA100 上约 20 小时。这是关键数字：一天，一台 node，可复现。这就是 LLaVA 扩散开的原因。

## Use It

`code/main.py` 实现：

1. 用纯 Python 实现的 2 层 MLP projector（toy scale 下 dim 16 → 32 → 32）。
2. Prompt-building pipeline：system prompt + 用 N 个 projected tokens 替换 `<image>` + user turn + assistant generation placeholder。
3. 可视化 576-token visual block 在 LLM context 中的样子（占 2k / 32k / 128k context 的百分比）。

## Ship It

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA-family checkpoint，它会运行一套 10-prompt vibes-eval suite（3 个 captioning、3 个 VQA、2 个 reasoning、2 个 refusal），并报告一个人类可读的 scorecard。它不是 benchmark，而是一个 smoke test，用来确认 projector 和 LLM 连接良好。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2 层 MLP projector 的可训练参数量。带 GELU 和 bias 时，它占 LLaVA-13B 的多少比例？

2. 构造一个“refusal”案例的 LLaVA prompt：图像中包含私人个体。写出预期 assistant response。为什么 LLaVA 应该 zero-shot 拒绝？需要什么训练数据来强化这种拒绝？

3. 阅读 LLaVA-NeXT blog 中的 AnyRes 部分。计算一张 1344x672 图像在 AnyRes 下的 visual token 数量。与 336x336 下的 base 576 tokens 对比。

4. LLaVA stage-1 projector 使用 captions 上的 LM loss 训练。如果跳过 stage 1，直接进入 stage 2（visual instruction tuning）会怎样？引用 Prismatic VLMs ablation（arXiv:2402.07865）回答。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO captions 生成 instructions。对一个新领域（医学 X-ray、卫星图像），描述生成 domain instructions 的四步 data pipeline。每一步可能出什么问题？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Projector | “MLP bridge” | 带 GELU 的 2 层 MLP，把 ViT dim 映射到 LLM dim |
| Image token | “<image> placeholder” | 推理前被 N 个 projected visual tokens 替换的 prompt marker |
| Visual instruction tuning | “LLaVA stage 2” | 在 GPT-4 生成的 (image, instruction, response) triplets 上训练 |
| Stage 1 alignment | “Projector pretraining” | 冻结 ViT 和 LLM，用 captions 上的 LM loss 训练 projector |
| AnyRes | “Multi-crop tiling” | 把高分辨率图像切成 tile grid，并拼接每个 tile 的 visual tokens |
| LLaVA-Instruct | “GPT-4-generated” | 从 COCO captions + GPT-4 合成的 158k instruction-response pairs |
| Vision encoder freeze | “Backbone locked” | CLIP weights 在 stage 1 不更新，有时 stage 2 也不更新 |
| ShareGPT4V | “Better captions” | GPT-4V 生成的 1M dense captions，用于更高质量 alignment |
| VQA | “Visual question answering” | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | “Design-space paper” | Karamcheti 2024 ablation，系统测试 projector 和 data choices |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dense captions 数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — design-space ablations。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图、多图和视频。
