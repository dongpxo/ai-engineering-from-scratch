# InternVL3：原生多模态预训练

> InternVL3 之前的每个 open VLM 都遵循同一个三步配方：拿一个在数万亿 text tokens 上训练好的 text LLM，接上 vision encoder，然后 fine-tune 连接处。这能工作，但会积累 alignment debt：text LLM 已经把完整 pretraining budget 花在 pure text 上，并不原生理解 visual tokens。当你事后加入 vision，LLM 必须重新学习如何把 visual input 和 text reasoning 联系起来，同时还不能忘掉 text。InternVL3（Zhu et al., 2025 年 4 月）拒绝 post-hoc 方法：一次 pretraining run，从第一步开始就交错训练 text 和 multimodal。结果是在 78B open params 下，MMMU-Pro 上匹配 Gemini 2.5 Pro。本课阅读 native pretraining 的理由，以及当你采用它时会改变什么。

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 minutes

## 学习目标

- 解释为什么 post-hoc VLM training 会积累 alignment debt，并引用三个可测量症状（catastrophic forgetting、answer drift、visual-text inconsistency）。
- 描述 InternVL3 的 native pretraining corpus mix，以及为什么 text : interleaved : caption 的比例重要。
- 比较 V2PE（variable visual position encoding）与 Qwen2-VL 的 M-RoPE。
- 说出 Visual Resolution Router (ViR) 和 Decoupled Vision-Language (DvD) 这两个 deployment optimizations。

## 问题

Post-hoc VLM training 是默认做法。LLaVA、BLIP-2、Qwen-VL、Idefics 都会拿一个已经 pretrained 的 LLM（Llama、Vicuna、Qwen、Mistral）再加 vision。训练阶段通常长这样：

1. Frozen LLM + frozen vision encoder + trainable projector，在 caption pairs 上训练，用来 align embeddings。
2. Unfreeze LLM，在 instruction data（LLaVA-Instruct、ShareGPT4V）上训练。
3. 可选的 task-specific fine-tune。

Alignment debt 会出现三个症状：

- Catastrophic forgetting。Post-hoc VLM 会忘掉 text-only skills。GSM8K 分数下降 5-10 点。Hellaswag 分数下降。Pure-text agents 退化。
- Answer drift。同一个 visual question 的小幅措辞变化会得到不同答案。Vision encoder 与 LLM 的连接，比 LLM 自身 tokens 的绑定更弱。
- Visual-text inconsistency。VLM 可以正确描述一张 image，然后回答一个与自己描述相矛盾的问题。Visual tokens 不像 text 那样参与 LLM 的 internal consistency checks。

这些症状有充分记录。MM1.5 Section 4 对它们做了量化。LLaVA-OneVision 的 ablations 也暗示了它们。Native pretraining 是答案。

## 核心概念

### Native Multimodal Pretraining

InternVL3 从零开始，在一个原生 multimodal 的 corpus 上训练。混合比例是：

- 40% text-only data（FineWeb、Proof-Pile-2 等）
- 35% interleaved image-text data（OBELICS、MMC4-style）
- 20% paired image-caption data
- 5% video-text data

Vision tokens、text tokens 和 cross-modal interactions 从第一个 gradient step 开始就共同参与同一个 loss。没有 alignment pretraining，没有 projector freezing stage，也没有需要恢复的 catastrophic forgetting。

Base model 使用单阶段训练。Instruction tuning 仍然在后面进行，但 base model 已经把 visual tokens 当作 first-class citizens 理解。

### V2PE (Variable Visual Position Encoding)

Qwen2-VL 使用 M-RoPE，并采用固定 axis allocation。InternVL3 引入 V2PE：position encoding 会根据 modality type（text、image、video）变化，并带有 learnable scaling。实践中：

- Text tokens 获得 1D position（text index）。
- Image patches 获得 2D position（row, col）。
- Video frames 获得 3D position（time, row, col）。

三者共享同一个 RoPE frequency base，但每个 band 的 hidden-dim allocation 是 learned parameter，而不是固定切分。这让模型在 pretraining 期间可以自由权衡 temporal 和 spatial frequency resolution。

V2PE 的 ablation claim：在相同 compute 下，相比 M-RoPE，video benchmarks 提升 1-2 点。不是革命，但更干净。

### Visual Resolution Router (ViR)

Deployment optimization。不是所有 images 都需要 full-resolution encoding。一张只有一个低细节物体的照片，如果用 1280px native encoding，会浪费 tokens。ViR 是一个小 classifier，在 encoding 前预测回答问题所需的最低 resolution。

Routing 有三个 tiers：low-res（256 tokens）、medium（576）、high（2048+）。在 production traffic 中，60% queries 用 low 或 medium 就足够。净效果：在相同 quality 下 throughput 提升 2-3x。

### Decoupled Vision-Language Deployment (DvD)

服务大型 VLM 时，vision encoder 对每张 image 只运行一次，但 LLM 会为每个 output token 自回归运行。两个组件的瓶颈不同（vision = conv + attention 的 GPU memory bandwidth；LLM = KV cache）。DvD 把它们拆到不同 GPUs 上，并在中间 streaming handoff。

对于 8B + 400M encoder model，DvD 相比 co-located 大约能让 per-node throughput 翻倍。

### Single-Stage vs Multi-Stage Quality

InternVL3 的主要 benchmark claim：78B params 时匹配 Gemini 2.5 Pro 的 MMMU-Pro。38B 时匹配 GPT-4o。8B 时领先 open-8B leaderboard。全部基于 single-stage pretrain + instruction-tune recipe。

Alignment-debt hypothesis 是可测量的：每获得单位 vision-benchmark gain，InternVL3-8B 在 text benchmarks（MMLU、GSM8K）上损失的点数少于 Qwen2.5-VL-7B。这个模型更像 generalist，因为训练是一体的，不是两段拼接。

### InternVL3.5 and InternVL-U

InternVL3.5（2025 年 8 月）扩展了这个 recipe。仍然是 native-pretrain approach，但数据更多、参数更多。MMMU improvements 是渐进式的。

InternVL-U（2026）加入 unified generation：在同一个 backbone 上添加 MMDiT heads 做 image output。"U" 代表 "Understanding + generation"，目标是追赶 Transfusion-style unified models（Lesson 12.13）。同一个 native-pretrain backbone 同时支持 understanding 和 generation heads。

### Native Pretraining 的取舍

Native pretraining 不是免费的：

- Compute。从零训练一个新 VLM 的成本和训练 text LLM 相同，需要数百万 GPU-hours。Post-hoc adaptation 可以复用已有 LLM weights，节省大部分成本。
- Data。大规模 interleaved image-text corpora 很稀缺。OBELICS 有 141M documents；MMC4 有 571M。Text alone 可以达到 15T tokens。Multimodal pretraining data scarcity 是硬约束。
- Base-LLM reuse。Native pretraining 放弃了之后直接换新 LLM 的选项。Post-hoc 允许你把 Llama-3.1 换成 Llama-4，只重新训练 adapter。

InternVL3 的赌注是：alignment debt 比 reuse loss 更糟。Benchmarks 支持这个主张。生产成本会阻挡未来 labs 便宜复现。Post-hoc VLMs 仍会存在，因为对多数项目来说它们更便宜。

## 实战使用

`code/main.py` 是一个 training-corpus mixer 和 ViR router simulator。它会：

- 接收一个目标 corpus mix（%text、%interleaved、%caption、%video），并计算每种 modality 的 expected steps。
- 在一批 queries 上模拟 ViR routing（distribution：50% low-detail、30% medium、20% high-detail），并报告 average token count。
- 给定 encoder vs LLM FLOPs，报告 DvD throughput estimates。
- 按 params、compute、data 和 expected alignment-debt symptoms 横向对比 post-hoc vs native pretraining。

## 交付

本课产出 `outputs/skill-native-vs-posthoc-auditor.md`。给定一个 proposed VLM training plan，它会审计应该选择 native 还是 post-hoc，标记 alignment-debt risk，并推荐 corpus mix。当你在规划新的 open-VLM project，需要选择 training strategy 时使用它。

## 练习

1. 估算 InternVL3-8B（native pretrain）和 LLaVA-OneVision-7B（post-hoc）之间的 compute delta。GPU-hours 的比例大约是多少？差距由什么解释？

2. InternVL3 报告的比例是 40% text / 35% interleaved / 20% caption / 5% video。如果你的目标任务偏 video-heavy，提出一个新比例，并说明为什么 base model 仍然需要大量 text 和 caption data。

3. 阅读 MM1.5 Section 4 关于 forgetting 的内容。说出 post-hoc training 中 regression 最大的具体 benchmark。这个 regression 损失了多少？

4. ViR 把 60% traffic 路由到 low-resolution encoding。它会误路由哪些 queries（本该 high-res 却发到 low-res）？提出三种 router-failure modes。

5. DvD 把 vision 和 LLM 拆到不同 GPUs。在哪种 traffic pattern 下，DvD 会降低 throughput，而不是提高 throughput？

## 关键术语

| Term | 人们通常怎么说 | 它实际是什么意思 |
|------|----------------|------------------|
| Native multimodal pretraining | "From scratch together" | Text + image + video tokens 从第 1 步开始共同参与 loss，而不是事后接上 |
| Alignment debt | "Post-hoc penalty" | 把 vision 接到 frozen LLM 后导致的 text skills 和 answer consistency 可测量退化 |
| V2PE | "Variable visual pos encoding" | Per-modality learnable position encoding allocation；InternVL3 的 M-RoPE successor |
| ViR | "Resolution router" | 一个小 classifier，在 encoding 前为每个 query 选择最低所需 resolution，以节省 inference tokens |
| DvD | "Decoupled deployment" | Vision encoder 在一张 GPU 上，LLM 在另一张 GPU 上，中间 stream handoff；大型 VLMs 可获得约 2x throughput |
| InternVL-U | "Unified understanding + generation" | 2026 follow-up，在 native-pretrain backbone 上加入 image-generation heads |
| Interleaved corpus | "OBELICS / MMC4" | Text 和 images 按自然 reading order 排列的 documents；native pretraining 的原材料 |

## 延伸阅读

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
