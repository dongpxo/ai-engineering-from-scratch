# 从 CLIP 到 BLIP-2：作为模态桥梁的 Q-Former

> CLIP 能对齐图像和文本，但不能生成 caption、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥梁解决了这件事：32 个 learnable query vectors 通过 cross-attention 关注冻结 ViT 的特征，然后直接插入冻结 LLM 的输入流。188M 参数的 bridge，把一个 11B LLM 接到了 ViT-g/14 上。到 2026 年，几乎所有基于 adapter 的 VLM（MiniGPT-4、InstructBLIP、LLaVA 的近亲）都是它的后代。本课会读懂 Q-Former 的架构，解释它的两阶段训练，并构建一个 toy version，把 visual tokens 喂给冻结 text decoder。

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 解释为什么在冻结 vision encoder 和冻结 LLM 之间使用可训练 bottleneck，比 end-to-end finetuning 更省成本、更稳定
- 实现一个 cross-attention block，让固定数量的 learnable queries attend 到外部 image features
- 走读 BLIP-2 的两阶段 pretraining：representation（ITC + ITM + ITG），然后 generative（使用冻结 decoder 的 LM loss）
- 比较 Q-Former 和 LLaVA 使用的更简单 MLP projector，并说明什么时候各自更合适

## 问题

你有一个冻结的 ViT，每张图像会产生 256 个 patch tokens，每个 token 维度是 1408。你还有一个冻结的 7B LLM，它期望 token embeddings 的维度是 4096。最直接的 bridge 是一个从 1408 到 4096 的 linear layer；它能工作，但把全部 256 个 patch tokens 都喂进 LLM context，会让每张图像额外占用 256 个 tokens。在 32 张图像的 batch 上，仅 visual modality 就会消耗 8192 个 tokens。

BLIP-2 问的是：能不能把 256-token image representation 压缩成少得多的 tokens（比如 32 个），同时保留足够信息，让 LLM 能 caption、回答问题、对图像进行推理？并且能不能在不触碰冻结 backbones 的情况下训练这个 bridge，把训练成本限制在 bridge 的参数上？

答案是：Q-Former。32 个 learnable "query" vectors 对 ViT 的 patch tokens 做 cross-attention，产生 32-token visual summary，供 LLM 使用。总共 188M 参数。在接触 LLM 之前，它先用 contrastive、matching 和 generative objectives 训练。

## 概念

### Learnable queries

Q-Former 的核心技巧是：不要让 LLM 的 text tokens 直接 attend 到 image patches，而是引入一组新的 32 个 learnable query vectors `Q`，让它们去 attend image patches。这些 queries 是模型参数；它们在训练中学习，并且每张图像都使用同一组 32 个 queries。

Cross-attention 之后，每个 query 都携带图像的压缩摘要，例如“描述主要物体”、“描述背景”、“数一数物体”等。Queries 并不会字面上专门对应某个语义标签；它们会学习任何能让 downstream losses 下降的编码方式。

### Architecture

Q-Former 是一个小型 transformer（12 layers，约 100M 参数），有两条路径：

1. Query path：32 个 query vectors 先经过 self-attention（彼此之间），然后对冻结 ViT 的 patch tokens 做 cross-attention，最后经过 FFN。
2. Text path：一个类似 BERT 的 text encoder，与 query path 共享 self-attention 和 FFN weights。Text path 上禁用 cross-attention。

训练时两条路径都会运行。Queries 和 text 通过共享 self-attention 交互，这意味着在需要文本条件的任务（ITM、ITG）中，queries 可以根据 text 调整。推理时如果只是交给 VLM 使用，只让 queries 流过，得到 32 个 visual tokens。

### 两阶段训练

BLIP-2 分两阶段 pretrain：

Stage 1：representation learning（没有 LLM）。三个 losses：

- ITC（image-text contrastive）：类似 CLIP 的 contrastive loss，用在 pooled query tokens 和 text CLS token 之间。
- ITM（image-text matching）：二分类器，判断 image-text pair 是否匹配；使用 hard-negative mining。
- ITG（image-grounded text generation）：对 text 使用 causal LM head，并以 queries 为条件。它迫使 queries 编码能被文本生成出来的内容。

只有 Q-Former 训练。ViT 冻结。没有 LLM 参与。

Stage 2：generative learning。接上一个冻结 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小 linear layer，把 32 个 query outputs 投影到 LLM 的 embedding dim。把它们 prepend 到 text prompt 前面。只在 concatenated prompt + image + caption sequence 的 LM loss 上训练 linear projection 和 Q-Former。

Stage 2 之后，Q-Former + projection 就是完整 visual adapter。推理时：image → ViT → Q-Former → linear proj → prepend 到 text → 冻结 LLM 生成输出。

### 参数经济学

BLIP-2 使用 ViT-g/14（1.1B，冻结）+ OPT-6.7B（6.7B，冻结）+ Q-Former（188M，训练）= 总共 8B，训练 188M。Q-Former 只占完整 stack 参数的约 2.4%。训练成本也反映了这一点：少量 A100 跑几天，而不是端到端训练几周。

质量上，BLIP-2 在 zero-shot VQA 上追平或超过 Flamingo-80B，同时小 50 倍。这个 bridge 是有效的。

### InstructBLIP 和 instruction-aware Q-Former

InstructBLIP（2023）给 Q-Former 增加了一个额外输入：instruction text 本身。在 cross-attention 时，queries 现在能同时访问 image patches 和 instruction。Queries 可以按 instruction 专门化（“count the cars”、“describe the mood”），而不是学习单一固定摘要。它在 held-out tasks 上带来了 benchmark gains。

### MiniGPT-4 和 projector-only 方法

MiniGPT-4 保留了 Q-Former，但只训练 output linear projection，其余全部冻结。便宜，但代价是质量：queries 是 BLIP-2 的，不是你的。它适合快速迭代，但不是最佳架构。

### 为什么 LLaVA 选择了更简单的方案

LLaVA（2023，第 12.05 课）把 Q-Former 替换成普通的 2-layer MLP，把每个 ViT patch token 投影到 LLM 空间；一个 24x24 grid 每张图像 576 tokens，全部喂给 LLM。压缩更差，但 LLM 可以直接 attend 到 raw patches。在当时这很有争议；到 2023 年末，它成为主流，因为 visual instruction data（LLaVA-Instruct-150k）证明 MLP 可以被训练到保留足够信号。取舍是：LLaVA 的 context 填得更快，但它自然扩展到 multi-image 和 video。

到 2026 年，领域分裂成两派：Q-Former 在 token budget 重要的场景中保留下来（long video、many images）；MLP projector 在 raw quality per token 是优先目标的场景中占主导。

### Gated cross-attention：祖先 Flamingo

Flamingo（第 12.04 课）早于 BLIP-2，并使用同一种 cross-attention 思路，但它不是单一 bridge，而是在每个冻结 LLM layer 上做。BLIP-2 证明了：只压缩到 input layer 也能工作。Gemini 和 Idefics 同时结合了两者：interleaved input tokens，再加上可选 gated cross-attention，用于 in-context few-shot。

### 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及多数因为 token budget 而使用它的 video-language models。
- Perceiver resampler：Flamingo 的变体（第 12.04 课）；Idefics family、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种都有效。决定问题是：你受限于 token budget，还是受限于 quality-per-token？

## 投入使用

`code/main.py` 构建了一个 stdlib Q-Former 风格的 cross-attention：

1. 模拟 256 个 image patch tokens（dim 128）。
2. 实例化 32 个 learnable queries（dim 128）。
3. 运行 scaled-dot-product cross-attention（queries 提供 Q，patches 提供 K/V）。
4. 通过 linear layer 投影到 LLM-dim（512）。
5. 输出 32 个 LLM-ready visual tokens。

所有数学都用纯 Python 实现（对向量进行 nested loops）。它是 toy，但 shape 正确。程序会打印 attention-weight matrix，让你看到每个 query 从哪些 patches 中取信息。

## 交付成果

本课产出 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（vision encoder token count、LLM context budget、deployment constraints、quality target），它会推荐 Q-Former、MLP 或 Perceiver resampler，并给出简短理由和每种 bridge 的参数量估算。

## 练习

1. 用 PyTorch 实现 cross-attention block。验证在 32 个 queries 和 256 个 keys/values 下，attention-weight matrix 是 32 x 256，并且 softmax 后每一行求和为 1。

2. 在 BLIP-2 stage 1 中，Q-Former 同时运行三个 losses：ITC、ITM、ITG。用 pseudo-code 写出每个 forward signature。哪一个需要 text encoder path 处于 active 状态？

3. 比较参数量：Q-Former（12 layers，768 hidden）vs 2-layer MLP projector（1408 → 4096，两层）。到什么 LLM scale 时，188M Q-Former 的成本会通过训练效率赚回来？

4. 阅读 BLIP-2 paper（arXiv:2301.12597）第 3.2 节，了解 Q-Former 如何初始化。解释为什么从 BERT-base 初始化（而不是随机初始化）能加速收敛。

5. 对一个 10 分钟视频，以 1 FPS 采样为 60 帧，计算每帧 token cost：（Q-Former → 32 tokens/frame）vs（MLP projector → 576 tokens/frame）。哪一个能装进 128k-token LLM context window？

## 关键术语

| Term | 人们常说 | 它真正的意思 |
|------|----------|-------------|
| Q-Former | "Querying transformer" | 一个小型 transformer，带 32 个 learnable query vectors，对冻结 ViT features 做 cross-attention |
| Learnable queries | "Soft prompt for vision" | 一组固定参数，作为 cross-attention 的 query side；按模型学习，对所有 inputs 共享 |
| Cross-attention | "Q from here, K/V from there" | Query、key、value 来自不同来源的 attention；queries 用它从 ViT patches 中取信息 |
| ITC | "Image-text contrastive" | 应用于 Q-Former pooled queries 和 text CLS 的 CLIP-style loss |
| ITM | "Image-text matching" | 在 hard-negative-mined pairs 上做二分类；迫使 queries 区分细粒度不匹配 |
| ITG | "Image-grounded text generation" | 文本在 queries 条件下生成的 causal LM loss；迫使 queries 编码 text-decodable content |
| Two-stage pretraining | "Representation then generative" | Stage 1 单独训练 Q-Former（ITC/ITM/ITG）；Stage 2 接入冻结 LLM，只训练 projection + Q-Former |
| Frozen backbone | "Do not finetune" | Vision encoder 和 LLM weights 固定；只训练 bridge |
| Projection head | "Linear to LLM dim" | 最后的 linear layer，把 Q-Former output 映射到 LLM embedding dimension |
| Perceiver resampler | "Flamingo's version" | 类似的 learnable-query cross-attention；Flamingo 在每层使用，而不是作为单一 bridge |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597)：核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086)：使用 ITC/ITM/ITG 三件套的前身。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651)："align before fuse"，stage 1 training 的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500)：instruction-aware Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592)：projector-only approach。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795)：learnable-query cross-attention 的通用架构。
