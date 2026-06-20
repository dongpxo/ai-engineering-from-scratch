# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以支撑随后五年的大想法：只用 noisy web image-caption pairs 和 contrastive loss，就能把 image encoder 和 text encoder 对齐到同一个 vector space。零 supervised labels。400M pairs。得到的 embedding space 能做 zero-shot classification、image-text retrieval，并作为 vision tower 接入每个 2026 年的 VLM。SigLIP 2（2025）用 sigmoid 替代 softmax，并以更低成本扩展到超越 CLIP。本课从 InfoNCE 到 sigmoid pairwise loss 推导数学，并用 stdlib Python 构建 training step。

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## Learning Objectives

- 从 mutual information 推导 InfoNCE loss，并实现一个 numerically-stable 的 vectorized 版本。
- 解释为什么 sigmoid pairwise loss（SigLIP）可以扩展到 batch 32768+，而不需要 softmax 所要求的 all-gather overhead。
- 通过构造 text templates（`a photo of a {class}`）并对 cosine similarity 取 argmax，运行 zero-shot ImageNet classification。
- 说出 CLIP / SigLIP pretraining 给你的四个 levers：batch size、temperature、prompt template、data quality。

## 问题

CLIP 之前的 vision 是 supervised 的。收集带标签的数据集（ImageNet：1.2M images，1000 classes），训练 CNN，然后上线。Labels 很昂贵，labels 会偏向标注者能达成共识的内容，并且如果不 finetuning，就不能迁移到新任务。

image-caption web 上有十亿级 loosely-labeled pairs，几乎免费。一张 golden retriever 的照片，alt text 写着 "my dog Max in the park"，它就携带了监督信号：文本描述了图像。问题是：你能把它变成有用的训练吗？

CLIP 的答案是：把 image-caption pairs 当作 matching task。给定一批 N 张 images 和 N 条 captions，学习把每张 image 匹配到自己的 caption，并和 N-1 个 distractors 区分开。监督信号是 "这两件东西属于同一对；这 N-1 个不属于"。没有 class labels。没有人工标注。只有 contrastive loss。

得到的 embedding space 能做的事情远超 CLIP 的训练目标。ImageNet zero-shot 能工作，是因为 "a photo of a cat" 会 embed 到接近 cats 图片的位置，尽管这些图片从未被显式标注为 cats。这就是催生每个 2026 VLM 的赌注。

## 概念

### 双编码器

CLIP 有两个 towers：

- Image encoder `f`：ViT 或 ResNet，为每张 image 输出一个 D-dim vector。
- Text encoder `g`：小 transformer，为每条 caption 输出一个 D-dim vector。

两个 towers 都会把输出 normalize 到 unit length。因此 similarity 是 `cos(f(x), g(y)) = f(x)^T g(y)`，因为二者都是 unit-norm。

对于一批 N 个 (image, caption) pairs，构建 shape 为 `(N, N)` 的 similarity matrix `S`：

```text
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是 learned temperature（CLIP 初始化为 0.07；在 log-space 中学习）。

### InfoNCE loss

CLIP 在 rows 和 columns 上使用 symmetric cross-entropy：

```text
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 会迫使每张 image 比 batch 中其他 caption 更匹配自己的 caption。"Negatives" 就是 batch 中所有其他 items。更大的 batch = 更多 negatives = 更强信号。CLIP 使用 batch 32k 训练；scale matters。

### Temperature

`tau` 控制 softmax 的 sharpness。低 tau → sharp distribution，带来 hard negative mining 效果。高 tau → soft，所有 samples 都会贡献。CLIP 学习 log(1/tau)，并做 clipping 防止 collapse。SigLIP 2 固定初始 tau，并改用 learned bias。

### 为什么 sigmoid 更好扩展（SigLIP）

Softmax 需要整个 similarity matrix 同步。在 distributed training 中，你必须把每个 embedding all-gather 到每个 replica，然后做 softmax。它的 communication 对 world size 是二次的。

SigLIP 用 element-wise sigmoid 替代 softmax：对每个 pair `(i, j)`，loss 都是一个二分类问题："这是不是匹配 pair？" positive class labels 是 diagonal，其余都是 negative。Loss 是：

```text
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当且仅当 `i == j`，否则为 0。每个 pair 的 loss 都独立。不需要 all-gather。每张 GPU 计算自己的 local block 并求和。SigLIP 2 能低成本扩展到 batch 32k-512k，而 CLIP 会需要成比例更多的 communication。

### Zero-shot classification

给定 N 个 class names，为每个 class 构建 text template：

```text
"a photo of a {class}"
```

用 text encoder embed 每个 template。用 image encoder embed 你的 image。Argmax cosine similarity = predicted class。不需要在 target classes 上训练。

Prompt templates 很重要。CLIP 原论文为每个 class 使用了 80 个 templates（plain、artistic、photo、painting 等），并平均 embeddings。ImageNet 提升了 3 个点。现代用法通常选择一两个 templates。

### Linear probes 与 finetuning

Zero-shot 是 baseline。Linear probe（在 frozen CLIP features 上为目标 classes 训练一个 linear layer）在 in-domain tasks 上超过 zero-shot。Full finetuning 在 in-domain 上超过 linear probe，但可能损害 zero-shot transfer。三种 regimes，对应三组 trade-offs。

### SigLIP 2: NaFlex 与 dense features

SigLIP 2（2025）增加了：

- NaFlex：单个模型处理 variable aspect ratios 和 resolutions。
- 更好的 dense features，用于 segmentation 和 depth estimation，目标是在 VLMs 中作为 frozen backbone。
- Multilingual：在 100+ languages 上训练，而 CLIP 只有 English。
- 1B param scale，而 CLIP 顶到 400M。

在 2026 年的 open VLMs 中，SigLIP 2 SO400m/14 是默认 vision tower。CLIP 仍然是 pure image-text retrieval 的默认选择，尤其当特定 LAION-2B training distribution 匹配你的 query pattern 时。

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN（Google, 2021）：和 CLIP 同样的想法，1.8B pair scale，90% noisy。证明 noisy data 能 scale。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的 open reproduction，多种 scales，是主流 open checkpoint。EVA-CLIP：从 masked image modeling 初始化；是 VLMs 的强 backbone。BASIC：Google 的 CLIP+ALIGN hybrid。它们属于同一个 family，只是 data 和 tuning 不同。

### Zero-shot ceiling

CLIP-class models 在 ImageNet zero-shot 上大约封顶于 76%（CLIP-G, OpenCLIP-G）。继续提升需要更大数据（SigLIP 2 达到 80%+）或架构变化（supervised heads、更多参数）。Benchmark 正在 saturating；真正价值是 downstream VLMs 消费的 embedding space。

```figure
multimodal-fusion
```

## Use It

`code/main.py` 实现了：

1. 一个 toy dual encoder（hash-based image features、text char features），让你在没有 numpy 的情况下看到 InfoNCE 的 shape。
2. 纯 Python 的 InfoNCE loss（通过 log-sum-exp 保证 numerical stability）。
3. 用于对比的 sigmoid pairwise loss。
4. 一个 zero-shot classification routine：计算与一组 text prompts 的 cosine similarity，并用 argmax 得到 prediction。

运行它并观察 loss curve。绝对数值是 toy 的；shape 与真实 CLIP trainer 输出一致。

## Ship It

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组 images（通过 path）和 target classes 列表，它会用 CLIP template 构建 text prompts，用指定 checkpoint（例如 `openai/clip-vit-large-patch14`）embed 两侧，并返回带 similarity scores 的 top-1 / top-5 predictions。该 skill 会拒绝对 prompt list 中不存在的 classes 作出声明。

## Exercises

1. 手动为 batch of 4 pairs 实现 InfoNCE。构建 4x4 similarity matrix，运行 softmax，取 diagonal，计算 cross-entropy。用这个手算结果验证你的 Python 实现。

2. SigLIP 除 temperature 外还使用 bias parameter `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 存在巨大 class imbalance（每行 negatives 远多于 positives）时，`b` 起什么作用？阅读 SigLIP Section 3（arXiv:2303.15343）。

3. 构建 cats vs dogs 的 zero-shot classifier。尝试两个 prompt templates：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张 test images 上测量 accuracy。Templates ensemble 是否胜过单个 template？

4. 计算 512-GPU、batch 32k 训练时，softmax InfoNCE 与 sigmoid pairwise 的 communication cost。哪个按 O(N) 扩展，哪个按 O(N^2) 扩展？引用 SigLIP Section 4。

5. 阅读 OpenCLIP scaling-laws paper（arXiv:2212.07143, Cherti et al.）。从图中复现它们关于 data scaling 的结论：在固定 model size 下，ImageNet zero-shot accuracy 与 training data size 之间的 log-linear relationship 是什么？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | 对 batch similarity matrix 做 cross-entropy；每个 item 的 positive 是其 paired item，negatives 是所有其他 items |
| Sigmoid loss | "SigLIP loss" | Per-pair binary cross-entropy；没有 softmax，没有 all-gather，在 distributed training 中低成本扩展 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logits 的 scalar；控制 distribution 的 sharpness |
| Zero-shot | "no-finetune classification" | 用 text prompts 构建 class embeddings，并按 cosine similarity 分类；不在 target classes 上训练 |
| Prompt template | "a photo of a ..." | class name 外面的 text scaffold；会影响 zero-shot accuracy 1-5 个点 |
| Dual encoder | "Two-tower" | 一个 image encoder + 一个 text encoder，输出到共享 D-dim space |
| Hard negative | "Tough distractor" | 与 positive 足够相似的 negative，迫使模型努力区分它们 |
| Linear probe | "Frozen + one layer" | 只在 frozen features 上训练 linear classifier；用于测量 feature quality |
| NaFlex | "Native flexible resolution" | SigLIP 2 在不 resize 的情况下接受任意 aspect ratio 和 resolution images 的能力 |
| Temperature scaling | "log-parametrized tau" | CLIP 参数化 `log(1/tau)` 以改善 gradients；并 clipping 防止 tau collapse 到接近 0 |

## Further Reading

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) —— CLIP paper。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) —— SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) —— multilingual + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) —— 使用 noisy web data 扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) —— OpenCLIP scaling laws。
