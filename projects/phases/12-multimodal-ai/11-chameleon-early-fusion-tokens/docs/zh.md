# Chameleon 与 Early-Fusion Token-Only 多模态模型

> 到目前为止，我们见过的每个 VLM 都把 image 和 text 分开。Visual tokens 来自 vision encoder，流入 projector，然后在 LLM 内部与 text 相遇。Vision vocabulary 和 text vocabulary 从不重叠。Chameleon（Meta, May 2024）提出了一个问题：如果它们重叠呢？训练一个 VQ-VAE，把 image 转成来自 shared vocabulary 的 discrete tokens sequence。每个 multimodal document 现在都是一个 sequence：text tokens 和 image tokens 交错排列，一个 autoregressive loss。副作用是：模型可以生成 mixed-modality outputs，在一次 inference call 中交替输出 text 和 image tokens。本课阅读 early-fusion 论点，并从头到尾构建一个 toy version。

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## Learning Objectives

- 解释 shared vocabulary + single loss 为什么会改变模型能做的事情。
- 描述 VQ-VAE 如何把 image tokenize 成与 transformer next-token objective 兼容的 discrete sequence。
- 说出 Chameleon 的 training-stability tricks：QK-Norm、dropout placement、LayerNorm ordering。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 路线，并描述什么时候该选择哪一个。

## 问题

Adapter-based VLMs（LLaVA、BLIP-2、Qwen-VL）把 text 和 image 当作两种不同的东西。Text token 走 `embed(text_token)`；image 走 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条 input paths，并在中途合并。

这带来三个后果：

1. LLM 只能消费 images，不能输出 images。Output 只有 text。
2. Mixed-modality documents（像文章中那样交替出现段落和图片）很别扭：你要么在模型外解析 multimodal input，要么 chain generations。
3. Distributional mismatch。Visual tokens 和 text tokens 位于 hidden space 的不同区域，制造微妙的 alignment issues。

Chameleon 拒绝这个前提：images 只是 shared vocabulary 中的 discrete tokens sequences。用 interleaved documents 训练模型，一个 loss，一个 autoregressive decoder，你就免费解锁了 mixed-modality generation。

## 概念

### VQ-VAE 作为 image tokenizer

Tokenizer 是一个 vector-quantized variational autoencoder。架构如下：

- Encoder：CNN + ViT，把 image 映射成 spatial feature map，比如 32x32 个 dim 256 的 features。
- Codebook：一个 learned vocabulary，包含 K 个 vectors（Chameleon 使用 8192），同样是 dim 256。
- Quantization：对每个 spatial feature，用 L2 distance 查找最近的 codebook entry。用 integer index 替换 continuous feature。
- Decoder：CNN，把 quantized features 解回 pixels。

训练方式：VAE reconstruction loss + commitment loss + codebook loss。Codebook indices 形成 image 的 discrete alphabet。

对 Chameleon 来说，一张 image 会变成 32*32 = 1024 个 tokens，这些 tokens 来自大小为 8192 的 vocabulary。把它们和 text tokens（来自 LLM 的 BPE vocabulary，例如 32000）concatenate。最终 vocabulary：40192。Transformer 看到的是一个 sequence，一个 loss。

### Shared vocabulary

Chameleon 的 vocabulary 组合了 text tokens、image tokens 和 modality separators。每个 token 都有单个 ID。Input embedding layer 把每个 ID 映射到 D-dim hidden vector。Output projection 把 hidden 映射回 vocab logits。Softmax 选择下一个 token，无论是什么 modality。

Separators 很重要：`<image>` 和 `</image>` tags 包住 image-token sequence。Generation 时，如果模型输出 `<image>`，downstream software 就知道接下来的 1024 个 tokens 是要送到 decoder 做 pixel rendering 的 VQ indices。

### Mixed-modality generation

Inference 是 shared vocabulary 中的 next-token prediction。示例 prompt："Draw a cat and describe it." Chameleon 输出：

```text
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型会自主选择顺序：可能先生成 image 再 text，先 text 再 image，或者交错生成。同一个 decoder，同一个 loss。

对比 adapter VLMs，它们的 generation 只有 text。Chameleon 重新打开了模型 output modalities 这个问题。

### Training stability：QK-Norm、dropout、LayerNorm ordering

Early-fusion training 在 scale 上不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在 attention 内部、dot product 之前，对 query 和 key projections 应用 LayerNorm。防止深层网络中 logit magnitude explosion。多个 post-2024 large models 使用了这个技巧。
- Dropout placement。在每个 residual-add 之后放 dropout，而不只是 attention 和 MLP 之后。当 image tokens 的 gradients 可能主导时，需要更多 regularization。
- LayerNorm ordering。Residual branch 上使用 Pre-LN（标准做法），再在最后一个 block 的 skip connection 上加一个额外 LN。稳定 final-layer gradient flow。

没有这些技巧，34B-param Chameleon training 会在多个 checkpoints diverge。有了这些技巧，它才会收敛。Training recipe 和 architecture 一样，都是这篇工作的贡献。

### Tokenizer 的 reconstruction ceiling

VQ-VAE 是 lossy 的。8192 个 codebook entries、每张 512x512 image 1024 tokens 时，reconstruction PSNR 大约封顶在 26-28 dB。它足以生成可识别图像，但明显差于 continuous-space diffusion（Stable Diffusion 3 达到 32+ dB）。

Tokenizer 是瓶颈。更好的 tokenizers（MAGVIT-v2、IBQ、SBER-MoVQGAN）会抬高 ceiling。Emu3（Lesson 12.12）仅靠更好的 tokenizer 就实现了 SDXL-quality generation。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（early fusion, shared vocab）：

- One loss, one decoder。
- 生成 mixed-modality output。
- Tokenizer 是 quality ceiling。
- 昂贵：inference path 上每张 generated image 都需要 VQ-VAE decoder。

BLIP-2 / LLaVA（late fusion, separate towers）：

- Vision in, text out only。
- 复用 pretrained LLM。
- Understanding 没有 tokenizer bottleneck。
- 便宜：single forward pass。

按任务选择。如果你需要 image generation，选 Chameleon family。如果你只需要 understanding，adapter-VLM 更简单，也复用了更多 pretrained compute。

### Fuyu 与 AnyGPT

Fuyu（Adept, 2023）是相关路线：完全跳过 separate vision encoder，把 raw image patches 像 tokens 一样送进 LLM 的 input projection，不用 tokenizer。它比 Chameleon 更简单，但失去了 shared-vocab output generation。

AnyGPT（Zhan et al., 2024）把 Chameleon 扩展到四种 modalities：text、image、speech、music。每种 modality 都用同样的 VQ-VAE trick，共享 transformer。Any-to-any generation。Lesson 12.16 会讲更多。

## Use It

`code/main.py` 构建一个 toy end-to-end early-fusion model：

- 一个 tiny VQ-VAE-style quantizer，把 8x8 patches 映射到 codebook indices（K=16）。
- 一个 shared vocabulary：(text ids 0..31) + (image ids 32..47) + (separators 48, 49)。
- 一个 toy autoregressive decoder（bigram table），在 synthetic captions + image-token sequences 上训练。
- 一个 sampling loop，给定 prompt 后输出交替的 text + image tokens。

代码有意保持 transformer 很小（bigrams），这样你可以端到端追踪 signal flow。

## Ship It

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定 product spec（只需 understand，还是 understand + generate、所需 image quality、cost budget），它会在 Chameleon-family（early fusion）和 LLaVA-family（late fusion）之间选择，并用 quantitative rules of thumb 给出理由。

## Exercises

1. Chameleon 使用 K=8192 个 codebook entries，并为每张 512x512 image 使用 1024 tokens。估算它相对于 24-bit RGB image 的 compression ratio。它是 lossy 吗？有多 lossy？

2. 一张 4K image（3840x2160）在同样 VQ-VAE density 下会产生多少 image tokens？Chameleon-style model 能在一次 inference call 中生成 4K image 吗？最先崩的是 context、tokenizer quality，还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定 64-dim query 和 key，展示 LayerNorm 前后的 dot product。为什么 depth 变深时 magnitude control 很重要？

4. 阅读 Chameleon Section 2.3 关于 training stability 的内容。描述论文观察到的 34B 模型在没有 QK-Norm 时的确切 failure mode。"norm explosion" signature 是什么？

5. 扩展 toy decoder，使它能在 text-only prompt 下输出 mixed-modality response。测量当 training-data distribution 为 60% text-first / 40% image-first 时，模型选择 image-first vs text-first 的频率。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Early fusion | "Unified tokens" | 从第一步开始就把 images 转成 discrete tokens，并与 transformer vocabulary 共享 |
| VQ-VAE | "Image tokenizer" | CNN + ViT + codebook，把 images 映射成 transformer 可预测的 integer indices |
| Shared vocabulary | "One dictionary" | 单一 token ID space，覆盖 text + image + modality separators |
| QK-Norm | "Attention stabilizer" | 在 query 和 key dot product 之前应用 LayerNorm，防止 norm blowup |
| Mixed-modality generation | "Text + image output" | 一次 inference 中自主生成交错的 text 和 image tokens |
| Codebook size | "K entries" | VQ-VAE 可量化到的 discrete vectors 数量；在 compression 和 fidelity 之间权衡 |
| Tokenizer ceiling | "Reconstruction limit" | 解码 VQ tokens 所能达到的最佳 PSNR；它限制模型的 image quality |

## Further Reading

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
