# Emu3：图像与视频生成的 Next-Token Prediction

> BAAI 的 Emu3（Wang et al., 2024 年 9 月）是一个本该终结 diffusion-versus-autoregressive 争论的 2024 结果。一个 Llama-style decoder-only transformer，只用 next-token-prediction objective 训练，在 text + VQ image tokens + 3D VQ video tokens 的 unified vocabulary 上，image generation 超过 SDXL，perception 超过 LLaVA-1.6。没有 CLIP loss。没有 diffusion schedule。Inference 时为了质量使用 classifier-free guidance，但核心 training objective 是带 teacher forcing 的 next-token prediction。发表在 Nature。本课读取 Emu3 的 thesis：为什么更好的 tokenizer 加 scale 就够了，并把它和 diffusion approaches 对比。

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## 学习目标

- 解释为什么 Emu3 的 single-loss next-token objective 能工作，尽管长期以来大家都认为 image quality 需要 diffusion。
- 描述 3D video tokenizer：spatiotemporal VQ codebook 是什么，为什么 patches 要跨越 time。
- 在 training compute、inference cost、quality ceiling 上比较 Emu3 与 Stable Diffusion XL。
- 说出同一个 Emu3 model 扮演的三种角色：Emu3-Gen（image gen）、Emu3-Chat（perception）、Emu3-Stage2（video gen）。

## 问题

到 2024 年为止，conventional wisdom 是：image generation 需要 diffusion。理由是：discrete image tokens 会丢失太多信息，无法重建细节；autoregressive sampling 会在数千个 tokens 上累积 error。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种 diffusion。Chameleon（Lesson 12.11）在小规模上部分反驳了这个观点，但 quality 没有追上 SDXL。

Emu3 正面攻击这个论点。它的 claim 是：更好的 visual tokenizer + 足够 scale + next-token loss = 在同一个也能做 perception 的模型里，得到超过 diffusion 的 image generation。

论文发表时这个赌注很有争议。两年后，open-source unified-generation family（Emu3、Show-o、Janus-Pro、Transfusion）已经成为 research 的默认路径；production frontier models 看起来也在使用某种 variant。

## 概念

### Emu3 tokenizer

关键成分是 visual tokenizer。Emu3 训练了一个 custom IBQ-class tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN family），每个 token 做 8x8 resolution reduction。一张 512x512 image 会变成 64x64 = 4096 tokens，codebook size 32768。

这比 Chameleon 在 512x512 下的 1024 tokens（K=8192）更大，但每个 token 更便宜（更小的 codebook lookups，更简单的 codec）。关键指标是 reconstruction PSNR 30.5 dB，能和 Stable Diffusion 的 continuous latent space（32 dB）竞争。

对于 video：3D VQ tokenizer 会把 spatiotemporal patch（4x4x4 pixels）编码成一个 integer。一个 4s clip，8 FPS，共 32 frames；在 256x256 下，空间和时间都做 4x reduction，token count 是 `(256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 tokens`。

Tokenizer quality 是 ceiling。Emu3 的贡献有一部分就是：“we trained a very good tokenizer.”

### Single-loss training

Emu3 只使用一个 objective：在 text tokens、2D image tokens 和 3D video tokens 共享的 vocabulary 上做 next-token prediction。训练时会用 modality-specific factors 乘到 weights 上，以平衡 contribution，但 loss function 完全相同。

训练数据混合：

- Image gen：`<text caption> <image> image_tokens </image>`
- Image perception：`<image> image_tokens </image> <question> text_tokens`
- Video gen：`<text caption> <video> video_tokens </video>`
- Video perception：类似。
- Text only：标准 NTP。

模型会从 data distribution 中学会何时 emit image tokens，何时 emit text tokens。当模型在 `<image>` tag 后预测 image tokens 时，generation 就出现了。

### Classifier-free guidance and temperature

Autoregressive image generation 在 inference 时使用 classifier-free guidance（CFG）会好很多。Emu3 使用了它：生成两次 logits，一次使用完整 caption，一次使用 empty caption，然后用 guidance weight 混合 logits（典型值 3.0-7.0）。这是 diffusion 使用的同一个 CFG trick，被借到了 autoregressive setting。

Temperature 很重要：太高会出 artifacts，太低会 mode collapse。Emu3 推荐 perception 使用 temperature 1.0，image generation 使用 0.8。

### Three roles, one model

Emu3 以三个 functionally distinct APIs 发布，但底层是同一组 weights：

- Emu3-Gen。Image generation。输入 text，输出 image tokens。
- Emu3-Chat。VQA 和 captioning。输入 image（tokens），输出 text。
- Emu3-Stage2。Video generation 和 video VQA。输入 text 或 video，输出 text 或 video。

没有 task-specific heads。只是不同 prompt templates。同一个 checkpoint。

### Benchmarks

来自 Emu3 paper（2024 年 9 月）：

- Image generation：在 MJHQ-30K FID 上超过 SDXL（5.4 vs 5.6），GenEval overall 上接近（0.54 vs 0.55，statistical tie），Deep-Eval composite 持平。
- Image perception：在 VQAv2 上超过 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致匹配。
- Video generation：4-second-clip quality 上，FVD 和 Sora-era publicly benchmarked models 竞争。

这些数字不是每项都赢，Emu3 会在这里让一分、那里赢一分，但 “next-token prediction is all you need” 这个 claim 在跨 modalities 上是站得住的。

### Compute cost

Emu3 使用 7B-parameter model，在约 300 billion multimodal tokens 上训练。GPU-hours 大致可与 Llama-2-7B pretraining 相比（A100-class silicon 上 2k-4k GPU-years）。Stable Diffusion 3 这类 diffusion models 的训练预算类似，但需要 separate text encoders 和更复杂 pipelines。

Inference 时，Emu3 每张 image 比 SDXL 慢：4096 image tokens 按 30 tok/s 生成，大约需要 2 分钟生成一张 512x512 image，而 SDXL 只需 2-5 秒。Speculative decoding 和 KV-cache optimization 会缩小差距，但不能完全消除它。Autoregressive image gen 很耗 compute；这是当前 trade-off。

### 为什么它重要

Emu3 更深层的贡献是概念性的。如果 next-token prediction 能 scale 到在 image generation 上匹配 diffusion，那么 unified-model path（one loss, one backbone, any modality）就是可行的。未来模型不需要 separate text encoders、separate diffusion schedulers、separate VAEs。一个 transformer，每种 modality 一个 tokenizer，然后 scale。

Show-o、Janus-Pro 和 InternVL-U 都在这个 thesis 上构建或挑战它。到 2025 年，中国 labs（BAAI、DeepSeek）在这个方向上的发表比美国 labs 更积极。

## Use It

`code/main.py` 构建两个 toy pieces：

- 一个 2D vs 3D VQ tokenizer count calculator：给定 resolution、patch、clip_length、FPS，计算 image vs video 的 token counts。
- 一个带 classifier-free guidance 和 temperature 的 autoregressive image-token sampler。

CFG implementation 匹配 Emu3 的 recipe：用 guidance weight 混合 conditional 和 unconditional logits。

## Ship It

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定 generation product spec（image 或 video、target resolution、quality tier、latency budget），它会计算 token counts、inference cost，并在 Emu3-family 和 diffusion 之间做选择。

## 练习

1. Emu3 在 8x8 reduction 下，每张 512x512 image 产生 4096 tokens。计算 1024x1024 和 2048x2048 的等价 token count。Inference latency 会发生什么？

2. 阅读 Emu3 Section 3.3 中的 video tokenizer。描述 3D VQ patch shape，以及为什么它是 4x4x4 而不是 8x8x1。

3. Classifier-free guidance weight 5.0 vs 3.0：视觉效果有什么区别？追踪 `code/main.py` 里的数学。

4. 计算 Emu3-7B 在 300B tokens 上的 training FLOPs，并与 Stable Diffusion 3 比较。哪一个训练更贵？

5. Emu3 在 FID 上超过 SDXL，但在 VQAv2 上没有超过 specialized VLMs。解释为什么 unified-loss approach 在不同 benchmarks 上相对 specialists 显示出不同优势。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Next-token prediction | “NTP” | 标准 autoregressive loss：给定 `token[0..i]` 预测 `token[i+1]`；只要被 tokenized，就适用于每种 modality |
| IBQ tokenizer | “Inverse bottleneck quantizer” | 一类 VQ-VAE，拥有更大 codebooks（32768+），reconstruction 优于 Chameleon 使用的 tokenizer |
| 3D VQ | “Spatiotemporal quantizer” | 由 `(time, row, col)` 索引的 codebook；一个 token 覆盖 4x4x4 pixel cube |
| Classifier-free guidance | “CFG” | 用 weight gamma 混合 conditional 和 unconditional logits；在 inference 时提升 image quality |
| Unified vocabulary | “Shared tokens” | Text + image + video 都来自同一个 integer space；模型预测接下来出现的任意 modality |
| MJHQ-30K | “Image gen benchmark” | 由 30k prompts 组成的 Midjourney-quality benchmark；Emu3 在这里报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
