# MIO 与 Any-to-Any Streaming 多模态模型

> GPT-4o 发布了一个多数 open models 还无法复刻的产品：一个能听语音、看视频，并实时说话回应的 agent。到 2024 年末，open ecosystem 的答案是 MIO（Wang et al.，2024 年 9 月）。MIO 把 text、image、speech 和 music 都 token 化，在 interleaved sequences 上训练一个 causal transformer，并能从任意模态生成任意模态。AnyGPT（Zhan et al.，2024 年 2 月）是 proof of concept；MIO 是 scale-up；Unified-IO 2（Allen AI，2023 年 12 月）是带 vision + action grounding 的近亲。本课阅读 any-to-any pattern：四个 tokenizers，一个 transformer，streaming-friendly decode。

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 minutes

## 学习目标

- 设计一个共享 vocabulary，让 text、image、speech 和 music tokens 共存且不冲突
- 从 compression + reconstruction 取舍角度比较 SEED-Tokenizer（images）和 SpeechTokenizer residual-VQ（speech）
- 解释构建 any-to-any generation 的四阶段 curriculum
- 说出三个 open any-to-any recipes 及其主要取舍：MIO、AnyGPT、Unified-IO 2

## 问题

一个统一多模态模型说起来容易，真正规模化很难。直到 2024 年，大多数 "any-to-any" systems 都是 pipeline：vision model → text representation → speech model → audio。每一跳都会丢失信息、增加 latency，并让训练更复杂。GPT-4o 的 demo video 展示了一个 subsecond response 的单模型替代方案；open systems 落后了几个月。

工程挑战：

- 每个 modality 都必须有 tokenizer，它要能为 reconstruction 做到足够低损压缩，并产生 transformer 能消费的 token rate。
- 单个 vocabulary 必须为 text（32k+）、image（16k+）、speech（4k+）、music（8k+）分配空间。最少也要四万多个 entries。
- Training data 必须覆盖每种 input-output pair（text→image、image→speech、speech→image 等），或者模型必须能组合泛化。
- Inference 必须足够快地 stream output tokens，才能达到 conversational latency（time-to-first-audio-byte <500ms）。

## 概念

### 四种模态的四个 tokenizers

MIO 的 tokenizer stack：

- Text：标准 BPE，vocab 约 32000。
- Image：SEED-Tokenizer（2023），一种 quantized VAE，离散 codebook 有 4096 entries，每张图 32x32 tokens。
- Speech：SpeechTokenizer residual-VQ（2023），把 16kHz waveform 编码成 8 个 hierarchical codebooks；第一层是 coarse content，后续层添加 prosody 和 speaker identity。
- Music：类似的 residual-VQ（Meta 的 MusicGen / Encodec family），4-8 个 codebooks。

每种 modality 都产生整数 tokens。这些 tokens 在共享 vocabulary 中获得互不重叠的 ID ranges：

```text
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

总计约 48k vocabulary。Input embedding 和 output projection 覆盖整个 vocabulary。

### Streaming decode

Speech generation 使用 residual-VQ。Transformer 预测 base（layer 0）speech tokens；后续 residual layers 由 parallel-decoded residual quantizer 预测。每个 layer 0 token 在 16kHz audio 下大约对应 50ms 音频。

Streaming pattern：

1. 用户对着麦克风说话；real-time audio tokenizer 每 50ms 发出 speech tokens。
2. MIO 在 tokens 到达时消费它们（prompt prefill + incremental forward）。
3. Output tokens 一边生成一边 stream；parallel speech decoder 以约 50-150ms latency 把它们转成 audio samples。
4. MIO 论文中的 time-to-first-audio-byte 约 300-500ms，接近 GPT-4o 的约 250ms。

Mini-Omni（arXiv:2408.16725）、GLM-4-Voice（arXiv:2412.02612）和 Moshi（arXiv:2410.00037）是互补的 streaming speech-LLM designs。Moshi 尤其值得注意：它在单张 GPU 上达到 160ms round-trip。

### 四阶段 curriculum

MIO 的 training curriculum：

1. Stage 1 — alignment。大规模 modality-pair corpora：text-image、text-speech、text-music。每个 pair 使用自己的 token vocabulary segment。训练共享 vocabulary。
2. Stage 2 — interleaved。Multi-modality interleaved documents（带图片和视频的 blogs、带 transcripts 的 podcasts 等）。训练 cross-modality context。
3. Stage 3 — speech-enhanced。额外 audio data，用来提升 speech quality，同时不损失 text capability。
4. Stage 4 — SFT。跨模态 instruction tuning：VQA、captioning、narration、speech-to-speech dialogue。

缺少某个 stage 会退化特定能力：跳过 stage 2，模型会丢失 cross-modality context；跳过 stage 3，speech 会很差。

### Chain-of-visual-thought

MIO 引入了 chain-of-visual-thought：模型会把 intermediate image tokens 作为 reasoning step 发出来。对于 “is the cat climbing a tree?”，模型会：

1. 发出 `<image>` tokens，渲染场景（来自 input image 或 sketch）。
2. 发出分析 sketch 的文本。
3. 发出 final answer。

渲染出来的 intermediate image 充当 scratchpad。Spatial-reasoning tasks 上 benchmarks 会提升。这个想法对应 text reasoning 里的 chain-of-thought。

### Any-to-any 的竞争者

- AnyGPT（arXiv:2402.12226）：4 modalities（text、image、speech、music），设计类似。
- Unified-IO 2（arXiv:2312.17172）：增加 vision action outputs、depth、normals。任务更多样，但规模更小。
- NExT-GPT（arXiv:2309.05519）：LLM + modality-specific diffusion decoders。不是单模型方案。
- CoDi（arXiv:2305.11846）：composable diffusion；通过共享 latent 做 any-to-any。

MIO 最接近 pure-token any-to-any。AnyGPT 是它的概念祖先。

### Latency budget

对 conversational product 来说，每个组件的 latency 都重要：

- Mic to audio tokens：约 50ms。
- Prefill（audio tokens + history）：8B model 上约 100ms。
- First output token：约 50ms。
- Parallel residual-VQ + speech decoder：约 100-150ms。

总 time-to-first-audio-byte：最低约 300ms。GPT-4o 声称约 250ms。Moshi 声称 160ms。MIO/AnyGPT 在公开 benchmarks 中大约是 400-600ms。

### 为什么 any-to-any 仍然很难

即使到 2026 年，open any-to-any models 仍然在两个轴上落后闭源模型：

- Speech quality。Residual-VQ tokenizer 是有损的；conversational speech 听起来比 ElevenLabs-class voices 更机械。
- Cross-modality reasoning。让模型“sing about what you see”仍然比 pure-vision tasks 更容易失败。

这些仍是开放研究问题。Qwen3-Omni（第 12.20 课）是 2025 年最先进的 open attempt。

## 投入使用

`code/main.py`：

- 定义 four-modality vocabulary allocation 并打印出来。
- 把一组 multimodal inputs（text、image、audio-clip、music）路由到 tokenizer router。
- 为 text-to-speech response 模拟 streaming decode，并统计 latency。
- 给定 encoder、prefill 和 decoder latencies，计算预期 time-to-first-audio-byte。

## 交付成果

本课产出 `outputs/skill-any-to-any-pipeline-auditor.md`。给定一个 conversational product spec（modalities in、modalities out、latency target），它会审计 MIO-family design choices，并计算 latency budget。

## 练习

1. 你的产品接受 speech input，并返回 speech output。端到端 latency budget target 应该是多少？列出会花时间的组件。

2. SpeechTokenizer residual-VQ 使用 8 个 codebooks。解释为什么 parallel-decoding residual levels 是必要的（相对于 sequential），以及它带来什么 latency savings。

3. 你的 vocabulary 有 32k text + 4k image + 4k speech。再加 8k music 和约 10 个 separators。在 hidden dim 4096 下，embedding-matrix parameter cost 是多少？

4. Chain-of-visual-thought 会发出一个 intermediate image。哪些问题会受益？哪些问题会被额外 tokens 伤害？

5. 阅读 Moshi（arXiv:2410.00037）。描述它的 "inner monologue" technique，并和 MIO 的 chain-of-visual-thought 比较。

## 关键术语

| Term | 人们常说 | 它真正的意思 |
|------|----------|-------------|
| Any-to-any | "Multimodal in/out" | 一个单模型，可以任意方向接受并发出 text、image、speech 和 music |
| Residual-VQ | "Speech tokenizer stack" | 多 codebook tokenization，每一层添加信息；base layer 是 content，后续层是 prosody |
| SEED-Tokenizer | "Image codes" | MIO 使用的离散 image tokenizer，带 4096-entry codebook |
| Chain-of-visual-thought | "Visual scratchpad" | 模型先生成 intermediate image 作为 reasoning step，然后再给 final answer |
| Time-to-first-audio-byte | "TTFAB" | 从用户语音到第一段 audio output 的 latency；<500ms 才有对话感 |
| Four-stage curriculum | "Training recipe" | Alignment -> interleaved -> speech-enhanced -> SFT，按这个顺序训练 |

## 延伸阅读

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
