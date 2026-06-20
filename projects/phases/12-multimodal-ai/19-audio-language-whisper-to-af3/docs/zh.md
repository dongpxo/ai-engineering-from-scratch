# 音频语言模型：从 Whisper 到 Audio Flamingo 3 的弧线

> Whisper（Radford et al., 2022 年 12 月）解决了语音识别——680k 小时弱监督多语言语音、一个简单的 encoder-decoder transformer，以及一个让后续每个 ASR 发布都必须引用它的 benchmark。但识别不是推理。问“这段录音里有哪些乐器”“说话者表达了什么情绪”“第 3 分钟发生了什么”，需要的是音频理解，而不是转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步搭起了这套栈：保留 Whisper 级 encoder，接上 Q-former，在 audio-text instruction data 上训练，加入 chain-of-thought reasoning。本课会走读这条弧线。

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## 学习目标

- 从 waveform 计算 log-Mel spectrogram：windowing、FFT、filter banks、log transform。
- 比较 encoder 选择：Whisper encoder、BEATs、AF-Whisper hybrid，以及它们各自什么时候胜出。
- 构建 audio Q-former：N 个 learnable queries 对 spectrogram patches 做 cross-attention。
- 解释 cascaded（Whisper-then-LLM）与 end-to-end audio-LLM training：为什么 end-to-end 更适合扩展到推理。

## 问题

语音识别已经被 Whisper 解决了。音频 OCR 已经变成商品能力。但“商品能力”止步于转录。如果模型不能对它听到的内容进行推理——时间、说话者、情绪、音乐结构、环境声音——单靠转录无法驱动产品功能。

三条显而易见的路线：

1. Cascade：Whisper 转录，LLM 对 transcript 推理。适合纯语音场景。对音乐、环境音、多说话者重叠、情绪会失败。

2. End-to-end audio-LLM：audio encoder 把 audio tokens 直接喂给 LLM，跳过转录。保留声学信息（情绪、说话者、环境）。需要新的训练数据。

3. Hybrid：audio encoder + text decoder，既能转录，也能推理。Qwen-Audio 和 Audio Flamingo 选择了这条路。

## 概念

### Log-Mel spectrogram：输入特征

每个 audio encoder 都从同一个特征开始：log-Mel spectrogram。

1. Resample 到 16 kHz。
2. 使用 25ms window、10ms hop 做 short-time Fourier transform。
3. 取 FFT 结果的 magnitude。
4. 应用 Mel filter banks（通常是 80 个 filter，在 0-8000 Hz 上按 log 间隔）扭曲到感知频率。
5. 做 log compress（log(1 + x)）以压缩动态范围。

结果是形状为 (T, 80) 的 2D array，其中 T 是时间帧数量。对一个 30 秒 clip、100 Hz frame rate 来说，形状是 (3000, 80)。

### Whisper 的 encoder

Whisper 的 encoder 是一个 12 层 ViT-style transformer，把 log-Mel spectrogram 作为时间帧序列处理。输出：每个时间帧一个 hidden-state vector。

对 ASR 来说，Whisper 的 decoder 是一个 cross-attention transformer，根据 encoder output 生成文本 token。标准 encoder-decoder。

对 ALM（audio-LLMs）来说，你希望把 encoder output 作为另一个 LLM 的输入。模式是：Whisper encoder 冻结，Q-former 可训练，LLM 冻结或微调。

### BEATs 与音频专用 encoder

Whisper 训练在以语音为主的数据上。它在音乐和环境音上更弱。

BEATs（Chen et al., 2022）是一个在 AudioSet 上训练的自监督 transformer。在同等参数量下，它比 Whisper 更好地捕获音乐和环境声音。

AF-Whisper（Audio Flamingo 3 的 hybrid）：把 Whisper + BEATs features concat 作为 audio input。Whisper 承载语言信号，BEATs 承载声学信号。

### Audio Q-former

模式与 BLIP-2 的 visual Q-former 相同。固定数量的 learnable queries（通常 32 或 64）对 audio encoder 的 output frames 做 cross-attention。这些 queries 变成 LLM 消费的 audio tokens。

训练 alignment stage：只训练 Q-former，在 audio-text pairs（AudioCaps、Clotho）上使用 contrastive + captioning losses。Instruction stage：end-to-end，解冻 LLM，在 instruction data 上训练。

### 这条弧线 — SALMONN、Qwen-Audio、AF3

SALMONN（Tang et al., 2023）：Whisper + BEATs + Q-former + LLaMA。第一个具备认真推理能力的开放 audio-LLM。在 MMAU benchmark 上 composite 约 0.55。

Qwen-Audio（Chu et al., 2023）：类似架构，在更丰富的数据集上训练，并针对多轮对话微调。MMAU 约 0.60。

LTU — Listen, Think, Understand（Gong et al., 2023）：显式推理数据，聚焦音频片段上的 chain-of-thought。更小，但更聚焦。

Audio Flamingo 3（Goel et al., 2025 年 7 月）：当前开放 SOTA。8B LLM backbone（Qwen2 7B）、Whisper-large encoder concat BEATs、64-query Q-former，在 1M+ audio-text instruction pairs 上训练。MMAU 0.72，在部分子任务上追平 proprietary frontier。

AF3 还引入了 audio 的 on-demand chain-of-thought：模型可以在最终答案前选择性输出 thinking tokens（“let me identify the instruments first: ...”）。开启 thinking 时，复杂推理任务准确率提升 3-5 个百分点。

### Cascaded vs end-to-end

Cascaded pipeline：

1. Whisper 把 audio 转录成 text。
2. LLM 对 text 推理。

它非常适合“summarize this podcast”。但会在这些任务上失败：

- “这首歌的 mood 是什么？”——mood 在声音里，不在文字里。
- “谁在说话，Alice 还是 Bob？”——需要 speaker identification。
- “爆炸发生在第几秒？”——temporal grounding 在文本中丢失。
- “这是真实音频还是生成音频？”——deepfake detection 需要声学特征。

End-to-end 会保留声学信号。Qwen-Audio 和 AF3 能原生处理音乐、环境和情绪。

### 2026 年生产 recipe

对一个新的 audio-understanding 产品：

- 如果目标是 transcription，没有音乐，没有情绪推断：使用 cascaded。
- 如果有音乐、情绪、多说话者或复杂音频推理：使用 AF3 / Qwen-Audio-family。

Cascaded 更便宜、更简单。End-to-end 能力更强。

### MMAU — 音频推理 benchmark

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年的音频推理 benchmark：

- 10,000 个 audio-text QA pairs，覆盖 speech、music、environmental sounds。
- 覆盖 classification、temporal reasoning、causal reasoning、open-ended QA。
- 测试 cascaded pipeline 系统性漏掉的内容。

开放 SOTA（AF3）为 0.72；proprietary frontier 约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。这个差距小于 VideoMME 的 open-vs-closed 差距，说明 audio-LLMs 正在成熟。

## Use It

`code/main.py`：

- 用 stdlib 实现 log-Mel spectrogram 计算：windowing、naive DFT、Mel filter-bank。
- Audio Q-former skeleton：给定 encoder output frames，计算 Q、K、V、attention，并输出 N 个 tokens。
- 在 toy task 上比较 cascaded-vs-end-to-end。

## Ship It

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（transcription、music tagging、emotion inference、multi-speaker diarization、environment classification），它会选择 cascaded、end-to-end AF3 或 hybrid。

## 练习

1. 对一个 30 秒 clip，在 16kHz、25ms window、10ms hop、80 Mel bins 下计算 log-Mel spectrogram 维度。换成 48kHz 会如何变化？

2. 为什么 Whisper 在音乐上表现较弱？BEATs 捕获了哪些 Whisper 没有捕获的 audio features？

3. Audio Q-former 使用 64 queries vs 32 queries：什么任务复杂度下 64 值得？32 在什么场景下节省 compute？

4. 阅读 AF3 Section 4 的 on-demand thinking。提出三个 chain-of-thought 最有帮助的音频任务。

5. 使用 AF3 的输出实现一个最小 diarization pipeline。你如何标记 speaker changes？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Log-Mel spectrogram | “Mel features” | 经过 Mel filter banks 后的 log-magnitude values 组成的 2D（time, frequency）array |
| Audio Q-former | “Audio Perceiver” | 从 audio encoder output 到固定长度 queries 的 cross-attention bottleneck，供 LLM 使用 |
| Cascaded | “ASR-then-LLM” | Whisper 先转录，text LLM 再推理的 pipeline；会丢失声学信息 |
| End-to-end | “Audio-LLM” | Audio features 通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | “Audio AudioSet encoder” | 在 AudioSet 上训练的 SSL transformer；擅长音乐 + 环境声音 |
| MMAU | “Audio reasoning bench” | 覆盖 speech、music、environment 的 10k QA pairs；2024 eval standard |
| On-demand thinking | “Audio CoT” | 模型可以在最终答案前选择性输出 reasoning tokens，准确率提升 3-5 个点 |

## 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
