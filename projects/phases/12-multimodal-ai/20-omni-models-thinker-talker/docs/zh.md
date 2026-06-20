# Omni Models：Qwen2.5-Omni 与 Thinker-Talker 拆分

> GPT-4o 在 2024 年 5 月的产品演示之所以具有冲击力，不是因为底层模型，而是因为产品形态：一个 voice interface，你说话，模型看到摄像头看到的内容，并在 250ms 内说回来。Open ecosystem 在 2024 和 2025 年剩下的时间里都在追赶这个 product surface。Qwen2.5-Omni（March 2025）是 reference open design：一个 Thinker（大型 text-generating transformer）加一个 Talker（并行 speech-generating transformer），二者通过 streaming speech tokens 连接。Mini-Omni 简化了它，Moshi 匹配了它的 latency，GLM-4-Voice 把它扩展到中文。本课阅读 Thinker-Talker 架构，以及让 streaming real-time dialogue 工作的 latency budget。

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## Learning Objectives

- 把 inference pipeline 拆成 Thinker（text reasoning）和 Talker（speech synthesis），并解释为什么 parallel streaming 有效。
- 按组件计算 conversational interaction 的 time-to-first-audio-byte（TTFAB）budget。
- 描述 TMRoPE 如何在 Thinker 内部跨 vision、audio 和 text 做 time-aligned position encoding。
- 说出三种 real-time conversational patterns：half-duplex、turn-taking、full-duplex。

## 问题

一个 real-time voice assistant 必须很快完成很多事：

1. 听见用户。Real-time speech tokenization，voice activity detection（VAD）判断用户什么时候说完。
2. 可选地看见环境。Camera input 以 2-4 FPS 流式进入 Thinker，并与 audio 一起处理。
3. 思考。基于 conversation history 组合 response。
4. 说话。合成 audio tokens，decode 成 waveform，并 stream 到用户 speakers。

每一步都会增加 latency。要有 conversational-feel，total round-trip 必须 < 500ms；低于这个阈值，用户就不再明显感到延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都必须 stream。不能 "batch everything then decode"。

## 概念

### Thinker and Talker

Qwen2.5-Omni 的分解：

- Thinker：一个 7B-80B text-generating transformer。消费 interleaved text + image + audio tokens。输出表示要说什么的 text tokens。
- Talker：一个较小的 speech-generating transformer（200M-1B）。消费 Thinker 的 text output tokens 加 recent speech-context tokens。输出 discrete speech tokens（residual-VQ indices）。
- Speech decoder：一个 streaming waveform decoder（SNAC、MoVQGAN family），把 speech tokens 实时转换成 audio samples。

这种分离很重要。Thinker 必须足够大，才能有好的 reasoning。Talker 可以很小，因为它的工作是 local 的：把 text 转成 speech tokens。更大的 Talker 并不会更 expressive，只会更慢。

二者并行运行：

1. Thinker 输出 text token t_i。
2. Talker 通过 streaming 消费 t_i，并输出 speech tokens s_i, s_{i+1}, ..., s_{i+k}。
3. Speech decoder 一边接收 speech tokens，一边输出 audio samples。
4. 当 Thinker 走到 text token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} stream 了 audio。

### TMRoPE：time-aligned multimodal positions

Thinker 需要整合 image frames（比如 4 FPS 到达）、audio frames（每秒 50 frames 到达）和 conversation history 中的 text。天真的 sequence order（先所有 images，再所有 audio，再 text）会丢失 temporal alignment。

TMRoPE 为每个 token 分配 absolute timestamp。Vision token at t=2.3s。Audio token at t=2.32s。用户说 "stop" 的 text token at t=2.35s。RoPE 按 timestamp 旋转 attention；模型会把它们看作时间上同时发生的内容。

这是让 "he waved while saying hello" 能成立的基础设施：模型在同一个 conceptual moment 看见 video frame 和 audio。

### Streaming speech synthesis

Speech tokens 必须 stream。Mini-Omni（Xie & Wu, 2024）提出 "language models can hear, talk while thinking in streaming"：Thinker output tokens 和 Talker output tokens 在同一个 sequence 中交错。Talker 在 Thinker 提交下一个 text token 时立刻触发。没有 batch boundaries。

Moshi（Défossez et al., October 2024）是最快的 open implementation。在单张 A100 上 TTFAB 为 160ms。架构是一个单一 7B transformer，在交替位置输出 text 和 speech tokens，并用 "inner monologue" 把 thinking stream 与 speaking stream 分开。这本质上是把 Thinker + Talker 融合进一个模型，并通过精心训练实现。

### VAD 与 turn-taking

Voice activity detection 在 input side 运行。两种模式：

- Half-duplex：用户说话，模型听。模型说话，用户听。通过 VAD silence detection（约 200ms）清晰交接。
- Full-duplex：双方可以同时说话。模型可以 backchannel（"uh-huh"）或 interrupt。难很多。Moshi 支持这一点。

Qwen2.5-Omni 默认支持 half-duplex，通过 silence threshold 做 turn-taking。Full-duplex 需要 application-layer handling。

### Qwen3-Omni（November 2025）

后继版本。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。Latency 接近 GPT-4o 的 250ms。Open weights。在 OmniBench 上与 Gemini 2.0 Live 竞争。

### Production latency budget

典型 streaming interaction：

- Mic -> audio tokens：40-80ms。
- Prefill（prompt + history）：7B 上 100-200ms，70B 上更久。
- First Thinker text token：40ms。
- Talker processes first text token：20ms。
- First speech tokens commit：40ms。
- Residual-VQ decode：30ms。
- Speech waveform decode：50-80ms。

Total TTFAB：7B 上 320-510ms，70B 上 600-900ms。Frontier quality 通常意味着 70B+，因此 frontier latency gap 会出现。

### Token-rate math

16kHz speech 搭配 50 Hz base speech tokens 时，你需要每秒输出 50 个 speech tokens。Talker 必须 emit ≥50 tok/s 才能跟上。在 H100 上，典型 LLM throughput 是 30-80 tok/s，因此小型（200-300M）Talker 足够快；7B Talker 会落后。

这就是为什么需要小型 dedicated Talker models，而不是 "just use the main model"。

## Use It

`code/main.py`：

- 用 mock token-emission rates 模拟 Thinker-Talker pipeline。
- 为可配置 model sizes 和 mic sample rates 计算 TTFAB。
- 用 VAD silence threshold 演示 half-duplex turn-taking。

## Ship It

本课产出 `outputs/skill-omni-streaming-budget.md`。给定一个 real-time voice product 的 target TTFAB 和 feature set（vision-in、bilingual、full-duplex），它会选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni，并为 Thinker/Talker sizing。

## Exercises

1. 你的 target TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，写出每个 component 的 latency。

2. Qwen2.5-Omni 使用 TMRoPE。描述这样一个 prompt 中模型看到什么：用户从 t=1s 开始说话，摄像头在 t=1.2s 捕捉到一个 gesture。

3. Full-duplex support 要求模型一边听一边输出 audio。提出一种能教会它这一点的 training data format。

4. 阅读 Moshi paper Section 4。描述 "inner monologue" separation，以及为什么它能避免 Thinker-Talker split。

5. 计算 throughput budget：为了跟上 16kHz speech、50 base-layer tokens/sec，Talker 必须以多快速度 emit tokens？

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Thinker | "Reasoning brain" | 大型 text-generating transformer，生成要说什么 |
| Talker | "Speech-generating mouth" | 小型 transformer，从 Thinker 的 text 生成 discrete speech tokens |
| TTFAB | "Latency budget" | Time-to-first-audio-byte：从用户 speech 结束到第一段 audio sample 输出的时间 |
| TMRoPE | "Time-aligned RoPE" | 使用 absolute timestamps 跨 vision、audio、text 做 position encoding |
| Half-duplex | "Turn-taking" | 用户和模型交替说话；VAD silence 检测用户已经说完 |
| Full-duplex | "Simultaneous" | 模型可以同时说话和聆听；支持 backchannel |
| Inner monologue | "Moshi separation" | 单模型设计，其中 thinking-stream 和 speaking-stream 交错出现 |

## Further Reading

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
