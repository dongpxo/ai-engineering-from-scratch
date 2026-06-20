# 百万 Token 上下文中的长视频理解

> 一段 1 小时、24 FPS 的 4K video，在 patch 并 embed 后，会产生约 6000 万 tokens。一个 2 小时 podcast episode 的 transcript 是 30,000 tokens。一部完整 Blu-ray feature film，即使用激进 pooling 压缩，也有数十万 tokens。Google 的 Gemini 1.5（2024 年 3 月）以 1000 万 token context 开启了这个时代，在小时级视频上做可靠的 needle-in-a-haystack recall。LWM（Liu et al., 2024 年 2 月）展示了 ring attention 的 scaling path。LongVILA 和 Video-XL 进一步扩展 ingestion。VideoAgent 用 agentic retrieval 替代 raw context。每种方法都在 compute、recall 和 engineering complexity 之间做了不同取舍。本课把它们并排阅读。

**Type:** Build
**Languages:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 minutes

## 学习目标

- 计算 long-form video 在不同 FPS 和 pooling 设置下的 total visual-token counts。
- 解释三条 scaling paths：brute context（Gemini 1.5）、ring attention（LWM）、token compression（LongVILA / Video-XL）。
- 比较 raw-context video VLMs 与 agentic-retrieval video VLMs（VideoAgent）在 accuracy 和 latency 上的差异。
- 为 30 分钟 video 设计 needle-in-a-haystack test，并测量特定 minute 的 recall。

## 问题

384 native resolution 下，一帧 Qwen2.5-VL-sized patches 大约是 729 tokens。经过 3x3 pooling 后，每帧是 81 tokens。30 分钟 clip、1 FPS = 1800 frames = 145,800 tokens。2025 年 open VLMs 可以处理，但很紧。2 FPS 时是 291,600 tokens，只有最大 context 的模型能放下。

2 小时电影、1 FPS 是 583k tokens。超过大多数 2026 open models；需要 Gemini 2.5 Pro，或者更激进地 pooling。

于是出现了三条 scaling paths。

## 核心概念

### Path 1: Brute Context (Gemini 1.5, Claude Opus)

把硬件砸到问题上。把 context 扩展到数百万 tokens，在一次 forward pass 中处理所有内容。

Gemini 1.5 Pro 发布时支持 1M tokens；Gemini 1.5 Ultra 到 10M；2026 年的 Gemini 2.5 Pro 可以可靠处理数小时 video。论文（arXiv:2403.05530）记录了在约 9.5M tokens 内，needle-in-a-haystack recall 达到 99.7%。

工程上：定制 attention implementation，带 memory hierarchy（local + global + sparse），再加上用于 long-context efficiency 的 MoE expert routing。完整细节没有发布。也不是 open-source。

### Path 2: Ring Attention (LWM, LongVILA)

Ring attention 把长 sequence 分布到多个 devices 上，形成一个 "ring"，每个 device 持有一个 chunk。跨完整 sequence 的 attention 通过 ring pattern 完成：每个 device 把自己的 chunk 发送给下一个 device，计算 partial attention，并聚合结果。

LWM（Liu et al., 2024）用这种方式训练了 1M-token context model。Training compute 随 context 线性扩展，而不是二次扩展；attention 的 quadratic hit 被分摊到 ring 的 devices 上。

LongVILA（arXiv:2408.10188）把这个 pattern 适配到 VLMs。1400-frame videos，每帧 192 tokens = 268k context，用 8-way parallelism 的 ring attention 训练。

### Path 3: Token Compression (Video-XL, LongVA)

比 brute context 更便宜：在 LLM 看到 sequence 之前先激进压缩。

Video-XL（arXiv:2409.14485）使用 visual summary token：每个 N frames 的 clip 产生一个 "summary" token，这个 token attend over 这 N 帧。Inference 时，LLM 每个 clip 只看到一个 summary token，context 大幅缩小。

LongVA 用 "long context transfer" 技术把 LLM context 从 200k 扩到 2M。先在 long-context text 上训练，再通过 shared representation transfer 到 long-context video。

Token compression 用特定 timestamp 的 recall 换 scalability。模型大致知道发生了什么，但有时会漏掉精确 frames。

### Path 4: Agentic Retrieval (VideoAgent)

不要把完整 video 喂给 LLM。把 video 当成 database，让 LLM 查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 读取 question。
2. LLM 向 retrieval tool 请求相关 clips（"show me segments with a cat"）。
3. Tool 返回匹配 clip timestamps。
4. LLM 通过 VLM 读取这些 clips。
5. LLM 组合 answer，或提出 follow-up queries。

这是 LLM-as-agent pattern 在 long video 上的应用。Inference 更便宜（只 encode 相关 clips），engineering 更难（retrieval quality 成为瓶颈）。

### Needle-in-a-Haystack Benchmarks

标准 long-context test：在 video 的随机位置插入一个独特 visual 或 textual marker，然后提出一个需要 recall 它的 query。

Metric：跨 video length 和 marker position 的 Recall@k。

Gemini 2.5 Pro 在最长 90 分钟 videos 上取得 >99% recall。Open 72B models（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟左右约 85-90%，超过 60 分钟后退化。

如果 retrieval tool 足够好，VideoAgent 在 2+ 小时内容上可以匹配或超过 raw-context models，因为 retrieval 能命中 needle。

### 该选哪条路径

对于 15 分钟 clip，并要求 frontier accuracy：open 72B + native context 通常可行。选 Qwen2.5-VL-72B。

对于 30 分钟到 1 小时内容：open 方案选 LongVILA 或 Video-XL；closed 方案选 Gemini 2.5 Pro。质量门槛很重要，frontier 往往是 closed。

对于 2+ 小时内容：选 VideoAgent 或类似 retrieval patterns。或者把内容总结成更小 chunks，再喂 hierarchical summaries。

### 2026 Production Pattern

实践中，生产长视频 pipelines 通常是 hybrid：

1. 对整个 video 运行 dynamic-FPS sampling + aggressive pooling（得到 100k-token global representation）。
2. 传给 72B VLM，得到 global summary。
3. 如果用户问细节问题，用 summary 作为 index 运行 agentic retrieval。

这结合了 brute-context 的 global understanding 和 retrieval 的 local detail。

## 实战使用

`code/main.py`：

- 计算 1 分钟到 3 小时 videos 在不同 FPS + pooling 下的 token budgets。
- 模拟 needle-in-a-haystack run：在随机 timestamp 注入 marker，提出问题，并为 recall 打分。
- 包含一个 agentic-retrieval router simulator，用来选择要喂给 downstream VLM 的 specific clips。

运行 budget table，感受 scale gap。

## 交付

本课产出 `outputs/skill-long-video-strategy-planner.md`。给定 video duration 和 query complexity，它会在 brute-context、compression 和 agentic retrieval 之间选择，并计算 latency + quality expectations。

## 练习

1. 一场 45 分钟 lecture，1 FPS，每帧 81 tokens。Total tokens 是多少？能放进哪些模型的 contexts？

2. 设计一个 needle-in-a-haystack test：你会在第几分钟注入 marker，exact query format 是什么？

3. 在 1 小时 video 上比较 brute-context Qwen2.5-VL-72B（80k context）和 VideoAgent（Claude 3.5 + retrieval）。谁在 recall 上赢？谁在 latency 上赢？

4. Ring attention 的 memory cost 随 sequence length 线性扩展，也随 device count 线性扩展。解释为什么，以及如果去掉 ring-rotation phase 会失败在哪里。

5. 阅读 Gemini 1.5 Section 5 关于 needle-in-a-haystack 的内容。论文对 1M 与 10M token boundary 的 recall 有什么发现？

## 关键术语

| Term | 人们通常怎么说 | 它实际是什么意思 |
|------|----------------|------------------|
| Brute context | "Just more tokens" | 把 LLM context 扩展到数百万 tokens；一次 pass 处理全部内容 |
| Ring attention | "LWM-style parallel" | Distributed attention pattern：每个 device 持有一个 chunk，并按 ring 旋转 |
| Token compression | "Summary tokens" | 在 LLM 之前用 learned compressor 减少每个 clip 的 tokens |
| Needle-in-haystack | "NIH test" | 在随机位置插入唯一 marker，并在 test time 要求模型 recall 它 |
| Agentic retrieval | "LLM as query planner" | LLM 向 retrieval tool 请求相关 clips，通过 VLM 读取，再组合 answer |
| VideoAgent | "Retrieval pattern for video" | 标准 agentic-retrieval design：question -> tool -> clip -> answer |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
