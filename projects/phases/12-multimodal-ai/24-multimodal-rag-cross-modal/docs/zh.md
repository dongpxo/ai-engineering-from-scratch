# 多模态 RAG 与跨模态检索

> Vision-native document RAG 只是其中一小片。生产级 multimodal RAG 要更宽：跨文本、图像、音频和视频检索，用于旅行规划（“find me a quiet vegan brunch with natural light”）、医疗分诊（“what injury matches this photo + these notes”）、电商（“outfits similar to this selfie, in my size”）和现场服务（“diagnose this engine sound plus photo of the part”）等工作流。2025 年的三篇综述——Abootorabi et al.、Mei et al.、Zhao et al.——把子问题规范化为：cross-modal retrieval、retrieval fusion、generation grounding、multimodal evaluation。本课会阅读这些综述，并设计一条生产 pipeline。

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time:** ~180 minutes

## 学习目标

- 设计 cross-modal retrieval：text → image、image → text、audio → video 等。
- 比较三种 fusion strategy：score fusion、attention-based fusion、MoE fusion。
- 解释 generation grounding：当 sources 是多种模态混合时，“cite your sources”长什么样。
- 说出 2025 年三篇 canonical multimodal RAG survey，以及它们的子问题 taxonomy。

## 问题

单模态 RAG 已经是成熟模式：embed query，embed chunks，retrieve，把结果塞进 LLM。Multimodal RAG 需要：

1. 多个 retrieval heads（每种模态都需要处于兼容空间的 embeddings）。
2. 跨模态检索结果 fusion。
3. 能引用多种模态 sources 的 generation grounding。
4. 覆盖 cross-modal signal 的 evaluation metrics。

2025 年的综述都收敛到同一套 taxonomy。

## 概念

### Cross-modal retrieval

给定模态 A 的 query，检索模态 B 的 documents。三种模式：

1. Shared embedding space。CLIP 和 CLAP 会在共享空间中产生 text + image / text + audio embeddings。跨模态 cosine similarity 可以直接使用。限制是只能覆盖 CLIP 训练过的配对类型。

2. Per-modality encoder + translation。Text encoder + image encoder + 一个小 translator module，在空间之间做映射。Gupta et al. 的 Sen2Sen 和其他 2024 设计属于这类。更灵活，但增加复杂度。

3. VLM as encoder。使用 VLM hidden states 作为检索表示。VLM 支持的任何模态都可用。质量更高，也更贵。

选择：text+image 用 CLIP / SigLIP 2；text+audio 用 CLAP；frontier 质量的 cross-modal 用 VLM-hidden-states。

### Fusion strategies

你检索到了 10 个结果：5 张图像、3 段文本、2 段音频。如何合并？

Score fusion（最便宜）。每个模态有自己的 retriever，每个 retriever 返回 scores。在模态内部 normalize scores，然后求和。简单，经常有效。

Attention-based fusion。拼接所有 retrieved items，让一个小 attention network 给它们加权。需要训练。

MoE fusion。Gating network 路由到 modality-specific experts。不同 query type 走不同路由，例如视觉问题会给图像更高权重。

生产默认：score fusion，并对 query 的主导模态加一点 bias。如果 A/B 测试在你的 domain 上显示 MoE 有清晰收益，再升级到 MoE。

### Generation grounding

LLM 应该引用每个 claim 由哪个 retrieved item 支撑。对多模态：

- Text source：标准 citation `[1]`。
- Image source：`[img 3]` 加一段短 caption。
- Audio：`[audio 2 at 0:34]`。

用 grounding-aware data 训练 generator：训练目标中的每个 claim 都标注 source index。推理时，模型就会自然发出 citations。

### 2025 年三篇综述

Abootorabi et al.（arXiv:2502.08826, “Ask in Any Modality”）：multimodal RAG taxonomy。覆盖 retrieval、fusion、generation。覆盖范围最广。

Mei et al.（arXiv:2504.08748, “A Survey of Multimodal RAG”）：聚焦 sub-task benchmarks 和 failure modes。对 evaluation design 很有用。

Zhao et al.（arXiv:2503.18016）：vision-focused survey。对 ColPali-family 工作覆盖很强。

读完三篇，就能掌握截至 2025 年春季的 state of the art。大多数子问题仍然是开放问题。

### MuRAG — foundational paper

MuRAG（Chen et al., 2022）是第一篇 multimodal RAG。它从 multimodal KB 中检索 image + text，并生成答案。在 VLM 浪潮之前证明了可行性。现代系统（REACT、VisRAG、M3DocRAG）都建立在它之上。

### 生产级 trip-planner 示例

Query：“find me a quiet vegan brunch with natural light.”

Pipeline：

1. 分解 query。“quiet” → audio/review keyword；“vegan brunch” → menu item；“natural light” → image feature。
2. 按模态检索：
   - Reviews 上的 text retrieval：“vegan brunch, quiet ambiance.”
   - Restaurant photos 上的 image retrieval：“natural light, airy.”
   - Ambient-sound clips 上的 audio retrieval：“low decibel, no music.”
3. 融合 scores。每家餐厅都有一个 composite score。
4. Top-k restaurants → VLM generator with all evidence → 带 citations 的回答。

这已经远远超过 text-RAG。每个模态都会补充文本单独遗漏的信号。

### Agentic multimodal RAG

Multi-hop：如果第一次检索没有返回高置信答案，LLM 会 reformulate 并再次检索。Phase 14 的 Agentic RAG patterns 同样适用于这里。例如：

- Retrieve initial top-10 → LLM 问 “too noisy, filter for <40 dB” → re-retrieve。
- Retrieve images → LLM 发现其中一张有菜单 → retrieve the menu text → answer。

这会增加复杂度，但能处理 single-shot retrieval 无法回答的查询。

### Evaluation

Cross-modal evaluation 仍然不成熟。常见 proxies：

- 每个模态的 Recall@k。
- Fused top-k accuracy。
- Human-judged end-to-end satisfaction。
- Task-specific 指标（bookings completed、purchases made）。

还没有一个标准 benchmark 覆盖所有模态。大多数论文都在 domain-specific tasks 上评估。

## Use It

`code/main.py`：

- 三个 mock retrievers（text、image、audio），运行在一个共享 restaurant corpus 上。
- Score fusion，用可配置 weights 合并 modality scores。
- 一个 generator stub，输出带 citations 的最终答案。
- 一个简单 agentic loop，在 confidence 较低时 reformulate query。

## Ship It

本课产出 `outputs/skill-multimodal-rag-designer.md`。给定一个带 multimodal query flow 的产品 spec，它会设计 retrievers、fusion、generator 和 evaluation。

## 练习

1. 提出一个 medical-triage multimodal RAG：query = injury photo + text symptoms。哪些模态会从哪个 KB 检索？

2. Score fusion 是简单加权和。它有什么 failure mode 是 MoE fusion 可以避免的？

3. 阅读 Abootorabi et al. taxonomy（Section 3）。三个 canonical sub-problems 是什么？它们如何映射到你选择的产品？

4. 为 trip-planner multimodal RAG 设计 eval spec。哪些 metrics 覆盖 image recall、audio recall 和 composite correctness？

5. Agentic multi-hop RAG 每个 round-trip 都有 latency tax。查询困难到什么程度时，accuracy gain 才值得付出 latency？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Cross-modal retrieval | “Query one modality, retrieve another” | 文本 query 检索图像；图像 query 检索文本；需要共享空间或 translator |
| Score fusion | “Combine scores” | 对各模态 retrieval scores 做加权求和；最简单的 fusion |
| MoE fusion | “Modality-routed experts” | Gating network 针对每个 query 选择该信任哪个模态的 scores |
| Grounded generation | “Cite your sources” | 答案中的每个 claim 都标注 source index |
| MuRAG | “First multimodal RAG” | 2022 年确立 multimodal RAG 模式的论文 |
| Agentic multi-hop | “Reformulate and retry” | 当第一轮 confidence 低时，LLM 重新查询 retrievers |

## 延伸阅读

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
