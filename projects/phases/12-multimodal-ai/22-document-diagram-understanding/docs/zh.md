# 文档与图表理解

> Documents 不是 photos。PDF、scientific paper、invoice 或 handwritten form 有 layout、tables、diagrams、footnotes、headers 和 semantic structure，这些是普通 image understanding 捕捉不到的。VLM 之前的 stack 是一条 pipeline：Tesseract OCR + LayoutLMv3 + table-extraction heuristics。VLM 浪潮用 OCR-free models 替代了它：Donut（2022）、Nougat（2023）、DocLLM（2023）可以直接 emit structured markup。到 2026 年，frontier 基本变成了 “把 page image 以 2576px native 喂给 Claude Opus 4.7”，structured-markup output 顺手就能得到。本课读取 document AI 的三个时代。

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## 学习目标

- 解释 document AI 的三个时代：OCR pipeline、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三种 input streams：text、layout（bbox）、image patches，以及 unified masking。
- 比较 Donut（OCR-free, image → markup）、Nougat（scientific paper → LaTeX）、DocLLM（layout-aware generative）、PaliGemma 2（VLM-native）。
- 为新任务选择 document model（invoices、scientific papers、handwritten forms、Chinese receipts）。

## 问题

“Understand this PDF” 看起来简单，其实很难。信息藏在：

- Text content（90% signal）。
- Layout（headers、footnotes、sidebars、two-column format）。
- Tables（rows、columns、merged cells）。
- Figures and diagrams。
- Handwritten annotations。
- Fonts and typography（title vs body）。

Raw OCR 会 dump text，却丢掉其他所有东西。一个处理 invoices 的系统必须知道 “Total: $1,245” 来自 bottom-right，而不是 footnote。

## 概念

### Era 1 — OCR pipeline（pre-2021）

经典 stack：

1. PDF → 每页 image。
2. Tesseract（或 commercial OCR）提取 text 和 per-word bounding boxes。
3. Layout analyzer 识别 blocks（header、table、paragraph）。
4. Table structure recognizer 解析 tables。
5. Domain rules + regex 提取 fields。

它适合干净的 printed text。遇到 handwriting、skewed scans、complex tables、non-English scripts 就会崩。每种 failure mode 都需要自定义 exception path。

### TrOCR（2021）

TrOCR（Li et al., arXiv:2109.10282）用一个在 synthetic + real text images 上训练的 transformer encoder-decoder，替换了 Tesseract 经典的 CNN-CTC。它在 handwritten 和 multilingual text 上明显胜出。它仍然是一条 pipeline（detector -> TrOCR -> layout），但 OCR step 大幅改善。

### Era 2 — OCR-free（2022-2023）

第一批 OCR-free models 的主张是：完全跳过 detection，直接把 image pixels 映射到 structured output。

Donut（Kim et al., arXiv:2111.15664）：

- Encoder-decoder transformer，encoder 是 Swin-B。
- Output 可以是 form understanding 的 JSON，summarization 的 markdown，或任何 task-specific schema。
- 没有 OCR，没有 layout，没有 detection。

Nougat（Blecher et al., arXiv:2308.13418）：

- 专门在 scientific papers 上训练。
- Output 是 LaTeX / markdown。
- 处理 equations、multi-column layout、figures。
- 每个 arXiv-parser 都会调用的模型。

这些是 specialists，不是 generalists。Donut 处理 scientific paper 会失败；Nougat 处理 invoice 会失败。

### LayoutLMv3（2022）

另一条路线是 LayoutLMv3（Huang et al., arXiv:2204.08387）。它保留 OCR，但加入 layout understanding：

- 三种 input streams：OCR text tokens、per-token 2D bounding boxes、image patches。
- 对三种 modalities 做 masked training objective（masked text、masked patches、masked layout）。
- Downstream：classification、entity extraction、table QA。

LayoutLMv3 是 OCR-based document understanding 的高峰。它在 forms 和 invoices 上很强。它需要 upstream OCR。在标准 document benchmarks 上，它是 VLM 之前最强的一类准确率。

### DocLLM（2023）

DocLLM（Wang et al., arXiv:2401.00908）是 LayoutLM 的 generative sibling。它在 layout tokens 条件下生成 free-form answers。对 document QA 更好；但仍依赖 OCR input。

### Era 3 — VLM-native（2024+）

到 2024 年，VLMs 已经足够好，可以完全替代 pipeline。把 full page image 以 high resolution 喂给 VLM，提问，得到答案。

- LLaVA-NeXT 336-tile AnyRes 适合小 documents。
- Qwen2.5-VL dynamic-resolution 可以原生处理 2048+ pixels。
- Claude Opus 4.7 支持 2576px documents。
- PaliGemma 2（2025 年 4 月）专门为 documents + handwriting 训练。

VLM-native 和 OCR-pipeline 的差距快速缩小。到 2026 年，VLM-native 在这些任务上胜出：

- Scene text（hand-written + printed, mixed scripts）。
- Complex tables with merged cells。
- Math equations embedded in text。
- Figures with text annotations。

OCR pipelines 仍然在这些场景胜出：

- Massive scale 的 pure-scan workloads，因为 per-page latency 很重要。
- Pipeline reliability（deterministic failures vs VLM hallucinations）。
- 需要 auditable OCR output 的 regulated environments。

### Claude 4.7 / GPT-5 frontier

在 2576-pixel native input 下，frontier VLMs 的 document understanding 接近 human accuracy。2026 年初的 benchmark numbers：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，pipelined LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

Closed-model gap 主要来自 resolution 和 base-LLM scale。7B open models 落后几分，但正在追赶。

### Math equations and LaTeX output

Scientific papers 需要精确的 equation LaTeX output。Nougat 专门为此训练。带 LaTeX targets 训练的 VLMs（Qwen2.5-VL-Math、Nougat derivatives）能生成可用的 LaTeX。没有 explicit LaTeX training 时，VLMs 会生成可读但不精确的转写。

2026 年的 scientific-paper pipeline：先对 PDF 跑 Nougat，再用 VLM 处理 tricky pages。

### Handwriting

Handwriting 仍然是最难的子任务。Mixed printed + handwritten（医生笔记、填写好的 forms）是 OCR pipelines 仍能在 cost 上胜过 VLMs 的地方。Handwritten-only VLMs 正在改善（Claude 4.7、PaliGemma 2）。

### 2026 recipe

对于新的 document-AI project：

- 大规模 pure-printed invoices：LayoutLMv3 + rules，cost-efficient。
- Mixed documents（scientific + handwritten + forms）：VLM-native（PaliGemma 2 或 Qwen2.5-VL）。
- Full arXiv ingestion：Nougat 处理 math，VLM 处理 figures。
- Regulatory：OCR pipeline + VLM validator 做 cross-check。

## Use It

`code/main.py`：

- 一个 toy layout-aware tokenizer：给定 `(text, bbox)` pairs，生成 LayoutLMv3-style input。
- 一个 Donut-style task schema generator：forms 的 JSON template。
- 比较 OCR-pipeline、Donut、Nougat 和 VLM-native 在每页上的 token budgets。

## Ship It

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个 document-AI project（domain、scale、quality、regulatory），它会在 OCR pipeline、OCR-free specialist 和 VLM-native 之间做选择。

## 练习

1. 你的项目每天处理 1000 万张 invoices。哪种 stack 能在不损失 accuracy 的情况下最小化 cost-per-page？

2. 为什么 LayoutLMv3 在 form QA 上优于 pure-CLIP-VLMs，却在 scene-text 上落后？bbox stream 放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM-native output 在 LaTeX fidelity 上胜过 Nougat 的 test case，再提出一个 Nougat 胜出的 case。

4. 阅读 PaliGemma 2 paper（Google, 2024）。相比 PaliGemma 1，提升 document accuracy 的关键 training-data addition 是什么？

5. 设计一个 regulatory-safe hybrid：OCR pipeline 作为 primary，VLM 作为 secondary cross-check。二者 disagree 时如何解决？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OCR pipeline | “Tesseract-style” | 分阶段 stack：detect -> OCR -> layout -> rules；deterministic，但 fragile |
| OCR-free | “Donut-style” | 跳过 explicit OCR 的 image-to-output transformer；single model |
| Layout-aware | “LayoutLM” | Input 包含 per-token bbox coordinates；跨 modalities 做 unified masking |
| VLM-native | “Frontier VLM” | 把 page image 直接以 high resolution 喂给 Claude/GPT/Qwen VLM；无 pipeline |
| DocVQA | “Doc benchmark” | Document VQA standard；最常被引用的 score |
| Markup output | “LaTeX / MD” | Structured output format，而不是 free-form text；支持 downstream automation |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
