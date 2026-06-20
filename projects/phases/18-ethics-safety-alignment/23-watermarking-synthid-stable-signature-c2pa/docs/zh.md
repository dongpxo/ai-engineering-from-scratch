# Watermarking — SynthID、Stable Signature、C2PA

> 三项技术构成了 2026 年 AI-generated-content provenance 的基础。SynthID（Google DeepMind）：image watermarking 于 2023 年 8 月推出；text+video 于 2024 年 5 月推出（Gemini + Veo）；text 版本于 2024 年 10 月通过 Responsible GenAI Toolkit 开源；2025 年 11 月随 Gemini 3 Pro 发布 unified multi-media detector。Text watermarking 会以难以察觉的方式调整 next-token sampling probabilities；image/video watermarks 可以经受 compression、cropping、filters、frame-rate changes。Stable Signature（Fernandez et al., ICCV 2023, arXiv:2303.15435）：fine-tune latent diffusion decoder，让每个输出都包含固定 message；生成图像被裁剪到只剩 10% 内容时，仍能以 >90% 检出率、FPR<1e-6 被检测到。后续论文 “Stable Signature is Unstable”（arXiv:2405.07145，2024 年 5 月）表明，fine-tuning 可以移除 watermark，同时保留质量。C2PA：cryptographically signed、tamper-evident metadata standard（C2PA 2.2 Explainer 2025）。Watermarking 和 C2PA 是互补的：metadata 可以被剥离，但能携带更丰富的 provenance；watermarks 能在 transcoding 后保留，但携带的信息更少。

**Type:** Build
**Languages:** Python (stdlib，token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## 学习目标

- 描述 token-level watermarking（SynthID-text style）以及它可被检测的机制。
- 描述 Stable Signature 和 2024 年破坏它的 removal attack。
- 说明 C2PA 的角色，以及为什么它与 watermarking 互补。
- 描述关键局限：model-specific signal、paraphrase 下的 robustness，以及 meaning-preserving attacks（arXiv:2508.20228）。

## 问题

2023-2024 年，deepfakes 和 AI-generated content 大规模进入政治与消费场景。Watermarking 是被提出的 technical provenance signal：在生成时标记内容，在之后检测它。2025 年的证据表明：没有任何 watermark 是无条件 robust 的；但如果与 C2PA metadata 分层组合，二者可以提供一个可用的 provenance story。

## 概念

### Text watermarking（SynthID-text style）

Kirchenbauer 等人 2023 年提出的机制，被 Google productionized：

1. 在每个 decoding step，hash 前 K 个 tokens，生成一个 pseudorandom partition，把 vocabulary 分成 “green” 和 “red” sets。
2. 通过给 green logits 加上 δ，让 sampling 偏向 green set。
3. 生成结果包含的 green tokens 会多于随机机会水平。

检测：重新 hash 每个 prefix，统计 generation 中的 green tokens，并计算 z-score。Watermarked text 的 z-score >0；human text 的 z-score 约为 0。

Properties：

- 对读者不可察觉（δ 足够小，quality loss 很轻）。
- 在能访问 vocabulary partition function 的情况下可检测。
- 不抗 paraphrase：重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google Responsible GenAI Toolkit 开源。

### Stable Signature（image）

Fernandez 等人，ICCV 2023。Fine-tune latent diffusion decoder，让每张生成图像都包含嵌入在 latent representation 中的固定 binary message。Detection 通过 neural decoder 从 latent 中解码。图像被裁剪到只剩 10% 内容时，仍能以 >90% 检出率、FPR<1e-6 被检测到。

2024 年 5 月 “Stable Signature is Unstable”（arXiv:2405.07145）：fine-tuning decoder 可以移除 watermark，同时保留 image quality。Adversarial post-generation fine-tuning 成本低；watermark 的 adversarial robustness 有限。

### SynthID unified detector（2025 年 11 月）

随 Gemini 3 Pro 一起发布：一个 multi-media detector，可以在同一个 API 中读取 text、image、audio 和 video 的 SynthID signals。它统一了 Google provenance stack。

### C2PA

Coalition for Content Provenance and Authenticity。Cryptographically signed tamper-evident metadata standard。C2PA 2.2 Explainer（2025）。C2PA manifest 会记录 provenance claims（谁创建、何时创建、发生了哪些 transformations），并由 creator 的 key 签名。

它与 watermarking 互补：

- Metadata 可以被剥离；watermarks 不容易被剥离。
- Metadata 信息丰富（完整 provenance chain）；watermarks 只携带 bits。
- C2PA 依赖 platform adoption；watermarks 会自动嵌入。

Google 在 Search、Ads 和 “About this image” 中集成了二者。

### 局限

- **Model-specific.** SynthID watermarks 来自启用 SynthID 的模型。未启用 SynthID 的模型生成内容不会带水印，因此 “no SynthID signal” 并不能证明 authenticity。
- **Paraphrase.** Text watermarks 无法经受 meaning-preserving paraphrase。
- **Transformation attacks.** arXiv:2508.20228（2025）展示了 meaning-preserving attacks，可以破坏 text watermarks 以及许多 image watermarks。
- **Fine-tune removal.** 如 “Stable Signature is Unstable” 所示，post-generation fine-tuning 可以移除嵌入式 watermarks。

### EU AI Act Article 50

AI-generated content labeling 的 Transparency Code（first draft 2025 年 12 月，second draft 2026 年 3 月，按照 [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content) 预计 final 于 2026 年 6 月发布）。截至 2026 年 4 月，该 Code 仍是 draft，timeline 可能变化。这是要求 technical layer 的 regulatory layer。Deepfakes 必须被 labeled。

### 它在 Phase 18 中的位置

第 22-23 课讨论模型 emit 的内容（private data、provenance signal）。第 27 课覆盖 training-data governance。第 24 课是要求这些 technical measures 的 regulatory framework。

## Use It

`code/main.py` 构建了一个玩具型 text watermark。Tokens 是整数 0..N-1；watermarked sampling 会偏向 hash-defined green set。Detector 会计算 green-token z-score。你可以观察 1000-token generations 中的 detection，看到 paraphrase 如何破坏信号，并测量 human text 上的 false-positive rate。

## Ship It

本课产出 `outputs/skill-provenance-audit.md`。给定一个带 provenance claim 的 content deployment，它会审计：watermark mechanism（如果有）、C2PA signing chain（如果有）、每一项的 adversarial robustness，以及 per-modality coverage。

## 练习

1. 运行 `code/main.py`。报告 watermarked 1000-token generation 与 human-authored text 的 z-scores。找出 95% confidence threshold 下的 false-positive rate。

2. 实现一个 paraphrase attack，把 30% 的 tokens 替换为 synonyms。重新测量 z-score。

3. 阅读 Kirchenbauer 等人 2023 年论文第 6 节关于 robustness 的内容。为什么 text watermarks 会在 paraphrase 下失败，但 image watermarks 能经受 cropping？

4. 设计一个使用 SynthID-text + C2PA metadata 的 deployment。描述 consumer 看到的 provenance chain。指出每个 component 的一个 failure mode。

5. 2024 年 “Stable Signature is Unstable” 结果表明 fine-tuning 可以移除 image watermark。设计一个 deployment control 来限制这种攻击，例如要求 fine-tuned checkpoints 必须 signed releases。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SynthID | “Google 的 watermark” | Cross-modal provenance signal；覆盖 text、image、audio、video |
| Token watermark | “Kirchenbauer-style” | Biased-sampling text watermark，可通过 green-token z-score 检测 |
| Stable Signature | “image watermark” | Fine-tuned-decoder watermark；ICCV 2023 |
| C2PA | “metadata standard” | Cryptographically signed tamper-evident provenance metadata |
| Paraphrase robustness | “改写会不会破坏它” | Text watermark property；当前仍有限 |
| Fine-tune removal | “adversarial unwatermark” | 通过 decoder fine-tuning 移除 image watermark 的攻击 |
| Cross-modal detector | “unified SynthID” | 2025 年 11 月发布的跨 modality unified API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226)：token-watermark mechanism。
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435)：image watermark paper。
- [“Stable Signature is Unstable” (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145)：removal attack。
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/)：cross-modal watermark。
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html)：metadata standard。
