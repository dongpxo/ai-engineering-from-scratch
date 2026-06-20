# 红队工具：Garak、Llama Guard、PyRIT

> 三个生产工具定义了 2026 年的 red-team stack。Llama Guard（Meta）是一个 Llama-3.1-8B classifier，在 14 个 MLCommons hazard categories 上 fine-tune；2025 年的 Llama Guard 4 是一个 12B natively multimodal classifier，从 Llama 4 Scout 剪枝而来。Garak（NVIDIA）是 open-source LLM vulnerability scanner，提供 static、dynamic 和 adaptive probes，用于 hallucination、data leakage、prompt injection、toxicity 和 jailbreaks。PyRIT（Microsoft）用于 multi-turn red-team campaigns，支持 Crescendo、TAP 和 custom converter chains，以进行深度 exploitation。Llama Guard 3 记录在 Meta 的 “Llama 3 Herd of Models”（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 记录在 arXiv:2411.17713 中；Garak 的 probe architecture 见 github.com/NVIDIA/garak。这些工具是 2026 年 red-team research（Lessons 12-15）与 deployment（Lesson 17+）之间的生产接口。

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## 学习目标

- 描述 Llama Guard 3/4 在 safety stack 中的位置：input classifier、output classifier，或者二者兼具。
- 说出 14 个 MLCommons hazard categories，并指出一个不那么显然的类别（Code Interpreter Abuse）。
- 描述 Garak 的 probe architecture：probes、detectors、harnesses。
- 描述 PyRIT 的 multi-turn campaign structure，以及它如何与 Garak probes 组合。

## 问题

Lessons 12-15 展示了 attack surface。生产部署需要可重复、可扩展的 evaluation。2026 年由三个工具主导：Llama Guard（defense classifier）、Garak（scanner）、PyRIT（campaign orchestrator）。它们分别瞄准 red-team lifecycle 的不同层。

## 概念

### Llama Guard (Meta)

Llama Guard 3 是一个 Llama-3.1-8B model，经过 fine-tune，可在 MLCommons AILuminate 14 categories 上进行 input/output classification：

- Violent crimes、non-violent crimes、sex-related、CSAM、defamation
- Specialized advice、privacy、IP、indiscriminate weapons、hate
- Suicide/self-harm、sexual content、elections、code-interpreter abuse

支持 8 种语言。用法：放在 LLM 前面（input moderation）、放在 LLM 后面（output moderation），或者两者都放。两种用途会产生不同的 training distributions；Llama Guard 3 作为单个模型同时处理二者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，在 mobile CPU 上约 30 tokens/s）是 quantized edge variant。

Llama Guard 4（2025 年 4 月）是 12B、natively multimodal，从 Llama 4 Scout 剪枝而来。它用一个能 ingest text + images 的 classifier，替代了之前的 8B text 和 11B vision predecessors。

### Garak (NVIDIA)

Open-source vulnerability scanner。架构如下：

- **Probes。** 面向 hallucination、data leakage、prompt injection、toxicity、jailbreaks 的 attack generators。包括 static（固定 prompts）、dynamic（生成 prompts）、adaptive（根据 target output 响应）。
- **Detectors。** 根据预期 failure modes 为 outputs 打分，例如 toxic、leaked、jailbroken。
- **Harnesses。** 管理 probe-detector pairs，运行 campaigns，并生成 reports。

TrustyAI 将 Garak 与 Llama-Stack shields（Prompt-Guard-86M input classifier、Llama-Guard-3-8B output classifier）集成，用于 end-to-end shielded-target evaluation。Tier-based scoring（TBSA）替代 binary pass/fail：同一个 probe 上，模型可能在 severity tier 3 通过，在 severity tier 5 失败。

### PyRIT (Microsoft)

Python Risk Identification Toolkit。用于 multi-turn red-team campaigns。核心组件包括：

- **Converters。** 转换 seed prompt，例如 paraphrase、encode、translate、roleplay。
- **Orchestrators。** 运行 campaign：Crescendo（escalation）、TAP（branching）、RedTeaming（custom loop）。
- **Scoring。** LLM-as-judge 或 classifier-as-judge。

PyRIT 是 Garak 更重的 cousin。Garak 会运行数千个 single-turn probes；PyRIT 会运行深入的 multi-turn campaigns，专门攻击特定 failure modes。

### Stack

把 Llama Guard 放在模型前后两侧。每晚运行 Garak 做 regression。Pre-release campaigns 用 PyRIT 跑。这就是 2026 年多数生产部署的默认配置。

### Evaluation pitfalls

- **Judge identity。** 三个工具都可以使用 LLM judge；judge calibration 会驱动报告中的 ASRs（Lesson 12）。在说明 tool 的同时，必须说明 judge。
- **Probe staleness。** Garak probes 会随着模型针对它们修补而老化。Adaptive probes（PAIR-shaped）比 static probes 老化更慢。
- **Llama Guard 对 benign content 的 FPR。** 早期 Llama Guard 版本会过度标记政治和 LGBTQ+ 内容；Llama Guard 3/4 的 calibrations 已经改进，但仍然不是 per-deployment calibrated。

### 它在 Phase 18 中的位置

Lessons 12-15 是 attack families。Lesson 16 是 production tooling。Lesson 17（WMDP）是 dual-use capability evaluation。Lesson 18 是把这些工具包进 policy structure 的 frontier safety frameworks。

## Use It

`code/main.py` 会构建一个 toy Llama Guard-style classifier（基于 14 categories 的 keyword + semantic features）、一个 toy Garak harness（probe-detector loop），以及一个 PyRIT-style multi-turn converter chain。你可以用这三个工具测试一个 mock target，并观察它们不同的 coverage signatures。

## Ship It

本 lesson 产出 `outputs/skill-red-team-stack.md`。给定一个 deployment description，它会指出三个工具中哪些适用、每个工具需要配置什么，以及应该用什么 regression cadence 运行。

## 练习

1. 运行 `code/main.py`。比较 Llama-Guard-style classifier 对 single-turn attacks 和 multi-turn attacks 的 detection rate。

2. 实现一个新的 Garak probe：base64-encoded harmful request。测量 Llama-Guard-style classifier 对它的 detection。

3. 给 PyRIT-style converter chain 扩展一个 “translate to French, then paraphrase” converter。重新测量 attack success。

4. 阅读 Llama Guard 3 的 hazard-category list。找出两个在合法 developer content 上，training data 现实中可能产生 high false-positive rates 的类别。

5. 比较 Garak 和 PyRIT 的 design principles。论证在哪种 deployment 中，每个工具分别是正确选择。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Llama Guard | “the classifier” | Fine-tuned Llama-3.1-8B/4-12B safety classifier，覆盖 14 个 hazard categories |
| Garak | “the scanner” | NVIDIA open-source vulnerability scanner；probes、detectors、harnesses |
| PyRIT | “the campaign tool” | Microsoft multi-turn red-team orchestrator；converters、orchestrators、scoring |
| Prompt-Guard | “the small classifier” | Meta 的 86M prompt-injection classifier，通常与 Llama Guard 搭配 |
| TBSA | “tier-based scoring” | Garak 的 tier-based pass/fail，用来替代 binary outcomes |
| Converter chain | “paraphrase + encode + ...” | PyRIT 用来构建 multi-step attacks 的 composition primitive |
| MLCommons hazard categories | “the 14 taxonomies” | Llama Guard 所针对的 industry-standard taxonomy |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)：8B classifier。
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713)：quantized mobile classifier。
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak)：scanner repo 和 documentation。
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT)：campaign toolkit。
