# 审核系统：OpenAI、Perspective、Llama Guard

> Production moderation systems 会把第 12-16 课定义的 safety policies operationalize。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，可在一次调用中分类 text + images；在 multilingual test set 上比上一版本好 42%；response schema 返回 13 个 category booleans——harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数 developers 免费。Layered patterns：Input moderation（pre-generation）、Output moderation（post-generation）、Custom moderation（domain rules）。Async parallel calls 可以隐藏 latency；flag 时返回 placeholder responses。Llama Guard 3/4（第 16 课）：14 个 MLCommons hazards、Code Interpreter Abuse、8 种语言（v3）、multi-image（v4）。Perspective API（Google Jigsaw）：早于 LLM-as-moderator 浪潮的 toxicity scoring；主要是 single-dimension toxicity，并带 severe-toxicity/insult/profanity variants；它是 content-moderation research 的 baseline。Deprecations：Azure Content Moderator 于 2024 年 2 月 deprecated，2027 年 2 月 retired，由 Azure AI Content Safety 替代。

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## 学习目标

- 描述 OpenAI Moderation API 的 category taxonomy，以及它与 Llama Guard 3 的 MLCommons set 有何不同。
- 描述 three moderation-layer pattern（input、output、custom），并说出每一层的一个 failure mode。
- 描述 Perspective API 作为 pre-LLM-era baseline 的位置，以及为什么它仍被用于 research。
- 说出 Azure deprecation timeline。

## 问题

第 12-16 课描述 attacks 和 defense tooling。第 29 课覆盖 deployed moderation systems：它们在用户接触产品的 surface 上把这些 defenses operationalize。Three-layer pattern 是 2026 年的默认配置。

## 概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。一次调用即可分类 text + images。对大多数 developers 免费。

Categories（response schema 中的 13 个 booleans）：

- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Multimodal support 适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余 categories 是 text-only。

在 `code/main.py` 的 code harness 中，为了教学简洁性，我们把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` sub-categories 折叠进 top-level parents。Production code 应该使用完整的 13-category schema。

在 multilingual test set 上，它比上一代 moderation endpoint 好 42%。它提供 per-category scores；applications 自行设置 thresholds。

### Llama Guard 3/4

第 16 课已经覆盖。它包含 14 个 MLCommons hazard categories（组织方式不同于 OpenAI 的 13 个 response-schema booleans）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是 natively multimodal，12B。

OpenAI 和 Llama Guard 的 taxonomies 有重叠，但并不相同。OpenAI 使用 “illicit” 作为宽泛 category；Llama Guard 则分别列出 “violent crimes” 和 “non-violent crimes”。Deployments 会根据自己的 policy-taxonomy fit 选择。

### Perspective API（Google Jigsaw）

早于 LLM-as-moderator 浪潮的 toxicity scoring system（pre-2020）。Categories：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。Primary score 是 single-dimension TOXICITY，并带 sub-dimension variants。

它被广泛用作 content-moderation research baseline，因为 API 稳定、有文档，并且有多年 calibration data。对于现代 LLM-adjacent use cases，Llama Guard 或 OpenAI Moderation 通常更合适。

### Three-layer pattern

1. **Input moderation。** 在 generation 之前分类 user prompt。如果 flagged，则 reject。Latency：一次 classifier call。
2. **Output moderation。** 在交付前分类 model output。如果 flagged，则替换为 refusal。Latency：generation 之后的一次 classifier call。
3. **Custom moderation。** Domain-specific rules（regex、allowlists、business policy）。可在 input 或 output 侧运行。

这三层在设计上是 sequential：input moderation 必须在 generation 前完成，output moderation 在 generation 后运行。Parallelism 发生在 layer 内部——对同一段 text 并发运行多个 classifiers（例如 OpenAI Moderation + Llama Guard + Perspective），可以隐藏每个 classifier 的 latency。作为可选优化，可以在 input moderation 完成且 token-1 streaming 被延迟时显示 placeholder response（“one moment, checking...”）。Flag behaviour 可配置：refuse、sanitize、escalate to human review。

### Failure modes

- **Input only。** 抓不到 output hallucinations（第 12-14 课的 encoding attacks 会绕过 input classifiers）。
- **Output only。** 允许任何 input 到达 model；增加 cost；把 internal reasoning 暴露给 attacker。
- **Custom only。** 无法稳健覆盖多个 categories；regexes 很脆弱。

Layered 是默认做法。相当于 belt-and-suspenders。

### Azure deprecation

Azure Content Moderator：2024 年 2 月 deprecated，2027 年 2 月 retired。它由 Azure AI Content Safety 替代；后者基于 LLM，并与 Azure OpenAI 集成。对 Azure deployments 来说，这是一项 2024-2027 的 field-level migration project。

### 它在 Phase 18 中的位置

第 16 课在 red-team context 中覆盖 moderation tooling。第 29 课覆盖 deployed moderation。第 30 课用当前 dual-use capability evidence 收尾。

## Use It

`code/main.py` 会构建一个 three-layer moderation harness：input moderator（keyword + category score）、output moderator（同一个 classifier 作用于 output）、custom moderator（domain rules）。你可以运行不同 inputs，并观察哪一层捕获了什么。

## Ship It

本课产出 `outputs/skill-moderation-stack.md`。给定一个 deployment，它会推荐 moderation stack configuration：input 使用哪个 classifier，output 使用哪个 classifier，使用哪些 custom rules，以及 edge cases 用什么 judge。

## 练习

1. 运行 `code/main.py`。让一个 benign、borderline 和 harmful input 通过三层。报告每一种 input 触发了哪一层。

2. 在 harness 中扩展 Perspective-API-style toxicity scoring，针对一个具体 category 打分。比较它的 threshold behaviour 与 category score 的差异。

3. 阅读 OpenAI Moderation API docs 和 Llama Guard 3 category list。把每个 OpenAI category 映射到最接近的 Llama Guard categories。找出三个无法 cleanly map 的 categories。

4. 为 code-assistant deployment（例如 GitHub Copilot）设计一个 moderation stack。识别最相关和最不相关的 categories，并提出 custom rules。

5. Azure Content Moderator 将于 2027 年 2 月 retired。规划一次迁移到 Azure AI Content Safety 的过程。识别迁移中风险最高的元素。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| OpenAI Moderation | “omni-moderation-latest” | 基于 GPT-4o 的 13-category（text）classifier，带部分 multimodal support |
| Perspective API | “Google Jigsaw toxicity” | Pre-LLM-era toxicity scoring baseline |
| Llama Guard | “MLCommons 14-category” | Meta 的 hazard classifier（v3：8B text，8 languages；v4：12B multimodal） |
| Input moderation | “pre-generation filter” | Model call 前作用于 user prompt 的 classifier |
| Output moderation | “post-generation filter” | 交付前作用于 model output 的 classifier |
| Custom moderation | “domain rules” | Deployment-specific rules（regex、allowlist、policy） |
| Layered moderation | “all three layers” | 标准 production deployment pattern |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — toxicity scoring
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure replacement
