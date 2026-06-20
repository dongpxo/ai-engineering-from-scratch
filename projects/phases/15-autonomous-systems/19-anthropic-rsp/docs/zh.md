# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年政策。它采用 two-tier mitigation：Anthropic 会单方面做什么，以及哪些内容被表述为 industry-wide recommendation（包括 RAND SL-4 security standards）。它把 Frontier Safety Roadmaps 和 Risk Reports 增加为 standing documents，而不是一次性交付物。它删除了 2023 年的 pause commitment。它引入 AI R&D-4 threshold：一旦跨过该阈值，Anthropic 必须发布一份 affirmative case，识别 misalignment risks 和 mitigations。Claude Opus 4.6 没有跨过该阈值。Anthropic 在 v3.0 公告中表示，“confidently ruling this out is becoming difficult”。SaferAI 给 2023 RSP 的评分是 2.2；他们把 v3.0 下调到 1.9，使 Anthropic 与 OpenAI 和 DeepMind 一起落入 “weak” RSP 类别。Qualitative thresholds 取代了 2023 年的 quantitative commitments；删除 pause clause 是最尖锐的倒退。

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 minutes

## 问题

Frontier labs 发布的 scaling policies 一部分是技术文档，一部分是治理文档，一部分是给监管者的信号。RSP v3.0 是 Anthropic 当前的文档。仔细阅读它很重要，并不是因为它具有强制合规约束（它没有），而是因为这种 framing 会塑造一家 lab 如何理解 catastrophic risk，以及它如何向公众沟通 trade-offs。

v3.0 vs v2.0 的 diff 才是有用单位。新增了什么：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 threshold。删除了什么：2023 年的 pause commitment。重新表述了什么：把 mitigation schedule 拆成 Anthropic-unilateral 和 industry-recommendation 的 two-tier 结构。外部评审机构 SaferAI 把评分从 2.2（v2）降到 1.9（v3.0）。这说明一份 scaling policy 可以看起来更 polished，却变得更不严谨。

## 概念

### Two-tier mitigation schedule

- **Anthropic unilateral actions**：无论其他 labs 怎么做，Anthropic 都会采取的行动。包括阈值以上停止训练、具体 security measures、具体 deployment gates。
- **Industry-wide recommendations**：Anthropic 认为行业应当集体采取的做法。包括 RAND SL-4 security standards。这些不是 Anthropic 自己的 commitments；它们是 policy advocacy。

Two-tier structure 在 v2 中不存在。这意味着读者需要看清每个 commitment 位于哪一栏。一个写在 “industry-wide recommendation” 栏里的 security measure 不是 Anthropic 的承诺；它是 Anthropic 的希望。

### AI R&D-4 threshold

这是 RSP v3.0 指出的下一个重要 capability level。具体来说：一个能够以 competitive cost 自动化相当一部分 AI research 的模型。一旦 Anthropic 认为某个模型跨过这个阈值，它必须在继续 scaling 前发布一份 affirmative case，识别 misalignment risks 和 mitigations。

根据 v3.0 公告，Claude Opus 4.6 没有跨过这个阈值。文档还补充说：“confidently ruling this out is becoming difficult.” 这句话很重要；它承认该阈值已经近到成为现实关切，而不是一个 speculative limit。

Lesson 6（Automated Alignment Research）和 Lesson 7（Recursive Self-Improvement）直接汇入这个阈值。自动化 alignment researchers 跨过 research-quality bars，就是 AI R&D-4 threshold 正在接近的证据。

### Frontier Safety Roadmaps 与 Risk Reports

v3.0 把两类 artifact 提升为 standing documents：

- **Frontier Safety Roadmap**：面向未来的文档，描述 planned safety work、capability expectations 和 mitigation research。
- **Risk Report**：特定模型发布后的回顾性文档，描述 observed capability 和 residual risk。

两者都是公开的。两者都会按声明 cadence 更新。它们的用途是：读者可以追踪 Anthropic 在 Roadmap 中说自己会做什么，并把它和 Risk Report 中实际报告的内容对比。

### 删除 pause clause

2023 RSP 包含一个明确的 pause commitment：如果模型跨过特定 capability thresholds，训练会暂停，直到 mitigations 就位。v3.0 用更软的表述替代了明确暂停（发布 affirmative case，并在 mitigations 足够时继续）。SaferAI 和其他分析者直接指出，这是新文档中最强的倒退。

支持这一变化的 policy argument 是：2023 年的 quantitative thresholds 到 2026 年代的 capability benchmarks 中变得不可达，因为 benchmarks 本身被重新缩放了。反方观点是：scaling policy 中的 pause clause 是一种 commitment device；删除它会削弱政策可信度。

### SaferAI 的降级

SaferAI 是一家独立机构，会给 RSP-style documents 评分。他们公开的评分是：2023 Anthropic RSP 得分 2.2（评分尺度中 4.0 是当前最佳 RSP，1.0 是 nominal）。v3.0 得分 1.9。这让 Anthropic 从 “moderate” 降到 “weak”，与 OpenAI 和 DeepMind 一起进入 weak category。

SaferAI 指出的降级因素包括：

- Qualitative thresholds 取代了 quantitative ones。
- Pause commitment 被删除。
- AI R&D-4 threshold 的 mitigations 被描述为 “affirmative case”，而不是具体措施。
- Review mechanisms 依赖 Anthropic 的 Safety Advisory Group，独立 oversight 有限。

### 本课不是什么

这不是一节 compliance 课。RSP v3.0 不是 regulation；没有东西强制 Anthropic 遵守它。本课训练的是用它应得的 specificity 和 skepticism 阅读文档。Scaling policies 是 frontier labs 关于 catastrophic-risk posture 发出的主要公开信号。读懂它们，是任何依赖 frontier capabilities 的人都需要的实用技能。

## Use It

`code/main.py` 实现了一个小型 decision engine，模拟 RSP 的 threshold-evaluation 形态：给定一个 candidate model 和一组 capability measurements，返回 AI R&D-4 threshold 是否被跨过、需要哪些 affirmative-case sections，以及 deployment 是否可以继续。它有意保持简单；重点是把文档逻辑显式化。

## Ship It

`outputs/skill-scaling-policy-review.md` 会根据 v3.0 reference 审查一份 scaling policy（Anthropic、OpenAI、DeepMind 或内部政策）：two-tier structure、thresholds、pause commitments、independent review。

## 练习

1. 运行 `code/main.py`。输入三个处于不同 capability levels 的 synthetic models。确认 threshold evaluator 的行为符合预期，并产出正确的 affirmative-case template。

2. 完整阅读 RSP v3.0（32 页）。找出所有位于 “industry-wide recommendation” tier 的 commitments。其中哪些在 v2 中会属于 “Anthropic unilateral”？

3. 阅读 SaferAI 的 RSP grading methodology。把他们的 rubric 应用于文档，复现 v3.0 的 1.9 分。哪一行 rubric 最主要地推动了降级？

4. 2023 年的 pause commitment 被删除了。提出一个替代 commitment，既保留政策可信度，又承认 2026 年 benchmark-rescaling 问题。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（Lesson 20）比较。选出一个 v3.0 更强的领域，再选出一个 Preparedness Framework 更强的领域。

## 关键术语

| Term | 常见说法 | 实际含义 |
|---|---|---|
| RSP | “Anthropic's scaling policy” | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | “Research-automation threshold” | 以 competitive cost 自动化 substantial AI research 的能力 |
| Affirmative case | “Safety justification” | 公开论证 risks 已被识别且 mitigations 足够 |
| Frontier Safety Roadmap | “Forward plan” | 关于 planned safety work 和 expected capabilities 的 standing document |
| Risk Report | “Retrospective on a model” | 模型发布后关于 observed capability 和 residual risk 的 standing document |
| Two-tier mitigation | “Unilateral vs industry” | Anthropic commitments 与 industry recommendations 被分开 |
| Pause commitment | “2023 clause” | 明确承诺暂停训练；在 v3.0 中被删除 |
| SaferAI rating | “Independent RSP grade” | 第三方 rubric；v3.0 得分 1.9（v2 是 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — the full 32-page policy.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — summary of changes from v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — standing document linked from RSP v3.0.
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — retrospective on the current frontier model.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — connects AI R&D-4 to measured autonomy.
