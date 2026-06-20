# 双重用途风险：Cyber、Bio、Chem、Nuclear Uplift

> 2026 年的 dual-use picture，可以按领域逐一理解。Bio/chem：第 17 课覆盖 WMDP；Anthropic 的 bioweapon-acquisition trial（2.53x uplift）和 OpenAI 2025 年 4 月 Preparedness Framework v2 warning（“on the cusp of meaningfully helping novices create known biological threats”）标志着拐点。Cyber（Anthropic 2025 年 11 月报告）：与中国有关的 state actors 使用 Claude 的 agentic coding tool 自动化了最高 90% 的 cyberattack campaign，human intervention 只剩 4-6 个步骤；OpenAI 的 “trusted access” pilot 为经过审查的 security organisations 提供 defensive dual-use work 所需的 capability access。Chem/bio execution gap erosion：经典防御是 “information access alone is insufficient”。具备视觉能力的 frontier models（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）可以观察 wet-lab video 并提供 real-time correction。2025 年 12 月：OpenAI 展示 GPT-5 迭代 wet-lab experiments，通过 AI-driven protocol optimization 实现 79x efficiency improvement。Novice-vs-expert pattern：AI 对 novices 提供更大的 relative uplift，但对 experts 提供更大的 absolute capability。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17（WMDP），Phase 18 · 18（safety frameworks），Phase 18 · 28（ecosystem）
**Time:** ~75 minutes

## 学习目标

- 描述 2024-2025 年的 bio-uplift narrative：“mild uplift” -> “on the cusp” -> “2.53x uplift insufficient to rule out ASL-3”。
- 描述 Anthropic 2025 年 11 月 cyber report：与中国有关的 automation 覆盖最高 90% 的 cyberattack campaign。
- 描述 chem/bio execution-gap erosion：具备视觉能力的模型对 wet-lab experiments 进行 real-time correction。
- 说出 novice-relative 与 expert-absolute asymmetry，以及它对 safety-case construction 的影响。

## 问题

第 17 课是 measurement methodology。第 30 课是 2026 年 measurement 的状态。2024 年到 2025 年末之间，图景发生了实质变化：每个 domain 都跨过了 2024 frameworks 没有预期到的 threshold。

## 概念

### Bio/chem uplift narrative

三个阶段（为保持连贯，重复第 17 课内容）：

1. **2024 “mild uplift”。** 早期 Preparedness/RSP evaluations 报告，相比 internet search，novices 只有小幅优势。
2. **2025 年 4 月 “on the cusp”。** OpenAI PF v2 警告，models 已经 “on the cusp of meaningfully helping novices create known biological threats”。
3. **2025 Anthropic bioweapon-acquisition trial。** 受控 novice study；acquisition-phase tasks 上有 2.53x uplift；不足以排除 ASL-3。

这个变化是质性的：“mild” 在十八个月内演化成 “plausibly enabling”，即使没有出现一次明显的 capability breakthrough。

### Chem/bio execution-gap erosion

历史防御：information 是必要但不充分条件；执行 protocol 的技能会阻挡 novices。2025 年具备视觉能力的 frontier models 部分打破了这个防御：

- **Real-time protocol correction。** GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1 可以观察 wet-lab video，并在 procedure 中途标记 errors。
- **OpenAI 2025 年 12 月 demonstration。** GPT-5 迭代 wet-lab experiments，通过 protocol optimization 实现 79x efficiency improvement。

含义：把 execution skill 当作防御的有效性正在削弱。Procurement 和 equipment gaps 仍然存在，但 tacit-knowledge gap 正在变窄。

### Cyber uplift（2025 年 11 月）

Anthropic 2025 年 11 月报告：与中国有关的 state actors 使用 Claude 的 agentic coding tool 自动化了 cyberattack campaign 的 80-90%。Human intervention 只在 4-6 个步骤中需要。

含义：

- Agentic coding 是 attack-automation primitive。之前的 AI cyber assistance 被限制在 code-snippet level；agentic workflows 把 reconnaissance、exploitation、post-exploitation 和 exfiltration 整合起来。
- 这 4-6 个 human steps 是 bottleneck；未来 capability gains 会减少这个数量。
- Defensive dual-use：OpenAI 的 “trusted access” pilot 为经过审查的 security organisations（成熟 incident-response firms、government）提供防御所需的 capability access。如果这个 pilot 扩大，access asymmetry 会偏向 defenders。

### Nuclear

在公开文档中，nuclear 是四个 CBRN domains 中分析最少的。Threat model 不同：难点主要是 fissile-material acquisition，而不是 information。AI 在 information layer 上的 uplift 对 novices 的实践帮助有限。2024-2025 年没有 major-lab report 识别出 nuclear-specific threshold crossing。

### Novice-relative vs expert-absolute

四个 domains 中都有一个模式：

- **Novice-relative uplift。** 高。Multiplicative。按照 Anthropic 2025 bio，为 2.53x。
- **Expert-absolute capability。** 上限高。Expert 从模型中获得的能力高于 novice，因为 expert 知道该问什么，也知道如何解释回答。

对 safety cases 的含义：只处理 novice uplift（通过 input filters、refusals、uncertainty）不足以控制 expert-absolute capability。还需要其他措施：elicitation-hardening、capability unlearning（第 17 课）和 control protocols（第 10 课）。

### Cross-domain synthesis

| Domain | 2024 | 2025 | Inflection |
|---|---|---|---|
| Bio | mild uplift | 2.53x uplift，接近 ASL-3 | acquisition-phase automation |
| Chem | mild uplift | 通过 vision 出现 execution-gap erosion | real-time wet-lab correction |
| Cyber | code assistance | 80-90% campaign automation | agentic coding |
| Nuclear | limited | limited | material-access bottleneck 仍然成立 |

三个 domains 跨过了 thresholds。一个仍受非信息性障碍约束。

### 它在 Phase 18 中的位置

第 30 课是 capstone：当前 dual-use picture，每个前置 lesson 都在为它提供测量、限制或治理能力。第 17-18 课给出 measurement 和 frameworks；第 12-16 课给出 evaluation tooling；第 24-25 课给出 regulatory 和 disclosure layer；第 28 课给出 research ecosystem。第 30 课是 evidence 落地的位置。

## Use It

没有代码。阅读 Anthropic 2025 年 11 月 cyber report、OpenAI 2025 年 4 月 Preparedness Framework v2 update，以及 Council on Strategic Risks 2025 AI x Bio wrapup。

## Ship It

本课产出 `outputs/skill-dual-use-triage.md`。给定一条 2026 capability claim 或 incident report，它会在四个 domains 之间进行 triage，并识别该 claim 影响的是 novice-relative uplift、expert-absolute capability，还是两者都有。

## 练习

1. 阅读 Anthropic 2025 年 11 月 cyber report。枚举 4-6 个 human-intervention steps，并论证其中哪一个最可能首先被下一代模型自动化。

2. Chem/bio execution gap 正在通过 vision 被削弱。设计一个 evaluation，用来测量 tacit-knowledge uplift，同时不跨越 ITAR/EAR 边界。

3. Nuclear uplift 看起来受 material access 限制。分别支持和反对这个观点：未来 AI breakthrough 可能改变这个 bottleneck。

4. 为一个具备 cyber capability 的 frontier model 构造 safety case（第 18 课 three-pillar），同时约束 novice 和 expert uplift。

5. 从四个 domains 中选择一个，根据 2024-2025 trajectory 写一段 2027 forecast。指出什么 evidence 会 falsify 你的 forecast。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Uplift | “AI helps attackers” | 可归因于 AI assistance 的 attacker capability 增长 |
| Novice-relative uplift | “multiplicative” | AI 相比 status-quo 对 novice 有多大帮助 |
| Expert-absolute capability | “ceiling” | Expert 能从模型中提取出的最大能力 |
| Execution gap | “doing vs knowing” | 历史防御：tacit wet-lab skill 阻挡 novices |
| Agentic coding | “autonomous attacks” | Multi-step autonomous cyber-task execution |
| Acquisition phase | “pre-synthesis steps” | Bio threat 中的 procurement、equipment、permit stages |
| Trusted access | “defender-only pilot” | OpenAI 2025 program，为经审查的 defenders 提供 capability access |

## 延伸阅读

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — Chinese-linked campaign automation
- [OpenAI — Preparedness Framework v2（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/) — bio “on the cusp”
- [Anthropic — RSP v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio thresholds
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — year-end synthesis
