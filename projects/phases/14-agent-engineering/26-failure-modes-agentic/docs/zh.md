# Failure Modes：Agents 为什么会坏掉

> MASFT（Berkeley, 2025）把 14 种 multi-agent failure modes 归入 3 个 categories。Microsoft 的 Taxonomy 说明了既有 AI failures 如何在 agentic settings 中被放大。Industry field data 收敛到五种反复出现的模式：hallucinated actions、scope creep、cascading errors、context loss、tool misuse。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## 学习目标

- 说出 MASFT 的三个 failure categories，并在每一类中至少说出四种具体模式。
- 解释为什么 agentic failure 会放大既有 AI failure modes（bias、hallucination）。
- 描述五种 industry-recurring modes 以及对应 mitigations。
- 实现一个 stdlib detector，用 failure-mode labels 标记 agent traces。

## 问题

团队交付的 agents 可能在 90% 的 traces 上工作正常。剩下 10% 的 failures 并不是随机噪声，它们会落入少数反复出现的 categories。一旦你能命名它们，就能监控它们并修复它们。

## 概念

### MASFT（Berkeley, arXiv:2503.13657）

Multi-Agent System Failure Taxonomy。14 种 failure modes 聚成 3 个 categories。Inter-annotator Cohen's Kappa 为 0.88，说明这些 categories 可以被稳定地区分。

核心主张：failures 是 multi-agent systems 的根本 design flaws，而不是可以通过更好的 base models 修好的 LLM limitations。

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 既有 AI failures（bias、hallucination、data leakage）会在 agentic settings 中被放大。
- Autonomy 会引入新的 failures：unintended action at scale、tool misuse、mission drift。
- 这份 whitepaper 是 agentic products 的 risk register。

### Characterizing Faults in Agentic AI（arXiv:2603.06847）

- Failures 来自 orchestration、internal state evolution 和 environment interaction。
- 不只是 “bad code” 或 “bad model output”。

### LLM Agent Hallucinations Survey（arXiv:2509.18970）

两种主要表现：

1. **Instruction-following Deviation**：agent 没有遵循 system prompt。
2. **Long-range Contextual Misuse**：agent 忘记或误用 earlier turns 中的 context。

Sub-intention errors：Omission（漏掉步骤）、Redundancy（重复步骤）、Disorder（步骤乱序）。

### 五种 industry-recurring modes

Arize、Galileo、NimbleBrain 2024-2026 年的 field analyses 收敛到：

1. **Hallucinated actions.** Agent 调用一个不存在的 tool，或编造 arguments。
2. **Scope creep.** Agent 把任务扩展到用户请求之外（创建额外 PRs、发送额外 emails）。
3. **Cascading errors.** 一次错误调用触发下游影响。一个 phantom SKU hallucination 触发四次 API calls，变成 multi-system incident。
4. **Context loss.** 长程任务忘记 early-turn constraints。
5. **Tool misuse.** 用错误 arguments 调用了正确 tool，或直接调用了错误 tool。

Cascading 是最致命的。Agents 无法区分 “I failed” 和 “the task is impossible”，并且经常在 400 errors 上 hallucinate 一个 success message 来关闭循环。

### Mitigation：每一步都加 gates

在 reasoning chain 的每一步添加 automated verification gates，对照 environment state 检查 factual grounding。具体来说：

- Per-step safety classifier（Lesson 21）。
- Tool-call argument validation（Lesson 06）。
- Cross-check retrieved content against known facts（Lesson 05, CRITIC）。
- 通过重新探测 state 检测 success hallucination（文件真的被创建了吗？）。

### Failure monitoring 容易在哪里出错

- **Tagging only crashes.** 大多数 agent failures 会产出看起来有效的 output。需要 content-level checks。
- **No baseline.** Drift detection 需要 last-known-good；没有它，你无法判断 “this is getting worse”。
- **Over-alerting.** 每个 failure 都发一页告警。应该 cluster 并 rate-limit。

## Build It

`code/main.py` 实现了一个 stdlib failure-mode tagger：

- 一个覆盖五种 modes 的 synthetic trace dataset。
- 每种 mode 对应 detector functions（基于 tool calls、outputs、repeat actions 的 signature patterns）。
- 一个 tagger，会标记每条 trace 并报告 mode distribution。

运行：

```bash
python3 code/main.py
```

输出：per-trace labels + aggregate distribution，这是 Phoenix 的 trace clustering 所展示内容的一个廉价复现。

## Use It

- **Phoenix** 用于 production drift clustering（Lesson 24）。
- **Langfuse** 用于 session replay + annotation。
- **Custom** 用于 observability platform 无法检测的 domain-specific signatures。

## Ship It

`outputs/skill-failure-detector.md` 会生成面向你的 domain 定制的 failure-mode detectors，并接入 trace store。

## 练习

1. 添加一个 “success hallucination” detector：agent 返回 success，但目标 state 没变。
2. 标记你构建过的某个 product 中的 100 条真实 traces。哪种 mode 占主导？修复它的成本是多少？
3. 实现一个 “cascade radius” metric：给定 step N 的 failure，它影响了多少个 downstream steps？
4. 阅读 MASFT 的 14 种 failure modes。挑选三个适用于你 product 的模式，编写 detectors。
5. 把一个 detector 接入 CI job：如果 >=5% 的 traces 被标记为某种 mode，就让 build 失败。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| MASFT | “Multi-agent failure taxonomy” | Berkeley 的 14-mode categorization |
| Cascading error | “Ripple failure” | 一个早期错误沿 N 个 steps 传播 |
| Context loss | “Forgot the constraint” | 长程 turn 丢掉 early-turn facts |
| Tool misuse | “Wrong tool / wrong args” | 调用本身有效，但 invocation 错误 |
| Success hallucination | “Faked completion” | Agent 在 400 上声称成功；state 没变 |
| Scope creep | “Overreach” | Agent 做了超过请求范围的事情 |
| Instruction-following deviation | “Disobedience” | 忽略 system prompt 或 user constraint |
| Sub-intention errors | “Plan bugs” | Plan execution 中的 omission、redundancy、disorder |

## 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure modes、3 categories
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — risk register
- [Arize Phoenix](https://docs.arize.com/phoenix) — drift clustering in practice
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when simpler patterns avoid modes entirely
