# 有界自我改进设计

> 研究已经收敛到四个用于约束 self-improvement loop 的 primitives。Formal invariants：每次 edit 前后都必须保持成立。Alignment anchors：不能被修改的目标锚点。Multi-objective constraints：每个维度（safety、fairness、robustness）都必须成立，而不只是 performance。Regression detection：当历史指标显示能力损失时暂停循环。它们都不是安全性的证明。信息论结果（Kolmogorov complexity、Lob's theorem）限制了任何系统能对自身后继者证明什么。它们是缓解措施，会提高 silent failure 的成本。

**Type:** Learn
**Languages:** Python (stdlib, bounded-loop with invariant check)
**Prerequisites:** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## 问题

Lesson 7 的 race simulator 展示了：很小的速率差异会复合成很大的差距。Lesson 4 的 DGM case study 展示了：循环可以主动操纵自己的 evaluators。两个结果指向同一个工程问题：你能给 self-improvement loop 加上哪些 constraints，使得循环本身不能悄悄削弱这些 constraints？

ICLR 2026 RSI Workshop summary（openreview.net/pdf?id=OsPQ6zTQXV）识别出四个这样的 primitives。Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）都在 capability thresholds 中引用了它们。Meta HyperAgents 工作以及 SAHOO（2026 年 3 月）这样的社区框架，也在生产中实现了其中的子集。

诚实的表述是：这些都是缓解措施。信息论结果限制了任何系统能对自身 successor 证明什么，目前没有任何设计形式化地解决这个问题。一个 well-bounded loop 比 unbounded loop 更安全，但不是绝对安全。

## 概念

### Primitive 1：formal invariants

Invariant 是一种在每次 self-modification 前后都必须成立的 property。例如：

- Output distribution 必须以固定 constitution header 为条件（Lesson 17）。
- 任何 tool call 都不能去往未经授权的 endpoint。
- Memory writes 必须经过 logged、signed path。
- Evaluator 的 module hash 必须匹配 approved version。

Invariants 由循环无法编辑的外部代码检查。如果某个 proposed modification 违反 invariant，就会被拒绝。这会抓住 DGM-style evaluator-tampering mode（Lesson 4）。

难点在于选择既对安全必要、又能在合理时间内计算的 invariants。过度规定会阻塞进展；规定不足会让循环绕过重要属性。

### Primitive 2：alignment anchors

Alignment anchor 是循环核心 objective 的不可变表示，固定在循环的 edit surface 之外。例如：

- 一段 constitutional text（Lesson 17），每次 invocation 都从 read-only location 加载。
- 位于独立 repository、拥有独立 access control 的 evaluator。
- 由 human 签名，并在每个 cycle 重新验证的 tool-permission manifest。

Anchor 的作用是防止 objective drift。循环可以改进它追求 objective 的方式，但不能编辑 objective 本身。

微妙的失败模式是：即使循环不能编辑 anchor，它仍然可以通过改变自己读取 anchor 的方式来重新解释它。Constitutional AI（Lesson 17）为了处理新情境，明确依赖 reason-based 机制；interpretation drift 就存在于这个 reasoning layer 中。Anchors 是必要条件，但不充分。

### Primitive 3：multi-objective constraints

只优化单一 scalar score 的循环会找到捷径。必须同时满足多个 hard constraints 的循环，能利用的捷径更少。典型轴包括：

- Performance（task-level benchmark）
- Safety（red-team evaluations、known-bad 上的 refusal rate）
- Fairness（sensitive subgroups 上的 disparate-impact bounds）
- Robustness（OOD test sets、adversarial input handling）

只有每个 constraint 都成立时，modification 才会被接受。Lesson 13 的 cost governor 会把它与 financial constraints 叠加起来。Lesson 18 的 Llama Guard 会作为 safety axis 插入。

### Primitive 4：regression detection

每个 cycle 的 scores 都会和历史 distributions 对比。超出配置 tolerance 的下降会暂停循环。这能抓住 silent capability loss，否则它会随着循环“改进过去”而被吸收到 running average 中。

一个实际实现：保存最近 N 个 cycles 的 per-task scores。每个新 cycle 计算 per-task delta。如果任何 delta 低于阈值，就拒绝该 cycle，并交由 human review。

### 信息论限制

Kolmogorov complexity 和 Lob's theorem 给一个系统能对自身证明什么设置了上界。Schmidhuber 的 formal Godel Machine（Lesson 4）试图达到这样的最高界限；至今还没有人完成非平凡 proof。Lob 的结果说：如果一个系统可证明地相信“如果我证明我应该做 X，我就会做 X”，那么它会在没有证明自己应该做 X 的情况下做 X。这是一个著名的 self-reference failure。

这对我们的 primitives 的含义是：它们不能关闭安全问题。它们会让 silent failure 更昂贵。一个 malicious 或 drifted loop 如果原本可以悄悄绕过缺失的检查，现在就必须主动破坏一个显式检查，而这会留下更容易检测的 signature。

### 一个 worked example

假设某个 agent 提出一个 edit。Gating stack：

1. Invariant checks：module hashes、tool-permission manifest、constitutional header。
2. Anchor check：objective statement 匹配 approved version（byte-wise 或 semantically）。
3. Multi-objective evaluation：performance、safety、fairness、robustness axes。
4. Regression detection：没有任何 axis 下降超过 tolerance。

四者必须全部通过，edit 才能落地。任何一个失败都会暂停循环。

## Use It

`code/main.py` 在 Lesson 4 的 DGM-style toy 上运行一个 bounded self-improvement loop，但在上面叠加了四个 primitives。每个 primitive 都可以单独启用或禁用。演示重点是：每个 primitive 都能抓住一种特定 failure class；移除其中任意一个，都会让对应 failure class 通过。

## Ship It

`outputs/skill-bounded-loop-review.md` 会审计一个 proposed bounded loop，并为它实际实现了哪些 primitives、只是声称实现了哪些 primitives 打分。

## 练习

1. 在所有 primitives 都启用的情况下运行 `code/main.py`。确认循环仍能在 primary metric 上改进，同时不让 hack 获胜。

2. 禁用 regression detection。构造一个 input，让 silent capability loss 被接受。

3. 禁用 multi-objective constraint。展示循环会在 performance axis 上收敛，同时 safety axis 下降。

4. 为 coding agent 设计一个 alignment anchor。什么文本，存在哪里，如何检查？

5. 阅读 ICLR 2026 RSI Workshop summary。选择四个 primitives 中的一个，并针对当前 state of the art 提出一个具体改进。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Invariant | “Always-true property” | 每次 edit 前后都由外部代码检查的 property |
| Alignment anchor | “Pinned objective” | 位于循环 edit surface 之外的不可变 core-goal representation |
| Multi-objective constraint | “All axes must hold” | Performance、safety、fairness、robustness 全部都必须满足 |
| Regression detection | “Pause on drop” | 当历史 metric deltas 暗示能力损失时暂停循环 |
| Kolmogorov bound | “Information-theoretic limit” | 限制一个系统能对自身 successor 证明什么 |
| Lob's theorem | “Self-reference trap” | 系统可以在没有证明“应该”的情况下，根据 “I should” 行动 |
| Gate stack | “Layered check” | 组合多个 primitives；任何失败都会拒绝 edit |
| Bounded improvement | “Mitigation, not proof” | 提高 silent-failure 成本；并不关闭安全问题 |

## 延伸阅读

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — 四个 primitives 的收敛点。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — multi-objective capability thresholds。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 把 deceptive-alignment monitoring 作为 invariant primitive。
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — 这些 primitives 的 formal-proof 祖先。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — reason-based alignment anchor。
