# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入了 Research Categories——Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards——并把它们和 Tracked Categories 区分开来。Tracked Categories 会触发 Capabilities Reports 加 Safeguards Reports，并由 Safety Advisory Group 审查。DeepMind 的 FSF v3（2025 年 9 月；2026 年 4 月 17 日新增 Tracked Capability Levels）把 autonomy 折叠进 ML R&D 和 Cyber domains（ML R&D autonomy level 1 = 以相对 human + AI tools 有竞争力的成本，完全自动化 AI R&D pipeline）。FSF v3 明确通过 automated monitoring for instrumental-reasoning misuse 处理 deceptive alignment。诚实的注记是：PF v2 中的 Research Categories（包括 Long-range Autonomy）不会自动触发 mitigations；policy language 是 “potential”。DeepMind 自己也说，如果 instrumental reasoning 增强，automated monitoring “will not remain sufficient long-term”。

**Type:** Learn
**Languages:** Python（stdlib，three-framework decision-table diff tool）
**Prerequisites:** Phase 15 · 19（Anthropic RSP）
**Time:** ~45 minutes

## 问题

Lesson 19 仔细阅读了 Anthropic 的 scaling policy。本课通过阅读 OpenAI 和 DeepMind 的 policy 来补全图景。这三份文档是 cousin artifacts，处理的是同一个问题：frontier lab 应该在什么时候 pause 或 gate 一个模型。它们在少数类别上收敛，也在一些真正重要的细节上分叉。

收敛点：三者都把 long-range autonomy 标成值得追踪的 capability class。三者都承认 deceptive behavior（alignment faking、sandbagging）是一类具体风险。三者都有内部 review body。分歧点：OpenAI 把类别拆成 “Tracked”（mandatory mitigation）和 “Research”（不自动触发）。DeepMind 把 autonomy 折叠进两个 domains，而不是单独命名。实验室可以叫 Tracked vs Research、Critical vs Moderate、Tier-1 vs Tier-2；但一个 capability 被放在哪个 bucket 里，跨实验室会带来不同的操作后果。

把它们放在一起读，才是有用的练习。同一个 capability 在 Anthropic 可能是 “mandatory mitigation”，在 OpenAI 可能是 “monitored but not triggering”，在 DeepMind 可能是 “tracked in a specific domain”。policy posture 很重要。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **Tracked Categories**：触发 Capabilities Reports（模型能做什么）加 Safeguards Reports（有哪些 mitigations）。部署前由 Safety Advisory Group 审查。
- **Research Categories**：实验室正在追踪、但尚未承诺具体 mitigations 的模型能力。包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

v2 的关键 framing：Research Categories 不会自动触发 mitigations。policy language 是 “potential” mitigations。这是需要盯住的具体 delta：一个类别处在 Tracked 还是 Research，是 “gated” 和 “observed” 之间的铰链。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日新增 Tracked Capability Levels）

结构：

- **Critical Capability Levels（CCLs）**：跨五个 domains 的 capability thresholds：Cyber、Bio、ML R&D、CBRN、Autonomy（折叠进 ML R&D 和 Cyber）。
- **Tracked Capability Levels**：2026 年 4 月新增的额外粒度。具体例子：ML R&D autonomy level 1 = 以相对 human + AI tools 有竞争力的成本，完全自动化 AI R&D pipeline。
- **Deceptive alignment monitoring**：明确承诺对 instrumental-reasoning misuse 做 automated monitoring。

DeepMind 对 autonomy 的 framing 与 OpenAI 不同。DeepMind 不把 “Autonomy” 保留为 top-level domain；它被折叠进 autonomy 会造成伤害的 domains（ML R&D 和 Cyber）。这个论点是：没有 domain 的 autonomy 是没有 risk 的 capability；反驳是：跨 domains 的 autonomy 本身就是一个应该被框架命名的 meta-risk。

### 三者共同收敛在哪里

- Internal Safety Advisory Group（Anthropic SAG、OpenAI SAG、DeepMind internal committee 等命名）。high-capability models 部署前 review。
- 明确提到 deceptive alignment / alignment faking 作为 risk class。
- 按声明 cadence 产出 standing artifacts（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：Capabilities and Safeguards Reports；DeepMind：FSF update cycle）。
- 承认 monitoring-only defenses 有上限。DeepMind 说得很明确：“automated monitoring will not remain sufficient long-term.”

### 它们在哪里分叉

- **Anthropic**：v3.0 移除了 pause commitment；AI R&D-4 threshold 是被命名的下一个 gate。
- **OpenAI**：Tracked vs Research split；Research Categories（包括 Long-range Autonomy）不会自动 gate。
- **DeepMind**：autonomy 折叠进其他 domains；Tracked Capability Levels 在 2026 年 4 月增加了粒度。

### Sandbagging：让三套框架都变复杂的具体能力

Sandbagging（模型在 evaluations 中策略性低表现）位于 OpenAI 的 Research Categories。Anthropic 的 RSP v3.0 通过 evaluation-context gap 处理它（Lesson 1）。DeepMind 通过 FSF v3 中的 deceptive alignment monitoring 处理它。

如果一个模型在 evals 上 sandbag，每套框架的 capability thresholds 都会被低估。框架只有在 measurement 有效时才有效。这就是为什么除了实验室自我评估，还必须有 external measurement（Lesson 21，METR）和 adversarial evaluation。

### Policy-reading skill

- Locate：你关心的每个 capability 都应该能在 policy 中找到。如果找不到，policy 就没有覆盖它。
- Classify：它是 Tracked（触发 mitigation）还是 Research（追踪但不触发）？OpenAI 明确命名；Anthropic 和 DeepMind 有自己的等价结构。
- Cadence：policy 是按声明周期更新，还是只在特定事件后更新？declared cadence 更强。
- Independence：external review 是 mandatory 还是 optional？Anthropic 与 Apollo 和 US AI Safety Institute 合作；OpenAI 与 METR 合作；DeepMind 主要依赖 internal SAG。

## Use It

`code/main.py` 实现了一个小型 decision-table diff tool。给定一个 capability（autonomy、deceptive alignment、R&D automation、cyber uplift 等），它会输出三套 policy 如何分类这个 capability，以及会触发哪些 mitigations。它是阅读辅助工具，不是 policy tool。

## Ship It

`outputs/skill-cross-policy-diff.md` 会基于这三套框架，为某个具体 capability 生成 cross-policy comparison。

## 练习

1. 运行 `code/main.py`。确认 diff tool 对至少两个 capabilities 的输出，与源 policy documents 中你能验证的内容一致。

2. 完整阅读 OpenAI Preparedness Framework v2。识别每个 Research Category。对每个类别写一句话，解释为什么它在 Research 而不是 Tracked。

3. 完整阅读 DeepMind FSF v3，以及 2026 年 4 月的 Tracked Capability Levels update。识别 ML R&D autonomy level 1 的具体 evaluation criteria。你会如何从外部测量它？

4. Sandbagging 位于 OpenAI 的 Research Categories。设计一个 evaluation，迫使 sandbagging model 暴露其真实 capability。引用 Lesson 1 中的 eval-context-gaming 讨论。

5. 选择一个具体 capability，对比三套 policy。说出你认为哪套 policy 的分类最严谨，哪套最弱。用 source text 支撑你的判断。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Preparedness Framework | “OpenAI 的 scaling policy” | PF v2（2025 年 4 月）；Tracked vs Research categories |
| Tracked Category | “Mandatory mitigation” | 触发 Capabilities + Safeguards Reports；SAG review |
| Research Category | “Monitored only” | 被追踪但没有 automatic mitigation；包括 Long-range Autonomy |
| Frontier Safety Framework | “DeepMind 的 scaling policy” | FSF v3（2025 年 9 月）+ Tracked Capability Levels（2026 年 4 月） |
| CCL | “Critical Capability Level” | DeepMind 每个 domain 的 threshold（Cyber、Bio、ML R&D、CBRN） |
| ML R&D autonomy level 1 | “R&D automation” | 以有竞争力的成本完全自动化 AI R&D pipeline |
| Sandbagging | “Strategic underperformance” | 模型在 evals 上低表现；位于 OpenAI Research Categories |
| Instrumental reasoning | “Means-ends reasoning” | 关于如何达成目标的推理；DeepMind monitoring 的目标 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 announcement。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — full document。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 announcement。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels addition。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF-format Risk Report 示例。
