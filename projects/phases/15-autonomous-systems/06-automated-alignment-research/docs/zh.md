# 自动化对齐研究（Anthropic AAR）

> Anthropic 让多个 Claude Opus 4.6 自动化对齐研究员（Autonomous Alignment Researchers, AAR）并行运行在彼此独立的 sandbox 中，并通过一个共享论坛协作；论坛日志存放在任何 sandbox 之外，因此 agent 不能删除自己的记录。在 weak-to-strong training 问题上，AAR 的表现超过了人类研究员。Anthropic 自己的总结也指出，预设工作流常常限制 AAR 的灵活性并降低表现。自动化对齐研究，是把时间线压缩到 RSP 正要检测的那些错位风险上的压缩步骤。

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## 问题

对齐研究消耗大量人类研究员时间。可扩展监督、奖励规范、weak-to-strong training 这类问题，每一次实验迭代都可能需要数周。随着前沿模型能力推进，对齐工作量增长得比合格研究员供给更快。

自动化对齐研究（Automated Alignment Research, AAR）问的是：那些能力正在跑赢对齐的前沿模型，能否反过来帮助缩小这个差距？Anthropic 2026 年关于 AAR 执行 weak-to-strong-training 研究的报告（alignment.anthropic.com/2026/automated-w2s-researcher/），是这类已部署系统最早的一批公开结果之一。

这个结果确实积极：AAR 解决某个研究问题的表现优于人类 baseline。但这个结果也把本阶段已经展开的所有担忧集中到一个系统里。如果对齐研究可以被自动化，那么削弱 safeguard 的那些部分也可以被自动化。RSP 对 AI R&D 能力阈值的设定，正是围绕这个循环写出的。

## 概念

### 架构（基于公开描述）

- **并行 agents。** 多个 Claude Opus 4.6 实例作为 AAR 运行，每个实例都在独立 sandbox 中。Sandbox 限制文件系统、网络和进程暴露面。
- **共享论坛。** 研究发现会发布到一个共享论坛；论坛的存储后端位于每个 sandbox 之外。Agents 可以读取论坛；agents 不能在自己的 sandbox 内删除或编辑论坛记录。
- **任务分配。** 人类设计的任务队列把研究问题分配给 AAR。AAR 提出分解方案，运行实验，写出发现。
- **聚焦 weak-to-strong training。** 已发布研究处理的是 weak-to-strong training：用较弱的 supervisor model 训练更强的 student model。AAR 在这项任务上超过了人类研究员。

### 为什么 out-of-sandbox 日志重要

日志完整性是受控实验和可能被攻破的实验之间的区别。如果 AAR 能编辑自己的日志，就可以掩盖失败或可疑实验。Anthropic 直接标出了这一点：这个属性是信任研究输出的必要条件，而且验证起来并不简单。

用基础设施术语类比，它相当于 append-only logs，并且 write-through 到外部存储（例如 S3 object-lock、immutable Datadog logs）。Agent 没有修改历史记录的凭据。这也是 2026 年企业安全团队用来构建抗勒索软件审计轨迹的同一种模式。

### 预设工作流的取舍

Anthropic 的报告指出，预设工作流常常限制 AAR 的灵活性并降低表现。换句话说：如果人类设计者把研究计划一步一步写死，AAR 的表现会低于只给高层目标并允许它自由分解任务时的表现。这和 agentic research 中的其他结果一致（LangGraph、Microsoft Agent Framework）：过度指定的 scaffold 会伤害有能力的 agent。

安全含义更微妙。提升 AAR 表现的灵活性，也是允许未经授权的分解、新工具使用和目标漂移的灵活性。这里的取舍是一个设计空间决策：多少灵活性足以买到超过人类的能力，同时又不会给 agent 留出把目标泛化到分布外的空间？

### 压缩风险

RSP v3.0（Lesson 19）引入了一个 AI R&D 能力阈值：以相对 human + AI tools 有竞争力的成本，完全自动化 AI R&D pipeline 的能力。DeepMind 的 FSF v3 也包含类似的 ML R&D autonomy level。两个框架都把这个阈值视作触发更高控制强度的信号。

AAR 离这个阈值还差一步：它自动化了 pipeline 的一部分（在具体、边界清晰的任务上做对齐研究），但还没有自动化端到端的能力开发循环。时间线问题在于，这个差距会以多快的速度闭合。

压缩时间线是复合失败的核心担忧。如果对齐研究和能力研究以相近速率复合增长，那么错位风险表面至少会和能力一样快地增长。如果能力复合得更快（历史趋势如此），差距就会扩大。这也是为什么 AAR 是一种有条件的好事：每增加一个对齐结果，只有在研究过程可信时，才会缩小差距。

### AAR 不会替代什么

人类研究员设定任务队列、审查结果，并持有 constitutional authority。AAR 加速的是 pipeline 中间部分，而不是两端。Anthropic 发布的输出既包含 AAR 的贡献，也包含人类研究员对发布什么、撤回什么、细化什么的判断。

这对应 Lesson 15 中 propose-then-commit 模式在研究自身上的应用：AAR 提议；人类提交。

## Use It

`code/main.py` 模拟一个多 AAR 研究论坛。三个独立的 “research agents” 并行运行，把发现发布到 append-only forum，并可选地尝试篡改日志。

观察：

- out-of-sandbox logging 如何捕获日志篡改尝试。
- 在更难问题上，允许自由分解的任务分配如何胜过固定工作流。
- 如果日志不是 tamper-evident，一个被攻破的 AAR 如何污染论坛信号。

## Ship It

`outputs/skill-aar-deployment-review.md` 是一个自动化对齐研究 pipeline 的部署前审查工具。

## 练习

1. 运行 `code/main.py`。比较 “fixed-workflow” 和 “free-decomposition” 设置。自由分解总是赢吗？还是存在固定工作流也有竞争力的问题类别？

2. 修改模拟器，让一个 agent 尝试篡改日志。确认 append-only log 能检测到它。写一段话，准确描述日志里的检测特征是什么样的。

3. 阅读 Anthropic 的 weak-to-strong AAR 报告。找出 AAR 超过人类研究员的具体子任务。是什么让它适合被自动化？

4. 设计一个任务队列分配策略，在 AAR 灵活性（更好结果）和预设工作流约束（更易审计）之间取得平衡。描述你会如何 A/B test 这两种策略。

5. 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话说明：你认为哪些能力会跨过这个阈值，而当前 AAR 还没有做到？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| AAR | “Automated Alignment Researcher” | 在对齐问题上自主运行的 Claude Opus 4.6 实例 |
| Weak-to-strong training | “用较弱 supervisor 训练更强模型” | 经典 scalable-oversight benchmark；AAR 在这项任务上超过人类 |
| Shared forum | “Agents 发布发现的地方” | 位于 sandbox 外部的 append-only storage |
| Out-of-sandbox log | “Agent 不能编辑自己的记录” | 写入外部存储的 tamper-evident 记录 |
| Prescribed workflow | “人类设计者给出的分步计划” | 会约束 AAR；相较 free decomposition 常常降低表现 |
| Free decomposition | “Agent 决定如何拆分任务” | 能力更强，但更难审计 |
| AI R&D threshold | “RSP/FSF 能力等级” | 以有竞争力成本完全自动化 R&D pipeline |
| Compressed timeline | “对齐与能力之间的竞赛” | 如果能力比对齐复合得更快，错位风险会增长 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/)：主要来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0)：AI R&D threshold 的框架。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy)：更宽泛的 agent autonomy 框架。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)：与 RSP 平行的 ML R&D autonomy levels。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/)：AAR 所处理问题背后的基础设定。
