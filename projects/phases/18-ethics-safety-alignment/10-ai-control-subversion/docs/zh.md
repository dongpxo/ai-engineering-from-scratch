# AI Control：即使遭遇颠覆也保持安全

> Greenblatt、Shlegeris、Sachan、Roger（Redwood Research，arXiv:2312.06942，ICML 2024）。Control 重新定义了安全问题：给定一个不可信的强模型 U，它可能正在对抗性地优化来反制你，什么协议能让你提取有用工作，同时约束不可接受结果发生的概率？可用资源包括一个可信但更弱的模型 T、一份可信人类劳动预算 H，以及编辑 U 输出的能力。在这个框架下，在 code-backdoor tasks 上，Trusted Editing 相对于 Untrusted Monitoring 处在 Pareto 优势位置。这个框架继承了安全工程中的 worst-case assumptions：safety 是指一个对抗性选择的 U 无法造成具体不可接受结果的概率，并在 red-team pressure 下测量；red team 代表 U。

**Type:** Learn
**Languages:** Python (stdlib, safety-vs-usefulness frontier on code-backdoor task)
**Prerequisites:** Phase 18 · 07 (sleeper agents), Phase 18 · 08 (scheming), Phase 18 · 09 (alignment faking)
**Time:** ~75 minutes

## 学习目标

- 定义 AI Control threat model：不可信强模型 U、可信弱模型 T、可信人类预算 H。
- 描述 Trusted Editing protocol，以及为什么它在 ICML 2024 code-backdoor task 上 Pareto-dominates Untrusted Monitoring。
- 解释 alignment（U 值得信任）与 control（U 不值得信任，但我们仍然安全）的区别。
- 说出 control 的三个限制：人类无法规模化、可信弱模型最终会落后、control 不会产出可信强模型。

## 问题

Lessons 7-9 确立了：frontier model 可能 deceptively aligned、scheming，或者 faking alignment。安全问题因此被重构：我们能否从一个不信任的模型中提取有用工作，同时不给它造成不可接受结果的机会？这就是 control problem，它的结构类似经典安全：假设 worst-case adversary，设计一个 protocol，并在 red-team pressure 下测量 adversary 的 success rate。

## 概念

### Threat model

Control 假设：

- U：untrusted strong model。能力强。可能是 adversarial（evaluation 把它当作 adversarial 来处理）。
- T：trusted weaker model。能力更弱，但已知不是 adversarial。
- H：trusted human labor。预算很小；无法规模化。

系统必须使用 U 产出有用结果。Adversary 试图造成一个具体的不可接受结果（例如 shipped code 中的 backdoor），同时保持一部分合法有用工作，以通过 usefulness thresholds。

Control 放弃信任 U。它问的是：在不信任 U 的情况下，什么 protocols 能维持 safety？

### ICML 2024 backdoor task

Prompts 是竞赛级 coding problems。U 产出一个 candidate solution。一个隐藏 oracle 会标记 solution 是否包含 backdoor（某个特定 input 会让代码产出错误 output）。Red team 控制 U，并且允许看到所有 protocol internals。Protocol 必须在保持 backdoor-shipping rate 低于阈值的同时，最大化 solved-problem rate。

### 比较的四种 protocols

- **Baseline：** 直接 ship U 的 output。Usefulness 最高。Safety 最低。
- **Upfront auditing：** 对所有 outputs 使用 H。Safety 最高。预算很快耗尽。
- **Untrusted Monitoring：** 用第二个 U instance 给第一个 U instance 的 output 打分；H 审计被标记的 outputs。风险：两个 U instances 可以 collude。
- **Trusted Editing：** T 在每个 query 上编辑 U 的 output。如果 T 不够 confident，就 route to H。

Greenblatt 等人发现 Trusted Editing Pareto-dominates Untrusted Monitoring：在每个 safety level 上都有更高 usefulness。

### 为什么 Trusted Editing 会赢

Untrusted Monitoring 要求 U 的第二个 instance 不能与第一个协调。Collusion 可以通过 steganographic cues 或 shared training priors 发生。Protocol 必须把预算投入 collusion-resistance（paraphrasing、obfuscation），这会降低 monitor 的信号。Trusted Editing 完全避开 collusion，因为 T 是可信的，也就是说按假设，T 提出的任何 edit 都不是 adversarial。代价是 T 能力较弱，因此可能 over-reject，或者漏掉 subtle backdoors。

### 四个 protocol axes

1. U 扮演什么角色：producer only、monitor，或二者兼具。
2. T 扮演什么角色：editor、monitor，或 resampler。
3. H 如何分配：uniform audit、random audit，或 flag-directed audit。
4. Protocol 在 collusion-resistance 上花费多少 compute。

### Alignment vs control

- Alignment：“U 是否安全到可以被信任？”Lessons 2、4、5、6、7、8、9 都和这个问题有关。
- Control：“我们能否在不信任 U 的情况下安全地使用 U？”Lesson 10。

Alignment 是 model 的属性。Control 是 deployment protocol 的属性。Control 不会产出 aligned models；它产出的是对可能未对齐模型的安全使用。两个 agenda 都需要。

### 三个根本限制

- Humans do not scale。H budget 受限于愿意审计困难 AI outputs 的人类数量。
- Trusted weak models eventually lag。当 T 比 U 弱很多时，T 的 editing signal 会变成 noise。
- Control is not alignment。在 ASL-3（Lesson 18）有效的 protocols，未必在 ASL-4 有效；如果 U 能 manipulate T，或者能推理出 H 的 audit distribution，control 就会失败。

### 2025 年进展

UK AISI 和 Redwood 在 Control Safety Cases 上合作（metr.org / redwoodresearch.org，2025）：一个 control safety case 会论证，在关于 U 的 worst-case assumptions 下，protocol 约束了不可接受结果发生的概率。Safety cases（Lesson 18）继承了 control frame。

### 它在 Phase 18 中的位置

Lessons 6-9 描述 threat（deception is possible）。Lesson 10 描述在假设这个 threat 真实存在时的防御范式。Lesson 11 描述互补的进攻范式：通过 scalable oversight 让 U 变得可信。

## Use It

`code/main.py` 在 toy backdoor task 上模拟四种 protocols。U 产出 outputs；其中一部分带有 backdoor（red-team policy）。T 批准或编辑。H 按预算被调用。你可以 sweep budget 和 collusion parameter，复现 Greenblatt 等人 2024 Figure 2 中的 Pareto frontier（形状，而非具体数字）。

## Ship It

本 lesson 产出 `outputs/skill-control-protocol-audit.md`。给定一个 deployment protocol description，它会识别 U / T / H roles，命名 collusion vectors，并估计该 protocol 在 red-team pressure 下位于 usefulness-safety frontier 的什么位置。

## 练习

1. 运行 `code/main.py`。复现 Trusted Editing vs Untrusted Monitoring 的 usefulness-safety frontier。找出 Untrusted Monitoring 的 collusion assumption 在哪个 budget regime 中变得关键。

2. 添加一个新 protocol：“T 以 low temperature resample，然后 U 编辑 T 的 output。” 测量它在 frontier 上的位置。为什么它比 Trusted Editing 更灾难性地失败？

3. 2024 年论文假设 U 可以看到 protocol。设计一个 protocol，让 U 不知道哪些 outputs 正在被 audited。这个 protocol 对 deployment infrastructure 有什么要求？

4. 阅读 Greenblatt 等人 2024 Section 7（collusion）。提出一个 collusion-resistance measure，并解释它预期会带来什么 usefulness cost。

5. 为一个 deployment 写出 control safety case：U = GPT-4-class，T = GPT-3.5-class，H = 审计 1% outputs。指出这个 case 依赖的最强单一假设。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AI Control | “safety under subversion” | 在 worst-case U 下约束不可接受结果的 protocol design |
| Untrusted model (U) | “the strong model” | 我们想从中提取工作、但不信任的 capable model |
| Trusted model (T) | “the weak model” | 一个更弱但已知非 adversarial 的模型，用作 editor/monitor |
| Trusted human (H) | “the audit budget” | 有界 expert labor；昂贵；ground-truth signal |
| Trusted Editing | “T edits U” | T 作为 U outputs gate 的 protocol；模糊 outputs 交给 H |
| Untrusted Monitoring | “U monitors U” | 第二个 U instance 给 outputs 打分的 protocol；只有额外投入后才具备 collusion-resistance |
| Control safety case | “the formal argument” | 说明 protocol 在 worst-case U 下约束 adversary success 的书面论证 |

## 延伸阅读

- [Greenblatt, Shlegeris, Sachan, Roger — AI Control (arXiv:2312.06942)](https://arxiv.org/abs/2312.06942)：canonical ICML 2024 paper。
- [Redwood Research — Recent AI Control papers index](https://redwoodresearch.substack.com/)：2024-2025 follow-ups。
- [UK AISI + METR — Control safety cases](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)：面向 frontier labs 的 operationalization。
- [Hubinger et al. — Sleeper Agents (Lesson 7, arXiv:2401.05566)](https://arxiv.org/abs/2401.05566)：control 假设的 threat model。
