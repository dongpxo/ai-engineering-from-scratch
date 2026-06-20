# AI Scientist v2：Workshop 级自主研究

> Sakana 的 AI Scientist v2（Yamada 等，arXiv:2504.08066）会跑完整研究循环：提出假设、编写代码、执行实验、生成图表、撰写论文、提交评审。它是第一个让生成论文通过 ICLR 2025 workshop peer review 的系统。独立评估（Beel 等）发现，42% 的实验因代码错误失败，文献综述也经常把已有概念误标为新颖。Sakana 自己的文档也警告：该代码库会执行 LLM 写出的代码，并建议用 Docker 隔离。这个正反两面的图景，正是本课的重点。

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## 问题

研究是开放式任务。和 AlphaEvolve 的算法搜索或 DGM 的 benchmark-bounded self-modification 不同，一个研究结果没有机器可直接检查的正确性标准。论文由评审判断，而不是由单元测试判断。这让循环更难闭合；一旦闭合也更有价值，因为研究正是复利式进展发生的地方。

AI Scientist v1（Sakana，2024）通过从人类写好的模板出发来闭合循环。LLM 在固定脚手架里填入实验。AI Scientist v2（Yamada 等，2025）去掉了模板要求，改用带 vision-language model critique loop 的 agentic tree search。系统会生成想法、实现实验、产出图表、撰写论文，并根据评审反馈迭代。

Peer review 的结论是：一篇由 v2 生成的论文被 ICLR 2025 workshop 接收（并披露了来源）。独立评估的结论是：系统离可靠还很远。两者同时成立。

## 概念

### 架构

1. **想法生成。** LLM 基于主题和既有文献提出研究想法。v1 使用模板；v2 在假设空间上做 agentic search。
2. **新颖性检查。** 文献检索步骤检查该想法是否已经发表。Beel 等的评估就在这一步发现了误标问题：已有方法经常被归类为新颖。
3. **实验计划。** Agent 起草实验协议并写代码。
4. **执行。** 代码在 sandbox 里运行。失败会反馈进 retry loop。Beel 等的测量显示，这一阶段有 42% 的实验因代码错误失败。
5. **图表生成。** Vision-language model 读取生成的图表，并为清晰度重写它们。这是 v2 的关键技术新增点。
6. **论文撰写。** LLM 起草论文，并和内部 reviewer 迭代。
7. **可选：提交。** 论文提交到某个 venue。

### Workshop 接收结果意味着什么

一篇由 v2 生成的论文通过了 ICLR 2025 workshop 的 peer review。作者向 program committee 披露了论文来源。这个接收结果是一个数据点；它不是声称系统“会做研究”的许可证。

重要背景是：workshop papers 的门槛低于主会论文。Peer review 本身有噪声；任何一天都会有一小部分投稿被接收。一次成功是 proof of concept，不是可靠性声明。Nature 2026 论文记录的是 end-to-end loop，而且那篇论文本身由人类研究者共同署名；它并不是“系统写了一篇 Nature 论文”。

### 独立评估发现了什么

Beel 等（arXiv:2502.14297）做了一次外部评估。核心发现：

- **实验失败。** 42% 的实验因代码错误失败（错误 import、shape mismatch、未定义变量）。Retry loop 捕获了一部分，但不是全部。
- **新颖性误标。** 文献检索步骤经常把已有概念标成新颖。这相当于研究领域里的 hallucination。
- **呈现质量缺口。** Vision-language figure critique 会产出 publication-grade visuals，从而遮盖底层实验弱点。

最后一条是本 phase 的重点。一个能产出可信外观、但没有做出可信研究的系统，比一个明显失败的系统更危险，而不是更安全。评估必须触达底层 claims，不能停在图表层面。

### Sandbox escape 风险

Sakana 自己的 repository README 提醒：

> 由于这个软件会执行 LLM 生成的代码，我们无法保证安全。它存在引入危险 packages、失控访问网络、启动非预期进程等风险。请自行承担风险，并考虑使用 Docker 隔离。

这就是未验证领域里 autonomy 的操作形态。LLM 写代码；代码运行；代码能做任何当前进程被允许做的事。如果 sandbox 没有硬性限制 filesystem、network 和 process actions，任何自导向研究 agent 都可能外泄数据、烧掉算力，或重写自身。

AlphaEvolve 的 sandbox story 更容易，因为它的 evaluator 很紧。AI Scientist v2 的循环会运行带开放目标的开放式代码。因此它需要更强隔离（Docker 是最低线；seccomp / gVisor 更好），并且每次产物离开系统前都需要人工 review。

### v2 在 frontier stack 中的位置

| System | Target | Output kind | Evaluator | Known failure |
|---|---|---|---|---|
| AlphaEvolve | algorithms | code | unit + benchmark | bounded by evaluator rigor |
| DGM | agent scaffolding | code | SWE-bench | reward hacking |
| AI Scientist v2 | research papers | text + code + figures | peer review (weak) | experiment failures, mislabeling, polish masking weakness |

三者之中，v2 的自动 evaluator 最弱、输出面最宽，也最容易直接生成公开 artifact。真正承担大部分安全工作的，是操作控制：sandbox、review 和 disclosure。

## Use It

`code/main.py` 会把 v2 循环模拟成一个 state machine：idea → novelty check → experiment → figure → writeup → review → accept-or-iterate。每个状态都有一个可配置的失败概率，取自 Beel 等的发现。运行模拟器 N 个循环，并统计：

- 多少想法能走到 submission。
- 多少 submission 会带有 polished paper 掩盖起来的 critical experimental flaw。
- Retry budgets 如何在 quality 和 yield 之间取舍。

## Ship It

`outputs/skill-ai-scientist-sandbox-review.md` 是一份 two-gate review checklist，用于任何 research-loop agent 的产物离开 sandbox 之前。

## 练习

1. 用默认参数运行 `code/main.py`。有多大比例的 loop runs 会产出一篇“clean” paper？有多大比例会产出一篇带有 experiment-failure flaw、但被 figure critique 润色遮盖的 paper？

2. 默认值已经使用 Beel 等的 42% / 25%。用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 重新运行，再用 `--experiment-failure 0.60 --novelty-mislabel 0.40` 重新运行。两次运行中 polished-but-flawed 的占比如何变化？

3. 阅读 Sakana 的 AI Scientist v2 repo README 中关于 sandbox requirements 的部分。说出两个你会为 multi-day autonomous run 增加的限制（Docker 之外）。

4. 阅读 Beel 等第 4 节关于 presentation-quality gap 的内容。设计一个额外 evaluator，用来抓住外观精美但实验有缺陷的论文。

5. 为 research-agent outputs 设计一套比“让一个 PhD 读每篇论文”更可扩展的 human-review protocol。找出 bottleneck，并围绕它设计。

## 关键术语

| Term | 常见说法 | 实际含义 |
|---|---|---|
| AI Scientist v1 | “Sakana's templated research agent” | 把实验填进固定 scaffold |
| AI Scientist v2 | “Template-free research agent” | 带 VLM figure critique 的 agentic tree search |
| Agentic tree search | “Branching research agent” | 并行扩展多个 experiment plans；由 internal critic 修剪 |
| Vision-language critique | “VLM polish on figures” | 多模态模型读取图表，并为清晰度重写 |
| Literature retrieval | “Novelty check” | 搜索 prior work 来确认想法新颖性；已有文档显示它会误标 |
| Polish masking | “Pretty paper, broken research” | 呈现质量超过实验质量，因而隐藏弱点 |
| Sandbox escape | “LLM code breaks out” | Agent 执行的代码做了 loop designer 没打算让它做的事 |

## 延伸阅读

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — paper.
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — vendor summary with peer-review context.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — external evaluation numbers.
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — the templated predecessor.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — broader framing of open-ended research agents.
