# Reflexion：语言强化学习

> 基于梯度的 RL 要修复一个失败模式，通常需要数千次试验和一整套 GPU 集群。Reflexion（Shinn 等，NeurIPS 2023）用自然语言完成同一件事：每次试验失败后，agent 写下一条反思，存进情节记忆，并让下一次试验基于这段记忆继续。这就是 Letta 的 sleep-time compute、Claude Code 的 `CLAUDE.md` 经验记录，以及 pro-workflow 的 learn-rule 背后的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## 学习目标

- 命名 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector），并解释情节记忆的作用。
- 用 stdlib 实现一个 Reflexion 循环，包含二元 evaluator、reflection buffer 和全新的重试。
- 针对给定任务，在 scalar、heuristic 和 self-evaluated feedback sources 之间做选择。
- 解释为什么语言强化可以捕捉到基于梯度的 RL 需要数千次试验才能修复的错误。

## 问题

一个 agent 任务失败了。在标准 RL 里，你会再跑数千次试验，计算梯度，更新权重。这个过程昂贵、缓慢，而且大多数生产 agent 并没有为每一次失败都重新训练的预算。

Reflexion（Shinn 等，arXiv:2303.11366）提出了另一个问题：如果 agent 只是思考自己为什么失败，并把这个想法放进下一次尝试的 prompt 里，会怎样？不更新权重。没有梯度。只是在试验之间保存自然语言。

结果是：在 ALFWorld 上，它超过了 ReAct 和其他非微调基线；在 HotpotQA 上，它相对 ReAct 有提升；在代码生成（HumanEval/MBPP）上，它达到当时的 state of the art。整个过程没有走一步梯度更新。

## 概念

### 三个组件

```text
Actor         : 生成一条 trajectory（ReAct-style loop）
Evaluator     : 给 trajectory 打分——binary、heuristic 或 self-eval
Self-Reflector: 针对失败写出一条自然语言反思
```

再加一个数据结构：

```text
Episodic memory: 之前反思的列表，会 prepend 到下一次试验的 prompt
```

一次 trial 会先运行 Actor。Evaluator 给它打分。如果分数低，Self-Reflector 生成一条反思（例如：“我选错工具，是因为我把问题误读成在问 X，但它其实在问 Y”）。这条反思进入情节记忆。下一次 trial 从头开始，但会看到这条反思。

### 三类 evaluator

1. **Scalar**：外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单，信号最强。
2. **Heuristic**：预定义失败特征。“如果 agent 连续两次产生同一个 action，就标记为卡住。”“如果 trajectory 超过 50 步，就标记为低效。”
3. **Self-evaluated**：LLM 给自己的 trajectory 打分。没有 ground truth 时需要它。信号较弱；最好和工具锚定的验证配合使用（Lesson 05 — CRITIC）。

2026 年的默认组合是：有 scalar 就用 scalar，没有就用 self-eval，再用 heuristics 作为安全护栏。

### 为什么它能泛化

Reflexion 与其说是一个全新算法，不如说是一个被命名的模式。几乎每个生产级“自愈”agent 都在跑某种变体：

- Letta 的 sleep-time compute（Lesson 08）：一个独立 agent 反思过去的对话，并写入 memory blocks。
- Claude Code 的 `CLAUDE.md` / “save memory” 模式：把反思捕获为 learnings，并 prepend 到未来 sessions。
- pro-workflow 的 `/learn-rule` 命令：把纠正捕获成显式规则。
- LangGraph 的 reflection nodes：一个 node 给输出打分，并在需要时路由到 refine。

它们都来自同一个洞察：自然语言已经足够丰富，可以在多次运行之间携带“我从失败中学到了什么”。

### 什么时候有效，什么时候无效

Reflexion 有效的条件：

- 有清晰的失败信号（测试失败、工具错误、答案错误）。
- 任务类别可复现（同一类问题以后还会再出现）。
- 反思有空间改善 trajectory（动作预算足够）。

Reflexion 无效的情况：

- agent 第一次尝试已经成功。
- 失败来自外部因素（网络断开、工具损坏）——反思“网络断了”不会帮助未来运行。
- 反思变成迷信——为一次偶发 flaky run 保存一段叙事。

2026 年的坑：memory rot。反思会累积；有些过时或错误；随着 episodic buffer 变大，重新运行会越来越慢。缓解方法：周期性压缩（Lesson 06）、给反思加 TTL，或用单独的 sleep-time cleanup agent（Letta）。

```figure
react-trace
```

## Build It

`code/main.py` 在一个玩具谜题上实现 Reflexion：生成一个 3 元素列表，使其总和等于目标值。Actor 输出候选列表；Evaluator 检查总和；Self-Reflector 写出一行诊断，说明哪里错了。反思会进入情节记忆，供下一次 trial 使用。

组件：

- `Actor`：一个脚本化 policy，看到 reflections 后会改进。
- `Evaluator.binary()`：基于目标总和的 pass/fail。
- `SelfReflector`：生成一行失败诊断。
- `EpisodicMemory`：带 TTL 语义的有界列表。

运行：

```bash
python3 code/main.py
```

trace 会展示三次 trials。Trial 1 失败并保存一条 reflection；trial 2 看到 reflection 后有所改进但仍然失败；trial 3 成功。和 baseline run（没有 reflection）对比一下——它会一直卡在 trial 1 的答案上。

## Use It

LangGraph 把 reflection 作为一种 node pattern 提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 会把 episodic buffer 外部化为 markdown 文件。Letta 的 sleep-time compute 在空闲时运行 Self-Reflector，让主 agent 保持低延迟。OpenAI Agents SDK 不直接内置 Reflexion；你可以用自定义 Guardrail 按分数拒绝 trajectories，再配合一个跨运行保留的 memory `Session` 来构建它。

## Ship It

`outputs/skill-reflexion-buffer.md` 会创建并维护一个 episodic buffer，包含 reflection capture、TTL 和 deduplication。给定一个任务类别和一次失败，它会输出一条真正能帮助下一次 trial 的反思，而不是泛泛的“下次更仔细”。

## 练习

1. 从 binary evaluator 切换到返回距离指标的 scalar evaluator（距离目标还差多少）。它收敛得更快吗？
2. 给 reflections 添加 10 次 trials 的 TTL。超过这个点后，旧 reflections 是伤害还是帮助？
3. 实现 heuristic evaluator：如果同一个 action 重复出现，就把 trial 标记为卡住。它会如何和 Self-Reflector 相互作用？
4. 用一个无视 reflections 的 adversarial Actor 运行 Reflexion。要让 Actor 注意到 reflections，最小的 reflection prompt engineering 是什么？
5. 阅读 Reflexion 论文第 4 节关于 AlfWorld 的内容。从概念上复现 130% success-rate improvement：它和 vanilla ReAct 的关键差异是什么？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Reflexion | “Self-correction” | Shinn 等 2023 年提出的模式：Actor、Evaluator、Self-Reflector 加 episodic memory |
| Verbal reinforcement | “Learning without gradients” | prepend 到下一次 trial prompt 的自然语言反思 |
| Episodic memory | “Per-task reflections” | 为一个任务类别保存的历史反思有界缓冲区 |
| Scalar evaluator | “Binary success signal” | 来自 ground truth 的 pass/fail 或数值分数 |
| Heuristic evaluator | “Pattern-based detector” | 预定义失败特征（例如 stuck-loop、too-many-steps） |
| Self-evaluator | “LLM-as-judge on own trace” | 没有 ground truth 时的低信号 fallback——应搭配工具锚定验证 |
| Memory rot | “Stale reflections” | episodic buffer 填满过时条目；用 compaction/TTL 修复 |
| Sleep-time reflection | “Async self-reflection” | 在热路径之外运行 Self-Reflector，让主 agent 保持快速 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)：canonical paper。
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)：生产中的 async reflection。
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：把 episodic buffer 作为 context 的一部分来管理。
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：reflection node pattern。
