# 心智理论与涌现式协调

> Li 等（arXiv:2310.10701）表明，合作文本游戏中的 LLM agents 会表现出 **涌现式高阶 Theory of Mind**（ToM）——推理另一个 agent 如何相信第三个 agent 的信念——但会因为 context management 和 hallucination 在长程规划上失败。Riedl（arXiv:2510.05174）在 population scale 上测量 higher-order synergy，并发现 **只有** ToM-prompt condition 会产生 identity-linked differentiation 和 goal-directed complementarity；低能力 LLM 只表现出虚假的 emergence。也就是说，coordination emergence 是 prompt-conditional 且 model-dependent 的，不是免费的。本课会实现一个最小 ToM-aware agent，在有无 ToM prompting 的情况下运行 cooperative task，并按 Riedl 2025 protocol 测量 coordination delta。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## 问题

Multi-agent coordination 经常看起来很神奇：agents 分工、互相预判、避免重复。通常，这种 “emergence” 是 prompt engineering 的产物——有人告诉 agents 要“coordinate”。移除 prompt，coordination 也会消失。

Riedl 2025 的发现更严格：在受控条件下，只有当 agents 被提示去推理 **other agents' minds**（ToM）时，coordination 才会涌现。没有 ToM prompt，即使是强模型也会表现出无法通过统计控制的协调模式。这对生产系统很重要：团队会发布“multi-agent coordination”功能，但这些功能 prompt-dependent 且 brittle。

本课把 ToM 作为一种具体能力（推理 beliefs about beliefs），构建一个最小 ToM-aware agent，并测量真正的 coordination 与 prompt dressing 的区别。

## 概念

### ToM 是什么意思

发展心理学：3 岁儿童认为每个人的内心世界都和自己一样。5 岁儿童理解他人可以有不同信念。7 岁儿童会推理 beliefs about beliefs（“她认为我以为球在杯子下面”）。这些分别是 zeroth、first 和 second-order ToM。

对 LLM agents 来说，ToM orders 映射为：

- **Zeroth-order：** 没有其他人模型。Agent 只基于自己的 observations 行动。
- **First-order：** Agent 拥有每个其他 agent 的 beliefs model。“Alice believes X.”
- **Second-order：** Agent 建模 recursive beliefs。“Alice believes that Bob believes X.”

Li 等 2023 发现，LLM agents 在 cooperative games 中会涌现 first- 和 second-order ToM，但会随着 long horizon 和不可靠 communication 而退化。

### Sally-Anne test 简述

1985 年的 false-belief test：Sally 把一颗弹珠放进篮子 A，然后离开。Anne 把弹珠移到篮子 B。Sally 回来后会去哪里找？具备 first-order ToM 的儿童会说篮子 A（Sally 的信念不同于现实）。不具备 ToM 的儿童会说篮子 B。

GPT-4 时代的 LLMs 在直接提问的 Sally-Anne-style tests 中可以通过。它们会在叙事很长、场景变化多次，或问题被间接表述时失败。这就是 2026 年 production LLMs 中 ToM 的实践状态。

### Riedl 的 coordination measurement

Riedl（arXiv:2510.05174）构建了 population-scale test：N 个 agents、一个 cooperative objective、可变 prompt conditions。测量：

1. **Identity-linked differentiation。** Agents 是否随时间发展出稳定 role distinctions？
2. **Goal-directed complementarity。** Agents 的 actions 是否彼此互补（不同 subtasks），而不是重复？
3. **Higher-order synergy。** 一个统计指标，用来衡量 group 是否达成了任何 subset 都无法达成的结果。

结果：只有在 ToM prompt condition 下，三个 metrics 才都产生高于 baseline 的信号。没有 ToM prompting 时，中等能力模型的 metrics 接近 chance。大模型在没有显式 ToM prompting 时会表现出一些 coordination，但效果小于显式 prompting。

### Coordination illusion

没有统计控制时，demo 中的 “emergent coordination” 往往反映的是：

- Prompt engineering 内置了 coordination（system prompts 中写着 “work together”）。
- Observer bias（我们会看见自己期待的模式）。
- 对成功 runs 的 post-hoc selection。

宣称 “emergent coordination” 但没有 measurable signal 的生产系统，应该被当作营销来看待。Measure before claiming。

### 最小 ToM-aware agent

结构：

```text
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` 属性就是 ToM state。First-order ToM 只保留一层。Second-order 会加入 `other_models[i][other_models_of_j]`——也就是我认为 agent i 认为 agent j 相信什么。

### 为什么 long-horizon 会造成伤害

Li 等记录了：context limits 会让 agents 忘记哪个 belief 属于谁。Hallucination 会把 false beliefs 加入 other-agent models。两者都会产生 “I thought he thought X” errors，并随时间复合。

论文和 2024-2026 年后续工作记录的缓解方法：

- **Prompt 中显式保存 ToM state。** 结构化格式：`{agent_id: belief_list}`。它会迫使 retrieval 保持 identity-belief binding。
- **更短的 reasoning chains。** 每轮更少 ToM updates 可以降低 compounding hallucination。
- **External ToM store。** 在 LLM context 外维护 model；每轮只注入相关部分。

### ToM 在生产中会在哪里失败

- **Adversarial settings。** 具备良好 ToM 的 agents 更容易被操纵（对方可以建模你的模型，再利用它）。
- **Heterogeneous teams。** 模型不同时，适用于一个 opponent 的 ToM model 无法泛化到另一个。
- **Ground-truth-dependent tasks。** ToM 关注 beliefs；如果 correctness 取决于 facts，ToM 可能会分散注意力。

### 你真正能测量的 coordination

团队的 coordination 是真实的，而不是 prompt-dressed 的，通常有三个实用信号：

1. **Complementarity over time。** 在 multi-turn task 中，agents 的 actions 是否覆盖不相交的 subtasks？
2. **Anticipation。** Agent A 在 T+1 的 action 是否依赖于对 B 在 T+2 action 的预测，而该预测后来被证明正确？
3. **Correction。** 当 A 在 T 轮误读 B 的 belief 时，A 是否在 T+2 前修正？

这些都可以在有日志的 multi-agent system 中测量。它们是 “coordination” 叙事的实质版本。

## Build It

`code/main.py` 实现：

- `ToMAgent`：追踪自身 beliefs 和每个 other-agent 的 belief models。
- 一个 cooperative task：三个 agents 必须从三个 boxes 中收集三个 tokens；每个 box 只能容纳一个 token。Agents 不能通信；它们通过彼此 actions 推断意图。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（带一层 belief model 的 ToM）。
- 在 200 个 randomized trials 上测量：completion rate、duplication rate（两个 agents 目标为同一个 box）、average turns to completion。

运行：

```bash
python3 code/main.py
```

Expected output：zeroth-order agents 的重复劳动率约为 35%，10 turns 内完成约 60% trials。First-order ToM agents 的重复率约为 5%，完成率约为 95%。这个 delta 就是 measurable coordination effect。

## Use It

`outputs/skill-tom-auditor.md` 是一个 skill，用来审计某个 multi-agent system 关于 “emergent coordination” 的声明。它会检查 prompt dressing、相对 control 的 statistical significance，以及 measured complementarity。

## Ship It

Coordination claims checklist：

- **Control condition。** 一个没有 coordination prompt 的系统版本。两个版本都要测量。
- **Statistical test。** 在你的 metric 上，system 和 control 的差异是否在 `p < 0.05` 上显著？
- **Complementarity measure。** 测量随时间的 action-disjointness，而不只是最终成功。
- **Failure-case log。** Agents miscoordinate 时，ToM state 看起来是什么样？
- **Model-capacity disclosure。** 如果效果在更小模型上消失，要说出来。

## 练习

1. 运行 `code/main.py`。确认 first-order ToM 会把 duplication rate 降低约 7 倍。扩展到 5 个 agents 和 5 个 boxes 时，这个差距还存在吗？
2. 实现 second-order ToM（agent A 建模 B 如何看待 C）。它相比 first-order 有改善吗？在哪类任务上？
3. 向 ToM state 注入一个 **hallucination**：每轮随机翻转一个 belief。这会让 first-order performance 下降多少？
4. 阅读 Li 等（arXiv:2310.10701）。复现 “long-horizon degradation” 发现：当 turns 从 10 增长到 30 时，你的 first-order ToM performance 如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在你的 simulation logs 上实现 higher-order synergy statistic。没有 ToM prompt condition 时，这个 effect 是否存在？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind | “Understanding others' minds” | 建模另一个 agent beliefs 的能力；按 order 分级（0、1、2+）。 |
| Sally-Anne test | “The false-belief test” | 1985 developmental psychology；LLMs 能通过直接版本，但会在复杂版本中失败。 |
| First-order ToM | “A believes X” | 建模另一个 agent 对 facts 的 beliefs。 |
| Second-order ToM | “A believes B believes X” | 更深一层的 recursive modeling。 |
| Identity-linked differentiation | “Stable roles over time” | Riedl 的 metric：roles 会持续，而不是随机出现。 |
| Goal-directed complementarity | “Disjoint actions” | Agents 瞄准不同 subtasks，而不是同一个。 |
| Higher-order synergy | “Group exceeds any subset” | Riedl 用来衡量真实 coordination 的统计指标。 |
| Coordination illusion | “It looks coordinated” | 没有 measurable signal 的 prompt-dressed coordination appearance。 |

## 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701)：cooperative games 中的 emergent ToM；long-horizon failure modes。
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174)：population-scale measurement；ToM prompting 是 load-bearing condition。
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595)：ToM 概念的 1978 年起源。
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/)：Sally-Anne paper（1985）。
