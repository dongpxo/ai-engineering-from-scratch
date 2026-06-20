# Self-Refine 与 CRITIC：迭代式输出改进

> Self-Refine（Madaan et al., 2023）让一个 LLM 在循环中扮演三个角色：generate、feedback、refine。7 个任务上的平均收益：绝对提升 +20。CRITIC（Gou et al., 2023）通过把 verification 路由到外部工具，让 feedback 步骤更可靠。到 2026 年，这个模式已经以 “evaluator-optimizer”（Anthropic）或 guardrail loop（OpenAI Agents SDK）的形式进入每个框架。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## 学习目标

- 说出 Self-Refine 的三个 prompts（generate、feedback、refine），并解释为什么 history 对 refine prompt 很重要。
- 解释 CRITIC 的关键洞察：如果没有外部 grounding，LLM 不擅长自我验证。
- 用 stdlib 实现一个带 history 和可选外部 verifier 的 Self-Refine loop。
- 将这个模式映射到 Anthropic 的 “evaluator-optimizer” workflow，以及 OpenAI Agents SDK 的 output guardrails。

## 问题

Agent 生成了一个几乎正确的答案。也许一行代码有语法错误。也许摘要太长。也许计划漏掉了一个边界情况。你想要的是：agent 批评自己的输出，然后修正它。

Self-Refine 证明了这件事可以只靠一个模型完成，不需要训练数据，也不需要 RL。但这里有个问题：LLM 在硬事实上的自我验证能力很差。CRITIC 给这个问题命名并给出修复方式：把 verify 步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文合在一起，定义了 2026 年迭代改进的默认形状：generate、verify（可能的话用外部工具）、refine，并在 verifier 通过时停止。

## 概念

### Self-Refine（Madaan et al., NeurIPS 2023）

一个 LLM，三个角色：

```text
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 会看到完整 history，也就是所有先前 outputs 和 critiques，所以它不会重复犯错。论文做过 ablation：去掉 history，质量会明显下降。

核心结果：在 7 个任务（math、code、acronym、dialog）上平均绝对提升 +20，包括 GPT-4。没有训练、没有外部工具、只用单个模型。

### CRITIC（Gou et al., arXiv:2305.11738, v4 Feb 2024）

Self-Refine 的弱点是：feedback 步骤本质上是一个 LLM 在给自己打分。对于事实性声明，这并不可靠（一个幻觉往往对产生它的模型来说也很有说服力）。CRITIC 用 `verify(task, output, tools)` 替换 `feedback(task, output)`，其中 `tools` 包括：

- 用于事实性声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定 verifier（单元测试、类型检查器、linters）。

Verifier 会产出一份基于工具结果的结构化 critique。Refiner 再基于这份 critique 进行修订。

核心结果：CRITIC 在事实性任务上超过 Self-Refine，因为 critique 有 grounding。在没有外部 verifier 的任务上（创意写作、格式化），CRITIC 会退化成 Self-Refine。

### 停止条件

常见形状有两种：

1. **Verifier passes.** 外部测试返回成功。有可用 verifier 时优先使用（单元测试、类型检查器、guardrail assertion）。
2. **No feedback issued.** 模型说 “the output is fine.” 成本更低，但不可靠；需要配合 max-iteration cap。

2026 年默认做法是把它们组合起来：“如果 verifier 通过，或模型说 fine 且 iterations >= 2，或 iterations >= max_iterations，就停止。”

### Evaluator-Optimizer（Anthropic, 2024）

Anthropic 在 2024 年 12 月的文章中，把它命名为五个 workflow patterns 之一。两个角色：

- Evaluator：为输出打分，并产出 critique。
- Optimizer：根据 critique 修订输出。

循环直到 evaluator 通过。这就是 Anthropic 语境里的 Self-Refine/CRITIC。Anthropic 补充的关键工程细节是：evaluator 和 optimizer prompts 应该明显不同，避免模型只是 rubber-stamp。

### OpenAI Agents SDK output guardrails

OpenAI Agents SDK 把这个模式做成了 “output guardrails”。Guardrail 是一个 validator，会在 agent 的最终输出上运行。如果 guardrail trip（抛出 `OutputGuardrailTripwireTriggered`），输出会被拒绝，agent 可以重试。Guardrails 可以调用工具（CRITIC-style），也可以是纯函数（Self-Refine-style）。

### 2026 年的坑

- **Rubber-stamp loops.** 同一个模型用同一种 prompt style 负责 generation 和 critique，最后会收敛到 “looks good to me.” 用结构明显不同的 prompts，或用更小、更便宜的模型来做 critique。
- **Over-refinement.** 每一次 refine pass 都会增加延迟和 tokens。预算 1-3 轮；再往后就升级到人工 review。
- **CRITIC on trivial tasks.** 如果没有外部 verifier，CRITIC 会退化成 Self-Refine；不要为了一个 stub verifier 支付额外延迟。

## Build It

`code/main.py` 在一个玩具任务上实现 Self-Refine 和 CRITIC：给定 topic，产出一个简短 bullet list。Verifier 检查格式（3 个 bullets，每条少于 60 个字符）。CRITIC 额外加入一个外部 “fact verifier”，会惩罚已知幻觉。

组件：

- `generate`：scripted producer。
- `feedback`：LLM-style self-critique。
- `verify_external`：CRITIC-style grounded verifier。
- `refine`：根据 history 重写输出。
- Stop condition：verifier passes 或最多 4 次迭代。

运行：

```bash
python3 code/main.py
```

对比 Self-Refine 和 CRITIC 的运行。CRITIC 会抓住一个 Self-Refine 漏掉的事实错误，因为外部 verifier 拥有 self-critic 没有的 grounding。

## Use It

Anthropic 的 evaluator-optimizer 是 Claude-friendly 语言里的这个模式。OpenAI Agents SDK 的 output guardrails 是 CRITIC-shaped（guardrails 可以调用工具）。LangGraph 提供的 reflection node 读起来像 Self-Refine。Google Gemini 2.5 Computer Use 加入的 per-step safety evaluator 是一种 CRITIC 变体：每个 action 在 commit 前都会被验证。

## Ship It

`outputs/skill-refine-loop.md` 会根据 task shape、verifier availability 和 iteration budget 配置一个 evaluator-optimizer loop。它会输出 generator、evaluator/verifier 和 optimizer prompts，以及 stop policy。

## 练习

1. 用 `max_iterations=1` 运行这个玩具 demo。CRITIC 还会有帮助吗？
2. 把外部 verifier 替换成一个有噪声的 verifier（随机 30% false positives）。Loop 会怎么做？这就是 2026 年大多数 guardrail stacks 的现实。
3. 实现一个 “generator-critic on different models” 变体：大模型负责生成，小模型负责 critique。它能超过 same-model 吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类 verification tools，并为每一类给一个例子。
5. 把 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 verifier role。SDK 哪些地方做错了，哪些地方做对了？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Self-Refine | “LLM that fixes itself” | 一个模型中的 generate -> feedback -> refine loop，并带 history |
| CRITIC | “Tool-grounded verification” | 用外部 verifier（搜索、代码、计算器、测试）替换 feedback |
| Evaluator-Optimizer | “Anthropic workflow pattern” | 两个角色：evaluator 打分，optimizer 修订；循环直到收敛 |
| Output guardrail | “Post-hoc check” | OpenAI Agents SDK validator，在 agent 产出输出后运行 |
| Verify step | “Critique phase” | 承重决策点：是 grounded，还是 self-rated |
| Refine history | “What the model already tried” | 先前 outputs + critiques，会 prepend 到 refine prompt；去掉后质量会崩 |
| Rubber-stamp loop | “Self-agreement failure” | 同 prompt critique 返回 “looks good”；用结构不同的 prompts 修复 |
| Stop condition | “Convergence test” | Verifier passes 或 no feedback 且达到 iteration cap；不要只用单一条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — canonical paper
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — tool-grounded verification
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer workflow pattern
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — output guardrails as CRITIC-shaped verifiers
