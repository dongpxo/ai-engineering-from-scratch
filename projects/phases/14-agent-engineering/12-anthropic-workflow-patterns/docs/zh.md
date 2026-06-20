# Anthropic 的 Workflow Patterns：简单优先于复杂

> Schluntz 和 Zhang（Anthropic, Dec 2024）区分了 workflows（预定义路径）和 agents（dynamic tool-use）。五种 workflow patterns 覆盖了大多数场景。先从直接 API calls 开始。只有当步骤无法预测时，才加入 agents。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## 学习目标

- 说出 Anthropic 的五种 workflow patterns：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解释 agent-vs-workflow 的区别，以及每一种方案的工程成本。
- 判断什么时候应该选择 workflow 而不是 agent，反过来也一样。
- 用 stdlib 和一个 scripted LLM 实现全部五种 patterns。

## 问题

很多团队会为本该只用一次 function call 解决的问题引入 multi-agent frameworks。成本是真实存在的：框架会增加层级，遮蔽 prompts，隐藏 control flow，并诱发过早复杂化。Schluntz 和 Zhang 在 2024 年 12 月的文章，是 industry 中最常被引用的反向提醒：从简单开始，只有当复杂性值得付费时才添加它。

## 概念

### Workflows vs agents

- **Workflow.** LLMs 和 tools 通过预定义 code paths 被编排。Engineers 拥有 graph。
- **Agent.** LLMs 动态指挥自己的 tools，并自己决定步骤。Model 拥有 graph。

两者都有位置。Workflows 更便宜、更快，也更容易 debug。Agents 能处理开放式问题，但 failure modes 更难推理。

### The augmented LLM

五种 patterns 的基础都是 augmented LLM：一个 LLM 接上三种能力——search（retrieval）、tools（actions）、memory（persistence）。任何 API call 都可以使用这些能力。

### 五种 patterns

1. **Prompt chaining.** 第 1 次调用的输出是第 2 次调用的输入。适合能清晰线性拆解的任务。步骤之间可以放可选的 programmatic gates。

2. **Routing.** 一个 classifier LLM 选择要调用哪个下游 LLM 或 tool。适合不同类别输入需要不同处理的场景（tier-1 support vs refund vs bug vs sales）。

3. **Parallelization.** 并发运行 N 次 LLM calls，然后聚合结果。两种形状：sectioning（不同 chunks）和 voting（同一个 prompt 跑 N 次，再 majority/synthesis）。

4. **Orchestrator-workers.** 一个 orchestrator LLM 动态决定要运行哪些 workers（也都是 LLMs），并综合它们的输出。它类似 agent loops，但 orchestrator 不会无限循环。

5. **Evaluator-optimizer.** 一个 LLM 提出答案，另一个 LLM 评估它。持续迭代直到 evaluator 通过。这是 Self-Refine（Lesson 05）的泛化版本。

### Workflows 胜过 agents 的地方

- **Predictable tasks.** 如果你能枚举步骤，就应该枚举。
- **Cost-bound tasks.** Workflows 有有界 step counts；agents 可能失控膨胀。
- **Compliance-bound tasks.** Auditors 想读 graph，而不是从 trajectories 里推断 graph。

### Agents 胜过 workflows 的地方

- **Open-ended research.** 下一步取决于上一步返回了什么。
- **Variable-length tasks.** 需要工作几分钟到几小时，而且 step count 未知。
- **Novel domains.** 你还不知道正确 workflow 是什么；先 exploration，之后再 codify。

### Context-engineering companion

“Effective context engineering for AI agents”（Anthropic 2025）形式化了相邻学科：200k window 是预算，不是容器。该包含什么、什么时候 compact、什么时候让 context 增长。这个主题会在 Phase 14 的 context compression lesson 中详细讲解（本课程 renumber 前是 Phase 14 较早的 lesson 06）。

## Build It

`code/main.py` 针对一个 `ScriptedLLM` 实现全部五种 workflow patterns：

- `prompt_chain(input, steps)`：顺序执行。
- `route(input, classifier, handlers)`：classification + dispatch。
- `parallel_vote(prompt, n, aggregator)`：N 次运行后聚合。
- `orchestrator_workers(task, workers)`：orchestrator 选择 workers。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)`：循环直到 pass。

运行：

```bash
python3 code/main.py
```

每个 pattern 都会打印自己的 trace。每种 pattern 的代码量约 10-15 行；框架的成本通常以 thousands of lines 计。

## Use It

- 大多数任务使用 direct API calls。
- 只有当 pattern 真的需要 durable state（LangGraph）、actor-model concurrency（AutoGen v0.4）或 role templating（CrewAI）时，才使用框架。
- 当你想要 Claude Code harness shape，但不想重新构建它时，再选择 Claude Agent SDK。

## Ship It

`outputs/skill-workflow-picker.md` 会根据 task description 选择正确 pattern，包含 decision rationale，以及当 workflows 不够用时 refactor 成 agent 的路径。

## 练习

1. 给 routing 实现 confidence threshold。低于 threshold -> escalate to human。对于 tier-1 support use case，这个 threshold 应该落在哪里？
2. 给 `parallel_vote` 添加 timeout。某一次 call hang 住会发生什么？缺失 votes 时如何 aggregate？
3. 把 `evaluator_optimizer` 变成一个 bandit：跨 iterations 保留 top-2 outputs，避免后期的好结果被后期的坏结果覆盖。
4. 组合 prompt chaining 和 routing：router 选择三条 chains 之一。测量 token cost，并和 single big-prompt alternative 对比。
5. 选择你的一个 production feature。画出 workflow graph。数一下 steps。这里真的用 agent 会更好吗？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Workflow | “Predefined flow” | 由 engineer 拥有的 LLM 和 tool calls graph |
| Agent | “Autonomous AI” | 由 model 拥有的 graph；动态指挥 tools |
| Augmented LLM | “LLM with tools” | LLM + search + tools + memory；原子单元 |
| Prompt chaining | “Sequential calls” | 第 N 次调用的输出是第 N+1 次调用的输入 |
| Routing | “Classifier dispatch” | 选择哪条 chain 或哪个 model 处理输入 |
| Parallelization | “Fan out” | N 个并发 calls；通过 sectioning 或 voting 聚合 |
| Orchestrator-workers | “Dispatcher agent” | Orchestrator LLM 动态选择 specialist LLMs |
| Evaluator-optimizer | “Proposer + judge” | 迭代直到 evaluator 通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — the five workflow patterns
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — the companion discipline
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — when stateful graphs earn their cost
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — the orchestrator-workers pattern, productized
