# ReWOO 与 Plan-and-Execute：解耦式规划

> ReAct 把思考和行动交织在同一条流里。ReWOO 把它们拆开：先一次性规划，再执行。Token 少 5 倍，HotpotQA 准确率提升 4%，而且你可以把 planner 蒸馏到 7B 模型。Plan-and-Execute 把这个模式泛化；Plan-and-Act 又把它扩展到网页导航。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分，比 ReAct 的交织循环更省 token，也更稳健。
- 实现一个 plan DAG、一个按依赖顺序执行的 executor，以及一个组合 worker 输出的 solver；全部使用 stdlib。
- 借助 Anthropic 2026 年的 “five workflow patterns” 框架，判断任务应该使用 plan-then-execute 还是交织式 ReAct。
- 识别什么时候需要 Plan-and-Act 的 synthetic plan data，尤其是长程网页或移动端任务。

## 问题

ReAct 的 thought-action-observation 交织循环简单而灵活，但每一次工具调用都必须带上完整的前文上下文，包括之前的所有 thought。Token 使用量会随着深度近似二次增长。更糟的是：当某个工具在循环中途失败时，模型必须从 error observation 里重新推导整套计划。

ReWOO（Xu et al., arXiv:2305.18323, May 2023）注意到了这一点，并押注于一个结构：先一次性规划，之后并行获取 evidence，最后组合答案。一次 LLM 调用负责规划，N 次 tool calls 负责 evidence（可并行），再一次 LLM 调用负责求解。它用更少的灵活性（计划是静态的）换来更好的 token efficiency 和更清晰的 failure modes。

## 概念

### 三个角色

- Planner：把 `user_question` 转成 `plan_dag`。
- Workers：执行 `plan_dag` 中的工具调用，产出 `evidence`，可并行。
- Solver：读取 `user_question`、`plan_dag` 和 `evidence`，组合出 `final_answer`。

Planner 产出一个 DAG。每个节点都会命名 tool、arguments，以及它依赖哪些更早的节点（例如 `#E1`、`#E2` 这样的引用）。Workers 按拓扑顺序执行节点。Solver 把所有东西拼接起来。

### 为什么 token 少 5 倍

ReAct 的 prompt 长度会随着 step count 线性增长。到了第 10 步，prompt 包含 thought 1、action 1、observation 1、thought 2、action 2、observation 2，以此类推。每个中间步骤还会重复带上原始 prompt。

ReWOO 只支付一次 planner prompt（较大）、N 个小 worker prompts（每个只是 tool call，没有 chain），以及一次 solver prompt。在 HotpotQA 上，论文测得 token 约少 5 倍，同时 absolute accuracy 提升 4 个百分点。

### 为什么它更稳健

如果 worker 3 在 ReAct 中失败，循环必须在中途围绕错误继续推理。在 ReWOO 中，worker 3 返回一个 error string；solver 会在原始计划的上下文里看到它，并能优雅降级。失败定位是 per-node 的，而不是 per-step 的。

### Planner 蒸馏

论文的第二个结果是：因为 planner 不看 observations，所以你可以用 175B teacher 产出的 planner outputs 来 fine-tune 一个 7B 模型。小模型负责 planning；大模型不需要参与 inference。这现在已经是标准做法：很多 2026 年的 production agents 会使用小 planner + 大 executor，或反过来。

### Plan-and-Execute（LangChain, 2023）

LangChain 团队在 2023 年 8 月的文章里，把 ReWOO 泛化成一个模式名称：Plan-and-Execute。前置 planner 产出 step list，executor 执行每一步，可选 replanner 在观察结果后修订计划。它比 ReWOO 更接近 ReAct（replanner 会把 observations 带回 planning），但仍保留 token savings。

### Plan-and-Act（Erdogan et al., arXiv:2503.09572, ICML 2025）

Plan-and-Act 把这个模式扩展到长程网页和移动端 agents。关键贡献是 synthetic plan data：一个带标签的 trajectory generator 会生成显式包含 plan 的训练数据。它用于 fine-tune planner models，让它们在 WebArena 这类任务里超过 30-50 步后仍能保持工作，而单条 ReAct trajectory 往往会失去连贯性。

### 什么时候选择哪一种

| Pattern | When |
|---------|------|
| ReAct | 短任务、未知环境、需要 reactive exception handling |
| ReWOO | 结构化任务、tools 已知、token 敏感、evidence 可并行 |
| Plan-and-Execute | 类似 ReWOO，但会在部分执行后 replanning |
| Plan-and-Act | 长程任务（>30 步）、网页 / 移动端 / computer-use |
| Tree of Thoughts | Search 值得付出成本（Lesson 04） |

Anthropic 2024 年 12 月的建议是：从最简单的开始。如果任务只是一轮 tool call 加一个 summary，就不要构建 ReWOO。如果任务是一个 40 步 research assignment，就不要只用 ReAct。

## Build It

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner`：一个 scripted policy，会从 prompt 产出 plan DAG。
- `Worker`：通过 registry 分发每个节点的 tool call。
- `Solver`：一个 scripted composition，读取 evidence 并产出最终答案。
- Dependency resolution：像 `#E1` 这样的引用会被替换成更早的 worker outputs。

这个 demo 会回答 “What is the population of the capital of France, rounded to millions?”，使用一个两步计划：（1）查首都，（2）查人口，然后求解。

运行：

```bash
python3 code/main.py
```

Trace 会先展示完整 plan，然后是 worker results，最后是 solver composition。把 token count（我们打印的是粗略 character count）和 ReAct-style interleaved run 对比一下；在这种结构化任务上，ReWOO 会赢。

## Use It

LangGraph 以 recipe 形式提供 Plan-and-Execute（`create_react_agent` 用于 ReAct，custom graphs 用于 plan-execute）。CrewAI 的 Flows 会直接编码这个模式：你先定义 tasks，然后 Flow DAG 执行它们。Plan-and-Act 的 synthetic data 方法目前仍然主要是研究；但 runtime pattern（显式 plan DAG）已经通过 LangGraph 和 CrewAI Flows 进入 production。

## Ship It

`outputs/skill-rewoo-planner.md` 会在给定 tool catalog 的情况下，从用户请求生成一个 ReWOO plan DAG。它会先验证 plan（无环、每个 reference 都能解析、每个 tool 都存在），再交给 executor。

## 练习

1. 为彼此独立的 plan nodes 并行化 worker execution。对于一个有 2 个 parallel groups 的 6-node DAG，它能带来什么？
2. 添加一个 replanner node：当任意 worker 返回 error 时触发。把 ReWOO 变成 Plan-and-Execute 所需的最小改动是什么？
3. 用一个小模型（7B class）替换 `Planner`，并让 `Solver` 继续使用 frontier model。比较 end-to-end quality：这种拆分在哪里会失败？
4. 阅读 ReWOO 论文第 4 节关于 planner distillation 的内容。从概念上复现 175B -> 7B 结果：你需要什么训练数据，又如何评分 plan quality？
5. 把这个玩具 demo 移植到 Plan-and-Act 的 trajectory shape：plan 是 sequence，而不是 DAG。哪些 tradeoffs 会变化？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| ReWOO | “Reasoning without observations” | 先规划，再并行获取 evidence，最后求解；planning prompt 中没有 observations |
| Plan-and-Execute | “LangChain's plan-execute pattern” | 带可选 replanner node 的 ReWOO，replanner 位于 execution 之后 |
| Plan-and-Act | “Scaled plan-execute” | 显式 planner/executor 拆分，并用 synthetic plan training data 支持长程任务 |
| Evidence reference | “#E1, #E2, ...” | Plan-node placeholder，会在 dispatch 时替换成之前的 worker output |
| Planner distillation | “Small planner, big executor” | 用大型 teacher 的 planner traces 来 fine-tune 小模型 |
| Token efficiency | “Fewer round trips” | 论文中相较 ReAct 在 HotpotQA 上少 5 倍 token |
| DAG executor | “Topological dispatcher” | 按依赖顺序运行 plan nodes；每一层可并行 |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — canonical paper
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — scaled planner-executor with synthetic plans
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — framework recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择能工作的最简单 pattern
