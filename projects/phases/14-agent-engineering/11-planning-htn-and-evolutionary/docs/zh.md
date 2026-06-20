# 使用 HTN 与演化搜索做规划

> 符号规划处理那些计划必须可证明正确的场景。演化代码搜索处理那些 fitness function 可以被机器检查的场景。ChatHTN（2025）和 AlphaEvolve（2025）展示了它们和 LLM 配合后分别能解锁什么。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 minutes

## 学习目标

- 解释 Hierarchical Task Networks：tasks、methods、operators、preconditions、effects。
- 描述 ChatHTN 的 hybrid loop：symbolic search 加上 LLM fallback decomposition。
- 解释 AlphaEvolve 的 evolutionary loop，以及为什么它只在有 programmatic evaluator 时才有效。
- 用 stdlib 实现一个 toy HTN planner 和一个 toy evolutionary search。

## 问题

ReWOO（Lesson 02）、Plan-and-Execute 和 ReAct 覆盖了大多数 agent planning。但有两类场景它们处理得不好：

1. **需要可证明正确性的计划。** 调度、飞行路径、合规 workflow——计划必须 by construction 就是 sound 的。一个有时会幻觉步骤的流畅 LLM plan 不能接受。
2. **带机器可检查 fitness function 的优化。** 矩阵乘法、调度启发式、编译器 passes——目标不是 “一个正确计划”，而是 “最好的计划”。

HTN planning 和 AlphaEvolve 解决的是两个不同问题。两者都把 LLM 当作放大器，而不是替代品。

## 概念

### Hierarchical Task Networks

一个 HTN 包含：

- **Tasks**：compound（需要被分解）和 primitive（可以直接执行）。
- **Methods**：把 compound task 分解成 subtasks 的方式，带 preconditions。
- **Operators**：带 preconditions 和 effects 的 primitive actions。
- **State**：一组 facts。

Planning：给定一个 goal task 和 initial state，找到一条分解路径，把 goal 拆成一串 primitive operators，并且这些 operators 的 preconditions 会按顺序得到满足。

HTN 比 LLM 早得多，并且至今仍是可证明正确计划的参考方法。

### ChatHTN（Gopalakrishnan 等人，2025）

ChatHTN（arXiv:2505.11814）把符号 HTN 与 LLM 查询交错起来：

1. 尝试用已有 methods 分解当前 compound task。
2. 如果没有 method 适用，就问 LLM：“在 state `s` 下，你会如何分解 `task`？”
3. 把 LLM 回复翻译成 candidate subtasks。
4. 根据 operator schema 做校验；拒绝无效分解。
5. 递归。

论文的核心主张是：生成的每个 plan 都可证明 sound，因为 LLM 建议只作为 candidate decompositions 进入系统，永远不能直接编辑 plan。符号层负责 correctness；LLM 扩展 method library。

Online method learning（OpenReview `gwYEDY9j2x`，2025 后续工作）加入了一个 learner，通过 regression 泛化 LLM 产生的 decompositions，可以把 LLM query 频率最多降低 75%。

### AlphaEvolve（Novikov 等人，2025）

AlphaEvolve（arXiv:2506.13131，DeepMind，2025 年 6 月）是另一种东西：由 Gemini 2.0 Flash/Pro ensemble 编排的 evolutionary code search。

循环如下：

1. 从一个 seed program + programmatic evaluator 开始；evaluator 会返回 fitness score。
2. LLM ensemble 提出 mutations。
3. 把 mutations 放进 evaluator 运行。
4. 保留最好的，继续 mutation。

已发表的成果包括：

- 56 年来首次改进 4x4 complex matrix multiplication 上 Strassen 之后的结果（48 次 scalar multiplications）。
- 通过 Borg scheduling heuristic 回收 0.7% Google compute。
- 在 frontier workload 上带来 32% FlashAttention speedup。

硬约束是：fitness function 必须 machine-checkable。对散文答案做 evolutionary search 不会收敛。

### 什么时候用哪个

| Problem class | Use | Why |
|---------------|-----|-----|
| 带硬约束的调度 | HTN + ChatHTN | 可证明 soundness |
| 编译器优化 | AlphaEvolve | Machine-checkable fitness |
| 多步任务执行 | ReAct / ReWOO | LLM in the loop，没有形式化保证 |
| 带测试的代码改进 | AlphaEvolve | Tests 就是 evaluator |
| 受政策约束的自动化 | HTN | Preconditions 编码 policy |

### 这个模式容易在哪里出错

- **HTN without operators.** 没有 precondition/effect schemas，soundness claim 就会崩掉。ChatHTN 的 “LLM suggests decomposition” 依赖 schema 拒绝无效动作。
- **AlphaEvolve without a real evaluator.** “问 LLM 代码是否更好” 不是 fitness function。Evaluator 必须确定性且快速。
- **Over-engineering.** 大多数 agent tasks 两者都不需要。优先使用 ReAct 或 ReWOO。

## Build It

`code/main.py` 实现了两个玩具版本：

- 一个 stdlib HTN planner，包含 operators、methods、preconditions、effects，以及在没有 method 匹配 compound task 时触发的 `LLMFallback`。“LLM” 是脚本化 decomposer，所以 planner 可以离线运行。
- 一个 stdlib evolutionary search，用于搜索 arithmetic programs：生成 expressions，使其在 test set 上最小化 `|f(x) - target|`。Evaluator 是确定性的。

运行它：

```bash
python3 code/main.py
```

Trace 会展示 HTN planner 如何分解一个 compound task（中途有一次 LLM fallback），以及 evolutionary loop 如何收敛到 target expression。

## Use It

- **HTN planners**：`pyhop`、`SHOP3`，或者为 domain-specific policy enforcement 自建 planner。
- **ChatHTN**：研究代码；这个模式（symbolic + LLM fallback）可以干净地移植到任意 HTN planner。
- **AlphaEvolve**：DeepMind paper；这个模式（ensemble + evaluator）可以复现。OpenEvolve 和类似开源 forks 正在出现。
- **Agent frameworks**：目前还没有框架内置 first-class HTN 或 AlphaEvolve。把它做成 subagent 或 background worker。

## Ship It

`outputs/skill-hybrid-planner.md` 会生成一个 hybrid planner scaffold（HTN 或 evolutionary），并明确限定 LLM 的角色。

## 练习

1. 给 HTN planner 增加 backtracking：当某个 operator 的 postcondition 在运行时失败，回滚并尝试下一个 method。
2. 给 ChatHTN 添加 LLM-method cache：当 LLM 在 state pattern `P` 下分解 task `T` 时，存储结果。下次调用先重新检查 method library。
3. 把 evolutionary search evaluator 换成真实 test suite。演化一个能通过 20 个 test cases 的 sort function；报告收敛需要多少 generations。
4. 阅读 AlphaEvolve 的 evaluator design notes。为你关心的 domain 设计一个 evaluator（SQL query optimization、test-suite minimization、deployment YAML）。
5. 组合两者：用 HTN 把 compound task 分解成 subtasks，然后在每个 subtask 的 primitive operator 上使用 evolutionary search。它在哪些地方有优势，哪些地方属于 over-engineering？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| HTN | “Hierarchical planner” | 带 operators、preconditions、effects 的 task decomposition |
| Method | “Decomposition rule” | 把 compound task 拆成 subtasks 的方式 |
| Operator | “Primitive action” | 带 precondition 和 effect 的具体步骤 |
| ChatHTN | “LLM + HTN” | 当没有 method 匹配时，symbolic planner 会询问 LLM |
| AlphaEvolve | “Evolutionary code search” | LLM ensemble mutation code；deterministic evaluator 负责选择 |
| Fitness function | “Evaluator” | 对 outputs 打分的确定性、machine-checkable score |
| Online method learning | “Cached LLM decomposition” | 存储并泛化 LLM plans，用来降低 query cost |

## 延伸阅读

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814)：symbolic + LLM hybrid planner。
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)：带 LLM mutations 的 evolutionary code search。
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：什么时候使用 planner，什么时候使用 simple loop。
