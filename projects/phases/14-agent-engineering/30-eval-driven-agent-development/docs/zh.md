# Eval-Driven Agent Development

> Anthropic 的建议是：“从简单 prompts 开始，用全面 evaluation 优化它们，只在需要时才加入 multi-step agentic systems。”Evaluation 不是最后一步。它是外层循环，驱动 Phase 14 中的每一个选择。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60 minutes

## 学习目标

- 命名三层 evaluation：static benchmarks、custom offline、online production，并说明每一层的用途。
- 解释 evaluator-optimizer 的紧密循环。
- 描述 2026 年最佳实践：evals 与 code 放在一起，在 CI 中运行，并作为 PR gate。
- 把 Phase 14 的每一课连接到它生成的 eval case。

## 问题

Agents 能通过 demos，却会在 production 中以 demos 无法预测的方式失败。Benchmarks 回答的是“这个模型是否广泛有能力？”，而不是“这个 agent 是否在为我的产品交付正确 patches？”答案是三层 evaluation：持续运行，并把每个 guardrail 和 learned rule 映射到一个 eval case。

## 概念

### 三层 evaluation

1. **Static benchmarks**：代码任务用 SWE-bench Verified（Lesson 19），浏览器 / 桌面任务用 WebArena/OSWorld（Lesson 20），generalist 用 GAIA（Lesson 19），tool use 用 BFCL V4（Lesson 06）。用于跨模型比较和 regression gating。Contamination 是真实问题：SWE-bench+ 发现 32.67% solution leakage。始终报告 Verified / +-audited scores。

2. **Custom offline evals**：你的产品形状：
   - LLM-as-judge（Langfuse、Phoenix、Opik，见 Lesson 24）。
   - Execution-based（运行 patch，检查 tests）。
   - Trajectory-based（把 action sequences 与 gold 对比；OSWorld-Human 显示 top agents 是 gold 的 1.4-2.7x）。

3. **Online evals**：production：
   - Session replays（Langfuse）。
   - Guardrail-triggered alerts（Lesson 16、21）。
   - Per-step cost / latency tracking（Lesson 23 OTel spans）。

### Evaluator-optimizer（Anthropic）

紧密循环：

1. Proposer 生成 output。
2. Evaluator 判断。
3. Refine，直到 evaluator 通过。

这是 Self-Refine（Lesson 05）的泛化。任何你在意的 agent flow，都可以包进 evaluator-optimizer 来提升可靠性。

### 2026 年最佳实践

- Evals 与 code 放在一起。
- 每个 PR 都在 CI 中运行。
- 用 eval scores gate merge（例如“不允许相对 main 回退超过 5%”）。
- 每个 guardrail 都映射到一个 eval case。
- 每个 learned rule（Reflexion、pro-workflow learn-rule）都映射到一个 failure case。

### 把 Phase 14 串起来

Phase 14 的每一课都会生成 eval cases：

| Lesson | Eval case it generates |
|--------|------------------------|
| 01 Agent Loop | Budget-exhausted、infinite-loop guard |
| 02 ReWOO | Tool 失败时 planner 能正确 replan |
| 03 Reflexion | Learned reflections 能应用到 retry |
| 05 Self-Refine/CRITIC | Judge 通过 refined output |
| 06 Tool Use | Argument coercion 有效；unknown tools 被拒绝 |
| 07-10 Memory | Retrieval citations 匹配 sources；stale facts 被 invalidate |
| 12 Workflow Patterns | 每种 pattern 都产生正确 output |
| 13 LangGraph | Resume 精确复现 state |
| 14 AutoGen Actors | DLQ 捕获 crashed handlers |
| 16 OpenAI Agents SDK | Guardrail 对正确 inputs 触发 |
| 17 Claude Agent SDK | Subagent results 返回 orchestrator |
| 19-20 Benchmarks | SWE-bench Verified score、WebArena success rate、OSWorld efficiency |
| 21 Computer Use | Per-step safety 捕获 injected DOM |
| 23 OTel | Spans 发出 required attributes |
| 26 Failure Modes | Detectors 标记 known failures |
| 27 Prompt Injection | PVE 拒绝 poisoned retrievals |
| 28 Orchestration | Supervisor 路由到正确 specialist |
| 29 Runtime Shapes | DLQ 处理 N% failure |

如果你的 eval suite 覆盖了这些 cases，就覆盖了 Phase 14。

### Eval-driven development 会在哪里失败

- **No baseline.** 没有 last-known-good 的 evals 很难读。要存 baselines。
- **LLM-judge without grounding.** Judges 也会幻觉。CRITIC pattern（Lesson 05）适用：judge 要基于外部工具 grounding。
- **Over-fitting to evals.** 为 eval 优化会偏离 production usefulness。要轮换 cases。
- **Flaky evals.** 非确定性 cases 会造成 false alarms。固定 seeds，snapshot state。

## Build It

`code/main.py` 是一个 stdlib eval harness：

- 带 categories（benchmark、custom、online）的 case registry。
- 一个脚本化 agent under test。
- Evaluator-optimizer loop：propose、judge、refine，直到 pass 或达到 max rounds。
- CI gate：聚合 pass rate，并与 baseline 比较 regression。

运行：

```bash
python3 code/main.py
```

输出：per-case pass/fail、regression flag、CI gate verdict。

## Use It

- 在 agent code 同一个 repo 中写 eval cases。
- 在每个 PR 上通过 CI 运行它们。
- 一旦 regression 就 fail build。
- 持续跟踪 pass rate。
- 把每次 production failure 都绑定到一个新 case。

## Ship It

`outputs/skill-eval-suite.md` 会为 agent product 构建一个三层 eval suite，包含 CI gates 和 regression tracking。

## 练习

1. 取一个你的 production failure。写一个能复现它的 eval case。你的 agent 现在能通过吗？
2. 为你的领域构建一个三维 LLM-judge rubric（factual、tone、scope）。给 50 个 sessions 打分。
3. 把 eval suite 接入 CI。当 regression >=5% 时 fail build。
4. 添加 trajectory-efficiency metric：agent 相比 gold trajectory 多走了多少步？
5. 把 Phase 14 的每一课映射到你 suite 中的 eval case。有没有缺口？那就是要补上的 gap。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | "Domain eval" | 基于你的产品形状做 LLM-as-judge / exec / trajectory |
| Online eval | "Production eval" | Session replay、guardrail alerts、cost/latency tracking |
| Evaluator-optimizer | "Propose-judge-refine" | 迭代直到 judge 通过 |
| CI gate | "Merge blocker" | Eval regression 时 fail build |
| Baseline | "Last-known-good" | 用来检测 regression 的参考分数 |
| Trajectory efficiency | "Steps over gold" | Agent step count 除以 human expert minimum |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：start simple, optimize with evals。
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)：curated benchmark。
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)：tool-use benchmark。
- [Langfuse docs](https://langfuse.com/)：实践中的 evals + session replay。
