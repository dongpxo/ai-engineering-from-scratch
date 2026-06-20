# 角色专门化：Planner、Critic、Executor、Verifier

> 2026 年最常见的 multi-agent decomposition：一个 agent 规划，一个执行，一个 critique 或 verify。MetaGPT（arXiv:2308.00352）把这件事形式化为编码进 role prompts 的 SOPs——Product Manager、Architect、Project Manager、Engineer、QA Engineer——遵循 `Code = SOP(Team)`。ChatDev（arXiv:2307.07924）通过 “chat chain” 把 designer、programmer、reviewer、tester 串起来，并加入 “communicative dehallucination”（agents 显式请求缺失细节）。verifier 是承重角色：Cemri et al.（MAST，arXiv:2503.13657）显示，每个 multi-agent failure 都能追溯到缺失或损坏的 verification。PwC 报告称，CrewAI 中的 structured validation loops 带来 7 倍 accuracy gain（10% → 70%）。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 minutes

## 问题

Generic multi-agent systems 会产出 generic output。三个 coders 在 group chat 里写出三种味道相近的平庸代码。你可以增加更多 agents、更多 rounds，仍然跨不过质量阈值。

修复方式不是更多 agents，而是 *不同* agents。分配不同 roles。给 critic 一些 planner 没有的 tools。给 verifier 一个客观 test suite。这样系统拥有的是带 grounded correction 的内部分歧，而不只是并行猜测。

## 概念

### 四个 canonical roles

**Planner.** 读取 goal，产出 step list 或 spec。Tools：knowledge retrieval、docs。Output：structured plan。

**Executor.** 一次读取一个 plan step，产出 artifact。Tools：实际工作 tools（code compiler、shell、API client）。Output：artifact。

**Critic.** 对照 planner 的 intent 读取 executor 的 output。Tools：对 artifact 的 read-only access、static analysis。Output：accept/reject with reasons。

**Verifier.** 读取 artifact 并运行 deterministic check。Tools：test runner、type checker、schema validator。Output：pass/fail with evidence。

Critic 是主观的、有 opinion 的，通常基于 LLM。Verifier 是客观的、deterministic 的，通常基于代码。它们不是同一个角色。

### MetaGPT 的 SOP pattern

MetaGPT（arXiv:2308.00352）把 software engineering SOPs 编码成 role prompts：

- **Product Manager** 写 PRD。
- **Architect** 产出 system design。
- **Project Manager** 拆分 tasks。
- **Engineer** 实现。
- **QA Engineer** 运行 tests。

每个 role 都有严格的 input/output schema。role prompt 说明这个 role *是什么*，以及它 *必须产出什么*。`Code = SOP(Team)` 这个 formulation 的意思是：deterministic SOPs 把一队 LLMs 变成可预测 pipeline。

### ChatDev 的 communicative dehallucination

ChatDev 增加了一个关键动作：当 executor 需要某个 plan 中没有给出的具体细节时，它会在继续之前显式询问 designer。这可以防止经典 LLM failure：看似合理地编造缺失细节。

实现方式：role prompt 中包含 “when you need specific information you were not given, ask the relevant role by name before producing output.”

### 为什么 verifier 最重要

Cemri et al.（MAST）追踪了 1642 个 multi-agent execution failures。其中 21.3% 是 verification gaps——系统发出了一个没人检查过的答案。剩下 79% 经常也能追溯到“有一个 check 静默失败，或根本没有运行”。Verification 是承重角色。

PwC 报告称（CrewAI deployments，2025），加入 structured validation loop 后 accuracy 从 10% 到 70%。一个 role 带来 7 倍收益。

### Critic vs verifier

- Critic 是一个 LLM，审查 artifact 的质量。主观。会被 plausible prose 欺骗。
- Verifier 是一个在 artifact 上运行的 deterministic program。客观。给出带 evidence 的 pass/fail。

两者都要用。Critic 能抓住 verifier 无法表达的 taste issues。Verifier 能抓住 critic 看不见的 bugs，因为它们只在 runtime 出现。

### Anti-pattern

你系统里的每个 role 都是 LLM，每个 role 的输出都是 “looks good to me”。这是经典 MAST failure mode。至少加入一个 verifier，让 pass/fail 由代码决定，而不是由 LLM 决定。

### Framework mappings

- **CrewAI**：`Agent(role, goal, backstory)` 是 textbook specialization surface。
- **LangGraph**：nodes 可以拥有 specialized prompts；edges 强制 pipeline。
- **AutoGen**：GroupChat 中使用带 one-word names 的 role-specific ConversableAgents。
- **OpenAI Agents SDK**：role-specialized Agents 之间通过 handoff tools 转交。

## Build It

`code/main.py` 实现了一个构建简单 Python function 的 4-role pipeline：

- **Planner** 产出 spec。
- **Executor** 生成 code string。
- **Critic**（LLM-simulated）标记明显问题。
- **Verifier** 在 sandbox（`exec`）中运行生成的 code，并对照 test case 检查。

Demo 会运行两次：一次 executor 产出正确代码（critic + verifier 都 pass），一次 executor 产出 off-spec code（critic 因为它看起来 plausible 而错过 bug，verifier 因 test failed 抓住它）。

运行：

```bash
python3 code/main.py
```

## Use It

`outputs/skill-role-designer.md` 接收一个 task，并产出 role roster（3-5 个 roles）、每个 role 的 input/output schema，以及 verifier check。在把 agents 接进框架之前先使用它。

## Ship It

Checklist：

- **至少一个 deterministic verifier。** 永远不要 all-LLM。
- **每个 role 都有 explicit I/O schema。** Planner 返回 spec，不是 prose；executor 读取那个 schema。
- **Communicative dehallucination。** Executor 缺少信息时必须问 planner；永远不要编造。
- **Critic/verifier ordering。** Critic 先跑（便宜，抓 design issues），verifier 后跑（慢，抓 bugs）。
- **Loop budget。** 最多 2 轮 critic-executor revision，再 escalation to human。

## 练习

1. 运行 `code/main.py`，观察 verifier 如何抓住 critic 错过的 bug。增加一个 static-analysis check（统计 `return` 出现次数）作为额外 verifier。它能抓住 runtime test 错过的什么？
2. 增加第 5 个 role：“requirements analyst”，把 user wish 翻译成 planner-ready spec。哪些 communicative dehallucination requests 应该向上流到它？
3. 阅读 MetaGPT Section 3（“Agents”）。列出 MetaGPT 五个 roles 各自的 input/output schema。
4. 阅读 ChatDev 的 chat-chain diagram（arXiv:2307.07924 Figure 3）。识别 communicative dehallucination 在哪里打断了一个原本会无限循环的 loop。
5. PwC 的 7 倍 accuracy gain 来自 verification loops。假设三个 adding verifier 不会有帮助的 tasks：它们的 correctness 无法 deterministic checking，或检查成本高到不可接受。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Role specialization | “Different agents, different jobs” | 为 planner/executor/critic/verifier roles 调整不同 system prompts。 |
| SOP pattern | “Encoded standard operating procedure” | MetaGPT 的 framing：每个 role 的严格 I/O schemas 把 team 变成 pipeline。 |
| Communicative dehallucination | “Ask before inventing” | ChatDev pattern：executor 缺少细节时问 planner，而不是编造。 |
| Critic | “LLM reviewer” | 主观、有 opinion 的 reviewer。抓 taste issues。会被 plausible prose 欺骗。 |
| Verifier | “Deterministic check” | 基于代码的 pass/fail。Test runner、type checker、schema validator。不会被 plausible prose 欺骗。 |
| Verification gap | “No one checked” | MAST failures 中的 21.3%。答案在没有会抓住 bug 的 check 时被发出。 |
| Revision loop | “Critic sends it back” | Critic rejection 触发 executor 带 feedback 重新运行。需要 budget。 |
| All-LLM anti-pattern | “Looks good to me” | 每个 role 都是 LLM，没有 deterministic check。经典 MAST failure。 |

## 延伸阅读

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — SOP-as-role-prompt reference paper
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；verification gaps 是 failures 的 21.3%
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — production role specification surface
