# 多 Agent 辩论与协作

> Du 等人（ICML 2024，“Society of Minds”）运行 N 个模型实例，让它们独立提出答案，然后在 R 轮中互相 critique，逐步收敛。它能提升事实性、规则遵循和推理能力。Sparse topology 在 token 成本上优于 full mesh。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 minutes

## 学习目标

- 解释 debate protocol：N 个 proposers、R 轮讨论，最后收敛到 shared answer。
- 描述为什么 debate 能提升事实性、规则遵循和推理能力。
- 解释 sparse topology：不是每个 debater 都需要看到所有其他 debaters。
- 用 stdlib 在 scripted LLM 上实现 full-mesh 和 sparse 两种 debate，并测量 token cost vs accuracy。

## 问题

Self-Refine（Lesson 05）是一个模型 critique 自己，有 groupthink 风险。CRITIC（Lesson 05）把 critique 锚定在外部工具上，但外部工具不总是可用。Debate 引入第三种模式：多个实例、交叉 critique、通过分歧收敛。

## 概念

### Society of Minds（Du 等人，ICML 2024）

- N 个模型实例独立回答同一个问题。
- 在 R 轮中，每个模型阅读其他模型的 proposals，并 critique 它们。
- 模型根据 critiques 更新自己的答案。
- R 轮后，返回收敛答案。

由于成本限制，原始实验使用 N=3、R=2。在困难问题上（MMLU、GSM8K、Chess Move Validity、biography generation），更多 agents 和更多 rounds 会提升准确率。

跨模型组合优于单模型 debate：ChatGPT + Bard 一起使用，比任意单独一个都更好。

### Sparse topology

“Improving Multi-Agent Debate with Sparse Communication Topology”（arXiv:2406.11776，2024-2025）显示，full-mesh debate 并不总是最优。Sparse topologies（star、ring、hub-and-spoke）可以用更低 token 成本匹配准确率。每个 debater 只看到一部分 peers。

含义：

- Full mesh N=5, R=3 = 5 × 3 = 15 个 proposals，每个 proposal 都阅读 4 个 peers = 60 次 critique ops。
- Star N=5, R=3（一个 hub + 4 个 spokes）= 15 个 proposals，spokes 只读 hub = 12 次 critique ops。

### Debate 什么时候有帮助

- **Factuality.** N 个独立 proposals，交叉检查会减少 hallucination。
- **Rule-following.** 比如棋步合法性：一个模型漏掉规则，其他模型能抓出来。
- **Open-ended reasoning.** 多种 framing 会逐渐收窄到正确答案。

### Debate 什么时候有害

- **Latency-sensitive UX.** N × R 个串行 rounds 会带来你可能无法接受的 latency。
- **Cost-sensitive scale.** 每个问题都需要 N × R tokens。
- **Simple factual lookups.** 一次查询比五个 agent 辩论便宜得多。

### 2026 年的实际形态

- **Anthropic orchestrator-workers**（Lesson 12）：带 synthesis step 的一种 debate 变体。
- **LangGraph supervisor**（Lesson 13）：central router + specialist agents 可以把 debate 实现为一个 node。
- **OpenAI Agents SDK**（Lesson 16）：agents 可以反复 handoff，完成 iterative critique。
- **Multi-agent evals**：把 debate 和 evaluator-optimizer 结合起来，生成 eval signal。

### 这个模式容易在哪里出错

- **Convergence collapse.** 所有 agents 都收敛到第一个错误答案。用 required disagreement rounds 缓解。
- **Hub failure.** 在 star topology 中，一个坏 hub 会污染所有人。轮换 hub，或使用多个 hubs。
- **Prompt homogenization.** 所有 agents 使用同一个 prompt，于是产出同样答案。使用 diverse prompts 和/或 diverse models。

## Build It

`code/main.py` 实现了 stdlib debate：

- `Debater` class（scripted LLM，带 per-debater opinion drift）。
- `FullMeshDebate` 和 `SparseDebate` runners。
- 三个问题：一个事实问题、一个规则问题、一个推理问题。
- Metrics：convergent answer、rounds to convergence、total critique ops。

运行它：

```bash
python3 code/main.py
```

输出会展示每个 protocol 的 accuracy 和 cost；sparse 在 2/3 个问题上以更低成本匹配 full mesh。

## Use It

- **Anthropic orchestrator-workers**：用于简单的 2-3-worker debates。
- **LangGraph**：用于带 checkpointing 的 stateful multi-round debate。
- **Custom**：用于研究或专门的 correctness guarantees。

## Ship It

`outputs/skill-debate.md` 会 scaffold 一个 multi-agent debate，支持配置 topology、N、R 和 convergence rule。

## 练习

1. 实现一个 “forced disagreement” 规则：第 1 轮中，每个 debater 都必须给出不同 proposal。测量它对 convergence speed 的影响。
2. 添加 confidence-weighted aggregation：debaters 返回 `(answer, confidence)`；aggregator 根据 confidence 加权。它有帮助吗？
3. 把其中一个 “agent” 换成带不同 opinions 的 scripted LLM。异质性会提高准确率吗？
4. 在你的 3 个问题上测量 full mesh vs sparse 的 token cost。画出 cost vs accuracy。
5. 阅读 Society of Minds paper。把你的 toy 移植到 N=5、R=3。什么会坏？什么会变好？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Debate | “Multi-agent critique” | N 个 proposers，R 轮 cross-critique，然后收敛 |
| Full mesh | “Everyone reads everyone” | 每个 debater 每轮都读取每个 peer |
| Sparse topology | “Limited peer view” | Debaters 只读取 peers 的一个子集 |
| Hub-and-spoke | “Star topology” | 一个 central debater，N-1 个 spokes 只读取 hub |
| Convergence | “Agreement” | Debaters 收敛到一个 shared answer |
| Society of Minds | “Du et al. debate paper” | ICML 2024 multi-agent debate 方法 |

## 延伸阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325)：canonical multi-agent debate。
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776)：sparse topology results。
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：orchestrator-workers 作为 debate 变体。
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)：单模型 self-critique counterpart。
