# Agents 的共识与拜占庭容错

> 经典分布式系统 BFT 遇上随机性的 LLM。2025-2026 年出现了三个研究方向：**CP-WBFT**（arXiv:2511.10400）用 confidence probe 给每张票加权；**DecentLLMs**（arXiv:2507.14928）去掉 leader，用并行 worker proposals 和 geometric-median aggregation；**WBFT**（arXiv:2505.05103）把 weighted voting 和 Hierarchical Structure Clustering 结合起来，拆分 Core 和 Edge nodes。“Can AI Agents Agree?”（arXiv:2603.01213）的诚实经验结果是：今天连 scalar agreement 都很脆弱，一个 deceptive agent 就能破坏 Mixture-of-Agents。BFT 必要但不充分。本课会构建一个最小 BFT protocol，注入三种 agent-specific attacks（byzantine lie、sycophantic conformity、correlated-error monoculture），并测量每种 consensus variant 如何应对。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## 问题

你有 N 个 LLM agents，每个都产出一个答案。它们不同意。Majority vote 选错了，因为两个 agents 是 correlated 的（同一个 base model、同一批 training data、同一种 failure modes）。第三个 agent 又以一种新方式错了，于是 majority 变成 false majority。

现在再加入一个 deceptive agent：它故意撒谎。或者一个 sycophantic agent：它同意最后发言的人。经典 BFT 的假设是 Byzantine nodes 占比 `f < n/3`，并且可以任意行动。2026 年的现实是：即使 honest，LLM nodes 也是 stochastic 的；不同模型之间存在 correlated errors；而且 agents 会受彼此输出影响。你不能把它们当作彼此独立的 Bernoulli voters。

经典 BFT（PBFT，1999）并没有错，只是不完整。它处理 arbitrary bit-flipping。它不处理“三个 honest agents 因为共享 training data 而拥有同一个 hallucination”。本课从 PBFT 的基础出发，再叠加三种 2025-2026 年的适配。

## 概念

### 经典 BFT 给你什么

Practical Byzantine Fault Tolerance（Castro & Liskov，OSDI 1999）可以容忍 `f < n/3` 个 Byzantine nodes。协议包含三个 phases（pre-prepare、prepare、commit）和两个 primitives（signed messages、quorum certificates）。它让 `n >= 3f + 1` 个 honest-or-malicious nodes 对一个单一 value 达成 agreement。

这些保证很强，但它假设：

1. **Independent faults.** Byzantines 不协同。
2. **Honest nodes 真的 honest.** Honest outputs 的正确性不是问题；协议只对齐 disagreement。
3. **问题有 ground-truth answer.** 对错误事实达成 consensus 仍然是 consensus。

LLM agents 违反了这三条。两个跑同一 base model 的 agents 会共享 faults。一个 “honest” LLM 仍会 hallucinate。对于 ambiguous questions，“truth” 就是 agents 决定的东西，并没有外部 oracle。

### 三种 LLM-specific attacks

**Byzantine lie.** 一个 agent 故意输出错误答案。如果 `f < n/3`，经典 BFT 可以处理。

**Sycophantic conformity.** 一个 agent 先读其他人的答案，再向最后发言者靠拢。它不是恶意的，但会与最响亮的声音相关。经典 BFT 无法阻止它，因为这个 agent 通过所有 signature checks。

**Correlated-error monoculture.** 三个 agents 共享同一个 base model。它们 hallucinate 同一个错误答案。Majority 是错的。经典 BFT 无能为力，因为三者都“诚实地”同意。

### 2025-2026 年的回应

**CP-WBFT**（arXiv:2511.10400）——Confidence-Probed Weighted BFT。每个 voter 给自己的 answer 附上 confidence probe（一个 self-reported probability，或一个独立 calibration model 的预测）。Vote weights 随 confidence 缩放。论文报告在 complete graphs 上 BFT improvement +85.71%。缓解对象：sycophantic conformity（跟风 agents 对自己主动提供的立场往往 confidence 较低）。

**DecentLLMs**（arXiv:2507.14928）——Leaderless。Worker agents 并行提出 proposals，evaluator agents 给 proposals 打分，最终答案是 scored positions 的 geometric median。在 `f < n/2` 时更 robust。缓解对象：Byzantine lie 和 correlated errors（geometric median 对 outliers robust，会拉向 dense cluster，而不是 model-biased average）。

**WBFT**（arXiv:2505.05103）——Weighted BFT with Hierarchical Structure Clustering。Vote weights 由 response quality 加历史学习出的 trust score 决定。Agents 被聚成 Core 和 Edge；Core agents 必须先达成 consensus，Edge agents 跟随。缓解对象：scalability（Core consensus 小且快），并且部分缓解 monoculture（Core 可以按 diversity 选择）。

### 经验结果：“Can AI Agents Agree?”（arXiv:2603.01213）

这篇论文在多个 frontier models 上测量 scalar agreement（LLM agents 对单一 numeric value 达成一致）。发现并不好看：

- 即使没有 adversaries，LLM agents 在许多 benchmarks 的 scalar questions 上 disagreement rate 也超过 30%。
- 一个采用 deceptive persona 的 agent，可以把 Mixture-of-Agents consensus 拉离 honest baseline 40 多个百分点。
- Disagreement rates 与 model diversity 相关：heterogeneous ensembles disagree 更多（好处：errors 更不相关），但 drift 更慢（坏处：time-to-agreement 更长）。

Takeaway：BFT 给你对齐 outputs 的机械结构，但它不告诉你对齐后的 output 是否正确。要把它和 verification（Phase 16 · 08 role specialization）、diversity（Phase 16 · 15 debate variants）以及 evaluator agents（Phase 16 · 24 benchmarks）组合起来。

### 核心协议，压到最小

一个面向 LLM agents 的最小 BFT round：

```text
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

Semantic clustering 是 LLM-specific twist。两个答案 “the study reports 4.2%” 和 “4.2% improvement” 属于同一个 cluster。朴素 string-equality 会漏掉。生产中可以使用便宜的 embedding model，或显式 canonicalization。

### Threshold tuning

`threshold` 参数决定什么时候 accept，什么时候 retry。太低：接受 weak majorities。太高：什么都不接受。经验范围：对于 `n=5-7` 个 agents，常见是 0.5-0.67；`n` 更小时需要更高。低于阈值时，升级给 human 或另一个 agent ensemble。

### Consensus 帮不上忙的地方

- **Ambiguous questions.** 如果问题没有 ground truth，consensus 就是 opinion。要把它叫作 opinion。
- **Compound questions.** “Write code and explain it” 是两个答案。分别投票。
- **Adversarial multi-round.** 如果 agents 能观察前几轮并模仿（Du 2023 debate），它们会开始彼此同意，而不管 truth。限制 rounds（通常 2-3 轮）。

## Build It

`code/main.py` 实现：

- `AgentVoter`：带 `(answer, confidence)` 的 scripted policy。
- `MajorityVote`：经典 plurality。
- `CPWBFT`：带 semantic clustering 的 confidence-weighted voting。
- `DecentLLMs`：在 scored proposals 上做 geometric-median aggregation。
- `Scenario`：在三种 attack patterns 下运行每个 aggregator。

实现的 attack patterns：

1. `byzantine`：一个 agent 以 high confidence 撒谎。
2. `sycophancy`：一个 agent 复制它看到的第一个答案，并匹配 confidence。
3. `monoculture`：三个 agents 共享一个 wrong answer（correlated error），confidence 中等。

运行：

```bash
python3 code/main.py
```

Expected output：一张 `(attack, aggregator) -> final answer` 表格，并高亮 correct answer。Plurality 会在 monoculture case 失败。CPWBFT 的 confidence weighting 会缓解 sycophancy。DecentLLMs 的 geometric median 会在 monoculture 少于总体一半时拉向 honest cluster。

## Use It

`outputs/skill-consensus-designer.md` 会为 multi-agent ensemble 设计 consensus protocol：clustering method、weighting、threshold，以及 sub-threshold rounds 的 escalation policy。

## Ship It

发布任何 consensus mechanism 前：

- **至少用上面的三种 patterns 做 attack-test。** 你的 protocol 应该 predictably fail，而不是 silent fail。
- **记录每个 minority cluster** 及其 provenance。Minority clusters 是 correlated errors 的 early-warning system。
- **强制 bounded rounds。** 不要 “keep debating until agreement”——这会奖励 sycophancy。
- **把 agreement 和 correctness 分开。** Consensus output 要进入 verifier；verifier 必须 independent of the ensemble。
- **监控 agreement rate。** 急剧上升意味着 conformity bias；急剧下降意味着 model drift。

## 练习

1. 运行 `code/main.py`。确认 plurality 会在 monoculture attack 中失败，但当 monoculture confidence 低于 0.7 时，CPWBFT 会部分缓解。
2. 添加第四种 attack pattern：**silent abstention**——一个 agent 拒绝回答（“I don't know”）。每个 aggregator 应如何处理 abstentions？实现你的选择。
3. 把 semantic clustering 从 string canonicalization 换成 embedding-similarity（使用任意 open-source embedding model）。Sycophancy attack 会发生什么变化？
4. 阅读 CP-WBFT（arXiv:2511.10400）。实现 confidence-probe calibration step（一个独立 calibration model 检查每个 agent 的 self-reported confidence）。测量 monoculture scenario 上的 accuracy gain。
5. 阅读 “Can AI Agents Agree?”（arXiv:2603.01213）。复现一个简化版 scalar-agreement experiment：三个 agents、一个 scalar question、deceptive-persona prompt。CPWBFT 或 DecentLLMs 能抓住它吗？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| BFT | “Byzantine fault tolerance” | Castro-Liskov 1999 protocol，用于在 `f < n/3` arbitrary faults 下达成 consensus。 |
| Byzantine | “Any bad behavior” | 一个 node 可以撒谎、丢消息、静默失败，也就是除了安全 crash 之外的任意行为。 |
| Confidence probe | “How sure are you?” | 附在 vote 上的 self-reported 或 calibrator-predicted probability。 |
| Semantic clustering | “Same answer, different words” | 计票前把等价答案分到同一组。 |
| Geometric median | “Robust center” | 使到样本点距离总和最小的点。不同于 mean，它对 outliers robust。 |
| Monoculture | “Same model, same failures” | Agents 共享 training data 或 base model 时产生 correlated errors。 |
| Sycophantic conformity | “Agreeing with the loud voice” | Agent 的 vote 偏向第一个或最响亮的发言者。 |
| Core/Edge | “Hierarchical BFT” | WBFT split：小规模 Core 先 consensus，Edge nodes 跟随。用于约束 latency。 |

## 延伸阅读

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf)：foundation。
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400)：按 confidence 加权 voting。
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928)：geometric-median aggregation。
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103)：用 Core/Edge split 约束 latency。
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213)：scalar-agreement fragility 和 deceptive-persona attack。
