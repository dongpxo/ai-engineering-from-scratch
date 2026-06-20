# Failure Modes：MAST、Groupthink、Monoculture、Cascading Errors

> 2026 年的参考 taxonomy 是 **MAST**（Cemri et al., NeurIPS 2025, arXiv:2503.13657），它来自 7 个 state-of-the-art open-source MAS 的 1642 条 execution traces，并显示 **41-86.7% failure rate**。三个根类别：**Specification Problems**（41.77%）——role ambiguity、unclear task definitions；**Coordination Failures**（36.94%）——communication breakdowns、state desync；**Verification Gaps**（21.30%）——missing validation、absent quality checks。**Groupthink** family（arXiv:2508.05687）补充了：monoculture collapse（同一个 base model → correlated failures）、conformity bias（agents 互相强化错误）、deficient theory of mind、mixed-motive dynamics、cascading reliability failures。级联例子：retry storms，payment failure 触发 order retries，order retries 触发 inventory retries，几秒内把 inventory service 打到 10x load，需要 circuit breakers。Memory poisoning：一个 agent 的 hallucination 进入 shared memory，下游 agents 把它当事实；accuracy 逐渐衰减，让 root-cause diagnosis 很痛苦。**STRATUS**（NeurIPS 2025）报告通过 specialized detection / diagnosis / validation agents，把 mitigation success 提高 1.5x。本课把 failure modes 当成一等工程目标。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Time:** ~75 minutes

## 问题

Multi-agent systems 在真实任务上会有 41-86.7% 的失败率（Cemri et al. 2025 在 7 个 open-source MAS 上测得）。这不是靠 “just add more agents” 能调好的。失败有结构性原因。MAST taxonomy 给你类别。本课把每个类别映射到具体的 detection、diagnosis 和 mitigation pattern，让这些数字不再看起来像任意噪声。

2026 年的生产实践是把 failure modes 当作 design inputs。你的架构还不能算 “good enough”，直到你能指着每个 MAST category，说出已经部署了哪种 mitigation。

## 概念

### MAST categories

**Specification Problems（41.77% of failures）。** Agent 的 task 没有被定义得足够严格。例子：

- Role ambiguity：两个 agents 都以为自己是 reviewer。
- Task underspecified：用户说 “summarize this”，但真正想要的是某个特定角度。
- Success criteria implicit：agent 无法判断自己是否成功。

Mitigations：

- 写 explicit role contracts。每个 agent 的 prompt 说明它做什么，**也说明它不做什么**。
- 每个 task 都有 acceptance tests。Agent 开始前，先定义 “done looks like X”。
- Pre-flight spec check：dispatch 前由一个独立 agent 审查 task definition。

**Coordination Failures（36.94%）。** Communication 或 state breakdowns。

例子：

- 两个 agents 没有同步就更新 shared state。
- Agents 之间 message 丢失（queue failure、timeout）。
- State drift：agent A 认为 task 已完成；agent B 仍在执行。

Mitigations：

- Versioned shared state with optimistic concurrency。
- Critical messages 要有 explicit acknowledgment（retry until acked）。
- 周期性 state-sync checkpoints；早发现 drift。

**Verification Gaps（21.30%）。** 输出没有 independent check。

例子：

- 一个 agent 声称成功；没人验证。
- 一串 agents 每个都信任前一个的输出。
- 对 emergent composed behavior 缺少 test coverage。

Mitigations：

- Independent verifier agent（第 13 课）。Read-only，并拥有独立 source access。
- Explicit handoff contract：“A 的输出必须通过 checker C，B 才能开始。”
- Outcome logging，供 post-hoc analysis。

### Groupthink family（arXiv:2508.05687）

当 agents 变得同质化或互相模仿时，会出现五类相关 failures：

**Monoculture collapse.** 同一个 base model 或 training data → correlated errors。三个 agents 共享一个 LLM，就会共享它的 hallucinations。

**Conformity bias.** Agents 向最响亮或最自信的 peer 调整，即使那个 peer 是错的。

**Deficient ToM.** Agents 无法建模彼此的 beliefs；coordination 崩掉（第 18 课）。

**Mixed-motive dynamics.** 部分对齐的 agents 会漂向 compromise-middle，最后谁都不满足。

**Cascading reliability failures.** 一个组件的 error pattern 触发依赖组件中的 error patterns。

### Cascading example：retry storm

一个经典 2026 incident pattern：

```text
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

修复方式是经典的：**circuit breakers**。当 downstream error rate 超过 threshold，短路并返回 cached 或 default results。再加上每个 request 的 capped retry budgets。

Circuit breakers 是少数可以直接从 distributed systems 借来、几乎不需要改造的 multi-agent failure mitigations。

### Memory poisoning（再次出现）

来自第 13 课：一个 agent 的 hallucination 变成 shared-memory fact；下游 agents 基于被污染事实推理。在 MAST 术语里，这是 shared-memory layer 上的 verification gap。

症状是 gradual accuracy decay。你不会得到 crash；你会得到缓慢 drift，而且 root-cause 很难找。

Mitigation：append-only log、provenance、unwritable verifier。第 13 课已经覆盖。

### STRATUS：用于 failure detection 的 specialized agents

STRATUS（NeurIPS 2025）报告说，部署以下 agents 后 mitigation success 提升 1.5x：

- **Detection agent.** 观察 symptom patterns（high disagreement、retry spikes、accuracy drift）。
- **Diagnosis agent.** 给定 symptoms，根据 MAST taxonomy 推断 likely root cause。
- **Validation agent.** Mitigation 应用后，检查 symptoms 是否清除。

这是 SRE-style incident response，用在 agent systems 上。三个 roles 都可以是带 specialized prompts 的 LLM agents。

### Failure-mode audit

2026 年的最佳实践是每年（或每个 major release）做 failure-mode audit：

1. **Trace sample.** 收集约 1000 条真实 execution traces。
2. **Categorize.** 对每条 trace 中的 failures，映射到 MAST + Groupthink categories。
3. **Compute failure-by-category rate.** 哪些 categories 主导你的系统？
4. **Rank mitigations.** 哪个 fix 能消除最多 failures？
5. **Pick 2-3 mitigations.** 实现；下个季度重新 audit。

纪律比具体选择更重要。没有 audits，failures 会混进噪声里，永远不会被系统性处理。

### 当系统 silent fail

最危险的 failure category 是 silent correctness failure。一个 loudly fail 的系统（crash、exception、alert）可以被监控。一个产出 plausible-but-wrong outputs 的系统，不能靠 exception logs 检测。这就是为什么 verification gaps 虽然按数量只占 21.30%，但每次 failure 的成本最高。

应投入：

- Sample-based human review。
- Golden-dataset regression tests。
- 重要 outputs 上的 cross-agent cross-checking。

### Failure vs slow failure

有些 failure 立即发生；有些 failure 很慢。Immediate failures（timeout、schema mismatch、auth error）便宜且容易检测。Slow failures（memory poisoning、monoculture drift、role ambiguity）检测和预防都更贵。

2026 年的工程动作是：instrument slow-failure proxies，在 drift 变成可见错误之前抓住它。Agreement rate、retry rate、output-length distribution，以及 consecutive agent versions 之间的 edit-distance 都是有用 proxies。

## Build It

`code/main.py` 实现：

- `FailureTaxonomy`：把 simulated incidents 分类到 MAST + Groupthink categories。
- `CircuitBreaker`：经典模式；error rate 超过 threshold 时打开。
- `RetryStormSimulator`：展示 cascading failure；可切换 circuit breaker on / off。
- `DetectionAgent`：scripted STRATUS-style symptom matcher。

运行：

```bash
python3 code/main.py
```

Expected output：

- 没有 circuit breaker 的 retry storm：inventory errors 爆炸（模拟）。
- 有 circuit breaker：限制在 threshold；返回 degraded-mode responses。
- Detection agent 标记该 pattern，并命名 MAST category。

## Use It

`outputs/skill-mast-auditor.md` 会对 multi-agent system 运行 MAST-style failure-mode audit。流程是：traces → categorization → mitigation ranking。

## Ship It

生产中的 failure-mode discipline：

- **MAST audit per quarter.** 不要 annual。Categories 会随着系统增长而变化。
- **Circuit breakers everywhere.** 每个指向 dependent service 的 outbound call 都要有。默认 open threshold 设在 5-10% error rate。
- **Golden datasets.** 小而高质量，人工审计过。每周跑 regression-test。
- **STRATUS trio.** Detection + Diagnosis + Validation agents 监控 production。先从 detection agent 开始；当 symptoms 变得 noisy 后再加 diagnosis。
- **Failure budget.** 为每个 category 的 failure rate 设置显式 SLO。超过 budget 就触发 stop-shipping conversation。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker 会限制 retry storm。调整 failure threshold，观察 tradeoff。
2. 实现一个 **slow-failure proxy**：3 个 parallel agents 的 agreement rate。当它急剧下降时触发 alert。通过逐渐让 agent outputs 相关来模拟 monoculture drift。
3. 阅读 Cemri et al.（arXiv:2503.13657）。选择他们的 7 个 MAS systems 之一，映射其 top 3 failure categories。它们和 MAST 预测相比如何？
4. 阅读 Groupthink paper（arXiv:2508.05687）。指出五种 patterns 中哪一种在生产中最难检测，并提出一个 proxy metric。
5. 为你熟悉的一个 multi-agent system 设计 STRATUS-style detection-diagnosis-validation trio。Detection 观察哪些 symptoms？Diagnosis 推荐什么 mitigations？Validation 如何确认它们有效？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MAST | “The 2026 taxonomy” | Cemri 2025；3 个 root categories + 14 个 failure sub-types。 |
| Specification Problem | “Role ambiguity” | Task 或 role 定义不足；agents 不知道该做什么。 |
| Coordination Failure | “State drift” | Agents 之间 communication 或 sync breakdown。 |
| Verification Gap | “No one checked” | Outputs 没有 independent validation 就被接受。 |
| Groupthink family | “Homogeneity failures” | Monoculture、conformity、deficient ToM、mixed-motive、cascading。 |
| Monoculture collapse | “Same model, same hallucinations” | 共享 base model 或 training data 导致 correlated errors。 |
| Retry storm | “Cascading error amplification” | 一个 failure 触发 retries，retries 又放大 downstream load。 |
| Circuit breaker | “Fail fast on error rate” | Error rate 超过 threshold 时 open；用 default 短路。 |
| STRATUS | “Incident response trio” | Detection + diagnosis + validation agents；mitigation success 提升 1.5x。 |
| Memory poisoning | “Hallucinations propagate” | Shared-memory fact 被污染；下游 agents 基于毒化事实推理。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)：MAST taxonomy，NeurIPS 2025。
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687)：monoculture、conformity 和 five-family taxonomy。
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/)：NeurIPS 2025 proceedings entry（detection + diagnosis + validation）。
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/)：canonical circuit-breaker reference。
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：production failure-mode notes。
