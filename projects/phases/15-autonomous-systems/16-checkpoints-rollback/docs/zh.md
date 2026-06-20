# Checkpoints 与 Rollback

> 每一次 graph-state transition 都会持久化。当 worker 崩溃时，它的 lease 过期，另一个 worker 会从最新 checkpoint 接手。Cloudflare Durable Objects 能把 state 保存数小时到数周。Propose-then-commit（第 15 课）为每个 action 定义 rollback plan。Post-action verification 把循环闭合。EU AI Act Article 14 要求 high-risk systems 必须有 effective human oversight；在实践中，这意味着 checkpoints 必须可查询，rollbacks 必须经过演练，audit trail 必须能跨部署存活。尖锐的 failure mode 是：如果没有 idempotency keys 和 precondition checks，一次 transient failure 后的 retry 可能会双重执行已经批准过的 action。Post-action verification 就是用来抓住它的。

**Type:** Learn
**Languages:** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time:** ~60 minutes

## 问题

Durable execution（第 12 课）让崩溃的 agent 可以恢复。Propose-then-commit（第 15 课）让已经批准的 action 可以审计。本课把两者接起来：当一个已经批准的 action 只执行了一部分、随后崩溃并恢复时，会发生什么？Rollback 什么时候运行？它基于什么 state 运行？

真实系统会用不同方式把这件事接起来：

- **LangGraph** 会把每一次 graph-state transition checkpoint 到 PostgreSQL。Worker 崩溃时，lease 释放，另一个 worker 从最新 checkpoint 恢复。Workflow 在 `interrupt()` 上暂停，而这个暂停本身也会持久化。
- **Cloudflare Durable Objects** 按 key 保存 state，能跨数小时或数周。把 computation 和 approved action 的 storage 放在一起。
- **Microsoft Agent Framework** 在 workflow API 中暴露 `Checkpoint` primitives；replay 加上 idempotency 覆盖 retries。

每一种情况下，真正有效的组合都是：idempotency key（防止 double-execute）+ precondition check（state 仍然是批准时所依据的状态）+ post-action verify（副作用确实发生了）+ verify-fail 时 rollback。

## 概念

### 每次 transition 都持久化

Graph-state transition 指的是任何把 workflow 从一个具名 state 移到另一个具名 state 的步骤。朴素实现只在特定 commit points 持久化；生产实现会持久化每一次 transition。成本只是多几次写入，相比可靠性收益很小：replay 可以落在任意位置，lease recovery 也能精确恢复。

### Lease recovery

Worker 崩溃时，workflow 不会丢失；lease（一个短期 claim，表示这个 worker 正在执行这次 run）只是过期。另一个 worker 会拿起最新 checkpoint 并继续执行。Lease 机制让生产系统能在 rolling deploys 中存活，而不会丢掉 in-flight work。

### Idempotency 加 preconditions

只有 idempotency 还不够。考虑一个 workflow 被批准执行：“当 A 的余额 > $1000 时，从 A 转 $100 给 B。” Workflow committed，执行中途崩溃，然后恢复。如果只检查 idempotency key，恢复后执行继续，transfer 只运行一次（这是正确的）。但如果在 crash 和 resume 之间，另一个 workflow 把 A 的余额降到了 $500 呢？Idempotency check 仍然通过；precondition 不通过。如果没有 precondition check，我们就制造了透支。

每个 consequential action 都需要两者：

- **Idempotency key**：防止 double-execute。
- **Precondition check**：确认 state 仍与批准 action 时一致。

### Post-action verification

“Tool returned 200” 不是 verification。真正的 verification 会重新读取目标 state，并确认副作用确实发生。常见模式：

- Database update：`UPDATE ... RETURNING *`，然后断言 returned row 匹配预期 state。
- Email send：提交后在 sent-folder 中检查 message ID。
- File write：把文件读回来并 hash。
- API call：对目标 resource 做后续 `GET`。

如果 verify 失败，workflow 就处于一个 known-bad state。Rollback 会介入。

### Rollback plans

Propose-then-commit（第 15 课）里的每个 consequential action 都携带 rollback plan。类型包括：

- **In-band rollback**：直接反转副作用（`INSERT` 后 `DELETE`，send 后 `Send-correction-email`）。
- **Compensating transaction**：一个新的 action，用来中和原始 action（标准 SAGA pattern）。
- **Out-of-band rollback**：提醒人类、暂停 workflow、保留 bad state 供调查。

No-op rollback（“我们无法撤销这件事”）必须在 proposal 中明说。没有 rollback 的 actions 在 commit time 需要更强的 HITL（第 15 课 challenge-and-response）。

### EU AI Act Article 14 的操作化解读

Article 14 要求 high-risk systems 具备 “effective human oversight”。在操作层面，implementers 通常把它读作：

- Checkpoints 可供 auditor 查询。
- Rollbacks 被演练过（至少端到端测试一次）。
- Audit trail 能跨部署存活（checkpoint backend 不是 ephemeral）。
- Failed verifications 会触发 alert，而不是静默写 log。

一个 workflow 如果在 mid-commit 崩溃、恢复并完成副作用，却没有 verify + rollback pathway，就无法通过 Article 14 的检验。

### 尖锐 failure mode：double-execute

这个领域最常见的生产事故：

1. Action 被批准，idempotency key 为 k。
2. Commit 开始，执行 action，返回 200。
3. Workflow 在持久化 “committed” status 之前崩溃。
4. Workflow 恢复；看到 “approved but not committed”；再次执行。
5. Side effect 触发两次。

缓解方式：执行前先持久化一个 “in-flight” intent，用 idempotency key 执行，然后只有在 post-action verification 成功后才标记 “committed”。如果 action 触发了，但 status write 失败，你知道应该 verify，并在必要时重新触发。如果 status write 成功而 action 失败，你会 verify，并通过 recovery path 精确触发一次。

## Use It

`code/main.py` 实现了一个带 checkpoint 的 workflow，包含 idempotency、preconditions、verify 和 rollback。Driver 模拟四个场景：clean run、crash 后 retry（idempotency 抓住）、precondition fail（workflow 不触发 action 并 abort）、verify fail（rollback 触发）。

## Ship It

`outputs/skill-rollback-rehearsal.md` 会为 proposed workflow 设计 rollback-rehearsal test，并审计 checkpoint backend 是否具备 audit-trail persistence。

## 练习

1. 运行 `code/main.py`。验证四个场景。对于 crash-during-commit case，确认 action 在 retries 之间只触发一次。

2. 修改 “mark as done first, then do it” pattern，让 status write 在 action 之后触发。重新运行 crash scenario。测量会触发多少 duplicate actions。

3. 为一个具体 production action 设计 rollback plan（例如 “post to a Slack channel”）。把它归类为 in-band、compensating 或 out-of-band，并说明理由。

4. 选一个你熟悉的 workflow。识别每一个 state transition。为每个 transition 标注 durability requirement（persist / do not persist）。数一数现在还有多少 transition 没有持久化。

5. Rehearsed-rollback test：设计一个端到端测试，运行真实 workflow，让它崩溃，并确认 rollback path 会触发。这个测试要断言什么？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Checkpoint | “Save point” | 每一次 graph-state transition 都持久化到 durable store |
| Lease | “Worker claim” | 短期 claim，表示某个 worker 正在执行一次 run；崩溃后过期 |
| Precondition | “State gate” | 断言 state 仍与 approved action 保持一致 |
| Post-action verify | “Re-read check” | 确认副作用确实在目标系统中发生 |
| In-band rollback | “Direct undo” | 用 inverse operation 反转副作用 |
| Compensating transaction | “SAGA undo” | 一个新的 action，用来中和原始 action |
| Mark-as-done-first | “Status write order” | 在 commit 返回前持久化 committed status |
| Article 14 | “EU AI Act human oversight” | 操作化含义：queryable checkpoints、rehearsed rollbacks、auditable trail |

## 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop)：checkpoint primitives 和 lease recovery。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/)：Durable Objects 作为 state substrate。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/)：regulatory baseline。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)：long-horizon workflows 的 reliability framing。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop)：Claude Code Routines 的 workflow shape。
