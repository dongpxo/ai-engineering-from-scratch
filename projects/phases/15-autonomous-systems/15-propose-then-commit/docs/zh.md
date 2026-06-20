# Human-in-the-Loop：Propose-Then-Commit

> 2026 年关于 HITL 的共识已经很具体。它不是“agent 询问，用户点击 Approve”。它是 propose-then-commit：proposed action 会带着 idempotency key 持久化到 durable store；展示给 reviewer 时包含 intent、data lineage、触及的 permissions、blast radius 和 rollback plan；只有收到 positive acknowledgement 后才 commit；执行后还要 verify，确认 side effect 真的发生了。LangGraph 的 `interrupt()` 加 PostgreSQL checkpointing、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现了同一种形状。典型 failure mode 是 rubber-stamp approval：“Approve?” 被点了，但没人真正 review。文档化 mitigation 是带显式 checklist 的 challenge-and-response。

**Type:** Learn
**Languages:** Python（stdlib，propose-then-commit state machine with idempotency）
**Prerequisites:** Phase 15 · 12（Durable execution），Phase 15 · 14（Tripwires）
**Time:** ~60 minutes

## 问题

agent 要执行一个 action。用户必须决定：批准还是不批准。如果这个决定是瞬间完成的，它很可能不是 review。如果这个决定是结构化的，它会更慢，但更可信。工程问题是：如何让结构化 review 成为阻力最小的路径。

2023 年代的 HITL pattern 是同步 prompt：“Agent wants to send email to X with body Y — approve?” 用户点击 Approve。大家都觉得系统安全了。实践中，这个 surface 很容易被 rubber-stamp：用户快速批准，approval 对风险预测很弱；当 agent 出错时，audit trail 只会显示一长串用户已经记不起来的 approvals。

2026 年的 pattern——propose-then-commit——把 HITL 移到 durable substrate 上，附加结构化 metadata，并要求 positive commit。每个 managed agent SDK 都发货了某个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名字不同；形状相同。

## 概念

### Propose-then-commit state machine

1. **Propose.** Agent 生成一个 proposed action。它会被持久化到 durable store（PostgreSQL、Redis、Durable Object）。记录包括：
   - intent：agent 为什么要这么做
   - data lineage：哪个 source 导致了这个 proposal
   - permissions touched：触及哪些 scopes / files / endpoints
   - blast radius：最坏情况是什么
   - rollback plan：如果 commit 了，如何撤销
   - idempotency key：每个 proposal 唯一；重复提交返回同一条 record
2. **Surface.** Reviewer 看到 proposal 和全部 metadata。reviewer 是一个人，不是 agent 审查自己。
3. **Commit.** Positive acknowledgement。action 执行。
4. **Verify.** 执行后读回 side effect 并确认。如果 verify step 失败，系统进入一个 known bad state，并触发 alerting。

### Idempotency key

没有 idempotency key，transient failure 后的 retry 可能让一次 approved action 被执行两次。具体例子：用户批准“从 A 向 B 转账 $100”。网络抖动。workflow retry。用户只批准了一次，但转账执行了两次。idempotency key 把 approval 绑定到一个唯一 side effect；第二次执行会成为 no-op。

这和 Stripe、AWS APIs 使用的 idempotency pattern 相同。Microsoft Agent Framework docs 明确把它复用于 agent approvals。

### Durability：为什么 approvals 必须比进程活得更久

approval waiting room 是一段不属于 agent 的 state。workflow 被暂停（Lesson 12）。approval 到达后，workflow 从那个精确位置恢复。这就是为什么 LangGraph 把 `interrupt()` 和 PostgreSQL checkpointing 搭配，而不是只用 in-memory state：两天后的 approval 仍然能找到完整 workflow。

### Rubber-stamp approvals 与 challenge-and-response mitigation

HITL 的默认 UI（“Approve” / “Reject” 按钮）会产生快速 approvals，但没有真正 review。文档化 mitigation 是 challenge-and-response checklist：在 Approve 按钮可用之前，要求 reviewer 对具体问题给出 positive answers。具体形状：

- “Do you understand what resource this touches? [ ]”
- “Have you verified the blast radius is acceptable? [ ]”
- “Do you have a rollback plan if this fails? [ ]”

这不是为了 bureaucratic friction 而增加 bureaucracy，而是一个 forcing function。无法勾选这些框的 reviewer，要么请求 clarification（escalation），要么拒绝（safe default）。Anthropic agent-safety research 明确把 checklist-driven HITL 作为 rubber-stamp approval patterns 的 mitigation。

### 什么算 consequential

不是每个 action 都需要 propose-then-commit。2026 年 guidance：

- **Consequential actions**（总是 HITL）：不可逆 writes、financial transactions、outbound communication、production database changes、destructive file-system operations。
- **Reversible actions**（有时 HITL）：本地文件 edits、staging-env changes、带清晰 rollback 的 reversible writes。
- **Reads and inspections**（从不 HITL）：读取文件、列出 resources、调用 read-only API。

### Post-action verification

“commit ran” 不等于 “side effect happened”。Network partition 和 race conditions 会造成 workflow 以为自己成功，但 backend 没有 persist。verify step 会在 commit 后重新读取目标 resource 来确认。这和 database transactions 中的 `RETURNING` clause，或者 AWS `PutObject` 后的 `GetObject` 是同一种模式。

### EU AI Act Article 14

Article 14 要求欧盟 high-risk AI systems 具备 effective human oversight。“Effective” 不是装饰性的。监管语言明确排除 rubber-stamp patterns。在 Microsoft Agent Governance Toolkit compliance docs 中，带 challenge-and-response 的 propose-then-commit，正是能经受 Article 14 审视的形状。

## Use It

`code/main.py` 用 stdlib Python 实现了一个 propose-then-commit state machine。Durable store 是 JSON file。Idempotency key 是 `(thread_id, action_signature)` 的 hash。driver 模拟三种情况：干净的 approval flow、transient failure 后的 retry（绝不能 double-execute），以及 rubber-stamp default 和 challenge-and-response flow 的对比。

## Ship It

`outputs/skill-hitl-design.md` 会审查一个 proposed HITL workflow 是否具备 propose-then-commit 形状，并标记缺失的 metadata、idempotency、verification 或 challenge-and-response layers。

## 练习

1. 运行 `code/main.py`。确认一个 approved proposal 的 retry 会使用 durable record，并且不会重新执行。然后把 idempotency key 改成包含 timestamp，展示 retry 会 double-execute。

2. 给 proposal record 增加一个 `rollback` field。模拟一次 verify step 失败的 execution。展示 rollback 会自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` docs。找出一个 API 包含但 toy engine 缺失的 metadata field。加入它，并解释它防范什么。

4. 为一个具体 action（例如“post to a public Twitter account”）设计 challenge-and-response checklist。reviewer 必须回答哪三个问题？为什么是这三个？

5. 选择一个同步 “Approve?” prompt 就足够的场景（不需要 durable store）。解释原因，并命名你接受的 risk class。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Propose-then-commit | “Two-phase approval” | 持久化 proposal + positive commit + verify |
| Idempotency key | “Retry-safe token” | 每个 proposal 唯一；第二次执行 no-ops |
| Data lineage | “Where it came from” | 导致 proposal 的具体 source content |
| Blast radius | “Worst case” | action 出错时的影响范围 |
| Rubber-stamp | “Fast approval” | 没有真正 review 就点击 “Approve” |
| Challenge-and-response | “Forcing checklist” | Reviewer 必须明确确认具体问题 |
| RequestInfoEvent | “MS Agent Framework primitive” | 带结构化 metadata 的 durable HITL request |
| `interrupt()` / `waitForApproval()` | “Framework primitives” | LangGraph / Cloudflare 中同一形状的等价 primitive |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、durable approvals。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为 long-horizon risk mitigation。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — high-risk systems 的 regulatory baseline。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕 oversight 的 constitutional framing。
