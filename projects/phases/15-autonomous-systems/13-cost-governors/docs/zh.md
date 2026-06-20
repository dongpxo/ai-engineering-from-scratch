# 动作预算、迭代上限与成本治理器

> 一个中型电商 agent 的月度 LLM 成本，在团队启用 “order-tracking” skill 后从 1,200 美元跳到 4,800 美元。这不是定价 bug。这是 agent 找到了一个新循环，并在循环里持续花钱。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）把防御这一类问题的方法规范化：per-request `max_tokens`、per-task token 和 dollar budgets、per-day/month caps、iteration caps、tiered model routing、prompt caching、context windowing、昂贵动作上的 HITL checkpoints、budget breach 时的 kill switches。Anthropic 的 Claude Code Agent SDK 也用不同名称提供同样的 primitives。Financial velocity limits，例如 10 分钟内花费超过 50 美元就切断访问，比 monthly caps 更快抓住循环。

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## 问题

Autonomous agents 每一轮都会花真钱。Chatbot 的坏输出是一条坏回复；agent 的坏循环是一张账单。行业文档里把这种 failure mode 称为 “Denial of Wallet”：agent 不停 reasoning，不停调用 tools，不停计费，而什么都没有阻止它，因为系统一开始就没有设计阻止机制。

修复方式不是一个数字。它是一组分布在不同时间尺度和粒度上的 limits：per-request、per-task、per-hour、per-day、per-month。设计良好的 stack 会在几分钟内抓住 runaway loop，在几小时内抓住 slow leak，在一天内抓住 bad release。当 agent 是 long-horizon 且 autonomous 时，同一套 stack 才能让预算真正存在。

这是一堂工程课：数学很简单，团队失败的地方是纪律。下面列出的 limits 都在 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档中被命名过。

## 概念

### Cost-governor stack

1. **每个 request 的 `max_tokens`。** 简单。防止单次调用生成无限 completion。
2. **Per-task token budget。** 整个 run 内不能超过 N 个 tokens。达到 cap 就 hard stop。
3. **Per-task dollar budget。** 与 token budget 相同，但单位是货币。Claude Code 中是 `max_budget_usd`。
4. **Per-tool call cap。** `WebFetch` calls、`shell_exec` calls 等各自不能超过 N 次。
5. **Iteration cap（`max_turns`）。** Agent loop 的总迭代次数；防止 infinite reasoning loops。
6. **Per-minute / per-hour / per-day / per-month cap。** Rolling windows。在不同时间尺度上抓住 leaks。
7. **Financial velocity limit。** 例如“如果 10 分钟内花费超过 50 美元，就切断访问”。在 monthly caps 触发前抓住 loop-based burn。
8. **Tiered model routing。** 默认使用较小模型；只有当 classifier 判断任务确实需要时，才升级到更大模型。
9. **Prompt caching。** System prompt 和稳定 context 存在 provider cache 中；重复发送的 token 成本接近零。
10. **Context windowing。** 通过 compaction / summarization 让 active context 低于阈值；直接降低 token 成本。
11. **昂贵动作上的 HITL checkpoints。** 在已知昂贵的 action 前（长 tool call、大下载、昂贵模型升级），要求 human tap。
12. **Budget breach 时的 kill switch。** 任一 cap 触发时中止 session。记录触发的 cap；重新启用必须走独立路径。

### 为什么需要 stack，而不是一个 cap

单个 monthly cap 只会在钱包已经烧完后抓住 runaway agent。单个 per-request cap 抓不住 session 级别的问题。不同 failure modes 需要不同时间尺度：

- **Runaway loop**（agent 卡在 5 秒一次的 retry 中）：由 velocity limit 抓住。
- **Slow leak**（agent 每个 task 做了约 2x 预期工作）：由 daily cap 抓住。
- **Bad release**（新版本使用了 5x tokens）：由 weekly / monthly cap 抓住。
- **Legitimate surge**（真实需求增长，而不是 bug）：由 hour / day cap 抓住，并留下清晰日志。

### Claude Code 的 budget surface

Claude Code Agent SDK 在公开文档中暴露了：

- `max_turns`：iteration cap。
- `max_budget_usd`：dollar cap；breach 时中止 session。
- `allowed_tools` / `disallowed_tools`：tool allowlist 和 denylist。
- Tool use 前的 hook points，用于 custom cost-accounting。

把它和 permission-mode ladder（Lesson 10）结合起来。没有 `max_budget_usd` 的 `autoMode` session 就是未治理的 autonomy。Anthropic 明确把 Auto Mode 描述为需要 budget controls；classifier 与 cost 是正交问题。

### EU AI Act、OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 覆盖 OWASP Agentic Top 10 以及 EU AI Act Article 14（human oversight）要求。对于欧盟生产环境，logging 和 cap enforcement 不是可选项。

### 观察到的 1,200 美元到 4,800 美元案例

Microsoft 文档中的真实案例：一个电商 agent 在新增 tool 后月度成本翻了三倍。该 tool 允许 agent 在每个 session 中轮询订单状态。没有 loop detection。没有 per-tool cap。没有 week-over-week growth alert。修复方式是 per-tool cap 加 daily-growth alert。这是一个模板：每个新的 tool surface 都是一个新的潜在循环；每个新 tool 都需要自己的 cap 和自己的 alert。

## Use It

`code/main.py` 会模拟有 layered cost-governor stack 和没有它时的 agent run。模拟 agent 在若干 turns 后漂移进 polling loop；layered stack 会在 velocity window 内抓住它，而单个 monthly cap 可能要几天后才会触发。

## Ship It

`outputs/skill-agent-budget-audit.md` 会审计一个 proposed agent deployment 的 cost-governor stack，并标记缺失的 layers。

## 练习

1. 运行 `code/main.py`。确认在 polling-loop trajectory 上，velocity limit 会先于 iteration cap 触发。然后禁用 velocity limit，测量 agent 在 iteration cap 抓住它之前“花费”了多少。

2. 为 browser agent（Lesson 11）设计一组 per-tool caps。哪个 tool 需要最严格的 cap？哪个 tool 可以无界运行而没有风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出 toolkit 命名的每一种 cap type。把每一种映射到一种 failure mode（runaway loop、slow leak、bad release、surge）。

4. 为一次 realistic task 的 overnight unattended run 定价（例如“triage 一个 repo 中的 50 个 issues”）。把 `max_budget_usd` 设置为点估计的 2x。解释为什么是 2x。

5. Claude Code 的 `max_budget_usd` 会在 session aggregate cost 上触发。设计一个你会在外部强制执行的 complementary velocity limit。什么会触发 cut-off？重新启用是什么流程？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Denial of Wallet | “Runaway bill” | Agent loop 在没有 cap 阻止的情况下持续产生花费 |
| max_tokens | “Per-request cap” | 单次 completion 大小的上限 |
| max_turns | “Iteration cap” | 一个 session 中 agent loop iterations 的上限 |
| max_budget_usd | “Dollar kill switch” | Session cost cap；breach 时中止 |
| Velocity limit | “Rate cap” | 短窗口内的花费限制（例如 50 美元 / 10 分钟） |
| Tiered routing | “Small model first” | 默认用便宜模型；只有 classifier 认为值得时才升级 |
| Prompt caching | “Cached system prompt” | Provider-side cache 把重复发送 token 的成本降到接近零 |
| HITL checkpoint | “Human approval gate” | 昂贵 action 前需要 human tap |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、tool allowlists。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor checkpoints。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — provider-side cost controls。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — caching mechanics。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon agents 的 cost profile。
