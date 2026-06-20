# 面向 LLM 生产环境的混沌工程

> 到 2026 年，面向 LLM 的 chaos engineering 已经成了独立实践。把 experiments 跑进 production 之前，必须先有这些前提：定义好的 SLI/SLO、trace+metric+log observability、automated rollback、runbooks、on-call。架构有四个 planes：control（experiment scheduler）、target（services、infra、data stores）、safety（guards + abort + traffic filters）、observability（metrics + traces + logs），以及进入 SLO adjustments 的 feedback。Guardrails 是必需的：如果 daily error-budget burn > expected 的 2x，burn-rate alerts 会暂停 experiments；suppression windows + trace-ID correlation 会去重 alert noise。Cadence：每周 small canary + SLO review；每月 game day + postmortem；每季度 cross-team resilience audit + dependency mapping。LLM-specific experiments 包括：memory overload、network failures、provider outages、malformed prompts、KV cache eviction storms。Tooling：Harness Chaos Engineering（LLM-derived recommendations、blast-radius downscaling、MCP tool integration）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes-native）。

**Type:** Learn
**Languages:** Python（stdlib，玩具版 chaos experiment runner）
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 说出五个 chaos engineering prerequisites（SLI/SLO、observability、rollback、runbooks、on-call），并解释为什么跳过任何一个都会破坏这项实践。
- 画出四个 planes（control、target、safety、observability）以及反馈到 SLO 的 feedback loop。
- 列出五个 LLM-specific experiments（memory overload、network fail、provider outage、malformed prompt、KV eviction storm）。
- 根据 stack 选择工具：Harness、LitmusChaos 或 Chaos Mesh。

## 问题

传统 stack 里的 chaos testing 已经相对成熟。LLM stack 会增加新的 failure modes。一个带 poison character 的 4K-token prompt 会让 tokenizer 卡 12 秒。上游 provider 返回 429；你的 gateway 重试；service 因 retry-amplified concurrency 而 OOM。Burst load 下的 KV cache eviction storm 会触发 re-prefill cascades，把 compute 打满。

这些都不会出现在 unit tests 里。Chaos engineering 就是你在用户发现之前先发现它们的方法。

## 概念

### Prerequisites

没有以下条件，不要在 production 里跑 chaos：

1. **SLI/SLO** — 已定义的 service-level indicators 和 objectives。
2. **Observability** — traces、metrics、logs，并接到 dashboards。
3. **Automated rollback** — Phase 17 · 20 的 policy-flag rollback。
4. **Runbooks** — 结构化 runbooks，见 Phase 17 · 23。
5. **On-call** — 有人负责响应。

缺任何一个，chaos 就会变成真实 incident。

### Four planes + feedback

**Control plane** — experiment scheduler（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target plane** — services、pods、nodes、load balancers、data stores。

**Safety plane** — kill switch、suppression windows、blast-radius limits、error-budget gates。

**Observability plane** — 正常 metrics + trace-ID correlation，用来区分 chaos-induced failures 和 natural failures。

**Feedback loop** — findings 会回流到 SLO adjustment、runbook updates 和 code fixes。

### Guardrails 是必需的

- **Burn-rate alert**：如果 daily error-budget burn 超过 expected 的 2x，就暂停 experiment。
- **Suppression windows**：在 experiment 期间，静默 blast radius 内的 non-experiment alerts。
- **Trace-ID correlation**：所有 experiment-induced errors 都携带 tag，让 on-call 可以 dedupe。

### 五个 LLM-specific experiments

1. **Memory overload** — 用 high-concurrency long-context requests 强制制造 KV cache preemption storm。观察：service 是 graceful shed，还是直接 crash？

2. **Network failure** — 切断 inference gateway 和 provider 之间的连接。观察：fallback 是否在 SLA 内启动？（Phase 17 · 19）

3. **Provider outage simulation** — 模拟 OpenAI 100% 返回 429。观察：routing 是否 failover 到 Anthropic？（Phase 17 · 16, 19）

4. **Malformed prompt** — 注入 tokenizer-stalling payload（例如深层嵌套 unicode、巨大 UTF-8 codepoint）。观察：单个 request 是否会锁住一个 worker？

5. **KV eviction storm** — 通过打满 vLLM block budget 强制 eviction。观察：LMCache 能否恢复，还是 service 会 degrade？

### Cadence

- **Weekly** — staging 中的小 canary experiments，可能加上 5% prod。
- **Monthly** — 针对特定 scenario 的 scheduled game day；跨团队参与；postmortem。
- **Quarterly** — cross-team resilience audit；更新 dependency map。

### Tooling

- **Harness Chaos Engineering** — commercial；AI-derived experiment recommendations；blast-radius downscaling；MCP tool integration。
- **LitmusChaos** — CNCF graduated；基于 Kubernetes workflow。
- **Chaos Mesh** — CNCF sandbox；Kubernetes-native CRD style。
- **Gremlin** — commercial；支持面广。
- **AWS FIS** / **Azure Chaos Studio** — managed cloud offerings。

### Starting small

第一个 experiment：在 steady traffic 下 pod-kill 一个 decode replica。观察 rerouting 和 recovery。如果这一步安全且表现正常，再升级到 network chaos。

第一个 LLM-specific experiment：注入一个 provider 429，持续 5 分钟。观察 fallback。大多数团队会发现 fallback 并没有被真正测过。

### 你应该记住的数字

- Four planes：control、target、safety、observability。
- Burn-rate pause：expected daily budget burn 的 2x。
- Cadence：weekly canary、monthly game day、quarterly audit。
- Five LLM experiments：memory、network、provider、malformed prompt、KV storm。

## Use It

`code/main.py` 会模拟三个带 safety plane gates 的 chaos experiments。它会报告哪些 experiments 会触发 burn-rate abort。

## Ship It

本课产出 `outputs/skill-chaos-plan.md`。给定 stack 和 maturity，它会选择前三个 experiments 和对应 tooling。

## 练习

1. 运行 `code/main.py`。哪个 experiment 会触发 burn-rate gate？为什么？
2. 为一个基于 vLLM 的 RAG service 设计前五个 chaos experiments。包含 success criteria。
3. 你的 burn-rate alert 暂停了一个 experiment。你如何判断 root cause 是 chaos 还是 natural？
4. 论证 chaos 应该跑在 production 还是只跑在 staging。什么时候 production 才是正确答案？
5. 说出三个 generic network-chaos 无法复现的 LLM-specific failure modes。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| SLI / SLO | “service targets” | Indicator + objective；必需 prerequisite。 |
| Blast radius | “scope” | Experiment 影响到的 services / users 集合。 |
| Burn-rate alert | “budget gate” | 当 error-budget burn rate > expected 的 2x 时触发。 |
| Game day | “monthly drill” | 定期跨团队 chaos exercise。 |
| LitmusChaos | “CNCF workflow” | Graduated CNCF Kubernetes chaos tool。 |
| Chaos Mesh | “CNCF CRD” | CNCF sandbox Kubernetes-native chaos。 |
| Harness CE | “commercial AI-assisted” | 带 AI recommendations 的 Harness chaos。 |
| Malformed prompt | “tokenizer bomb” | 会卡住 tokenization 的 input。 |
| KV eviction storm | “preemption cascade” | 大规模 eviction 触发 re-prefills。 |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
