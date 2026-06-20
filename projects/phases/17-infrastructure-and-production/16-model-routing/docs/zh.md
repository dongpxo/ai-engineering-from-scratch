# 模型路由作为降本原语

> 一个 dynamic broker 会评估每个 request（task type、token length、embedding similarity、confidence），把简单查询发给 cheap model，把复杂查询升级到 frontier model。这也叫 model cascading。生产案例显示，在美国 / 英国 / 欧盟部署中，iso-quality 下可节省 20-60% 成本；高流量 SaaS 上 30% 的 routing efficiency improvement 会变成六位数年度节省。2026 年的背景是 LLM inference prices 每年下降约 10x——从 2022 年末到 2026 年，一个 GPT-4-class token 从 $20/M 降到约 $0.40/M。大部分下降来自更好的 serving stacks（Phase 17 · 04-09），不是硬件。Routing 是你把这轮价格下降转化为 margin、同时不让 product regression 的方法。失败模式是 cheap-model drift：route 把 40% 流量推给较弱模型，reasoning tasks 上 quality 掉 3-5%，一个季度都没人发现。Routes 要用 online quality metrics gate，而不是只靠 offline eval sets。

**Type:** Learn
**Languages:** Python（stdlib，toy cascading router simulator）
**Prerequisites:** Phase 17 · 01（Managed LLM Platforms），Phase 17 · 19（AI Gateways）
**Time:** ~60 minutes

## 学习目标

- 解释 model cascading：cheap-first with confidence check，low confidence 时 escalate。
- 枚举四个 routing signals（task classification、prompt length、embedding similarity to known-hard set、self-confidence from first-pass）。
- 在目标 routing split 和 quality loss tolerance 下，计算 expected blended cost。
- 说出能抓住 cheap-model creep 的 drift-monitoring metric（online quality gate）。

## 问题

你的 service 在 GPT-5 上每月花 $80k。Analytics 显示，70% queries 很简单：“what time is it in Paris?”、“rephrase this sentence.” 一个 Haiku-class model 能以 3% 的成本完美处理这些请求。剩下 30% 需要 GPT-5 的 reasoning——coding、math、multi-step planning。

如果你把 70% route 到 cheap model，30% route 到 expensive model，账单会在同等 product quality 下下降约 65%。这就是 routing。难点是构建 broker 时不让质量退化。

## 概念

### 四个 routing signals

1. **Task classification**：simple/complex/codegen/math/chat。可以是 rules-based classifier、小 LLM（Haiku-class at $0.25/M），或到 labeled buckets 的 embedding similarity。输出：route = cheap / balanced / frontier。

2. **Prompt length**：prompts >4K tokens 通常需要 frontier 来保持 coherence。prompts <500 tokens 通常不需要。

3. **Embedding similarity to known-hard set**：如果 query 接近一个 known-hard bucket（cosine > 0.88），直接 escalate 到 frontier。

4. **Self-confidence from first-pass**：先发给 cheap；如果 model 的 log-probs 显示 low confidence，或它拒答，或输出 hedging language，就在 frontier 上 retry。这会给约 10% traffic 增加 P95 latency，但在另外 90% traffic 上节省 50%+。

### 三种 patterns

**Pre-route**（classifier up front）：增加约 5-10ms latency；整体最快。

**Cascade**（cheap-first，low confidence 时 escalate）：median latency 约 1.2x（cheap run plus verify），被 escalated 的请求约 2x。质量底线最好。

**Ensemble route**（对一部分样本并行运行 cheap 和 frontier，再用 reward-model pick）：质量最高，成本最高；只用于 critical A/B。

### 实现方式

AI gateways（Phase 17 · 19）会暴露 routing。LiteLLM 有带 fallback 和 cost-routing 的 `router` config。Portkey 有 guards + routing。Kong AI Gateway 有 plugin-based routing。OpenRouter 的 model marketplace 暴露 recommendation API。

Open-source：RouteLLM（LMSYS）、Not Diamond（commercial）、Prompt Mule。

### 2026 price curve

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier（GPT-5, Claude 4） | — | ~$3-10/M | new tier |

大多数改进来自 serving efficiency——Phase 17 · 04-09 的核心课程转化成 provider-side cost drops。Routing 让你在 app layer 捕获这些收益，而不是等待所有用户都迁移到 cheap tier。

### Drift 才是真风险

你的 route 把 40% 流量发给 cheap model。六个月后，task distribution 变化了（用户变得更复杂，问题更长）。Router 没有发现，因为它的 classifier 是用 Q1 data 训练的。Quality 静默下降。没人足够大声地抱怨。直到 competitor benchmark 中你输了，才发现问题。

用 online quality metrics gate routes：

- 每条 route 上的 user thumbs-up / thumbs-down。
- 每条 route 上对 held-out sample（5%）做 automated LLM-judge。
- Escalation rate：如果 cascade up-route >30%，cheap model 被 over-routed。
- 每条 route 的 refusal rate。

### 你应该记住的数字

- 2026 routing savings at iso-quality：案例研究中为 20-60%。
- 2022-2026 LLM price drop：整体约每年 10x。
- GPT-4-level 2022 vs 2026：~$20/M → ~$0.40/M。
- Cascade latency impact：median 约 1.2x，escalated 约 2x（约 10% traffic）。

## Use It

`code/main.py` 会在 mixed workload 上模拟 pre-route、cascade 和 ensemble。它会报告 blended cost、quality loss 和 escalation rate。

## Ship It

本课产出 `outputs/skill-router-plan.md`。给定 workload 和 quality budget，它会选择 routing pattern 和 signals。

## 练习

1. 运行 `code/main.py`。在什么 accuracy floor 下 cascade 会击败 pre-route？
2. 你的 user base 是 30% enterprise（complex queries）、70% free tier（simple）。设计 routing split。用什么 online metric gate 它？
3. 一条 route 让 quality 掉 2%，但节省 40%。能 ship 吗？取决于 product——两边都论证一下。
4. 使用 OpenAI / Anthropic APIs 的 logprobs 实现 confidence check。你会从哪个 threshold 开始？
5. 六个月里，escalation rate 从 8% 爬到 22%。诊断三个原因，并给出每个原因的 fix。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | “cost broker” | 每个 request 动态选择 model |
| Model cascade | “cheap-first escalate” | 先跑 cheap，low confidence 时 fall through 到 frontier |
| Pre-route | “classify first” | 前置 classifier；不 re-run |
| Ensemble route | “parallel pick” | 运行多个模型，由 reward-model 选择最佳 |
| Escalation rate | “uprouted %” | Cascade requests 中被 escalated 的比例 |
| RouteLLM | “LMSYS router” | OSS router library |
| Not Diamond | “commercial router” | SaaS model-routing product |
| Drift | “cheap creep” | Distribution shift 发生但 router 没发现 |
| Online quality gate | “live check” | 对 live traffic 做 automated LLM-judge sampling |

## 延伸阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives。
