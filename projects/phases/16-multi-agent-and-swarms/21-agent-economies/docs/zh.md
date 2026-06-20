# Agent 经济、Token 激励与 Reputation

> 长程 autonomous agents（METR 的 1-hour 到 8-hour work-curve）需要经济能动性。正在形成的 **5-layer stack** 是：**DePIN**（physical compute）→ **Identity**（W3C DIDs + reputation capital）→ **Cognition**（RAG + MCP）→ **Settlement**（account abstraction）→ **Governance**（Agentic DAOs）。Production agent-incentive networks 包括 **Bittensor**（TAO subnets 奖励 task-specific models）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET token）和 **Gonka**（transformer-based PoW，把 compute 重新分配给 productive AI tasks）。学术工作包括：AAMAS 2025 的 decentralized LaMAS 使用 **Shapley-value credit attribution** 来公平奖励贡献 agents；Google Research 的 “Mechanism design for large language models” 提出带 second-price payment 的 **token auctions**，适用于 monotone aggregation。本课会构建一个最小 agent marketplace，把 Shapley-value credit attribution 应用到 multi-agent pipeline，并运行一个 second-price token auction，让 game-theory machinery 落到具体代码上。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

当 agents 共同产出价值、但需要分别获得奖励时，multi-agent systems 会变复杂。经典机制——equal split、last-contributor-takes-all——要么不公平，要么容易被 game。通过 Shapley values 做 coalition-based rewarding 在构造上公平，但计算成本高。2025-2026 年的文献推进了一些有用近似：Shapley sampling、monotone aggregation auctions，以及从 confirmed contributions 中累积的 on-chain reputation。

除了 credit attribution，领域也开始转向真正的 economic agents：Bittensor TAO 奖励 mining compute 来 fine-tune subnet-specific models；Fetch.ai/ASI 用 FET tokens 奖励 ASI-1 Mini LLM usage；Gonka 把 transformer proof-of-work 重新分配给 productive AI tasks。能自主交易的 agents 今天已经存在；问题是如何对齐 incentives。

本课把 agent economies 当作一个具体 problem family：credit attribution、mechanism design 和 reputation，并用最小数学构建每个部分，让这些想法真正落地。

## 概念

### 5-layer agent-economy stack

1. **DePIN（physical compute）。** 出租 GPU、storage、bandwidth 的 decentralized infrastructure。Bittensor subnets、Render Network、Akash。它不是 agent-specific；agents 使用它。
2. **Identity。** W3C Decentralized Identifiers（DIDs）给每个 agent 一个独立于平台的 durable ID。Reputation 累积到 DID 上。Agent Network Protocol（ANP）使用 DID 作为 discovery layer。
3. **Cognition。** Agent 的 reasoning loop：LLM + RAG + MCP。这是其他阶段构建的东西。
4. **Settlement。** Account abstraction（ERC-4337）让 agents 可以从自己的 balances 里支付 gas，而不需要持有 ETH。Agents 可以为 services、彼此或 compute 付费。
5. **Governance。** Agentic DAOs：humans *and* agents 都能对 protocol changes 投票，voting power 绑定到 reputation。

不是每个 production system 都使用全部五层。Bittensor 使用 1、2，部分使用 3 和 4，不使用 5。OpenAI agents 除了 3 之外不使用其他层。这个 stack 是参考地图，不是要求。

### Bittensor、Fetch.ai、Gonka：实际运行的是什么

**Bittensor（TAO）。** Subnets 是专门任务（language modeling、image generation、forecasting）。Miners 提交 model outputs。Validators 排名这些 outputs；stake-weighted scoring 分配 TAO rewards。每个 subnet 都有自己的 evaluation。经济学 lesson：为 task-specific output quality 付费，而不是为使用的 compute 付费。

**Fetch.ai / ASI Alliance。** ASI-1 Mini LLM 运行在 Fetch.ai network 上；users 用 FET tokens 支付 inference。这里 agents-as-peers 的叙事更强：Fetch 上的一个 agent 可以调用另一个 agent 完成任务，并用 FET 支付。

**Gonka。** Transformer proof-of-work：“work” 是 transformer 的 forward passes。Miners 通过运行 inference tasks 获得收益，这些 tasks 有已知正确 outputs（来自 training data）。这是 resource-productive PoW，而不是 hash-based PoW。

截至 2026 年 4 月，这三者都是 production-grade。Payoff distribution 不同。Bittensor 奖励相对于 subnet validators 的 quality；Fetch 奖励 paying users 衡量出的 utility；Gonka 奖励 verifiable inference work。

### Shapley-value credit attribution

三个 agents 协作完成一个 task。output score 是 0.8。谁贡献了多少？

Shapley value：满足四条公理（efficiency、symmetry、linearity、null）的唯一 credit allocation。对于 agent `i`：

```text
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是 ordering `O` 中位于 `i` 前面的 agents 集合。实践上：枚举所有 permutations，记录每个 agent 在每个 permutation 中的 marginal contribution，然后取平均。

对于 N=3 agents，有 6 个 permutations。对于 N=10，有 3.6M 个，所以实践中会 sample orderings，而不是全部枚举。

### Aggregation 的 second-price auction

Google Research（“Mechanism design for large language models”）提出用 second-price token auctions 聚合 LLM outputs。设置是：N 个 agents 各自提出一个 completion；每个 agent 对被选中有一个 private value。auctioneer 选择 highest-value proposal，并支付 *second-highest* value。在 monotone aggregation 下（value 取决于哪个 proposal 被选中，而不是投了多少 bids），这是 truthful 的：agents 会 bid 自己的 true value。

这对 LLM systems 很重要：你可以把 completion tasks 外包给多个价格不同的 agents；auction 会选择最好的并公平支付，而且 agents 没有 incentive 去 misreport。

### Reputation capital

一个 DID-bound reputation score 会从 confirmed contributions 中累积。一个简单 update rule：

```text
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

其中 decay factor `alpha` 接近 1。Reputation：

- 对 routing decisions 来说很便宜（“把 hard tasks 发给 high-rep agents”）。
- 伪造成本高（随时间累积，并绑定到 DID）。
- 可以被 slashed：verification 失败的 contributions 会扣分。

### AAMAS 2025 decentralized LaMAS

LaMAS proposal（AAMAS 2025）结合了 DID identity、Shapley-value credit attribution 和一个简单 auction mechanism。核心主张：把 credit attribution step decentralized，会让系统可审计，并免疫 single-point manipulation。

### Economics 会在哪里崩掉

- **Price oracle manipulation。** 如果 credit function 可以被 game，agents 就会 game 它。每个 mechanism 都需要 adversarial test。
- **Sybil attacks。** 一个 operator 启动 N 个 fake agents 来膨胀自己的 contribution。DIDs 会减速但无法阻止它；reputation cost-to-forge 才是 mitigation。
- **Verification cost。** Credit attribution 的公平性只和 verifier 一样强。如果 verification 便宜（小 LLM），它可以被 game；如果昂贵（human panel），系统就无法 scale。
- **Regulatory overhang。** Agent economies 会与 financial regulation 交叉。截至 2026 年，Bittensor、Fetch 和 Gonka 在一些 jurisdictions 中都处于法律灰区。

### Agent economies 什么时候有意义

- **有异质 operators 的 open networks。** 没有单个 team 控制全部 agents。
- **Verifiable outputs。** 没有 verification，credit attribution 就只是猜测。
- **Long-horizon workflows。** One-shot tasks 不能从 reputation accumulation 中获得太多收益。
- **Tokenized payments 在你的 jurisdiction 中 legally viable。**

在封闭 corporate systems 中，economics 会让位于更简单的 allocation（managers 分配工作，metrics 是内部的）。经济学文献主要适用于 open networks。

## Build It

`code/main.py` 实现：

- `shapley(value_fn, agents)`：小 N 下通过枚举做 exact Shapley computation。
- `second_price_auction(bids)`：truthful mechanism；winner 支付 second-highest。
- `Reputation`：带 exponential decay 和 slashing 的 DID-bound reputation。
- Demo 1：三个 agents 协作，exact Shapley 分配 credit。
- Demo 2：五个 agents 为一个 task slot 竞价；second-price auction 选出 winner + payment。
- Demo 3：100 轮 task assignment，agents 有异质 rep；warmup 后，rep-weighted routing 优于 random。

运行：

```bash
python3 code/main.py
```

Expected output：每个 agent 的 Shapley values；展示 truthful-bid equilibrium 的 auction result；rep-weighted routing 在 warmup 后比 random 多 10-20% quality gain。

## Use It

`outputs/skill-economy-designer.md` 会设计一个最小 agent economy：选择 identity layer、credit attribution mechanism、payment mechanism 和 reputation rule。

## Ship It

在 2026 年运行 agent economy：

- **从 reputation 开始，而不是 tokens。** Reputation 实现便宜，并且单独就有价值；tokens 会增加 legal 和 economic complexity。
- **Reward 前先 verify。** 永远不要在没有 independent verification step 时分配 credit。Self-reported quality 会累积 sybil games。
- **Shapley-sample，不要 Shapley-exact。** Sample 100-1000 个 orderings；exact enumeration 不可扩展。
- **限制 decay factor，并给 reputation 设 floor。** Unbounded decay 会抹掉 legitimate contributors；too-slow decay 会奖励 stale high-rep agents。
- **对 mechanisms 做 adversarial audit。** 开放 network 前先运行 red-team scenarios。每个 mechanism 都有 game theory；你想先找到漏洞，而不是让 attackers 找到。

## 练习

1. 运行 `code/main.py`。确认 Shapley values 加总等于 total value（efficiency axiom）。修改 value function；Shapley allocations 是否按预期方向变化？
2. 实现 Shapley *sampling*（对 K 个 orderings 做 Monte Carlo）。K 如何影响 approximation accuracy？对 N=4 与 exact 结果对比。
3. 在 auction 前实现 coalition-forming step：agents 可以合并成 teams，并作为一个 unit bid。哪些 coalitions 会形成？结果是否比 individual bidding 更 Pareto-better？
4. 阅读 Google Research mechanism-design post。找出一个一旦被违反就会破坏 truthfulness 的 assumption。这个 failure mode 在 LLM setting 中是什么样子？
5. 阅读 AAMAS 2025 decentralized LaMAS paper。在 synthetic task 上为 10 个 agents 实现它们的 Shapley step。Exact computation 需要多久？100 draws 的 sampling 能有多接近？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| DePIN | “Decentralized physical infrastructure” | Token-incentivized compute/storage/bandwidth。Bittensor、Akash、Render。 |
| DID | “Decentralized identifier” | W3C portable IDs spec。Agent reputation 绑定到 DID，而不是平台。 |
| ERC-4337 | “Account abstraction” | 可以 sponsor gas 的 contract accounts，使 agent payments 成为可能。 |
| Shapley value | “Fair credit attribution” | 满足 efficiency、symmetry、linearity、null 的唯一 allocation。 |
| Second-price auction | “Vickrey auction” | Truthful mechanism：winner 支付 second-highest bid。兼容 monotone aggregation。 |
| Reputation capital | “Accumulated quality score” | 从 confirmed contributions 得来的 DID-bound score；随时间 decay。 |
| Agentic DAO | “Agents + humans govern” | 把 agent voters 作为 first-class 的 DAO，voting power 绑定 reputation。 |
| TAO / FET / GPU credits | “Token denominations” | Bittensor TAO、Fetch.ai FET，以及各种 DePIN tokens。 |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 年关于 5-layer agent-economy stack 的 survey
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 带 monotone aggregation 的 token auctions
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value credit attribution
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet structure 和 reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM 和 FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — identity foundation
