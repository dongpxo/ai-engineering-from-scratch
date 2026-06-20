# Prompt Caching 与 Semantic Caching 经济学

> **价格快照日期：2026-04。** 下面的数字说法来自本课发布时抓取的供应商价格表；在下游引用前，请先用链接文档核对最新价格。

> 缓存发生在两层。L2（provider-level）prompt/prefix caching 会为重复 prefix 复用 attention KV。Anthropic 的 prompt-caching 文档宣称，长 prompt 上最多可降低 90% 成本、降低 85% latency；对 Claude 3.5 Sonnet 而言，cache read 是 $0.30/M，fresh input 是 $3.00/M，5-minute TTL，1-hour TTL 选项有 2x write premium（docs.anthropic.com，2026-04）。OpenAI prompt caching 会自动应用于 ≥1024 tokens 的 prompt，并把 cached input 定价为相对 fresh 约 90% 折扣（platform.openai.com，2026-04）；精确的 per-model cached rate 取决于实时价格表。L1（app-level）semantic caching 会在 embedding similarity 命中时完全跳过 LLM。供应商说的 “95% accuracy” 指的是匹配正确性，不是 hit rate；生产 reported hit rates 从 10%（open-ended chat）到 70%（structured FAQ）不等。两家 provider 都没有发布官方 baseline，所以这些数字应当视为 community telemetry，而不是保证。生产陷阱是：parallelization 会杀掉 caching（第一次 cache write 完成前发出的 N 个并行请求会让花费膨胀数倍），prefix 内的 dynamic content 会彻底阻止 cache hit。ProjectDiscovery 通过把 dynamic text 移出 cacheable prefix，把 hit rate 从 7% 提升到 74%（2025-11）。

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 区分 L2 prompt/prefix caching（provider 侧 KV 复用）与 L1 semantic caching（相似 prompt 命中时绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记，以及两种 TTL 选项（5-minute 与 1-hour）的价格倍数。
- 在给定 hit rate、prompt/response mix 和 token prices 时，计算预期月度节省。
- 说出会让账单膨胀 5-10x 的 parallelization 反模式，以及会让 hit rate 崩掉的 dynamic-content 反模式。

## 问题

你给 RAG 服务加了 prompt caching。账单没有下降。你测量 hit rate，只有 7%。你的 prompts 看起来是静态的，但其实不是：system prompt 里包含了精确到分钟的当前日期、request ID，以及为了多样性而随机重排的示例。每个请求都写入一个新的 cache entry，读取次数为零。

另一个问题是，你的 agent 会为每个用户问题并行运行十个 tool calls。十个请求都在第一次 cache write 完成前到达 provider。十次写入，零次读取。你的账单是“开启 caching 后”本该出现的成本的 5-10 倍。

Caching 是 protocol，不是 flag。两层缓存，两种不同 failure modes。

## 概念

### L2 — provider prompt/prefix caching

Provider 会为可缓存 prefix 存储 attention KV，并在下一个匹配相同 prefix 的请求中复用它。你只为 write 支付一次成本，read 近乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 series）**：请求中使用显式 `cache_control` marker。你标记哪些 blocks 可缓存。TTL：5-minute（write cost 是 base 的 1.25x）或 1-hour（write cost 是 base 的 2x）。Cache reads：Claude 3.5 Sonnet 上是 $0.30/M，而 fresh 是 $3.00/M，即便宜 10x（docs.anthropic.com，截至 2026-04）。不同模型（Opus/Haiku）价格不同；始终交叉核对实时 pricing page。

**OpenAI**：对 ≥1024 tokens 的 prompts 自动缓存（platform.openai.com，2026-04）。没有显式 flag。按当前 gpt-4o/gpt-5 rate cards，cached input 大约比 fresh 便宜 10x。Docs 和 release notes 都没有发布官方 hit-rate baseline；community reports 在精心设计 prompt 时大多集中在 30-60%。监控 `usage.cached_tokens` 来测你自己的结果。

**Google（Gemini）**：通过显式 API 做 context caching；1M-token context 让 caching 的收益更大。

**Self-hosted（vLLM、SGLang）**：Phase 17 · 06 讲 RadixAttention，也就是在你自己的 compute 上应用同一种模式。

### L1 — app-level semantic caching

在完全调用 LLM 前，先 hash prompt、embed prompt，然后查找相似 cached request（cosine similarity 高于阈值，通常 0.95+）。命中时返回 cached response。未命中时调用 LLM，并缓存结果。

Open-source：Redis Vector Similarity、GPTCache、Qdrant。Commercial：Portkey Cache、Helicone Cache。

供应商的 accuracy claims 指的是返回的 cached response 在语义上合适的频率，而不是命中频率。生产 hit rates：

- Open-ended chat：10-15%。
- Structured FAQ / support：40-70%。
- Code questions：20-30%（小变体会杀掉 hits）。
- Voice agents repeating prompts：50-80%（voice normalization 固定了集合）。

### Parallelization 反模式

你的 agent 并行发起 10 个 tool calls。十个请求都有相同的 4K-token system prompt。Anthropic cache writes 是 per-request 的；第一次 cache-write 大约在 provider 看到 prompt 后 300 ms 完成。请求 2-10 在同一个毫秒窗口到达，每个都会看到 cache miss。你支付 10 次 write premium，得到 0 次 read discount。

修复方式：batch with sequential-first。先单独发出 request 1，等 1 的 cache 填充后再发出 2-10。第一个 tool call 增加 300 ms；账单节省 5-10x。

### Dynamic content 反模式

你的 system prompt 看起来像：

```text
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复方式：把真正静态的内容移到 cacheable prefix，把 dynamic content 追加到 cache boundary 之后：

```text
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 通过这种方式把 cache hit rate 从 7% 提升到 74%，并公开了拆解过程。

### 为 overnight workloads 叠加 batch + cache

Batch APIs（Phase 17 · 15）在 24-hour turnaround 下给 50% 折扣。Cached input 还能在此基础上再获得约 10x。Overnight classification、labeling、report generation 这类 workloads 可以通过叠加 batch 和 cache，把成本降到 synchronous-uncached cost 的约 10%。

### 你应该记住的数字

价格点来自 2026-04 抓取的供应商文档，并且每几个月都会漂移；依赖它们前要重新检查。

- Anthropic cached read：Claude 3.5 Sonnet 上 $0.30/M，约比 fresh input 便宜 10x（docs.anthropic.com）。
- Anthropic cache write premium：1.25x（5-min TTL）或 2x（1-hour TTL）。
- OpenAI auto-cache：适用于 ≥1024 tokens 的 prompts；cached input 在当前 rate cards 上约为 fresh input 的 10%（platform.openai.com）。
- Semantic cache hit rate（community-reported）：open chat 约 10%；structured FAQ 最高约 70%。不是供应商文档化 baseline。
- ProjectDiscovery：通过把 dynamic 内容移出 prefix，hit rate 从 7% 提升到 74%（project blog，2025-11）。
- Parallelization 反模式：当 N 个并行请求错过第一次 cache write 时，常见报告显示账单膨胀 5-10x。

## Use It

`code/main.py` 会在混合 workload 上模拟 L1 + L2 caching。它会报告 hit rates、bill，并展示 parallelization penalty。

## Ship It

本课产出 `outputs/skill-cache-auditor.md`。给定 prompt template 和 traffic，它会审计 cacheability，并建议如何重构。

## 练习

1. 运行 `code/main.py`。切换 parallelization flag。账单变化有多大？
2. 你的 system prompt 里有日期。把它移出去。展示 before/after hit rate 数学。
3. 在给定 request arrival rate 时，计算 1-hour TTL（2x write）相对 5-minute TTL（1.25x write）的 break-even。
4. Semantic cache 在 0.95 threshold 下命中 20%。在 0.85 时命中 50%，但你看到错误 cached responses。选择正确 threshold 并说明理由。
5. 你为每个用户问题 batch 10 个 parallel sub-queries。改写成 cache-friendly 的形式，同时不增加 end-to-end latency。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| L2 prompt cache | “prefix cache” | Provider 为重复 prefix 存储 KV |
| `cache_control` | “Anthropic cache marker” | 标记 cacheable blocks 的显式属性 |
| Cache write premium | “write tax” | 第一次 miss-to-cache 的额外成本（1.25x 或 2x） |
| L1 semantic cache | “embedding cache” | 调用 LLM 前在 app 层做 hash-and-embed |
| GPTCache | “LLM caching lib” | 流行的 OSS L1 cache library |
| Cache hit rate | “hits / total” | 由 cache 服务的请求比例 |
| Parallelization anti-pattern | “the N-write trap” | N 个并行请求会 miss cache N 次 |
| Dynamic content trap | “the time-in-prompt trap” | Prefix 中的 dynamic bytes 会杀掉 hit rate |
| RadixAttention | “intra-replica cache” | SGLang 的 prefix-cache 实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义和 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为和 eligibility。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
