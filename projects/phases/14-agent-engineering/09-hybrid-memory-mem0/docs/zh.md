# Hybrid Memory：Vector + Graph + KV（Mem0）

> Mem0（Chhikara et al., 2025）把 memory 看成三个并行 stores：vector 用于 semantic similarity，KV 用于快速 fact lookup，graph 用于 entity-relationship reasoning。Retrieval 时由一个 scoring layer 融合三路结果。这是 2026 年 external memory 的 production standard。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## 学习目标

- 解释为什么单一 store（只有 vector、只有 graph、只有 KV）不足以支撑 agent memory。
- 说出 Mem0 的三个并行 stores，以及每个 store 分别优化什么。
- 描述 Mem0 的 fusion scoring：relevance、importance、recency；并解释为什么它是 weighted sum，而不是层级规则。
- 使用 stdlib 实现一个玩具版 three-store memory：`add()` 会写入三种 stores，`search()` 会融合检索结果。

## 问题

对于三类 query，总有一类会让单一 store 失效：

- **Semantic similarity**： “what did we discuss about agent drift last week?” Vector 胜出；KV 和 graph 会漏掉。
- **Fact lookup**： “what is the user's phone number?” KV 胜出；vector 太浪费，graph 又过度设计。
- **Relationship reasoning**： “which customers share the same billing entity?” Graph 胜出；vector 和 KV 回答不了。

Production agents 会在同一个 session 中发出这三类 query。单一 store memory 总会对其中两类不合适。Mem0 的贡献，是把三种 stores 接到同一个 `add`/`search` surface 后面，并用一个 scoring function 融合它们。

## 概念

### 三个并行 stores

Mem0（arXiv:2504.19413, April 2025）在 `add(text, user_id, metadata)` 时：

1. 从文本中抽取 candidate facts（这是一个 LLM-driven step）。
2. 把每条 fact 写入 vector store（embedding），用于 semantic search。
3. 把每条 fact 按 `(user_id, fact_type, entity)` 写入 KV store，用于 O(1) lookup。
4. 把每条 fact 作为 typed edges 写入 graph store（Mem0g），用于 relationship queries。

在 `search(query, user_id)` 时：

1. Vector store 按 embedding cosine 返回 top-k。
2. KV store 按从 query 推导出的 `(user_id, type, entity)` 返回 direct hits。
3. Graph store 返回从 query entities 可达的 subgraph。
4. Scoring layer 融合三路结果。

### Fusion scoring

```text
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance**：vector cosine、KV exact match、graph path weight。
- **Importance**：在 write time 标记或通过学习得到（某些 facts 更重要：names、IDs、policies）。
- **Recency**：根据最近一次 write 或 read 以来经过的时间做 exponential decay。

Weights 会按 product 调整。Chat agents 使用更高的 `w_recency`；compliance agents 使用更高的 `w_importance`；retrieval agents 使用更高的 `w_relevance`。

### Mem0g 与 temporal reasoning

Mem0g 增加了 conflict detector。当一条新 fact 与现有 edge 矛盾时，现有 edge 会被标记为 invalid，但不会被删除。Temporal queries（例如 “what was the user's city in March?”）会遍历 valid-at-time subgraph。

这正是 Letta 的 invalidation pattern 所泛化出的 compliance-grade behavior。

### Benchmark numbers

Mem0 论文报告了如下 2025 年结果：

- **LoCoMo**（long-form conversation memory）：91.6
- **LongMemEval**（long-horizon episodic memory）：93.4
- **BEAM 1M**（1M-token memory benchmark）：64.1

对比 baselines（full-context 128k LLM、flat vector store、flat KV）都落后 10+ points。Benchmarks 本身不能单独证明选择合理，operational shape 才是关键；但这些数字说明 fusion design 不是一个微小优化。

### Scope taxonomy

Mem0 按 scope 拆分 memory：

- **User memory**：跨 sessions 持久化，以 `user_id` 为 key。
- **Session memory**：在单个 thread 内持久化。
- **Agent memory**：per-agent instance state。

每一次 write 都会选择一个 scope。Retrieval 可以跨 scopes 查询，并为每个 scope 设置权重。不加思考地混合 scopes，就是产生 “the assistant told Alice about Bob's project” 这类事故的方式。

### 这个模式会在哪里出错

- **Embedding drift.** 前一百个 queries 看起来正确的 vector results，会随着 corpus 增长而退化。需要定期重新 embedding top-N-used records。
- **KV schema creep.** `(user_id, type, entity)` 看起来很简单，直到每个团队都开始添加自己的 `type`。每季度审计一次 type set。
- **Graph explosion.** 一个 noisy extractor 每条 message 添加 50 条 edges。限制每次 `add` call 的 graph writes；丢弃 low-confidence edges。

## Build It

`code/main.py` 使用 stdlib 实现 three-store pattern：

- `VectorStore`：用朴素 token-overlap similarity 代替 embedding。
- `KVStore`：以 `(user_id, fact_type, entity)` 为 key 的 dict。
- `GraphStore`：typed edges（subject、relation、object、valid）。
- `Mem0`：顶层 facade，提供 `add()`、`search()`、fusion scoring 和 scope-aware retrieval。
- 一个 multi-user、multi-session conversation 的 worked trace。

运行：

```bash
python3 code/main.py
```

输出会展示三条独立 recall paths，以及融合后的 top-k。在 `main()` 顶部调整 scoring weights，观察 ranking 如何变化。

## Use It

- **Mem0 (Apache 2.0)**：production-ready。可用 Postgres + Qdrant + Neo4j self-host，也可以使用 managed cloud。
- **Letta**：three-tier core / recall / archival；自带或接入你自己的 vector 和 graph backends。
- **Zep**：带 temporal KG 和 fact extraction 的商业替代品。
- **Custom builds**：当你需要精确控制 extractor（compliance）或 fusion weights（recency 占主导的 voice agents）时使用。

## Ship It

`outputs/skill-hybrid-memory.md` 会生成一个 three-store memory scaffold，包含 fusion scorer、scope taxonomy 和 temporal invalidation。

## 练习

1. 用真实 embedding model（sentence-transformers、Ollama、OpenAI embeddings）替换玩具版 vector similarity。在一段 synthetic long conversation 上测量 recall@10。经过 1000 次 writes 后，ranking 会 drift 吗？
2. 添加 temporal query：`search(query, as_of=timestamp)`。只返回在该时间点或之前 valid 的 records。哪一个 store 需要最多改动？
3. 实现 conflict detector：如果 incoming fact 与一条 graph edge 矛盾，就 invalidate 旧 edge，并记录新旧两条。用 “user lives in Berlin” -> “user lives in Lisbon” 测试。
4. 扩展 fusion scorer，加入 `user_feedback` 维度（对 retrieved records 点 thumbs-up）。你如何防止 gaming（agent 只返回它已经 liked 的 records）？
5. 阅读 Mem0 docs（`docs.mem0.ai`）。把玩具 demo 移植成 `mem0` client calls。用相同的 20 个 test queries 比较 retrieval quality。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Hybrid memory | “Vector plus graph plus KV” | 三个 stores 并行写入，在 retrieval 时融合 |
| Fact extraction | “Memory ingestion” | 把文本拆成 (entity, relation, fact) tuples 的 LLM step |
| Fusion scoring | “Relevance ranking” | Relevance、importance、recency 的 weighted sum |
| Scope | “Memory namespace” | user / session / agent；决定谁能看到什么 |
| Mem0g | “Memory graph” | 带 temporal validity 的 typed edges，用于 relationship queries |
| Temporal invalidation | “Soft delete” | 把矛盾 edges 标记为 invalid；永不删除 |
| Embedding drift | “Retrieval rot” | 随着 corpus 增长，vector quality 会退化；需要定期 re-embed |

## 延伸阅读

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — original paper
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — production API、SDKs、managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — virtual-context predecessor
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — three-tier sibling design
