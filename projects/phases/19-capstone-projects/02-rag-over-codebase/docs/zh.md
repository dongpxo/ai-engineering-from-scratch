# Capstone 02 — 基于代码库的 RAG（跨仓库语义搜索）

> 到 2026 年，每个严肃的工程组织都会运行一个 internal code search：它理解 meaning，而不只是 strings。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的 enterprise graph、Aider 的 repomap、Pinterest 的 internal MCP——形态都相同。Ingest many repos，用 tree-sitter parse，按 function- 和 class-level chunks embedding，hybrid-search，re-rank，并带 citations 回答。本 capstone 要求你构建一个系统：它能处理跨 10 个 repos 的 2M 行代码，并且能在每次 git push 后完成 incremental re-indexing。

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 hours

## 问题

到 2026 年，每个 frontier coding agent 都会带一个 codebase retrieval layer，因为 context windows 本身无法解决 cross-repo questions。Claude 的 1M-token context 有帮助；但它并没有消除 ranked retrieval 的必要性。对 raw chunks 做 naive cosine search，会在 generated code、monorepo duplication，以及很少被 import 的 long-tail symbols 上污染结果。Production answer 是对 AST-aware chunks 做 hybrid（dense + BM25）search，加上 re-ranker，并由 symbol references graph 支撑。

你通过索引一个真实 fleet 来学习它——不是一个 tutorial repo——并测量 MRR@10、citation faithfulness 和 incremental freshness。Failure modes 是 infrastructure 层面的：一个 100k-file monorepo、一次 retouch 半数文件的 push、一个需要跨四个 repos 才能正确回答的 query。

## 概念

AST-aware ingestion pipeline 会用 tree-sitter parse 每个 file，提取 function 和 class nodes，并在 node boundaries 上 chunk，而不是按固定 token windows 切分。每个 chunk 有三种 representation：dense embedding（Voyage-code-3 或 nomic-embed-code）、sparse BM25 terms，以及一个简短的 natural-language summary。Summary 增加第三种可检索 modality——用户问 “how is X authorized”，summary 里可能写着 “authz”，即使代码里只有 `check_permission`。

Retrieval 是 hybrid。一个 query 会同时触发 dense 和 BM25 searches，merge top-k，然后把 union 交给 cross-encoder re-ranker（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。Re-ranked list 会进入 long-context synthesizer（带 prompt caching 的 Claude Sonnet 4.7，或 self-hosted Llama 3.3 70B），并要求每个 claim 都引用 file 和 line range。没有 citations 的 answers 会被 post-filter reject。

Incremental freshness 是 infrastructure problem。Git push 会触发 diff：哪些 files 变化了，哪些 symbols 变化了。只有受影响 chunks 会重新 embed。受影响的 cross-file symbol edges（imports、method calls）会重新计算。Index 可以保持 consistent，而不必每个 commit 都重新处理 2M 行代码。

## 架构

```text
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- Parsing：tree-sitter，带 17 种 language grammars（Python、TS、Rust、Go、Java、C++ 等）
- Dense embeddings：Voyage-code-3（hosted）或 nomic-embed-code-v1.5（self-host），bge-code-v1 fallback
- Sparse index：Tantivy（Rust），BM25F，并按 symbol name vs body 做 field-weighted
- Vector DB：Qdrant 1.12 with hybrid search，或 pgvector + pgvectorscale（适合 under 50M vectors 的 teams）
- Chunk summary model：Claude Haiku 4.5 或 Gemini 2.5 Flash，prompt-cached
- Re-ranker：Cohere rerank-3 或 self-hosted bge-reranker-v2-gemma-2b
- Orchestration：ingestion 用 LlamaIndex Workflows，query agent 用 LangGraph
- Synthesizer：Claude Sonnet 4.7（1M context）with prompt caching
- Symbol graph：Neo4j（managed）或 kuzu（embedded），用于 import 和 call edges
- Observability：每个 retrieval + synthesis step 都有 Langfuse spans

## Build It

1. **Ingestion walker。** 在每个 push hook 上遍历 git history。收集 changed files。对每个 file，用 tree-sitter parse，提取 function 和 class nodes 及其完整 source span。Emit chunk records `{repo, path, start_line, end_line, symbol, body}`。

2. **Chunk summarizer。** 把 chunks batch 到 Haiku 4.5 calls，并对 system preamble 使用 prompt caching。Prompt：“Summarize this function in one sentence, naming its public contract and side effects.” 将 summary 与 chunk 一起存储。

3. **Embedding pool。** 两个 parallel queues：dense（Voyage-code-3 batch 128）和 summary（同一模型，但输入 summary string）。把 vectors 写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 index。** Field-weighted Tantivy index：symbol name weight 4、symbol body weight 1、summary weight 2。这样既支持 “find the function named X”，也支持 “find the function that does X”。

5. **Symbol graph。** 对每个 chunk 记录 edges：imports（this file uses symbol Y from repo Z）、calls（this function calls method M on class C）、inheritance。存入 kuzu。Query time 用它跨 repo boundaries 扩展 retrieval。

6. **Query agent。** LangGraph 有三个 nodes。`retrieve` 并行触发 dense + BM25，并按 (repo, path, symbol) deduplicate。`rerank` 在 top-50 上运行 cross-encoder，并保留 top-10。`synth` 用 reranked chunks 作为 context 调 Claude Sonnet 4.7，缓存 system prompt，并要求 file:line citations。

7. **Citation enforcement。** Parse model output；任何没有 `(repo/path:start-end)` anchor 的 claim 都会被标记为 re-ask 或 drop。只向用户返回 cited-only answer。

8. **Incremental re-index。** 每次 webhook 触发时，计算 symbol-level diff。只重新 embed text 变化的 chunks。对 imports 变化的 chunks 重新计算 symbol edges。目标指标：对于 2M-LOC fleet，一次 50-file push 在 60 秒内完成 re-index。

9. **Eval。** 标注 100 个 cross-repo questions，并给出 gold file:line answers。测量 MRR@10、nDCG@10、citation faithfulness（带 verifiable anchors 的 claims 占比），以及 p50/p99 latency。

## Use It

```console
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Ship It

Deliverable skill：`outputs/skill-codebase-rag.md`。给定一组 repos，它会搭建 ingestion pipeline、hybrid index 和 query agent，并针对任意 cross-repo question 返回带 citation 的 answer。Rubric：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Retrieval quality | 在 100-question held-out set 上测 MRR@10 和 nDCG@10 |
| 20 | Citation faithfulness | Answer claims 中带 verifiable file:line anchors 的比例 |
| 20 | Latency and scale | 在 indexed corpus size 上，10k QPS 时的 p95 query latency |
| 20 | Incremental indexing correctness | 50-file commit 从 git push 到 searchable 的时间 |
| 15 | UX and answer formatting | Citation clickability、snippet previews、follow-up affordance |
| **100** | | |

## 练习

1. 把 Voyage-code-3 换成 self-hosted nomic-embed-code。测量 MRR@10 delta。报告启用 re-ranking 后 gap 是否缩小。

2. 向 corpus 注入 20% generated code（LLM-produced boilerplate）并重新 evaluate。观察 retrieval poisoning。向 payload 添加 `"generated"` flag，并 down-weight 这些 hits。

3. 在你的 corpus size 上 benchmark Qdrant hybrid search vs pgvector + pgvectorscale。报告 batch size 1 时的 p99。

4. 添加 sampling-based drift check：每周重新运行 100-question eval。MRR@10 drop > 5% 时 alert。

5. 扩展到 cross-language symbol resolution：一个 Python function 通过 gRPC 调用 Go service。使用 symbol graph 将它们 link 起来。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| AST-aware chunking | “Function-level splits” | 在 tree-sitter node boundaries 上切代码，而不是固定 token windows |
| Hybrid search | “Dense + sparse” | 并行运行 BM25 和 vector search，merge top-k，再 rerank |
| Cross-encoder rerank | “Second-stage rank” | 一起打分每个 (query, candidate) pair 的模型，比 cosine 更准确 |
| Prompt caching | “Cached system prompt” | 2026 Claude / OpenAI feature，可让重复 prefix tokens 最高打 90% 折扣 |
| Symbol graph | “Code graph” | 跨 files 和 repos 的 imports、calls、inheritance edges |
| Citation faithfulness | “Grounded answer rate” | 用户能点击 anchor 并阅读 referenced span 来验证的 claims 占比 |
| Incremental re-index | “Push-to-search time” | 从 git push 到 changed symbols 可被 query 的 wall-clock time |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — production cross-repo code intelligence
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本 capstone 的 reference deep-dive
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — commercial symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — reference implementation
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 details
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder reference
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — internal-platform reference
