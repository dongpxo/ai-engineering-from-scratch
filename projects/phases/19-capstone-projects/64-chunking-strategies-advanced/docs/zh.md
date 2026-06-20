# Chunking 策略对比

> Chunking 决定了你的 retriever 最多能召回什么内容。边界切错了，下游再好的 embedding model、reranker、LLM 都补不回这个损伤。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 lessons 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (lessons 20-29)
**Time:** ~90 minutes

## 学习目标

- 从零实现五种 chunking 策略：fixed-window、sentence、recursive-split、semantic clustering 和 structural markdown headers。
- 在带 gold-labeled answer spans 的 fixture corpus 上测量 recall@k，并解释为什么一种策略在 prose 上胜出，而另一种策略在 technical documents 上胜出。
- 阅读 chunk-length distribution，并识别每种策略注入的 failure modes：orphan sentences、mid-symbol cuts、header-only chunks、semantic drift。
- 不运行 benchmark，只通过检查三种属性为新 corpus 选择默认策略：document type、average paragraph length，以及格式是否携带显式结构。

## 问题

每条 RAG pipeline 都从把源文档切成片段开始：片段要足够小，让 embedding model 能处理；也要足够大，让每个片段承载一个自洽的想法。切在哪里不是一个普通 hyperparameter。它决定了 retriever 最多能返回什么。

如果 query 问的是 "what does the budget abort threshold look like"，只有当包含 abort threshold 的 chunk 可以被召回时，系统才可能答对。如果 fixed-window splitter 把 threshold value 和周围上下文切开，embedding 会移动到另一个 cluster，BM25 score 会下降，rerankers 看到的是噪声，LLM 最终生成的答案也会错。2024 年论文 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" 测到，仅仅因为 chunking 选择不同，retrieval recall 就会出现 35 个百分点的绝对波动。2025 年关于 contextual chunk headers 的后续工作缩小了这个差距，但没有完全消除它。

本课会并排构建五种策略，在带 gold-labeled answer spans 的 fixture corpus 上运行它们，让你亲自阅读 recall numbers。

## 概念

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Fixed-window

最粗暴的 baseline。每 N 个字符切一次。可以选择 overlap，让在位置 N 被切断的句子，在从 `N - overlap` 开始的 chunk 里完整出现。它快速、deterministic，但边界质量很差。把它当 control，不要当 default。

### Sentence

用 regex 或简单 state machine 在句子边界切分。把一个或多个句子打包成一个 chunk，直到接近目标字符预算。它不会切在单词中间，但仍然会切在段落中间和章节中间。很多早期 RAG pipeline 都用它作默认策略；对没有其它结构的 prose 来说，它也是合理选择。

### Recursive split

这是 2023 年左右的库普及起来的层级策略。先尝试最强 separator（double newline、paragraph），不行就退回下一级（single newline），然后退回 sentence，最后退回 character。当 chunk 符合预算时 recursion 终止。它适合结构不一致的文档，因为它会按区域自适应。

### Semantic clustering

先 embed 每个句子。再把共享 topic centroid 的连续句子聚成 cluster。当下一个句子和 running centroid 的 similarity 低于 threshold 时切开。边界反映语义，而不是字符数量。它构建更慢，并且依赖 embedding model，但面对段落内部切换话题的文档时更稳。

### Structural markdown headers

对于携带显式结构的文档（markdown、reStructuredText、RFC-style numbered sections），可以按 heading boundaries 切分。每个 chunk 由 heading 加它下面的所有内容组成，一直延伸到同级或更高级 heading 出现为止。这会得到每个 topic 下最小的 chunk，但前提是 corpus 本身格式良好。

### recall@k 如何衡量边界选择

一条 gold-labeled query 会携带答案 span 在源文档中的精确 character offsets。Chunking 之后，你要问：retriever 返回的 top-k chunks 中，有没有任何一个和 gold span 重叠？如果有，这条 query 的 recall@k 就是 1；如果没有，就是 0。对 query set 求平均。对每种策略运行同样的 evaluation，策略之间的差距就会显示哪种 boundary policy 能扛住你的 corpus。

## Build It

`code/main.py` 实现：

- `fixed_window(text, size, overlap)` - baseline。
- `sentence_chunks(text, target)` - 简单的 sentence packer。
- `recursive_split(text, separators, target)` - 层级 recursion。
- `semantic_chunks(text, similarity_threshold)` - 基于 deterministic mock embedding 的 centroid-based clustering。
- `structural_markdown(text)` - header-aware splitter。
- `mock_embed(text, dim)` - hash-based embedding，让整个 loop 可以 offline 运行。
- `DenseIndex` - 和 Phase 19 Track B hybrid retrieval lesson 使用的 shape 一致。
- `eval_recall(strategy, corpus, queries, k)` - comparison loop。
- `main()` 会在 fixture corpus 上运行每种策略，并打印 recall@k table。

运行：

```bash
python3 code/main.py
```

输出是一个小表格，每种策略一行，每个 k 一列。Sentence 在 structured fixture 上会输。Structural-markdown 在 markdown fixture 上会赢。Recursive 在 mixed fixture 上站得住，因为 recursion 会自适应。Semantic clustering 会在没有有用 structural cues 的 prose fixture 上胜出。

## 表格不会替你掩盖的 failure modes

**Orphan sentences.** Sentence packing 会产生漏掉 topic sentence 的 chunk。Embedding 随后会指向错误 cluster。

**Mid-symbol cuts.** Fixed-window 在 code 或 YAML 里会把 identifier 切成两半。这两半 embed 后基本就是噪声。

**Header-only chunks.** Structural markdown 可能发出只包含 `## Title` 的 chunk。过滤掉它们，或者把下一个 chunk 的第一段接上去。

**Semantic drift.** 当 corpus 整体话题高度一致时，semantic clustering 会切得太少。一个 5000 字符的 chunk 会把许多具体答案塞进一个 diffuse embedding。把 semantic 和 hard character cap 组合起来。

**Stale embeddings.** Semantic clustering 使用 embedding model。如果你换了模型，chunks 也会变化。把 chunk model 和 retrieval model 分别 pin 住，或者一起重建 index。

## 不运行 benchmark 时如何选择默认策略

新 corpus 的默认 chunker 由三个属性决定。

| Property | Value | Default |
|----------|-------|---------|
| Document type | Prose with no structure | Recursive split, target 800 |
| Document type | Markdown / RFC / API docs | Structural markdown |
| Document type | Code | AST-aware (out of scope; see Phase 19 lesson 02) |
| Paragraph length | Long, single topic | Sentence, target 500 |
| Paragraph length | Short, mixed topics | Semantic, threshold 0.6 |

拿不准时，选择 recursive split。它是最强的单策略 baseline。

## Use It

Production patterns：

- 上线新 pipeline 之前先运行 eval；不要相信你的库默认使用的策略。
- 每次更换 embedding model 或 corpus mix 时都重新运行 eval；赢家依赖 corpus。
- 在每个 chunk 的 metadata 中持久化 strategy name，这样后续才能归因 regressions。

## Ship It

第 69 课的 Track F end-to-end RAG system 会把这里选出的 chunker 作为第一阶段。第 68 课的 eval harness 会读取与本课 `eval_recall` 返回值同形状的 recall@k。选出在你的 corpus 上胜出的策略，并把它传给后续系统。

## 练习

1. 添加第六种策略：使用 `tiktoken` 的 token-window，而不是 character counts。把它和同一 fixture 上的 fixed-window 对比。
2. 向 prose fixture 注入 30 percent 的 code blocks。重新运行表格。解释为什么除了 structural markdown 之外，每种策略都会损失 recall。
3. 把 deterministic embedding 替换成你项目真实 provider 的 embedding。测量 semantic-clustering recall delta。报告策略之间的差距是变宽还是变窄。
4. 给每个 chunk 添加一个 `summary` field：一句话的 centroid description。把 summary append 到 chunk body 后重新运行 eval。测量 recall lift。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Recall@k | "Did we get the right chunk?" | top-k chunks 中任意一个和 gold answer span 重叠的 query 占比 |
| Chunk overlap | "Sliding window" | 把上一个 chunk 的最后 N 个字符重新包含到下一个 chunk 中 |
| Structural splitter | "Header-aware chunks" | 在 H1/H2/H3 boundaries 切分；heading text 也是 chunk 的一部分 |
| Semantic chunker | "Topic-aware chunks" | Embed sentences，根据 centroid similarity 聚类，并在 drift 时切开 |
| Centroid drift | "Topic shift" | running mean 和下一个 sentence 的 cosine similarity 跌破 threshold |

## 延伸阅读

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 lesson 06 - RAG fundamentals
- Phase 11 lesson 07 - advanced RAG
- Phase 19 lesson 65 - hybrid retrieval that ranks the chunks produced here
- Phase 19 lesson 68 - the eval harness that scores the strategy choice in production
