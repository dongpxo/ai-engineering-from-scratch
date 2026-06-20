# ColPali 与 Vision-Native Document RAG

> 传统 RAG 会把 PDFs parse 成 text，切成 chunks，embed chunks，再存 vectors。每一步都会丢失信号：OCR 会漏掉 chart data，chunking 会切断 table rows，text embeddings 会忽略 figures。ColPali（Faysse et al., 2024 年 7 月）提出了一个更简单的问题：为什么一定要抽取 text？直接通过 PaliGemma embed page image，使用 ColBERT-style late interaction 做 retrieval，并保留 document 携带的所有 layout、figures、fonts 和 formatting signal。公开 benchmark：在视觉信息丰富的 documents 上，end-to-end accuracy 比 text-RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这个 pattern。本课阅读 vision-native RAG thesis，并构建一个小型 ColPali-like indexer。

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## 学习目标

- 解释 bi-encoder retrieval（每个 document 一个 vector）和 late-interaction retrieval（每个 document 多个 vectors）的区别。
- 描述 ColBERT 的 MaxSim operation，以及 ColPali 如何把它从 text tokens 泛化到 image patches。
- 构建一个小型 ColPali-like indexer：page → patch embeddings → query-term embeddings 上的 MaxSim → top-k pages。
- 在 invoices / financial reports use case 上，比较 ColPali + Qwen2.5-VL generator 和 text-RAG + GPT-4。

## 问题

PDF 上的 Text-RAG 会丢掉 document 的大部分信息。Financial report 的 Q3 revenue growth 通常在 chart 里；medical report 的 findings 在 annotated images 里；legal contract 的 signature block 是 layout fact，不是 text fact。

Text-RAG pipeline：

1. PDF → text via OCR / pdftotext。
2. Text → 300-500 token chunks。
3. Chunk → bi-encoder embedding（一个 vector）。
4. User query → embedding → cosine similarity → top-k chunks。
5. Chunks + query → LLM。

五个 lossy steps。Charts 没有捕获。Tables 被 chunk 切断。Multi-column layout 被展平。Figure annotations 消失。

ColPali 的修复方式：跳过 OCR，直接 embed page image。使用 ColBERT-style late interaction 做 retrieval，让模型在 query time attend 到 fine-grained patches。

## 核心概念

### ColBERT (2020)

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是一种 text retrieval 方法。它不是每个 document 产生一个 vector，而是每个 token 产生一个 vector。Query time：

- Query tokens 拥有自己的 embeddings（N_q vectors）。
- Document tokens 拥有 embeddings（N_d vectors，通常已缓存）。
- Score = 对每个 query token，取所有 document tokens 的 cosine similarity 最大值，然后求和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim operation。每个 query token 都会 "pick" 最匹配的 document token。最终 score 是这些最大值之和。

优点：recall 强，能处理 term-level semantics。缺点：每个 document 有 N_d vectors，storage expensive。

### ColPali

ColPali（Faysse et al., arXiv:2407.01449）把 ColBERT pattern 应用到 images。

- 每个 page 由 PaliGemma（ViT + language）encode 成 patch embeddings：每页 N_p vectors。
- 每个 user query（text）被 encode 成 query-token embeddings：N_q vectors。
- Score = Σ_i max_j cos(q_i, p_j)，也就是在 query-text-tokens 和 page-image-patches 上做 MaxSim。
- 按 total score 取回 top-k pages。

Document ingestion 时：用 PaliGemma embed 每一页，存储所有 patch embeddings。Query time：embed query tokens，对所有 stored page embeddings 计算 MaxSim，返回 top-k pages。

优点：在视觉信息丰富的 documents 上，end-to-end 比 text-RAG 高 20-40%。每个 patch-vector 捕获 local layout 和 content。

缺点：每页 N_p patches × 4-byte floats × D-dim vectors，storage 增长很快。可用 PQ / OPQ quantization 缓解。

### ColQwen2 and ColSmol

ColQwen2（illuin-tech, 2024-2025）把 PaliGemma 换成 Qwen2-VL。Base encoder 更强，retrieval 更好。

ColSmol 是用于 local / edge use 的小规模 variant。约 1B params 的 ColSmol retriever 可以跑在消费级 GPU 上。

### VisRAG

VisRAG（Yu et al., arXiv:2410.10594）是另一个 variant：它不在 patches 上做 MaxSim，而是用 VLM 把每页 pool 成一个 single vector，再做 bi-encoder retrieval。Indexing 更快、storage 更小，但 recall 更弱。

Quality-vs-cost trade-off：要质量选 ColPali，要 scale 选 VisRAG。

### M3DocRAG

M3DocRAG（Cho et al., arXiv:2411.04952）把 multi-modal retrieval 扩展到 multi-page multi-document reasoning。它会跨 documents 检索 pages，并为 VLM 组合 multi-page context。

### ViDoRe — Benchmark

ColPali 的 companion benchmark。Visual Document Retrieval Evaluation。任务包括 financial reports、scientific papers、administrative documents、medical records、manuals。Metric：nDCG@5。

ColPali-v1 在 ViDoRe 上约 80% nDCG@5；同一批 documents 上的 text-RAG 约 50-60%。

### End-to-End RAG Pipeline

Vision-native RAG 的流程：

1. Ingest：PDF → page images → PaliGemma encoding → store all patch embeddings。
2. Query：user text → query-token embeddings → MaxSim against all indexed pages → top-k pages。
3. Generate：top-k page images + query → VLM（Qwen2.5-VL or Claude）→ answer。

全程没有 OCR。Figures、charts、fonts、layout 都流入 answer。

### Storage Math

一份 50-page financial report，每页 729 patches，128-dim embeddings：

- ColPali：50 * 729 * 128 * 4 bytes = ~18 MB raw，PQ 后 ~4 MB。
- Text-RAG：50 chunks * 768-dim * 4 bytes = ~150 kB。

ColPali 每个 document 的 storage 约为 text-RAG 的 30x。Scale 上，OPQ / PQ 能把它降到约 5-10x，通常可以接受。

### Text-RAG 什么时候仍然赢

- 没有 layout signal 的 pure-text documents（wiki articles、chat logs）。Text-RAG 更简单、storage 更便宜。
- Storage 成本主导的 multi-million-page archives。
- 严格监管要求必须同时保留 extractable OCR text。

除此之外，在 2026 年的 financial reports、scientific papers、legal contracts、medical records、UX documentation 上，vision-native RAG 胜出。

## 实战使用

`code/main.py`：

- Toy patch encoder：把一个 "page"（小型 feature-vector grid）映射为 patch embeddings array。
- MaxSim scorer：计算 query token embedding set 和 page patch set 之间的 ColBERT-style score。
- Index 5 个 toy pages，运行 3 个 queries，并返回带 scores 的 top-k。

## 交付

本课产出 `outputs/skill-vision-rag-designer.md`。给定 document-RAG project，它会在 ColPali / ColQwen2 / VisRAG / text-RAG 之间做选择，并估算 storage。

## 练习

1. 一份 200-page annual report，每页 729 patches，128-dim embedding，4-byte floats。计算 raw storage 和 PQ-compressed（8x）storage。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。相比 simple mean similarity，这个 sum 捕获了什么？

3. ColPali 把 pages index 为 patch sets。如果我们改成 word-level indexing（像 ColBERT 那样），会有什么变化？取舍是什么？

4. 为一个 1M-page corpus 设计 end-to-end pipeline，latency budget 是每 query 500ms。选择 ColQwen2 / VisRAG，并说明理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述 multi-page attention pattern，以及它和 single-page ColPali retrieval 有什么不同。

## 关键术语

| Term | 人们通常怎么说 | 它实际是什么意思 |
|------|----------------|------------------|
| Late interaction | "ColBERT-style" | 使用 per-token 或 per-patch embeddings + MaxSim 的 retrieval，而不是 single doc vector |
| MaxSim | "Max-over-patches" | 对每个 query token，选择 similarity 最高的 document token；再对 query 求和 |
| Bi-encoder | "Single-vector" | 每个 document 一个 vector；更快，但丢失 granularity |
| Multi-vector | "Many-vectors-per-doc" | 每个 document / page 存 N_p vectors；storage cost 增长，但 recall 改善 |
| Patch embedding | "Page feature" | VLM encoder 为每个 image patch 生成的一个 vector，按 page 缓存 |
| ViDoRe | "Vision doc bench" | ColPali 的 visual document retrieval benchmark suite |
| PQ quantization | "Product quantization" | 在缩小 storage 约 8x 的同时尽量保持 vector similarity 的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
