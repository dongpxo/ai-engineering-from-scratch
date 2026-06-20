# RAG（Retrieval-Augmented Generation）

> 你的 LLM 只知道训练截止时间以前的东西。它不知道你公司的文档、你的 codebase，也不知道上周的会议纪要。RAG 通过检索相关文档并把它们塞进 prompt 来解决这个问题。这是生产 AI 中部署最广的模式。如果你只从这门课里构建一个东西，就构建一条 RAG pipeline。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 (LLMs from Scratch), Phase 11 Lessons 01-05
**Time:** ~90 minutes
**Related:** Phase 5 · 23 (Chunking Strategies for RAG) 讲六种 chunking algorithms，以及每种何时更优。Phase 5 · 22 (Embedding Models Deep Dive) 讲如何选择 embedder。Phase 11 · 07 (Advanced RAG) 讲 hybrid search、reranking 和 query transformation。

## 学习目标

- 构建完整的 RAG pipeline：document loading、chunking、embedding、vector storage、retrieval 和 generation
- 使用 vector database（ChromaDB、FAISS 或 Pinecone）和正确 indexing 实现 semantic search
- 解释为什么对 knowledge-grounded applications 来说，RAG 通常优于 fine-tuning（cost、freshness、attribution）
- 用 retrieval metrics（precision、recall）和 generation metrics（faithfulness、relevance）评估 RAG 质量

## 问题

你为公司构建了一个 chatbot。客户问：“What's the refund policy for enterprise plans?” LLM 给出了一段关于典型 SaaS 退款政策的泛泛回答。真实政策埋在 200 页的 internal wiki 里：enterprise customers 有 60-day window，并且按比例退款。LLM 从来没见过这份文档。它不可能知道训练时没有学过的东西。

Fine-tuning 是一种方案。拿 LLM 在你的内部文档上继续训练，然后部署更新后的模型。这能工作，但问题很严重。Fine-tuning 要花费数千美元 compute。一旦文档发生变化，模型立刻过期。你无法知道模型回答来自哪个 source。如果公司下个月收购另一条产品线，你还要再 fine-tune 一次。

RAG 是另一种方案。保持模型不变。当问题进来时，在你的 document store 中搜索相关 passages，把它们粘到 question 前面的 prompt 里，然后让模型把这些 passages 当作 context 来回答。Document store 可以在几分钟内更新。你能清楚看到检索到了哪些 documents。模型本身从不改变。这就是 RAG 成为生产主流模式的原因：它更便宜、更新鲜、更可审计，并且适用于任何 LLM。

## 概念

### RAG Pattern

完整模式只有四步：

```mermaid
graph LR
    Q["User Query"] --> R["Retrieve"]
    R --> A["Augment Prompt"]
    A --> G["Generate"]
    G --> Ans["Answer"]

    subgraph "Retrieve"
        R --> Embed["Embed query"]
        Embed --> Search["Search vector store"]
        Search --> TopK["Return top-k chunks"]
    end

    subgraph "Augment"
        TopK --> Format["Format chunks into prompt"]
        Format --> Combine["Combine with user question"]
    end

    subgraph "Generate"
        Combine --> LLM["LLM generates answer"]
        LLM --> Cite["Answer grounded in retrieved docs"]
    end
```

Query -> Retrieve -> Augment prompt -> Generate。每个 RAG system 都遵循这个模式。Production RAG systems 之间的差异都在每一步的细节里：如何 chunk、如何 embed、如何 search，以及如何构造 prompt。

### 为什么 RAG 胜过 Fine-Tuning

| Concern | Fine-tuning | RAG |
|---------|------------|-----|
| Cost | 每次 training run $1,000-$100,000+ | 每次 query $0.01-$0.10（embedding + LLM） |
| Freshness | 重新训练前都会过期 | 通过 re-indexing docs 几分钟内更新 |
| Auditability | 无法追踪 answer 到 source | 可以展示精确 retrieved passages |
| Hallucination | 仍然会自由 hallucinate | Grounded in retrieved documents |
| Data privacy | Training data 被烘进 weights | Documents 留在你的 vector store |

Fine-tuning 会永久改变模型的 weights。RAG 只是临时改变模型的 context。对大多数应用来说，临时 context 正是你想要的。

Fine-tuning 胜出的唯一场景是：你需要模型采用一种特定 style、tone 或 reasoning pattern，而且无法只靠 prompting 达成。对于 factual knowledge retrieval，RAG 每次都赢。

### Embedding Models

Embedding model 会把 text 转换成 dense vector。相似文本会在这个 high-dimensional space 中产生靠得很近的 vectors。“How do I reset my password?” 和 “I need to change my password” 虽然共享词很少，却会生成几乎相同的 vectors。“The cat sat on the mat” 则会生成非常不同的 vector。

常见 embedding models（2026 lineup，完整分析见 Phase 5 · 22）：

| Model | Dimensions | Provider | Notes |
|-------|-----------|----------|-------|
| text-embedding-3-small | 1536 (Matryoshka) | OpenAI | 大多数用例的最佳 price/performance |
| text-embedding-3-large | 3072 (Matryoshka) | OpenAI | 更高 accuracy，可截断到 256/512/1024 |
| Gemini Embedding 2 | 3072 (Matryoshka) | Google | 顶级 MTEB retrieval；8K context |
| voyage-4 | 1024/2048 (Matryoshka) | Voyage AI | Domain variants（code、finance、law） |
| Cohere embed-v4 | 1024 (Matryoshka) | Cohere | 强 multilingual，128K context |
| BGE-M3 | 1024 (dense + sparse + ColBERT) | BAAI (open-weight) | 一个模型输出三种 views |
| Qwen3-Embedding | 4096 (Matryoshka) | Alibaba (open-weight) | 顶级 open-weight retrieval score |
| all-MiniLM-L6-v2 | 384 | Open-weight (Sentence Transformers) | Prototyping baseline |

本课会用 TF-IDF 构建自己的简单 embedding。不是因为 production systems 会用 TF-IDF，而是因为它能把概念具体化：text 进去，vector 出来，相似文本产生相似 vectors。

### Vector Similarity

给定两个 vectors，如何衡量 similarity？有三种选择：

**Cosine similarity**：两个 vectors 夹角的 cosine。范围从 -1（相反）到 1（相同）。它忽略 magnitude，只关心方向。这是 RAG 的默认选择。

```text
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

**Dot product**：raw inner product。更大的 vectors 会得到更高 scores。当 magnitude 携带信息时有用，比如更长 documents 可能更相关。

```text
dot(a, b) = sum(a_i * b_i)
```

**L2 (Euclidean) distance**：vector space 中的直线距离。距离越小越相似。它对 magnitude differences 敏感。

```text
L2(a, b) = sqrt(sum((a_i - b_i)^2))
```

Cosine similarity 是标准选择。它通过 magnitude normalization 很好地处理不同长度的 documents。别人说 “vector search” 时，几乎总是在说 cosine similarity。

### Chunking Strategies

Documents 太长，不能作为单个 vectors 来 embed。一个 50-page PDF 可能会产生很差的 embedding，因为它包含几十个 topics。你应该把 documents 切成 chunks，然后分别 embed 每个 chunk。

**Fixed-size chunking**：每 N 个 tokens 切一次。简单、可预测。512-token chunk 加 50-token overlap 意味着 chunk 1 是 tokens 0-511，chunk 2 是 tokens 462-973，以此类推。Overlap 可以避免在倒霉的边界把一句话切断。

**Semantic chunking**：在自然边界处切分。比如 paragraphs、sections 或 markdown headers。每个 chunk 都是一个 coherent unit of meaning。实现更复杂，但 retrieval 更好。

**Recursive chunking**：先尝试最大边界，比如 section headers。如果某个 section 仍然太大，就按 paragraph boundaries 切。如果 paragraph 仍然太大，就按 sentence boundaries 切。这就是 LangChain RecursiveCharacterTextSplitter 的思路，实践中很好用。

Chunk size 比很多人想象的更重要：

- 太小（64-128 tokens）：每个 chunk 缺少 context。“It increased 15% last quarter” 如果不知道 “it” 指什么，就没有意义。
- 太大（2048+ tokens）：每个 chunk 覆盖多个 topics，稀释 relevance。你搜索 revenue data，却拿到一个 10% 关于 revenue、90% 关于 headcount 的 chunk。
- Sweet spot（256-512 tokens）：context 足够自洽，同时 focus 足够相关。

大多数 production RAG systems 使用 256-512 token chunks，带 50-token overlap。Anthropic 的 RAG guidelines 推荐这个范围。

### Vector Databases

有了 embeddings 之后，你需要地方存储和搜索它们。选项包括：

| Database | Type | Best for |
|----------|------|----------|
| FAISS | Library (in-process) | Prototyping, small to medium datasets |
| Chroma | Lightweight DB | Local development, small deployments |
| Pinecone | Managed service | Production without ops overhead |
| Weaviate | Open source DB | Self-hosted production |
| pgvector | Postgres extension | Already using Postgres |
| Qdrant | Open source DB | High-performance self-hosted |

本课会构建一个简单的 in-memory vector store。它把 vectors 存在 list 里，然后做 brute-force cosine similarity search。这等价于使用 flat index 的 FAISS。它大概能扩展到 100,000 vectors，再多就会变慢。Production systems 使用 HNSW 这类 approximate nearest neighbor（ANN）算法，在 milliseconds 内搜索数百万 vectors。

### 完整 Pipeline

```mermaid
graph TD
    subgraph "Indexing (offline)"
        D["Documents"] --> C["Chunk"]
        C --> E["Embed each chunk"]
        E --> S["Store vectors + text"]
    end

    subgraph "Querying (online)"
        Q["User query"] --> QE["Embed query"]
        QE --> VS["Vector search (top-k)"]
        VS --> P["Build prompt with chunks"]
        P --> LLM["LLM generates answer"]
    end

    S -.->|"same vector space"| VS
```

Indexing phase 对每个 document 运行一次，或者在 documents 更新时运行。Querying phase 在每次用户请求时运行。生产中，indexing 可能要用数小时处理数百万 documents。Querying 必须在一秒内响应。

### 真实参数

大多数 production RAG systems 使用这些参数：

- **k = 5 to 10** retrieved chunks per query
- **Chunk size = 256 to 512 tokens**，带 50-token overlap
- **Context budget**：每次 query 2,500-5,000 tokens 的 retrieved content
- **Total prompt**：约 8,000-16,000 tokens（system prompt + retrieved chunks + conversation history + user query）
- **Embedding dimension**：384-3072，取决于 model
- **Indexing throughput**：使用 API embeddings 时每秒 100-1,000 documents
- **Query latency**：retrieval 50-200ms，generation 500-3000ms

```figure
rag-chunking
```

## Build It

### Step 1: Document Chunking

```python
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

### Step 2: TF-IDF Embeddings

我们构建一个简单的 embedding function。TF-IDF（Term Frequency-Inverse Document Frequency）不是 neural embedding，但它会用捕捉 word importance 的方式把 text 转成 vectors。Document 中频繁出现的 words 有更高 TF。Corpus 中稀有的 words 有更高 IDF。两者相乘后，重要且有区分度的 words 会在 vector 中获得高值。

```python
import math
from collections import Counter

def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)

def compute_tf(text, vocab):
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    return [count.get(word, 0) / total for word in vocab]

def compute_idf(documents, vocab):
    n = len(documents)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf

def tfidf_embed(text, vocab, idf):
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]
```

### Step 3: Cosine Similarity Search

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def search(query_embedding, stored_embeddings, top_k=5):
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

### Step 4: Prompt Construction

这一步就是 RAG 里的 “augmented” 发生的地方。取出 retrieved chunks，把它们格式化进 prompt，然后让 LLM 根据提供的 context 回答。

```python
def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
```

### Step 5: The Complete RAG Pipeline

```python
class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.idf = []

    def index(self, documents):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        self.chunks = all_chunks
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]

    def query(self, question, top_k=5):
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, top_k)
        retrieved = [(self.chunks[i], score) for i, score in results]
        prompt = build_rag_prompt(
            question, [chunk for chunk, _ in retrieved]
        )
        return prompt, retrieved
```

### Step 6: Generation (simulated)

Production 中，这一步会调用 LLM API。本课为了模拟 generation，会从 retrieved context 中抽取最相关的 sentence。

```python
def simple_generate(prompt, retrieved_chunks):
    query_words = set(prompt.lower().split("question:")[-1].split())
    best_sentence = ""
    best_score = 0
    for chunk in retrieved_chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(sentence.lower().split())
            overlap = len(query_words & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
    return best_sentence if best_sentence else "I don't have enough information."
```

## Use It

使用真实 embedding model 和 LLM 时，代码几乎不变：

```python
from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

或者使用 Anthropic：

```python
import anthropic

client = anthropic.Anthropic()

def generate(prompt):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

Pipeline 是一样的。换 embedding function。换 generation function。Retrieval logic、chunking、prompt construction 都完全相同，不取决于你用哪种 models。

要做 scale 级别的 vector storage，可以把 brute-force search 替换成正式 vector database：

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

results = collection.query(
    query_texts=["What is the refund policy?"],
    n_results=5
)
```

Chroma 会在内部处理 embedding（默认使用 all-MiniLM-L6-v2），并把 vectors 存到本地数据库。同一个 pattern，不同的 plumbing。

## Ship It

本课产出：

- `outputs/prompt-rag-architect.md`：一个用于为特定 use cases 设计 RAG systems 的 prompt
- `outputs/skill-rag-pipeline.md`：一个教 agents 如何构建和 debug RAG pipelines 的 skill

## 练习

1. 把 TF-IDF embeddings 替换成简单 bag-of-words 方法（binary：word 出现则为 1，否则为 0）。在 sample documents 上比较 retrieval quality。TF-IDF 应该更好，因为它会给 rare words 更高权重。

2. 实验 chunk sizes：在同一组 documents 上尝试 50、100、200、500 words。每种 size 都运行同样 5 个 queries，并统计有多少 query 在 top-3 中返回 relevant chunk。找出 retrieval quality 达峰的 sweet spot。

3. 为每个 chunk 增加 metadata（source document name、chunk position）。修改 prompt template，加入 source attribution，让 LLM 引用来源。

4. 实现一个简单 evaluation：给定 10 组 question-answer pairs，把每个 question 跑过 RAG pipeline，并测量 retrieved chunks 中包含 answer 的比例。这就是 retrieval recall at k。

5. 构建 conversation-aware RAG pipeline：维护最近 3 轮对话历史，并把它们和 retrieved chunks 一起放进 prompt。用 follow-up questions 测试，比如先问 pricing，再问 “What about enterprise?”

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| RAG | “AI that reads your docs” | 检索相关 documents，把它们粘到 prompt 中，然后生成 grounded in those documents 的 answer |
| Embedding | “Convert text to numbers” | 文本的 dense vector representation，其中相似 meanings 会产生相似 vectors |
| Vector database | “Search engine for AI” | 为存储 vectors 并按 similarity 找 nearest neighbors 优化的数据存储 |
| Chunking | “Split docs into pieces” | 把 documents 拆成更小 segments（通常 256-512 tokens），让每段都能独立 embed 和 retrieve |
| Cosine similarity | “How similar are two vectors” | 两个 vectors 夹角的 cosine；1 = 方向相同，0 = 正交，-1 = 相反 |
| Top-k retrieval | “Get the k best matches” | 从 vector store 返回与 query 最相似的 k 个 chunks |
| Context window | “How much text the LLM can see” | LLM 单次 request 能处理的最大 tokens 数；retrieved chunks 必须放得进去 |
| Augmented generation | “Answer using given context” | 使用 retrieved documents 作为 context 生成 response，而不是只依赖 trained knowledge |
| TF-IDF | “Word importance scoring” | Term Frequency 乘以 Inverse Document Frequency；按 words 在 corpus 中的区分度加权 |
| Indexing | “Preparing docs for search” | 离线完成 chunking、embedding 和 storing documents 的过程，让它们能在 query time 被搜索 |

## 延伸阅读

- Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks” (2020)：Facebook AI Research 的原始 RAG paper，形式化了 retrieve-then-generate pattern
- Anthropic's RAG documentation (docs.anthropic.com)：关于 chunk sizes、prompt construction 和 evaluation 的实践指南
- Pinecone Learning Center, “What is RAG?”：用清晰视觉解释 RAG pipeline，并补充 production considerations
- Sentence-BERT: Reimers & Gurevych (2019)：all-MiniLM embedding models 背后的 paper，展示如何训练用于 semantic similarity 的 bi-encoders
- [Karpukhin et al., “Dense Passage Retrieval for Open-Domain Question Answering” (EMNLP 2020)](https://arxiv.org/abs/2004.04906)：DPR paper，证明 dense bi-encoder retrieval 在 open-domain QA 上胜过 BM25，并奠定现代 RAG retrievers 的模式。
- [LlamaIndex High-Level Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)：构建 RAG pipelines 时需要理解的主要概念：data loaders、node parsers、indices、retrievers、response synthesizers。
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)：另一种风格的 orchestrator；用 chain-of-runnables 视角表达同一个 retrieve-then-generate pattern。
