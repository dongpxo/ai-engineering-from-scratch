# 面向预训练的数据管线

> 模型是一面镜子。它会反射你喂给它的数据。喂进去的是垃圾，它就会用完美流畅的语言反射出垃圾。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 · Lessons 01-02（Tokenizers，Building a Tokenizer）
**Time:** ~90 minutes

## 学习目标

- 构建一个流式数据管线，在不把全部文本载入内存的情况下，对 TB 级文本做 tokenization、chunking、shuffle 和 batching
- 实现真实预训练管线中使用的数据质量过滤：去重、语言检测与内容过滤
- 创建固定长度训练序列，并正确处理 attention mask 与文档边界
- Profile 管线吞吐量，确保 dataloader 能跟上 GPU 训练速度

## 问题

你已经有了 tokenizer。现在你需要数据。

不是一个 dataset。不是一份 CSV 文件。而是 TB 级文本：清洗过、去重过、按质量过滤过、tokenize 成固定长度序列，并且能以随机 batch 的形式高速送出，快到你的 8-GPU 集群永远不用等下一批数据。

大多数人以为训练 LLM 的核心是模型架构。其实不是。Llama 3 用了 15.6 万亿个 token。GPT-3 用了 3000 亿个 token。DeepSeek-V2 用了 8.1 万亿个 token。这三个模型的架构大体相同：堆叠 transformer blocks，里面是 attention 和 feedforward layers。输出质量的差异，绝大多数来自数据。

DeepMind 的 Chinchilla 论文把这件事说得更精确。给定一个 compute budget，模型参数量和训练 token 数之间存在最优比例。Chinchilla 证明，2022 年的大多数模型都严重 undertrained：相对于它们看到的数据量，参数太多了。一个用 1.4 万亿 token 训练的 70B 参数模型（符合 Chinchilla 最优比例）超过了一个用 3000 亿 token 训练的 280B 模型（Gopher）。

你的数据管线决定了模型学到的是语言，还是噪声。

## 概念

### 数据从哪里来

每个大型语言模型都会在一组混合来源上训练。多数实验室会严密保密确切的数据配比，但我们已经知道足够多的类别，可以理解整体格局。

| Source | Size | Quality | Used By |
|--------|------|---------|---------|
| Common Crawl | ~250 TB raw | 低（需要重度过滤） | GPT-3, Llama, most open models |
| Wikipedia | ~20 GB | 高 | Every major LLM |
| GitHub code | ~1 TB+ | 中（大量重复和死代码） | StarCoder, CodeLlama, DeepSeek-Coder |
| Books (BookCorpus, Pile) | ~100 GB | 高 | GPT-2, GPT-3, early models |
| Academic papers (arXiv, S2ORC) | ~100 GB | STEM 领域高 | Llama, Galactica |
| StackOverflow, Reddit | ~100 GB | 中 | Llama, Falcon |
| Curated web (C4, RefinedWeb) | ~5 TB | 中到高（已预过滤） | T5, Falcon |

Llama 3 披露过自己的数据混合：大约 50% web data，25% code，13% books and academic papers，8% math data，4% multilingual web data。总量是 15.6 万亿 token，来自超过 5 TB 的原始文本。

比例和总量一样重要。Web data 太多，模型会变成 Reddit 复读机。代码太少，它就不会编程。数学太少，它推理会失败。把这个 mix 调好，是训练 LLM 最难的部分之一，而且没有公式：只能靠实验和评估。

### 数据清洗

原始 web data 很脏。典型的 Common Crawl dump 会包含：

- HTML tags 和 JavaScript
- 样板式 headers、footers、navigation menus
- 重复页面（精确重复和近重复）
- 机器生成的 spam
- Personally identifiable information（PII）
- 低质量文本（关键词列表、SEO spam）
- 被编码成文本的非文本内容

清洗不是可选项。它决定了模型是生成连贯段落，还是输出混着商品列表的 HTML tags。

```mermaid
graph TD
    A["原始文本"] --> B["移除 HTML"]
    B --> C["语言检测"]
    C --> D["质量过滤"]
    D --> E["去重"]
    E --> F["PII 移除"]
    F --> G["干净文本"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

每一步都会消除一种噪声：

**HTML stripping：** 移除所有 markup，只保留可见文本。`trafilatura` 或 `readability` 这类库可以提取正文，同时丢弃导航、广告和样板内容。

**Language detection：** 使用 fastText 的语言识别模型（lid.176.bin）给每个文档分类。只保留目标语言。如果一个文档被分类为英文，但 confidence 低于 0.8，它很可能不是干净英文。

**Quality filtering：** 这里开始有意思。RefinedWeb（Falcon 背后的 dataset）使用基于 perplexity 的过滤器：先在 Wikipedia 上训练一个小语言模型，再给每个文档打分。高 perplexity 表示这个文档不像 Wikipedia，通常意味着 spam、关键词列表或机器生成内容。perplexity 超过阈值的文档会被移除。

**Deduplication：** 影响最大的单个清洗步骤。Common Crawl 里有海量重复页面：法律免责声明、cookie 提示、服务条款。用重复内容训练会浪费 compute，还可能让模型记忆并逐字吐出特定段落。

**PII removal：** 姓名、邮箱、电话号码、社会安全号码。结构化 PII 可以用 regex 检测；上下文中的姓名通常需要 NER 模型。

### 用 MinHash 去重

精确去重很简单：给每个文档做 hash，移除重复项。但真正的问题是近重复。两份同一篇新闻文章，周围广告略有差异，就是近重复。主体内容 95% 相同，但按 byte 比较并不相等。

MinHash + Locality-Sensitive Hashing（LSH）可以高效解决这个问题。

```mermaid
graph LR
    A["文档"] --> B["Shingling"]
    B --> C["MinHash signature"]
    C --> D["LSH buckets"]
    D --> E["候选文档对"]
    E --> F["Jaccard similarity"]
    F --> G["去重后的集合"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

核心思路：

1. **Shingling：** 把每个文档转成一组 n-grams（例如词级或字符级 5-grams）。`the quick brown fox` 使用 3-word shingles 会得到 `{"the quick brown", "quick brown fox"}`。

2. **MinHash：** 对每个文档的 shingle set 计算 k 个 hash 值。每个 hash 值是在不同 hash function 下，所有 shingles 的最小 hash。这样会生成一个固定大小的 signature，用来近似任意两个文档之间的 Jaccard similarity。

3. **LSH：** 根据 MinHash signature 的若干 bands，把文档分到 buckets 里。同一个 bucket 里的文档是候选近重复项。这样可以避免两两比较所有文档：只比较候选项。

4. **Verify：** 对每个候选文档对计算精确 Jaccard similarity。如果 similarity 超过阈值（通常是 0.8），就移除其中一个副本。

Llama 团队报告说，去重移除了大约 38% 的 web data。这不是小数字。超过三分之一的 Common Crawl 都是重复或近重复内容。

### 序列打包

模型需要固定长度的输入序列。你的文档长度却各不相同。有些只有 50 个 token。有些有 50,000 个 token。

朴素做法：把每个文档都 pad 到最大 sequence length。这样会把大量 compute 浪费在对学习毫无贡献的 padding token 上。

更好的做法：把多个文档 pack 到同一个序列中，并用 end-of-sequence token 分隔。一个 2048-token sequence 可能包含三个短文档，中间用 `[EOS]` 分开。

```mermaid
graph TD
    subgraph Naive Packing
        A1["Doc A (200 tokens)"] --> P1["[PAD] x 1848"]
        A2["Doc B (500 tokens)"] --> P2["[PAD] x 1548"]
        A3["Doc C (100 tokens)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing
        B1["Doc A (200) | Doc B (500) | Doc C (100) | Doc D (400) | Doc E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff
```

attention mask 必须正确设置。同一个 packed sequence 里，Document A 的 token 不应该 attend 到 Document B 的 token。这需要 block-diagonal attention mask。

长文档会被截断，或在 sequence boundary 被切成 chunks。切分点很重要：在句子中间切断，会迫使模型看到不完整的思路。有些管线会尽可能把切分点对齐到段落或句子边界。

### Chinchilla Scaling Law

给定固定 compute budget C（以 FLOPs 计），最优模型大小 N 和 dataset size D 遵循：

```text
N_opt ~ C^0.5
D_opt ~ C^0.5
```

实际含义是：模型大小和 dataset size 应该大致等比例扩展。参数量增加 10x 的模型，需要大约 10x 的训练 token，才能达到相同 loss 水平。

| Model | Parameters | Training Tokens | Chinchilla-Optimal? |
|-------|-----------|----------------|-------------------|
| GPT-3 | 175B | 300B | 否（undertrained 3-4x） |
| Chinchilla | 70B | 1.4T | 是（设计目标） |
| Llama 2 | 70B | 2T | Overtrained（有意为之） |
| Llama 3 | 70B | 15T | 重度 overtrained |

Llama 3 故意违反 Chinchilla law。Meta 发现，使用远超 compute-optimal 比例的数据进行 overtraining，会得到更适合推理部署的模型。额外训练成本只付一次，但更小的模型会永远更便宜地被服务。这有时被称为 “inference-optimal” scaling approach，并且从 2024 年起成为行业标准。

## 动手构建

### Step 1：文本清洗

移除 HTML，规范化空白，去掉非文本内容。这里会用一个公有领域文本（Project Gutenberg）作为小语料。

```python
import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
```

这个 quality filter 会抓住 SEO spam（ALL CAPS）、机器生成噪声（特殊字符比例过高）和 stub pages（太短）。仅仅这三项检查，就能从 web crawls 中移除数量惊人的垃圾。

### Step 2：MinHash 去重

从零实现 MinHash。不需要外部库，只用 `hashlib`。

```python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
```

`num_hashes=128` 和 `bands=16` 这两个参数控制 precision-recall tradeoff。更多 hashes 会给出更准确的 similarity estimate。更多 bands 会提高 recall（抓住更多重复项），代价是更多 false positives。对典型 web text 来说，这组值效果不错。

### Step 3：Tokenize 并打包序列

取清洗、去重后的文本，tokenize 它，再 pack 成训练用的固定长度序列。

```python
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
```

### Step 4：训练用 DataLoader

产出随机化的 packed sequence batches。这正是 training loop 会消费的东西。

```python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
```

### Step 5：Dataset Statistics

计算真正重要的数字：total tokens、unique tokens、compression ratio 和 document length distribution。

```python
from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
```

Compression ratio 告诉你 tokenizer 在这个 corpus 上有多高效。英文文本通常会压缩到每个 token 约 3-4 个字符。如果你看到每个 token 只有 1.5 个字符，说明 tokenizer 切得太碎。如果看到 8+，说明它学到了非常 domain-specific 的 merges。

Sequence utilization 告诉你 packed sequences 里有多少是真实数据，多少是 padding。低于 90% 意味着 packing 效率不好：你正在把 compute 浪费在 padding tokens 上。

## 投入使用

### 与 HuggingFace Datasets 对比

用 HuggingFace 的 datasets library 加载同一份 corpus，并比较管线速度。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
```

HuggingFace pipeline 底层使用 Rust tokenizers，并且会在 4 个 cores 上并行处理。你的纯 Python pipeline 会慢 10-50x。这个差距正是生产团队使用 compiled tokenizers 的原因。算法是一样的。实现语言造成了差异。

## 交付成果

本课会产出一个用于验证和调试 LLM 训练数据质量的 prompt。见 `outputs/prompt-data-quality-checker.md`。

## 练习

1. **Easy：** 使用简单 heuristic（字符集分析）给 cleaning pipeline 增加 language detection。只保留英文文档，并测量移除了多少文档。
2. **Medium：** 在 MinHash 近重复去重之外，实现基于 SHA-256 hashes 的精确去重。用 web-scraped corpus 比较两种方法各自抓到了多少重复项。
3. **Hard：** 构建一个基于 perplexity 的 quality filter。在 Wikipedia text 上训练一个小 bigram language model，按 perplexity 给每个文档打分，并移除最差的 20%。比较在 filtered 和 unfiltered data 上训练时的模型输出质量。

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|----------|----------|
| Common Crawl | “互联网” | 一个每月爬取 web 的非营利组织，原始数据约 250 TB，是多数 LLM 训练数据的起点 |
| MinHash | “某种 hash 技巧” | 一种用固定大小 signatures 估计集合间 Jaccard similarity 的技术，使大规模近重复检测成为可能 |
| LSH | “Locality-Sensitive Hashing” | 一种把相似 items 分到同一 bucket 的方法，把 pairwise comparisons 从 O(n^2) 降到近似线性 |
| Sequence packing | “拼接文档” | 用正确 attention masks 把多个文档装入固定长度序列，消除 padding 浪费 |
| Chinchilla scaling | “多用数据训练” | 给定固定 compute budget，最优性能需要大致等比例扩展模型大小和训练 token 数 |
| Fertility | “每个词多少 token” | 每个词平均被拆成多少 token；GPT-4 英文约 1.3，非拉丁文字通常更高 |
| Data mixing | “选择训练数据” | code、text、math、multilingual data 之间的比例；没有公式，需要实验 |
| Perplexity filter | “质量打分” | 用一个小语言模型给文档打分；高 perplexity 表示文本不像干净参考数据 |
| Deduplication | “移除副本” | 消除精确重复和近重复文档；通常会移除原始 web data 的 30-40% |
| Attention mask | “哪些 token 可以看哪些 token” | 一种 binary mask，在 packed sequences 中阻止 attention 跨越文档边界 |

## 延伸阅读

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) -- 改变我们理解数据规模方式的论文
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) -- 如何把 Common Crawl 过滤成高质量数据
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) -- Llama 2 的数据管线细节
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) -- 为什么去重比你想的更重要
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900) -- MinHash 的原始论文
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) -- 15.6T tokens、data mixing ratios 和 filtering pipeline
