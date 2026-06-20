# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词训练一个嵌入。GloVe 分解共现矩阵。FastText 嵌入词的组成片段。BPE 则把这条路接到了 Transformer。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## 学习目标

- 实现 GloVe 的共现矩阵构建与带权矩阵分解训练循环
- 解释为什么 FastText 用字符 n-gram 为未登录词组合出可用向量
- 从计数与合并规则出发实现一个最小 BPE 词表学习器
- 比较 GloVe、FastText 与 BPE 在 OOV、形态变化和 Transformer 输入上的取舍
- 选择适合传统文本流水线、预训练词向量和语言模型训练的分词策略

## 问题

Word2Vec 留下了两个开放问题。

第一，同时还有一条研究路线不是做在线 skip-gram 更新，而是直接分解共现矩阵（LSA、HAL）。Word2Vec 的迭代式方法真的更好吗？还是差异只是来自两类方法处理计数的方式不同？**GloVe** 回答了这个问题：只要损失函数设计得足够细致，矩阵分解可以追平甚至超过 Word2Vec，而且训练成本更低。

第二，这两种方法都没有处理未见词的方案。`Zoomer-approved`、`dogecoin`、上周才出现的专有名词、罕见词根的每一种屈折形式，都会掉进词表之外。**FastText** 用字符 n-gram 解决这个问题：一个词是它各个片段的和，其中包括词素，所以即使是未登录词也能得到合理向量。

第三，Transformer 出现后，问题又变了。词级词表大约到百万级就会碰到上限，但真实语言远比这开放。**字节对编码（BPE）** 及其近亲通过学习高频子词单元的词表来覆盖所有输入。现代 LLM 的每一个现代 tokenizer，本质上都是子词 tokenizer。

本课会依次走过这三种方法，并解释什么时候该用哪一种。

## 概念

**GloVe（Global Vectors）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 出现在词 `i` 上下文中的次数。训练向量，使 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权，避免高频词对支配训练。完成。

**FastText。** 一个词是它的字符 n-gram 加上词本身的总和。`where` 会变成 `<wh, whe, her, ere, re>, <where>`。词向量就是这些组成向量的和。训练方式与 Word2Vec 相同。好处是：未见词（`whereupon`）也能由已知 n-gram 组合出来。

**BPE（Byte-Pair Encoding）。** 从单个字节（或字符）组成的词表开始。统计语料中的每个相邻 pair。把最高频的 pair 合并成一个新 token。重复 `k` 轮。结果是一个包含 `k + 256` 个 token 的词表，其中高频序列（`ing`、`tion`、`the`）会成为单个 token，罕见词则被拆成熟悉片段。任何句子都能被 token 化。

## Build It

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

这里有两个值得点名的移动部件。权重函数 `f(x) = (x/x_max)^alpha` 会降低超高频词对（比如 `(the, and)`）的权重，避免它们支配损失。最终嵌入是 `W`（中心词）和 `W_tilde`（上下文词）两张表的和。把两者相加是论文中使用的技巧，通常比只使用其中一张表效果更好。

### FastText：感知子词的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词都由它的一组 n-gram 表示（通常是 3 到 6 个字符）。词嵌入是这些 n-gram 嵌入的和。做 skip-gram 训练时，把它接到 Word2Vec 原本使用单个向量的位置即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见词，只要它的一部分 n-gram 已知，你仍然可以得到向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，所以两者会落在相近位置。

### BPE：学习出来的子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮会合并最常见的相邻 pair。经过足够多轮之后，高频子串（`low`、`est`、`tion`）会成为单个 token，罕见词则会被干净地拆开。

真实的 GPT、BERT、T5 tokenizer 会学习 3 万到 10 万个 merge。结果是：任何文本都会被 token 化成一段长度有界的已知 ID 序列，不再有 OOV。

## Use It

实践中，你很少会自己训练这些模型。你通常会加载预训练 checkpoint。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 Transformer 时代，如果要使用 BPE 风格的子词 tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```text
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（这是 GPT-2 的约定）。每个现代 tokenizer 都是 BPE 变体、WordPiece（BERT），或者 SentencePiece（T5、LLaMA）。

### 什么时候选哪一个

| 场景 | 选择 |
|------|------|
| 通用预训练词向量，不需要容忍 OOV | GloVe 300d |
| 通用预训练词向量，必须处理拼写错误、新词或形态丰富语言 | FastText |
| 任何进入 Transformer 的输入（训练或推理） | 模型随附的 tokenizer。不要替换。 |
| 从零训练自己的语言模型 | 先在自己的语料上训练 BPE 或 SentencePiece tokenizer |
| 使用线性模型做生产文本分类 | 仍然是 TF-IDF。见第 02 课。 |

## Ship It

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **Easy.** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠度。你应该会看到大量共享片段（`pla`、`lay`、`play`），这就是 FastText 能很好迁移到形态变体上的原因。
2. **Medium.** 扩展 `learn_bpe`，跟踪词表增长。画出 tokens-per-corpus-character 随 merge 数变化的曲线。你应该会看到一开始压缩很快，然后逐渐逼近约 2-3 chars per token 的水平。
3. **Hard.** 在莎士比亚全集上训练一个 1k-merge 的 BPE。比较常见词与罕见专有名词的 tokenization。测量前后的平均每词 token 数。写下让你意外的发现。
