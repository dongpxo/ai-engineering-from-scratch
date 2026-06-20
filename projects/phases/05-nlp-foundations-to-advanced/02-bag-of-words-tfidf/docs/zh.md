# 词袋、TF-IDF 与文本表示

> 先计数，再思考。即使到了 2026 年，在定义清楚的任务上，TF-IDF 仍然常常胜过 embedding。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01（Text Processing），Phase 2 · 02（Linear Regression from Scratch）
**Time:** ~75 minutes

## 问题

模型需要数字。你手里只有字符串。

每条 NLP 流水线都必须回答同一个问题：怎样把长度可变的 token 流，变成分类器能吃进去的固定长度向量？这个领域最早稳定下来的答案，是那个最笨但有效的答案：数词。做一个向量。

这个向量承载过的生产级 NLP，比任何 embedding 模型都多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP benchmark 的第一个十年。到了 2026 年，实践者在狭窄分类任务上仍然会先试它。它快、可解释，而且在“词是否出现”才是关键的任务上，经常和 4 亿参数 embedding 模型没有可见差别。

这一课会从零构建 bag of words，再构建 TF-IDF。然后展示 scikit-learn 如何用三行代码做同一件事。最后说明什么时候它会失败，以及什么时候该换成 embedding。

## 概念

**Bag of Words（BoW，词袋）** 会丢掉顺序。对每篇文档，统计词表中每个词出现了多少次。向量长度就是词表大小。位置 `i` 存的是第 `i` 个词的计数。

**TF-IDF** 会重新加权 BoW。一个出现在每篇文档里的词没有多少信息量，所以要压低权重。一个在语料库中很少见、但在某篇文档里频繁出现的词通常是信号，所以要放大权重。

```text
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的频率，`df` 是文档频率（包含该词的文档数量），`N` 是总文档数。`log` 会让到处都出现的词的权重保持有界。

关键性质：两者都会产生稀疏向量，并且每个轴都可以解释。你可以查看训练后分类器的权重，直接读出哪些词把文档推向哪个类别。对 768 维 BERT embedding，你无法这样做。

```figure
bow-tfidf
```

## Build It

### Step 1：构建词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：已经分好词的文档列表（任何词级 tokenizer 都可以；本课 `code/main.py` 使用了一个简化的小写版本）。输出：`{word: index}` 字典。稳定的插入顺序意味着，索引 0 是第一篇文档里第一次出现的词。不同实现会有不同约定；scikit-learn 会按字母排序。

### Step 2：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词表索引。条目 `[i][j]` 表示“第 `j` 个词在第 `i` 篇文档里出现了多少次”。文档 1 里 `cat` 是 2，因为它确实出现了两次。文档 0 里 `ran` 是 0，因为它没有出现。

### Step 3：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

这里有两个值得点名的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保一个出现在所有文档里的词仍然有 IDF 1（而不是 0），这与 scikit-learn 的默认行为一致。其他实现会使用原始的 `log(N/df)`。两种都能用；平滑版本更友好。

### Step 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词表词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在三篇文档里，所以它的 IDF 低。`dog` 只出现一次，所以它的 IDF 高。这些向量是稀疏的（大多数条目很小），有区分力的词会凸显出来。

### Step 5：对每行做 L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不归一化，较长文档会得到更大的向量，并主导相似度分数。L2 归一化会把每篇文档放到单位超球面上。此时，行与行之间的 cosine similarity 就只是点积。

## Use It

scikit-learn 提供了生产级版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 会在一次调用里完成 tokenization、词表构建和 BoW。`TfidfVectorizer` 会再加上 IDF 加权和 L2 归一化。两者都会返回稀疏矩阵。对于 10 万篇文档，dense 版本放不进内存；除非分类器强制要求 dense，否则保持 sparse。

会彻底改变结果的几个旋钮：

| Arg | Effect |
|-----|--------|
| `ngram_range=(1, 2)` | 包含 bigram。通常会提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 篇文档中的词。在噪声数据上可以裁剪词表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。不用硬编码列表，也能近似移除 stopword。 |
| `stop_words="english"` | scikit-learn 内置 stopword 列表。依赖任务；情感分析不应该删除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)`，而不是原始 `tf`。当某个词在一篇文档里重复很多次时有帮助。 |

### TF-IDF 到 2026 年仍然会赢的场景

- 垃圾邮件检测、主题打标、日志异常标记。词是否出现才是重点；语义细微差别不是。
- 低数据量场景（几百个标注样本）。TF-IDF 加 logistic regression 不需要预训练成本。
- 延迟敏感的任何地方。TF-IDF 加线性模型可以在微秒级回答。让 transformer 给文档做 embedding 通常需要 10-100ms。
- 必须解释预测的系统。检查分类器系数即可。最大的正向词就是原因。

### TF-IDF 会失败的场景

语义盲区。考虑这两篇文档：

- "The movie was not good at all."
- "The movie was excellent."

一篇是负面评论，一篇是正面评论。它们的 TF-IDF 重叠正好是 `{the, movie, was}`。词袋分类器必须记住 `not` 靠近 `good` 时会翻转标签。有足够数据时它能学到，但永远不会像理解 syntax 的模型那样优雅。

另一个失败点：推理时遇到词表外词。一个在 IMDb 评论上训练的 BoW 模型，如果训练时从没见过 `Zoomer-approved` 这个 token，就完全不知道该怎么处理。subword embeddings（第 04 课）可以处理这种情况。TF-IDF 不能。

### Hybrid：TF-IDF 加权 embedding

2026 年，中等数据量分类任务的实用默认方案是：用 TF-IDF 权重作为 word embeddings 上的 attention。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从 embedding 得到语义能力，又从 TF-IDF 得到罕见词强调。分类器在 pooled vector 上训练。在大约 5 万个标注样本以下的情感、主题、意图分类中，它通常会胜过单独使用任意一边。

## Ship It

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **（Easy）** 在 L2 归一化后的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档的分数是 1.0，词表完全不相交的文档分数是 0.0。
2. **（Medium）** 给 `bag_of_words` 增加 `n-gram` 支持。参数 `n` 会统计 `n`-gram。测试 `n=2` 时，`["the", "cat", "sat"]` 会产生 `["the cat", "cat sat"]` 的 bigram 计数。
3. **（Hard）** 用 GloVe 100d vectors（下载一次并缓存）构建上面的 TF-IDF-weighted-embedding hybrid。在 20 Newsgroups 数据集上，把分类准确率与普通 TF-IDF、普通 mean-pooled embeddings 进行比较。报告哪种方法在哪些场景下获胜。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| BoW | 词频向量 | 一篇文档中词表词的计数。会丢掉顺序。 |
| TF | 词频 | 一个词在一篇文档中的计数，可选地按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数量。 |
| IDF | 逆文档频率 | 平滑后的 `log(N / df)`。会压低到处都出现的词。 |
| Sparse vector | 大多是零 | 词表通常有 1 万到 10 万个词；任意单篇文档里大多数词都不存在。 |
| Cosine similarity | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)：规范 API 参考，也包含每个旋钮的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210)：让 TF-IDF 成为十年默认方法的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2)：关于旧方法何时获胜、为什么获胜的 2026 年视角。
