# 词性标注与句法解析

> 语法曾经有一段时间不时髦。后来每个 LLM 流水线都需要验证结构化抽取，语法又回来了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 学习目标

- 解释 POS tag、Penn Treebank tagset 与 Universal Dependencies tagset 的区别
- 实现 most-frequent-tag baseline，并理解它为什么能成为词性标注的下限
- 用转移概率、发射概率和 Viterbi 解码实现一个 bigram HMM tagger
- 区分 constituency parsing 与 dependency parsing 的输出结构
- 判断什么时候应该停止从零实现，并在生产流水线中调用 spaCy、stanza 或 trankit

## 问题

第 01 课承诺过：词形还原需要词性标签。如果不知道 `running` 是动词，lemmatizer 就不能把它还原成 `run`。如果不知道 `better` 是形容词，它也不能把它还原成 `good`。

这个承诺背后藏着一个完整子领域。词性标注会为每个 token 分配语法类别。句法解析会恢复句子的树结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年打磨这两件事。后来深度学习把它们压缩成预训练 Transformer 之上的 token classification 任务，研究社区也就继续向前走了。

但应用社区没有放下它们。每个结构化抽取流水线的底层仍然会用到 POS 和依存树。LLM 生成的 JSON 会根据语法约束做验证。问答系统会用依存句法解析拆解查询。机器翻译质量评估器会检查 parse tree 的对齐情况。

值得了解。本课会介绍 tagset、baseline，以及你应该停止从零实现、转而调用 spaCy 的那个边界。

## 概念

**词性标注（POS tagging）** 会给每个 token 标上语法类别。**Penn Treebank（PTB）** tagset 是英语里的默认选择。它有 36 个标签，区分粒度对普通读者来说有点挑剔：`NN` 单数名词、`NNS` 复数名词、`NNP` 单数专有名词、`VBD` 过去式动词、`VBZ` 第三人称单数现在时动词，等等。**Universal Dependencies（UD）** tagset 更粗（17 个标签），并且与具体语言无关；它已经成为跨语言工作的默认选择。

```text
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法解析（syntactic parsing）** 会产出一棵树。两种主要风格：

- **成分句法解析（constituency parsing）。** 名词短语、动词短语、介词短语彼此嵌套。输出是一棵由非终结类别（NP、VP、PP）构成的树，单词是叶子节点。
- **依存句法解析（dependency parsing）。** 每个词都有一个它依赖的 head word，并带有一个语法关系标签。输出是一棵树，其中每条边都是一个 `(head, dependent, relation)` 三元组。

依存句法解析在 2010 年代胜出，因为它能干净地跨语言泛化，尤其适合词序自由的语言。

```text
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build It

### Step 1：most-frequent-tag baseline

最笨但能工作的 POS tagger。对每个词，预测它在训练集中最常出现过的 tag。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown corpus 上，这个 baseline 能达到约 85% accuracy。不算好，但这是任何严肃模型都不该低于的下限。

### Step 2：bigram HMM tagger

对序列的联合概率建模：

```text
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（给定前一个 tag 的当前 tag 概率）和发射概率（给定 tag 的 word 概率）。用计数和 Laplace smoothing 估计两者。用 Viterbi 解码（在 tag lattice 上做动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 上的 bigram HMM 能达到约 93% accuracy。从 85% 跳到 93%，主要靠的是转移概率：模型学会了 `DET NOUN` 很常见，而 `NOUN DET` 很少见。

### Step 3：为什么现代 tagger 会超过它

转移概率和发射概率都是局部的。它们捕捉不到 `saw` 在 "I bought a saw" 里是名词，但在 "I saw the movie" 里是动词。带任意特征的 CRF（后缀、词形、前后词、词本身）能达到约 97%。BiLSTM-CRF 或 Transformer 能达到 98% 以上。

这个任务的上限由标注者分歧决定。人类标注者在 Penn Treebank 上的一致率大约是 97%。超过 98% 的模型很可能已经在过拟合测试集。

### Step 4：依存句法解析速写

从零完整实现依存句法解析超出了本课范围；经典教材处理见 Jurafsky 和 Martin。需要知道两个经典家族：

- **基于转移（transition-based）** 的 parser（arc-eager、arc-standard）像 shift-reduce parser 一样工作：它们读取 token，把 token shift 到栈上，再应用创建 arc 的 reduce action。贪心解码速度很快。经典实现是 MaltParser。现代神经版本是 Chen 和 Manning 的 transition-based parser。
- **基于图（graph-based）** 的 parser（Eisner's algorithm、Dozat-Manning biaffine）会为每条可能的 head-dependent edge 打分，然后选择最大生成树。更慢，但更准确。

对大多数应用工作，调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```text
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就会浮现出来。

## Use It

每个生产级 NLP 库都会把 POS 和依存句法 parser 作为标准流水线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快、准，并与 tokenization、NER、lemmatization 集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford 的 CoreNLP 后继者。在 60 多种语言上达到一线水平。
- **trankit**。基于 Transformer，UD accuracy 很好。
- **NLTK**。`pos_tag`。可用，但慢、偏老。适合教学。

### 2026 年它仍然重要的地方

- **词形还原。** 第 01 课需要 POS 才能正确 lemmatize。永远如此。
- **从 LLM 输出做结构化抽取。** 验证生成句子是否满足语法约束（例如主谓一致、必需修饰语）。
- **基于方面的情感分析。** 依存句法解析会告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 可以通过 parse 拆成结构化约束。
- **跨语言迁移。** UD tag 和依存关系与具体语言无关，可以对新语言做 zero-shot 结构化分析。
- **低计算量流水线。** 如果你不能部署 Transformer，POS + dependency parse + gazetteer 也能走得相当远。

## Ship It

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **Easy.** 在一个小型已标注语料（例如 NLTK 的 Brown 子集）上使用 most-frequent-tag baseline，测量 held-out 句子的 accuracy。验证约 85% 的结果。
2. **Medium.** 训练上面的 bigram HMM，并报告每个 tag 的 precision/recall。HMM 最容易混淆哪些 tag？
3. **Hard.** 使用 spaCy 的依存句法解析，从 1000 句样本中抽取 subject-verb-object triples。在 50 个手工标注 triple 上评估。记录抽取失败的位置（通常是被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| POS tag | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准 tagset | 英语专用。细分动词时态和名词数。 |
| Universal Dependencies | 多语言 tagset | 比 PTB 更粗；语言中立；是跨语言工作的默认选择。 |
| Dependency parse | 句子树 | 每个词有一个 head，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 在给定 emissions 和 transitions 时，找出概率最高的 tag 序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing，第 8 章和第 18 章](https://web.stanford.edu/~jurafsky/slp3/) — POS 与 parsing 的经典教材处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 跨语言 tagset 与 treebank 集合，每个多语言 parser 都会用到。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 暴露的每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 把神经 parser 带入主流的论文。
