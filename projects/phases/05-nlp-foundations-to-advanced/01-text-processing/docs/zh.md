# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理就是中间的桥。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题

模型不能直接读懂 “The cats were running.”。它读到的是整数。

每个 NLP 系统开头都会面对同样三个问题。一个词从哪里开始。词根是什么。当有帮助时，怎样把 “run”、“running”、“ran” 当作同一件事；当不该混在一起时，又怎样把它们当作不同的东西。

分词错了，模型就会从垃圾里学习。如果 tokenizer 把 `don't` 当作一个 token，但另一个流程把它拆成 `do n't` 两个 token，训练分布就被拆开了。如果 stemmer 把 `organization` 和 `organ` 压成同一个 stem，主题建模就会崩掉。如果 lemmatizer 需要词性上下文，而你没有传入，动词就会被当作名词处理。

这一课会从零构建三个预处理步骤，然后展示 NLTK 和 spaCy 怎样做同一件事，让你看清其中的取舍。

## 概念

三个操作。每个都有自己的职责和失败模式。

**Tokenization（分词）** 把字符串切分成 tokens。“Token” 这个词刻意保持模糊，因为正确粒度取决于任务。经典 NLP 常用 word-level。Transformer 常用 subword。没有空格的语言可能用 character。

**Stemming（词干提取）** 用规则砍掉后缀。快，激进，也很笨。`running -> run`。`organization -> organ`。第二个就是失败模式。

**Lemmatization（词形还原）** 使用语法知识，把词还原到词典形式。更慢，更准确，需要 lookup table 或 morphological analyzer。`ran -> run`（需要知道 “ran” 是 “run” 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且可以容忍噪声时，用 stemming（搜索索引、粗略分类）。当语义重要时，用 lemmatization（问答、语义搜索、任何用户会直接阅读的内容）。

```figure
edit-distance
```

## 动手构建

### Step 1：一个正则词级 tokenizer

最简单可用的 tokenizer 会按非字母数字字符切分，同时把标点保留为独立 token。它不完美，也不是最终方案，但一行就能跑起来。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列。带可选内部撇号的单词（`don't`、`it's`）。纯数字。任何单个非空白、非字母数字字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式：`3pm` 会被拆成 `['3', 'pm']`，因为我们在字母串和数字串之间做了 alternation。对多数任务来说这已经够用。URL、email、hashtag 都会被拆坏。生产环境里，要把这些模式放在更通用的模式之前。

### Step 2：一个 Porter stemmer（只做 step 1a）

完整 Porter algorithm 有五个阶段的规则。只做 step 1a 就能覆盖最常见的英语后缀，也足够讲清这个模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下读这些规则。`ies -> i` 这条规则解释了为什么 `ponies -> poni`，而不是 `pony`。真正的 Porter step 1b 会继续修正它。规则之间会竞争。更早的规则获胜。顺序比任何单条规则都更重要。

### Step 3：一个基于 lookup table 的 lemmatizer

真正的 lemmatization 需要形态学。一个可控的教学版本可以用小型 lemma table 加 fallback。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键教学点。`watched` 不在表里，而我们的 fallback 只处理 `ing`。真正的 lemmatization 会覆盖 `ed`、不规则动词、比较级形容词、带音变的复数（`children -> child`）。这就是为什么生产系统会使用 WordNet、spaCy 的 morphologizer，或完整的 morphological analyzer。

### Step 4：把它们串起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是 POS tagger。Phase 5 · 07（POS Tagging）会构建一个。现在先把所有内容默认当作 `NOUN`，并明确承认这个限制。

## 使用它

NLTK 和 spaCy 都提供了生产级版本。每个只需要几行。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 会处理 contractions、Unicode，以及你的正则漏掉的边界情况。`PorterStemmer` 会运行全部五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 词性标签翻译成 WordNet 的缩写集合。上面的翻译胶水代码，就是多数教程会跳过的那一小段。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```text
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整条 pipeline 都藏在 `nlp(text)` 后面。Tokenization、POS tagging 和 lemmatization 都会运行。大规模场景下比 NLTK 更快。开箱准确率也更好。代价是你不太容易替换单个组件。

### 什么时候选哪个

| Situation | Pick |
|-----------|------|
| 教学、研究、替换组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer pipeline（无论如何都会用模型自己的 tokenizer） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 两个几乎没人提醒你的失败模式

多数教程讲完算法就停了。真实 preprocessing pipeline 里有两个问题会咬人，而且它们几乎从不被覆盖。

**Reproducibility drift（可复现性漂移）。** NLTK 和 spaCy 会在不同版本之间改变 tokenization 和 lemmatizer 行为。spaCy 2.x 里产生 `['do', "n't"]` 的输入，在 3.x 里可能产生 `["don't"]`。你的模型是在一个分布上训练的。现在 inference 却跑在另一个分布上。准确率悄悄变差，却没人知道原因。把库版本固定在 `requirements.txt` 里。写一个 preprocessing regression test，冻结 20 个样例句子的预期 tokenization。每次升级都跑它。

**Training / inference mismatch（训练/推理不一致）。** 训练时使用激进预处理（lowercase、stopword removal、stemming），部署时却直接喂原始用户输入，性能会直接塌掉。这是生产 NLP 中最常见的失败。如果训练时做了预处理，推理时必须运行完全相同的函数。把 preprocessing 作为函数随模型包一起发布，而不是留成一个由服务团队重写的 notebook cell。

## 交付它

一个可复用的 prompt，用来帮助工程师选择 preprocessing strategy，而不需要先读三本教材。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **Easy.** 扩展 `tokenize`，让 URL 保持为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产生一个 URL token。
2. **Medium.** 实现 Porter step 1b。如果一个词包含元音并以 `ed` 或 `ing` 结尾，就移除这个后缀。处理双辅音规则（`hopping -> hop`，不是 `hopp`）。
3. **Hard.** 构建一个 lemmatizer：优先把 WordNet 当作 lookup table；当 WordNet 没有 entry 时，fallback 到你的 Porter stemmer。在带标签语料上，把准确率和 plain WordNet、plain Porter 做比较。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的任意单位。可以是 word、subword、character 或 byte。 |
| Stem | 词根 | 基于规则剥离后缀得到的结果。不一定是真实单词。 |
| Lemma | 词典形式 | 你会去词典里查的形式。需要语法上下文才能准确计算。 |
| POS tag | 词性 | NOUN、VERB、ADJ 这类类别。准确 lemmatization 需要它。 |
| Morphology | 词形规则 | 单词如何根据时态、数量、格等变化形式。Lemmatization 依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今仍是最清楚的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实 pipeline 是怎样连接起来的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的 tokenization 边界情况。
