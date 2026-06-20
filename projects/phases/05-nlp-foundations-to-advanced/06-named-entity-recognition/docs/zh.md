# 命名实体识别

> 把名字抽出来。听起来很简单，直到你遇到模糊边界、嵌套实体和领域术语。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## 问题

“Apple sued Google over its iPhone search deal in the US.” 里面有五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（也许算）、US（GPE）。好的 NER 系统会把它们全部抽出来，并给出正确类型。差的系统会漏掉 iPhone，把作为水果的 Apple 和作为公司的 Apple 混在一起，还会把 “US” 标成 PERSON。

NER 是每条结构化抽取 pipeline 下面的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人回答的 grounding、法律合同抽取。你不一定直接看见它，但你总是在依赖它。

这一课会沿着经典路线（rule-based、HMM、CRF）走到现代路线（BiLSTM-CRF，再到 transformers）。每一步都解决了前一步的一个具体限制。这个演进模式本身就是本课重点。

## 概念

**BIO tagging**（或 BILOU）把实体抽取变成 sequence-labeling 问题。给每个 token 打上 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）。

```text
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体会串起来：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意 span。

架构演进：

- **Rule-based.** Regex + gazetteer lookups。对已知实体 precision 高，对新实体 coverage 为零。
- **HMM.** Hidden Markov Model。给定 tag 的 token emission probability，以及 tag-to-tag transition probability。用 Viterbi decode。在标注数据上训练。
- **CRF.** Conditional Random Field。像 HMM，但它是 discriminative 的，所以可以混合任意 features（word shape、capitalization、neighboring words）。到 2026 年，在低资源部署里仍然是经典生产主力。
- **BiLSTM-CRF.** 用 neural features 替代手工特征。LSTM 双向读取句子，顶部 CRF layer 强制 tag sequence 保持一致。
- **Transformer-based.** Fine-tune BERT，加 token-classification head。准确率最好。计算量最大。

```figure
ner-bio-tagging
```

## 动手构建

### Step 1：BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Step 2：手工特征

对经典（非 neural）NER 来说，features 就是关键。常用特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词来说是高信号特征。

### Step 3：一个简单的 rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产 gazetteers 会有从 Wikipedia 和 DBpedia 抓取来的数百万条目。Coverage 很好。Disambiguation（公司 Apple vs 水果 apple）很糟。这就是统计模型胜出的原因。

### Step 4：CRF 这一步（草图，不做完整实现）

如果没有概率论基础，从零用 50 行写完整 CRF 并不会带来多少启发。这里改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 regularization。`all_possible_transitions=True` 让模型学习非法序列（例如 `O` 后面直接跟 `I-ORG`）概率很低，这就是 CRF 在你不用手写约束的情况下仍能强制 BIO consistency 的方式。

### Step 5：BiLSTM-CRF 增加了什么

Features 变成 learned。输入是 token embeddings（GloVe 或 fastText）。LSTM 从左到右、从右到左读取句子。拼接后的 hidden states 进入 CRF output layer。CRF 仍然负责强制 tag-sequence consistency；LSTM 用学到的表示替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF layer 可以用 `torchcrf.CRF`（pip install pytorch-crf）。相对 hand-crafted CRF，它的提升可以测到，但除非你有数万条标注句子，否则提升通常比你想象的小。

## 使用它

spaCy 开箱就带生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```text
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意，`iPhone` 被标成了 `ORG` 而不是 `PRODUCT`。spaCy small model 的产品实体覆盖较弱。large model（`en_core_web_lg`）会更好。transformer model（`en_core_web_trf`）还会更好。

Hugging Face 的 BERT-based NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```text
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X tokens 合并成一个 span。没有它，你拿到的是 token-level labels，必须自己合并。

### LLM-based NER（2026 年的选择）

Zero-shot 和 few-shot LLM NER 现在在许多领域已经能和 fine-tuned models 竞争；当标注数据稀缺时，它们的优势尤其明显。

- **Zero-shot prompting.** 给 LLM 一组 entity types 和一个 example schema。要求输出 JSON。开箱可用；在新领域上准确率中等。
- **ZeroTuneBio-style prompting.** 把任务拆成 candidate extraction → meaning explanation → judgment → re-check。多阶段 prompt（不是 one-shot）能显著提高 biomedical NER 准确率。同样模式也适用于法律、金融、科学领域。
- **Dynamic prompting with RAG.** 每次 inference 都从一个小型标注 seed set 中检索最相似的 labeled examples；动态构造 few-shot prompt。在 2026 年 benchmark 中，这能让 GPT-4 biomedical NER F1 比 static prompting 高 11-12%。
- **Per-entity-type decomposition.** 对长文档来说，一次调用同时抽取所有 entity types 时，长度越长 recall 越差。改成每种 entity type 单独跑一轮抽取。推理成本更高，但准确率明显更高。这是临床笔记和法律合同的标准模式。

截至 2026 年的生产建议：在收集训练数据之前，先做一个 LLM zero-shot baseline。很多时候 F1 已经足够好，你根本不需要 fine-tune。

### 经典 NER 仍然胜出的地方

即使有 LLM 可用，经典 NER 在这些场景里仍然会赢：

- Latency budget 低于 50ms。
- 你有数千条标注样本，并且需要 98%+ F1。
- 领域 ontology 稳定，pretrained CRF 或 BiLSTM 能很好迁移。
- 监管约束要求 on-prem、non-generative model。

### 它会在哪里崩掉

- **Domain shift.** 在 CoNLL 上训练的 NER 跑法律合同，可能比 gazetteer 还差。要在你的领域上 fine-tune。
- **Nested entities.** “Bank of America Tower” 同时是 ORG 和 FACILITY。标准 BIO 无法表示 overlapping spans。你需要 nested NER（multi-pass 或 span-based models）。
- **Long entities.** “United States Federal Deposit Insurance Corporation.” Token-level models 有时会把它拆开。使用 `aggregation_strategy` 或 post-process。
- **Sparse types.** 医疗 NER 标签可能是 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不知道这些。Scispacy 和 BioBERT 是这类任务的起点。

## 交付它

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **Easy.** 实现 `bio_to_spans`（`spans_to_bio` 的反向函数），并在 10 个句子上验证 round-trip consistency。
2. **Medium.** 在 CoNLL-2003 English NER dataset 上训练上面的 sklearn-crfsuite CRF。用 `seqeval` 报告 per-entity F1。典型结果：约 84 F1。
3. **Hard.** 在一个特定领域 NER dataset（医疗、法律或金融）上 fine-tune `distilbert-base-cased`。和 spaCy small model 对比。记录 data leakage checks，并写下让你意外的发现。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NER | 抽取名字 | 给 token spans 打类型标签（PERSON、ORG、GPE、DATE，等等）。 |
| BIO | Tagging scheme | `B-X` 表示开始，`I-X` 表示继续，`O` 表示外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（last）和 `U-X`（unit），让边界更清楚。 |
| CRF | Structured classifier | 建模 labels 之间的 transitions，而不只是 emissions。强制序列有效。 |
| Nested NER | Overlapping entities | 一个 span 和它的 sub-span 分别是不同实体。BIO 无法表达。 |
| Entity-level F1 | 正确的 NER 指标 | 预测 span 必须和真实 span 完全一致。Token-level F1 会夸大准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。Canonical。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入后来成为标准的 token-classification pattern。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个 attribute 的实用参考。
