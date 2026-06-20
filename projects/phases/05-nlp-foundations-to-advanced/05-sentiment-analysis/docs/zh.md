# 情感分析

> NLP 的经典任务。关于 classical text classification 你需要知道的大多数内容，都会在这里出现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## 问题

"The food was not great." 是正面还是负面？

情感分析听起来很简单。评论者说自己喜欢或不喜欢某个东西。给句子打标签。但它之所以成为 NLP 的经典任务，是因为每个看起来容易的 case 背后都藏着难点。Negation 会翻转含义。Sarcasm 会反转含义。"Not bad at all" 虽然包含两个负面编码的词，却是正面的。Emojis 可能比周围文本携带更多信号。Domain vocabulary 很重要，例如音乐评论里的 `tight` 和时尚评论里的 `tight` 含义不同。

Sentiment 是 classical NLP 的工作实验室。如果你理解每个 naive baseline 为什么都有具体 failure mode，你也就理解了每个更丰富的 model 为什么会被发明出来。本课会从零构建一个 Naive Bayes baseline，加入 logistic regression，并指出那些让 production sentiment 变成 compliance-grade problem 的陷阱。

## 概念

Classical sentiment 是一个两步 recipe。

1. **Represent.** 把文本变成 feature vector。可以用 BoW、TF-IDF 或 n-grams。
2. **Classify.** 在 labeled examples 上拟合 linear model，例如 Naive Bayes、logistic regression 或 SVM。

Naive Bayes 是能工作的最笨 model。它假设在给定 label 后，每个 feature 都彼此 independent。根据 counts 估计 `P(word | positive)` 和 `P(word | negative)`。Inference 时，把 probabilities 相乘。"naive" 的 independence assumption 错得有点好笑，但结果却强得惊人。原因是：在 sparse text features 和中等规模 data 下，classifier 更关心每个 word 倾向哪一侧，而不是精确倾向多少。

Logistic regression 修复了 independence assumption。它为每个 feature 学一个 weight，包括 negative weights。`not good` 作为 bigram feature 会得到负权重。Naive Bayes 无法对它从未标注过的 bigrams 做到这一点。

```figure
sentiment-logits
```

## Build It

### Step 1: 一个真实的 mini-dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

有意保持很小。真实工作会使用数万条 examples（IMDb、SST-2、Yelp polarity）。数学是一样的。

### Step 2: 从零实现 multinomial Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing（alpha=1.0）就是 Laplace smoothing。没有它，某个 class 中未见过的 word 会有 zero probability，log 会爆掉。实践中常见 `alpha=0.01`。`alpha=1.0` 是教学默认值。

### Step 3: 从零实现 logistic regression

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization 在这里很重要。Text features 是 sparse 的；没有 L2，model 会记住 training examples。从 `0.01` 开始再 tune。

### Step 4: 处理 negation（failure mode）

考虑 "not good" 和 "not bad"。BoW classifier 看到的是 `{not, good}` 和 `{not, bad}`，然后根据 training 中哪种更多来学习。Bigram classifier 看到的是 `not_good` 和 `not_bad`，会把它们学习成不同 features。通常这就足够了。

如果你没有 bigrams，一个更粗糙但有效的修复方式是：**negation scoping**。在 negation word 之后，直到下一个 punctuation 之前，给 tokens 加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同 features。Classifier 可以给它们相反的 weight。三行 preprocessing，就能在 sentiment benchmarks 上带来可测量的 accuracy jump。

### Step 5: 真正重要的 evaluation metrics

如果 classes imbalanced，只看 accuracy 会误导人。真实 sentiment corpora 通常是 70-80% positive 或 70-80% negative；constant-majority classifier 可以拿到 80% accuracy，但毫无价值。下面每一项都要报告：

- **Per-class precision and recall.** 每个 class 一组。对它们做 macro-average，得到一个尊重 class balance 的单个数字。
- **Macro-F1（imbalanced data 的 primary metric）.** Per-class F1 scores 的平均值，等权重。Classes imbalanced 时用它代替 accuracy。
- **Weighted-F1（alternative）.** 和 macro 类似，但按 class frequency 加权。当 imbalance 本身有 business meaning 时，和 macro-F1 一起报告。
- **Confusion matrix.** 原始 counts。在信任任何 scalar metric 之前，一定先检查它；它会揭示 model 混淆的是哪一对 classes。
- **Per-class error samples.** 每个 class 抽 5 个错误 predictions。读它们。没有什么可以替代直接阅读真实 errors。

对于严重 imbalanced data（> 95-5 ratio），报告 **AUROC** 和 **AUPRC**，不要报告 accuracy。AUPRC 对 minority class 更敏感，而这通常正是你关心的对象（spam、fraud、rare sentiment）。

**要避免的常见 bug。** 在 imbalanced data 上报告 micro-F1 而不是 macro-F1，会得到一个看起来很高的数字，因为它被 majority class 主导。Macro-F1 会强迫你看到 minority-class performance。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Use It

scikit-learn 可以用六行正确完成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三件事。`stop_words=None` 会保留 negations。`ngram_range=(1, 2)` 会加入 bigrams，所以 `not_good` 会成为一个 feature。`sublinear_tf=True` 会压低重复 words 的影响。这三个 flags 是 SST-2 上 75%-accurate baseline 和 85%-accurate baseline 的差别。

### 什么时候该用 transformer

- Sarcasm detection。Classical models 在这里会失败。没有例外。
- Sentiment 在文档中途转向的 long reviews。
- Aspect-based sentiment。"Camera was great but battery was terrible." 你需要把 sentiment 归因到 aspects。只能用 transformers 或 structured output models。
- Non-English、low-resource languages。Multilingual BERT 会免费给你一个 zero-shot baseline。

如果你需要上面任意一项，跳到 phase 7（transformers deep dive）。否则，在 TF-IDF + bigrams + negation handling 上使用 Naive Bayes 或 logistic regression，就是你的 2026 production baseline。

### Reproducibility trap（又一次）

Retraining sentiment models 是常规操作。Re-evaluating 它们却不是。Papers 中报告的 accuracy numbers 使用特定 split、特定 preprocessing、特定 tokenizer。如果你拿新 model 和一个没有使用相同 pipeline 的 baseline 比较，就会得到误导性的 deltas。永远在你自己的 pipeline 上重新生成 baseline，而不是使用 paper 的数字。

## Ship It

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **Easy.** 把 `apply_negation` 作为 preprocessing step 加入 scikit-learn pipeline，并在一个小型 sentiment dataset 上测量 F1 delta。
2. **Medium.** 实现 class-weighted logistic regression（在 scikit-learn 中传入 `class_weight="balanced"`，或者自己推导 gradient）。在 synthetic 90-10 class imbalance 上测量效果。
3. **Hard.** 在 sentiment model 的 residuals 上训练第二个 classifier，构建 sarcasm detector。记录你的 experimental setup。当 accuracy 低于 chance 时提醒读者（2-class sarcasm 的 chance-level 约为 50%，大多数第一次尝试都会落在那里）。

## 关键术语

| Term | 人们通常怎么说 | 它实际是什么意思 |
|------|----------------|------------------|
| Polarity | Positive or negative | Binary label；有时扩展到 neutral 或 fine-grained（5-star）。 |
| Aspect-based sentiment | Per-aspect polarity | 把 sentiment 归因到文本中提到的具体 entities 或 attributes。 |
| Negation scoping | Reversing nearby tokens | 在 "not" 之后的 tokens 前加 `NOT_`，直到 punctuation。 |
| Laplace smoothing | Adding 1 to counts | 防止 Naive Bayes 中出现 zero-probability features。 |
| L2 regularization | Shrinking weights | 在 loss 中加入 `lambda * sum(w^2)`。对 sparse text features 至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — foundational survey。很长，但前四节覆盖了 classical 部分的全部核心。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 这篇 paper 表明，在 short text 上，bigrams + Naive Bayes 很难被击败。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 和每个你会 tune 的 knob 的 reference。
