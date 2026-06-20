# 处理不平衡数据

> 当 99% 的 data 都是 “normal” 时，accuracy 就是谎言。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-09 (especially evaluation metrics)
**Time:** ~90 minutes

## 学习目标

- 从零实现 SMOTE，并解释 synthetic oversampling 和 random duplication 的区别
- 使用 F1、AUPRC 和 Matthews Correlation Coefficient 评估 imbalanced classifiers，而不是 accuracy
- 比较 class weighting、threshold tuning 和 resampling strategies，并为给定 imbalance ratio 选择正确 approach
- 构建一个完整 imbalanced data pipeline，组合 SMOTE、class weights 和 threshold optimization

## 问题

你构建了一个 fraud detection model。它得到 99.9% accuracy。你开始庆祝。然后你发现它对每一笔 transaction 都预测 “not fraud”。

这不是 bug。当只有 0.1% transactions 是 fraudulent 时，这反而是理性的做法。Model 学会了永远猜 majority class 可以最小化 overall error。它技术上正确，但完全没用。

这种情况出现在所有真正重要的 classification 场景里。Disease diagnosis：1% positive rate。Network intrusion：0.01% attacks。Manufacturing defects：0.5% defective。Spam filtering：20% spam。Churn prediction：5% churners。Minority class 越重要，它往往越稀有。

Accuracy 失败，是因为它把所有 correct predictions 都当成同等价值。正确标记一笔 legitimate transaction，和正确抓到 fraud，都只算 accuracy 的一分。但抓到 fraud 才是这个 model 存在的全部理由。我们需要 metrics、techniques 和 training strategies，强迫 model 关注 rare but important class。

## 概念

### 为什么 Accuracy 失效

考虑一个有 1000 个 samples 的 dataset：990 个 negative，10 个 positive。一个永远预测 negative 的 model：

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| Actually Positive | 0 (TP) | 10 (FN) |
| Actually Negative | 0 (FP) | 990 (TN) |

Accuracy = (0 + 990) / 1000 = 99.0%

这个 model 抓到 zero fraud。Zero disease。Zero defects。但 accuracy 说它有 99%。这就是为什么 accuracy 对 imbalanced problems 很危险。

### 更好的 Metrics

**Precision** = TP / (TP + FP)。所有被 flag 为 positive 的样本中，多少是真的？High precision 意味着 false alarms 少。

**Recall** = TP / (TP + FN)。所有真实 positive 中，我们抓到了多少？High recall 意味着 missed positives 少。

**F1 Score** = 2 * precision * recall / (precision + recall)。Harmonic mean。相比 arithmetic mean，它会更强地惩罚 precision 和 recall 之间的极端不平衡。

**F-beta Score** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。当 beta > 1，recall 更重要。当 beta < 1，precision 更重要。Fraud detection 中常用 F2（漏掉 fraud 比 false alarm 更糟）。

**AUPRC**（Area Under Precision-Recall Curve）。类似 AUC-ROC，但对 imbalanced data 信息量更高。Random classifier 的 AUPRC 等于 positive class rate（不像 ROC 那样是 0.5）。这让 improvements 更容易被看见。

**Matthews Correlation Coefficient** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。范围从 -1 到 +1。只有当 model 在两个 classes 上都做得好，才会得到 high score。即使 classes size 差异很大，它也保持 balanced。

对于上面那个 “always predict negative” model：precision = 0/0（undefined，通常设为 0），recall = 0/10 = 0，F1 = 0，MCC = 0。这些 metrics 会正确识别这个 model 没有价值。

### Imbalanced Data Pipeline

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### SMOTE：Synthetic Minority Oversampling Technique

Random oversampling 会复制已有 minority samples。这能工作，但容易 overfit，因为 model 会反复看到 identical points。

SMOTE 会创建新的 synthetic minority samples：它们 plausible，但不是 copies。Algorithm：

1. 对每个 minority sample x，在其他 minority samples 中找到 k nearest neighbors
2. 随机选择一个 neighbor
3. 在 x 和这个 neighbor 之间的 line segment 上创建新 sample

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这会在真实 minority points 之间 interpolation，在 feature space 的同一区域创建 samples，而不是只复制 existing data。

```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### Sampling Strategies Compared

**Random Oversampling**：复制 minority samples，让它匹配 majority count。
- Pros：简单，没有 information loss
- Cons：exact duplicates 会造成 overfitting，增加 training time

**Random Undersampling**：移除 majority samples，让它匹配 minority count。
- Pros：training 快，简单
- Cons：丢弃可能有用的 majority data，variance 更高

**SMOTE**：通过 interpolation 创建 synthetic minority samples。
- Pros：生成新 data points，相比 random oversampling 减少 overfitting
- Cons：可能在 decision boundary 附近创建 noisy samples，不考虑 majority class distribution

| Strategy | Data Changed | Risk | When to Use |
|----------|-------------|------|-------------|
| Oversample | Minority duplicated | Overfitting | Small datasets, moderate imbalance |
| Undersample | Majority removed | Information loss | Large datasets, want fast training |
| SMOTE | Synthetic minority added | Boundary noise | Moderate imbalance, enough minority samples for k-NN |

### Class Weights

不要改变 data，而是改变 model 如何看待 errors。给 minority class 的 misclassification 更高 weight。

对于一个 binary problem，950 个 negative samples，50 个 positive samples：
- Negative class weight = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- Positive class weight = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

Positive class 得到 19x weight。错分一个 positive sample 的 cost 等于错分 19 个 negative samples。Model 被迫关注 minority class。

在 logistic regression 中，这会修改 loss function：

```text
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于 sample i 的 class。

Class weights 在 expectation 上和 oversampling 数学等价，但不创建新 data points。这让它更快，也避免 duplicated samples 带来的 overfitting 风险。

### Threshold Tuning

大多数 classifiers 会输出 probability。默认 threshold 是 0.5：如果 P(positive) >= 0.5，就预测 positive。但 0.5 是任意的。当 classes imbalanced 时，optimal threshold 通常低得多。

流程：
1. Train 一个 model
2. 在 validation set 上得到 predicted probabilities
3. 从 0.0 到 1.0 扫描 thresholds
4. 在每个 threshold 上计算 F1（或你选择的 metric）
5. 选择最大化 metric 的 threshold

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

某个 fraudulent transaction 可能得到 P(fraud) = 0.15。在 threshold 0.5 下，它会被分类为 not fraud。在 threshold 0.10 下，它会被正确抓到。Probability calibration 不如 ranking 重要：只要 fraud 的 probabilities 高于 non-fraud，就存在能分开它们的 threshold。

### Cost-Sensitive Learning

Class weights 的 generalization。不要使用统一 costs，而是指定具体 misclassification costs：

| | Predict Positive | Predict Negative |
|--|---|---|
| Actually Positive | 0 (correct) | C_FN = 100 |
| Actually Negative | C_FP = 1 | 0 (correct) |

漏掉一笔 fraudulent transaction（FN）的 cost 是 false alarm（FP）的 100 倍。Model 优化 total cost，而不是 total error count。

当你能估计 real-world costs 时，这是最 principled approach。Missed cancer diagnosis 的 cost 和导致 extra biopsy 的 false alarm 完全不同。显式写出这些 costs，会迫使 model 做正确 tradeoff。

### Decision Flowchart

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

```figure
class-imbalance
```

## Build It

### Step 1：生成 imbalanced dataset

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### Step 2：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### Step 3：Random oversampling 和 undersampling

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### Step 4：带 class weights 的 Logistic regression

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### Step 5：Threshold tuning

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### Step 6：Evaluation functions

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### Step 7：比较所有 approaches

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件会在一个 script 中运行所有这些内容，并打印 results。

## Use It

使用 scikit-learn 和 imbalanced-learn，这些 techniques 都是一行级别的调用：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

From-scratch implementations 会展示每种 technique 到底在做什么。SMOTE 只是 minority class 上的 k-NN interpolation。Class weights 会乘进 loss。Threshold tuning 是一个扫描 cutoffs 的 for-loop。没有魔法。

## Ship It

本课产出：
- `outputs/skill-imbalanced-data.md`，一个用于处理 imbalanced classification problems 的 decision checklist

## 练习

1. **Borderline-SMOTE**：修改 SMOTE implementation，只为接近 decision boundary 的 minority points 生成 synthetic samples（它们的 k-nearest neighbors 中包含 majority class samples）。在 classes overlap 的 dataset 上与 standard SMOTE 比较 results。

2. **Cost matrix optimization**：实现 cost-sensitive learning，把 cost matrix 作为 parameter。创建一个函数，接收 cost matrix 并返回最小化 expected cost 的 optimal predictions。使用不同 cost ratios（1:10、1:100、1:1000）测试，并画出 precision-recall tradeoff 如何变化。

3. **Threshold calibration**：实现 Platt scaling（在 model raw outputs 上 fit 一个 logistic regression 来产生 calibrated probabilities）。比较 calibration 前后的 precision-recall curve。展示 calibration 不会改变 ranking（AUC 保持不变），但会让 probabilities 更有意义。

4. **Ensemble with balanced bagging**：训练多个 models，每个 model 都在一个 balanced bootstrap sample 上训练（all minority + random subset of majority）。平均它们的 predictions。把这个 approach 和单个 SMOTE model 比较。测量 performance 和 runs 之间的 variance。

5. **Imbalance ratio experiment**：取一个 balanced dataset，逐步提高 imbalance ratio（50/50、70/30、90/10、95/5、99/1）。对每个 ratio，分别用 SMOTE 和不用 SMOTE 训练。画出两种 approaches 的 F1 vs imbalance ratio。SMOTE 从哪个 ratio 开始产生 meaningful difference？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Class imbalance | “One class has way more samples” | Dataset 中 classes 的 distribution 明显 skewed，导致 models 偏向 majority class |
| SMOTE | “Synthetic oversampling” | 在 existing minority samples 和它们的 k-nearest minority neighbors 之间 interpolation，创建新的 minority samples |
| Class weights | “Making errors on rare classes more expensive” | 用 class-specific weights 乘进 loss function，让 model 更重地惩罚 minority misclassification |
| Threshold tuning | “Moving the decision boundary” | 把 classification probability cutoff 从默认 0.5 改成能优化目标 metric 的 value |
| Precision-recall tradeoff | “You cannot have both” | 降低 threshold 会抓到更多 positives（higher recall），但也会 flag 更多 false positives（lower precision），反之亦然 |
| AUPRC | “Area under the PR curve” | 把 precision-recall curve 汇总成一个数；在 heavily imbalanced classes 下比 AUC-ROC 更有信息量 |
| Matthews Correlation Coefficient | “The balanced metric” | Predicted labels 和 actual labels 之间的 correlation；只有 model 在两个 classes 上都表现好时才会 high |
| Cost-sensitive learning | “Different mistakes cost different amounts” | 把 real-world misclassification costs 纳入 training objective，让 model 优化 total cost，而不是 error count |
| Random oversampling | “Duplicate the minority” | 重复 minority class samples 来平衡 class counts；简单，但有 overfit duplicated points 的风险 |

## 延伸阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813)，original SMOTE paper，仍是 imbalanced learning 中引用最多的工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907)，覆盖 sampling、cost-sensitive 和 algorithmic approaches 的 comprehensive survey
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)，包含 SMOTE variants、undersampling strategies 和 pipeline integration 的 Python library
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)，在 imbalanced problems 中何时以及为什么优先使用 PR curves 而不是 ROC curves
