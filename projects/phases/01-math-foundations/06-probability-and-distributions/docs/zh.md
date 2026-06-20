# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## 学习目标

- 从零实现 Bernoulli、categorical、Poisson、uniform 和 normal 分布的 PMF 与 PDF
- 计算期望、方差，并使用中心极限定理解释为什么 Gaussian 如此常见
- 使用数值稳定技巧（减去最大 logit）构建 softmax 和 log-softmax 函数
- 从 logits 计算交叉熵损失，并把它连接到负对数似然

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型要从 50,000 个候选词中选择下一个词。一个扩散模型通过从学习到的分布中采样来生成图像。这些全都是概率在发挥作用。

模型做出的每个预测都是一个概率分布。每个损失函数都在衡量预测分布与真实分布之间的距离。每个训练步骤都在调整参数，让一个分布看起来更像另一个分布。没有概率，你就无法读懂机器学习论文，无法调试模型，也无法理解为什么训练损失会变成 NaN。

## 概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的一个子集。概率会把事件映射到 0 到 1 之间的数字。

```text
抛硬币：
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

掷一次骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了整个概率论：

1. 对任意事件 A，P(A) >= 0
2. P(S) = 1，也就是总会发生某个结果
3. 当 A 和 B 不可能同时发生时，P(A or B) = P(A) + P(B)

其他所有内容，包括 Bayes 定理、期望和分布，都来自这三条规则。

### 条件概率与独立性

P(A|B) 表示在 B 已经发生的条件下，A 发生的概率。

```text
P(A|B) = P(A and B) / P(B)

例子：一副扑克牌
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

如果知道一个事件是否发生，并不会给另一个事件提供任何信息，那么这两个事件就是独立的：

```text
独立：       P(A|B) = P(A)
等价于：     P(A and B) = P(A) * P(B)
```

抛硬币之间是独立的。不放回地抽牌不是独立的。

### 概率质量函数 vs 概率密度函数

离散随机变量有概率质量函数（PMF）。每个结果都有一个可以直接读出的具体概率。

```text
PMF: P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数（PDF）。单个点上的密度不是概率。概率来自对某个区间上的密度进行积分。

```text
PDF: f(x)

P(a <= X <= b) = 从 a 到 b 对 f(x) 积分

f(x) 可以大于 1（它是密度，不是概率）
从 -inf 到 +inf 对 f(x) dx 积分 = 1
```

这个区别在机器学习中很重要。分类输出是 PMF（离散选择）。VAE 的潜在空间使用 PDF（连续变量）。

### 常见分布

**Bernoulli：** 一次试验，两个结果。用于建模二分类。

```text
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical：** 一次试验，k 个结果。用于建模多分类（softmax 输出）。

```text
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform：** 所有结果等可能。用于随机初始化。

```text
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal（Gaussian）：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```text
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson：** 固定区间内罕见事件的计数。用于建模事件发生率。

```text
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望与方差

期望是结果的加权平均。

```text
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差衡量结果围绕均值的分散程度。

```text
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在机器学习中，期望会以损失函数的形式出现，也就是数据分布上的平均损失。方差告诉你模型有多稳定。梯度的高方差意味着训练很嘈杂。

### 联合分布与边缘分布

联合分布 P(X, Y) 会一起描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 雨伞）：

| | Y=0（无伞） | Y=1（有伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（下雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布会把另一个变量求和消掉：

```text
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表中的行总和与列总和就是边缘概率。

### 为什么正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均值）会收敛到正态分布，无论原始分布是什么。

```text
掷 1 个骰子：  均匀分布（平的）
2 个骰子的平均值：  三角形（中间更高）
30 个骰子的平均值：接近完美的钟形曲线

这对任意起始分布都成立。
```

这就是为什么：

- 测量误差近似正态分布（由许多小的独立来源组成）
- 神经网络的权重初始化会使用正态分布
- SGD 中的梯度噪声近似正态分布（许多样本梯度的和）
- 在给定均值和方差的情况下，正态分布是最大熵分布

### 对数概率

原始概率会造成数值问题。把很多很小的概率连乘，很快就会下溢成零。

```text
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0（约 30 项后下溢）
```

对数概率可以解决这个问题。乘法会变成加法。

```text
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> 有限数字（不会下溢）
```

规则：

- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0，因为 0 < P <= 1
- 越负表示越不可能
- 交叉熵损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 会把它们转换成有效的概率分布。

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

性质：
  - 所有输出都在 (0, 1) 内
  - 所有输出之和为 1
  - 保留输入的相对顺序
  - exp() 会放大 logits 之间的差异
```

softmax 技巧：在取指数之前减去最大 logit，防止溢出。

```text
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

结果相同，没有溢出。
```

Log-softmax 会把 softmax 和 log 结合起来，以获得数值稳定性。PyTorch 的交叉熵损失内部就是这样做的。

### 采样

采样就是从一个分布中抽取随机值。在机器学习中：

- Dropout 会随机采样哪些神经元被置零
- 数据增强会采样随机变换
- 语言模型会从预测分布中采样下一个 token
- 扩散模型会采样噪声，并逐步去噪

从任意分布中采样需要使用逆变换采样、拒绝采样或重参数化技巧等方法。重参数化技巧用于 VAE。

```figure
gaussian-pdf
```

## Build It

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含全部可视化的完整实现位于 `code/probability.py`。

## Use It

使用 NumPy 和 SciPy，上面所有内容都可以写成一行：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零实现过这些内容。现在你知道这些库调用到底在做什么。

## 练习

1. 为指数分布实现逆变换采样。通过采样 10,000 个值，并把直方图与真实 PDF 对比来验证。

2. 为两个加权骰子构建联合分布表。计算边缘分布，并检查这两个骰子是否独立。

3. 一个 5 类分类器输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别是索引 3。计算交叉熵损失，然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 写一个函数，输入一组对数概率，返回最可能的序列、总对数概率，以及等价的原始概率。用一个包含 50 个词的句子来测试，其中每个词的概率都是 0.01。

## 关键术语

| 术语 | 人们常说 | 它真正的含义 |
|------|----------|--------------|
| 样本空间 | “所有可能性” | 实验中每个可能结果组成的集合 S |
| PMF | “概率函数” | 给出每个离散结果精确概率的函数，所有概率之和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。对某个区间积分才能得到概率 |
| 条件概率 | “给定某事后的概率” | P(A\|B) = P(A and B) / P(B)。Bayesian 思维和 Bayes 定理的基础 |
| 独立性 | “它们互不影响” | P(A and B) = P(A) * P(B)。知道一个事件不会告诉你另一个事件的信息 |
| 期望 | “平均值” | 所有结果按概率加权后的和。损失函数就是一种期望 |
| 方差 | “有多分散” | 相对于均值的平方偏差的期望。高方差 = 嘈杂、不稳定的估计 |
| 正态分布 | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于中心极限定理，它无处不在 |
| 中心极限定理 | “平均值会变成正态” | 许多独立样本的均值会收敛到正态分布，无论来源分布是什么 |
| 联合分布 | “两个变量放在一起” | P(X, Y) 描述 X 和 Y 每一种结果组合的概率 |
| 边缘分布 | “把另一个变量求和消掉” | P(X) = sum_y P(X, Y)。从联合分布中恢复一个变量的分布 |
| 对数概率 | “概率的 log” | log P(x)。把乘法变成加法，防止长序列中的数值下溢 |
| Softmax | “把分数变成概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。把实数 logits 映射成有效概率分布 |
| 交叉熵 | “损失函数” | -sum(p_true * log(p_predicted))。衡量两个分布有多不同。越低越好 |
| Logits | “模型原始输出” | softmax 之前的未归一化分数。名称来自 logistic 函数 |
| 采样 | “抽随机值” | 按照概率分布生成值。模型就是这样生成输出的 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 用可视化方式证明为什么平均值会变成正态
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 覆盖本课及更多内容的简明概率参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性很重要，以及如何实现它
