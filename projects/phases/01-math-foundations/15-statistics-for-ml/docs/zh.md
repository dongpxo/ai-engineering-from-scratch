# 机器学习统计学

> 统计学告诉你模型是真的有效，还是只是运气好。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## 学习目标

- 从零计算 descriptive statistics、Pearson/Spearman correlation 和 covariance matrices
- 执行 hypothesis tests（t-test、chi-squared），并正确解释 p-values 和 confidence intervals
- 使用 bootstrap resampling，在没有分布假设的情况下为任意 metric 构建 confidence intervals
- 使用 effect size measures 区分 statistical significance 和 practical significance

## 问题

你训练了两个模型。Model A 在测试集上得分 0.87。Model B 得分 0.89。你部署了 Model B。三周后，生产指标比以前更差。发生了什么？

Model B 实际上并没有优于 Model A。0.02 的差异只是噪声。你的测试集太小，或者方差太高，或者两者都有。你把随机性包装成了改进，然后发布了它。

这种事一直发生。Kaggle leaderboard 震荡。论文无法复现。A/B tests 只基于几百个样本就宣布 winner。根因总是一样的：有人跳过了统计学。

统计学给你工具来区分信号和噪声。它告诉你差异什么时候是真的、你应该有多大信心，以及你需要多少数据才能相信结果。每个 ML pipeline、每次模型比较、每个实验都需要统计学。没有它，你只是在猜。

## 概念

### Descriptive Statistics：总结你的数据

在建模之前，你需要知道数据长什么样。Descriptive statistics 会把一个数据集压缩成几个数字，用来捕捉它的形状。

**Measures of central tendency** 回答“中间在哪里？”

```text
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Mean 是平衡点。Median 是中点。当它们分离时，说明分布是 skewed。收入分布通常 mean >> median（亿万富翁造成右偏）。训练过程中的 loss distributions 经常 mean << median（easy samples 造成左偏）。

**Measures of spread** 回答“数据分散到什么程度？”

```text
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Percentiles** 会把排序后的数据分成 100 等份。25th percentile（Q1）表示 25% 的值落在这个点以下。50th percentile 是 median。75th percentile 是 Q3。

```text
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在 ML 中，你会关心 inference latency、prediction confidence distributions 和 error distributions 的 percentiles。一个平均误差很低、但 P99 error 很糟的模型，可能在 safety-critical applications 中完全不可用。

**Sample vs population statistics。** 从样本计算 variance 时，除以 (n-1) 而不是 n。这叫 Bessel's correction。它补偿了 sample mean 不是真实 population mean 这一事实。如果分母用 n，你会系统性低估真实 variance。用 (n-1)，估计是 unbiased 的。

```text
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实践中：如果 n 很大（几千个样本），差异可以忽略。如果 n 很小（几十个样本），它就很重要。

### Correlation：变量如何共同变化

Correlation 衡量两个变量之间线性关系的强度和方向。

**Pearson correlation coefficient** 衡量线性关联：

```text
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假设关系是线性的，并且两个变量大致服从正态分布。它对 outliers 敏感。单个极端点就能把 r 从 0.1 拉到 0.9。

**Spearman rank correlation** 衡量单调关联：

```text
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**什么时候用哪个：**

```text
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金规则：** correlation does not imply causation。冰淇淋销量和溺水死亡人数相关，因为两者都会在夏天增加。模型准确率和参数数量相关，但增加参数并不会自动提升准确率（见：overfitting）。

### Covariance Matrix

两个变量之间的 covariance 衡量它们如何一起变化：

```text
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于 d 个特征，covariance matrix C 是一个 d x d 矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线元素 C[i][i] 是每个特征的 variance。

```text
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的联系。** PCA 会对 covariance matrix 做 eigendecomposition。Eigenvectors 是 principal components（最大方差方向）。Eigenvalues 告诉你每个 component 捕获了多少 variance。这正是 Lesson 10 讲过的内容，但现在你能看出为什么 covariance matrix 是该分解的正确对象：它编码了数据中所有成对线性关系。

**与 correlation 的联系。** Correlation matrix 是 standardized variables（每个变量除以自己的 standard deviation）的 covariance matrix。Correlation 会对 covariance 做归一化，让所有值落在 [-1, 1]。

### Hypothesis Testing

Hypothesis testing 是一个在不确定性下做决策的框架。你从一个 claim 开始，收集数据，然后判断数据是否与这个 claim 一致。

**设置：**

```text
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-value** 是：假设 H0 为真，看到像你观测到的数据一样极端的数据的概率。它不是 H0 为真的概率。这是统计学中最常见的误解。

```text
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence intervals** 给出某个 parameter 的 plausible values 范围：

```text
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Confidence interval 的宽度告诉你 estimate 的精度。宽 interval 表示不确定性高。窄 interval 表示估计很精确（但如果数据有 bias，它不一定准确）。

### t-test

t-test 比较 means。有几种版本。

**One-sample t-test：** population mean 是否不同于某个 hypothesized value？

```text
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test（independent）：** 两个 group means 是否不同？

```text
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test：** 当 measurements 成对出现时使用（例如同一个模型在同一批 data splits 上被评估）：

```text
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在 ML 中，paired t-test 很常见：你在同样的 10 个 cross-validation folds 上运行两个模型，然后逐 fold 比较它们的 scores。

### Chi-squared Test

Chi-squared test 检查 observed frequencies 是否匹配 expected frequencies。它适用于 categorical data。

```text
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML Models 的 A/B Testing

ML 中的 A/B testing 不等同于 web A/B testing。模型比较有一些特殊挑战：

```text
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```text
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistical Significance vs Practical Significance

一个结果可以 statistically significant，但 practically meaningless。只要数据足够多，即使微不足道的差异也会变得 statistically significant。

```text
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size** 量化差异有多大，并且与 sample size 无关：

```text
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

永远同时报告 p-value 和 effect size。p-value 告诉你差异是否真实。effect size 告诉你差异是否重要。

### Multiple Comparison Problem

当你测试许多 hypotheses 时，其中一些会因为偶然而“significant”。如果你在 alpha = 0.05 下测试 20 件事，即使没有任何真实效果，也预期会有 1 个 false positive。

```text
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni correction：** 用 tests 数量除以 alpha。

```text
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在 ML 中，当你跨多个 metrics 比较模型、测试许多 hyperparameter configurations，或在多个 datasets 上评估时，这一点很重要。

### Bootstrap Methods

Bootstrapping 通过对数据做有放回重采样，估计某个 statistic 的 sampling distribution。它不需要对底层分布做假设。

**算法：**

```text
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap confidence interval（percentile method）：**

```text
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么 bootstrap 对 ML 很重要：**

```text
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用于 model comparison 的 bootstrap：**

```text
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比 paired t-test 更鲁棒，因为它不做分布假设。

### Parametric vs Non-parametric Tests

**Parametric tests** 假设特定分布（通常是 normal）：

```text
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Non-parametric tests** 不做分布假设：

```text
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**什么时候用 non-parametric：**

```text
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**什么时候用 parametric：**

```text
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在 ML experiments 中，n 通常很小（5 或 10 个 cross-validation folds），所以 Wilcoxon signed-rank 这类 non-parametric tests 往往比 t-tests 更合适。

### Central Limit Theorem：实践含义

CLT 说的是：随着 n 增长，sample means 的分布会趋近 normal distribution，无论底层 population distribution 是什么。

```text
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**为什么这对 ML 很重要：**

```text
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不会做什么：**

```text
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 论文中的常见统计错误

1. **在训练集上测试。** 这会保证 overfitting。始终留出模型训练期间从未见过的数据。

2. **没有 confidence intervals。** 只报告单个 accuracy 数字而不报告 uncertainty，会让结果难以复现和验证。

3. **忽略 multiple comparisons。** 测试 50 个 configurations，然后不做 correction 就报告最好的那个，会抬高 false positive rates。

4. **混淆 statistical 和 practical significance。** 对 0.01% accuracy improvement 得到 p-value 0.001 并没有实际意义。

5. **在 imbalanced data 上使用 accuracy。** 如果数据集中 99% 都是 negative class，一个 trivial classifier 就能达到 99% accuracy。应使用 precision、recall、F1 或 AUC。

6. **Cherry-picking metrics。** 只报告你的模型赢的 metric。诚实评估要报告所有相关 metrics。

7. **Train/test splits 之间泄漏信息。** 在 split 前 normalize，或使用未来数据预测过去。

8. **小测试集且没有 variance estimates。** 在 100 个样本上评估并声称提升 2%，这只是噪声，不是信号。

9. **在数据不独立时假设独立。** 来自同一个病人的医学图像、同一篇文档中的多个句子。组内 observations 是相关的。

10. **P-hacking。** 不断尝试不同 tests、subsets 或 exclusion criteria，直到 p < 0.05。结果只是搜索过程的产物。

## Building It

你将实现：

1. **从零实现 descriptive statistics**（mean、median、mode、standard deviation、percentiles、IQR）
2. **Correlation functions**（Pearson 和 Spearman，以及 covariance matrix）
3. **Hypothesis tests**（one-sample t-test、two-sample t-test、chi-squared test）
4. **Bootstrap confidence intervals**（适用于任意 statistic，无需假设）
5. **A/B test simulator**（生成数据、测试、检查 Type I 和 Type II errors）
6. **Statistical vs practical significance demo**（展示大 n 如何让一切都“significant”）

全部从零实现，只使用 `math` 和 `random`。不使用 numpy，不使用 scipy。

## 关键术语

| 术语 | 定义 |
|---|---|
| Mean | 值的总和除以数量。对 outliers 敏感。 |
| Median | 排序后位于中间的值。对 outliers 鲁棒。 |
| Standard deviation | Variance 的平方根。用原始单位衡量 spread。 |
| Percentile | 低于该值的数据占给定百分比。 |
| IQR | Interquartile range。Q3 减 Q1。中间 50% 的 spread。 |
| Pearson correlation | 衡量两个变量之间的线性关联。范围 [-1, 1]。 |
| Spearman correlation | 使用 ranks 衡量单调关联。 |
| Covariance matrix | 所有 features 之间成对 covariances 组成的矩阵。 |
| Null hypothesis | 默认假设，通常是没有 effect 或没有 difference。 |
| p-value | 在 null hypothesis 为真时，观测到这么极端的数据的概率。 |
| Confidence interval | 给定 confidence level 下 parameter 的 plausible values 范围。 |
| t-test | 检验 means 是否显著不同。使用 t-distribution。 |
| Chi-squared test | 检验 observed frequencies 是否不同于 expected frequencies。 |
| Effect size | 差异大小，与 sample size 无关。Cohen's d 很常见。 |
| Bonferroni correction | 用 tests 数量除以 significance threshold，以控制 false positives。 |
| Bootstrap | 有放回重采样，用来估计 sampling distributions。 |
| Type I error | False positive。在 H0 为真时拒绝 H0。 |
| Type II error | False negative。在 H0 为假时未能拒绝 H0。 |
| Statistical power | 正确拒绝 false H0 的概率。Power = 1 减 Type II error rate。 |
| Central limit theorem | 随着 sample size 增长，sample means 会收敛到 normal distribution。 |
| Parametric test | 假设数据服从特定分布（通常是 normal）的 test。 |
| Non-parametric test | 不做分布假设。通常基于 ranks 或 signs。 |
