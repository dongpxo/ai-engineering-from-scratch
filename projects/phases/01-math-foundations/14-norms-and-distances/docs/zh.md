# 范数与距离

> 你的距离函数定义了什么叫“相似”。选错了，下游的一切都会坏掉。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 学习目标

- 从零实现 L1、L2、cosine、Mahalanobis、Jaccard 和 edit distance 函数
- 为给定 ML 任务选择合适的距离度量，并解释为什么其他选择会失败
- 将 L1 和 L2 范数与 LASSO、Ridge regularization 及其几何约束区域联系起来
- 演示同一个数据集在不同 metric 下会产生不同的 nearest neighbors

## 问题

你有两个向量。也许它们是 word embeddings。也许它们是 user profiles。也许它们是 pixel arrays。你需要知道：它们有多接近？

答案完全取决于你选择哪个距离函数。两个数据点在一个 metric 下可能是 nearest neighbors，在另一个 metric 下却相距很远。你的 KNN classifier、recommendation engine、vector database、clustering algorithm、loss function，全都依赖这个选择。选错了，模型就会优化错误的东西。

不存在通用最佳距离。L2 适合空间数据。Cosine similarity 主导 NLP。Jaccard 处理集合。Edit distance 处理字符串。Mahalanobis 会考虑相关性。Wasserstein 会搬运概率质量。每一种都编码了关于“相似”含义的不同假设。

这一课会从零构建每种主要距离函数，展示什么时候应该使用哪一种，并演示同一份数据如何因为 metric 不同而产生完全不同的 nearest neighbors。

## 概念

### Norms：衡量向量大小

Norm 衡量一个向量的“大小”。两个向量之间的每个距离函数，都可以写成它们差值的 norm：d(a, b) = ||a - b||。所以理解 norms，就是理解 distances。

### L1 Norm（Manhattan distance）

L1 norm 会把所有分量的绝对值相加。

```text
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它叫 Manhattan distance，因为它衡量的是你在城市网格上走了多远：只能沿坐标轴移动，不能走对角线。

```text
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

什么时候使用 L1：

- 高维稀疏数据（text features、one-hot encodings）
- 你希望对 outliers 更鲁棒时（单个巨大差异不会主导结果）
- Feature selection 问题（L1 regularization 会促进稀疏性）

与 L1 regularization（Lasso）的联系：在 loss function 中加入 ||w||_1，会惩罚权重绝对值之和。这会把小权重推到精确的 0，实现自动 feature selection。L1 penalty 会在 weight space 中形成菱形约束区域，而菱形的角落位于坐标轴上，也就是某些 weights 为 0 的地方。

与 loss functions 的联系：Mean Absolute Error（MAE）是 predictions 和 targets 之间 L1 distance 的平均值。它对所有错误做线性惩罚，所以相比 MSE 更能抵抗 outliers。

### L2 Norm（Euclidean distance）

L2 norm 是直线距离。它是所有分量平方和的平方根。

```text
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是几何课里学过的距离。n 维空间中的 Pythagoras。

```text
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

什么时候使用 L2：

- 低到中等维度的连续数据
- 特征尺度可比时
- 物理距离（空间数据、传感器读数）
- 像素级图像相似度

与 L2 regularization（Ridge）的联系：在 loss function 中加入 ||w||_2^2，会惩罚大权重。与 L1 不同，它不会把 weights 推到 0。它会按比例把所有 weights 缩向 0。L2 penalty 会形成圆形约束区域，所以坐标轴上没有角。权重会变小，但很少精确为 0。

与 loss functions 的联系：Mean Squared Error（MSE）是 L2 distances 的平方平均。平方会比小错误更强烈地惩罚大错误。

```text
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp Norms：通用族

L1 和 L2 是 Lp norm 的特殊情况：

```text
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同 p 值会产生不同形状的“unit balls”（距离原点为 1 的所有点的集合）：

```text
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity Norm（Chebyshev distance）

当 p 趋近 infinity 时，Lp norm 会收敛到最大绝对分量。

```text
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两个点之间的距离由它们差异最大的那个维度决定。其他维度都会被忽略。

```text
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

什么时候使用 L-infinity：

- 任意单一维度的 worst-case deviation 很重要时
- 棋盘游戏（国际象棋里的王按 L-infinity 移动：任意方向一步成本都是 1）
- 制造公差（每个维度都必须在规格范围内）

### Cosine Similarity 与 Cosine Distance

Cosine similarity 衡量两个向量之间的角度，忽略它们的 magnitude。

```text
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的范围从 -1（方向相反）到 +1（方向相同）。垂直向量的 cosine similarity 是 0。

Cosine distance 会把它转换成距离：cosine_distance = 1 - cosine_similarity。范围从 0（方向相同）到 2（方向相反）。

```text
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么 cosine 主导 NLP 和 embeddings：在文本里，文档长度不应该影响相似度。一篇关于猫的文档，即使长度是另一篇关于猫文档的两倍，也仍然应该是“相似”的。Cosine similarity 忽略 magnitude（长度），只关心方向。词分布相同但长度不同的两篇文档会指向同一方向，得到 cosine similarity 1.0。

什么时候使用 cosine similarity：

- 文本相似度（TF-IDF vectors、word embeddings、sentence embeddings）
- 任何 magnitude 是噪声、direction 是信号的领域
- 推荐系统（user preference vectors）
- Embedding search（vector databases 几乎总是使用 cosine 或 dot product）

### Dot Product Similarity vs Cosine Similarity

两个向量的 dot product 是：

```text
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity 是把 dot product 除以两个 magnitude 后的结果。当两个向量已经 unit-normalized（magnitude = 1）时，dot product 和 cosine similarity 完全相同。

```text
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

它们的差异在于：dot product 包含 magnitude 信息。magnitude 更大的向量会得到更高 dot product 分数。在某些检索系统中，这很重要，因为你希望“热门”item 排得更高。Magnitude 会成为隐式的质量或重要性信号。

```text
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：

- 当你想要纯方向相似度时，使用 cosine similarity
- 当 magnitude 携带有意义的信息时，使用 dot product
- 许多 vector databases（Pinecone、Weaviate、Qdrant）允许你在二者之间选择
- 如果 embeddings 已经 L2-normalized，选择哪个都不重要

### Mahalanobis Distance

Euclidean distance 平等对待所有维度。但如果特征相关，或具有不同尺度，L2 会给出误导性结果。

Mahalanobis distance 会考虑数据的 covariance structure。

```text
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的 covariance matrix。

直觉上：Mahalanobis distance 会先对数据做 decorrelate 和 normalize（whitening），然后在转换后的空间中计算 L2 distance。如果 S 是 identity matrix（特征不相关且方差为 1），Mahalanobis distance 就退化为 Euclidean distance。

```text
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

什么时候使用 Mahalanobis distance：

- Outlier detection（与均值 Mahalanobis distance 很大的点是 outliers）
- 特征尺度不同且存在相关性时的分类
- 你有足够数据可以可靠估计 covariance matrix 时
- 制造质量控制（multivariate process monitoring）

### Jaccard Similarity（用于集合）

Jaccard similarity 衡量两个集合的重叠。

```text
J(A, B) = |A intersect B| / |A union B|
```

它的范围从 0（没有重叠）到 1（集合相同）。Jaccard distance = 1 - Jaccard similarity。

```text
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

什么时候使用 Jaccard：

- 比较 tags、categories 或 features 的集合
- 基于词是否出现的 document similarity（不是词频）
- Near-duplicate detection（Jaccard 的 MinHash 近似）
- 比较 binary feature vectors（presence/absence data）
- 评估 segmentation models（Intersection over Union = Jaccard）

### Edit Distance（Levenshtein Distance）

Edit distance 计算把一个字符串转换成另一个字符串所需的最少单字符操作数。操作包括：insert、delete 或 substitute。

```text
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

它用 dynamic programming 计算。填一个矩阵，其中 entry (i, j) 是字符串 A 前 i 个字符与字符串 B 前 j 个字符之间的 edit distance。

```text
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

什么时候使用 edit distance：

- 拼写检查和纠错
- DNA sequence alignment（带 weighted operations）
- Fuzzy string matching
- 脏文本数据去重

### KL Divergence（不是距离，但经常被当成距离用）

KL divergence 衡量一个概率分布与另一个概率分布有多不同。Lesson 09 已经讲过它，但它属于这次讨论，因为人们经常把它当作“距离”使用，尽管它不是。

```text
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL divergence 不对称。

```text
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足 distance metric 的基本要求。它也不满足 triangle inequality。它是 divergence，不是 distance。

Forward KL（D_KL(P || Q)）是 “mean-seeking”：Q 会尝试覆盖 P 的所有 modes。  
Reverse KL（D_KL(Q || P)）是 “mode-seeking”：Q 会聚焦在 P 的单个 mode 上。

你会在这些地方看到 KL divergence：

- VAEs（ELBO 中的 KL 项把 latent distribution 推向 prior）
- Knowledge distillation（student 尝试匹配 teacher 的 distribution）
- RLHF（KL penalty 让 fine-tuned model 靠近 base model）
- Policy gradient methods（约束 policy updates）

### Wasserstein Distance（Earth Mover's Distance）

Wasserstein distance 衡量把一个概率分布转换成另一个概率分布所需的最小“工作量”。可以这样想：如果一个分布是一堆土，另一个分布是一个坑，你需要把多少土搬多远？

```text
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于 1D distributions，它会简化成两个 cumulative distribution functions 绝对差的积分：

```text
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 重要：

- 它是真正的 metric（对称，满足 triangle inequality）
- 即使两个分布不重叠，它也能提供 gradients（KL divergence 会变成 infinity）
- 这个性质让它成为 Wasserstein GANs（WGANs）的核心，解决了原始 GAN 的训练不稳定问题

```text
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

什么时候使用 Wasserstein：

- GAN training（WGAN、WGAN-GP）
- 比较可能不重叠的 distributions
- Optimal transport problems
- 图像检索（比较 color histograms）

### 为什么不同任务需要不同距离

| Task | Best distance | Why |
|------|---------------|-----|
| Text similarity | Cosine | Magnitude 是噪声，direction 是含义 |
| Image pixel comparison | L2 | 空间关系重要，features 尺度可比 |
| Sparse high-dim features | L1 | 鲁棒，不会放大罕见的大差异 |
| Set overlap (tags, categories) | Jaccard | 数据天然是 set-valued，不是 vectorial |
| String matching | Edit distance | 操作符合人类编辑直觉 |
| Outlier detection | Mahalanobis | 考虑 feature correlations 和 scales |
| Comparing distributions | KL divergence | 衡量用 Q 而不是 P 损失了多少信息 |
| GAN training | Wasserstein | 即使 distributions 不重叠也提供 gradients |
| Embeddings (vector DB) | Cosine or dot product | Embeddings 通常把意义编码在 direction 中 |
| Recommendation | Dot product | Magnitude 可以编码 popularity 或 confidence |
| DNA sequences | Weighted edit distance | 不同 nucleotide pair 的 substitution costs 不同 |
| Manufacturing QC | L-infinity | 任意维度的 worst-case deviation 都重要 |

### 与 Loss Functions 的联系

Loss functions 就是应用到 predictions 和 targets 上的 distance functions。

```text
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与 Regularization 的联系

Regularization 会在 loss function 中加入权重的 norm penalty。

```text
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 会产生 sparsity 而 L2 不会：想象 2D weight space 中的约束区域。L1 是菱形，L2 是圆。Loss function 的 contours（椭圆）最可能在菱形角落接触约束区域，而那里某个 weight 为 0。它们接触圆时是在平滑点上，两个 weights 通常都非零。

### Nearest Neighbor Search

每个 distance function 都隐含一个 nearest neighbor search 问题：给定一个 query point，在数据集中找到最近的点。

在一个有 n 个点、d 个维度的数据集中，exact nearest neighbor search 每次 query 是 O(n * d)。对于大数据集，这太慢了。

Approximate Nearest Neighbor（ANN）算法用少量准确率换取巨大速度提升：

```text
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代 vector databases 中的主流算法。它构建一个多层 graph，每个 node 连接到它的 approximate nearest neighbors。搜索从顶层开始（稀疏、长跳跃），逐层下降到底层（密集、短跳跃）。

```figure
norm-unit-balls
```

## Build It

### 第 1 步：所有 norm 和 distance functions

完整实现见 `code/distances.py`。每个函数都只用基础 Python math 从零构建。

### 第 2 步：同一份数据，不同距离，不同 neighbors

`distances.py` 中的 demo 会创建一个数据集，选择一个 query point，然后展示 nearest neighbor 如何随 distance metric 改变。在 L1 下“最近”的点，在 L2 或 cosine 下可能并不是最近。

### 第 3 步：Embedding similarity search

代码包含一个 mock embedding similarity search，会使用 cosine similarity 与 L2 distance 找到与 query 最相似的“documents”，展示 rankings 可能不同。

## Use It

最常见的实践用途：在 vector database 中寻找相似 items。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索 vector database 时，底层发生的就是这些。Embedding model 把文本映射成向量。Vector database 计算 query vector 与每个存储向量之间的 cosine similarity（或 dot product），并使用 ANN algorithms 避免逐个检查全部向量。

## 练习

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity distances。验证对任意一对点，总有 L-inf <= L2 <= L1。证明为什么这个顺序必然成立。

2. 创建两个向量，让 cosine similarity 很高（> 0.9），但 L2 distance 很大（> 10）。从几何上解释发生了什么。然后创建两个向量，让 cosine similarity 很低（< 0.3），但 L2 distance 很小（< 0.5）。

3. 实现一个函数，输入 dataset 和 query point，分别返回 L1、L2、cosine 和 Mahalanobis distance 下的 nearest neighbor。找一个数据集，让四者对哪个点最近产生分歧。

4. 使用 CDF 方法，手算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein distance。然后计算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的距离。哪个更大？为什么？

5. 实现 MinHash 来近似 Jaccard similarity。生成 100 个随机集合，计算所有 pair 的精确 Jaccard，并使用 50、100、200 个 hash functions 的 MinHash 近似进行比较。绘制 approximation error。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Norm | “向量大小” | 一个把向量映射到非负 scalar 的函数，满足 triangle inequality、absolute homogeneity，并且只有 zero vector 的 norm 为 0 |
| L1 norm | “Manhattan distance” | 分量绝对值之和。在优化中产生 sparsity。对 outliers 鲁棒 |
| L2 norm | “Euclidean distance” | 分量平方和的平方根。Euclidean space 中的直线距离 |
| Lp norm | “Generalized norm” | 绝对分量 p 次方和的 p 次根。L1 和 L2 是特殊情况 |
| L-infinity norm | “Max norm” 或 “Chebyshev distance” | 最大绝对分量值。是 p 趋近 infinity 时 Lp 的极限 |
| Cosine similarity | “向量之间的角度” | 按两个 magnitude 归一化后的 dot product。范围从 -1 到 +1。忽略向量长度 |
| Cosine distance | “1 minus cosine similarity” | 把 cosine similarity 转成 distance。范围从 0 到 2 |
| Dot product | “未归一化的 cosine” | 分量乘积之和。等于 cosine similarity 乘以两个 magnitude |
| Mahalanobis distance | “考虑相关性的距离” | 在使用数据 covariance matrix 进行 whitened（decorrelated and normalized）的空间中的 L2 distance |
| Jaccard similarity | “集合重叠” | 交集大小除以并集大小。用于 sets，不是 vectors |
| Edit distance | “Levenshtein distance” | 把一个字符串转换成另一个字符串所需的最少 insertions、deletions 和 substitutions |
| KL divergence | “分布之间的距离” | 不是真正的 distance（不对称）。衡量用 Q 编码 P 会多花多少 bits |
| Wasserstein distance | “Earth mover's distance” | 把质量从一个分布搬到另一个分布所需的最小工作量。是真正的 metric |
| Approximate nearest neighbor | “ANN search” | 比 exact search 快得多地寻找近似最近点的算法（HNSW、LSH、IVF） |
| HNSW | “vector DB algorithm” | Hierarchical Navigable Small World graph。用于快速 approximate nearest neighbor search 的多层 graph |
| L1 regularization | “Lasso” | 把 weights 的 L1 norm 加到 loss 中。把 weights 推到 0（sparsity） |
| L2 regularization | “Ridge” 或 “weight decay” | 把 weights 的 squared L2 norm 加到 loss 中。把 weights 缩向 0，但不产生 sparsity |
| Elastic Net | “L1 + L2” | 结合 L1 和 L2 regularization。比单独使用任意一种更适合处理相关 feature groups |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的 billion-scale ANN search 库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将 Earth Mover's distance 引入 GANs 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基础 ANN 算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，cosine similarity 在 embeddings 中成为默认选择的关键背景
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中 distance metrics 和 neighbor algorithms 的实践指南
