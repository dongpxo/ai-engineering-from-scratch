# 线性系统

> 求解 Ax = b 是数学中最古老、同时仍在驱动神经网络的问题。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors & Matrices), 03 (Matrix Transformations)
**Time:** ~120 minutes

## 学习目标

- 使用带 partial pivoting 的 Gaussian elimination 和 back substitution 求解 Ax = b
- 用 LU、QR 和 Cholesky decompositions 分解矩阵，并解释每种方法适合什么场景
- 推导 least squares 的 normal equations，并把它们连接到 linear regression 和 ridge regression
- 使用 condition number 诊断 ill-conditioned systems，并应用 regularization 来稳定求解

## 问题

每次训练 linear regression，你都在求解一个 linear system。每次计算 least-squares fit，你都在求解一个 linear system。每次 neural network layer 计算 `y = Wx + b`，它都在计算 linear system 的一侧。当你添加 regularization，你就在修改这个 system。当你使用 Gaussian processes，你会 factor 一个矩阵。当你为了 Mahalanobis distance 反转 covariance matrix，你也在求解 linear system。

方程 Ax = b 到处都在出现。A 是已知 coefficients 的 matrix。b 是已知 outputs 的 vector。x 是你想找到的 unknowns vector。在 linear regression 中，A 是 data matrix，b 是 target vector，x 是 weight vector。整个模型可以归结为：找到 x，让 Ax 尽可能接近 b。

本课会从零构建求解这个方程的所有主要方法。你会理解为什么有些方法快、有些方法稳定，为什么有些方法只适用于 square systems，而其他方法可以处理 overdetermined systems，以及为什么 matrix 的 condition number 决定了你的答案是否有意义。

## 概念

### Ax = b 的几何含义

一个 linear equations system 有几何解释。每个 equation 定义一个 hyperplane。Solution 是所有 hyperplanes 相交的 point（或 point set）。

```text
2x + y = 5          Two lines in 2D.
x - y  = 1          They intersect at x=2, y=1.
```

```mermaid
graph LR
    A["2x + y = 5"] --- S["Solution: (2, 1)"]
    B["x - y = 1"] --- S
```

可能发生三种情况：

```mermaid
graph TD
    subgraph "One Solution"
        A1["Lines intersect at a single point"]
    end
    subgraph "No Solution"
        A2["Lines are parallel — no intersection"]
    end
    subgraph "Infinite Solutions"
        A3["Lines are identical — every point is a solution"]
    end
```

在 matrix form 中，"one solution" 意味着 A is invertible。"No solution" 意味着 system is inconsistent。"Infinite solutions" 意味着 A has a null space。大多数 ML 问题属于 "no exact solution" 类别，因为 equations（data points）比 unknowns（parameters）更多。这就是 least squares 出场的地方。

### Column picture vs row picture

有两种方式理解 Ax = b。

**Row picture。** A 的每一行定义一个 equation。每个 equation 是一个 hyperplane。Solution 是它们全部相交的位置。

**Column picture。** A 的每一列是一个 vector。问题变成：A 的 columns 的什么 linear combination 会产生 b？

```text
A = | 2  1 |    b = | 5 |
    | 1 -1 |        | 1 |

Row picture: solve 2x + y = 5 and x - y = 1 simultaneously.

Column picture: find x1, x2 such that:
  x1 * [2, 1] + x2 * [1, -1] = [5, 1]
  2 * [2, 1] + 1 * [1, -1] = [4+1, 2-1] = [5, 1]   check.
```

Column picture 更基础。如果 b 位于 A 的 column space 中，system 就有 solution。如果 b 不在其中，你要在 column space 中找到离 b 最近的 point。这个 closest point 就是 least-squares solution。

### Gaussian elimination

Gaussian elimination 会把 Ax = b 转换成 upper triangular system Ux = c，然后用 back substitution 求解。这是最直接的方法。

Algorithm：

```text
1. For each column k (the pivot column):
   a. Find the largest entry in column k at or below row k (partial pivoting).
   b. Swap that row with row k.
   c. For each row i below k:
      - Compute multiplier m = A[i][k] / A[k][k]
      - Subtract m times row k from row i.
2. Back substitute: solve from the last equation upward.
```

Example：

```text
Original:
| 2  1  1 | 8 |       R2 = R2 - (2)R1     | 2  1   1 |  8 |
| 4  3  3 |20 |  -->  R3 = R3 - (1)R1 --> | 0  1   1 |  4 |
| 2  3  1 |12 |                            | 0  2   0 |  4 |

                       R3 = R3 - (2)R2     | 2  1   1 |  8 |
                                       --> | 0  1   1 |  4 |
                                           | 0  0  -2 | -4 |

Back substitute:
  -2 * x3 = -4    -->  x3 = 2
  x2 + 2  = 4     -->  x2 = 2
  2*x1 + 2 + 2 = 8 --> x1 = 2
```

Gaussian elimination 的代价是 O(n^3) operations。对一个 1000x1000 system 来说，大约是十亿次 floating-point operations。它很快，但如果你需要用同一个 A 求解多个 systems，还有更好的办法。

### Partial pivoting：为什么重要

没有 pivoting，Gaussian elimination 可能失败，或者产生垃圾结果。如果 pivot element 是 0，你会除以 0。如果它很小，你会放大 rounding errors。

```text
Bad pivot:                       With partial pivoting:
| 0.001  1 | 1.001 |            Swap rows first:
| 1      1 | 2     |            | 1      1 | 2     |
                                 | 0.001  1 | 1.001 |
m = 1/0.001 = 1000              m = 0.001/1 = 0.001
R2 = R2 - 1000*R1               R2 = R2 - 0.001*R1
| 0.001  1     | 1.001   |      | 1      1     | 2     |
| 0     -999   | -999.0  |      | 0      0.999 | 0.999 |

x2 = 1.000 (correct)            x2 = 1.000 (correct)
x1 = (1.001 - 1)/0.001          x1 = (2 - 1)/1 = 1.000 (correct)
   = 0.001/0.001 = 1.000        Stable because the multiplier is small.
```

在 limited precision 的 floating-point arithmetic 中，未 pivot 的版本可能丢失 significant digits。Partial pivoting 总是选择当前可用的最大 pivot，以最小化 error amplification。

### LU decomposition

LU decomposition 会把 A factor 成 lower triangular matrix L 和 upper triangular matrix U：A = LU。L matrix 存储 Gaussian elimination 中的 multipliers。U matrix 是 elimination 的结果。

```text
A = L @ U

| 2  1  1 |   | 1  0  0 |   | 2  1   1 |
| 4  3  3 | = | 2  1  0 | @ | 0  1   1 |
| 2  3  1 |   | 1  2  1 |   | 0  0  -2 |
```

为什么 factor，而不是直接 eliminate？因为一旦有了 L 和 U，对任何新的 b 求解 Ax = b 都只需要 O(n^2)：

```text
Ax = b
LUx = b
Let y = Ux:
  Ly = b    (forward substitution, O(n^2))
  Ux = y    (back substitution, O(n^2))
```

O(n^3) 的成本只在 factorization 时支付一次。之后每次 solve 都是 O(n^2)。如果你需要用同一个 A 和不同的 b vectors 求解 1000 个 systems，LU 会在总工作量上节省约 1000/3 倍。

带 partial pivoting 时，你得到的是 PA = LU，其中 P 是记录 row swaps 的 permutation matrix。

### QR decomposition

QR decomposition 会把 A factor 成 orthogonal matrix Q 和 upper triangular matrix R：A = QR。

Orthogonal matrix 满足 Q^T Q = I。它的 columns 是 orthonormal vectors。乘以 Q 会保持 lengths 和 angles 不变。

```text
A = Q @ R

Q has orthonormal columns: Q^T Q = I
R is upper triangular

To solve Ax = b:
  QRx = b
  Rx = Q^T b    (just multiply by Q^T, no inversion needed)
  Back substitute to get x.
```

QR 在求解 least-squares problems 时比 LU numerically more stable。Gram-Schmidt process 会逐列构建 Q：

```text
Given columns a1, a2, ... of A:

q1 = a1 / ||a1||

q2 = a2 - (a2 . q1) * q1        (subtract projection onto q1)
q2 = q2 / ||q2||                (normalize)

q3 = a3 - (a3 . q1) * q1 - (a3 . q2) * q2
q3 = q3 / ||q3||

R[i][j] = qi . aj    for i <= j
```

每一步都会移除沿着所有 previous q vectors 的 component，只留下新的 orthogonal direction。

### Cholesky decomposition

当 A 是 symmetric（A = A^T）并且 positive definite（所有 eigenvalues 为正）时，你可以把它 factor 成 A = L L^T，其中 L 是 lower triangular。这就是 Cholesky decomposition。

```text
A = L @ L^T

| 4  2 |   | 2  0 |   | 2  1 |
| 2  5 | = | 1  2 | @ | 0  2 |

L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]    for i > j
```

Cholesky 比 LU 快约两倍，并且只需要一半 storage。它只适用于 symmetric positive definite matrices，但这些矩阵经常出现：

- Covariance matrices 是 symmetric positive semi-definite（加上 regularization 后是 positive definite）。
- Gaussian processes 中的 kernel matrix 是 symmetric positive definite。
- Convex function 在 minimum 处的 Hessian 是 symmetric positive definite。
- A^T A 永远是 symmetric positive semi-definite。

在 Gaussian processes 中，你用 Cholesky factor kernel matrix K，然后求解 K alpha = y 来得到 predictive mean。Cholesky factor 也会给出 marginal likelihood 所需的 log-determinant：log det(K) = 2 * sum(log(diag(L)))。

### Least squares：当 Ax = b 没有 exact solution

如果 A 是 m x n 且 m > n（equations 比 unknowns 多），system 就是 overdetermined。它没有 exact solution。你转而最小化 squared error：

```text
minimize ||Ax - b||^2

This is the sum of squared residuals:
  sum((A[i,:] @ x - b[i])^2 for i in range(m))
```

Minimizer 满足 normal equations：

```text
A^T A x = A^T b
```

推导：展开 ||Ax - b||^2 = (Ax - b)^T (Ax - b) = x^T A^T A x - 2 x^T A^T b + b^T b。对 x 求 gradient，并设为 0：2 A^T A x - 2 A^T b = 0。

```text
Original system (overdetermined, 4 equations, 2 unknowns):
| 1  1 |         | 3 |
| 1  2 | x     = | 5 |       No exact x satisfies all 4 equations.
| 1  3 |         | 6 |
| 1  4 |         | 8 |

Normal equations:
A^T A = | 4  10 |    A^T b = | 22 |
        | 10 30 |            | 63 |

Solve: x = [1.5, 1.7]

This is linear regression. x[0] is the intercept, x[1] is the slope.
```

### Normal equations = linear regression

这个连接是精确的。在 linear regression 中，data matrix X 每一行对应一个 sample，每一列对应一个 feature。Target vector y 每个 entry 对应一个 sample。Weight vector w 满足：

```text
X^T X w = X^T y
w = (X^T X)^(-1) X^T y
```

这是 linear regression 的 closed-form solution。每次调用 `sklearn.linear_model.LinearRegression.fit()` 都会计算这个式子（或通过 QR / SVD 计算等价形式）。

给 matrix 加上 regularization term lambda * I，就得到 ridge regression：

```text
(X^T X + lambda * I) w = X^T y
w = (X^T X + lambda * I)^(-1) X^T y
```

Regularization 会让 matrix 更 well-conditioned（更容易准确 invert），并通过把 weights shrink toward zero 来防止 overfitting。当 lambda > 0 时，matrix X^T X + lambda * I 总是 symmetric positive definite，所以你可以用 Cholesky 来求解。

### Pseudoinverse（Moore-Penrose）

Pseudoinverse A+ 把 matrix inversion 推广到 non-square 和 singular matrices。对任意 matrix A：

```text
x = A+ b

where A+ = V Sigma+ U^T    (computed via SVD)
```

Sigma+ 的构造方式是：对每个 nonzero singular value 取 reciprocal，然后转置结果。如果 A = U Sigma V^T，那么 A+ = V Sigma+ U^T。

```text
A = U Sigma V^T        (SVD)

Sigma = | 5  0 |       Sigma+ = | 1/5  0  0 |
        | 0  2 |                | 0  1/2  0 |
        | 0  0 |

A+ = V Sigma+ U^T
```

Pseudoinverse 会给出 minimum-norm least-squares solution。如果 system 有：
- One solution：A+ b 给出它。
- No solution：A+ b 给出 least-squares solution。
- Infinite solutions：A+ b 给出 ||x|| 最小的那个。

NumPy 的 `np.linalg.lstsq` 和 `np.linalg.pinv` 内部都使用 SVD。

### Condition number

Condition number 衡量 solution 对 input 中微小变化有多敏感。对 matrix A，condition number 是：

```text
kappa(A) = ||A|| * ||A^(-1)|| = sigma_max / sigma_min
```

其中 sigma_max 和 sigma_min 是最大和最小 singular values。

```text
Well-conditioned (kappa ~ 1):        Ill-conditioned (kappa ~ 10^15):
Small change in b -->                Small change in b -->
small change in x                    huge change in x

| 2  0 |   kappa = 2/1 = 2          | 1   1          |   kappa ~ 10^15
| 0  1 |   safe to solve            | 1   1+10^(-15) |   solution is garbage
```

经验规则：
- kappa < 100：safe，solution accurate。
- kappa ~ 10^k：你的 floating-point arithmetic 会损失大约 k 位 precision。
- kappa ~ 10^16（对 float64）：solution meaningless。Matrix effectively singular。

在 ML 中，features nearly collinear 时会出现 ill-conditioning。Regularization（添加 lambda * I）会把 condition number 从 sigma_max / sigma_min 改善为 (sigma_max + lambda) / (sigma_min + lambda)。

### Iterative methods：conjugate gradient

对非常大的 sparse systems（数百万 unknowns），LU 或 Cholesky 这样的 direct methods 太昂贵。Iterative methods 会通过多次迭代改善 guess 来近似 solution。

Conjugate gradient（CG）在 A 是 symmetric positive definite 时求解 Ax = b。它在 exact arithmetic 中最多 n 次 iterations 就能找到 exact solution，但如果 A 的 eigenvalues clustered，通常会快得多地收敛。

```text
Algorithm sketch:
  x0 = initial guess (often zero)
  r0 = b - A x0           (residual)
  p0 = r0                 (search direction)

  For k = 0, 1, 2, ...:
    alpha = (rk . rk) / (pk . A pk)
    x_{k+1} = xk + alpha * pk
    r_{k+1} = rk - alpha * A pk
    beta = (r_{k+1} . r_{k+1}) / (rk . rk)
    p_{k+1} = r_{k+1} + beta * pk
    if ||r_{k+1}|| < tolerance: stop
```

CG 用于：
- Large-scale optimization（Newton-CG method）
- 求解 PDE discretizations
- Kernel methods，其中 kernel matrix 太大，无法 factor
- 为其他 iterative solvers 做 preconditioning

Convergence rate 取决于 condition number。Conditioned 更好的 systems 收敛更快，这也是 regularization 有帮助的另一个原因。

### 完整图景：什么时候用哪种方法

| Method | Requirements | Cost | Use case |
|--------|-------------|------|----------|
| Gaussian elimination | Square, nonsingular A | O(n^3) | 一次性求解 square system |
| LU decomposition | Square, nonsingular A | O(n^3) factor + O(n^2) solve | 用同一个 A 做多次 solve |
| QR decomposition | Any A (m >= n) | O(mn^2) | Least squares，numerically stable |
| Cholesky | Symmetric positive definite A | O(n^3/3) | Covariance matrices、Gaussian processes、ridge regression |
| Normal equations | Overdetermined (m > n) | O(mn^2 + n^3) | Linear regression（small n） |
| SVD / pseudoinverse | Any A | O(mn^2) | Rank-deficient systems、minimum-norm solutions |
| Conjugate gradient | Symmetric positive definite, sparse A | O(n * k * nnz) | Large sparse systems，k = iterations |

### 与 ML 的联系

本课中的每种方法都会出现在 production ML 中：

**Linear regression。** Closed-form solution 求解 normal equations X^T X w = X^T y。它可以通过 Cholesky（如果 n 很小）、QR（如果 numerical stability 很重要）或 SVD（如果 matrix 可能 rank-deficient）完成。

**Ridge regression。** 向 X^T X 添加 lambda * I。Regularized system (X^T X + lambda * I) w = X^T y 总是可以通过 Cholesky 求解，因为当 lambda > 0 时，X^T X + lambda * I 是 symmetric positive definite。

**Gaussian processes。** Predictive mean 需要求解 K alpha = y，其中 K 是 kernel matrix。K 的 Cholesky factorization 是标准方法。Log marginal likelihood 使用 log det(K) = 2 sum(log(diag(L)))。

**Neural network initialization。** Orthogonal initialization 使用 QR decomposition 创建 columns orthonormal 的 weight matrices。这能防止 deep networks 中的 signal collapse。

**Preconditioning。** Large-scale optimizers 使用 incomplete Cholesky 或 incomplete LU 作为 conjugate gradient solvers 的 preconditioners。

**Feature engineering。** X^T X 的 condition number 会告诉你 features 是否 collinear。如果 kappa 很大，就 drop features 或 add regularization。

```figure
linear-system-conditioning
```

## Build It

### Step 1：Gaussian elimination with partial pivoting

```python
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x
```

### Step 2：LU decomposition

```python
def lu_decompose(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U

def lu_solve(P, L, U, b):
    n = len(b)
    Pb = P @ b.astype(float)

    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x
```

### Step 3：Cholesky decomposition

```python
def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L
```

### Step 4：Least squares via normal equations

```python
def least_squares_normal(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return gaussian_elimination(AtA, Atb)

def ridge_regression(A, b, lam):
    n = A.shape[1]
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b
    L = cholesky(AtA)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (Atb[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L.T[i, i]
    return x
```

### Step 5：Condition number

```python
def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]
```

## Use It

把这些 pieces 组合起来，在真实数据上做 linear regression 和 ridge regression：

```python
np.random.seed(42)
X_raw = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X_raw @ w_true + np.random.randn(100) * 0.1

X = np.column_stack([np.ones(100), X_raw])

w_ols = least_squares_normal(X, y)
print(f"OLS weights (ours):    {w_ols}")

w_np = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS weights (numpy):   {w_np}")
print(f"Max difference: {np.max(np.abs(w_ols - w_np)):.2e}")

w_ridge = ridge_regression(X, y, lam=1.0)
print(f"Ridge weights (ours):  {w_ridge}")

from sklearn.linear_model import Ridge
ridge_sk = Ridge(alpha=1.0, fit_intercept=False)
ridge_sk.fit(X, y)
print(f"Ridge weights (sklearn): {ridge_sk.coef_}")
```

## Ship It

本课产出：
- `code/linear_systems.py`，包含 Gaussian elimination、LU decomposition、Cholesky decomposition、least squares 和 ridge regression 的 from-scratch implementations
- 一个 working demonstration，展示 normal equations 和 sklearn 的 LinearRegression 会产生相同 weights

## 练习

1. 使用你的 Gaussian elimination、你的 LU solver，以及 `np.linalg.solve` 求解 system `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]`。验证三者在 floating-point tolerance 内给出相同答案。

2. 生成一个 50x5 random matrix X 和 target y = X @ w_true + noise。分别用 normal equations、QR（通过 `np.linalg.qr`）、SVD（通过 `np.linalg.svd`）和 `np.linalg.lstsq` 求解 w。比较四个 solutions。测量 X^T X 的 condition number，并解释它如何影响你信任哪种方法。

3. 通过让两列几乎相同来创建 nearly singular matrix（例如 column 2 = column 1 + 1e-10 * noise）。计算它的 condition number。分别在有 regularization（add 0.01 * I）和无 regularization 时求解 Ax = b。比较 solutions 和 residuals。解释为什么 regularization 有帮助。

4. 为一个 100x100 random symmetric positive definite matrix 实现 conjugate gradient algorithm。统计它需要多少 iterations 才能收敛到 tolerance 1e-8。与理论上的最多 n 次 iterations 比较。

5. 在 size 为 10、50、200、500 的 symmetric positive definite matrices 上，为你的 Cholesky solver、LU solver 和 `np.linalg.solve` 计时。绘制结果。验证 Cholesky 大约比 LU 快 2 倍。

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|----------|----------|
| Linear system | “Solve for x” | 一组 linear equations Ax = b。寻找 x 意味着找到在 transformation A 下产生 output b 的 input。 |
| Gaussian elimination | “Row reduce” | 通过 row operations 系统地把 diagonal 下方 entries 清零，得到可用 back substitution 求解的 upper triangular system。O(n^3)。 |
| Partial pivoting | “Swap rows for stability” | 在 column k 中 eliminate 之前，把该列 absolute value 最大的 row 换到 pivot position。避免除以很小的数。 |
| LU decomposition | “Factor into triangles” | 写成 A = LU，其中 L 是 lower triangular（存储 multipliers），U 是 upper triangular（eliminated matrix）。把 O(n^3) 成本摊销到多次 solves 上。 |
| QR decomposition | “Orthogonal factorization” | 写成 A = QR，其中 Q 有 orthonormal columns，R 是 upper triangular。对 least squares 比 LU 更稳定。 |
| Cholesky decomposition | “Square root of a matrix” | 对 symmetric positive definite A，写成 A = LL^T。成本约为 LU 的一半。用于 covariance matrices、kernel matrices 和 ridge regression。 |
| Least squares | “Best fit when exact is impossible” | 当 system overdetermined（equations 比 unknowns 多）时，最小化 squared residuals 之和 ||Ax - b||^2。 |
| Normal equations | “The calculus shortcut” | A^T A x = A^T b。把 ||Ax - b||^2 的 gradient 设为 0。这就是 linear regression 的 closed-form solution。 |
| Pseudoinverse | “Inversion for non-square matrices” | 通过 SVD 得到 A+ = V Sigma+ U^T。对任意 matrix（square 或 rectangular，singular 或 not）给出 minimum-norm least-squares solution。 |
| Condition number | “How trustworthy is this answer” | kappa = sigma_max / sigma_min。衡量对 input perturbations 的敏感度。大约损失 log10(kappa) 位 precision。 |
| Ridge regression | “Regularized least squares” | 求解 (X^T X + lambda I) w = X^T y。添加 lambda I 会改善 conditioning，并把 weights shrink toward zero。防止 overfitting。 |
| Conjugate gradient | “Iterative Ax=b for big matrices” | 用于 symmetric positive definite systems 的 iterative solver。最多 n 步收敛。适合 factorization 太昂贵的大型 sparse systems。 |
| Overdetermined system | “More data than parameters” | 在 m-by-n system 中 m > n。不存在 exact solution。Least squares 会找到最佳近似。这就是每个 regression problem。 |
| Back substitution | “Solve from the bottom up” | 给定 upper triangular system，先求解最后一个 equation，再向上 substitute。O(n^2)。 |
| Forward substitution | “Solve from the top down” | 给定 lower triangular system，先求解第一个 equation，再向下 substitute。O(n^2)。用于 LU solve 的 L step。 |

## 延伸阅读

- [MIT 18.06: Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (Gilbert Strang) -- linear systems 和 matrix factorizations 的权威课程
- [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html) (Trefethen & Bau) -- 理解 numerical stability、conditioning，以及 algorithms 为什么失败的标准参考
- [Matrix Computations](https://www.cs.cornell.edu/cv/GolubVanLoan4/golubandvanloan.htm) (Golub & Van Loan) -- 覆盖各类 matrix algorithms 的百科式参考
- [3Blue1Brown: Inverse Matrices](https://www.3blue1brown.com/lessons/inverse-matrices) -- 关于求解 Ax = b 几何意义的 visual intuition
