# 机器学习微积分

> 导数告诉你哪个方向是下坡。神经网络学习所需的，就是这件事。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60 minutes

## 学习目标

- 计算常见 ML 函数（x^2、sigmoid、交叉熵）的数值导数和解析导数
- 从零实现梯度下降，在 1D 和 2D 中最小化损失函数
- 推导线性回归模型的梯度，并通过手动更新权重来训练它
- 解释 Hessian 矩阵、Taylor 级数近似，以及它们和优化方法之间的联系

## 问题

你有一个包含数百万个权重的神经网络。每个权重都是一个旋钮。你需要弄清楚每一个旋钮该往哪个方向拧，才能让模型稍微少错一点。微积分给你的正是这个方向。

没有微积分，训练神经网络就意味着随机尝试改动，然后祈祷结果变好。有了导数，你就能准确知道每个权重如何影响误差。每一次，你都把每个旋钮朝正确方向拧。

## 概念

### 什么是导数？

导数衡量变化率。对于函数 y = f(x)，导数 f'(x) 告诉你：如果把 x 轻轻移动一点点，y 会变化多少？

从几何上看，导数就是某一点处切线的斜率。

**f(x) = x^2：**

| x | f(x) | f'(x)（斜率） |
|---|------|---------------|
| 0 | 0    | 0（平坦，在底部） |
| 1 | 1    | 2 |
| 2 | 4    | 4（这一点处的切线斜率） |
| 3 | 9    | 6 |

在 x=2 时，斜率是 4。如果你把 x 向右移动很小一段，y 大约会增加这段距离的 4 倍。在 x=0 时，斜率是 0。你正处在碗底。

形式化定义：

```text
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

在代码里，你不会真的求极限，而是使用一个非常小的 h。这就是数值导数。

### 偏导数：一次只看一个变量

真实函数通常有很多输入。神经网络损失依赖成千上万个权重。偏导数会把除一个变量以外的所有变量都视为常量，然后对这个变量求导。

```text
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     （把 y 当作常量）
df/dy = 3x + 2y     （把 x 当作常量）
```

每个偏导数都在回答：如果我只轻轻改变这个权重，损失会如何变化？

### 梯度：所有偏导数组成的向量

梯度把每个偏导数收集成一个向量。对于函数 f(x, y, z)，梯度是：

```text
grad f = [ df/dx, df/dy, df/dz ]
```

梯度指向函数上升最快的方向。要最小化函数，就朝相反方向走。

**f(x,y) = x^2 + y^2 的等高线图：**

这个函数形成一个碗形曲面，等高线是同心圆。最小值位于 (0, 0)。

| 点 | grad f | -grad f（下降方向） |
|----|--------|---------------------|
| (1, 1) | [2, 2]（指向上坡，远离最小值） | [-2, -2]（指向下坡，靠近最小值） |
| (0, 0) | [0, 0]（平坦，在最小值处） | [0, 0] |

这就是梯度下降的图像版：计算梯度，取反，然后迈一步。

### 与优化的联系

训练神经网络就是优化。你有一个损失函数 L(w1, w2, ..., wn)，它衡量模型错得有多严重。你想最小化它。

```text
梯度下降更新规则：

  w_new = w_old - learning_rate * dL/dw

对于每个权重：
  1. 计算损失对该权重的偏导数
  2. 从权重中减去这个偏导数的一小部分
  3. 重复
```

学习率控制步长。太大，会越过最低点。太小，会爬得很慢。

**损失地形（1D 切片）：**

当权重 w 变化时，损失函数 L(w) 会形成一条有峰有谷的曲线。

| 特征 | 描述 |
|------|------|
| 全局最小值 | 整条曲线上的最低点，也就是最佳解 |
| 局部最小值 | 比附近点低、但不是整体最低的山谷 |
| 斜率 | 梯度下降会从任意起点沿斜率向下走 |

梯度下降会顺着斜率下山。它可能卡在局部最小值里，但在高维空间中（数百万个权重），这通常不是实践中的主要问题。

### 数值导数 vs 解析导数

计算导数有两种方式。

解析导数：手动应用微积分规则。对于 f(x) = x^2，导数是 f'(x) = 2x。精确，而且快速。

数值导数：使用定义来近似。计算 f(x+h) 和 f(x-h)，其中 h 很小，然后取差值。

```text
数值导数（中心差分）：

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 在实践中通常很好用
```

数值导数更慢，但适用于任何函数。解析导数很快，但需要你推导公式。神经网络框架使用第三种方式：自动微分，它会机械地计算精确导数。你会在 Phase 3 看到它。

### 简单函数的手算导数

这些导数会在 ML 中反复出现。

```text
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

对于 f(x) = x^2：

```text
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      斜率向左倾斜（下降）
  -1    1      -2      斜率向左倾斜（下降）
   0    0       0      平坦（最小值！）
   1    1       2      斜率向右倾斜（上升）
   2    4       4      斜率向右倾斜（上升）
```

对于 f(w) = wx + b，且 x=3、b=1：

```text
f(w) = 3w + 1    f'(w) = 3

对 w 的导数就是 x。
如果 x 很大，w 的一个小变化就会导致输出发生大变化。
```

### 链式法则

当函数相互组合时，链式法则告诉你如何求导。

```text
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

示例：y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

神经网络是函数链：input -> linear -> activation -> linear -> activation -> loss。反向传播就是从输出到输入反复应用链式法则。这就是整个算法。

### Hessian 矩阵

梯度告诉你斜率。Hessian 告诉你曲率。

Hessian 是二阶偏导数组成的矩阵。对于函数 f(x1, x2, ..., xn)，Hessian 的第 (i, j) 个元素是：

```text
H[i][j] = d^2f / (dx_i * dx_j)
```

对于二变量函数 f(x, y)：

```text
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**Hessian 在临界点（gradient = 0）告诉你什么：**

| Hessian 性质 | 含义 | 示例曲面 |
|--------------|------|----------|
| 正定（所有特征值 > 0） | 局部最小值 | 开口向上的碗 |
| 负定（所有特征值 < 0） | 局部最大值 | 开口向下的碗 |
| 不定（特征值正负混合） | 鞍点 | 马鞍形曲面 |

**示例：** f(x, y) = x^2 - y^2（鞍点函数）

```text
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2（一个为正，一个为负）
--> (0, 0) 是鞍点
```

与 f(x, y) = x^2 + y^2（碗形）对比：

```text
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2（都为正）
--> (0, 0) 是局部最小值
```

**为什么 Hessian 对 ML 重要：**

Newton 方法使用 Hessian 来采取比梯度下降更好的优化步。它不只是沿斜率走，还会考虑曲率：

```text
Newton's update:    w_new = w_old - H^(-1) * gradient
Gradient descent:   w_new = w_old - lr * gradient
```

Newton 方法收敛更快，因为 Hessian 会“重新缩放”梯度：陡峭方向步子更小，平坦方向步子更大。

问题在于：对于一个有 N 个参数的神经网络，Hessian 是 N x N 的。一个拥有 100 万个参数的模型需要一个包含 1 万亿个元素的矩阵。这就是为什么我们会使用近似。

| 方法 | 使用什么 | 代价 | 收敛性 |
|------|----------|------|--------|
| 梯度下降 | 只使用一阶导数 | 每步 O(N) | 慢（线性） |
| Newton 方法 | 完整 Hessian | 每步 O(N^3) | 快（二次） |
| L-BFGS | 从梯度历史近似 Hessian | 每步 O(N) | 中等（超线性） |
| Adam | 每个参数的自适应学习率（对角 Hessian 近似） | 每步 O(N) | 中等 |
| 自然梯度 | Fisher 信息矩阵（统计意义上的 Hessian） | 每步 O(N^2) | 快 |

实践中，Adam 是深度学习默认常用的优化器。它通过跟踪每个参数梯度的滑动均值和方差，以低成本近似二阶信息。

### Taylor 级数近似

任何平滑函数都可以在局部用多项式近似：

```text
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

包含的项越多，近似越好，但只在点 x 附近成立。

**为什么 Taylor 级数对 ML 重要：**

- **一阶 Taylor = 梯度下降。** 当你使用 f(x + h) ~ f(x) + f'(x)*h 时，你正在做线性近似。梯度下降最小化这个线性模型，选择 h = -lr * f'(x)。

- **二阶 Taylor = Newton 方法。** 使用 f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2 时，你得到一个二次模型。最小化它会得到 h = -f'(x)/f''(x)，也就是 Newton 步。

- **损失函数设计。** MSE 和交叉熵都是平滑的，这意味着它们的 Taylor 展开表现良好。这不是偶然。平滑损失会让优化更可预测。

```text
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   只有函数值        随机搜索
1st order (linear)     斜率              梯度下降
2nd order (quadratic)  曲率              Newton 方法
Higher orders          更细结构          ML 中很少使用
```

关键洞见是：所有基于梯度的优化，本质上都是在局部近似损失函数，然后走向这个近似函数的最小值。

### ML 中的积分

导数告诉你变化率。积分计算累积量，也就是曲线下面积。

在 ML 中，你很少手算积分，但这个概念无处不在：

**概率。** 对于密度为 p(x) 的连续随机变量：

```text
P(a < X < b) = integral from a to b of p(x) dx
```

概率密度曲线在 a 和 b 之间的面积，就是落在这个范围内的概率。

**期望值。** 按概率加权后的平均结果：

```text
E[f(X)] = integral of f(x) * p(x) dx
```

数据分布上的期望损失就是一个积分。训练最小化的是它的经验近似。

**KL 散度。** 衡量两个分布有多不同：

```text
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```

用于 VAE、知识蒸馏和贝叶斯推断。

**归一化常数。** 在贝叶斯推断中：

```text
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```

分母是对所有可能参数值的积分。它通常不可解，这就是为什么我们会使用 MCMC 和变分推断等近似方法。

| 积分概念 | 它在 ML 中出现的位置 |
|----------|----------------------|
| 曲线下面积 | 从密度函数得到概率 |
| 期望值 | 损失函数、风险最小化 |
| KL 散度 | VAE、策略优化、蒸馏 |
| 归一化 | 贝叶斯后验、softmax 分母 |
| 边际似然 | 模型比较、证据下界（ELBO） |

### 计算图中的多变量链式法则

链式法则不只适用于排成一条线的标量函数。在神经网络中，变量会分叉，也会合并。下面展示导数如何流过一个简单前向传播：

```mermaid
graph LR
    x["x（输入）"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"loss fn"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

反向传播从右到左计算梯度：

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

每条边都会乘上局部导数。任意参数的梯度，就是从损失到该参数路径上所有局部导数的乘积。当路径分叉并合并时，你会把各条路径的贡献相加（多变量链式法则）。

反向传播就是这么回事：从输出到输入，系统地在计算图上应用链式法则。

### Jacobian 矩阵

当一个函数把向量映射到向量时（例如神经网络层），它的导数就是一个矩阵。Jacobian 包含每个输出对每个输入的所有偏导数。

对于 f: R^n -> R^m，Jacobian J 是一个 m x n 矩阵：

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

你不会手算神经网络里的 Jacobian。PyTorch 会处理。但知道它存在，可以帮你理解反向传播中的形状：如果一层把 R^n 映射到 R^m，它的 Jacobian 就是 m x n。梯度会通过这个矩阵的转置向后传播。

### 为什么这对神经网络重要

神经网络里的每个权重都会得到一个梯度。梯度告诉你如何调整这个权重，从而降低损失。

```mermaid
graph LR
    subgraph Forward["前向传播"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["反向传播"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

每次权重更新：

- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

前向传播计算预测和损失。反向传播计算损失对每个权重的梯度。然后每个权重都朝下坡方向迈一小步。重复数百万次。这就是深度学习。

```figure
derivative-tangent
```

## Build It

### 第 1 步：从零实现数值导数

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

数值导数会在很多小数位上匹配解析导数。

### 第 2 步：偏导数与梯度

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### 第 3 步：用梯度下降寻找 f(x) = x^2 的最小值

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

从 x=5 出发，每一步都会更靠近 x=0（最小值）。

### 第 4 步：在 2D 函数上做梯度下降

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### 第 5 步：比较数值导数和解析导数

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### 第 6 步：数值计算 Hessian

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

鞍点函数的 Hessian 特征值为 2 和 -2（符号混合，确认它是鞍点）。碗形函数的特征值为 2 和 2（都为正，确认它是最小值）。

### 第 7 步：Taylor 近似的实际效果

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

在 x0=0 附近，sin(x) ~ x（一阶 Taylor）。当 h 很小时，近似非常好；当 h 变大时，近似会失效。这就是为什么梯度下降在小学习率下效果最好：每一步都假设线性近似是准确的。

### 第 8 步：为什么这对神经网络重要

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

每个基于梯度的训练循环都遵循这个模式：预测，计算损失，计算梯度，更新权重。

## Use It

使用 NumPy，同样的操作更快也更简洁：

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

你已经从零构建了梯度下降。PyTorch 会自动化梯度计算，但更新循环是一样的。

## 练习

1. 使用两次调用 `numerical_derivative` 来实现 `numerical_second_derivative(f, x)`。验证 x^3 在 x=2 处的二阶导数是 12。
2. 使用梯度下降寻找 f(x, y) = (x - 3)^2 + (y + 1)^2 的最小值。从 (0, 0) 出发。答案应该收敛到 (3, -1)。
3. 给梯度下降循环添加动量：维护一个速度向量，用来累积过去的梯度。在 f(x) = x^4 - 3x^2 上比较有动量和无动量的收敛速度。

## 关键术语

| 术语 | 人们常说 | 它实际表示 |
|------|----------|------------|
| 导数 | “斜率” | 函数在某一点处的变化率。告诉你输入每变化一个单位，输出会变化多少。 |
| 偏导数 | “一个变量的导数” | 在其他变量保持不变时，对某一个变量求导。 |
| 梯度 | “最陡上升方向” | 所有偏导数组成的向量。指向让函数增长最快的方向。 |
| 梯度下降 | “向下走” | 从参数中减去梯度（乘以学习率），以降低损失。它是神经网络训练的核心。 |
| 学习率 | “步长” | 控制每次梯度下降步子大小的标量。太大：发散。太小：收敛缓慢。 |
| 链式法则 | “把导数相乘” | 对复合函数求导的规则：df/dx = df/dg * dg/dx。它是反向传播的数学基础。 |
| Jacobian | “导数矩阵” | 当函数把向量映射到向量时，Jacobian 是所有输出对所有输入的偏导数组成的矩阵。 |
| 数值导数 | “有限差分” | 通过在两个相邻点计算函数值，并求它们之间的斜率来近似导数。 |
| 反向传播 | “反向模式自动微分” | 使用链式法则，从输出到输入逐层计算梯度。神经网络正是这样学习的。 |
| Hessian | “二阶导数矩阵” | 所有二阶偏导数组成的矩阵。描述函数曲率。在临界点处 Hessian 正定意味着局部最小值。 |
| Taylor 级数 | “多项式近似” | 使用函数在某点的导数来近似附近函数值：f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ...。它是理解梯度下降和 Newton 方法为什么有效的基础。 |
| 积分 | “曲线下面积” | 某个量在一个范围内的累积。在 ML 中，积分定义概率、期望值和 KL 散度。 |

## 延伸阅读

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) - 关于导数、积分和链式法则的视觉直觉
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) - 梯度如何流过神经网络层
