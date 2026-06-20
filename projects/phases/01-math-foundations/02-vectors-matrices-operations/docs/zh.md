# 向量、矩阵与运算

> 每个神经网络，本质上都是矩阵乘法加上一些额外步骤。

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 学习目标

- 构建一个 Matrix 类，支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法与矩阵乘法，并解释它们分别适用于什么场景
- 只使用从零实现的 Matrix 类，写出一个稠密神经网络层：`relu(W @ x + b)`
- 解释广播规则，以及神经网络框架中的偏置加法是如何工作的

## 问题

你想构建一个神经网络。你打开代码，看见这一行：

```python
output = activation(weights @ input + bias)
```

这里的 `@` 是矩阵乘法。`weights` 是一个矩阵，`input` 是一个向量。如果你不知道这些运算在做什么，这一行就像魔法。如果你知道，它就是一个神经网络层的完整前向传播，只用了三个操作。

模型处理的每张图像都是一组像素值矩阵。每个词嵌入都是一个向量。每个神经网络的每一层都是一次矩阵变换。你不理解矩阵运算，就很难真正构建 AI 系统，就像不会变量也很难写程序一样。

这一课会从零开始建立这种熟练度。

## 概念

### 向量：有顺序的数字列表

向量是一串有顺序的数字，同时带有方向和大小。在 AI 中，向量可以表示数据点、特征或参数。

```text
v = [3, 4]        -- 二维向量
w = [1, 0, -2]    -- 三维向量
```

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度，也就是大小，是 5，也就是 3-4-5 直角三角形。

### 矩阵：数字网格

矩阵是一个二维网格，由行和列组成。一个 m x n 矩阵有 m 行、n 列。

```text
A = | 1  2  3 |     -- 2x3 矩阵（2 行，3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵会把输入向量变换成输出向量。一个拥有 784 个输入和 128 个输出的层，会使用一个 128x784 的权重矩阵。

### 为什么形状很重要

矩阵乘法有一条严格规则：`(m x n) @ (n x p) = (m x p)`。中间两个维度必须相同。

```text
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

内侧维度：784 = 784  -- 有效
```

如果你在 PyTorch 中遇到 shape mismatch 错误，原因就在这里。

### 运算地图

| 运算 | 它做什么 | 神经网络中的用途 |
|------|----------|------------------|
| 加法 | 逐元素组合 | 给输出加偏置 |
| 标量乘法 | 缩放每个元素 | 学习率 * 梯度 |
| 矩阵乘法 | 变换向量 | 层的前向传播 |
| 转置 | 交换行和列 | 反向传播 |
| 行列式 | 一个数字摘要 | 检查是否可逆 |
| 逆矩阵 | 撤销一个变换 | 求解线性系统 |
| 单位矩阵 | 什么都不改变的矩阵 | 初始化、残差连接 |

### 逐元素乘法 vs 矩阵乘法

这个区别经常让初学者卡住。

逐元素乘法：把相同位置的元素相乘。两个矩阵必须有相同形状。

```text
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：计算行与列之间的点积。内侧维度必须匹配。

```text
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同运算，不同结果，不同规则。

### 广播

当你把一个偏置向量加到一组输出矩阵上时，两者形状并不相同。广播会把较小的数组“拉伸”到合适的形状。

```text
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播会把这个向量沿着行方向复制：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都会自动完成这件事。理解广播，可以避免你在形状看似不匹配但代码仍能运行时感到困惑。

```figure
vector-projection
```

## Build It

### 第 1 步：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：看它运行

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个稠密层：`output = relu(W @ x + b)`。每个神经网络中的每个稠密层，本质上都在做这件事。

## Use It

NumPy 可以用更少的代码完成上面所有操作，并且快几个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符会调用 `__matmul__`。NumPy 使用 C 和 Fortran 编写的优化 BLAS 例程来实现它。数学相同，速度可以快上 100 倍。

NumPy 中的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 会自动把一维偏置广播到两行上。每个神经网络框架中的偏置加法都是这样工作的。

## Ship It

这一课会产出一个提示词，用几何直觉来讲解矩阵运算。见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类，是我们在 Phase 3, Lesson 10 中构建迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()`，确认你得到的是单位矩阵。用三个不同的 2x2 矩阵试一试。当行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix 类，使用伴随矩阵方法计算 3x3 矩阵的逆。用 NumPy 的 `np.linalg.inv` 对照测试。

3. **构建一个两层网络。** 只使用你的 Matrix 类，不使用 NumPy，创建一个两层神经网络：input (3) -> hidden (4) -> output (2)。随机初始化权重，运行一次前向传播，并验证所有形状都是正确的。

## 关键术语

| 术语 | 人们常说 | 它真正的含义 |
|------|----------|--------------|
| 向量 | “一个箭头” | 一串有顺序的数字。在 AI 中，是高维空间中的一个点。 |
| 矩阵 | “一张数字表” | 一个线性变换。它把向量从一个空间映射到另一个空间。 |
| 矩阵乘法 | “就是把数字乘起来” | 第一个矩阵的每一行与第二个矩阵的每一列做点积。顺序很重要。 |
| 转置 | “翻一下” | 交换行和列。把 m x n 矩阵变成 n x m 矩阵。在反向传播中很关键。 |
| 行列式 | “矩阵算出来的某个数” | 衡量矩阵把面积（二维）或体积（三维）缩放了多少。为零意味着这个变换压扁了某个维度。 |
| 逆矩阵 | “撤销这个矩阵” | 能反转该变换的矩阵。只有当行列式不为零时才存在。 |
| 单位矩阵 | “无聊的矩阵” | 矩阵中的乘以 1。用于残差连接，例如 ResNet。 |
| 广播 | “神奇的形状修复” | 通过沿缺失维度重复，把较小数组拉伸到较大数组的形状。 |
| 逐元素 | “普通乘法” | 对相同位置的元素相乘。两个数组必须形状相同，或可以广播。 |

## 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 对本课覆盖的每个运算提供可视化直觉
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的精确广播规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习的简明线性代数参考
