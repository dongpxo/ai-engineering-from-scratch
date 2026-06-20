# 从零实现反向传播

> 反向传播是让学习成为可能的算法。没有它，神经网络只是昂贵的随机数生成器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.02 (Multi-Layer Networks)
**Time:** ~120 minutes

## 学习目标

- 实现一个基于 Value 的 autograd 引擎，它能构建计算图，并通过拓扑排序计算梯度
- 使用链式法则推导加法、乘法和 sigmoid 的 backward pass
- 只用从零实现的反向传播引擎，在 XOR 和圆形分类任务上训练多层网络
- 识别深层 sigmoid 网络中的梯度消失问题，并解释为什么梯度会指数级缩小

## 问题

你的网络有一个隐藏层，包含 768 个输入和 3072 个输出。这就是 2,359,296 个权重。它做出了错误预测。哪些权重导致了这个错误？如果逐个测试每个权重，就意味着要跑 230 多万次前向传播。反向传播能在一次 backward pass 中计算全部 230 多万个梯度。这不是优化，而是“可训练”和“不可能训练”的区别。

朴素做法是：取一个权重，把它轻微扰动一下，再跑一次前向传播，测量 loss 是上升还是下降。这样你就得到了这个权重的梯度。现在对网络里的每个权重都做一遍。再乘以数千个训练步骤和数百万个数据点。你需要地质年代级别的时间才能训练出任何有用的东西。

反向传播解决了这个问题。一次前向传播，一次反向传播，计算出所有梯度。诀窍是把微积分中的链式法则系统地应用到计算图上。这正是让深度学习变得可行的算法。没有它，我们还会困在玩具问题里。

## 概念

### 链式法则，应用到网络中

你在 Phase 01, Lesson 05 中见过链式法则。快速回顾：如果 y = f(g(x))，那么 dy/dx = f'(g(x)) * g'(x)。你沿着链条把导数乘起来。

在神经网络中，这条“链”就是从输入到 loss 的操作序列。每一层应用权重、加上偏置、通过激活函数。loss 函数把最终输出和目标值进行比较。反向传播会沿着这条链反向追踪，计算每个操作对误差的贡献。

### 计算图

每次前向传播都会构建一张图。每个节点是一个操作（乘法、加法、sigmoid）。每条边会向前传递数值，向后传递梯度。

```mermaid
graph LR
    x["x"] --> mul["*"]
    w["w"] --> mul
    mul -- "z1 = w*x" --> add["+"]
    b["b"] --> add
    add -- "z2 = z1 + b" --> sig["sigmoid"]
    sig -- "a = sigmoid(z2)" --> loss["Loss"]
    y["target"] --> loss
```

前向传播：数值从左到右流动。x 和 w 产生 z1 = w*x。加上 b 得到 z2。Sigmoid 给出激活值 a。再用 loss 函数把 a 和目标 y 进行比较。

反向传播：梯度从右到左流动。从 dL/da 开始，也就是 loss 如何随激活值变化。乘以 da/dz2，也就是 sigmoid 的导数，得到 dL/dz2。然后拆成 dL/db（因为 z2 = z1 + b，所以它等于 dL/dz2）和 dL/dz1。接着 dL/dw = dL/dz1 * x，dL/dx = dL/dz1 * w。

图中的每个节点在 backward pass 中只有一个职责：接收上游传来的梯度，乘以自己的局部导数，再把结果传下去。

### 前向与反向

```mermaid
graph TB
    subgraph Forward["Forward Pass"]
        direction LR
        f1["Input x"] --> f2["z = Wx + b"]
        f2 --> f3["a = sigmoid(z)"]
        f3 --> f4["Loss = (a - y)^2"]
    end
    subgraph Backward["Backward Pass"]
        direction RL
        b4["dL/dL = 1"] --> b3["dL/da = 2(a-y)"]
        b3 --> b2["dL/dz = dL/da * a(1-a)"]
        b2 --> b1["dL/dW = dL/dz * x\ndL/db = dL/dz"]
    end
    Forward --> Backward
```

前向传播会保存每个中间值：z、a、每一层的输入。反向传播需要这些保存下来的值来计算梯度。这就是反向传播核心处的内存-计算权衡：用内存保存激活值，换取速度，也就是用一次 pass 取代数百万次。

### 梯度如何流过网络

对于一个 3 层网络，梯度会链式穿过每一层：

```mermaid
graph RL
    L["Loss"] -- "dL/da3" --> L3["Layer 3\na3 = sigmoid(z3)"]
    L3 -- "dL/dz3 = dL/da3 * sigmoid'(z3)" --> L2["Layer 2\na2 = sigmoid(z2)"]
    L2 -- "dL/dz2 = dL/da2 * sigmoid'(z2)" --> L1["Layer 1\na1 = sigmoid(z1)"]
    L1 -- "dL/dz1 = dL/da1 * sigmoid'(z1)" --> I["Input"]
```

在每一层，梯度都会乘以 sigmoid 的导数。sigmoid 的导数是 a * (1 - a)，最大值是 0.25（当 a = 0.5 时）。三层之后，梯度最多已经乘上 0.25^3 = 0.0156。十层之后：0.25^10 = 0.000001。

### 梯度消失

这就是梯度消失问题。Sigmoid 把输出压缩到 0 和 1 之间。它的导数永远小于 0.25。堆叠足够多的 sigmoid 层之后，梯度会缩小到几乎没有。早期层几乎学不到东西，因为它们收到的梯度接近 0。

```text
sigmoid(z):     Output range [0, 1]
sigmoid'(z):    Max value 0.25 (at z = 0)

After 5 layers:   gradient * 0.25^5 = 0.001x original
After 10 layers:  gradient * 0.25^10 = 0.000001x original
```

这就是为什么深层 sigmoid 网络几乎无法训练。修复办法，也就是 ReLU 及其变体，是 Lesson 04 的主题。现在先理解：反向传播本身工作得很好。问题在于它要穿过什么。

### 推导两层网络的梯度

下面是一个具体网络的数学：输入 x，隐藏层使用 sigmoid，输出层使用 sigmoid，loss 使用 MSE。

Forward pass:

```text
z1 = W1 * x + b1
a1 = sigmoid(z1)
z2 = W2 * a1 + b2
a2 = sigmoid(z2)
L = (a2 - y)^2
```

Backward pass（逐步应用链式法则）：

```text
dL/da2 = 2(a2 - y)
da2/dz2 = a2 * (1 - a2)
dL/dz2 = dL/da2 * da2/dz2 = 2(a2 - y) * a2 * (1 - a2)

dL/dW2 = dL/dz2 * a1
dL/db2 = dL/dz2

dL/da1 = dL/dz2 * W2
da1/dz1 = a1 * (1 - a1)
dL/dz1 = dL/da1 * da1/dz1

dL/dW1 = dL/dz1 * x
dL/db1 = dL/dz1
```

每个梯度都是从 loss 回溯而来的局部导数乘积。这就是反向传播的全部。

```figure
backprop-vanishing
```

## Build It

### Step 1: Value 节点

我们计算中的每个数字都会变成一个 Value。它保存自己的 data、自己的 gradient，以及自己是如何被创建出来的，这样它就知道如何向后计算梯度。

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

此时还没有梯度（0.0），也没有 backward 函数（no-op）。`_children` 会记录哪些 Values 生成了当前 Value，这样稍后就能对图做拓扑排序。

### Step 2: 带 backward 函数的操作

每个操作都会创建一个新的 Value，并定义梯度如何通过它向后流动。

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

对于加法：d(a+b)/da = 1，d(a+b)/db = 1。所以两个输入都会直接收到输出的梯度。

对于乘法：d(a*b)/da = b，d(a*b)/db = a。每个输入都会收到“另一个输入的值乘以输出梯度”。

这里的 `+=` 非常关键。一个 Value 可能会被多个操作使用。它的梯度是来自所有路径的梯度之和。

### Step 3: Sigmoid 和 Loss

```python
import math

def sigmoid(self):
    x = self.data
    x = max(-500, min(500, x))
    s = 1.0 / (1.0 + math.exp(-x))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
        self.grad += (s * (1 - s)) * out.grad

    out._backward = _backward
    return out
```

Sigmoid 的导数是：sigmoid(x) * (1 - sigmoid(x))。我们已经在前向传播中算出了 sigmoid(x) = s。直接复用它，不需要额外计算。

```python
def mse_loss(predicted, target):
    diff = predicted + Value(-target)
    return diff * diff
```

单输出的 MSE 是：(predicted - target)^2。我们把减法表示成加上一个取负的 Value。

### Step 4: Backward Pass

拓扑排序可以确保节点按正确顺序处理，也就是某个节点的梯度已经完全累积之后，才通过它继续向下传播。

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

从 loss 开始（gradient = 1.0，因为 dL/dL = 1）。沿着排序后的图反向遍历。每个节点的 `_backward` 会把梯度推给它的子节点。

### Step 5: Layer 和 Network

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        scale = (2.0 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class Network:
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
```

Neuron 接收输入，计算加权和加偏置，然后应用 sigmoid。权重初始化会按 sqrt(2/n_inputs) 缩放，避免深层网络中的 sigmoid 饱和。Layer 是一组 Neuron。Network 是一组 Layer。`parameters()` 方法会收集所有可学习的 Values，方便我们更新它们。

### Step 6: 在 XOR 上训练

```python
random.seed(42)
net = Network([2, 4, 1])

xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

learning_rate = 1.0

for epoch in range(1000):
    total_loss = Value(0.0)
    for inputs, target in xor_data:
        x = [Value(i) for i in inputs]
        pred = net(x)
        loss = mse_loss(pred, target)
        total_loss = total_loss + loss

    net.zero_grad()
    total_loss.backward()

    for p in net.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {total_loss.data:.6f}")

print("\nXOR Results:")
for inputs, target in xor_data:
    x = [Value(i) for i in inputs]
    pred = net(x)
    print(f"  {inputs} -> {pred.data:.4f} (expected {target})")
```

观察 loss 如何下降。网络会从随机预测走向正确的 XOR 输出，整个过程完全由反向传播计算梯度，并把权重朝正确方向轻轻推动。

### Step 7: 圆形分类

在 Lesson 02 中，你为圆形分类手动调过权重。现在让网络自己学习这些权重。

```python
random.seed(7)

def generate_circle_data(n=100):
    data = []
    for _ in range(n):
        x1 = random.uniform(-1.5, 1.5)
        x2 = random.uniform(-1.5, 1.5)
        label = 1.0 if x1 * x1 + x2 * x2 < 1.0 else 0.0
        data.append(([x1, x2], label))
    return data

circle_data = generate_circle_data(80)

circle_net = Network([2, 8, 1])
learning_rate = 0.5

for epoch in range(2000):
    random.shuffle(circle_data)
    total_loss_val = 0.0
    for inputs, target in circle_data:
        x = [Value(i) for i in inputs]
        pred = circle_net(x)
        loss = mse_loss(pred, target)
        circle_net.zero_grad()
        loss.backward()
        for p in circle_net.parameters():
            p.data -= learning_rate * p.grad
        total_loss_val += loss.data

    if epoch % 200 == 0:
        correct = 0
        for inputs, target in circle_data:
            x = [Value(i) for i in inputs]
            pred = circle_net(x)
            predicted_class = 1.0 if pred.data > 0.5 else 0.0
            if predicted_class == target:
                correct += 1
        accuracy = correct / len(circle_data) * 100
        print(f"Epoch {epoch:4d} | Loss: {total_loss_val:.4f} | Accuracy: {accuracy:.1f}%")
```

这里使用在线 SGD，也就是每个样本之后就更新权重，而不是累积完整 batch。这会更快打破对称性，并避免完整 loss landscape 上的 sigmoid 饱和。每个 epoch 都打乱数据，可以防止网络记住样本顺序。

不需要手动调权重。网络会自己发现圆形决策边界。这就是反向传播的力量：你定义架构、loss 函数和数据。算法会找出权重。

## Use It

PyTorch 用几行代码完成上面的全部内容。核心思想完全一样：autograd 会在前向传播期间构建计算图，再沿着它反向追踪以计算梯度。

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
criterion = nn.MSELoss()

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("PyTorch XOR Results:")
with torch.no_grad():
    for i in range(4):
        pred = model(X[i])
        print(f"  {X[i].tolist()} -> {pred.item():.4f} (expected {y[i].item()})")
```

`loss.backward()` 就是你的 `total_loss.backward()`。`optimizer.step()` 就是你手动写的 `p.data -= lr * p.grad`。`optimizer.zero_grad()` 就是你的 `net.zero_grad()`。同一个算法，工业级实现。PyTorch 会处理 GPU 加速、混合精度、梯度检查点和数百种层类型。但 backward pass 仍然是同一个链式法则，应用在同一类计算图上。

训练会运行前向传播、反向传播，然后更新权重。推理只运行前向传播。没有梯度，没有更新。这个区别很重要，因为生产环境中发生的是推理。当你调用 Claude 或 GPT 这样的 API 时，你正在运行推理：你的 prompt 向前流过网络，token 从另一端出来。权重不会改变。理解反向传播很重要，因为正是它塑造了那个网络里的每一个权重。

## Ship It

本课产出：

- `outputs/prompt-gradient-debugger.md` - 一个可复用提示词，用于诊断任何神经网络中的梯度问题（消失、爆炸、NaN）

## 练习

1. 给 Value 类添加 `__sub__` 方法（a - b = a + (-1 * b)）。然后实现 `__neg__` 方法。用一个简单表达式，比如 (a - b)^2，和手算结果对比，验证梯度是否正确。

2. 给 Value 添加 `relu` 方法（输出 max(0, x)，导数在 x > 0 时为 1，否则为 0）。把隐藏层中的 sigmoid 换成 relu，再次在 XOR 上训练。比较收敛速度。你应该会看到训练更快，这会预告 Lesson 04。

3. 在 Value 上实现一个用于整数幂的 `__pow__` 方法。用它把 `mse_loss` 换成真正的 `(predicted - target) ** 2` 表达式。验证梯度和原实现一致。

4. 给训练循环添加梯度裁剪：调用 `backward()` 之后，把所有梯度裁剪到 [-1, 1]。训练一个更深的网络（4+ 层 sigmoid），比较有无裁剪时的 loss 曲线。这是你抵御梯度爆炸的第一道防线。

5. 构建一个可视化：在 XOR 训练完成后，打印网络中每个参数的梯度。找出哪一层的梯度最小。这会演示你在概念部分读到的梯度消失问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 反向传播 | “网络在学习” | 一种算法，通过沿计算图反向应用链式法则，为每个权重计算 dL/dw |
| 计算图 | “网络结构” | 一个有向无环图，节点是操作，边向前携带数值、向后携带梯度 |
| 链式法则 | “把导数乘起来” | 如果 y = f(g(x))，那么 dy/dx = f'(g(x)) * g'(x)，这是反向传播的数学基础 |
| 梯度 | “最陡上升方向” | loss 对某个参数的偏导数，告诉你如何改变该参数以降低 loss |
| 梯度消失 | “深层网络学不动” | 梯度在穿过 sigmoid 等饱和激活层时指数级缩小 |
| 前向传播 | “运行网络” | 通过顺序应用每层操作，从输入计算输出，并保存中间值 |
| 反向传播过程 | “计算梯度” | 反向遍历计算图，并用链式法则在每个节点累积梯度 |
| 学习率 | “学得有多快” | 更新权重时控制步长的标量：w_new = w_old - lr * gradient |
| 拓扑排序 | “正确顺序” | 一种图节点排序方式，每个节点都出现在它依赖的所有节点之后，确保梯度在传播前已完全累积 |
| Autograd | “自动微分” | 在前向计算期间构建计算图并自动计算梯度的系统，也就是 PyTorch 引擎所做的事 |

## 延伸阅读

- Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" (1986) - 让反向传播成为主流、并打开多层网络训练大门的论文
- 3Blue1Brown, "Neural Networks" series (https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - 关于反向传播以及梯度如何流过网络的优秀可视化解释
