# 优化器

> 梯度下降告诉你往哪个方向走。它没有告诉你走多远、多快。SGD 是指南针。Adam 是带实时路况的 GPS。

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.05 (Loss Functions)
**Time:** ~75 minutes

## 学习目标

- 在 Python 中从零实现 SGD、带 momentum 的 SGD、Adam 和 AdamW 优化器
- 解释 Adam 的 bias correction 如何补偿训练早期从零初始化的矩估计
- 在同一个任务上演示为什么 AdamW 比带 L2 正则的 Adam 泛化更好
- 为 transformers、CNN、GAN 和微调任务选择合适的优化器和默认超参数

## 问题

你已经算出了梯度。你知道第 4,721 个权重应该减少 0.003，才能降低 loss。但 0.003 是什么单位？应该按什么比例缩放？第 1 步和第 1,000 步应该移动同样的量吗？

普通梯度下降会在每一步对每个参数使用同一个学习率：w = w - lr * gradient。这会带来三个问题，让神经网络训练在实践中变得痛苦。

第一，震荡。Loss landscape 很少像一个平滑碗，更像一条又长又窄的山谷。梯度指向横穿山谷的方向（陡峭方向），而不是沿着山谷前进的方向（平缓方向）。梯度下降会在狭窄维度上来回弹跳，却只在真正有用的方向上缓慢前进。你见过这种现象：loss 很快下降，然后进入平台期，不是因为模型收敛了，而是因为它在震荡。

第二，对所有参数使用同一个学习率是错的。有些权重需要大更新（它们还在早期欠拟合阶段）。另一些权重只需要小更新（它们已经接近最优值）。适合前者的学习率会毁掉后者，反过来也一样。

第三，鞍点。在高维空间中，loss landscape 有大量平坦区域，梯度接近 0。普通 SGD 会以梯度的速度爬过这些区域，而这个速度实际上接近 0。模型看起来卡住了。它并没有真的卡住，只是处在一个平坦区域，另一边仍然有可用的下降方向。但 SGD 没有机制把它推过去。

Adam 解决了这三个问题。它为每个参数维护两个运行平均值：平均梯度（momentum，用来处理震荡）和平方梯度的平均值（adaptive rate，用来处理不同尺度）。再加上前几步的 bias correction，它给了你一个默认超参数就能解决 80% 问题的优化器。本课会从零构建它，这样你就能准确理解它在另外 20% 情况下何时、为何失效。

## 概念

### 随机梯度下降（SGD）

最简单的优化器。在一个 mini-batch 上计算梯度，然后朝相反方向迈一步。

```text
w = w - lr * gradient
```

“随机”指的是你使用随机子集（mini-batch）来估计梯度，而不是使用整个数据集。这种噪声其实有用，它有助于逃离尖锐的局部最小值。但噪声也会造成震荡。

学习率是唯一旋钮。太高：loss 发散。太低：训练永远跑不完。最优值取决于架构、数据、batch size 和当前训练阶段。现代网络中普通 SGD 的典型值范围是 0.01 到 0.1。但即使在单次训练中，理想学习率也会变化。

### Momentum

“球滚下山坡”的类比被用得很多，但它确实准确。你不再只按当前梯度迈步，而是维护一个会累积过去梯度的速度。

```text
m_t = beta * m_{t-1} + gradient
w = w - lr * m_t
```

Beta（通常为 0.9）控制保留多少历史信息。当 beta = 0.9 时，momentum 大约是最近 10 个梯度的平均值（1 / (1 - 0.9) = 10）。

为什么它能修复震荡：指向同一方向的梯度会累积。方向反复翻转的梯度会互相抵消。在狭窄山谷中，“横穿”分量每一步都会变号，因此会被抑制。“沿着山谷”的分量保持一致，因此会被放大。结果就是在有用方向上平滑加速。

真实数字：在条件很差的 loss landscape 上，单独使用 SGD 可能需要 10,000 步。带 momentum 的 SGD（beta=0.9）通常只需要 3,000-5,000 步。这个加速不是微小改进。

### RMSProp

第一个真正有效的逐参数自适应学习率方法。由 Hinton 在 Coursera 课程中提出（从未正式发表）。

```text
s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)
```

s_t 追踪平方梯度的运行平均值。长期拥有大梯度的参数会除以一个大数（更小的有效学习率）。梯度较小的参数会除以一个小数（更大的有效学习率）。

这解决了“所有参数使用同一个学习率”的问题。一个一直得到大更新的权重可能已经接近目标，应该放慢。一个一直只得到小更新的权重可能训练不足，应该加速。

Epsilon（通常为 1e-8）用于防止某个参数还没更新过时发生除零。

### Adam：Momentum + RMSProp

Adam 结合了两种想法。它为每个参数维护两个指数移动平均：

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        (first moment: mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       (second moment: variance)
```

**Bias correction** 是大多数解释会略过的关键细节。在第 1 步，m_1 = (1 - beta1) * gradient。当 beta1 = 0.9 时，它就是 0.1 * gradient，比真实梯度小了十倍。移动平均还没热起来。Bias correction 会补偿这一点：

```text
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

在 beta1 = 0.9 的第 1 步：m_hat = m_1 / (1 - 0.9) = m_1 / 0.1 = 真实梯度。到第 100 步时，(1 - 0.9^100) 近似为 1.0，所以修正项消失。Bias correction 对前约 10 步很重要，约 50 步后基本无关。

更新规则：

```text
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Adam 默认值：lr = 0.001，beta1 = 0.9，beta2 = 0.999，epsilon = 1e-8。这些默认值适用于 80% 的问题。当它们不适用时，先改 lr。再改 beta2。几乎不要改 beta1 或 epsilon。

### AdamW：正确的 Weight Decay

L2 正则会把 lambda * w^2 加到 loss 上。在普通 SGD 中，这等价于 weight decay，也就是每一步从权重中减去 lambda * w。在 Adam 中，这种等价关系会失效。

Loshchilov 和 Hutter 的洞见是：当你把 L2 加到 loss 中，然后 Adam 处理梯度时，自适应学习率也会缩放正则项。梯度方差大的参数得到的正则更少。方差小的参数得到的正则更多。这不是你想要的。你想要的是不管梯度统计如何，都对参数进行均匀正则。

AdamW 通过在 Adam 更新之后直接对权重应用 weight decay 来修复这一点：

```text
w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w
```

Weight decay 项（lr * lambda * w）不会被 Adam 的自适应因子缩放。每个参数都会得到相同的比例收缩。

这看起来像小细节。不是。几乎在所有任务上，AdamW 都比 Adam + L2 正则收敛到更好的解。它是 PyTorch 中训练 transformer、扩散模型和大多数现代架构的默认优化器。BERT、GPT、LLaMA、Stable Diffusion 都使用 AdamW 训练。

### 学习率：最重要的超参数

```mermaid
graph TD
    LR["Learning Rate"] --> TooHigh["Too high (lr > 0.01)"]
    LR --> JustRight["Just right"]
    LR --> TooLow["Too low (lr < 0.00001)"]

    TooHigh --> Diverge["Loss explodes<br/>NaN weights<br/>Training crashes"]
    JustRight --> Converge["Loss decreases steadily<br/>Reaches good minimum<br/>Generalizes well"]
    TooLow --> Stall["Loss decreases slowly<br/>Gets stuck in suboptimal minimum<br/>Wastes compute"]

    JustRight --> Schedule["Usually needs scheduling"]
    Schedule --> Warmup["Warmup: ramp from 0 to max<br/>First 1-10% of training"]
    Schedule --> Decay["Decay: reduce over time<br/>Cosine or linear"]
```

如果你只调一个超参数，就调学习率。学习率改变 10 倍，比你做的大多数架构选择都更重要。常见默认值：

- SGD: lr = 0.01 到 0.1
- Adam/AdamW: lr = 1e-4 到 3e-4
- 微调预训练模型: lr = 1e-5 到 5e-5
- 学习率 warmup: 在前 1-10% 训练步中线性升高

### 优化器对比

```mermaid
flowchart LR
    subgraph "Optimization Path"
        SGD_P["SGD<br/>Oscillates across valley<br/>Slow but finds flat minima"]
        Mom_P["SGD + Momentum<br/>Smoother path<br/>3x faster than SGD"]
        Adam_P["Adam<br/>Adapts per-parameter<br/>Fast convergence"]
        AdamW_P["AdamW<br/>Adam + proper decay<br/>Best generalization"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P
```

### 每种优化器何时胜出

```mermaid
flowchart TD
    Task["What are you training?"] --> Type{"Model type?"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + Momentum<br/>lr=0.1, momentum=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"Fine-tuning"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"Don't know yet"| Default["Start with AdamW<br/>lr=3e-4, wd=0.01"]
```

```figure
optimizer-trajectory
```

## Build It

### Step 1: 普通 SGD

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

### Step 2: 带 Momentum 的 SGD

```python
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

### Step 3: Adam

```python
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

### Step 4: AdamW

```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]
```

### Step 5: 训练对比

在 Lesson 05 的圆形数据集上，用四种优化器训练同一个两层网络。比较收敛情况。

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## Use It

PyTorch 优化器会处理参数组、梯度裁剪和学习率调度：

```python
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

固定模式永远是：zero_grad、forward、loss、backward、（clip）、step、（schedule）。记住这个顺序。顺序写错，比如在 `optimizer.step()` 之前调用 `scheduler.step()`，是细微 bug 的常见来源。

对于 CNN，很多实践者仍然偏好 SGD + momentum（lr=0.1，momentum=0.9，weight_decay=1e-4），配合 step 或 cosine schedule。SGD 会找到更平坦的最小值，通常泛化更好。对于 transformers 和 LLM，带 warmup + cosine decay 的 AdamW 是通用默认选择。没有测量依据，不要和共识硬碰硬。

## Ship It

本课产出：

- `outputs/prompt-optimizer-selector.md` - 一个决策提示词，用于为任意架构选择正确的优化器和学习率

## 练习

1. 实现 Nesterov momentum，也就是在“前瞻”位置（w - lr * beta * v）而不是当前位置计算梯度。在圆形数据集上比较它和标准 momentum 的收敛情况。

2. 实现学习率 warmup schedule：在前 10% 训练步骤中从 0 线性升高到 max_lr，然后 cosine decay 到 0。比较 Adam + warmup 和没有 warmup 的 Adam。测量在圆形数据集上达到 90% 准确率需要多少个 epoch。

3. 在 Adam 训练期间追踪每个参数的有效学习率。有效学习率是 lr * m_hat / (sqrt(v_hat) + eps)。绘制第 10、50 和 200 步后的有效学习率分布。所有参数都在以同样速度更新吗？

4. 实现梯度裁剪（按全局范数裁剪）。把最大梯度范数设为 1.0。用较高学习率（Adam 的 lr=0.01）分别在有裁剪和无裁剪情况下训练。用 10 个随机种子统计有多少次运行发散（loss 变成 NaN）。

5. 在大权重网络上比较 Adam 和 AdamW。把所有权重初始化为 [-5, 5] 中的随机值（远大于正常值）。使用 weight_decay=0.1 训练 200 个 epoch。绘制两种优化器训练期间权重 L2 范数的变化。AdamW 应该表现出更快的权重收缩。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 学习率 | “步长” | 梯度更新上的标量乘数，是训练中影响最大的单个超参数 |
| SGD | “基础梯度下降” | 随机梯度下降：用 mini-batch 计算梯度，并通过减去 lr * gradient 更新权重 |
| Momentum | “滚动小球类比” | 过去梯度的指数移动平均，用来抑制震荡并加速一致方向 |
| RMSProp | “自适应学习率” | 用近期梯度的运行 RMS 除以每个参数的梯度，从而均衡学习率 |
| Adam | “默认优化器” | 把 momentum（一阶矩）和 RMSProp（二阶矩）结合起来，并对初始步骤做 bias correction |
| AdamW | “正确版 Adam” | 带解耦 weight decay 的 Adam，直接对权重应用正则，而不是通过梯度应用 |
| Bias correction | “运行平均的 warmup” | 除以 (1 - beta^t)，用于补偿 Adam 矩估计从零初始化带来的偏差 |
| Weight decay | “缩小权重” | 每一步减去权重值的一部分，是一种惩罚大权重的正则化方法 |
| 学习率调度 | “随时间改变 lr” | 训练期间调整学习率的函数；warmup + cosine decay 是现代默认做法 |
| 梯度裁剪 | “限制梯度范数” | 当梯度向量范数超过阈值时把它缩小，用来防止梯度更新爆炸 |

## 延伸阅读

- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) - Adam 原始论文，包含收敛分析和 bias correction 推导
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017) - 证明 L2 正则和 weight decay 在 Adam 中不等价，并提出 AdamW
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) - 引入 LR range test 和循环调度，减少固定学习率调参需求
- Ruder, "An Overview of Gradient Descent Optimization Algorithms" (2016) - 关于各种优化器变体的优秀综述，比较和直觉都很清晰
