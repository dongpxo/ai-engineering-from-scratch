# PyTorch 入门

> 你已经用活塞和曲轴搭出了引擎。现在学习大家实际会开的那辆车。

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.10 (Build Your Own Mini Framework)
**Time:** ~75 minutes

## 学习目标

- 使用 PyTorch 的 nn.Module、nn.Sequential 和 autograd 构建并训练神经网络
- 使用 PyTorch tensors、GPU acceleration 和标准训练循环（zero_grad、forward、loss、backward、step）
- 将你从零构建的 mini framework 组件转换为对应的 PyTorch 写法
- 在同一个任务上 profile 并比较纯 Python 框架和 PyTorch 的训练速度

## 问题

你已经有了一个能工作的迷你框架。Linear layers、ReLU、dropout、batch norm、Adam、DataLoader、训练循环。它可以用纯 Python 在圆形分类问题上训练一个 4 层网络。

但它在同一个问题上也比 PyTorch 慢 500 倍。

你的迷你框架用嵌套 Python loops 一次处理一个样本。PyTorch 会把同样的操作分发给优化过的 C++/CUDA kernels，并在 GPU 上运行。在单张 NVIDIA A100 上，PyTorch 可以在大约 6 小时内用 ImageNet（128 万张图片）训练一个 ResNet-50（2560 万参数）。你的框架在同一任务上大约需要 3000 小时，而且前提是它不会先耗尽内存。

速度不是唯一差距。你的框架没有 GPU 支持。没有自动微分，每个 module 的 backward() 都是你手写的。没有序列化。没有分布式训练。没有 mixed precision。也没有不用 print statements 就能调试 gradient flow 的办法。

PyTorch 填补了所有这些缺口。而且它保留了你已经构建过的同一套心智模型：Module、forward()、parameters()、backward()、optimizer.step()。概念一一迁移。语法几乎相同。区别在于，PyTorch 把十年的系统工程封装在你从零设计过的同一个接口后面。

## 概念

### 为什么 PyTorch 赢了

2015 年，TensorFlow 要求你先定义静态计算图，然后才能运行任何东西。你先构建图，编译它，再把数据喂进去。调试意味着盯着 graph visualizations。修改架构意味着从头重建计算图。

PyTorch 在 2017 年带着另一种哲学出现：eager execution。你写 Python。它立刻运行。`y = model(x)` 会现在就计算 y，而不是“往稍后才会计算 y 的图里添加一个节点”。这意味着标准 Python 调试工具可以工作。print() 可以工作。pdb 可以工作。forward pass 里的 if/else 可以工作。

到 2020 年，市场已经给出答案。PyTorch 在 ML 研究论文中的占比从 7%（2017）上升到 75% 以上（2022）。Meta、Google DeepMind、OpenAI、Anthropic 和 Hugging Face 都把 PyTorch 作为主要框架。TensorFlow 2.x 也采用 eager execution 作为回应，这等于默认承认 PyTorch 的设计是正确的。

这一课的启示是：developer experience 会复利增长。一个慢 10% 但调试快 50% 的框架，每次都会赢。

### Tensors

Tensor 是一个多维数组，具有三个关键属性：shape、dtype 和 device。

```python
import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list
```

**Shape** 是维度。标量的 shape 是 ()，向量是 (n,)，矩阵是 (m, n)，一批图片是 (batch, channels, height, width)。

**Dtype** 控制精度和内存。

| dtype | Bits | Range | Use case |
|-------|------|-------|----------|
| float32 | 32 | ~7 decimal digits | 默认训练 |
| float16 | 16 | ~3.3 decimal digits | Mixed precision |
| bfloat16 | 16 | Same range as float32, less precision | LLM 训练 |
| int8 | 8 | -128 to 127 | 量化推理 |

**Device** 决定计算发生在哪里。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

每个操作都要求所有 tensors 位于同一个 device。这是初学者最常遇到的 PyTorch 错误：`RuntimeError: Expected all tensors to be on the same device`。修复方式是在计算前把所有东西移动到同一个 device。

**Reshaping** 是常数时间操作，它改变的是元数据，不是数据。

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions
```

### Autograd

你的迷你框架要求你为每个 module 实现 backward()。PyTorch 不需要。它会把 tensors 上的每个操作记录到一个有向无环图中（computational graph），然后反向遍历这个图，自动计算梯度。

```mermaid
graph LR
    x["x (leaf)"] --> mul["*"]
    w["w (leaf, requires_grad)"] --> mul
    mul --> add["+"]
    b["b (leaf, requires_grad)"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w
```

与你的框架相比，关键区别是：PyTorch 使用 tape-based autodiff。前向传播期间，每个操作都会追加到一条“tape”上。调用 `.backward()` 会反向重放这条 tape。

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

Autograd 的三条规则：

1. 只有设置了 `requires_grad=True` 的 leaf tensors 会累积梯度
2. 梯度默认会累积，每次 backward pass 前调用 `optimizer.zero_grad()`
3. `torch.no_grad()` 会禁用梯度跟踪（评估时使用）

### nn.Module

`nn.Module` 是 PyTorch 中每个神经网络组件的基类。你已经在第 10 课构建过这个抽象。PyTorch 的版本还增加了自动参数注册、递归 module discovery、device management 和 state dict serialization。

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

当你在 `__init__` 中把 `nn.Module` 或 `nn.Parameter` 赋值为属性时，PyTorch 会自动注册它。`model.parameters()` 会递归收集每个注册参数。这就是为什么你不再需要像 mini framework 那样手动收集权重。

关键构建块：

| Module | What it does | Parameters |
|--------|-------------|------------|
| nn.Linear(in, out) | Wx + b | in*out + out |
| nn.Conv2d(in_ch, out_ch, k) | 2D convolution | in_ch*out_ch*k*k + out_ch |
| nn.BatchNorm1d(features) | Normalize activations | 2 * features |
| nn.Dropout(p) | Random zeroing | 0 |
| nn.ReLU() | max(0, x) | 0 |
| nn.GELU() | Gaussian error linear | 0 |
| nn.Embedding(vocab, dim) | Lookup table | vocab * dim |
| nn.LayerNorm(dim) | Per-sample normalization | 2 * dim |

### Loss Functions and Optimizers

PyTorch 提供了你构建过的所有东西的生产级版本。

**Loss functions**（来自 `torch.nn`）：

| Loss | Task | Input |
|------|------|-------|
| nn.MSELoss() | Regression | Any shape |
| nn.CrossEntropyLoss() | Multi-class classification | Logits (not softmax) |
| nn.BCEWithLogitsLoss() | Binary classification | Logits (not sigmoid) |
| nn.L1Loss() | Regression (robust) | Any shape |
| nn.CTCLoss() | Sequence alignment | Log probabilities |

注意：`CrossEntropyLoss` 内部组合了 `LogSoftmax` + `NLLLoss`。传入 raw logits，不要传 softmax outputs。这是一个常见错误，会悄悄产生错误梯度。

**Optimizers**（来自 `torch.optim`）：

| Optimizer | When to use | Typical LR |
|-----------|-------------|-----------|
| SGD(params, lr, momentum) | CNNs, well-tuned pipelines | 0.01--0.1 |
| Adam(params, lr) | Default starting point | 1e-3 |
| AdamW(params, lr, weight_decay) | Transformers, fine-tuning | 1e-4--1e-3 |
| LBFGS(params) | Small-scale, second-order | 1.0 |

### 训练循环

每个 PyTorch 训练循环都遵循同一个 5 步模式。你已经在第 10 课见过它。

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop Each Epoch
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end
```

标准模式：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

Batch loop 里面只有五行。训练 GPT-4、Stable Diffusion 和 LLaMA 的也是这五行。架构会变。数据会变。这五行不会变。

### Dataset and DataLoader

PyTorch 的 `Dataset` 是一个抽象类，有两个方法：`__len__` 和 `__getitem__`。`DataLoader` 会包裹它，提供 batching、shuffling 和 multi-process data loading。

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

`num_workers=4` 会启动 4 个进程并行加载数据，同时 GPU 在当前 batch 上训练。对于受磁盘限制的工作负载（大图像、音频），仅这一点就能让训练速度翻倍。

### GPU Training

把模型移动到 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

这会递归地把每个参数和 buffer 移动到 GPU。然后在训练中移动每个 batch：

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**Mixed precision** 会在现代 GPU（A100、H100、RTX 4090）上把内存用量减半，并将吞吐量提升约 2 倍：forward/backward 使用 float16，master weights 保持 float32。

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 对比：Mini Framework vs PyTorch vs JAX

| Feature | Mini Framework (L10) | PyTorch | JAX |
|---------|---------------------|---------|-----|
| Autodiff | Manual backward() | Tape-based autograd | Functional transforms |
| Execution | Eager (Python loops) | Eager (C++ kernels) | Traced + JIT compiled |
| GPU support | No | Yes (CUDA, ROCm, MPS) | Yes (CUDA, TPU) |
| Speed (MNIST MLP) | ~300s/epoch | ~0.5s/epoch | ~0.3s/epoch |
| Module system | Custom Module class | nn.Module | Stateless functions (Flax/Equinox) |
| Debugging | print() | print(), pdb, breakpoint() | Harder (JIT tracing breaks print) |
| Ecosystem | None | Hugging Face, Lightning, timm | Flax, Optax, Orbax |
| Learning curve | You built it | Moderate | Steep (functional paradigm) |
| Production use | Toy problems | Meta, OpenAI, Anthropic, HF | Google DeepMind, Midjourney |

```figure
dropout-mask
```

## Build It

一个只使用 PyTorch primitives 在 MNIST 上训练的 3 层 MLP。不用 high-level wrappers。不用 `torchvision.datasets`。我们自己下载并解析原始数据。

### Step 1：从原始文件加载 MNIST

MNIST 由 4 个 gzip 压缩文件组成：训练图像（60,000 x 28 x 28）、训练标签、测试图像（10,000 x 28 x 28）、测试标签。我们下载它们并解析二进制格式。

```python
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
```

### Step 2：定义模型

一个 3 层 MLP：784 -> 256 -> 128 -> 10。ReLU activations。Dropout 用于 regularization。为了保持简单，不使用 batch norm。

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
```

输出层会产生 10 个 raw logits（每个数字一个）。不要加 softmax，`CrossEntropyLoss` 会在内部处理。

参数量：784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146。按现代标准很小。GPT-2 small 有 124M。这可以在几秒内训练完。

### Step 3：训练循环

标准的 forward-loss-backward-step 模式。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
```

注意评估期间的 `torch.no_grad()`。它会禁用 autograd，减少内存使用并加速推理。如果没有它，PyTorch 会构建一个你永远不会使用的 computational graph。

### Step 4：把所有东西接起来

```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
```

10 个 epoch 后的预期输出：约 97.8% 测试准确率。CPU 训练时间：约 30 秒。GPU：约 5 秒。在你的 mini framework 中用同样架构训练：约 45 分钟。

## Use It

### 快速对比：Mini Framework vs PyTorch

| Mini Framework (Lesson 10) | PyTorch |
|---------------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` then `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| No GPU | `model.to("cuda")` |
| Manual backward for every module | Autograd handles everything |

接口几乎相同。差别都在底层。

### 保存和加载模型

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

始终保存 `state_dict()`（参数字典），不要保存 model object。保存 model object 会使用 pickle，一旦你重构代码就容易坏。State dicts 更可移植。

### 学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

PyTorch 内置 15+ 种 schedulers：StepLR、ExponentialLR、CosineAnnealingLR、OneCycleLR、ReduceLROnPlateau。它们都接入同一个 optimizer interface。

## Ship It

本课产出两个 artifacts：

- `outputs/prompt-pytorch-debugger.md` -- 一个诊断常见 PyTorch 训练失败的 prompt
- `outputs/skill-pytorch-patterns.md` -- 一个 PyTorch 训练模式的 skill reference

## 练习

1. **添加 batch normalization。** 在每个 linear layer 后插入 `nn.BatchNorm1d`（放在 activation 前）。比较它与只用 dropout 版本的测试准确率和训练速度。Batch norm 应该能用更少 epoch 达到 98%+。

2. **实现 learning rate finder。** 用指数增长的 learning rate（从 1e-7 到 1.0）训练一个 epoch。绘制 loss vs LR。最佳 LR 是 loss 开始上升之前的值。用它为 MNIST 模型选择更好的 LR。

3. **迁移到 GPU 并使用 mixed precision。** 在训练循环中添加 `torch.amp.autocast` 和 `GradScaler`。在 GPU 上分别测量使用和不使用 mixed precision 的吞吐量（samples/second）。在 A100 上，预期约 2 倍加速。

4. **构建 custom Dataset。** 下载 Fashion-MNIST（格式与 MNIST 相同，但内容是服装物品）。实现一个带有 `__getitem__` 和 `__len__` 的 `FashionMNISTDataset(Dataset)` 类。训练同一个 MLP 并比较准确率。Fashion-MNIST 更难，预期约 88%，而不是约 98%。

5. **把 Adam 替换成 SGD + momentum。** 使用 `SGD(params, lr=0.01, momentum=0.9)` 训练。比较收敛曲线。然后添加 `CosineAnnealingLR` scheduler，看看 SGD 是否能在第 10 个 epoch 前追上 Adam。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Tensor | “多维数组” | 一个有类型、知道 device、且每个操作都内置 automatic differentiation 支持的数组 |
| Autograd | “自动反向传播” | 一个 tape-based 系统，会在 forward pass 中记录操作，再反向重放以计算精确梯度 |
| nn.Module | “一层” | 任意可微计算块的基类：注册参数、支持嵌套、处理 train/eval modes |
| state_dict | “模型权重” | 一个从参数名映射到 tensors 的 OrderedDict，是训练后模型的可移植、可序列化表示 |
| .backward() | “计算梯度” | 反向遍历 computational graph，为每个 requires_grad=True 的 leaf tensor 计算并累积梯度 |
| .to(device) | “移动到 GPU” | 递归地把所有参数和 buffers 转移到指定 device（CPU、CUDA、MPS） |
| DataLoader | “数据管道” | 一个从 Dataset 中 batching、shuffling，并可选地并行加载数据的 iterator |
| Mixed precision | “使用 float16” | 用 float16 进行 forward/backward 来提速，同时保留 float32 master weights 以保证数值稳定 |
| Eager execution | “现在就运行” | 操作在被调用时立即执行，而不是延迟到之后的编译步骤；这是区分 PyTorch 和 TF 1.x 的核心设计选择 |
| zero_grad | “重置梯度” | 在下一次 backward pass 前把所有参数梯度设为 0，因为 PyTorch 默认会累积梯度 |

## 延伸阅读

- Paszke 等人，“PyTorch: An Imperative Style, High-Performance Deep Learning Library”（2019）-- 解释 PyTorch 设计取舍的原始论文
- PyTorch Tutorials：“Learning PyTorch with Examples”（https://pytorch.org/tutorials/beginner/pytorch_with_examples.html）-- 从 tensors 到 nn.Module 的官方路径
- PyTorch Performance Tuning Guide（https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html）-- mixed precision、DataLoader workers、pinned memory 和其他生产优化
- Horace He，“Making Deep Learning Go Brrrr”（https://horace.io/brrr_intro.html）-- 解释为什么 GPU training 快，并包含 PyTorch 相关优化策略
