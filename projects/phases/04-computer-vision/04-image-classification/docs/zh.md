# 图像分类

> 分类器就是一个从像素到类别概率分布的函数。其他一切都是管线工程。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lesson 09（Model Evaluation），Phase 3 Lesson 10（Mini Framework），Phase 4 Lesson 03（CNNs）
**Time:** ~75 minutes

## 学习目标

- 在 CIFAR-10 上构建端到端图像分类流水线：dataset、augmentation、model、training loop、evaluation
- 解释 dataloader、loss、optimizer、scheduler、augmentation 各自的作用，并预测其中任意一个损坏时会如何体现在 loss curve 上
- 从零实现 mixup、cutout 和 label smoothing，并说明什么时候值得加入它们
- 阅读 confusion matrix 和 per-class precision/recall 表，在总体 accuracy 之外诊断数据集和模型失败

## 问题

几乎每个落地的视觉任务，在某个层面都会归约为图像分类。Detection 会分类 region。Segmentation 会分类 pixel。Retrieval 会按和 class centroid 的相似度排序。把分类做好，也就是把 dataset loop、augmentation policy、loss 和 evaluation 做对，是会迁移到本阶段每个其他任务上的能力。

大多数分类 bug 不在模型里，而在流水线里：坏掉的 normalization，没有 shuffle 的训练集，会扭曲 label 的 augmentation，被训练数据污染的 validation split，或者 epoch 30 之后悄悄发散的 learning rate。一个正确配置下能在 CIFAR-10 上达到 93% 的 CNN，常常会因为坏掉的流水线只得到 70-75%，而且 loss curve 全程看起来还很合理。

这一课会手工接好整条流水线，让每个部分都可检查。你不会使用 `torchvision.datasets` 里任何可能藏住 bug 的东西。

## 概念

### 分类流水线

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个 loop 里的每条边都可能藏着 bug。Cross-entropy 接收的是 raw logits，不是 softmax output，所以在 loss 之前写任何 `model(x).softmax()`，都会悄悄算出错误梯度。Augmentation 只作用于 input，不作用于 label，除了 mixup，因为它会同时混合两者。`optimizer.zero_grad()` 必须每一步执行一次；跳过它会累积梯度，看起来像一个极不稳定的 learning rate。这些 bug 都不会报错，只会把学习曲线压平。

### Cross-entropy、logits 和 softmax

分类器会为每张图像产生 `C` 个数字，叫做 logits。应用 softmax 会把它们转换成概率分布：

```text
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Cross-entropy 衡量正确类别的 negative log probability：

```text
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

右边的形式是数值稳定版本，也就是 log-sum-exp。PyTorch 的 `nn.CrossEntropyLoss` 会把 softmax 和 NLL 融合到一个 op 中，并直接接收 raw logits。自己先应用 softmax 几乎总是 bug，因为你会计算 `log(softmax(softmax(z)))`，这是没有意义的量。

### 为什么 augmentation 有效

CNN 对平移有 inductive bias，这来自权重共享；但它没有内置对 crop、flip、colour jitter 或 occlusion 的不变性。教会它这些不变性的唯一方法，是让它看到能体现这些变化的像素。训练时的每个随机变换都在表达：“这两张图像有相同 label；请学习能忽略差异的特征。”

| 变换 | 含义 |
|------|------|
| Original crop | “dog facing left” |
| Flip | “dog facing right”，同一 label，不同像素 |
| Rotate(+15) | “dog, slight tilt” |
| Colour jitter | “dog in warmer light” |
| RandomErasing | “dog with patch missing” |

规则：augmentation 必须保持 label。Cutout 和 rotation 用在数字上可能把 “6” 变成 “9”；对这种数据集，你要使用更小的 rotation range，并选择符合数字不变性的 augmentation。

### Mixup 和 cutmix

普通 augmentation 会变换像素，但保持 one-hot label 不变。**Mixup** 和 **cutmix** 会打破这个规则，因为它们会同时插值 input 和 label。

```text
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

它有效的原因是：模型不再记忆尖锐的 one-hot target，而是学习在类别之间插值。Training loss 会上升，test accuracy 会上升。对任何分类器来说，这是最便宜的 robustness 升级之一。

### Label smoothing

Label smoothing 是 mixup 的近亲。与其训练 `[0, 0, 1, 0, 0]`，不如训练 `[eps/C, eps/C, 1-eps, eps/C, eps/C]`，其中 `eps` 是 0.1 这类小值。它阻止模型产生过度尖锐的 logits，并以几乎零成本改善 calibration。PyTorch 1.10 起，`nn.CrossEntropyLoss(label_smoothing=0.1)` 已经内置了它。

### Accuracy 之外的评估

总体 accuracy 会隐藏类别不平衡。一个 90-10 的二分类器，如果永远预测多数类，也能得到 90%。真正告诉你发生了什么的工具是：

- **Per-class accuracy**：每个类别一个数字，会立刻暴露表现差的类别。
- **Confusion matrix**：C x C 网格，row i col j 表示真实类别 i 被预测成类别 j 的数量；对角线是正确，非对角线才是模型错误生活的地方。
- **Top-1 / Top-5**：正确类别是否在最高 1 个或最高 5 个预测里；Top-5 对 ImageNet 很重要，因为 “Norwich terrier” 和 “Norfolk terrier” 这种类别确实有歧义。
- **Calibration (ECE)**：一个 0.8 confidence 的预测，是否真的有 80% 概率正确？现代网络系统性过度自信；temperature scaling 或 label smoothing 可以修复它。

```figure
receptive-field
```

## Build It

### Step 1: 一个确定性的合成数据集

CIFAR-10 存在磁盘上。为了让这一课可复现且快速，我们构建一个看起来像 CIFAR 的合成数据集：32x32 RGB 图像，带有模型必须学习的 class-specific structure。完全相同的流水线可以不改动地用于真实 CIFAR-10。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类别都有自己的颜色 palette 和 frequency pattern，再加 Gaussian noise，迫使模型学习信号，而不是记忆像素。十个类别，每类一千张图像，然后打乱。

### Step 2: Normalisation 和 augmentation

每条视觉流水线都有这两类 transform。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

Crop 之前用 reflect-pad，而不是 zero-pad，因为黑边是一种信号，模型会学会以一种无用方式忽略它。

### Step 3: Mixup

Mixup 会在训练步骤内部混合两张图像和两个 label。这里把它实现成 batch transform，让它靠近 forward pass，而不是藏在 dataset 里。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是针对 soft-label distribution 的 cross-entropy。当 target 正好是 one-hot 时，它会退化成普通 cross-entropy。

### Step 4: Training loop

完整配方：遍历一次数据，每个 batch 做一次梯度，每个 epoch step 一次 scheduler。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次写 training loop，都检查这五个不变量：

1. 训练前 `model.train()`，评估前 `model.eval()`，因为它们会切换 dropout 和 batchnorm 行为。
2. `.zero_grad()` 在 `.backward()` 之前执行。
3. 累积 metrics 时使用 `.item()`，避免保留 computation graph。
4. Evaluation 期间使用 `@torch.no_grad()`，节省内存和时间，也防止微妙事故。
5. 对 raw logits 做 argmax，不对 softmax 做 argmax；结果相同，但少一个 op。

### Step 5: 拼起来

使用上一课的 `TinyResNet`，训练几个 epoch，然后评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，它会在五个 epoch 内接近完美 validation accuracy。这正是重点：流水线是正确的，模型能学到可学习的东西。把 dataset 换成真实 CIFAR-10，同一套 loop 不用修改也能训练到约 90%。

### Step 6: 读取 confusion matrix

Accuracy 从来不会告诉你模型具体在哪里失败。Confusion matrix 会。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类别，列是预测类别。如果类别 3 和 5 之间出现一团非对角计数，说明模型混淆这两个类别，也给了你一个起点：收集有针对性的数据，或者设计 class-specific augmentation。

## Use It

`torchvision` 会把上面的所有东西封装成惯用组件。对真实 CIFAR-10，完整流水线就是四行加一个 training loop。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

注意两件事：mean/std 是 **dataset-specific**，是在 CIFAR-10 训练集上计算的，不是 ImageNet；reflect pad 是社区默认的 crop policy。把 ImageNet stats 复制到这里，会悄悄漏掉大约 1% accuracy，直到有人 profile 模型时才会发现。

## Ship It

这一课产出：

- `outputs/prompt-classifier-pipeline-auditor.md`：一个提示词；审计训练脚本是否违反上面的五个不变量，并指出第一个违规点。
- `outputs/skill-classification-diagnostics.md`：一个技能；给定 confusion matrix 和 class names 后，总结 per-class failure，并提出影响最大的单个修复。

## 练习

1. **（Easy）** 在合成数据集上，用同一个模型分别开启和关闭 mixup 训练五个 epoch。画出两者的 train loss 和 val loss。解释为什么 mixup 的 train loss 更高，但 val accuracy 相近或更好。
2. **（Medium）** 实现 Cutout，也就是在每张训练图像中随机置零一个 8x8 方块，并做 ablation：no augmentation、hflip+crop、hflip+crop+cutout、hflip+crop+mixup。报告每组 val accuracy。
3. **（Hard）** 构建 CIFAR-100 流水线（100 类，相同输入尺寸），复现一个 ResNet-34 训练 run，使结果在 published accuracy 的 1% 内。加分项：扫三个 learning rate 和两个 weight decay，记录到本地 CSV，并生成最终 confusion-matrix-top-confusions 表。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Logits | “Raw outputs” | 每张图像的 pre-softmax C 维向量；cross-entropy 期望这些值，而不是 softmax 后的值 |
| Cross-entropy | “The loss” | 正确类别的 negative log-probability；在一个稳定 op 中组合 log-softmax 和 NLL |
| DataLoader | “The batcher” | 给 dataset 加上 shuffling、batching 和可选 multi-worker loading；一半训练 bug 都会先怪到它头上 |
| Augmentation | “Random transforms” | 训练时任何保持 label 的像素级变换；教会 CNN 它原生没有的不变性 |
| Mixup / Cutmix | “Mix two images” | 同时混合 input 和 label，让分类器学习平滑插值，而不是硬边界 |
| Label smoothing | “Softer targets” | 把 one-hot 替换成 `(1-eps, eps/(C-1), ...)`；改善 calibration，并稍微提升 accuracy |
| Top-k accuracy | “Top-5” | 正确类别位于概率最高的 k 个预测中；用于有真实歧义类别的数据集 |
| Confusion matrix | “Where errors live” | C x C 表，entry (i, j) 统计真实类别 i 被预测成 j 的图像数；对角线正确，非对角线告诉你该修什么 |

## 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/)：单页讲清训练流水线的经典材料
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187)：许多小技巧叠加起来，给 ImageNet 上的 ResNet 加 3-4% accuracy
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412)：mixup 原始论文，三页理论加上很有说服力的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)：证明现代网络 miscalibrated，并用一个标量参数修复它的论文
