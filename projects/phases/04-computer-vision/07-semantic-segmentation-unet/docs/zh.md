# 语义分割：U-Net

> 分割就是在每个像素上做分类。U-Net 通过把 downsampling encoder 和 upsampling decoder 配对，并在两者之间接上 skip connections，让这件事真正可行。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03（CNNs），Phase 4 Lesson 04（Image Classification）
**Time:** ~75 minutes

## 学习目标

- 区分 semantic、instance 和 panoptic segmentation，并为给定问题选择正确任务
- 在 PyTorch 中从零构建 U-Net，包括 encoder blocks、bottleneck、带 transposed convolutions 的 decoder，以及 skip connections
- 实现 pixel-wise cross-entropy、Dice loss，以及当前医学和工业分割任务默认使用的 combined loss
- 阅读每个 class 的 IoU 和 Dice metrics，并诊断低分来自 small-object recall、boundary accuracy 还是 class imbalance

## 问题

分类对每张图输出一个 label。检测对每张图输出若干 box。分割对每个像素输出一个 label。对大小为 `H x W` 的 input，output 是 shape 为 `H x W`（semantic）或 `H x W x N_instances`（instance）的 tensor。也就是说，每张图不是一个 prediction，而是几百万个 predictions。

分割的结构，解释了它为什么支撑几乎所有 dense-prediction 视觉产品：医学影像（tumour masks）、自动驾驶（road、lane、obstacle）、卫星图像（building footprints、crop boundaries）、文档解析（layout zones）、机器人（graspable regions）。这些任务都不能靠给 object 画一个 box 解决；它们需要精确轮廓。

架构问题说起来很简单，解决起来不简单：你需要网络同时看到图像的 global context（这是什么场景）和 local pixel detail（到底哪个像素是 road，哪个是 pavement）。标准 CNN 为了获得 context 会压缩空间维度，这会丢掉 detail。U-Net 是同时拿到两者的设计。

## 概念

### Semantic vs instance vs panoptic

```mermaid
flowchart LR
    IN["Input image"] --> SEM["Semantic<br/>(pixel → class)"]
    IN --> INS["Instance<br/>(pixel → object id,<br/>only foreground classes)"]
    IN --> PAN["Panoptic<br/>(every pixel → class + id)"]

    style SEM fill:#dbeafe,stroke:#2563eb
    style INS fill:#fef3c7,stroke:#d97706
    style PAN fill:#dcfce7,stroke:#16a34a
```

- **Semantic** 会说“这个像素是 road，那个像素是 car”。两辆挨在一起的车会合并成同一个 blob。
- **Instance** 会说“这个像素是 car #3，那个像素是 car #5”。它忽略 background stuff（“stuff” 指 sky、road、grass）。
- **Panoptic** 把两者统一起来：每个像素都有 class label，每个 instance 都有唯一 id，stuff 和 things 都被分割。

这一课覆盖 semantic。下一课（Mask R-CNN）覆盖 instance。

### U-Net 的形状

```mermaid
flowchart LR
    subgraph ENC["Encoder (contracting)"]
        E1["64<br/>H x W"] --> E2["128<br/>H/2 x W/2"]
        E2 --> E3["256<br/>H/4 x W/4"]
        E3 --> E4["512<br/>H/8 x W/8"]
    end
    subgraph BOT["Bottleneck"]
        B1["1024<br/>H/16 x W/16"]
    end
    subgraph DEC["Decoder (expanding)"]
        D4["512<br/>H/8 x W/8"] --> D3["256<br/>H/4 x W/4"]
        D3 --> D2["128<br/>H/2 x W/2"]
        D2 --> D1["64<br/>H x W"]
    end
    E4 --> B1 --> D4
    E1 -. skip .-> D1
    E2 -. skip .-> D2
    E3 -. skip .-> D3
    E4 -. skip .-> D4
    D1 --> OUT["1x1 conv<br/>classes"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style BOT fill:#fef3c7,stroke:#d97706
    style DEC fill:#dcfce7,stroke:#16a34a
```

Encoder 会把 spatial resolution 减半四次，同时把 channels 翻倍。Decoder 反过来：把 spatial resolution 翻倍四次，同时把 channels 减半。Skip connections 会在每个 resolution 上，把匹配的 encoder features 和 decoder features concatenate 起来。最后的 1x1 conv 在 full resolution 上把 `64 -> num_classes`。

为什么 skip connections 必不可少：当 decoder 尝试输出 pixel-level predictions 时，它只见过很小的 feature maps。没有 skips，它无法准确定位边缘，因为这些信息已经在 encoder 中被压缩掉了。Skip connections 把 encoder 下采样过程中计算出的 high-resolution feature maps 交给 decoder。

### Transposed vs bilinear upsample

Decoder 必须扩展 spatial dimensions。有两个选择：

- **Transposed convolution**（`nn.ConvTranspose2d`）：learnable upsample。历史上的 U-Net 默认选择。如果 stride 和 kernel size 不能整除，可能产生 checkerboard artifacts。
- **Bilinear upsample + 3x3 conv**：先平滑 upsample，再接 conv。Artifacts 更少，参数更少，现在是现代默认选择。

两者在真实项目里都会出现。对第一个 U-Net 来说，bilinear 更稳。

### Pixel grid 上的 cross-entropy

对 C 类 semantic segmentation，模型 output 是 `(N, C, H, W)`。Target 是 `(N, H, W)`，里面是 integer class IDs。Cross-entropy 和分类场景完全一样，只是应用到每个空间位置：

```text
Loss = mean over (n, h, w) of -log( softmax(logits[n, :, h, w])[target[n, h, w]] )
```

PyTorch 的 `F.cross_entropy` 原生支持这个 shape。不需要 reshape。

### Dice loss 以及为什么需要它

Cross-entropy 平等对待每个像素。当一个 class 主导画面时，这就是错的（医学影像：99% background，1% tumour）。网络可以通过全部预测 background 拿到 99% accuracy，但依然毫无用处。

Dice loss 直接优化 predicted mask 和 true mask 的 overlap：

```text
Dice(p, y) = 2 * sum(p * y) / (sum(p) + sum(y) + epsilon)
Dice_loss = 1 - Dice
```

其中 `p` 是某个 class 的 sigmoid/softmax probability map，`y` 是 binary ground-truth mask。只有 overlap 完美时，loss 才是 0。因为它基于 ratio，class imbalance 不再重要。

实践中使用 **combined loss**：

```text
L = L_cross_entropy + lambda * L_dice       (lambda ~ 1)
```

Cross-entropy 在训练早期提供稳定梯度；Dice 在训练后段专注于真正匹配 mask shape。这个组合是医学影像默认选择，在任何 class-imbalanced dataset 上都很难被击败。

### Evaluation metrics

- **Pixel accuracy**：预测正确的像素比例。便宜。但在 imbalanced data 上坏掉的原因和分类 accuracy 一样。
- **IoU per class**：每个 class mask 的 intersection over union；跨 classes 平均就是 mIoU。
- **Dice（pixels 上的 F1）**：类似 IoU；`Dice = 2 * IoU / (1 + IoU)`。医学影像偏好 Dice，自动驾驶社区偏好 IoU；它们是单调相关的。
- **Boundary F1**：衡量 predicted boundaries 和 ground-truth boundaries 的接近程度，会惩罚很小的偏移。对半导体检测这类高精度任务很重要。

报告每个 class 的 IoU，而不只是 mIoU。Mean IoU 会隐藏一个只有 15% 的 class，尤其当其他九个 class 都在 85% 时。

### Input resolution trade-off

U-Net 的 encoder 会把 resolution 减半四次，所以 input 必须能被 16 整除。医学图像通常是 512x512 或 1024x1024。自动驾驶 crop 通常是 2048x1024。U-Net 的 memory cost 随 `H * W * C_max` 缩放；在 1024x1024 且 bottleneck channels 为 1024 时，一次 forward pass 已经会用掉数 GB VRAM。

两个标准 workaround：

1. Tile the input：处理带 overlap 的 256x256 tiles，然后 stitch 回去。
2. 用 dilated convolutions 替换 bottleneck，让 spatial resolution 保持更高，同时扩大 receptive field（DeepLab family）。

对第一个模型来说，256x256 input 加 64-channel-base U-Net 可以在 8 GB VRAM 上舒服训练。

## Build It

### Step 1: Encoder block

两个 3x3 convs，配 batch norm 和 ReLU。第一个 conv 改变 channel count；第二个保持不变。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

这个 block 会被全网复用。`bias=False` 是因为 BN 的 beta 会处理 bias。

### Step 2: Down 和 up blocks

```python
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
```

只检查 spatial shape（`shape[-2:]`）可以处理尺寸不能被 16 整除的 input；安全的 `F.interpolate` 会在 concat 前对齐 tensor。比较完整 shape 还会因为 channel-count differences 触发，这种差异应该是响亮的错误，而不是静默 interpolate。

### Step 3: U-Net

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16 + base * 8, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u3 = Up(base * 4 + base * 2, base * 2)
        self.u4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)

net = UNet(in_channels=3, num_classes=2, base=32)
x = torch.randn(1, 3, 256, 256)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

Output shape 是 `(1, 2, 256, 256)`，spatial size 和 input 相同，channel 数等于 `num_classes`。`base=32` 时大约 7.7M parameters。

### Step 4: Losses

```python
def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, num_classes, lam=1.0):
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce.item(), "dice": dc.item()}
```

Dice 会先按 class 计算再平均（macro Dice）。`eps` 防止 batch 中缺失某个 class 时除以零。

### Step 5: IoU metric

```python
@torch.no_grad()
def iou_per_class(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious[c] = (inter / union) if union > 0 else torch.tensor(float("nan"))
    return ious
```

返回长度为 C 的 vector。`nan` 标记 batch 中不存在的 classes；计算 mIoU 时不要把这些 class 平均进去。

### Step 6: 用于端到端验证的 synthetic dataset

在彩色背景上生成 shapes，让网络必须学习 shape，而不是 pixel colour。

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader

def synthetic_segmentation(num_samples=200, size=64, seed=0):
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    for i in range(num_samples):
        bg = rng.uniform(0, 1, (3,))
        images[i] = bg
        masks[i] = 0
        num_shapes = rng.integers(1, 4)
        for _ in range(num_shapes):
            cls = int(rng.integers(1, 3))
            color = rng.uniform(0, 1, (3,))
            cx, cy = rng.integers(10, size - 10, size=2)
            r = int(rng.integers(4, 12))
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            if cls == 1:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            else:
                mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            images[i][mask] = color
            masks[i][mask] = cls
        images[i] += rng.normal(0, 0.02, images[i].shape)
        images[i] = np.clip(images[i], 0, 1)
    return images, masks


class SegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[i]).long()
        return img, mask
```

三个 classes：background (0)、circles (1)、squares (2)。网络必须学会区分 shape。

### Step 7: Training loop

```python
def train_one_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    loss_sum, total = 0.0, 0
    iou_sum = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, _ = combined_loss(logits, y, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        iou_sum += iou_per_class(logits, y, num_classes).nan_to_num(0)
    return loss_sum / total, iou_sum / len(loader)
```

在 synthetic dataset 上跑 10-30 epochs，观察 shape classes 的 mIoU 爬到 0.9 以上。注意，`nan_to_num(0)` 会把 batch 中不存在的 classes 当作 0；为了得到准确的 per-class IoU，evaluation 时应该按 presence 做 mask，并用 `torch.nanmean` 跨 batches 平均，而不是像这里这样直接平均。

## Use It

在 production 中，`segmentation_models_pytorch`（“smp”）会用任意 torchvision 或 timm backbone 封装所有标准 segmentation architectures。三行即可：

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)
```

做真实工作时还值得知道：

- **DeepLabV3+** 用 dilated convs 替换 max-pool-based downsampling，让 bottleneck 保持 resolution；在 satellite 和 driving data 上边界更快更准。
- **SegFormer** 把 conv encoder 换成 hierarchical transformer；在很多 benchmarks 上是当前 SOTA。
- **Mask2Former** / **OneFormer** 在单一 architecture 中统一 semantic、instance 和 panoptic segmentation。

这三者在 `smp` 或 `transformers` 中都可以作为 drop-in replacements，使用相同 data loader。

## Ship It

这一课产出：

- `outputs/prompt-segmentation-task-picker.md`：一个 prompt，为给定任务在 semantic、instance 和 panoptic segmentation 之间做选择，并指出适合的 architecture。
- `outputs/skill-segmentation-mask-inspector.md`：一个 skill，报告 class distribution、predicted-mask statistics，以及 under-predicted 或 boundary-blurred 的 classes。

## 练习

1. **（Easy）** 为 binary segmentation task（foreground vs background）实现 `bce_dice_loss`。在 synthetic two-class dataset 上验证：当 foreground 只占 5% pixels 时，combined loss 比单独 BCE 收敛更快。
2. **（Medium）** 把 `nn.Upsample + conv` up-block 换成 `nn.ConvTranspose2d` up-block。在 synthetic dataset 上训练两者并比较 mIoU。观察 transposed-conv 版本的 checkerboard artifacts 出现在哪里。
3. **（Hard）** 选择一个真实 segmentation dataset（Oxford-IIIT Pets、Cityscapes mini split 或一个医学子集），训练 U-Net，让它达到 `smp.Unet` reference 2 个 IoU points 以内。报告 per-class IoU，并识别哪些 classes 从加入 Dice loss 中受益最多。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Semantic segmentation | “Label every pixel” | 对每个像素做 C 类分类；同一 class 的 instances 会合并 |
| Instance segmentation | “Label every object” | 分离同一 class 的不同 instances；只处理 foreground |
| Panoptic segmentation | “Semantic + instance” | 每个像素都有 class；每个 thing instance 也有唯一 id |
| Skip connection | “U-Net bridge” | 把 encoder features concatenate 到匹配 resolution 的 decoder features 中；保留 high-frequency detail |
| Transposed conv | “Deconvolution” | Learnable upsampling；可能产生 checkerboard artifacts |
| Dice loss | “Overlap loss” | `1 - 2|A ∩ B| / (|A| + |B|)`；直接优化 mask overlap，并且对 class imbalance 鲁棒 |
| mIoU | “Mean intersection over union” | 跨 classes 平均的 IoU；segmentation 的社区标准 metric |
| Boundary F1 | “Boundary accuracy” | 只在 boundary pixels 上计算的 F1 score；对精度要求高的任务很重要 |

## 进一步阅读

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)：原始 paper；大家都在复制的图在第 2 页
- [Fully Convolutional Networks (Long et al., 2015)](https://arxiv.org/abs/1411.4038)：第一篇让 segmentation 成为端到端 conv problem 的 paper
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)：production segmentation 的 reference；包含所有标准 architectures 和所有标准 losses
- [Lessons learned from training SOTA segmentation (kaggle.com competitions)](https://www.kaggle.com/code/iafoss/carvana-unet-pytorch)：讲解真实数据上 TTA、pseudo-labeling 和 class weights 为什么重要
