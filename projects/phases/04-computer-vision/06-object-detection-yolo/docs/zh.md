# 目标检测：从零实现 YOLO

> 检测就是在 feature map 的每个位置同时做分类和回归，然后用非极大值抑制清理重复框。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03（CNNs），Phase 4 Lesson 04（Image Classification），Phase 4 Lesson 05（Transfer Learning）
**Time:** ~75 minutes

## 学习目标

- 解释 grid-and-anchor 设计如何把检测变成 dense prediction 问题，并说清 output tensor 中每个数字的含义
- 计算两个 box 之间的 Intersection-over-Union，并从零实现 non-maximum suppression
- 在 pretrained backbone 顶部构建一个最小 YOLO-style head，包括 classification、objectness 和 box-regression losses
- 阅读 detection metric row（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95），并判断下一步应该调哪个 knob

## 问题

分类会说：“这张图里是一只狗。”检测会说：“像素 `(112, 40, 280, 210)` 处有一只狗，`(400, 180, 560, 310)` 处有一只猫，画面里没有别的东西。”这个结构变化，也就是从每张图一个 label 变成可变数量的带标签 box，正是每个自动驾驶系统、每个安防产品、每个文档版面解析器和每条工厂视觉产线依赖的能力。

检测也是视觉领域所有工程权衡同时出现的地方。你希望 box 足够准确（regression head），每个 box 的类别正确（classification head），模型知道什么时候没有东西需要检测（objectness score），并且每个真实物体只对应一个 prediction（non-maximum suppression）。任何一项没做好，pipeline 要么漏检物体，要么报告幻觉框，要么用十五个位置略有不同的框反复预测同一个物体。

YOLO（You Only Look Once，Redmon et al. 2016）的设计让这一切能通过 conv net 的一次 forward pass 实时运行，而这些结构决策至今仍是现代 detector（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）的骨架。理解核心之后，每个变体都只是同一组部件的重新排列。

## 概念

### 把检测看成 dense prediction

分类器对每张图输出 C 个数字。YOLO-style detector 对每张图输出 `(S x S x (5 + C))` 个数字，其中 S 是 spatial grid size。

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

每个 `S * S` grid cell 会预测 `B` 个 box。对每个 box：

- 4 个数字描述几何：`tx, ty, tw, th`。
- 1 个数字是 objectness score：“这个 cell 中心附近是否有物体？”
- C 个数字是 class probabilities。

每个 cell 总计：`B * (5 + C)`。对 VOC 来说，如果 `S=13, B=2, C=20`，每个 cell 就是 50 个数字。

### 为什么需要 grids 和 anchors

朴素回归会为每个 object 预测绝对坐标 `(x, y, w, h)`。这对 conv network 很难，因为平移图像时，不应该让所有 prediction 都按同样幅度平移；每个 object 都是被空间位置锚定的。grid 的回答是：把每个 ground-truth box 分配给其中心落入的 grid cell；只有这个 cell 对该 object 负责。

Anchor 解决第二个问题。一个 3x3 conv 很难从 16-pixel receptive field 的 feature cell 中直接回归出一个 500 像素宽的 box。所以我们在每个 cell 预定义 `B` 个 prior box shape（anchors），再从每个 anchor 预测小的 delta。模型学的是选择合适 anchor 并轻微调整它，而不是从零回归。

```text
Anchor box priors（416x416 input 示例）：

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

在每个 grid cell，每个 anchor 都发出 (tx, ty, tw, th, obj, c_1, ..., c_C)。
```

现代 detector 常常使用 FPN，在不同 resolution 上使用不同 anchor set：浅层高分辨率 map 用 small anchors，深层低分辨率 map 用 large anchors。想法相同，只是尺度更多。

### 解码 predictions

原始的 `tx, ty, tw, th` 不是 box 坐标；它们是 regression targets，需要先转换再绘制：

```text
centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)
```

`sigmoid` 会把 centre offsets 限制在 cell 内。`exp` 让 width 可以相对 anchor 自由缩放，且不会发生符号翻转。`stride` 把 grid coordinates 缩放回像素。自 YOLOv2 以来，每个 YOLO 版本都使用这种 decode step。

### IoU

检测中两个 box 之间的通用相似度指标：

```text
IoU(A, B) = area(A intersect B) / area(A union B)
```

IoU = 1 表示完全相同；IoU = 0 表示没有重叠。Prediction 和 ground-truth box 之间的 IoU 决定一个 prediction 是否算 true positive（通常 IoU >= 0.5）。两个 prediction 之间的 IoU 是 NMS 用来去重的依据。

### Non-maximum suppression

在相邻 anchor 上训练的 conv network，常常会为同一个 object 预测多个重叠 box。NMS 保留最高 confidence 的 prediction，并删除所有与它 IoU 高于阈值的其他 prediction。

```text
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

目标检测的典型 threshold 是 0.45。近年的 detector 会用 `soft-NMS`、`DIoU-NMS`，或者直接学习 suppression（RT-DETR）来替代标准 NMS，但它们的结构目的相同。

### Loss

YOLO loss 是三个 loss 加权相加：

```text
L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)
```

只有包含 object 的 cell 会贡献 box-regression 和 classification losses。不含 object 的 cell 只贡献 objectness loss（教模型保持沉默）。`lambda_noobj` 通常很小（约 0.5），因为绝大多数 cell 都是空的，否则它们会主导 total loss。

现代变体会把 MSE box loss 换成 CIoU / DIoU（直接优化 IoU），用 focal loss 处理 class imbalance，并用 quality focal loss 平衡 objectness。三组件结构不变。

### Detection metrics

Accuracy 不能直接迁移到检测。能用的是这四个数字：

- **Precision@IoU=0.5**：被计为 positive 的 predictions 中，有多少是真的正确。
- **Recall@IoU=0.5**：真实 object 中，我们找到了多少。
- **AP@0.5**：IoU threshold 为 0.5 时 precision-recall curve 的面积；每个 class 一个数字。
- **mAP@0.5:0.95**：在 IoU thresholds 0.5、0.55、...、0.95 上求 AP 平均。COCO metric；最严格，也最有信息量。

四个都要报告。一个 detector 如果 mAP@0.5 很强但 mAP@0.5:0.95 很弱，说明定位大致正确但不够紧；用更好的 box-regression loss 修。一个 detector 如果 precision 高、recall 低，说明太保守；降低 confidence threshold，或者提高 objectness weight。

## Build It

### Step 1: IoU

整节课的 workhorse。它作用在两个 `(x1, y1, x2, y2)` 格式的 box array 上。

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回值是一个 `(N_a, N_b)` 的 pairwise IoU matrix。要和单个 ground-truth box 比较时，把其中一个 array 做成 shape `(1, 4)` 即可。

### Step 2: Non-max suppression

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

它是 deterministic 的，排序带来 `O(N log N)`，在相同 input 上会匹配 `torchvision.ops.nms` 的行为。

### Step 3: Box encoding 和 decoding

在 pixel coordinates 和网络实际回归的 `(tx, ty, tw, th)` targets 之间转换。

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试：先 encode 一个 box，再 decode。你应该能拿回一个非常接近原始 box 的结果（如果 `tx` 不在 post-sigmoid range 内，sigmoid inverse 不会完美可逆，所以会有微小差异）。

### Step 4: 一个最小 YOLO head

在 feature map 上做一个 1x1 conv，然后 reshape 成 `(B, S, S, num_anchors, 5 + C)`。

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

Output shape 是 `(N, H, W, num_anchors, 5 + C)`。最后一维保存 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### Step 5: Ground-truth assignment

对每个 ground-truth box，决定由哪个 `(cell, anchor)` 负责。

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

Anchor selection 是“和 ground truth 形状 IoU 最高的 anchor”——这是一个廉价 proxy，和 YOLOv2/v3 的 assignment 匹配。v5 及之后的版本使用更复杂的策略（task-aligned matching、dynamic k），但只是对同一想法的细化。

### Step 6: 三个 losses

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

每篇 YOLO tutorial 要么 hardcode、要么 sweep 的五个 hyper-parameters。比例很重要：`lambda_coord=5, lambda_noobj=0.5` 呼应原始 YOLOv1 paper，并且仍然是合理的默认值。

### Step 7: Inference pipeline

Decode raw head output，应用 sigmoid/exp，根据 objectness 设 threshold，然后做 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这就是完整 eval path：head -> decode -> threshold -> NMS。

## Use It

`torchvision.models.detection` 提供了 production detector，它们有相同的概念结构。加载 pretrained model 只需要三行。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对 real-time inference pipeline 来说，`ultralytics`（YOLOv8/v9）是标准选择：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。模型会在内部处理 decoding 和 NMS，并返回与你上面构建的相同 `boxes / scores / labels` triple。

## Ship It

这一课产出：

- `outputs/prompt-detection-metric-reader.md`：一个 prompt，把一行 `precision, recall, AP, mAP@0.5:0.95` 转成一句诊断，以及最有用的下一个实验。
- `outputs/skill-anchor-designer.md`：一个 skill，给定 ground-truth boxes 数据集后，对 `(w, h)` 运行 k-means，并返回每个 FPN level 的 anchor sets 以及你选择 anchor 数量所需的 coverage statistics。

## 练习

1. **（Easy）** 实现 `box_iou`，并在 1,000 对 random boxes 上与 `torchvision.ops.box_iou` 比较。验证 max absolute difference 小于 `1e-6`。
2. **（Medium）** 把 `yolo_loss` 改成使用 `CIoU` box loss，而不是 MSE。用一个 100-image synthetic dataset 展示：在相同 epoch 数下，CIoU 会比 MSE 收敛到更好的最终 mAP@0.5:0.95。
3. **（Hard）** 实现 multi-scale inference：把同一张图用三种 resolution 输入模型，合并 box predictions，最后运行一次 NMS。在 held-out set 上测量相对 single-scale inference 的 mAP lift。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Anchor | “Box prior” | 每个 grid cell 中预定义的 box shape；网络从它预测 delta，而不是预测绝对坐标 |
| IoU | “Overlap” | 两个 box 的 intersection-over-union；检测中的通用相似度指标 |
| NMS | “Deduplicate” | 贪心算法，保留最高分 predictions，并删除重叠超过阈值的其他 predictions |
| Objectness | “Is there something here” | 每个 anchor、每个 cell 的 scalar，预测是否有物体中心落在该 cell |
| Grid stride | “Downsample factor” | 每个 grid cell 对应的像素数；416-px input 配 13-grid head 时 stride 为 32 |
| mAP | “Mean average precision” | precision-recall curve 下方面积的平均值，先按 class 平均；COCO 还会按 IoU threshold 平均 |
| AP@0.5 | “PASCAL VOC AP” | IoU threshold 为 0.5 的 average precision；较宽松的指标版本 |
| mAP@0.5:0.95 | “COCO AP” | 在 IoU thresholds 0.5..0.95、步长 0.05 上求平均；更严格，也是当前社区标准 |

## 进一步阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — 开创性 paper；之后的每个 YOLO 都是对这个结构的细化
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) — 引入 multi-scale FPN-style heads 的 paper；图仍然最清晰
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com) — 当前 production reference；覆盖 dataset formats、augmentations、training recipes
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) — 对完整 detector zoo 最好的 plain-English 导览；理解 DETR、RetinaNet、FCOS 和 YOLO 之间关系时很有价值
