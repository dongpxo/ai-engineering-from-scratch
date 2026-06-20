# 单目深度与几何估计

> Depth map 是一张 single-channel image，其中每个 pixel 都表示它到 camera 的距离。过去只从一帧 RGB 中预测 depth，如果没有 stereo 或 LiDAR，几乎不可能。到 2026 年，一个 frozen ViT encoder 加 lightweight head，已经能接近 ground truth 的几个百分点以内。

**Type:** Build + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14（ViT），Phase 4 Lesson 17（Self-Supervised Vision），Phase 4 Lesson 07（U-Net）
**Time:** ~60 minutes

## 学习目标

- 区分 relative depth 和 metric depth，并说明每个 production model（MiDaS、Marigold、Depth Anything V3、ZoeDepth）解决的是哪一种
- 使用 Depth Anything V3（DINOv2 backbone）对任意 single images 预测 depth，无需 calibration
- 解释为什么单张 image 也能产生 depth（perspective cues、texture gradients、learned priors），以及它无法恢复什么（absolute scale、occluded geometry）
- 使用 depth map 和 pinhole camera intrinsics，把 2D detections lift 到 3D points

## 问题

Depth 是 2D computer vision 中缺失的轴。给定 RGB，你知道物体出现在 image plane 的哪里；但不知道它们离你有多远。Depth sensors（stereo rigs、LiDAR、time-of-flight）能直接解决这个问题，但它们昂贵、脆弱，而且 range 有限制。

Monocular depth estimation，也就是从单张 RGB frame 预测 depth，过去只能产生模糊且不可靠的输出。到 2026 年，大型 pretrained encoders 改变了这一点：Depth Anything V3 使用 frozen DINOv2 backbone，并能生成跨 indoor、outdoor、medical 和 satellite domains 泛化的 depth maps。Marigold 把 depth 重写成 conditional diffusion problem。ZoeDepth 则回归真正的 metric distances。

Depth 也是 2D detection 和 3D understanding 之间的桥梁：把 detected box 的 pixels 乘以 depth，就能把 2D object lift 成 3D point cloud。这是每个 AR occlusion system、每个 obstacle-avoidance pipeline，以及每个“pick up the cup”机器人的核心。

## 概念

### Relative vs metric depth

- **Relative depth**：没有真实世界单位的有序 `z` values。“Pixel A 比 pixel B 更近，但距离比例没有锚定到 metres。”
- **Metric depth**：从 camera 出发的绝对距离，单位是 metres。要求模型学到 image cues 和真实距离之间的统计关系。

MiDaS 和 Depth Anything V3 输出 relative depth。Marigold 输出 relative depth。ZoeDepth、UniDepth 和 Metric3D 输出 metric depth。Metric models 对 camera intrinsics 敏感；relative models 不敏感。

### Encoder-decoder pattern

```mermaid
flowchart LR
    IMG["Image (H x W x 3)"] --> ENC["Frozen ViT encoder<br/>(DINOv2 / DINOv3)"]
    ENC --> FEATS["Dense features<br/>(H/14, W/14, d)"]
    FEATS --> DEC["Depth decoder<br/>(conv upsampler,<br/>DPT-style)"]
    DEC --> DEPTH["Depth map<br/>(H, W, 1)"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style DEC fill:#fef3c7,stroke:#d97706
    style DEPTH fill:#dcfce7,stroke:#16a34a
```

Depth Anything V3 冻结 encoder，只训练 DPT-style decoder。Encoder 提供 rich features；decoder 把这些 features 插值回 image resolution，并回归 depth。

### 为什么一张 image 也能产生 depth

2D image 包含许多与 depth 相关的 monocular cues：

- **Perspective**：3D 中的平行线会在 2D 中汇聚。
- **Texture gradient**：远处 surfaces 的 texture 更小、更密。
- **Occlusion order**：较近 objects 会遮挡较远 objects。
- **Size constancy**：已知 objects（cars、humans）提供近似 scale。
- **Atmospheric perspective**：outdoor scenes 中远处 objects 看起来更朦胧、更偏蓝。

在数十亿 images 上训练的 ViT 会 internalise 这些 cues。只要数据足够多、backbone 足够强，monocular depth 即使没有显式 3D supervision，也能达到合理 accuracy。

### Monocular depth 不能做什么

- **Absolute metric scale**：如果没有 intrinsics 或 scene 中已知大小的 object，网络可以预测“cup 是 spoon 距离的两倍”，但无法知道 cup 是 1 m 还是 10 m 远。
- **Occluded geometry**：椅背没有被看见，就无法可靠推断。
- **真正 untextured / reflective surfaces**：mirrors、glass、uniform walls。网络会报告 plausible 但错误的 depth。

### 2026 年的 Depth Anything V3

- Vanilla DINOv2 ViT-L/14 作为 encoder（frozen）。
- DPT decoder。
- 使用来自多种来源的 posed image pairs 训练（除了 photometric consistency，不需要显式 depth supervision）。
- 从 **任意数量的 visual inputs 中预测 spatially consistent geometry，无论是否已知 camera poses**。
- 在 monocular depth、any-view geometry、visual rendering、camera pose estimation 上达到 SOTA。

当你在 2026 年需要 depth 时，这是要调用的 drop-in model。

### Marigold：用于 depth 的 diffusion

Marigold（Ke et al., CVPR 2024）把 depth estimation 重写成 conditional image-to-image diffusion。Conditioning：RGB。Target：depth map。它使用 pretrained Stable Diffusion 2 U-Net 作为 backbone。输出 depth maps 在 object boundaries 上异常清晰。取舍是：inference 比 feed-forward models 更慢（10-50 denoising steps）。

### Intrinsics 和 pinhole camera

要把带 depth `d` 的 pixel `(u, v)` lift 到 camera coordinates 中的 3D point `(X, Y, Z)`：

```text
fx, fy, cx, cy = camera intrinsics
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

Intrinsics 来自 EXIF metadata、calibration pattern，或 monocular intrinsics estimator（Perspective Fields、UniDepth）。没有 intrinsics 时，你仍然可以假设 60-70° FOV 和中等分辨率 principals 来渲染 point cloud，这对 visualisation 可用，但不能用于 measurement。

### Evaluation

两个标准 metrics：

- **AbsRel**（absolute relative error）：`mean(|d_pred - d_gt| / d_gt)`。越低越好。Production models 通常在 0.05-0.1。
- **delta < 1.25**（threshold accuracy）：满足 `max(d_pred/d_gt, d_gt/d_pred) < 1.25` 的 pixels 比例。越高越好。SOTA 为 0.9+。

对于 relative depth（Depth Anything V3、MiDaS），evaluation 使用这两个 metrics 的 scale-and-shift invariant versions。

## Build It

### Step 1: Depth metrics

```python
import torch

def abs_rel_error(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return (torch.abs(pred - target) / target.clamp(min=1e-6)).mean().item()


def delta_accuracy(pred, target, threshold=1.25, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    ratio = torch.maximum(pred / target.clamp(min=1e-6), target / pred.clamp(min=1e-6))
    return (ratio < threshold).float().mean().item()
```

Evaluation 前永远要 mask invalid depth pixels（zero、NaN、saturated）。

### Step 2: Scale-and-shift alignment

对于 relative-depth models，在计算 metrics 前先把 prediction 对齐到 ground truth。最小二乘拟合 `a * pred + b = target`：

```python
def align_scale_shift(pred, target, mask=None):
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.flatten()
        t = target.flatten()
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    coeffs, *_ = torch.linalg.lstsq(A, t.unsqueeze(-1))
    a, b = coeffs[:2, 0]
    return a * pred + b
```

评估 MiDaS / Depth Anything 时，在 `abs_rel_error` 前先运行 `align_scale_shift`。

### Step 3: 把 depth lift 成 point cloud

```python
import numpy as np

def depth_to_point_cloud(depth, intrinsics):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


depth = np.random.uniform(0.5, 4.0, (240, 320))
intr = (320.0, 320.0, 160.0, 120.0)
pc = depth_to_point_cloud(depth, intr)
print(f"point cloud shape: {pc.shape}  (H, W, 3)")
```

一个函数，所有 3D-lifted application 都会用到。把 point cloud 导出到 `.ply`，然后在 MeshLab 或 CloudCompare 中打开。

### Step 4: 用 synthetic depth scene 做 smoke test

```python
def synthetic_depth(size=96):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    # Floor: linear gradient from near (top) to far (bottom)
    depth = 1.0 + (yy / size) * 4.0
    # Box in the middle: closer
    mask = (np.abs(xx - size / 2) < size / 6) & (np.abs(yy - size * 0.6) < size / 6)
    depth[mask] = 2.0
    return depth.astype(np.float32)


gt = torch.from_numpy(synthetic_depth(96))
pred = gt + 0.3 * torch.randn_like(gt)  # simulated prediction
aligned = align_scale_shift(pred, gt)
print(f"before align  absRel = {abs_rel_error(pred, gt):.3f}")
print(f"after align   absRel = {abs_rel_error(aligned, gt):.3f}")
```

### Step 5: Depth Anything V3 usage（reference）

```python
import torch
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-v2-large")

image = Image.open("street.jpg").convert("RGB")
out = pipe(image)
depth_np = np.array(out["depth"])
```

三行。`out["depth"]` 是 PIL grayscale；转换成 numpy 后再做数学运算。对于 Depth Anything V3，等模型发布后替换 model id 即可；API 不变。

## Use It

- **Depth Anything V3**（Meta AI / ByteDance，2024-2026）：relative depth 的默认选择。Production 中最快的 ViT-large-backbone model。
- **Marigold**（ETH，2024）：视觉质量最高，inference 慢。
- **UniDepth**（ETH，2024）：metric depth，并估计 camera intrinsics。
- **ZoeDepth**（Intel，2023）：metric depth；较老，但仍可靠。
- **MiDaS v3.1**：legacy 但稳定；适合作为 comparison baseline。

典型 integration pattern：

1. RGB frame 到达。
2. Depth model 生成 depth map。
3. Detector 生成 boxes。
4. 通过 depth 把 box centroids lift 到 3D；如果有 point cloud，就与其 merge。
5. Downstream：AR occlusion、path planning、object-size estimation、stereo replacement。

对于 real-time use，Depth Anything V2 Small（INT8 quantised）在消费级 GPU 上以 `518x518` 输入可达到约 30 fps。

## Ship It

本课产出：

- `outputs/prompt-depth-model-picker.md`：根据 latency、metric-vs-relative need 和 scene type，在 Depth Anything V3、Marigold、UniDepth、MiDaS 之间选择。
- `outputs/skill-depth-to-pointcloud.md`：一个 skill，用正确的 intrinsics handling 从 depth maps 构建 point clouds，并导出到 `.ply`。

## 练习

1. **（Easy）** 在你桌面的任意 10 张 images 上运行 Depth Anything V2。把 depth 保存为 grayscale PNGs 并检查。找出一个 predicted depth 看起来错误的 object，并解释 monocular cues 为什么失败。
2. **（Medium）** 给定来自 Depth Anything V2 的 RGB + depth，把它 lift 成 point cloud，并用 `open3d` 渲染。比较两个 scenes（indoor / outdoor），并记录哪个看起来更可信。
3. **（Hard）** 拍五对 images，每对只改变一个已知 object 的位置（例如 bottle 向 camera 靠近 30 cm）。用 UniDepth 在两张图上预测 metric depth。报告 predicted distance delta 与真实 30 cm 的差距。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Monocular depth | “Single-image depth” | 从一帧 RGB 做 depth estimation，没有 stereo 或 LiDAR |
| Relative depth | “Ordered depth” | 没有真实世界单位的有序 z-values |
| Metric depth | “Absolute distance” | 以 metres 为单位的 depth；需要 calibration 或带 metric supervision 训练的模型 |
| AbsRel | “Absolute relative error” | `|d_pred - d_gt| / d_gt` 的 mean；标准 depth metric |
| Delta accuracy | “delta < 1.25” | Prediction 在 ground truth 25% 以内的 pixels 比例 |
| Pinhole camera | “fx, fy, cx, cy” | 用来把 `(u, v, d)` lift 成 `(X, Y, Z)` 的 camera model |
| DPT | “Dense Prediction Transformer” | 冻结 ViT encoders 上方用于 depth 的 conv-based decoder |
| DINOv2 backbone | “The reason it works” | 不需要 depth labels 也能跨 domains 泛化的 self-supervised features |

## Further Reading

- [Depth Anything V3 paper page](https://depth-anything.github.io/) — 使用 DINOv2 encoder 的 SOTA monocular depth
- [Marigold (Ke et al., CVPR 2024)](https://marigoldmonodepth.github.io/) — diffusion-based depth estimation
- [UniDepth (Piccinelli et al., 2024)](https://arxiv.org/abs/2403.18913) — metric depth with intrinsics
- [MiDaS v3.1 (Intel ISL)](https://github.com/isl-org/MiDaS) — canonical relative-depth baseline
- [DINOv3 blog post (Meta)](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) — 提升 depth accuracy 的 encoder family
