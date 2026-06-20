# 多目标跟踪与视频记忆

> Tracking 是 detection 加 association。每一帧都检测。再把这一帧的 detections 和上一帧的 tracks 按 ID 匹配起来。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06（YOLO Detection），Phase 4 Lesson 08（Mask R-CNN），Phase 4 Lesson 24（SAM 3）
**Time:** ~60 minutes

## 学习目标

- 区分 tracking-by-detection 和 query-based tracking，并说出算法家族（SORT、DeepSORT、ByteTrack、BoT-SORT、SAM 2 memory tracker、SAM 3.1 Object Multiplex）
- 从零实现 IoU + Hungarian assignment，用于 classic tracking-by-detection
- 解释 SAM 2 的 memory bank，以及为什么它比 IoU-based association 更能处理 occlusion
- 阅读三个 tracking metrics（MOTA、IDF1、HOTA），并为给定 use case 选择最重要的指标

## 问题

Detector 会告诉你单帧中 objects 在哪里。Tracker 会告诉你 frame `t` 中的哪个 detection，和 frame `t-1` 中的哪个 detection 是同一个 object。没有它，你就无法统计 objects 穿过一条线的次数，无法在 occlusion 中跟踪一个球，也无法知道“car #4 已经在车道里待了 8 秒”。

Tracking 是每个面向视频的产品都需要的能力：sports analytics、surveillance、autonomous driving、medical video analysis、wildlife monitoring、wordmark counting。核心 building blocks 是共享的：per-frame detector、motion model（Kalman filter 或更丰富的模型）、association step（基于 IoU / cosine / learned features 的 Hungarian algorithm），以及 track lifecycle（birth、update、death）。

2026 年带来了两个新模式：**SAM 2 memory-based tracking**（使用 feature-memory，而不是 motion-model association）和 **SAM 3.1 Object Multiplex**（为同一 concept 的多个 instances 共享 memory）。这一课先走 classical stack，再进入 memory-based approach。

## 概念

### Tracking-by-detection

```mermaid
flowchart LR
    F1["Frame t"] --> DET["Detector"] --> D1["Detections at t"]
    PREV["Tracks up to t-1"] --> PREDICT["Motion predict<br/>(Kalman)"]
    PREDICT --> PRED["Predicted tracks at t"]
    D1 --> ASSOC["Hungarian assignment<br/>(IoU / cosine / motion)"]
    PRED --> ASSOC
    ASSOC --> UPDATE["Update matched tracks"]
    ASSOC --> NEW["Birth new tracks"]
    ASSOC --> DEAD["Age unmatched tracks; delete after N"]
    UPDATE --> NEXT["Tracks at t"]
    NEW --> NEXT
    DEAD --> NEXT

    style DET fill:#dbeafe,stroke:#2563eb
    style ASSOC fill:#fef3c7,stroke:#d97706
    style NEXT fill:#dcfce7,stroke:#16a34a
```

2026 年你会遇到的每个 tracker，都是这个 loop 的某种变体。区别在于：

- **SORT**（2016）：Kalman filter + IoU Hungarian。简单、快速，没有 appearance model。
- **DeepSORT**（2017）：SORT + 每条 track 一个 CNN-based appearance feature（ReID embedding）。更能处理 crossing。
- **ByteTrack**（2021）：把 low-confidence detections 作为第二阶段来关联；不需要 appearance features，但在 MOT17 上表现很强。
- **BoT-SORT**（2022）：Byte + camera motion compensation + ReID。
- **StrongSORT / OC-SORT**：ByteTrack descendants，拥有更好的 motion 和 appearance。

### 用一段话理解 Kalman filter

Kalman filter 为每条 track 维护一个 state `(x, y, w, h, dx, dy, dw, dh)` 和一个 covariance。每一帧先用 constant-velocity model **predict** state，再用 matched detection **update**。当 prediction uncertainty 高时，update 会更信任 detection。这样可以得到平滑 trajectories，并能让 track 穿过短暂 occlusion（1-5 frames）。

每个 classical tracker 都在 motion-prediction step 中使用 Kalman filter。

### Hungarian algorithm

给定一个 `M x N` cost matrix（tracks x detections），找到让 total cost 最小的一对一 assignment。Cost 通常是 `1 - IoU(track_bbox, detection_bbox)`，或者 appearance features 的 negative cosine similarity。Runtime 是 O((M+N)^3)；当 M、N 到 ~1000 时，用 `scipy.optimize.linear_sum_assignment` 在 Python 中仍然足够快。

### ByteTrack 的关键想法

标准 trackers 会丢弃 low-confidence detections（< 0.5）。ByteTrack 会把它们保留下来作为 **second-stage candidates**：在 tracks 和 high-confidence detections 匹配之后，unmatched tracks 会再尝试用稍微宽松一点的 IoU threshold 匹配 low-confidence detections。这样能恢复短暂 occlusions，减少 crowds 附近的 ID switches。

### SAM 2 memory-based tracking

SAM 2 通过维护每个 instance 的 **memory bank** 来处理视频，里面保存 per-instance spatio-temporal features。给定某一帧上的 prompt（click、box、text），它会把 instance 编码进 memory。后续 frames 中，memory 会与新帧 features 做 cross-attention，decoder 则为同一个 instance 在新帧中产生 mask。

没有 Kalman filter，没有 Hungarian assignment。Association 隐式发生在 memory-attention operation 中。

优点：
- 对大型 occlusions 更 robust（memory 会跨很多 frames 携带 instance identity）。
- 与 SAM 3 的 text prompts 结合时支持 open-vocabulary。
- 不需要单独的 motion model。

缺点：
- 对 many-object tracking 来说比 ByteTrack 慢。
- Memory bank 会增长；context window 有上限。

### SAM 3.1 Object Multiplex

之前的 SAM 2 / SAM 3 tracking 会为每个 instance 保持单独 memory bank。50 个 objects，就是 50 个 memory banks。Object Multiplex（2026 年 3 月）把它们折叠成一个 shared memory，并使用 **per-instance query tokens**。成本相对 instances 数量呈 sub-linear scaling。

Multiplex 是 2026 年 crowd tracking 的新默认选择：concert crowds、warehouse workers、traffic intersections。

### 必须知道的三个指标

- **MOTA（Multi-Object Tracking Accuracy）**：`1 - (FN + FP + ID switches) / GT`。按 error type 加权；这是一个会混合 detection 和 association failures 的单一指标。
- **IDF1（ID F1）**：ID precision 和 ID recall 的 harmonic mean。专门关注每个 ground-truth track 是否能随时间保持同一个 ID。对 ID-switch-sensitive tasks 来说比 MOTA 更好。
- **HOTA（Higher Order Tracking Accuracy）**：分解为 detection accuracy（DetA）和 association accuracy（AssA）。2020 年以来的 community standard；最全面。

Surveillance（谁是谁）：报告 IDF1。Sports analytics（统计传球）：HOTA。一般 academic comparison：HOTA。

## Build It

### Step 1: IoU-based cost matrix

```python
import numpy as np


def bbox_iou(a, b):
    """
    a, b: (N, 4) arrays of [x1, y1, x2, y2].
    Returns (N_a, N_b) IoU matrix.
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

### Step 2: Minimal SORT-style tracker

为了简洁，这里省略固定 constant-velocity Kalman；我们只使用简单 IoU association。在 production 中，Kalman predict 是必需的。`sort` Python package 提供完整版本。

```python
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, bbox, frame):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.hits = 1

    def update(self, bbox, frame):
        self.bbox = bbox
        self.last_frame = frame
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1
            return [(t.id, t.bbox) for t in self.tracks]

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array(detections) if len(detections) else np.empty((0, 4))

        iou = bbox_iou(track_boxes, det_boxes) if len(det_boxes) else np.zeros((len(track_boxes), 0))
        cost = 1 - iou
        cost[iou < self.iou_threshold] = 1e6

        matched_track = set()
        matched_det = set()
        if cost.size > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    self.tracks[r].update(det_boxes[c], frame)
                    matched_track.add(r); matched_det.add(c)

        for i, d in enumerate(det_boxes):
            if i not in matched_det:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks]
```

60 行。输入 per-frame detections，返回 per-frame track IDs。真实系统会加入 Kalman predict、ByteTrack 的 second-stage re-match，以及 appearance features。

### Step 3: Synthetic trajectory test

```python
def synthetic_frames(num_frames=20, num_objects=3, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-5, 5, size=(num_objects, 2))
    frames = []
    for f in range(num_frames):
        dets = []
        for i in range(num_objects):
            cx, cy = starts[i] + f * velocities[i]
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10])
        frames.append(dets)
    return frames


tracker = SimpleTracker()
for f, dets in enumerate(synthetic_frames()):
    tracks = tracker.step(dets, f)
```

三个沿直线移动的 objects 应该在全部 20 frames 中保持自己的 IDs。

### Step 4: ID-switch metric

```python
def count_id_switches(tracks_per_frame, gt_per_frame):
    """
    tracks_per_frame:  list of list of (track_id, bbox)
    gt_per_frame:      list of list of (gt_id, bbox)
    Returns number of ID switches.
    """
    prev_assignment = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        t_boxes = np.array([b for _, b in tracks])
        g_boxes = np.array([b for _, b in gts])
        iou = bbox_iou(g_boxes, t_boxes)
        for g_idx, (gt_id, _) in enumerate(gts):
            j = iou[g_idx].argmax()
            if iou[g_idx, j] > 0.5:
                t_id = tracks[j][0]
                if gt_id in prev_assignment and prev_assignment[gt_id] != t_id:
                    switches += 1
                prev_assignment[gt_id] = t_id
    return switches
```

这是一个简化的 IDF1-adjacent metric：统计 ground-truth object 改变 assigned predicted track ID 的次数。真实 MOTA / IDF1 / HOTA tooling 在 `py-motmetrics` 和 `TrackEval` 中。

## Use It

2026 年的 production trackers：

- `ultralytics`：内置 YOLOv8 + ByteTrack / BoT-SORT。`results = model.track(source, tracker="bytetrack.yaml")`。默认选择。
- `supervision`（Roboflow）：ByteTrack wrappers 加 annotation utilities。
- SAM 2 / SAM 3.1：通过 `processor.track()` 做 memory-based tracking。
- Custom stack：detector（YOLOv8 / RT-DETR）+ `sort-tracker` / `OC-SORT` / `StrongSORT`。

选择方式：

- Pedestrians / cars / boxes，30+ fps：**ByteTrack with ultralytics**。
- Crowd 中同一 class 的很多 instances：**SAM 3.1 Object Multiplex**。
- 有严重 occlusions，且 appearance 可识别：**DeepSORT / StrongSORT**（ReID features）。
- Sports / complex interactions：**BoT-SORT** 或 learned trackers（MOTRv3）。

## Ship It

这一课产出：

- `outputs/prompt-tracker-picker.md`：根据 scene type、occlusion patterns 和 latency budget，在 SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1 中做选择。
- `outputs/skill-mot-evaluator.md`：针对 ground-truth tracks，写出完整 MOTA / IDF1 / HOTA evaluation harness。

## Exercises

1. **（Easy）** 用 3、10、30 个 objects 运行上面的 synthetic tracker。报告每种情况下的 ID-switch count。指出 simple IoU-only association 从哪里开始失败。
2. **（Medium）** 在 association 前加入 constant-velocity Kalman predict step。展示短暂（2-3 frame）occlusions 不再造成 ID switches。
3. **（Hard）** 通过 `transformers` 集成 SAM 2 的 memory-based tracker，作为 alternative tracker backend。在一段 30 秒 crowd clip 上同时运行 SimpleTracker 和 SAM 2，并为 5 个显著人物手动标注 ground-truth IDs，比较 ID-switch counts。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Tracking-by-detection | “Detect then associate” | Per-frame detector + 基于 IoU / appearance 的 Hungarian assignment |
| Kalman filter | “Motion predict” | 用于平滑 track predictions 和处理 occlusion 的 linear dynamics + covariance |
| Hungarian algorithm | “Optimal assignment” | 求解 minimum-cost bipartite matching problem；`scipy.optimize.linear_sum_assignment` |
| ByteTrack | “Low-confidence second pass” | 把 unmatched tracks 重新匹配到 low-confidence detections，以恢复短暂 occlusions |
| DeepSORT | “SORT + appearance” | 加入用于 cross-frame matching 的 ReID feature；更利于 ID preservation |
| Memory bank | “SAM 2 trick” | 跨 frames 存储的 per-instance spatio-temporal features；cross-attention 替代显式 association |
| Object Multiplex | “SAM 3.1 shared memory” | 使用 per-instance queries 的单个 shared memory，用于快速 many-object tracking |
| HOTA | “Modern tracking metric” | 分解为 detection 和 association accuracy；community standard |

## Further Reading

- [SORT（Bewley et al., 2016）](https://arxiv.org/abs/1602.00763)：minimal tracking-by-detection paper
- [DeepSORT（Wojke et al., 2017）](https://arxiv.org/abs/1703.07402)：加入 appearance feature
- [ByteTrack（Zhang et al., 2022）](https://arxiv.org/abs/2110.06864)：low-confidence second pass
- [BoT-SORT（Aharon et al., 2022）](https://arxiv.org/abs/2206.14651)：camera motion compensation
- [HOTA（Luiten et al., 2020）](https://arxiv.org/abs/2009.07736)：decomposed tracking metric
- [SAM 2 video segmentation（Meta, 2024）](https://ai.meta.com/sam2/)：memory-based tracker
- [SAM 3.1 Object Multiplex（Meta, March 2026）](https://ai.meta.com/blog/segment-anything-model-3/)
