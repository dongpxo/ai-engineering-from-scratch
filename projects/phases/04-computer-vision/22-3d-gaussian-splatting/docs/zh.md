# 从零开始 3D Gaussian Splatting

> 一个 scene 是由数百万个 3D Gaussians 组成的 cloud。每个 Gaussian 都有 position、orientation、scale、opacity，以及一个依赖 viewing direction 的 colour。把它们 rasterise，对 rasterisation 做 backprop，就结束了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 13（3D Vision & NeRF），Phase 1 Lesson 12（Tensor Operations），Phase 4 Lesson 10（Diffusion basics optional）
**Time:** ~90 minutes

## 学习目标

- 解释为什么到 2026 年，3D Gaussian Splatting 已经取代 NeRF，成为 photorealistic 3D reconstruction 的 production default
- 说出每个 Gaussian 的六类参数（position、rotation quaternion、scale、opacity、spherical harmonics colour、optional feature），以及每类贡献多少 floats
- 从零实现一个使用 `alpha` compositing 的 2D Gaussian splatting rasterizer，并说明 3D 情况如何投影到同一个 loop
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建一个 scene，并导出到 `KHR_gaussian_splatting` glTF extension，或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` schema

## 问题

NeRF 把一个 scene 存成 MLP 的 weights。每个 rendered pixel 都需要沿 ray 做数百次 MLP queries。训练需要数小时，渲染需要数秒，而且 weights 无法直接编辑。如果你想移动 scene 里的椅子，就得重新训练。

3D Gaussian Splatting（Kerbl、Kopanas、Leimkühler、Drettakis，SIGGRAPH 2023）替换了这一切。一个 scene 是一组显式 3D Gaussians。Rendering 是 100+ fps 的 GPU rasterisation。Training 只需要几分钟。Editing 是直接的：平移一组 Gaussians，就移动了椅子。到 2026 年，Khronos Group 已经 ratify 了 Gaussian splats 的 glTF extension，OpenUSD 26.03 发布了 Gaussian splat schema，Zillow 和 Apartments.com 用它们渲染房产，大多数新的 3D reconstruction 论文也都是围绕核心 3DGS idea 的变体。

Mental model 很简单，但数学里有足够多 moving parts，所以大多数介绍会从 rasterisation 开始，然后跳过 projection 和 spherical harmonics。本课会构建完整流程：先做 2D 版本，再扩展到 3D。

## 概念

### 一个 Gaussian 携带什么

一个 3D Gaussian 是 space 中的 parametric blob，带有这些 attributes：

```text
position         mu         (3,)    centre in world coordinates
rotation         q          (4,)    unit quaternion encoding orientation
scale            s          (3,)    log-scales per axis (exponentiated at render time)
opacity          alpha      (1,)    post-sigmoid opacity [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   view-dependent colour
```

Rotation + scale 会构造一个 `3x3` covariance：`Sigma = R S S^T R^T`。这就是 Gaussian 在 3D 中的 shape。Spherical harmonics 让 colour 可以随 viewing direction 改变：specular highlights、subtle sheen、view-dependent glow，而不需要存储 per-view textures。使用 SH degree 3 时，每个 colour channel 有 16 个 coefficients，仅 colour 就需要每个 Gaussian 48 个 floats。

一个 scene 通常有 1-5 million Gaussians。每个 Gaussian 大约存 60 个 floats（3 + 4 + 3 + 1 + 48 + misc）。一个五百万 Gaussian 的 scene 大约是 240 MB，远小于带 per-point texture 的等价 point cloud，也比高分辨率重渲染所需的 NeRF MLP weights 小一个数量级。

### Rasterisation，不是 ray marching

```mermaid
flowchart LR
    SCENE["Millions of 3D Gaussians<br/>(position, rotation, scale,<br/>opacity, SH colour)"] --> PROJ["Project to 2D<br/>(camera extrinsics + intrinsics)"]
    PROJ --> TILES["Assign to tiles<br/>(16x16 screen-space)"]
    TILES --> SORT["Depth-sort<br/>per tile"]
    SORT --> ALPHA["Alpha-composite<br/>front-to-back"]
    ALPHA --> PIX["Pixel colour"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全都 GPU-friendly。没有 per-pixel MLP query。一张 RTX 3080 Ti 可以用 147 fps 渲染 600 万个 splats。

### Projection step

World position 为 `mu`、3D covariance 为 `Sigma` 的 3D Gaussian，会投影成 screen position 为 `mu'`、2D covariance 为 `Sigma'` 的 2D Gaussian：

```text
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

2D Gaussian 的 footprint 是一个 ellipse，它的 axes 是 `Sigma'` 的 eigenvectors。这个 ellipse 内的每个 pixel 都会收到该 Gaussian 的 contribution，权重为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha-compositing 规则

对一个 pixel 来说，所有覆盖它的 Gaussians 会按 back-to-front 排序（或者等价地，用反向公式做 front-to-back）。Colour 使用和 1980 年代以来所有 semi-transparent rasteriser 相同的公式合成：

```text
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour
```

这和 **NeRF 的 volumetric render 是同一个方程**，只是这里是在一组显式 sparse Gaussians 上求和，而不是沿 ray 做 dense samples。这个一致性解释了为什么 rendered quality 可以匹配 NeRF：两者都在积分同一个 radiance-field equation。

### 为什么它可微

每一步：projection、tile assignment、alpha compositing、SH evaluation，都对 Gaussian parameters 可微。给定一张 ground-truth image，计算 rendered pixel loss，通过 rasteriser backprop，使用 gradient descent 更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 iterations，Gaussians 会找到正确的位置、scales 和 colours。

### Densification 和 pruning

固定数量的 Gaussians 无法覆盖复杂 scene。Training 包含两个 adaptive mechanisms：

- **Clone**：当某个 Gaussian 的 gradient magnitude 很高但 scale 很小时，在当前位置 clone 一个 Gaussian；这表示这里需要更多 detail。
- **Split**：当一个大尺度 Gaussian 的 gradient 很高时，把它 split 成两个更小的 Gaussians；一个大 Gaussian 太 smooth，无法拟合该区域。
- **Prune**：删除 opacity 低于 threshold 的 Gaussians；它们没有贡献。

Densification 每隔 N iterations 运行一次。一个 scene 通常会从 SfM points 初始化的约 100k Gaussians，增长到训练结束时的 1-5M。

### 一段话讲 spherical harmonics

View-dependent colour 是 unit sphere 上的函数 `c(direction)`。Spherical harmonics 是 sphere 的 Fourier basis。截断到 degree `L`，每个 channel 会得到 `(L+1)^2` 个 basis functions。为一个新 view 计算 colour，就是把 learned SH coefficients 和在 viewing direction 上求值的 basis 做 dot product。Degree 0 = 一个 coefficient = constant colour。Degree 3 = 16 个 coefficients = 足够捕捉 Lambertian shading、specular 和轻微 reflection。3D Gaussian Splatting 论文默认使用 degree 3。

### 2026 年 production stack

```text
1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 和 generative variants

- **4D Gaussian Splatting**：Gaussians 是 time 的函数；用于 volumetric video（Superman 2026、A$AP Rocky 的 "Helicopter"）。
- **Generative splats**：text-to-splat models（World Labs 的 Marble），可以 hallucinate 整个 scenes。
- **3D Gaussian Unscented Transform**：NVIDIA NuRec 的 variant，用于 autonomous driving simulation。

## Build It

### Step 1: 一个 2D Gaussian

我们先构建一个 2D rasteriser。3D 情况在 projection 之后会退化到它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 会为每个 `(Gaussian, pixel)` pair 计算 quadratic form `diff^T Sigma^-1 diff`。

### Step 2: 2D splatting rasteriser

Front-to-back alpha-compositing。2D 中 depth 没有实际意义，所以我们用一个 learned per-Gaussian scalar 排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

它不快，真实实现会使用 tile-based CUDA kernels。但这里的 math 完全正确，而且 fully differentiable。

### Step 3: 一个可训练的 2D splat scene

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是 unconstrained parameters，并在 render time 通过正确 activation 映射到有效范围。这是每个 3DGS implementation 的标准模式。

### Step 4: 把 2D Gaussians 拟合到 target image

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 steps，64 个 Gaussians 会聚到两个 shapes 上。这就是完整想法：对显式 geometric primitives 做 gradient descent。

### Step 5: 从 2D 到 3D

3D extension 保留同一个 loop。新增的是：

1. 每个 Gaussian 的 rotation 是 quaternion，而不是单个 angle。
2. Covariance 是 `R S S^T R^T`，其中 `R` 由 quaternion 构造，`S = diag(exp(log_scale))`。
3. Projection `(mu, Sigma) -> (mu', Sigma')` 使用 camera extrinsics 和在 `mu` 处的 perspective projection Jacobian。
4. Colour 变成 spherical-harmonics expansion；在 viewing direction 上求值。
5. Depth-sort 使用真实 camera-space z，而不是 learned scalar。

每个 production implementation（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都在 GPU 上用 tile-based CUDA kernels 做同一件事。

### Step 6: Spherical harmonics evaluation

Degree 3 以内的 SH basis 每个 channel 有 16 项。求值如下：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

Learned `sh_coeffs` 存储这个 Gaussian “在每个方向上的 colour”。Render time 时，你用当前 view direction 求值，得到一个 3-vector RGB。

## Use It

真实 3DGS 工作请使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS trainer。典型 scene 在 RTX 4090 上需要 10-30 分钟。

2026 年重要的 export options：

- `.ply`：raw Gaussian cloud（portable，文件最大）。
- `.splat`：PlayCanvas / SuperSplat quantised format。
- glTF `KHR_gaussian_splatting`：Khronos standard，可跨 viewers 使用（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat`：USD-native，用于 NVIDIA Omniverse 和 Vision Pro pipelines。

对于 4D / dynamic scenes，`4DGS` 和 `Deformable-3DGS` 用 time-varying means 和 opacities 扩展同一套 machinery。

## Ship It

本课产出：

- `outputs/prompt-3dgs-capture-planner.md`：一个 prompt，为给定 scene type 规划 capture session（照片数量、camera path、lighting）。
- `outputs/skill-3dgs-export-router.md`：一个 skill，根据 downstream viewer 或 engine 选择正确 export format（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **（Easy）** 在另一个 synthetic image 上运行上面的 2D splat trainer。把 `num_splats` 设为 `[16, 64, 256]`，并为每个设置绘制 MSE vs step。找出 diminishing returns 的位置。
2. **（Medium）** 扩展 2D rasteriser，让每个 Gaussian 的 RGB colours 可以通过 degree-2 harmonic 依赖一个 scalar “view angle”。在一对 target images 上训练，并验证模型能重建两者。
3. **（Hard）** Clone `nerfstudio`，并在你拥有的任意 scene（桌面、植物、人脸、房间）的 20-photo capture 上训练 `splatfacto`。导出到 glTF `KHR_gaussian_splatting`，并在 viewer（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）中打开。报告 training time、Gaussians 数量和 rendered fps。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 3DGS | “Gaussian splats” | 用数百万个 3D Gaussians 显式表示 scene，每个 Gaussian 有 position、rotation、scale、opacity、SH colour |
| Covariance | “Shape of the Gaussian” | `Sigma = R S S^T R^T`；一个 Gaussian 的 orientation 和 anisotropic scale |
| Alpha compositing | “Back-to-front blend” | 与 NeRF volumetric render 相同的方程，现在作用在显式 sparse set 上 |
| Densification | “Clone and split” | 在 reconstruction under-fit 的区域 adaptive 添加新 Gaussians |
| Pruning | “Delete low-opacity” | 删除训练中 opacity 接近 0 的 Gaussians |
| Spherical harmonics | “View-dependent colour” | Sphere 上的 Fourier basis；把 colour 存成 viewing direction 的函数 |
| Splatfacto | “nerfstudio's 3DGS” | 2026 年训练 3DGS 最容易的路径 |
| `KHR_gaussian_splatting` | “glTF standard” | Khronos 2026 extension，让 3DGS 可跨 viewers 和 engines 移植 |

## Further Reading

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — original paper
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — production-quality CUDA rasteriser
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — reference training recipe
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 portable format
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — industry overview
