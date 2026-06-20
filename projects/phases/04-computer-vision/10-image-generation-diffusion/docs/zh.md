# 图像生成：Diffusion Models

> Diffusion model 学的是去噪。训练它从 noisy image 中移除一小点 noise，再把这个过程反向重复一千次，你就得到了 image generator。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 07（U-Net），Phase 1 Lesson 06（Probability），Phase 3 Lesson 06（Optimizers）
**Time:** ~75 minutes

## 学习目标

- 推导 forward noising process `x_0 -> x_1 -> ... -> x_T`，并解释为什么 closed-form `q(x_t | x_0)` 对任意 t 都成立
- 实现 DDPM-style training objective：回归每一步加入的 noise，并实现一个从 pure noise 走回 image 的 sampler
- 构建一个 time-conditioned U-Net，小到可以在 CPU 上训练，并能预测任意 timestep 的 noise
- 解释 DDPM 和 DDIM sampling 的区别，以及什么时候该用哪一个（Lesson 23 会深入讲 flow matching 和 rectified flow）

## 问题

GANs 是 one-shot 生成：noise 进来，image 出去，一次 forward pass。它们很快，也很难训练。Diffusion models 是 iterative 生成：从 pure noise 开始，一小步一小步 denoise，image 逐渐浮现。它们很慢，但容易训练。过去五年里，后一个性质压倒了一切：任何小团队都可以训练 diffusion model，并得到合理 samples；GAN training 则是一门要从多年失败 run 中学出来的手艺。

除了训练稳定性，diffusion 的 iterative structure 也是现代图像生成能力的来源：text conditioning、inpainting、image editing、super-resolution、controllable style。Sampling loop 的每一步都是注入新 constraint 的位置。这个 hook 解释了为什么 Stable Diffusion、Imagen、DALL-E 3、Midjourney 以及你会使用的每个 controllable image model 都基于 diffusion。

这一课构建最小 DDPM：forward noising、backward denoising、training loop。下一课（Stable Diffusion）会把它接入一个 production system：VAE、text encoder 和 classifier-free guidance。

## 概念

### Forward process

取一张 image `x_0`。加入很小的 Gaussian noise 得到 `x_1`。再加入一点点 noise 得到 `x_2`。持续 T 步，直到 `x_T` 几乎无法和 pure Gaussian noise 区分。

```text
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
```

`beta_t` 是一个很小的 variance schedule，通常在 T=1000 步中从 0.0001 线性增长到 0.02。每一步都会略微缩小 signal，并注入新的 noise。

### Closed-form jump

一步一步加 noise 是一条 Markov chain，但数学上可以折叠：你可以一步从 `x_0` 直接采样 `x_t`。

```text
Define alpha_t = 1 - beta_t
Define alpha_bar_t = prod_{s=1..t} alpha_s

Then:
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

Equivalently:
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon ~ N(0, I)
```

这个单一方程就是 diffusion 可实践的全部原因。训练时你随机选择一个 `t`，直接从 `x_0` 采样 `x_t`，然后一步训练；不需要模拟完整 Markov chain。

### Reverse process

Forward process 是固定的。Reverse process `p(x_{t-1} | x_t)` 是 neural network 要学习的东西。Diffusion models 不直接预测 `x_{t-1}`；它们预测 step t 加入的 noise `epsilon`，再由数学公式推导出 `x_{t-1}`。

```mermaid
flowchart LR
    X0["x_0<br/>(clean image)"] --> Q1["q(x_t|x_0)<br/>add noise"]
    Q1 --> XT["x_t<br/>(noisy)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["predicted epsilon"]
    EPS --> LOSS["MSE against<br/>true epsilon"]

    XT -.->|sampling| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|repeat 1000x| X0S["x_0 (sampled)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### Training loss

每个 training step：

1. Sample 一张真实 image `x_0`。
2. 从 [1, T] 中均匀 sample 一个 timestep `t`。
3. Sample noise `epsilon ~ N(0, I)`。
4. 计算 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`。
5. 用网络预测 `epsilon_theta(x_t, t)`。
6. 最小化 `|| epsilon - epsilon_theta(x_t, t) ||^2`。

就是这样。Neural network 学会预测任意 timestep 的 noise。Loss 是 MSE。没有 adversarial game，没有 collapse，没有 oscillation。

### Sampler（DDPM）

生成时，从 `x_T ~ N(0, I)` 开始，每次向后走一步。

```text
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0
```

关键在于：虽然一般情况下 reverse conditional 并没有 closed form，但对这个特定 Gaussian forward process 来说，它有。那些看起来难看的 coefficients 正是 Bayes' rule 给你的结果。

### 为什么是 1000 步

Forward noise schedule 的选择，是为了让每一步加入刚好足够少的 noise，使 reverse step 近似 Gaussian。步数太少，reverse step 会离 Gaussian 很远，网络很难建模。步数太多，sampling 会变贵且收益递减。T=1000 加 linear schedule 是 DDPM 默认选择。

### DDIM：快 20 倍的 sampling

Training 不变。Sampling 改变。DDIM（Song et al., 2020）定义了一个 deterministic reverse process，可以在不重新训练的情况下跳过 timesteps。用 DDIM 50 steps sampling，通常能得到接近 1000-step DDPM 的质量。每个 production system 都使用 DDIM 或更快的变体（DPM-Solver、Euler ancestral）。

### Time conditioning

网络 `epsilon_theta(x_t, t)` 需要知道自己正在 denoise 哪个 timestep。现代 diffusion models 通过 sinusoidal time embeddings 注入 `t`（和 transformers 中的 positional encoding 是同一个思路），并把它加到每个 U-Net level 的 feature maps 中。

```text
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

没有 time conditioning，网络必须从 image 本身猜测 noise level；这也能工作，但 sample efficiency 差很多。

## Build It

### Step 1: Noise schedule

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

预计算一次，然后在 training 和 sampling 时按 index gather。

### Step 2: Forward diffusion（q_sample）

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

一行 closed form。`t` 是一批 timesteps，batch 中每张 image 一个。

### Step 3: 一个很小的 time-conditioned U-Net

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

这是一个 two-level U-Net，把 time conditioning 注入 bottleneck。真实 images 需要加深、加宽。

### Step 4: Training loop

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

这就是完整 training loop。没有 GAN game，没有 specialised loss，只有一次 MSE call。

### Step 5: Sampler（DDPM）

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

生成一批 samples 需要 1000 次 forward passes。真实代码里你会把它换成 DDIM 50-step sampler。

### Step 6: DDIM sampler（deterministic，约快 20 倍）

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0` 是完全 deterministic（同一个 noise input 总会产生同一个 output）。`eta=1` 会恢复 DDPM。

## Use It

Production work 使用 `diffusers`：

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

这个库提供 ready-made schedulers（DDPM、DDIM、DPM-Solver、Euler、Heun）、可配置 U-Nets、text-to-image 和 image-to-image pipelines，以及 LoRA fine-tuning helpers。

Research work 中，`k-diffusion`（Katherine Crowson）有最忠实的 reference implementations 和最好的 sampling variants。

## Ship It

这一课产出：

- `outputs/prompt-diffusion-sampler-picker.md`：一个 prompt，会根据 quality target、latency budget 和 conditioning type 选择 DDPM / DDIM / DPM-Solver / Euler。
- `outputs/skill-noise-schedule-designer.md`：一个 skill，给定 T 和 target corruption level 后，产生 linear、cosine 或 sigmoid beta schedule，以及 signal-to-noise ratio 随时间变化的 diagnostic plots。

## 练习

1. **（Easy）** Visualise forward process：取一张 image，绘制 `t in [0, 100, 250, 500, 750, 1000]` 时的 `x_t`。验证 `x_1000` 看起来像 pure Gaussian noise。
2. **（Medium）** 在 synthetic-circles dataset 上训练 TinyUNet 20 epochs，并 sample 16 个 circles。比较 DDPM（1000 steps）和 DDIM（50 steps）sampling：它们用同一个 noise seed 会产生类似 images 吗？
3. **（Hard）** 实现 cosine noise schedule（Nichol & Dhariwal, 2021）：`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`。用 linear 和 cosine schedules 训练同一个模型，并展示 cosine 在低 step counts 下会给出更好的 samples。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Forward process | “Add noise over time” | 固定 Markov chain，在 T steps 内把 image 腐蚀成 Gaussian noise |
| Reverse process | “Denoise step by step” | 学到的 distribution，从 noise 一步步走回 image |
| Epsilon prediction | “Predict the noise” | Training target：`epsilon_theta(x_t, t)` 预测 step t 加入的 noise |
| Beta schedule | “Noise amounts” | 长度为 T 的小 variances 序列，定义每一步加入多少 noise |
| alpha_bar_t | “Cumulative retain factor” | 截止时间 t 的 `(1 - beta_s)` 连乘；t 越大，剩余 signal 越少 |
| DDPM sampler | “Ancestral, stochastic” | 从 conditional Gaussian 中采样每个 `x_{t-1}`；1000 steps |
| DDIM sampler | “Deterministic, fast” | 把 sampling 改写成 deterministic ODE；20-100 steps 可获得类似质量 |
| Time conditioning | “Tell the model which t” | 把 t 的 sinusoidal embedding 注入 U-Net，让模型知道 noise level |

## 进一步阅读

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)：让 diffusion 变得实用、并在 FID 上超过 GANs 的 paper
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)：cosine schedule 和 v-parameterisation
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502)：让 real-time inference 成为可能的 deterministic sampler
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364)：对每个 diffusion design choice 的统一视角；当前最佳 reference
