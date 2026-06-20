# 视觉编码器 Patch

> 读取像素的视觉模型也需要一个像素 tokenizer。Patch embedding 就是这个 tokenizer。把图像切成一个方格网格，展平每个小方块，通过一层线性投影，再加上 2D 位置讯号，让 transformer 知道每个方块原本在图像里的位置。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37 (Track B foundations)
**Time:** ~90 minutes

## 学习目标

- 把图像 token 化成一段固定长度的 patch embeddings 序列。
- 实现基于 `Conv2d` 的 patch projection，并让它匹配 unfold-then-linear 的数学形式。
- 构建 deterministic 的 2D sinusoidal position embedding，让 token 顺序编码空间位置。
- 在 synthetic fixture 上验证 patch 数量、embedding shape，以及 `Conv2d`/unfold 的等价性。

## 问题

Transformer 吃的是一串向量。图像是一个 3 通道网格。如果把每个像素都当成一个 token，序列长度会爆炸：一张 224x224 RGB 图像有 150,528 个 token，12 层 transformer 的 attention 根本承受不起。如果把整张图像读成一个巨大的扁平向量，又会丢掉局部性，而 attention 层无法凭空把这种结构恢复回来。编码器前端的任务，就是把像素网格压缩成几百个 token，让每个 token 总结一个方形区域。

Patch embedding 用一层线性投影解决这个问题。把一张 224x224 图像切成 16x16 patch，会得到 14x14 的 patch 网格，共 196 个 patch。每个 patch 从 `(3, 16, 16) = 768` 个像素值展平成一个向量，然后一层 linear layer 把它映射到模型的 hidden dimension。Transformer 看到的是 196 个维度为 `hidden` 的 token（常见是 768），再加上一个 CLS token。这就是后续网络能消化的序列长度。

## 概念

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### 为什么用 patch，而不是像素

Attention 的计算量随序列长度平方增长。196 个 token 的序列，每个 head 每一层需要 `196 * 196 = 38,416` 个 attention scores；150,528 个 token 的序列需要 `150,528 * 150,528 = 22.6 billion` 个。Patch 带来了大约 590,000x 的 attention 计算量缩减，而单个 16x16 区域已经足够承载高层视觉任务所需的信号。代价是一个 patch 内部的细粒度空间细节会丢失，所以当精细定位很重要时，下游 multimodal stack 往往还会运行第二条高分辨率分支。

### 为什么一层线性投影就够了

每个 patch 都被当成一个独立向量。投影层会学习一组基：边缘检测器、颜色滤波器、简单纹理。单层 linear layer 很小（ViT-Base 中是 `768 * 768 = 589,824` 个参数），训练也快。更深的 convolutional stem 确实存在（也就是 "hybrid" ViT），但扁平线性投影是标准做法，大多数现代 open-weight encoder 都采用这个形状。

### `Conv2d` 技巧

一个不带 padding 的 `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)`，在数值上等价于 unfold-then-linear，因为每个输出位置都是把 patch 像素和一个 filter 做点积。这个 convolution 就是 patch projection。大多数生产代码库会这么写，因为它在 GPU 上更快，还少了一次 reshape。

### 位置嵌入

投影之后，token 本身不携带顺序。2D sinusoidal embedding 会给每个 token 一个固定讯号，用来编码它的 `(row, col)` 位置。Embedding 维度的一半用多个频率的 sin/cos 编码行位置，另一半编码列位置。这个编码是 deterministic 的，所以你可以在不重新训练的情况下切换分辨率，并且能干净地插值到模型训练时从未见过的网格。

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

对 224 分辨率的 ViT-Base/16 来说，projection 有 590,592 个参数，CLS token 有 768 个参数，sinusoidal position 没有参数。下一课（59）会把一个 12 层 transformer 叠到这个前端上。

### 把等价性当作 sanity check

Patch 步骤有两种写法：`Conv2d` projection，以及显式的 unfold-then-linear。对于同一组权重，它们必须产生相同输出。如果不相同，就说明 unfold 数学写错了，而整个 encoder 都建在不稳的地基上。本课的 tests 会专门覆盖这个等价性。

## Build It

`code/main.py` 实现：

- `PatchEmbed`，一个包装 `Conv2d` 做 patch projection 的 `nn.Module`。
- `sinusoidal_2d(grid_h, grid_w, dim)`，一个 stateless function，用来构建 2D position table。
- `VisionFrontEnd`，把 patch embedding、CLS prepend 和 position addition 组合成一次 forward pass。
- `synthesize_image(seed)` helper，用 `numpy.random` 构造 deterministic 的 224x224x3 fixture。
- 一个 demo：把一张 fixture image 跑过前端，并打印输出 shape、CLS token norm，以及 position embedding 的一行。

运行：

```bash
python3 code/main.py
```

输出：224x224 fixture 会被 token 化为 shape `(1, 197, 768)` 的序列。第一个 token 是 CLS；后面 196 个是 patch tokens。Position embedding 的 norm 在同一行内保持一致，这就是 sinusoidal signature。

## Use It

同样的 patch 前端出现在所有现代 vision-language model 里：CLIP ViT-L/14、SigLIP、DINOv2、Qwen-VL family 和 InternVL stack 都从一个 `Conv2d` patch projection 加 position signal 开始。不同模型家族的差异发生在下游（CLS vs no-CLS pooling、register tokens、14 vs 16 的不同 patch sizes、通过 interpolated positions 做 dynamic resolution）。本课里的 frontend 是这些模型共同站立的底座。

## Tests

`code/test_main.py` 覆盖：

- patch count 匹配 `(image_size / patch_size) ** 2`
- output shape 匹配 `(batch, num_patches + 1, hidden)`
- 在小 fixture 上，`Conv2d` projection 等于 manual unfold-then-linear
- sinusoidal position table 在多次调用之间保持 deterministic
- CLS token 会广播到 batch 维度，且没有 leakage

运行：

```bash
python3 -m unittest code/test_main.py
```

## 练习

1. 把 sinusoidal position 替换成 learned `nn.Parameter`，并在一个很小的 synthetic classification task 上比较 first-epoch loss。固定分辨率下 learned positions 会赢；训练后改变分辨率时 sinusoidal 会赢。

2. 把 `Conv2d` 换成显式的 `nn.Unfold` 加 `nn.Linear`，并断言输出在 float tolerance 内相同。数学相同，只是两种写法。

3. 增加对非方形 patch sizes 的支持（例如给宽屏输入使用 32x16），并验证 position table 能处理非方形网格。

4. 在 batch sizes 1、8、64 下 profile patch step。Patch projection 很少是瓶颈；下游 attention layers 才是主要开销。

5. 把前端当成 frozen feature extractor，在一个 4-class synthetic shape dataset（circles、squares、triangles、stars）上训练。CLS token 输出应该能被线性分离。

## 关键术语

| Term | What it means |
|------|---------------|
| Patch | 图像的一个方形子区域，通常是 14x14 或 16x16 |
| Patch embedding | 把一个展平后的 patch 线性投影到 hidden dim |
| Sequence length | Patch tokenization 之后的 token 数量，通常还要加上 CLS |
| Sinusoidal position | 用固定 sin/cos 讯号编码 2D grid coordinates |
| CLS token | 加在序列开头的 learned vector，用作 pooling head |

## 延伸阅读

- An Image is Worth 16x16 Words（ViT, 2021），了解最初的 patch-embed framing。
- Attention Is All You Need（2017），了解这里扩展到 2D 的 sinusoidal position 公式来源。
- DINOv2 paper，了解 register tokens；你可以把它作为第六个练习添加进来。
