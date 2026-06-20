# AI 中的复数

> -1 的平方根并不“虚幻”。它是 rotations、frequencies，以及半个 signal processing 的关键。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04 (linear algebra, calculus)
**Time:** ~60 minutes

## 学习目标

- 在 rectangular form 和 polar form 中执行 complex arithmetic（add、multiply、divide、conjugate）
- 应用 Euler's formula，在 complex exponentials 和 trigonometric functions 之间转换
- 使用 complex roots of unity 实现 Discrete Fourier Transform
- 解释 complex rotations 如何支撑 transformers 中的 RoPE 和 sinusoidal positional encodings

## 问题

你打开一篇 Fourier transforms 论文，里面到处都是 `i`。你看 transformer positional encodings，看到不同 frequencies 下的 `sin` 和 `cos`，也就是 complex exponentials 的 real 和 imaginary parts。你读 quantum computing，又发现一切都写在 complex vector spaces 中。

Complex numbers 看起来很抽象。一个建立在 -1 的平方根上的 number system，像是数学 trick。但它不是 trick。它是 rotations 和 oscillations 的自然语言。每当某个东西旋转、振动或 oscillate，complex numbers 都是正确工具。

如果不理解 complex numbers，你就无法理解 Discrete Fourier Transform。无法理解 FFT。无法理解现代 language models 中 RoPE（Rotary Position Embedding）如何工作。也无法理解原始 Transformer paper 中 sinusoidal positional encodings 为什么使用那些 frequencies。

本课会从零构建 complex arithmetic，把它连接到 geometry，并精确展示 complex numbers 在 machine learning 中出现的位置。

## 概念

### 什么是 complex number？

Complex number 有两部分：real part 和 imaginary part。

```text
z = a + bi

where:
  a is the real part
  b is the imaginary part
  i is the imaginary unit, defined by i^2 = -1
```

就是这样。你把 number line 扩展成一个 plane。Real numbers 位于一条 axis 上。Imaginary numbers 位于另一条 axis 上。每个 complex number 都是这个 plane 中的一个 point。

### Complex arithmetic

**Addition。** Real parts 相加，imaginary parts 相加。

```text
(a + bi) + (c + di) = (a + c) + (b + d)i

Example: (3 + 2i) + (1 + 4i) = 4 + 6i
```

**Multiplication。** 使用 distributive law，并记住 i^2 = -1。

```text
(a + bi)(c + di) = ac + adi + bci + bdi^2
                 = ac + adi + bci - bd
                 = (ac - bd) + (ad + bc)i

Example: (3 + 2i)(1 + 4i) = 3 + 12i + 2i + 8i^2
                            = 3 + 14i - 8
                            = -5 + 14i
```

**Conjugate。** 翻转 imaginary part 的符号。

```text
conjugate of (a + bi) = a - bi
```

一个 complex number 与其 conjugate 的乘积永远是 real：

```text
(a + bi)(a - bi) = a^2 + b^2
```

**Division。** 将 numerator 和 denominator 同时乘以 denominator 的 conjugate。

```text
(a + bi) / (c + di) = (a + bi)(c - di) / (c^2 + d^2)
```

这样会消掉 denominator 中的 imaginary part，得到干净的 complex number。

### Complex plane

Complex plane 会把每个 complex number 映射到 2D point。Horizontal axis 是 real axis，vertical axis 是 imaginary axis。

```text
z = 3 + 2i  corresponds to the point (3, 2)
z = -1 + 0i corresponds to the point (-1, 0) on the real axis
z = 0 + 4i  corresponds to the point (0, 4) on the imaginary axis
```

Complex number 同时是一个 point，也是从 origin 出发的 vector。这个 dual interpretation 让 complex numbers 对 geometry 非常有用。

### Polar form

Plane 中的任意 point 都可以用它到 origin 的 distance 和它相对 positive real axis 的 angle 来描述。

```text
z = r * (cos(theta) + i*sin(theta))

where:
  r = |z| = sqrt(a^2 + b^2)     (magnitude, or modulus)
  theta = atan2(b, a)             (phase, or argument)
```

Rectangular form（a + bi）适合 addition。Polar form（r, theta）适合 multiplication。

**Polar form 中的 multiplication。** Magnitudes 相乘，angles 相加。

```text
z1 = r1 * e^(i*theta1)
z2 = r2 * e^(i*theta2)

z1 * z2 = (r1 * r2) * e^(i*(theta1 + theta2))
```

这就是为什么 complex numbers 非常适合 rotations。乘以 magnitude 为 1 的 complex number，就是 pure rotation。

### Euler's formula

Complex exponentials 和 trigonometry 之间的桥梁：

```text
e^(i*theta) = cos(theta) + i*sin(theta)
```

这是本课最重要的公式。当 theta = pi：

```text
e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0i = -1

Therefore: e^(i*pi) + 1 = 0
```

五个 fundamental constants（e、i、pi、1、0）被一个 equation 连接起来。

### 为什么 Euler's formula 对 ML 重要

Euler's formula 说明 `e^(i*theta)` 会随着 theta 变化沿 unit circle 移动。theta = 0 时，你在 (1, 0)。theta = pi/2 时，你在 (0, 1)。theta = pi 时，你在 (-1, 0)。theta = 3*pi/2 时，你在 (0, -1)。完整 rotation 是 theta = 2*pi。

这意味着 complex exponentials 就是 rotations。而 rotations 在 signal processing 和 ML 中无处不在。

### 与 2D rotations 的联系

把 complex number (x + yi) 乘以 e^(i*theta)，会让 point (x, y) 绕 origin 旋转 theta angle。

```text
Rotation via complex multiplication:
  (x + yi) * (cos(theta) + i*sin(theta))
  = (x*cos(theta) - y*sin(theta)) + (x*sin(theta) + y*cos(theta))i

Rotation via matrix multiplication:
  [cos(theta)  -sin(theta)] [x]   [x*cos(theta) - y*sin(theta)]
  [sin(theta)   cos(theta)] [y] = [x*sin(theta) + y*cos(theta)]
```

两者产生完全相同的结果。Complex multiplication 就是 2D rotation。Rotation matrix 只是用 matrix notation 写出来的 complex multiplication。

```mermaid
graph TD
    subgraph "Complex Multiplication = 2D Rotation"
        A["z = x + yi<br/>Point (x, y)"] -->|"multiply by e^(i*theta)"| B["z' = z * e^(i*theta)<br/>Point rotated by theta"]
    end
    subgraph "Equivalent Matrix Form"
        C["vector [x, y]"] -->|"multiply by rotation matrix"| D["[x cos theta - y sin theta,<br/> x sin theta + y cos theta]"]
    end
    B -.->|"same result"| D
```

### Phasors and rotating signals

Complex exponential e^(i*omega*t) 是一个以 angular frequency omega 绕 unit circle 旋转的 point。随着 t 增加，这个 point 会描出 circle。

这个 rotating point 的 real part 是 cos(omega*t)。Imaginary part 是 sin(omega*t)。Sinusoidal signal 是 rotating complex number 的 shadow。

```text
e^(i*omega*t) = cos(omega*t) + i*sin(omega*t)

Real part:      cos(omega*t)    -- a cosine wave
Imaginary part: sin(omega*t)    -- a sine wave
```

这就是 phasor representation。你不再追踪一条扭动的 sine wave，而是追踪一个平滑旋转的 arrow。Phase shifts 变成 angle offsets。Amplitude changes 变成 magnitude changes。Signals 相加变成 vector addition。

### Roots of unity

N-th roots of unity 是 unit circle 上等距分布的 N 个 points：

```text
w_k = e^(2*pi*i*k/N)    for k = 0, 1, 2, ..., N-1
```

当 N = 4，roots 是：1、i、-1、-i（四个 compass points）。
当 N = 8，你会得到四个 compass points 加四个 diagonals。

Roots of unity 是 Discrete Fourier Transform 的基础。DFT 会把 signal 分解成这些 N 个 equally-spaced frequencies 上的 components。

### 与 DFT 的联系

Signal x[0], x[1], ..., x[N-1] 的 Discrete Fourier Transform 是：

```text
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)
```

每个 X[k] 衡量 signal 与第 k 个 root of unity 的相关程度，也就是 frequency k 上的 complex sinusoid。DFT 会把 signal 拆成 N 个 rotating phasors，并告诉你每个 phasor 的 amplitude 和 phase。

### 为什么 i 并不 imaginary

"Imaginary" 这个词是历史意外。Descartes 带着贬义使用它。但 i 并不比 negative numbers 更“虚幻”，就像人们最初也拒绝负数一样。Negative numbers 回答“从 3 中减去什么会得到 5？”Imaginary unit 回答“什么数平方后得到 -1？”

更有用的理解是：i 是 90-degree rotation operator。把 real number 乘以 i 一次，你旋转 90 degrees 到 imaginary axis。再乘以 i 一次（i^2），你又旋转 90 degrees，现在指向 negative real direction。这就是为什么 i^2 = -1。它并不神秘。它是由两个 quarter-turns 组成的 half-turn。

这就是 complex numbers 为什么在 engineering 中无处不在。任何旋转的东西，electromagnetic waves、quantum states、signal oscillations、positional encodings，都天然适合用 complex numbers 描述。

### Complex exponentials vs trigonometric functions

在 Euler's formula 之前，工程师把 signals 写成 A*cos(omega*t + phi)：amplitude A、frequency omega、phase phi。这能用，但 arithmetic 很痛苦。添加两个不同 phases 的 cosines 需要 trigonometric identities。

使用 complex exponentials，同一个 signal 写成 A*e^(i*(omega*t + phi))。添加两个 signals 只是添加两个 complex numbers。Multiplying（modulating）只是乘 magnitudes 并加 angles。Phase shifts 变成 angle additions。Frequency shifts 变成乘以 phasors。

整个 signal processing 领域切换到 complex exponential notation，是因为数学更干净。"Real signal" 永远只是 complex representation 的 real part。Imaginary part 被带着做 bookkeeping，让所有 algebra 自然成立。

### 与 transformers 的联系

**Sinusoidal positional encodings**（原始 Transformer paper）：

```text
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

这些 sin 和 cos pairs 是不同 frequencies 下 complex exponentials 的 real 和 imaginary parts。每个 frequency 为 encoding position 提供不同的 "resolution"。Low frequencies 变化慢（coarse position）。High frequencies 变化快（fine position）。它们合在一起为每个 position 提供独特的 frequency fingerprint。

**RoPE（Rotary Position Embedding）** 更进一步。它显式地用 complex rotation matrices 乘 query 和 key vectors。两个 tokens 之间的 relative position 变成 rotation angle。Attention 使用这些 rotated vectors 计算，使模型通过 complex multiplication 感知 relative position。

| Operation | Algebraic Form | Geometric Meaning |
|-----------|---------------|-------------------|
| Addition | (a+c) + (b+d)i | Plane 中的 vector addition |
| Multiplication | (ac-bd) + (ad+bc)i | Rotate and scale |
| Conjugate | a - bi | 沿 real axis 反射 |
| Magnitude | sqrt(a^2 + b^2) | 到 origin 的 distance |
| Phase | atan2(b, a) | 相对 positive real axis 的 angle |
| Division | multiply by conjugate | Reverse rotation and rescale |
| Power | r^n * e^(i*n*theta) | Rotate n times, scale by r^n |

```mermaid
graph LR
    subgraph "Unit Circle"
        direction TB
        U1["e^(i*0) = 1"] -.-> U2["e^(i*pi/2) = i"]
        U2 -.-> U3["e^(i*pi) = -1"]
        U3 -.-> U4["e^(i*3pi/2) = -i"]
        U4 -.-> U1
    end
    subgraph "Applications"
        A1["Euler's formula:<br/>e^(i*theta) = cos + i*sin"]
        A2["DFT uses roots of unity:<br/>e^(2*pi*i*k/N)"]
        A3["RoPE uses rotation:<br/>q * e^(i*m*theta)"]
    end
    U1 --> A1
    U1 --> A2
    U1 --> A3
```

```figure
roots-of-unity
```

## Build It

### Step 1：Complex class

构建一个 Complex number class，支持 arithmetic、magnitude、phase，以及 rectangular 和 polar forms 之间的 conversion。

```python
import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)
```

### Step 2：Polar conversion and Euler's formula

```python
def to_polar(z):
    return z.magnitude(), z.phase()

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))
```

Verify：`euler(theta).magnitude()` 应该始终是 1.0。`euler(0)` 应该给出 (1, 0)。`euler(pi)` 应该给出 (-1, 0)。

### Step 3：Rotation

把 point (x, y) 旋转 angle theta，就是一次 complex multiplication：

```python
point = Complex(3, 4)
rotated = point * euler(math.pi / 4)
```

Magnitude 保持不变。只有 angle 发生变化。

### Step 4：DFT from complex arithmetic

```python
def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result
```

这是 O(N^2) DFT。每个 output X[k] 都是 signal samples 乘以 roots of unity 后的总和。

### Step 5：Inverse DFT

Inverse DFT 会从 spectrum 重构 original signal。与 forward DFT 相比，只有两个变化：翻转 exponent 中的 sign，并除以 N。

```python
def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result
```

这会给你 perfect reconstruction。应用 DFT，再应用 IDFT，你会在 machine precision 内拿回 original signal。没有信息丢失。

### Step 6：Roots of unity

```python
def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]
```

验证两个 properties：
- 每个 root 的 magnitude 都恰好为 1。
- 所有 N 个 roots 的和为 zero（它们由于 symmetry 相互抵消）。

这些 properties 让 DFT invertible。Roots of unity 构成 frequency domain 的 orthogonal basis。

## Use It

Python 内置支持 complex numbers。Literal `j` 表示 imaginary unit。

```python
z = 3 + 2j
w = 1 + 4j

print(z + w)
print(z * w)
print(abs(z))

import cmath
print(cmath.phase(z))
print(cmath.exp(1j * cmath.pi))
```

对 arrays，numpy 原生处理 complex numbers：

```python
import numpy as np

z = np.array([1+2j, 3+4j, 5+6j])
print(np.abs(z))
print(np.angle(z))
print(np.conj(z))
print(np.real(z))
print(np.imag(z))

signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128))
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(128, d=1/128)
```

## Ship It

运行 `code/complex_numbers.py` 生成 `outputs/skill-complex-arithmetic.md`。

## 练习

1. **Complex arithmetic by hand。** 计算 (2 + 3i) * (4 - i)，并用代码验证。然后计算 (5 + 2i) / (1 - 3i)。在 complex plane 上画出两个结果，并检查 multiplication 是否 rotate and scale 了第一个 number。

2. **Rotation sequence。** 从 point (1, 0) 开始。连续乘以 e^(i*pi/6) 十二次。验证 12 次 multiplication 后回到 (1, 0)。打印每一步 coordinates，确认它们描出 regular 12-gon。

3. **DFT of a known signal。** 创建一个 signal：sin(2*pi*3*t) 和 0.5*sin(2*pi*7*t) 的和，采样 32 points。运行你的 DFT。验证 magnitude spectrum 在 frequencies 3 和 7 处有 peaks，并且 7 处 peak 的高度是 3 处的一半。

4. **Roots of unity visualization。** 计算 8th roots of unity。验证它们的和为 zero。验证任意 root 乘以 primitive root e^(2*pi*i/8) 会得到下一个 root。

5. **Rotation matrix equivalence。** 对 10 个 random angles 和 10 个 random points，验证 complex multiplication 与使用 2x2 rotation matrix 做 matrix-vector multiplication 得到相同结果。打印 maximum numerical difference。

## 关键术语

| Term | What it means |
|------|---------------|
| Complex number | 形如 a + bi 的 number，其中 a 是 real part，b 是 imaginary part，并且 i^2 = -1 |
| Imaginary unit | 数字 i，定义为 i^2 = -1。它在哲学意义上并不 imaginary，而是一个 rotation operator |
| Complex plane | x-axis 为 real、y-axis 为 imaginary 的 2D plane。也叫 Argand plane |
| Magnitude (modulus) | 到 origin 的 distance：sqrt(a^2 + b^2)。写作 \|z\| |
| Phase (argument) | 相对 positive real axis 的 angle：atan2(b, a)。写作 arg(z) |
| Conjugate | 沿 real axis 的 mirror image：a + bi 的 conjugate 是 a - bi |
| Polar form | 把 z 表示为 r * e^(i*theta)，而不是 a + bi。让 multiplication 变简单 |
| Euler's formula | e^(i*theta) = cos(theta) + i*sin(theta)。连接 exponentials 和 trigonometry |
| Phasor | 表示 sinusoidal signal 的 rotating complex number e^(i*omega*t) |
| Roots of unity | 对 k = 0 到 N-1 的 N 个 complex numbers e^(2*pi*i*k/N)。Unit circle 上 N 个 equally spaced points |
| DFT | Discrete Fourier Transform。使用 roots of unity 将 signal 分解成 complex sinusoidal components |
| RoPE | Rotary Position Embedding。使用 complex multiplication 在 transformer attention 中编码 relative position |

## 延伸阅读

- [Visual Introduction to Euler's Formula](https://betterexplained.com/articles/intuitive-understanding-of-eulers-formula/) - 用较少 heavy notation 建立 geometric intuition
- [Su et al.: RoFormer (2021)](https://arxiv.org/abs/2104.09864) - 引入使用 complex rotations 的 Rotary Position Embedding 的论文
- [Vaswani et al.: Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 包含 sinusoidal positional encodings 的原始 Transformer paper
- [3Blue1Brown: Euler's formula with introductory group theory](https://www.youtube.com/watch?v=mvmuCPvRoWQ) - 解释为什么 e^(i*pi) = -1 的 visual explanation
- [Needham: Visual Complex Analysis](https://global.oup.com/academic/product/visual-complex-analysis-9780198534464) - complex numbers 最好的 visual treatment，充满 geometric insight
- [Strang: Introduction to Linear Algebra, Ch. 10](https://math.mit.edu/~gs/linearalgebra/) - linear algebra 和 eigenvalues 语境中的 complex numbers
