# Fourier 变换

> 每个 signal 都是一组 sine waves 的和。Fourier transform 会告诉你是哪几组。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04, 19 (complex numbers)
**Time:** ~90 minutes

## 学习目标

- 从零实现 DFT，并将它与 O(N log N) 的 Cooley-Tukey FFT 验证对齐
- 解释 frequency coefficients：从 signal 中提取 amplitude、phase 和 power spectrum
- 应用 convolution theorem，通过 FFT multiplication 执行 convolution
- 将 Fourier frequency decomposition 连接到 transformer positional encodings 和 CNN convolution layers

## 问题

一段 audio recording 是 pressure measurements 随时间变化的 sequence。Stock price 是 days 上的 values sequence。Image 是 space 上 pixel intensities 的 grid。这些都是 time domain（或 space domain）中的 data。你看到的是某个 index 上不断变化的 values。

但很多 patterns 在 time domain 中不可见。这个 audio signal 是 pure tone 还是 chord？这个 stock price 是否有 weekly cycle？这张 image 是否有 repeating texture？这些问题都在问 frequency content，而 time domain 会把它藏起来。

Fourier transform 会把 data 从 time domain 转换到 frequency domain。它拿到一个 signal，并把它 decomposes 成不同 frequencies 的 sine waves。每个 sine wave 都有 amplitude（有多强）和 phase（从哪里开始）。Fourier transform 会同时告诉你这两者。

这对 ML 很重要，因为 frequency-domain thinking 到处都在出现。Convolutional neural networks 执行 convolution，而 convolution 在 frequency domain 中就是 multiplication。Transformer positional encodings 使用 frequency decomposition 表示 position。Audio models（speech recognition、music generation）处理 spectrograms，也就是 sound 的 frequency representations。Time series models 寻找 periodic patterns。理解 Fourier transform 会给你处理这些内容所需的 vocabulary。

## 概念

### DFT definition

给定 N 个 samples x[0], x[1], ..., x[N-1]，Discrete Fourier Transform 会产生 N 个 frequency coefficients X[0], X[1], ..., X[N-1]：

```text
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

for k = 0, 1, ..., N-1
```

每个 X[k] 都是 complex number。它的 magnitude |X[k]| 告诉你 frequency k 的 amplitude。它的 phase angle(X[k]) 告诉你该 frequency 的 phase offset。

关键洞察：`e^(-2*pi*i*k*n/N)` 是 frequency k 上的 rotating phasor。DFT 计算 signal 与 N 个 equally-spaced frequencies 的 correlation。如果 signal 在 frequency k 上有 energy，correlation 就大。如果没有，它就接近 zero。

### 每个 coefficient 的含义

**X[0]：DC component。** 这是所有 samples 的和，与 mean 成比例。它表示 signal 的 constant（zero-frequency）offset。

```text
X[0] = sum_{n=0}^{N-1} x[n] * e^0 = sum of all samples
```

**X[k] for 1 <= k <= N/2：positive frequencies。** X[k] 表示每 N 个 samples 中 k 个 cycles 的 frequency。k 越高，frequency 越高（oscillation 越快）。

**X[N/2]：Nyquist frequency。** 这是用 N 个 samples 能表示的最高 frequency。超过这个频率，就会出现 aliasing：high frequencies 会伪装成 low frequencies。

**X[k] for N/2 < k < N：negative frequencies。** 对 real-valued signals，X[N-k] = conj(X[k])。Negative frequencies 是 positive frequencies 的 mirror images。这就是为什么有用信息位于前 N/2 + 1 个 coefficients 中。

### Inverse DFT

Inverse DFT 会从 frequency coefficients 重构 original signal：

```text
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)

for n = 0, 1, ..., N-1
```

它与 forward DFT 只有两个差异：exponent 中的 sign 是 positive（不是 negative），并且有 1/N normalization factor。

Inverse DFT 是 perfect reconstruction。没有信息丢失。你可以从 time domain 到 frequency domain，再回来，不产生任何 error。DFT 是 change of basis：它把同样的信息重新表达在不同 coordinate system 中。

### FFT：让它变快

按上面的定义计算 DFT 是 O(N^2)：对 N 个 output coefficients 中的每一个，都要对 N 个 input samples 求和。N = 1 million 时，这就是 10^12 operations。

Fast Fourier Transform（FFT）会用 O(N log N) 计算同样的结果。N = 1 million 时，大约是 20 million operations，而不是一万亿。这让 frequency analysis 变得 practical。

Cooley-Tukey algorithm（最常见的 FFT）使用 divide and conquer：

1. 把 signal 分成 even-indexed 和 odd-indexed samples。
2. 递归计算每一半的 DFT。
3. 使用 "twiddle factors" e^(-2*pi*i*k/N) 合并两个 half-size DFTs。

```text
X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]          for k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - e^(-2*pi*i*k/N) * O[k]    for k = 0, ..., N/2 - 1

where E = DFT of even-indexed samples
      O = DFT of odd-indexed samples
```

这种 symmetry 意味着 recursion 的每一层做 O(N) work，并且有 log2(N) 层。Total：O(N log N)。

```mermaid
graph TD
    subgraph "8-point FFT (Cooley-Tukey)"
        X["x[0..7]<br/>8 samples"] -->|"split even/odd"| E["Even: x[0,2,4,6]"]
        X -->|"split even/odd"| O["Odd: x[1,3,5,7]"]
        E -->|"4-pt FFT"| EK["E[0..3]"]
        O -->|"4-pt FFT"| OK["O[0..3]"]
        EK -->|"combine with twiddle factors"| XK["X[0..7]"]
        OK -->|"combine with twiddle factors"| XK
    end
    subgraph "Complexity"
        C1["DFT: O(N^2) = 64 multiplications"]
        C2["FFT: O(N log N) = 24 multiplications"]
    end
```

FFT 要求 signal length 是 power of 2。实践中，signals 会 zero-pad 到下一个 power of 2。

### Spectral analysis

**Power spectrum** 是 |X[k]|^2，也就是每个 frequency coefficient 的 squared magnitude。它展示每个 frequency 上有多少 energy。

**Phase spectrum** 是 angle(X[k])，也就是每个 frequency 的 phase offset。对大多数 analysis tasks，你关心 power spectrum，而忽略 phase。

```text
Power at frequency k:  P[k] = |X[k]|^2 = X[k].real^2 + X[k].imag^2
Phase at frequency k:  phi[k] = atan2(X[k].imag, X[k].real)
```

### Frequency resolution

DFT 的 frequency resolution 取决于 samples 数量 N 和 sampling rate fs。

```text
Frequency of bin k:      f_k = k * fs / N
Frequency resolution:    delta_f = fs / N
Maximum frequency:       f_max = fs / 2  (Nyquist)
```

要分辨两个很接近的 frequencies，你需要更多 samples。要捕捉 high frequencies，你需要更高 sampling rate。

### Convolution theorem

这是 signal processing 中最重要的结果之一，并且与 CNNs 直接相关。

**Time domain 中的 convolution 等于 frequency domain 中的 pointwise multiplication。**

```text
x * h = IFFT(FFT(x) . FFT(h))

where * is convolution and . is element-wise multiplication
```

为什么这重要：

- 两个长度为 N 和 M 的 signals 做 direct convolution 需要 O(N*M) operations。
- FFT-based convolution 需要 O(N log N)：transform both，multiply，transform back。
- 对 large kernels，FFT convolution 快得多。
- 这正是 large receptive fields 的 convolutional layers 中发生的事情。

注意：DFT 计算的是 circular convolution（signal 会 wrap around）。要做 linear convolution（不 wraparound），先把两个 signals zero-pad 到长度 N + M - 1。

```mermaid
graph LR
    subgraph "Time Domain"
        TA["Signal x[n]"] -->|"convolve (slow: O(NM))"| TC["Output y[n]"]
        TB["Filter h[n]"] -->|"convolve"| TC
    end
    subgraph "Frequency Domain"
        FA["FFT(x)"] -->|"multiply (fast: O(N))"| FC["FFT(x) * FFT(h)"]
        FB["FFT(h)"] -->|"multiply"| FC
        FC -->|"IFFT"| FD["y[n]"]
    end
    TA -.->|"FFT"| FA
    TB -.->|"FFT"| FB
    FD -.->|"same result"| TC
```

### Windowing

DFT 假设 signal 是 periodic 的，也就是把 N 个 samples 当作一个 infinitely repeating signal 的一个 period。如果 signal 的起点和终点不是同一个 value，那么 boundary 会出现 discontinuity，并表现为 spurious high-frequency content。这叫 spectral leakage。

Windowing 会在计算 DFT 前，把 signal 两端逐渐 taper 到 zero，以减少 leakage。

常见 windows：

| Window | Shape | Main lobe width | Side lobe level | Use case |
|--------|-------|----------------|-----------------|----------|
| Rectangular | Flat（no window） | Narrowest | Highest (-13 dB) | Signal 在 N samples 中恰好 periodic |
| Hann | Raised cosine | Moderate | Low (-31 dB) | 通用 spectral analysis |
| Hamming | Modified cosine | Moderate | Lower (-42 dB) | Audio processing、speech analysis |
| Blackman | Triple cosine | Wide | Very low (-58 dB) | Side lobe suppression 很关键时 |

```text
Hann window:    w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
```

在 DFT 前，将 window 与 signal element-wise 相乘：`X = DFT(x * w)`。

### DFT properties

| Property | Time Domain | Frequency Domain |
|----------|-------------|-----------------|
| Linearity | a*x + b*y | a*X + b*Y |
| Time shift | x[n - k] | X[f] * e^(-2*pi*i*f*k/N) |
| Frequency shift | x[n] * e^(2*pi*i*f0*n/N) | X[f - f0] |
| Convolution | x * h | X * H (pointwise) |
| Multiplication | x * h (pointwise) | X * H (circular convolution, scaled by 1/N) |
| Parseval's theorem | sum \|x[n]\|^2 | (1/N) * sum \|X[k]\|^2 |
| Conjugate symmetry (real input) | x[n] real | X[k] = conj(X[N-k]) |

Parseval's theorem 说明 total energy 在两个 domains 中相同。Energy 在 transform 中被 conserved。

### 与 positional encodings 的联系

原始 Transformer 使用 sinusoidal positional encodings：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

每个 dimension pair（2i, 2i+1）以不同 frequency oscillate。Frequencies 从 high（dimension 0,1）到 low（最后几个 dimensions）按 geometric spacing 分布。这给每个 position 一个跨 frequency bands 的 unique pattern，类似 Fourier coefficients 如何唯一识别一个 signal。

它提供的关键 properties：

- **Uniqueness：** 没有两个 positions 拥有相同 encoding。
- **Bounded values：** sin 和 cos 始终在 [-1, 1]。
- **Relative position：** Position p+k 的 encoding 可以表达为 position p encoding 的 linear function。Model 可以学习关注 relative positions。

### 与 CNNs 的联系

Convolution layer 通过在 signal 或 image 上滑动 learned filter（kernel）来作用于 input。数学上，这就是 convolution operation。

根据 convolution theorem，它等价于：
1. FFT the input
2. FFT the kernel
3. 在 frequency domain 中 multiply
4. IFFT the result

Standard CNN implementations 使用 direct convolution（对小型 3x3 kernels 更快）。但对 large kernels 或 global convolution，FFT-based approaches 快得多。一些 architectures（如 FNet）完全用 FFT 替代 attention，以 O(N log N) 而不是 O(N^2) complexity 达到有竞争力的 accuracy。

### Spectrograms and the Short-Time Fourier Transform

单次 FFT 会给出整个 signal 的 frequency content，但不会告诉你这些 frequencies 什么时候出现。一个 chirp（frequency 随时间升高的 signal）和一个 chord（所有 frequencies 同时存在）可能有相同的 magnitude spectrum。

Short-Time Fourier Transform（STFT）通过在 signal 的 overlapping windows 上计算 FFT 来解决这个问题。结果是 spectrogram：一个 2D representation，一条 axis 是 time，另一条 axis 是 frequency。每个点的 intensity 表示该 time、该 frequency 上的 energy。

```text
STFT procedure:
1. Choose a window size (e.g., 1024 samples)
2. Choose a hop size (e.g., 256 samples -- 75% overlap)
3. For each window position:
   a. Extract the windowed segment
   b. Apply a Hann/Hamming window
   c. Compute FFT
   d. Store the magnitude spectrum as one column of the spectrogram
```

Spectrograms 是 audio ML models 的标准 input representation。Speech recognition models（Whisper、DeepSpeech）处理 mel-spectrograms，也就是把 frequencies 映射到 mel scale 的 spectrograms，它更符合 human pitch perception。

### Aliasing

如果 signal 包含高于 fs/2（Nyquist frequency）的 frequencies，以 rate fs 采样时会产生 aliased copies。一个 90 Hz signal 以 100 Hz 采样，看起来会与 10 Hz signal 完全相同。仅凭 samples 无法区分它们。

```text
Example:
  True signal: 90 Hz sine wave
  Sampling rate: 100 Hz
  Apparent frequency: 100 - 90 = 10 Hz

  The samples from the 90 Hz signal at 100 Hz sampling rate
  are identical to the samples from a 10 Hz signal.
  No amount of math can recover the original 90 Hz.
```

这就是为什么 analog-to-digital converters 会包含 anti-aliasing filters，在 sampling 前移除 Nyquist 以上的 frequencies。在 ML 中，如果 downsampling feature maps 时没有 proper low-pass filtering，就会出现 aliasing。一些 architectures 用 anti-aliased pooling layers 解决这个问题。

### Zero-padding does not increase resolution

一个常见误解：FFT 前 zero-padding signal 会提高 frequency resolution。不会。Zero-padding 只是在已有 frequency bins 之间做 interpolation，让 spectrum 看起来更平滑。但它无法揭示 original samples 中不存在的 frequency detail。

真正的 frequency resolution 只取决于 observation time T = N / fs。要分辨相隔 delta_f 的两个 frequencies，你至少需要 T = 1 / delta_f seconds 的 data。任何 zero-padding 都无法改变这个 fundamental limit。

```figure
fourier-synthesis
```

## Build It

### Step 1：DFT from scratch

O(N^2) DFT 直接来自 definition。

```python
import math

class Complex:
    ...

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    return result
```

### Step 2：Inverse DFT

结构相同，positive exponent，并除以 N。

```python
def idft(X):
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * w
        result.append(Complex(total.real / N, total.imag / N))
    return result
```

### Step 3：FFT（Cooley-Tukey）

Recursive FFT 要求长度是 power-of-2。Split into even and odd，recurse，再用 twiddle factors combine。

```python
def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result
```

### Step 4：Spectral analysis helpers

```python
def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]

def convolve_fft(x, h):
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = idft(Y)
    return [y[n].real for n in range(N)]
```

## Use It

真实工作中，使用 numpy 的 FFT，它由高度优化的 C libraries 支撑。

```python
import numpy as np

signal = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(256, d=1/256)

power = np.abs(spectrum) ** 2

positive_freqs = freqs[:len(freqs)//2]
positive_power = power[:len(power)//2]
```

Windowing 和更高级的 spectral analysis：

```python
from scipy.signal import windows, stft

window = windows.hann(256)
windowed = signal * window
spectrum = np.fft.fft(windowed)
```

Convolution：

```python
from scipy.signal import fftconvolve

result = fftconvolve(signal, kernel, mode='full')
```

Spectrograms：

```python
from scipy.signal import stft

frequencies, times, Zxx = stft(signal, fs=sample_rate, nperseg=256)
spectrogram = np.abs(Zxx) ** 2
```

Spectrogram matrix 的 shape 是 (n_frequencies, n_time_frames)。每一列是一个 time window 的 power spectrum。这就是 audio ML models 消费的 input。

## Ship It

运行 `code/fourier.py` 生成 `outputs/prompt-spectral-analyzer.md`。

## 练习

1. **Pure tone identification。** 创建一个 signal：unknown frequency（1 到 50 Hz 之间）的单个 sine wave，以 128 Hz 采样 1 second。使用你的 DFT 识别 frequency。验证答案匹配。现在加入 standard deviation 0.5 的 Gaussian noise 并重复。Noise 如何影响 spectrum？

2. **FFT vs DFT verification。** 生成一个 length 64 的 random signal。分别计算 DFT（O(N^2)）和 FFT。验证所有 coefficients 在 1e-10 以内匹配。对 length 256、512、1024、2048 的 signals 计时两个 functions。绘制 DFT time 与 FFT time 的 ratio。

3. **Convolution theorem proof by example。** 创建 signal x = [1, 2, 3, 4, 0, 0, 0, 0] 和 filter h = [1, 1, 1, 0, 0, 0, 0, 0]。直接计算它们的 circular convolution（nested loop）。再通过 FFT 计算（transform、multiply、inverse transform）。验证结果匹配。然后通过合适的 zero-padding 做 linear convolution。

4. **Windowing effects。** 创建一个 signal：10 Hz 和 12 Hz（非常接近）的两个 sine waves 之和。以 128 Hz 采样 1 second。分别在 no window、Hann window、Hamming window 下计算 power spectrum。哪个 window 最容易区分两个 peaks？为什么？

5. **Positional encoding analysis。** 为 d_model = 128 和 max_pos = 512 生成 sinusoidal positional encodings。对每一对 positions (p1, p2)，计算它们 encodings 的 dot product。展示 dot product 只依赖 |p1 - p2|，不依赖 absolute positions。距离增加时 dot product 会怎样？

## 关键术语

| Term | What it means |
|------|---------------|
| DFT (Discrete Fourier Transform) | 将 N 个 time-domain samples 转换成 N 个 frequency-domain coefficients。每个 coefficient 是与该 frequency 上 complex sinusoid 的 correlation |
| FFT (Fast Fourier Transform) | 计算 DFT 的 O(N log N) algorithm。Cooley-Tukey algorithm 会递归 split even/odd indices |
| Inverse DFT | 从 frequency coefficients 重构 time-domain signal。公式与 DFT 相同，但 exponent sign 翻转，并乘以 1/N scaling |
| Frequency bin | DFT output 中的每个 index k 表示 frequency k*fs/N Hz。"bin" 是 discrete frequency slot |
| DC component | X[0]，zero-frequency coefficient。与 signal mean 成比例 |
| Nyquist frequency | fs/2，在 sampling rate fs 下可表示的 maximum frequency。更高 frequencies 会 alias |
| Power spectrum | \|X[k]\|^2，每个 frequency coefficient 的 squared magnitude。展示 energy 在 frequencies 上的分布 |
| Phase spectrum | angle(X[k])，每个 frequency component 的 phase offset。Analysis 中通常忽略 |
| Spectral leakage | 把 non-periodic signal 当作 periodic 处理导致的 spurious frequency content。可通过 windowing 减少 |
| Window function | DFT 前应用的 tapering function（Hann、Hamming、Blackman），用于减少 spectral leakage |
| Twiddle factor | FFT butterfly computation 中用于 combine sub-DFTs 的 complex exponential e^(-2*pi*i*k/N) |
| Convolution theorem | Time domain 中的 convolution 等于 frequency domain 中的 pointwise multiplication。Signal processing 和 CNNs 的基础 |
| Circular convolution | Signal 会 wrap around 的 convolution。这是 DFT 自然计算的形式 |
| Linear convolution | 没有 wraparound 的 standard convolution。通过 DFT 前 zero-padding 实现 |
| Parseval's theorem | Total energy 在 Fourier transform 中保持不变。sum \|x[n]\|^2 = (1/N) sum \|X[k]\|^2 |
| Aliasing | Sampling rate 不足时，高于 Nyquist 的 frequencies 表现为 lower frequencies |

## 延伸阅读

- [Cooley & Tukey: An Algorithm for the Machine Calculation of Complex Fourier Series (1965)](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/) - 改变 computing 的原始 FFT paper
- [3Blue1Brown: But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) - Fourier transforms 最好的 visual introduction
- [Lee-Thorp et al.: FNet: Mixing Tokens with Fourier Transforms (2021)](https://arxiv.org/abs/2105.03824) - 在 transformers 中用 FFT 替代 self-attention
- [Smith: The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/) - 免费在线教材，深入覆盖 FFT、windowing 和 spectral analysis
- [Vaswani et al.: Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 从 Fourier frequency decomposition 派生的 sinusoidal positional encodings
- [Radford et al.: Whisper (2022)](https://arxiv.org/abs/2212.04356) - 使用 mel-spectrograms 作为 input representation 的 speech recognition
