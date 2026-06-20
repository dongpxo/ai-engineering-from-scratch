# GPU 设置与云端环境

> 用 CPU 训练很适合学习。真正训练模型时，你需要 GPU。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 是否可用。
- 在 Google Colab 中配置 T4 GPU，用于免费的云端实验。
- 对 CPU 和 GPU 上的矩阵乘法做基准测试，并测量加速比。
- 使用 fp16 经验法则估算你的 VRAM 能容纳的最大模型。

## 问题

Phase 1-3 中的大多数课程在 CPU 上运行都没问题。但一旦开始训练 CNN、transformer 或 LLM（Phase 4+），你就需要 GPU 加速。CPU 上需要 8 小时的训练，GPU 上可能 10 分钟就结束。

你有三种选择：本地 GPU、云端 GPU，或 Google Colab（免费）。

## 概念

```text
你的选择：

1. 本地 NVIDIA GPU
   成本：$0（如果你已经拥有）
   设置：安装 CUDA + cuDNN
   最适合：经常使用、大型数据集

2. Google Colab（免费层）
   成本：$0
   设置：无需设置
   最适合：快速实验、家里没有 GPU

3. 云端 GPU（Lambda、RunPod、Vast.ai）
   成本：$0.20-2.00/hr
   设置：SSH + install
   最适合：认真训练、大模型
```

## Build It

### 选项 1：本地 NVIDIA GPU

检查你是否有可用 GPU：

```bash
nvidia-smi
```

安装带 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 验证 GPU

把本课程中的 notebook 直接上传到 Colab 即可。

### 选项 3：云端 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没关系。

大多数课程都可以在 CPU 上完成。需要 GPU 的课程会明确说明，并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It：GPU vs CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，比较 CPU 和 GPU 时间。
2. 如果你没有 GPU，在 Google Colab 上运行它并比较结果。
3. 检查你有多少 GPU memory，并估算能放下的最大模型（经验法则：fp16 每个参数 2 bytes）。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| CUDA | “GPU programming” | NVIDIA 的并行计算平台，让你可以在 GPU 上运行代码 |
| VRAM | “GPU memory” | GPU 上的视频内存，独立于系统 RAM，会限制模型大小 |
| fp16 | “Half precision” | 16-bit floating point；内存占用是 fp32 的一半，精度损失通常很小 |
| Tensor Core | “Fast matrix hardware” | 用于矩阵乘法的专用 GPU core，比普通 core 快 4-8 倍 |
