# 调试与性能分析

> 最糟糕的 AI bug 不会崩溃。它们会静悄悄地用垃圾数据训练，然后汇报一条漂亮的 loss curve。

**Type:** Build
**Language:** Python
**Prerequisites:** Lesson 1 (Dev Environment), basic PyTorch familiarity
**Time:** ~60 minutes

## 学习目标

- 使用 conditional `breakpoint()` 和 `debug_print`，在训练中途检查 tensor shapes、dtypes 和 NaN values。
- 使用 `cProfile`、`line_profiler` 和 `tracemalloc` profile training loops，找出 bottlenecks。
- 检测常见 AI bugs：shape mismatches、NaN loss、data leakage 和 wrong-device tensors。
- 设置 TensorBoard，可视化 loss curves、weight histograms 和 gradient distributions。

## 问题

AI code 的失败方式和普通代码不同。Web app 会带着 stack trace 崩溃。配置错误的 training loop 会跑 8 小时，烧掉 200 美元 GPU 时间，最后产出一个对每个输入都预测均值的模型。代码从未报错。Bug 可能是 tensor 放在了错误 device 上、忘记 `.detach()`，或 labels 泄漏进了 features。

你需要能在这些 silent failures 浪费时间和算力之前抓住它们的 debugging tools。

## 概念

AI debugging 有三个层级：

```mermaid
graph TD
    L3["3. Training Dynamics<br/>Loss curves, gradient norms, activations"] --> L2
    L2["2. Tensor Operations<br/>Shapes, dtypes, devices, NaN/Inf values"] --> L1
    L1["1. Standard Python<br/>Breakpoints, logging, profiling, memory"]
```

大多数人会直接跳到第 3 层（盯着 TensorBoard 看）。但 80% 的 AI bugs 都活在第 1 层和第 2 层。

## Build It

### Part 1: Print Debugging（是的，它有效）

Print debugging 经常被轻视。它不该被轻视。对 tensor code 来说，一条有目标的 print statement 比逐步走 debugger 更有效，因为你需要一次看到 shapes、dtypes 和 value ranges。

```python
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
```

在每个可疑操作之后调用它。找到 bug 后，移除这些 prints。简单。

### Part 2: Python Debugger（pdb 和 breakpoint）

内置 debugger 在 AI 工作中被低估了。把 `breakpoint()` 放进 training loop，然后交互式检查 tensors。

```python
def training_step(model, batch, criterion, optimizer):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if loss.item() > 100 or torch.isnan(loss):
        breakpoint()

    loss.backward()
    optimizer.step()
```

Debugger 停下来之后，常用命令：

- `p outputs.shape` 检查 shapes
- `p loss.item()` 查看 loss value
- `p torch.isnan(outputs).sum()` 统计 NaNs
- `p model.fc1.weight.grad` 检查 gradients
- `c` 继续，`q` 退出

这就是 conditional debugging。你只在看起来不对的时候停下来。对于 10,000-step training run，这很重要。

### Part 3: Python Logging

当 debugging 超过一次快速检查时，用 logging 替代 print statements。

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
logger.error("NaN loss at step %d, stopping", step)
```

Logging 给你 timestamps、severity levels 和 file output。当训练在凌晨 3 点失败时，你想要的是 log file，而不是已经滚出屏幕的 terminal output。

### Part 4: 为代码片段计时

知道时间花在哪里，是优化的第一步。

```python
import time

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")

with Timer("data loading"):
    batch = next(dataloader_iter)

with Timer("forward pass"):
    outputs = model(batch)

with Timer("backward pass"):
    loss.backward()
```

常见发现：data loading 占训练时间的 60%。修复方式是在 DataLoader 中设置 `num_workers > 0`，不是换更快的 GPU。

### Part 5: cProfile 和 line_profiler

当你需要的不只是 manual timers：

```bash
python -m cProfile -s cumtime train.py
```

这会按 cumulative time 排序展示每个 function call。要做 line-by-line profiling：

```bash
pip install line_profiler
```

```python
@profile
def train_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

# Run with: kernprof -l -v train.py
```

### Part 6: Memory Profiling

#### 使用 tracemalloc 分析 CPU Memory

```python
import tracemalloc

tracemalloc.start()

# your code here
model = build_model()
data = load_dataset()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
for stat in top_stats[:10]:
    print(stat)
```

#### 使用 memory_profiler 分析 CPU Memory

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    raw = read_csv("data.csv")       # watch memory jump here
    processed = preprocess(raw)       # and here
    return processed
```

用 `python -m memory_profiler your_script.py` 运行，可以看到 line-by-line memory usage。

#### 使用 PyTorch 分析 GPU Memory

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

当你遇到 OOM（Out of Memory）：

1. 降低 batch size（永远先试这个）
2. 使用 `torch.cuda.empty_cache()` 释放 cached memory
3. 对大型 intermediates 使用 `del tensor`，随后调用 `torch.cuda.empty_cache()`
4. 使用 mixed precision（`torch.cuda.amp`）把 memory usage 减半
5. 对非常深的 models 使用 gradient checkpointing

### Part 7: 常见 AI Bugs 以及如何抓住它们

#### Shape Mismatch

这是最常见的 bug。Tensor 的 shape 是 `[batch, features]`，但模型期望 `[batch, channels, height, width]`。

```python
def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()
```

用一个 sample batch 运行一次。它会映射出模型中每一次 shape transformation。

#### NaN Loss

NaN loss 表示某些东西爆炸了。常见原因：

- Learning rate 太高
- Custom loss 中除以零
- 对零或负数取 log
- RNNs 中 exploding gradients

```python
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in {name}")
        return True
    return False
```

#### Data Leakage

你的模型在 test set 上达到 99% accuracy。听起来很好。它其实是 bug。

```python
def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both train and test")
        return True
    return False
```

还要检查 temporal leakage：用未来数据预测过去。Split 之前先按 timestamp 排序。

#### Wrong Device

Tensors 在不同 devices 上（CPU vs GPU）会导致 runtime errors。但有时候一个 tensor 静悄悄地留在 CPU 上，而其他都在 GPU 上，训练只是跑得很慢。

```python
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"  WARNING: tensor {i} on {t.device}, model on {model_device}")
```

### Part 8: TensorBoard Basics

TensorBoard 会展示训练过程中内部发生了什么。

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

for step in range(num_steps):
    loss = train_step(model, batch)

    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
```

启动它：

```bash
tensorboard --logdir=runs
```

需要关注什么：

- **Loss not decreasing**：Learning rate 太低，或 model architecture 有问题
- **Loss oscillating wildly**：Learning rate 太高
- **Loss goes to NaN**：Numerical instability（见上面的 NaN section）
- **Train loss decreasing, val loss increasing**：Overfitting
- **Weight histograms collapsing to zero**：Vanishing gradients
- **Gradient histograms exploding**：需要 gradient clipping

### Part 9: VS Code Debugger

对于 interactive debugging，用 `launch.json` 配置 VS Code：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Training",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

点击 gutter 设置 breakpoints。使用 Variables pane 检查 tensor properties。Debug Console 允许你在执行中途运行任意 Python expressions。

这对逐步查看 data preprocessing pipelines 很有用，因为你想看到每一步 transformation。

## Use It

下面这套 debugging workflow 能抓住大多数 AI bugs：

1. **Before training**：用 sample batch 运行 `check_shapes`。确认 input 和 output dimensions 符合预期。
2. **First 10 steps**：对 loss、outputs 和 gradients 使用 `debug_print`。确认没有 NaN，values 位于合理范围。
3. **During training**：记录 loss、learning rate 和 gradient norms。使用 TensorBoard 可视化。
4. **When something breaks**：在 failure point 放 `breakpoint()`。交互式检查 tensors。
5. **For performance**：计时 data loading、forward pass 和 backward pass。如果接近 OOM，就 profile memory。

## Ship It

运行 debugging toolkit script：

```bash
python phases/00-setup-and-tooling/12-debugging-and-profiling/code/debug_tools.py
```

查看 `outputs/prompt-debug-ai-code.md`，那里有一个 prompt，可以帮助诊断 AI-specific bugs。

## 练习

1. 运行 `debug_tools.py`，阅读每个 section 的输出。修改 dummy model 来引入一个 NaN（提示：在 forward pass 中除以零），观察 detector 如何抓住它。
2. 使用 `cProfile` profile 一个 training loop，并找出最慢的 function。
3. 使用 `tracemalloc` 找出 data loading pipeline 中哪一行分配了最多 memory。
4. 为一个简单 training run 设置 TensorBoard，并判断模型是否 overfitting。
5. 在 training loop 中使用 `breakpoint()`。练习从 debugger prompt 中检查 tensor shapes、devices 和 gradient values。
