# Gradient Checkpointing 与 Activation Recomputation

> Backprop 会保留每一个 intermediate activation。到了 70B 参数和 128K context，每个 rank 的 activations 会达到 3 TB。Checkpointing 用 FLOPs 换 memory：不保存，而是重新计算。问题是应该丢掉哪些 segments，答案不是“全部丢掉”。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题

训练 transformer 时，每一层都要保存 backward 会求导的每个 op 所需的 inputs：attention inputs、Q/K/V projections、softmax output、FFN inputs、norm outputs 和 residual stream。对于 hidden size 为 `d`、sequence length 为 `L`、batch 为 `B` 的一层，这大约是每层 `12 * B * L * d` 个 floats。

当 `d=8192, L=8192, B=1` 时，BF16 下每层就是 800 MB。64-layer model 是 51 GB activations；这还没乘上 microbatch size，没加 attention-softmax intermediates（每个 head 是 `L^2`），也没考虑 tensor-parallel partial copies。

账单是两面的：BF16 weights 加 optimizer state 可能能塞进 80GB，但 activations 会把你推过上限。Gradient checkpointing，也叫 activation recomputation，是标准修复方法。丢掉大多数 activations；在 backward 期间重做 forward，把它们拿回来。代价：额外 FLOPs。收益：memory 按 checkpoint segments 与 total layers 的比例下降。

朴素地做，checkpointing 每个 step 大约会多花 33% forward-pass FLOPs。做得好，也就是按 Korthikanti et al. 的 “smart selection” 做 selective checkpointing，你可以用低于 5% 的 FLOP overhead 换来 5x memory savings。再叠加 FP8 matmuls、FSDP offload 和 expert-parallel MoE，这真的很重要：memory 和浪费的 compute 你都承担不起。

## 概念

### Backward 实际需要什么

`output = layer(input)`。Backward 想要 `grad_input` 和 `grad_params`。为了计算它们，它需要：

- `input`，用于计算 linear layers 中的 `grad_params = input.T @ grad_output`
- 一些 activation derivative intermediates，比如 ReLU/GELU/softmax 的导数依赖 activation value

Forward pass 会自动把这些保存在 autograd graph 中。每个 `tensor.retain_grad()`，以及每个需要 input 的 op，都会保留一个引用。

### Naive Full Checkpointing

把 network 拆成 `N` 个 segments。Forward 期间只保存每个 segment 的 *input*。当 backward 需要 intermediates 时，重新运行该 segment 的 forward pass，把它们 materialize 出来，然后再求导。

例子：32-layer transformer 拆成 32 个 segments，每个 segment 1 层。

- Memory：保存 32 个 layer-inputs（小）vs 保存 `32 * 每层 activation volume`（巨大）。
- Extra compute：每个 segment 多做 1 次 forward，也就是总共约多 33% forward FLOPs。因为 backward 是 2x forward，完整 step 从 `1 + 2 = 3` 个单位变成 `1 + 1 + 2 = 4` 个单位。

这是 Chen et al. 2016 的原始方案：每 `sqrt(L)` 层放一个 checkpoint，用来平衡 memory 和 compute。对于 L=64，也就是 8 个 checkpoints。

### Selective Checkpointing（Korthikanti 2022）

并不是所有 activations 的成本都一样。Attention softmax output 是 `B*L*L*heads`，随 sequence length *二次增长*。FFN hidden activation 是 `B*L*4d`，线性增长。长序列时，softmax 占主导。

Selective checkpointing 保留便宜的 activations，比如 linear projections 和 residuals，只重新计算昂贵的 activations，比如 attention。你付出很少 FLOPs 来 recompute，却省下 O(L^2) memory。

Megatron-Core 把它实现为 “selective” activation recomputation。多数 2024+ frontier training runs 都在用它。

### Offload

除了 recompute，还有一种替代方案：在 forward 和 backward 之间把 activations 发送到 CPU RAM。它需要 PCIe bandwidth；当 idle bandwidth 超过 rematerialization 成本时有收益。混合策略很常见：一些 layers checkpoint，另一些 offload。

FSDP2 把 offload 作为 first-class option。GPU 被 memory 卡住、但 CPU-GPU transfer 还有余量时，offload 特别有效。

### Recompute Cost Model

在 `L` 层中每 `k` 层做一次 naive checkpointing 时，每 step FLOPs：

```text
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Selective checkpointing 只 recompute attention kernel，而不是整个 layer：

```text
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Memory Savings Model

每层 activation volume：`A`。对于 `L` 层，总 activation memory：`L * A`。

Full checkpoint（segment size 1）：只保存 `L * input_volume`，对标准 transformer 来说约为 `L * 1/10 A`。节省约 `9 * L * A * 1/10`。

每 `k` 层 checkpoint：保存 `L/k * A`，再加上 active segment 内 `k-1` 层的 activations。

当 `k = sqrt(L)` 时，memory 和 recompute cost 都按 `sqrt(L)` 缩放。这是 uniform-cost layers 下的最优取舍。

### 什么时候不要 Checkpoint

- Pipeline stage 内已经 in-flight 的最内层。它们无论如何都要完成。
- 如果第一层和最后一层主导了 stage compute，就不要 checkpoint 它们；不过这在 transformers 里很少见。
- 已经使用 FlashAttention 的 attention kernels。Flash 本身已经快速 recompute softmax，所以额外 layer-level checkpointing 的收益不大。

### Implementation Patterns

1. **Function wrapper：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包裹一个 segment。PyTorch 只保存 `input`，在 backward 时 recompute 其他所有东西。

2. **Decorator-based：** 把 layers 标记为 checkpointable；trainer 在 config time 决定哪些 segments 被 wrapper 包起来。

3. **Manual explicit recompute：** 自己写 backward pass，调用一个 custom `recompute_forward`，用保存的 input 复制 forward。

三种方式的函数结果相同。Wrappers 是标准 idiom。

### 与 TP / PP / FP8 的交互

- **Tensor parallel：** checkpoint inputs 在 recompute 时必须 gather 或 rescatter；要处理 communication cost。
- **Pipeline parallel：** 典型模式是 checkpoint 每个 pipeline-stage 的 forward，这样 reverse-order microbatches 可以复用 activation memory。
- **FP8 recompute：** recompute 期间更新的 amax histories 必须匹配原始 forward，否则 FP8 scale 会漂移。多数 frameworks 会 snapshot scale。

## 动手构建

### Step 1：带 Segments 的 Toy Model

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Step 2：需要所有 Activations 的 Naive Backward

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Step 3：Checkpoint-Every-k Memory

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Step 4：Cost Model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Step 5：Memory Estimator

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Step 6：Optimal Segment Size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Step 7：Selective Checkpoint Decision

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`，PyTorch 中的 canonical wrapper。它包裹一个 function，只保存 inputs，在 backward 时 recompute。
- **Megatron-Core activation recomputation**：支持 `selective`、`full` 和 `block` modes。2024+ frontier training 中的标准做法。
- **FSDP2 offload**：FSDP2 shards 中使用带 `offload_policy` 的 `module.to_empty(device="cpu")`，把 activations shard 到 CPU，而不是 recompute。
- **DeepSpeed ZeRO-Offload**：为 optimizer states 和 activations 提供 CPU offload，与 checkpointing 互补。

## 交付它

本课会产出 `outputs/prompt-activation-recompute-policy.md`，这是一个 prompt：输入你的 model config（layers、hidden、seq、batch）和可用 GPU memory，它会输出 per-layer recompute policy（none / selective / full / offload）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（full activations），再运行 `model_forward_checkpointed` + `model_backward_checkpointed`（segments）。Parameter gradients 必须在 machine precision 内完全一致。

2. 从 1 到 `L` sweep segment size `k`。绘制 FLOP overhead 和 memory。找到曲线的 knee。

3. 实现 selective checkpointing：保存 attention-module input，但不保存其 intermediates。对一个 32-layer model，在 `seq=8192` 下测量它相对 full-layer checkpointing 的 FLOP overhead。

4. 添加 offload。把 segment inputs 保存到模拟的 “CPU buffer”（一个单独 list）。把 “PCIe bandwidth” 计为 bytes/time，并找出 offload 和 recompute 的 breakeven point。

5. 用真实 PyTorch transformer benchmark 有无 `torch.utils.checkpoint` 的差异。测量 memory（通过 `torch.cuda.max_memory_allocated`）和 step time。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | “通过重做 forward 省 memory” | 只保存 segment inputs；在 backward 期间 recompute intermediates，以获得支持 gradient 的 tensors |
| Activation recomputation | “和 checkpointing 一样” | 同一技术在 HPC 语境下的名字 |
| Segment size (k) | “每个 checkpoint 覆盖多少层” | 其 intermediates 会被丢弃并一起 rematerialized 的层数 |
| Selective checkpointing | “Korthikanti 的技巧” | 只 recompute 存储昂贵的 activations（attention softmax），保留便宜的部分 |
| Full checkpointing | “朴素版本” | Recompute 每个 segment 中每一层的 intermediates |
| Block checkpointing | “Coarse-grained” | Checkpoint 整个 transformer blocks；粒度最大 |
| FLOP overhead | “compute tax” | 每 step 额外 FLOPs = `(recompute FLOPs) / (fwd + bwd FLOPs)`；naive 约 33%，selective 约 5% |
| Activation offload | “发到 CPU” | 在 forward 到 backward 之间把 activations 移到 CPU RAM；是 recompute 的替代方案 |
| sqrt-L rule | “经典最优值” | 对 uniform-cost layers，最优 checkpoint spacing 是 `sqrt(L)` 层 |
| Attention-softmax volume | “O(L^2) 问题” | `L^2 * heads * batch` 个 floats；在 long contexts 下主导 activation memory |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- formalize gradient checkpointing 的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selective activation recomputation 和形式化 cost analysis
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过 reverse-mode rematerialization 实现 constant-memory 的替代方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模 activation offload
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 和 block modes
