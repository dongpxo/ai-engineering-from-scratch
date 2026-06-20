# Attention 变体：Sliding Window、Sparse、Differential

> Full attention 是一个圆。每个 token 都能看到每个 token，memory 为此付出代价。四种变体改变这个圆的形状，并拿回一半成本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## 问题

Full attention 在 sequence length 上需要 `O(N²)` memory 和 `O(N²)` compute。对一个 128K-context Llama 3 70B 来说，每层就是 160 亿个 attention entries，再乘以 80 层。Flash Attention（Lesson 12）隐藏了 `O(N²)` activation memory，但没有改变算术成本：每个 token 仍然 attend 到每个其他 token。

三类变体会改变 attention matrix 自身的 topology：

1. **Sliding window attention（SWA）。** 每个 token 只 attend 到固定窗口内的邻居，而不是完整 prefix。Memory 和 compute 降到 `O(N · W)`，其中 `W` 是 window。代表：Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **Sparse / block attention。** 只有选中的 `(i, j)` pairs 会被打分；其余位置被强制为 zero weight。代表：Longformer、BigBird、OpenAI sparse transformer。
3. **Differential attention。** 用独立的 Q/K projections 计算两张 attention maps，然后相减。它会消除把权重泄漏到前几个 tokens 的 "attention sink"。代表：Microsoft 的 DIFF Transformer（2024）。

这些方案可以共存。2026 年的 frontier model 往往会混用：大多数 layers 是 SWA-1024，每五层有一层 global full attention，还有少数 differential heads 用来清理 retrieval。Gemma 3 的 5:1 SWA-to-global ratio 是当前教科书默认值。

## 概念

### Sliding Window Attention（SWA）

位置 `i` 的每个 query 只 attend 到 `[i - W, i]`（causal SWA）或 `[i - W/2, i + W/2]`（bidirectional）中的位置。窗口外 tokens 在 score matrix 中得到 `-inf`。

```text
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

当 `N = 8192` 且 `W = 1024` 时，score matrix 期望上只有 1024 × 8192 个 non-zero rows，相当于减少 8 倍。

**KV cache 会随 SWA 变小。** 每层只需要保留最后 `W` 个 tokens 的 K 和 V。对一个 Gemma-3-ish config（1024 window，128K context）来说，KV cache 会下降 128 倍。

**质量成本。** SWA-only transformers 在 long-range retrieval 上会吃力。修复方式：把 SWA layers 和 full-attention layers 交错。Gemma 3 使用 5:1 SWA:global。Mistral 7B 使用 causal-SWA stack，让信息通过重叠窗口向前“流动”：每一层都会把 effective receptive field 扩展 `W`，`L` 层后模型可以 attend 到 `L × W` tokens 之前。

### Sparse / Block Attention

预先选择一个 `N × N` sparsity pattern。三个 canonical shapes：

- **Local + strided（OpenAI sparse transformer）。** Attend 到最近 `W` 个 tokens，再加上之前每隔 `stride` 的 token。以 `O(N · sqrt(N))` compute 捕捉 local 和 long-range。
- **Longformer / BigBird。** Local window + 一小组 global tokens（例如 `[CLS]`），这些 tokens attend 到所有位置、也被所有位置 attend + random-sparse links。在同等质量下经验上能得到 2 倍 context。
- **Native Sparse Attention（DeepSeek，2025）。** 学习哪些 `(Q, K)` blocks 重要；在 kernel level 跳过 zero blocks。兼容 FlashAttention。

Sparse attention 是 kernel-engineering story。数学很简单（mask score matrix）；收益来自从不把 zero entries load 到 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 让 custom sparse patterns 成为 PyTorch 中的一等公民。

### Differential Attention（DIFF Transformer，2024）

Regular attention 有一个 "attention sink" 问题：softmax 强制每一行 sum to 1，所以那些没有特别想 attend 到任何内容的 tokens，会把权重倒到第一个 token（或前几个 tokens）上。这会偷走本该给真实内容的 capacity。

Differential attention 通过计算**两张** attention maps 并相减来修复：

```text
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是 learned scalar（通常 0.5-0.8）。A1 捕捉真实内容权重；A2 捕捉 sink。相减会抵消 sink，把权重重新分配给 relevant tokens。

Reported results（Microsoft 2024）：perplexity 降低 5-10%，同等训练长度下 effective context 延长 1.5-2 倍，needle-in-haystack retrieval 更尖锐。

### Variant Comparison

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | every model's default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2x context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

```figure
gqa-kv-sharing
```

## Build It

见 `code/main.py`。我们实现一个 causal mask comparator，在 toy sequence 上并排展示 full、SWA、local+strided 和 differential attention。

### Step 1：Full causal mask（baseline）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自 Lesson 07 的 baseline。Lower triangular；diagonal 上方权重为 zero。

### Step 2：Sliding window causal mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

只有一个参数：`window`。当 `window >= n` 时，你会恢复 full causal attention。当 `window = 1` 时，每个 token 只 attend 到自己。

### Step 3：Local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window，加上从 sequence 开头到当前位置、每隔 `stride` 的 token。随着额外 layers 增加，receptive field 会以 log steps 增长。

### Step 4：Differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次 attention pass，再用 learned mixing coefficient 相减。在代码中，我们比较 single vs differential 的 attention-sink heatmap，观察 sink 如何塌陷。

### Step 5：KV cache sizes

打印 `N = 131072` 时每层、每种变体的 cache size。SWA 和 sparse variants 会下降 10-100 倍。Differential 会翻倍。要有意识地支付你的 memory bill。

## Use It

2026 production patterns：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接收一个 mask function：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成 custom Triton kernel。对常见 patterns，速度距离 FlashAttention-3 在 10% 以内，而且 mask function 是 Python callable。

**什么时候选哪一种：**

- **Pure full attention**：每层都用，适合 ~16K context 以内，或 retrieval quality 最重要的场景。
- **SWA + global mix**：long context（>32K），training 和 inference memory-bound。2026 年 32K 以上的默认选择。
- **Sparse block attention**：custom kernel、custom pattern。保留给 specialized workloads（retrieval、audio）。
- **Differential attention**：任何 attention-sink contamination 会伤害效果的 workload（long-context RAG、needle-in-haystack）。

## Ship It

见 `outputs/skill-attention-variant-picker.md`。这个 skill 会根据 target context length、retrieval demands 和 training/inference compute profile，为新模型选择 attention topology。

## 练习

1. **Easy。** 运行 `code/main.py`。验证 `window=4` 的 SWA 会把每行最后 4 个 tokens 之外的所有位置置零。验证 `window=n` 会 bit-identically 复现 full causal attention。
2. **Medium。** 在 Lesson 07 capstone 上实现 `window=1024` 的 causal SWA。在 tinyshakespeare 上训练 1,000 steps。相比 full attention，val loss 回退多少？Peak memory 降低多少？
3. **Hard。** 在 capstone model 中实现 Gemma-3-style 5:1 layer mix（5 层 SWA，1 层 global）。在 matched parameters 下，比较 pure-SWA、pure-global baselines 的 loss、memory 和 generation quality。
4. **Hard。** 实现每个 head 一个 learned `λ` 的 differential attention。在 synthetic retrieval task（one needle，2,000 distractors）上训练。与 matched parameters 的 single-attention baseline 比较 retrieval accuracy。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Sliding window attention (SWA) | "Local attention" | 每个 query attend 到最后 `W` 个 tokens；KV cache 缩到 `O(W)`。 |
| Effective receptive field | "How far back the model sees" | 在 `L`-layer SWA stack 且 window 为 `W` 时，最多能看到 `L × W` tokens 之前。 |
| Longformer / BigBird | "Local + global + random" | Sparse patterns，带少数 always-attending global tokens；早期 long-context approach。 |
| Native Sparse Attention | "DeepSeek's kernel trick" | 学习 block-level sparsity；在 kernel level 跳过 zero blocks，同时保持质量。 |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer：从第一张 attention map 中减去 learned `λ` 倍的第二张 map，以抵消 attention sinks。 |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization 强制 rows sum to 1；uninformative queries 会把权重倒到 position 0。 |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API，把任意 mask functions 编译成 FlashAttention-shape kernels。 |
| Layer type mix | "5:1 SWA-to-global" | 在 stack 中交错 sparse 和 full attention layers，以更低 memory 保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — canonical sliding-window + global-token paper。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的 local+strided pattern。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global mix。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 mix，window=1024，现在是 textbook default。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer paper。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的 learned-sparsity attention。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — Use It 中 mask-as-callable pattern 的 API reference。
