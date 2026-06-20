# Native Sparse Attention（DeepSeek NSA）

> 在 64k tokens 下，attention 会吃掉 70-80% 的 decode latency。每个 open-model lab 都有修它的计划。DeepSeek 的 NSA（ACL 2025 best paper）是留下来的那个：三条并行 attention branches，通过 learned gate 组合在一起：compressed coarse-grained tokens、selectively retained fine-grained tokens，以及用于 local context 的 sliding windows。它 hardware-aligned（kernel-friendly）、natively trainable（能用于 pre-training，而不是 inference 时外挂），并且在 64k decodes 上比 FlashAttention 更快，同时达到或超过 full attention 质量。本课会端到端构建这三条 branches，并说明为什么这种 sparsity 可以端到端 differentiable。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12（KV cache, flash-attention），Phase 7 · 15（attention variants），Phase 10 · 16（differential attention）
**Time:** ~60 minutes

## 学习目标

- 说清 NSA 的三条 attention branches，以及每条 branch 捕捉什么信息
- 解释为什么 NSA 是 “natively trainable”，而早期 sparse-attention methods 多数只能在 inference 阶段使用
- 在 64k context 下，根据 compression block size 和 selection top-k，计算 NSA 相对 full attention 的 attention compute savings
- 用 stdlib Python 在短 synthetic sequence 上实现三分支组合，并验证 gating weights 的行为

## 问题

Full attention 在 sequence length N 时，需要 `O(N^2)` 时间和每层 `O(N)` KV cache。到了 64k tokens，compute 和 memory bandwidth 数字会变得灾难性。NSA paper 的理论测量估计是：在 64k 下，attention 占总 decode latency 的 70-80%。下游的一切：TTFT、tokens/sec、每百万 tokens 成本，都会被 attention cost 主导。

Sparse attention 是显然的答案。早期尝试大致分成两类。Fixed-pattern sparsity（sliding-window、strided、block-local）会丢掉信息，并在 long-range recall tasks 上失败。Inference-time sparsity（KV cache pruning、H2O、StreamingLLM）应用在 dense attention 预训练出来的模型上，只能恢复一部分潜在 speedup，因为模型从来没有被要求通过 sparse pattern 路由信息。

Native Sparse Attention（Yuan et al., DeepSeek + PKU + UW, ACL 2025 best paper, arXiv:2502.11089）同时解决两件事：sparsity pattern 在 pre-training 中学习，并且实现成 kernel-aligned algorithm，inference 时真的能交付 compute savings。两年后，NSA 或它的直接后继，很可能会成为每个 frontier long-context model 的默认 attention。

## 概念

### 三条并行 branches

对每个 query，NSA 会针对 KV cache 的三种不同视图运行三次 attention：

1. **Compressed branch。** Tokens 被按大小为 `l` 的 blocks 分组（通常是 32 或 64）。每个 block 通过一个小 learned MLP 压缩成一个 summary token。Query 会 attend 到这些 compressed tokens，从而获得 whole sequence 的 coarse-grained view。

2. **Selected branch。** 利用 compressed branch 的 attention scores，识别出对当前 query 最相关的 top-k blocks。系统会读取这些 blocks 里的 fine-grained（未压缩）tokens，然后 query 对它们全部做 attention。可以把 compressed-branch attention 理解成 selection 的 routing signal。

3. **Sliding-window branch。** Query 会 attend 到最近的 `W` 个 tokens（通常是 512），用于 local context。这条 branch 捕捉 structure-heavy 的 short-range patterns，例如 syntax 和 local coreference，而另外两条 branch 可能会漏掉这些。

三条 branch 的输出通过一个 learned per-position gate 组合：

```text
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是一个 query 上的小 MLP 输出的 gate weights。它们不一定要加和为 1，可以独立地给每条 branch 加权。

### 为什么它是 “natively trainable”

Selection step（top-k blocks）是离散的。离散操作会切断 gradient flow。早期 sparse-attention 工作要么跳过 selection 的 backprop（限制训练），要么使用 continuous relaxations，但这些 relaxation 在 inference 时并不能带来真正 sparsity。

NSA 绕开了这个问题：compressed-branch attention 本身就是一个对 whole sequence 的 differentiable coarse-grained attention。Top-k operation 只是复用 compressed branch 的 top attention scores，来决定加载哪些 fine-grained blocks。Gradients 会流过 compressed-branch scores（它们既影响 compressed output，也影响 selection logic），selected blocks 对 final output 的贡献也是 differentiable 的。不可微的 `top_k` operation 在 forward computational graph 上是一个 no-op：它只控制哪些 blocks 会从 memory 中加载。

这就是 NSA 可以端到端用于 pre-training 的原因。模型会联合学习如何通过三条 branches 路由信息，产生一种 sparse pattern，而这个 pattern 在 inference 时确实能交付承诺的 speedup。

### Hardware-aligned kernel

NSA 的 kernel 是为现代 GPU memory hierarchy 设计的。Kernel 按 GQA groups 加载 queries（outer loop），为每个 group 拉取对应 sparse KV blocks（inner loop），并在 SRAM 上运行 attention。因为每个 query group 看到同一批 selected blocks（selection 是 per-query-group，不是 per-query-head），KV loads 可以在 group 内摊销。Arithmetic intensity 会保持较高水平。

论文报告说，Triton kernels 在 64k decodes 上比 FlashAttention 快 9x，而且 speedup ratio 会随 sequence length 增长而提高。Forward 和 backward kernels 都已提供。

### Compute budget

设 `N` 为 sequence length，`l` 为 compression block size，`k` 为 top-k selection count，`w` 为 sliding window，`b` 为 selected block size（通常等于 `l`）。

- Compressed branch：每个 query 有 `O(N/l)` 个 keys，所以 total 是 `O(N * N / l)`。
- Selected branch：每个 query 有 `O(k * b)` 个 keys，所以 total 是 `O(N * k * b)`。
- Sliding branch：每个 query 有 `O(w)` 个 keys，所以 total 是 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512` 时：per-query cost 是 `1000 + 1024 + 512 = 2536 keys`。Full attention 是 `64000 keys`。Compute 降低 25x。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512` 时：per-query cost 是 `2000 + 1024 + 512 = 3536 keys`。Full attention 是 `128000 keys`。降低 36x。收益会随 sequence length 增长而提高，这正是它的意义。

### 和其他方法相比

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|----------------|------------------------|-------------------|
| Sliding window only | yes | yes | fails |
| Strided / block-sparse | yes | yes | partial |
| KV pruning (H2O, StreamingLLM) | N/A（inference-time） | yes | partial |
| MoBA (Moonshot) | partial | yes | good |
| NSA | yes（natively） | yes（64k 时 9x） | matches full attention |

MoBA（Moonshot, arXiv:2502.13189）几乎同期发表，也采用类似 “three-is-better-than-one” 的思路，把 MoE principle 应用于 attention blocks。NSA 和 MoBA 是理解 2026 long-context pre-training 时最值得记住的两个 architectures。

```figure
sliding-window-attention
```

## 动手构建

`code/main.py` 会在一段短 synthetic sequence 上实现三条 branches，并展示：

- Compression MLP（为了教学清晰，这里使用一个简单的 mean-pool baseline；真实 NSA 使用 learned MLP）。
- 由 compressed-branch scores 驱动的 top-k block selection。
- 对最后 `w` 个 tokens 的 sliding-window attention。
- Gated combination。
- 与 full attention 对比的 compute-count printout。

### Step 1：把 tokens 压缩成 blocks

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2：Compressed-branch attention

运行 query 对 compressed keys 的 softmax attention。Compressed-branch scores 同时作为 top-k selection 的信号。

### Step 3：Top-k block selection

选择得分最高的 `k` 个 compressed blocks 的 indices。加载这些 blocks 中的原始 uncompressed tokens，并在它们上运行 attention。

### Step 4：Sliding-window attention

取最后 `w` 个 tokens，并在它们上运行标准 attention。

### Step 5：Gate + combine

Query 上的一个小 MLP 产生三个 gate weights。最终输出是三条 branch outputs 的 weighted sum。

### Step 6：Compute counting

打印每条 branch 每个 query attend 到的 keys 数量和总数。与 `N`（full attention）对比。在一个 1024-token synthetic 上，使用 `l = 32, k = 4, w = 128` 时，NSA 每个 query 看到 `32 + 128 + 128 = 288` 个 keys，而 full attention 是 1024 个，少了 3.5x。

## 投入使用

NSA 正在 DeepSeek 自己的 long-context pre-training pipeline 中使用。截至 2026 年 4 月，public inference stacks 的集成状态如下：

- **DeepSeek internal**：native，已发布 weights 使用 NSA 或它的后继 DSA（Deepseek Sparse Attention）。
- **vLLM**：针对 DeepSeek-V3.x weights 的实验性 NSA support 正在开发中。
- **SGLang**：已有 NSA benchmarks；production path 跟随 vLLM。
- **llama.cpp / CPU**：不支持；在 CPU throughput 下，kernel decomposition 的 overhead 不值得。

什么时候考虑 NSA：

- 面向 64k-plus context、并且有认真 compute budget 的 pre-training 或 continued-training run。
- 推理 DeepSeek 自己的 long-context checkpoints。它们的 weights 是 NSA-native。

什么时候不要用：

- Serving 一个已有的 dense-attention pre-trained model。没有 continued training，你无法 retrofit NSA。
- Context 低于 16k。三分支 overhead 会压过 savings。
- Batch-1 interactive chat。Latency-sensitive decode 会受益，但主要是在 long contexts 下。

## 交付成果

本课会产出 `outputs/skill-nsa-integrator.md`。给定一个 long-context pre-training run specification，它会生成 NSA integration plan：compression block size、top-k、sliding window、gate MLP width、kernel choice，以及能证明架构切换值得的 specific long-context evals。

## 练习

1. 在 1024-token synthetic 上运行 `code/main.py`。扫过三组 `(l, k, w)` presets 并打印 compute counts。找出 key-count per query 最低，同时在 needle-in-haystack test 上相对 full attention 保持 95% recall 的 preset。

2. 用一个 tiny learned MLP（2-layer, hidden 32）替换 mean-pool compressor。在一个 block signal 是 block 平均值的 synthetic task 上训练它。测量它在 held-out data 上相对 mean-pool baseline 的 perplexity gap。

3. 实现 gate MLP。它以 query 为输入，输出三个 scalars。展示 gate 的行为合理：random queries 上接近 uniform weighting；当 query 命中 far-back block 时，对 selected branch 给出较大权重。

4. 计算 NSA-enabled 70B model 在 128k context 下的 KV cache memory budget。KV heads 是 8，head dim 是 128，BF16。与 full attention 和 MLA 比较（Phase 10 · 14 展示过 MLA 的数字）。找出 NSA 的 fine-grained branch KV cache 等于 full attention 的 sequence length。

5. 阅读 NSA paper（arXiv:2502.11089）Section 4，并用三句话解释为什么 compressed branch 的 attention scores 会被复用于 top-k selection，而不是计算一个单独 routing score。把答案和 gradient flow 联系起来。

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|----------|----------|
| Compressed branch | “Coarse view” | 对 block-averaged keys 做 attention，让每个 query 用 O(N/l) 个 keys 获得 global context |
| Selected branch | “Top-k blocks” | 对 compressed-branch scores 最高的 `k` 个 blocks 内的 fine-grained tokens 做 attention |
| Sliding window | “Local context” | 对最后 `W` 个 tokens 做 attention，用于 short-range patterns |
| Native trainability | “带着 sparsity 预训练” | Sparsity pattern 在 pre-training 中学习，而不是 inference 时外挂 |
| Compression block size l | “Coarse view 的 group size” | 多少 tokens 合并成一个 summary；典型值 32-64 |
| Top-k | “保留哪些 blocks” | 读取 uncompressed tokens 的 compressed blocks 数量；典型值 16 |
| Sliding window W | “Local attention radius” | 通常是 512；太短会伤害 local coherence，太长会浪费 compute |
| Branch gate | “三条 branch 怎么混合” | Per-position MLP 输出，用来给三条 branches 的贡献加权 |
| Hardware alignment | “Kernel-friendly sparsity” | Sparse pattern 的选择方式能让真实 GPU kernel 达到理论 speedup |
| DSA | “NSA 的后继” | Deepseek Sparse Attention，DeepSeek lineage 中接在 NSA 之后的 architecture |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 原论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 面向的 architecture family
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，MoE-style attention over blocks
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — sliding-window 起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA 改进的 inference-time sparsity baseline
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA kernels 在 64k 下超过的 full-attention baseline
