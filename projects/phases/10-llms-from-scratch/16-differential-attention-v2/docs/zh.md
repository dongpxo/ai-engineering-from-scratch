# Differential Attention (V2)

> Softmax attention 会把少量 probability 分散到每个不匹配的 token 上。到了 100k tokens，这些 noise 会累积并淹没 signal。Differential Transformer（Ye et al., ICLR 2025）通过把 attention 计算为两个 softmax 的差来修复它，减去 shared noise floor。DIFF V2（Microsoft，2026 年 1 月）是 production-stack rewrite：decode latency 匹配 baseline Transformer，不需要 custom kernels，兼容 FlashAttention。这一课从 V1 到 V2 端到端讲清楚，并给出一个可用 stdlib Python 运行的 toy implementation，演示 difference operation。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## 学习目标

- 精确说明为什么 softmax attention 存在 noise floor，以及它为什么会随 context length 增长。
- 推导 differential attention formula，并解释为什么 subtraction 会抵消 shared noise component，同时保留 signal。
- 梳理 V1 到 V2 的 diff：什么变快了，什么变简单了，什么变稳定了，以及为什么这些变化对 production pre-training 必要。
- 用 pure Python 从零实现 differential attention，并在 synthetic signal-plus-noise query 上实证验证 noise-cancellation property。

## 问题

Standard softmax attention 有一个数学性质，在 scale 上会变成运行层面的麻烦。对 query `q`，attention weights 是 `softmax(qK^T / sqrt(d))`。Softmax 永远不能产生精确的 zeros，每个不匹配 token 都会得到一些 positive mass。这些 residual mass 是 noise，而且它会随 context length 扩大。在 128k tokens 下，即使每个不匹配 token 只得到 0.001% probability，127,999 个 token 加起来也会贡献约 12% 的总量。模型必须学会绕开一个会随 context 增长的 noise floor。

经验上，这表现为 attention-head interference：long-context RAG 中 hallucinated citations，100k-token retrieval tasks 上 lost-in-the-middle failures，以及 32k 之后 needle-in-haystack benchmarks 的细微 accuracy degradation。Differential Transformer paper（arXiv:2410.05258, ICLR 2025）测量了这个差距：相同 size 下，DIFF Transformers 相比 baselines 有更低 perplexity、更高 long-context accuracy，以及更少 hallucinations。

DIFF V1 有三个问题，让它难以进入 frontier pre-training pipelines。它的 value cache 在每个 decode step 必须加载两次，它需要破坏 FlashAttention compatibility 的 custom CUDA kernels，而且它的 per-head RMSNorm 在 70B+ scale 的 long-run training 中不稳定。DIFF V2（Microsoft unilm blog，2026 年 1 月 20 日）修复了这三点。这一课会走过两个版本，构建 difference operator，并在 toy query 上 benchmark noise cancellation。

## 概念

### Softmax 的 noise floor

对 query `q` 和 keys `K = [k_1, ..., k_N]`，attention weights 是：

```text
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会是 zero。如果 `k_i` 和 `q` 完全无关，score `q . k_i` 也不是 0，而是围绕 zero 波动，variance 为 `||q||^2 / d`。Softmax normalization 之后，每个无关 token 仍然对 weighted sum 贡献 `O(1/N)`。无关 tokens 的总贡献是 `O((N-1)/N) = O(1)`，这不是一个小量。

模型真正想要的是接近 hard top-k 的东西：匹配 tokens 上高权重，其他地方接近 zero。Softmax 太平滑，无法直接做到这一点。

### Differential idea

把每个 head 的 Q 和 K projections 分成两份：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两个 attention maps：

```text
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```text
DiffAttn = (A_1 - lambda * A_2) V
```

Subtraction 会抵消两个 maps 共享的 noise distribution。如果两个 maps 在 127k 个无关 tokens 上都有大致 uniform weight（random initialization 时会如此），它们会被抵消。Signal，也就是少数真正 relevant tokens 上的 peaked weight，只有在两个 maps 中以相同 magnitude 出现时才会抵消；模型训练后这通常不会发生。

`lambda` 是每个 head 的 learnable scalar，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负。`lambda_init` 默认是类似 0.8 的小正数。

### 为什么它像 headed noise-canceling

想象两个 noisy microphones 录下同一个声音。两者都录到说话者，也都录到 correlated background noise。一个减去另一个，共享 noise 会下降。声音能保留下来，因为两个 signals 在 phase 或 amplitude 上差异足够大，不会完全抵消。Per-head `lambda` 学到的正是这种 balance。

### V1 vs V2：diff

V1 保持 parameter count 与 baseline Transformer 相同。为了让每个 head 有两个 queries，它把 head dimension 减半。这牺牲了 head expressiveness，更痛的是，它让每个 head 的 value cache 减半。Decode 每步必须加载 value cache 两次（每个 softmax branch 一次）。结果：尽管 parameter count 匹配，decode 仍然慢于 baseline。

V2 会把 query heads 数量翻倍，同时保持 KV heads 不变（从 up-projection 借 parameters）。Head dimension 保持和 baseline 相同。Subtraction 之后，额外 dimension 会被 project 回去，以匹配 baseline Transformer 的 O_W projection。三件事同时发生：

1. Decode speed 匹配 baseline（KV cache 只加载一次）。
2. FlashAttention 原样运行（不需要 custom kernel）。
3. Decode 时 arithmetic intensity 上升（每从 HBM 加载一个 byte，有更多 compute）。

V2 还移除了 V1 用于稳定 subtraction 的 per-head RMSNorm。在 70B-class pre-training scales 上，那个 RMSNorm 会让 late training 不稳定。V2 用更简单的 initialization scheme 替代它，在没有额外 module 的情况下保持 training stable。

### 什么时候该用它

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | 更干净的 attention maps，更少 hallucinated citations |
| Needle-in-haystack benchmarks | 32k 之后有明显 accuracy lift |
| Multi-document QA | 更少 cross-document interference |
| Code completion at 8k | 提升有限，不值得改 architecture |
| Short chat (< 4k) | 基本和 baseline 无法区分 |

价值会随 context length 增长。4k tokens 时 noise floor 小到 standard attention 足够好。128k 时它已经在伤害你。

### 它如何叠加 2026 年的其他 knobs

| Feature | Compatible with DIFF V2? |
|---------|--------------------------|
| GQA | Yes（V2 增加 Q heads，不增加 KV heads） |
| MLA (DeepSeek) | 原则上 Yes，但还没有公开 paper 组合二者 |
| MoE | Yes（attention 独立于 MLP block） |
| RoPE | Yes（不变） |
| YaRN / long-context scaling | Yes（正是 DIFF 最有帮助的地方） |
| FlashAttention | V2 中 Yes（V1 中 no） |
| Speculative decoding | Yes（attention change 对 spec-decode loop 不可见） |

```figure
differential-attention
```

## 动手构建

`code/main.py` 用 pure Python 实现 differential attention。一个带已知 signal-plus-noise 结构的 toy query，可以让你直接测量 noise-cancellation ratio。

### Step 1：standard softmax attention

Stdlib matrix ops：lists of lists、manual matmul、带 max subtraction 的 numerically stable softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2：把 Q、K 分成两半

V1 style：把 head dimension 减半。V2 style：保持 head dimension，并把 heads 数量翻倍。Toy implementation 为了教学清晰使用 V1；数学完全相同，只有 bookkeeping 不同。

### Step 3：两个 softmax branches + subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：output weights 可以为负。这没问题，value cache 仍然可以处理 signed contributions。后续 V projection 会吸收 sign。

### Step 4：noise cancellation measurement

构建一个 length 1024 的 synthetic sequence。把 signal token 放在已知位置，其余填入 noise。计算 (a) standard softmax attention 在 signal position 上的 weight，和 (b) differential attention weight。测量两者的 signal-to-noise ratio。DIFF attention 通常能产生更高的 signal-to-noise ratio，提升倍数从 3x 到 10x 不等，取决于两个 branches 被训练得有多不同。

### Step 5：V1 vs V2 parameter accounting

给定 config（hidden=4096, heads=32, d_head=128），打印：

- Baseline Transformer：Q、K、V 各自 size 为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各自 size 为 `hidden * hidden`，V size 为 `hidden * hidden`（不变），内部 head dim 减半。增加 per-head `lambda` parameters（O(heads * d_head)）。
- DIFF V2：Q size 为 `2 * hidden * hidden`，K size 为 `hidden * hidden`，V size 为 `hidden * hidden`。Extra dim 会在 O_W 前 project 回去。增加同样的 `lambda` parameters。

Toy 会测量 V2 的额外 parameter cost（每个 attention block 约额外 `hidden * hidden`）并打印出来。

## 使用它

截至 2026 年 4 月，DIFF V2 还没有在每个 production inference server 中出货，但 vLLM 和 SGLang 的集成正在推进。与此同时，这种 pattern 已经出现在：

- Microsoft internal long-context production models。
- 面向 256k+ context 的多个 open model training runs 中的 research replications。
- 在 alternate layers 上组合 DIFF attention 和 sliding-window attention 的 hybrid architectures。

2026 年你会在这些情况下考虑它：

- 从零训练一个面向 64k+ effective context 的新模型。从一开始加入 differential attention；之后 retraining 很贵。
- Fine-tuning 一个 long-context model，且 lost-in-the-middle failures 主导 eval。Q projections 上的 LoRA 可以近似 DIFF structure。

你不会在这些情况下使用它：

- 你正在 serve 一个已有的 pre-trained dense model，而且它的 long-context performance 稳定。对 existing weights 来说，retraining cost 很少能回本。
- 你的 context 永远低于 16k。Noise floor 可以忽略。

## 交付它

这一课会产出 `outputs/skill-diff-attention-integrator.md`。给定 model architecture、target context length、hallucination profile 和 training budget，它会生成一个 integration plan，用于把 differential attention 加入新的 pre-training run 或 LoRA fine-tune。

## 练习

1. 运行 `code/main.py`。验证 synthetic query 上 differential attention 报告的 signal-to-noise ratio 高于 standard softmax attention。改变 noise amplitude，展示 standard attention 开始不可用的 crossover point。

2. 对一个 7B-class model（hidden=4096, heads=32, d_head=128, 32 layers），计算从 baseline 到 DIFF V1、从 baseline 到 DIFF V2 的 parameter-count delta。展示哪些 components 增加了 parameters，哪些保持不变。

3. 阅读 DIFF V1 paper（arXiv:2410.05258）的 Section 3 和 DIFF V2 Hugging Face blog 的 Section 2。用两句话解释为什么 V1 per-head RMSNorm 是必要的，以及为什么 V2 可以移除它而不导致 training divergence。

4. 实现一个 ablation：分别用 `lambda = 0`（pure first softmax）和 `lambda = 1`（full subtraction）计算 differential attention。在 synthetic query 上测量 signal-to-noise 如何随 sweep 变化。找出最大化 signal-to-noise 的 `lambda`。

5. 把 toy 扩展到 GQA + DIFF V2。选择 8 KV heads 和 32 Q heads。展示 KV cache size 与相同 (8, 32) 配置的 baseline GQA model 匹配。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Differential attention | "Two softmaxes minus each other" | 把 Q、K 分成两半，计算两个 softmax maps，用 lambda scale 后从第一个中减去第二个，再乘以 V |
| Noise floor | "The non-zero tail of softmax" | Softmax 给每个无关 token 分配的 O(1/N) weight，在 long contexts 中累加为 O(1) |
| lambda | "The subtraction scale" | Per-head learnable scalar，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "The ICLR 2025 version" | 原始 Differential Transformer；把 head dim 减半以保持 parameter count，需要 custom kernel，decode 更慢 |
| DIFF V2 | "The January 2026 fix" | 保持 KV heads，Q heads 翻倍；匹配 baseline decode speed，并兼容 FlashAttention |
| Per-head RMSNorm | "The V1 stabilizer" | V1 在 difference 后应用的 extra norm；V2 移除它以防 late-training instability |
| Signal-to-noise ratio | "How much attention is wasted" | True signal position 上的 weight 与无关 positions 平均 weight 的比值 |
| Lost in the middle | "Long-context failure mode" | Long context 中间位置文档 retrieval accuracy 下降的经验现象；DIFF attention 会缓解它 |
| Arithmetic intensity | "FLOPs per byte loaded" | V2 在 decode 时通过每次 KV load 翻倍 queries 提高的比值；对 memory-bound decode 很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含 noise-cancellation theory 和 long-context ablations。
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — production-stack rewrite，匹配 baseline decode，兼容 FlashAttention。
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 理论分析 subtraction 为什么能恢复 pretrained attention structure。
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — parameter-sharing variant。
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 所 subtract 的 baseline Transformer。
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF attention 面向的 long-context benchmark。
