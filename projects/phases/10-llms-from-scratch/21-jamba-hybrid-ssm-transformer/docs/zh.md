# Jamba — Hybrid SSM-Transformer

> State space models（SSMs）和 transformers 想要的东西不同。Transformers 用 quadratic cost 的 attention 换质量。SSMs 用 recurrence 换 linear-time inference 和 constant memory，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把二者放进同一个模型：每 7 个 Mamba layers 配 1 个 Transformer layer，每隔一个 block 使用 MoE，并且 256k context window 可以装进单张 80GB GPU。Mamba-3（ICLR 2026）用 complex-valued state spaces 和 MIMO projections 强化 SSM 侧。这一课端到端阅读两种 architecture，并解释为什么 pure-SSM 和 pure-Transformer 的 long-context 尝试都没能持续 scaling，而 hybrid recipe 活了三年。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## 学习目标

- 解释 Jamba block 中的三个 primitives：Transformer layers、Mamba layers、MoE，以及 1:7:even interleaving recipe。
- 从高层说明 SSM 的 recurrence 长什么样，以及为什么它能实现 constant-memory inference。
- 计算 Jamba model 在 256k context 下的 KV cache footprint，并和 pure-Transformer model 的需求比较。
- 说出 Mamba-3 的三个创新：exponential-trapezoidal discretization、complex-valued state update、MIMO，以及每个创新针对的问题。

## 问题

Attention 对 sequence length 是 quadratic。State space models 是 linear。这个差异会不断放大：在 256k tokens 下，一个 Transformer attention map 每个 head 有 65B entries；SSM 的 recurrent state 不随 sequence length 增长，而是固定大小。

Pure-SSM models（Mamba、Mamba-2）在小规模下能匹配 Transformer perplexity，但在 state-tracking tasks 上落后，并且在某些 in-context retrieval 类别中失败。直觉是：SSMs 会把 history 压缩进 fixed state；history 很长时，信息会泄漏。Attention 可以精确记住所有东西，但要付 quadratic cost。

显然的修复方式：两个都用。把 Transformer layers 放在 exact recall 重要的地方。其他地方用 SSM layers。调 ratio。Jamba 是第一个在 scale 上交付这种 hybrid recipe 的 production-grade model（52B total，12B active，256k context，单张 80GB GPU）。Jamba 1.5 把家族扩展到 398B total / 94B active。Mamba-3（ICLR 2026）是当前最好的 pure-SSM baseline，未来的 hybrid 可以围绕它重建。

这一课会阅读三篇 paper，并产出一个“如何选择正确 ratio”的 mental model。

## 概念

### 一页讲清 SSM

State space model 通过 fixed-size state `h` 处理 sequence `x_1, ..., x_N`：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步，state 通过 linear dynamics `A` 演化，接收 input `B x_t`，并输出 `C h_t`。`A, B, C` 都可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。Memory 是 constant。Inference 是每 token O(1)。

建模质量的技巧在于 `A` 的结构。S4（Gu 2021）使用高度 structured matrix，可以在 training 时高效地作为 long convolution 计算。Mamba（Gu, Dao 2023）把 fixed `A, B, C` 替换成 data-dependent ones（也就是 "selective" 部分）。Mamba-2（2024）进一步简化结构。Mamba-3（2026）又在特定位置把复杂度加回来。

关键性质：对于 decoder LLM，SSM layer 可以作为 attention layer 的 drop-in replacement，只是它使用 fixed-size per-layer state，而不是不断增长的 KV cache。

### Jamba block

Jamba block 根据两个数字交错 layers：

- `l`：attention-to-Mamba ratio。Jamba 使用 `l = 8`，也就是每 7 个 Mamba layers 配 1 个 Transformer layer（7 Mamba + 1 Attention = 每组 8 layers）。
- `e`：MoE frequency。Jamba 使用 `e = 2`，也就是每隔一层应用 MoE。

一个 block 内的 layer sequence：

```text
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba block 是 8 layers。4 个 blocks 深（总共 32 layers）时，你会得到 28 个 Mamba layers 和 4 个 Attention layers。其中 16 个使用 MoE。

### 为什么是 1:7 ratio

AI21 做了 ablations：什么 attention-to-Mamba ratio 能在 perplexity-per-parameter 和 long-context evals 上的 in-context recall 之间取得最好平衡？

- 太多 attention（1:1）：质量上升，但 memory 和 speed 变差。
- 太少 attention（1:15）：memory 很好，但 in-context retrieval 失败。
- Sweet spot：1:7 或 1:8。

直觉是：Transformer layers 处理 exact recall 和 state tracking。Mamba layers 处理便宜的大部分 processing。

### Positional encoding

Mamba layers 本身是 position-aware 的（通过 recurrence）。原始 Mamba-based hybrids 中的 attention layers 没有使用 RoPE，因为 SSM layers 提供了 position info。Jamba 1.5 给 attention layers 加入 RoPE，以提升 longer-context generalization，这是基于 empirical long-context evaluation 的 post-hoc refinement。

### Memory budget

对于一个 Jamba-1 shape（32 layers：28 Mamba + 4 Attention，hidden 4096，32 attention heads）：

- KV cache（只有 attention layers）：256k BF16 下是 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个 attention layers 贡献 KV cache。
- SSM state：每个 token prefix 是 `28 * hidden * state_size`，但这是每层 fixed-size，不随 sequence length 增长。典型 Mamba state 是每个 feature 16，hidden 4096：总共 `28 * 4096 * 16 * 2 = 3.7 MB`。

和 pure Transformer 比较：32 layers、相同 hidden、32 heads 的 full MHA，在 256k BF16 下是 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV cache 降低 8x。即便和多数 2024 models 使用的 GQA(8) baseline 比（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 hybrid 在 16 GB 下仍然小 2x。

这就是 AI21 所说的“single 80GB GPU 上的 256k context”。Full-MHA pure Transformer 的 KV cache 装不下；即便 GQA baseline，也几乎不给 weights 和 activations 留空间；Jamba 可以。

### Mamba-3：2026 年的 pure-SSM baseline

Mamba-3（ICLR 2026，arXiv:2603.15569）在 pure-SSM 侧引入三个创新：

1. **Exponential-trapezoidal discretization.** 用更有表达力的 recurrence 替代 Mamba-2 中的 Euler-method discretization。Convolution-like operation 被应用在 core recurrence 内部的 state-input 上，而不是作为 `x_t` 上的 outer convolution。

2. **Complex-valued state update.** 之前的 Mamba 把 state matrix 从 complex（S4）降到 real diagonal（Mamba），再到 scaled identity（Mamba-2）。Mamba-3 重新引入 complex values，相当于 state 上的 data-dependent rotary embedding。这恢复了之前 real-valued simplifications 失去的 state-tracking capabilities。

3. **Multi-input multi-output (MIMO) projections.** 不再使用 per-feature scalar projections，而是使用 matrix-valued projections。在不增加 decode latency 的情况下，提升 modeling power 和 inference-time hardware utilization。

在 1.5B parameters 下，Mamba-3 相比 Gated DeltaNet 平均 downstream accuracy 提升 0.6 points；MIMO variant 再增加 1.2，总提升 1.8 points。在相同 state size 下，Mamba-3 用一半 state 就能匹配 Mamba-2。

Mamba-3 还没有作为 production hybrid 大规模出货，但它显然是下一代 Jamba-class model 中 SSM 侧的候选。

### 什么时候该选择 hybrid

Hybrids 在这些情况下获胜：

- Context 足够长，pure Transformer KV cache 开始痛苦（64k+）。
- Tasks 混合 short-range structure（适合 SSM）和 long-range recall（需要 Transformer）。
- 你想部署在 single-GPU memory budget 上，而 Transformer KV cache 自己都装不下。

Hybrids 在这些情况下吃亏：

- Context 很短（低于 16k）。SSM overhead 被浪费；pure Transformer 足够好。
- Tasks 需要 everywhere-to-everywhere attention（deep reasoning、multi-document cross-reference）。Hybrid 中 attention layers 稀疏，会伤害它。
- 你正在 scale 到 trillion-parameter frontier models。Pure-Transformer + MLA + MoE（DeepSeek-V3 style）目前赢得 capability race。

### 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|-------|--------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 年的格局：pure-Transformer MoE 主导 frontier，但 hybrids 占据 256k+ context niche。Mamba-3 的 state-tracking gains 可能会让下一代 hybrid 的 ratio 更低，也就是更多 SSM、更少 attention。

```figure
swiglu-ffn
```

## 使用它

`code/main.py` 是一个 hybrid architectures 的 memory calculator。给定 SSM-Transformer ratio 和 hidden-size / layer-count config，它会计算：

- 目标 context 下的 KV cache。
- SSM state memory。
- 一组 model shapes 在 context N 下的 total memory。

Calculator 支持：

- Pure-Transformer baseline（KV cache 随 N 增长）。
- Jamba-style 1:7 hybrid。
- Pure-SSM（完全没有 KV cache）。

公开 shapes 的数字直接来自 Jamba-1 和 Jamba-1.5 papers；hypothetical variants 使用 extrapolation。

真实 deployment 的 integration considerations：

- 多数 production inference servers（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k context 下，Jamba 的 memory advantage 会体现在 concurrent-request throughput 上。同样 VRAM 中，你能放下更多 Jamba sequences，而不是 Transformer sequences。
- Mamba-3 作为 standalone model 还没有 production 出货，目前是 1.5B research preview。

## 交付它

这一课会产出 `outputs/skill-hybrid-picker.md`。给定 workload specification（context length profile、task mix、memory budget），它会在 pure Transformer、Jamba-style hybrid 和 pure SSM 之间给出推荐，并明确说明 memory 与 quality tradeoffs。

## 练习

1. 运行 `code/main.py`，计算 32-layer pure Transformer（hidden 4096，32 heads）在 256k context 下的 KV cache，以及相同 shape 的 Jamba-1 hybrid。验证 AI21 paper 声称的约 8x memory reduction。

2. 修改 calculator，建模 1:3 hybrid（4 Mamba : 1 Attention）和 1:15 hybrid（14 Mamba : 1 Attention）。绘制 KV cache vs ratio。在什么 ratio 下，KV cache 会等于 SSM state memory？

3. 阅读 Jamba paper（arXiv:2403.19887）的 Section 3。解释为什么 AI21 使用 Mamba-1，而不是更快的 Mamba-2。提示：hybrid ablation section 记录了这一点。

4. 计算 Jamba 1.5 Large（398B total，94B active）中 MoE-every-other-layer 的 parameter overhead。把 active ratio 和 DeepSeek-V3（37B/671B）比较，并解释为什么 Jamba architecture 会把 active ratio 推高。

5. 阅读 Mamba-3 paper（arXiv:2603.15569）的 Section 3。用三句话解释为什么 complex-valued state update 等价于 data-dependent rotary embedding。把答案和 Phase 7 · Lesson 04 的 RoPE derivation 联系起来。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| State space model (SSM) | "Recurrence with a fixed state" | 带 learned recurrence `h_t = A h_{t-1} + B x_t` 的 layer；每个 token constant memory |
| Selective SSM | "Mamba's trick" | Data-dependent A、B、C parameters，让模型以 linear time 获得类似 gating 的 selectivity |
| Attention-to-Mamba ratio | "How many attention layers" | 在 Jamba 中，`l = 8` 表示每 7 个 Mamba layers 配 1 个 attention layer |
| Jamba block | "The 8-layer group" | 一个 attention + 七个 Mamba + alternate positions 上的 MoE |
| SSM state | "The hidden buffer" | 替代 Mamba layers 中 KV cache 的 fixed-size per-layer state |
| 256k context | "Jamba's flagship number" | Jamba-1 能装进单张 80GB GPU 的 sequence length；pure Transformer 在这个大小装不下 |
| Mamba-3 | "2026 pure SSM" | 当前最佳 pure-SSM architecture，带 complex state + MIMO；hybrids 会围绕它重建 baseline |
| MIMO | "Multi-input multi-output" | Mamba-3 的创新，用 matrix-valued projections 替代 scalar per-feature |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | 更有表达力的 recurrence，涵盖 Mamba-2 的 Euler-method discretization |
| Hybrid architecture | "Mix attention and SSM" | 任何交错 Transformer 和 SSM layers 的模型；Jamba 是 production archetype |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba paper，ratio ablations，256k context claim。
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — scaled-up family，398B/94B 和 12B/52B public releases。
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的 selective SSM paper。
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化版 structured-state-space successor。
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — complex-valued state、MIMO、2026 pure-SSM frontier。
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 paper，LLM 中 SSM genealogy 的起点。
