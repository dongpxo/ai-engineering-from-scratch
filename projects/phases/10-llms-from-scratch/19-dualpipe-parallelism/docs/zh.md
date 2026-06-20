# DualPipe 并行

> DeepSeek-V3 在 2,048 张 H800 GPUs 上训练，MoE experts 分散在不同 nodes。跨节点 expert all-to-all communication 的成本达到每 1 GPU-hour compute 对应 1 GPU-hour comm。GPU 有一半时间在空转。DualPipe（DeepSeek，2024 年 12 月）是一种 bidirectional pipeline，它把 forward 和 backward computation 与它们触发的 all-to-all comms 重叠起来。Bubbles 下降，throughput 上升；而保留两份 model-parameter copies（名字里的 “dual”）在 Expert Parallelism 已经把 experts 分散到各 ranks 时并不昂贵。本课是一个 Learn-type walkthrough，解释 DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV refinement 用稍微更紧的 bubble 换掉了 2x parameter cost。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## 学习目标

- 说出 DualPipe forward-backward chunk 的四个组成部分，并解释为什么每一部分都有自己的 overlap window
- 解释大规模 pipeline bubble problem，以及 “bubble-free” 在实践中和营销话术中分别是什么意思
- 手工追踪一个 8 个 PP ranks、16 个 micro-batches 的 DualPipe schedule，并确认 forward 和 reverse streams 如何填满彼此的 idle slots
- 说明 DualPipeV（Sea AI Lab，2025）的取舍：在 Expert Parallelism 不活跃时，用略大的 bubble 换掉 2x parameter replication

## 问题

在 2k H800 GPUs 上训练一个 671B MoE model，会撞上三个互相叠加的瓶颈：

1. **Memory pressure。** 每张 GPU 持有模型的一部分。Sequence 8k、61 layers、128 heads 下的 activation memory 非常巨大。
2. **Pipeline bubbles。** 传统 pipeline parallelism（GPipe、1F1B）会让 GPU 在等待本 stage 的 input 或 gradient 时空转。8 个 stages 时，即使用 1F1B scheduling，也可能有大约 12% 的 GPU time 变成 bubble。
3. **Cross-node all-to-all。** 带 expert parallelism 的 MoE 会把 experts 分散到 nodes。每次 forward pass 都会触发一次 all-to-all，把 tokens 分发给对应 experts，还会触发另一次 all-to-all 做 combine。在 2k GPUs 上，这很容易变成 1:1 的 compute-to-comm ratio。

每个问题都有单独解法：gradient checkpointing 处理 memory，Zero Bubble（Sea AI Lab，2023）处理 pipeline bubbles，expert-parallel comm kernels 处理 all-to-all。DualPipe 做的是让这些解法协同工作。它在一个 forward-backward chunk 内重叠 compute 和 comm，同时从 pipeline 两端注入 micro-batches，并用由此产生的 schedule 把 all-to-all 隐藏在 compute windows 里。

报告结果：DeepSeek-V3 的 14.8T-token training run 中，pipeline bubbles 几乎被消除，GPU utilization 超过 95%。

## 概念

### Pipeline parallelism refresher

把一个 N-layer model 分到 P 个 devices 上。Device `i` 持有 layers `i * N/P .. (i+1) * N/P - 1`。一个 micro-batch 从 devices 0 到 P-1 走 forward，然后从 P-1 到 0 走 backward。每个 device 只能在上游 device 发来 output 后开始自己的 forward stage，也只能在下游 device 发来 upstream gradient 后开始 backward。

GPipe（Huang et al., 2019）一次调度一个 micro-batch，会浪费大部分 GPU time。1F1B（Narayanan et al., 2021）为多个 micro-batches 交错 forward 和 backward passes。Zero Bubble（Qi et al., 2023）把 backward pass 拆成两部分，也就是 backward-for-input（B）和 backward-for-weights（W），并调度它们填补 bubble。Zero Bubble 之后，pipeline 已经接近紧凑。

DualPipe 是下一步。它在此基础上加入两个想法：

### Idea 1：chunk decomposition

每个 forward chunk 被拆成四个 components：

- **Attention。** Q/K/V projections、attention、output projection。
- **All-to-all dispatch。** 跨节点通信，把 tokens 发送到对应 experts。
- **MLP。** MoE expert computation。
- **All-to-all combine。** 跨节点通信，把 expert outputs 收回来。

Backward chunk 会加入这些 components 的 gradient 版本。DualPipe 对它们进行调度，让 all-to-all dispatch 与下一个 chunk 的 attention compute 并行，让 all-to-all combine 与后续 chunk 的 MLP compute 并行。

### Idea 2：bidirectional scheduling

大多数 pipeline schedules 从 stage 0 注入 micro-batches，并流向 stage P-1。DualPipe 从两端同时注入 micro-batches。Stage 0 会看到从这里出发的 forward micro-batches；stage P-1 也会看到从那里出发的 forward micro-batches。两条 streams 在中间相遇。

为了让它工作，device `i` 必须同时持有 early-pipeline layer `i` 和 late-pipeline layer `P - 1 - i`。这就是 DualPipe 中 “dual” 的部分：每个 device 保留两份自己需要服务的 model layers（一份服务每个方向）。在 DeepSeek-V3 的规模上，这是 2x parameter replication cost。但它是可承受的，因为 Expert Parallelism 已经把 MoE experts 分得很薄，复制两份 non-expert layers 只是小事。

关键点是：一个方向的 forward stream 和另一个方向的 backward stream，正好在单向 schedule 的 bubbles 位置重叠。Bubbles 消失了。

### 手工追踪 schedule

考虑 `P = 4` ranks、8 个 micro-batches，拆成 4 个 forward / 4 个 reverse。时间从左到右移动，行表示 device ranks。

```text
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读 `F4/F5R` 这种 notation：rank 1 正在同一个 time slot 内运行 micro-batch 4 的 forward（沿 pipeline 从左到右），同时运行 micro-batch 5 的 forward（从右到左）。这就是 “bidirectional” 在操作层面的含义。

在 rank 2，cross streams 更早重叠；在 rank 0 和 P-1，它们最晚重叠。在 schedule 的 stable middle phase，每个 rank 都会运行某个方向的 forward，并与另一个方向的 backward 重叠。Compute 保持忙碌。Forward pass 的 all-to-all dispatches 被隐藏在 backward compute 里。All-to-all combines 被隐藏在 forward compute 里。Bubbles 被挤出去了。

### Bubble accounting

标准 1F1B pipeline bubble，也就是每个 rank 浪费的时间：

```text
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble refinement 会把它降下来，但不是降到零。DualPipe 在 stable phase 中，如果 micro-batch count 可以被 `2 * pipeline depth` 整除，就有 zero bubble。在 stable phase 之外，也就是 warmup 和 cooldown 中，仍然会有一些 bubble，但它不会随着 micro-batches 数量增长。这是论文强调的关键性质。

营销话术： “bubble-free”。技术表述：bubbles 不随 micro-batch count 增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）显示，完整 zero-bubble 只有在 Expert Parallelism 不是瓶颈时成立；如果存在 EP-driven all-to-all，总会有一些 scheduling compromise。

### DualPipeV：refinement

Sea AI Lab（2025）观察到，当 EP comm overlap 不是重点时，2x parameter replication 很浪费。他们的 DualPipeV schedule 把 bidirectional injection 折成一种 “V-shape” schedule，并且只运行一份 parameter copy。Bubble 略大于 DualPipe，但 memory savings 很可观。DeepSeek 在其 open-source DualPipe implementation 中把 DualPipeV 作为 EP-off mode 采用。

取舍如下：

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### 对 14.8T-token run 意味着什么

DeepSeek-V3 的 pre-training 在 2,048 张 H800 GPUs 上消耗了 14.8T tokens，大约用了 2.8M GPU-hours。如果使用 naive 1F1B，他们会损失 12-15% 给 pipeline bubbles，也就是 340-420K GPU-hours，足够训练一个完整的 70B model。DualPipe 回收了其中大部分。没有内部 logs 很难直接量化它的贡献，但论文声称整个训练平均 GPU utilization 超过 95%。

对较小 runs（少于 1k GPUs）来说，DualPipe 过重了。Pipeline bubbles 相对总成本更小，而 dense-model training 很少遇到 all-to-all bottleneck。对多千 GPU 规模的 frontier MoE training 来说，它实际上是必需的。

### 它在 stack 中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 在 ranks 间 shard model parameters；DualPipe 在 ranks 间调度 compute。两者可以组合。
- 与 **ZeRO-3** gradient sharding 兼容。两份 copy replication 的 bookkeeping 需要和 ZeRO 的 sharded gradients 配合。
- 需要针对特定 cluster topology 调优的 **custom all-to-all kernels**。DeepSeek 的 open-source kernels 是 reference implementation。

```figure
expert-capacity
```

## 使用它

`code/main.py` 是一个 pipeline schedule simulator。它接收 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 各自的 stable-phase utilization。它是教学工具，数字匹配论文中的定性 claim，但不是 production measured speedup 的声明。

Simulator 的价值在于：用不同 P 和 micro-batch counts 运行它，观察 1F1B 的 bubble fraction 如何增长，而 DualPipe 不会。

真实 training run 的 integration considerations：

- 选择能被 micro-batch count 整除的 pipeline-parallel depth。
- 确保 expert-parallel mesh 支持 bidirectional all-to-all。DeepSeek 的 kernels 是 reference。
- 第一次做时，预期会花一周 debug schedule 本身。Bookkeeping 很繁琐。
- 监控每个 rank 的 GPU utilization，而不只是 aggregate。DualPipe 的收益来自收紧 stragglers。

## 交付它

本课会产出 `outputs/skill-dualpipe-planner.md`。给定 training cluster specification（GPU count、topology、interconnect、model shape），它会推荐 pipeline parallelism strategy、要使用的 scheduling algorithm，以及目标规模下的 expected bubble fraction。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU utilization difference，并把它表达为每训练一百万 tokens 可回收多少 GPU-hours。

2. 手工画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的 schedule table。用 micro-batch ID 和 direction 标记每个 time slot。识别第一个没有 bubbles 的 time slot。

3. 阅读 DeepSeek-V3 technical report（arXiv:2412.19437）的 Figure 5。识别 DualPipe forward chunk 内 all-to-all dispatch 的 overlap window。解释 compute schedule 如何隐藏它。

4. 分别计算 DualPipe 对一个 `P=8` pipeline stages 的 70B dense model，以及一个 `P=16` pipeline stages 的 671B MoE model 造成的 2x parameter overhead。说明为什么 MoE case 的 overhead 按比例更小（大多数参数是 experts，并且被 shard 到大型 EP group 中）。

5. 比较 DualPipe 和 Chimera（2021 年的一个竞争性 bidirectional scheduler）。以论文 Section 3.4 为参考，识别 DualPipe 增加了 Chimera 没有的两个具体性质。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline bubble | “每个 rank 的 idle time” | Pipeline stage 等待 input 或 gradient 时浪费的 GPU cycles |
| 1F1B | “默认 pipeline schedule” | One forward / one backward interleaved scheduling；DualPipe 要击败的 baseline |
| Zero Bubble | “Sea AI Lab 2023” | 把 backward 拆成 B（input gradient）和 W（weight gradient）；几乎完全收紧 pipeline |
| DualPipe | “DeepSeek-V3 schedule” | Bidirectional pipeline + compute-comm overlap；bubbles 不随 micro-batch count 增长 |
| DualPipeV | “Cut-in-half” | V-shape refinement，去掉 2x parameter replication，代价是稍大的 bubbles |
| Chunk | “Pipeline work unit” | 一个 micro-batch 通过一个 pipeline stage 的 forward 或 backward pass |
| All-to-all dispatch | “把 tokens 发送给 experts” | Cross-node comm，把 tokens 路由到它们被分配的 MoE experts |
| All-to-all combine | “把 expert outputs 收回来” | MLP 之后汇集 expert outputs 的 cross-node comm |
| Expert Parallelism (EP) | “Experts across GPUs” | 在 ranks 间 shard MoE experts，让不同 GPUs 持有不同 experts |
| Pipeline Parallelism (PP) | “Layers across GPUs” | 在 ranks 间 shard model layers；DualPipe 调度的维度 |
| Bubble fraction | “浪费的 GPU time” | `bubble_time / total_time`；DualPipe 将它推向零 |

## 延伸阅读

- [DeepSeek-AI -- DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) -- 主要 DualPipe 参考
- [DeepSeek -- DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) -- open-source reference implementation，包括 DualPipeV（Cut-in-half）mode
- [Qi et al. -- Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) -- Zero Bubble predecessor
- [Sea AI Lab -- DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) -- 影响 DeepSeek EP-off mode 的 DualPipeV analysis
- [Narayanan et al. -- PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) -- DualPipe 对比的 1F1B schedule
- [Huang et al. -- GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) -- 原始 pipeline parallelism paper 和 bubble problem
