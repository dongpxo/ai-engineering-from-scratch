# Async and Hogwild! Inference

> Speculative decoding（Phase 10 · 15）是在一个 sequence 内并行化 tokens。Multi-agent frameworks 会跨完整 sequences 并行化，但要求显式协调（voting、sub-task splitting）。Hogwild! Inference（Rodionov et al., arXiv:2504.06261）做的是另一件事：让同一个 LLM 的 N 个实例并行运行，并共享一个 key-value cache。每个 worker 都能立刻看到其他 workers 生成的 tokens。现代 reasoning models，例如 QwQ、DeepSeek-R1，可以通过这个 shared cache 自我协调，不需要任何 fine-tuning。这个方法仍处于实验阶段，但它打开了一条全新的 inference parallelism 轴线，和 spec decode 正交。本课会用 stdlib Python 实现一个 two-worker Hogwild! simulator，并解释 shared-cache collaboration 为什么会从现有模型的 reasoning abilities 中涌现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12（inference optimization），Phase 10 · 15（speculative decoding）
**Time:** ~60 minutes

## 学习目标

- 描述三种常见 parallel-LLM topologies（voting、sub-task、Hogwild!），并说明每一种适合解决什么问题
- 说清 Hogwild! 的核心设置：multiple workers、一个 shared KV cache、通过 self-prompting 涌现协调
- 根据 worker count `N`、task-level parallelism `p` 和 coordination overhead `c`，计算 Hogwild! 的 wall-time speedup
- 在 toy problem 上实现 two-worker Hogwild! simulator，并观察 emergent task division

## 问题

现代 LLM 通过生成很长的 reasoning chains 来解决难题。5000 tokens 的 step-by-step logic 很常见；深层数学题上出现数万 tokens 也并不罕见。在 70B 模型 35 tokens/sec 的 decode 速度下，50k tokens 需要 24 分钟。这不是 interactive experience。

Speculative decoding（Phase 10 · 15）通过在一个 sequence 内并行化，带来 3-5x speedup。再往后，autoregressive decoding 的顺序依赖就是硬上限。每个新 token 都依赖此前所有 tokens。

显然的问题是：能否跨 sequences 并行化？在同一个问题上运行同一模型的多个 copies，让它们合作，让它们分工？

已有方法包括：voting ensembles（运行 N 个模型，选 majority answer）、tree-of-thought（分叉 reasoning paths 再重组），以及 multi-agent frameworks（给每个 agent 分配 sub-task，再使用 coordinator）。这些方法都能在特定任务域里提供帮助。但它们也都会引入显式 coordination machinery：voting rules、branch-and-prune logic、agent-to-agent messaging protocols。

Hogwild! Inference 采用另一种方式。N 个 workers 共享一个 KV cache。每个 worker 都能立刻看到其他 workers 生成的 tokens，就像那些 tokens 已经在它自己的 context 中一样。这些 workers 在没有任何 training 或 fine-tuning 的情况下，会自己想出如何分工。现代 reasoning models（QwQ、DeepSeek-R1、Claude-family reasoning mode）可以读取 shared cache，并说出类似 “I see worker 2 already handled the base case, so I'll work on the inductive step.” 的内容。

截至 2026 年 4 月，speedup 依赖 workload，且仍处于实验阶段。但这个想法值得掌握，因为它打开了一条新的 inference parallelism 轴线。

## 概念

### 设置

初始化 N 个 worker processes，它们都运行同一个 LLM。不要给每个 worker 单独的 KV cache，而是维护一个 shared cache。当 worker `i` 生成 token `t_j` 时，这个 token 会写入 shared cache 的下一个位置。当 worker `k` 进行下一步时，它读取 cache 的当前状态，其中包含所有 N 个 workers 到目前为止生成的全部内容。

在 step time，workers 会竞争写入 tokens。没有 per-worker position index：cache 是一个单独增长的 sequence。顺序由写入到达时间决定。

### 为什么 coordination 会涌现

Workers 共享一个 prompt。通常类似：“You are one of N instances working together on this problem. Each instance reads the shared memory and can see what other instances have written. Avoid redundant work.” 这个 prompt 加上 shared cache 就足够了。Reasoning models 会读取 cache，注意到问题的哪些部分已经被尝试过，并且（经常但不总是）转向未探索部分。

Hogwild! paper（Rodionov et al., 2025）报告了这些现象：

- Workers 会制定 plans，并通过 cache 和其他 workers 沟通。
- Workers 会注意到其他 workers reasoning 中的错误，并指出它们。
- Workers 会在 plan 失败时适应，并提出 alternatives。
- 当被提示检查 redundancy 时，workers 能检测到重复工作并 pivot。

这些都不需要 fine-tuning。Emergent behavior 来自模型已经具备的 reasoning capabilities。

### 命名

论文名称借用了 Hogwild! SGD（Recht et al., 2011），一种 asynchronous-update optimizer。类比是：SGD 的 asynchronous workers 都写入同一个 shared parameter vector；Hogwild! Inference 的 workers 都写入同一个 shared KV cache。二者都依赖 empirical convergence，而不是 synchronization guarantees。

### RoPE 让这件事可行

Rotary Position Embeddings（RoPE, Su et al. 2021）通过 Q 和 K vectors 中的 rotation 来编码 position information。因为 positions 是 rotations，而不是烙死的 offsets，一个 token 的 position 可以 shift，而不需要重新计算 KV cache entry。当 worker `i` 写入 shared cache 的 position `p` 时，其他 workers 读取这个 position 可以直接使用 cached entry，不需要 re-rotation。

在 learned-position 或 absolute-position model 中，Hogwild! 每次 concurrent write 都需要 cache invalidation。RoPE 让 cache 保持稳定。

### Wall-time math

设 `T_serial` 为一个 worker 独自解决问题所需时间。设 `p` 为 task-level parallelizable fraction。设 `c` 为 per-step coordination overhead（读取扩展 cache，决定写什么）。

Single-worker time：`T_serial`。

如果 coordination 免费，N-worker Hogwild! time 是：`T_serial * ((1 - p) + p / N)`。这是经典 Amdahl。

带 coordination overhead 时：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要让 worker productive，`c` 必须相对 per-step decode time 足够小。对于会生成 5k+ tokens 的 reasoning models，workers 可以花掉数百 tokens 的 coordination overhead，仍然跑赢单 worker。对于 short chat tasks，coordination 会主导成本，Hogwild! 会比 serial 更差。

### 具体例子

Reasoning problem：10k tokens 的 chain-of-thought。假设问题有 `p = 0.7` parallelizable content（不同 proof strategies、不同 case analyses），并且每个 worker 有 `c = 200` tokens 的 coordination overhead。使用 `N = 4` workers：

- Serial time：10000 decode steps。
- Hogwild! time：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 decode steps。
- Speedup：10000 / 5550 = 1.8x。

这很温和。但在更长的 reasoning problems（50k tokens）上，coordination overhead 会被摊薄，speedup 会推进到 2.5-3x。Hogwild! 就像 inference 版本的 thread-level parallelism，只是它让你能用自然语言写 multi-threaded code。

### 什么时候考虑 Hogwild!

- Long reasoning problems（数千 tokens），并且任务能跨独立 sub-goals 并行化。
- 已经被训练成 step-by-step thinking 的 reasoning models。Non-reasoning models 不会很好地自我协调。
- 有足够 VRAM 的 single-node deployments，可以容纳 shared cache 和 N 个 worker processes。Cache 是 shared 的，但每个 worker 仍有自己的 activation memory。

### 什么时候不要用

- Short interactive chat。Coordination overhead 会主导。
- 不能并行化的任务（单条 linear proof、单次 compilation）。N=1 就是上限。
- Non-reasoning models。不会涌现 coordination。
- Multi-node deployments。Shared cache 需要非常快的 cross-worker synchronization。Intra-node 可以，cross-node 会是 latency disaster。

### 实验状态

截至 2026 年 4 月，Hogwild! 是一种 research method，并有开源 PyTorch implementation。Production adoption 还没有发生。三个 blockers：

1. 跨 concurrent processes 管理 shared KV cache 是不简单的工程问题。
2. Emergent coordination 依赖任务；benchmarks 仍在构建中。
3. Speedups 相对 speculative decoding 已经带来的提升而言较温和；二者可以组合，但组合后的工程又多一层。

值得了解。值得实验。还不值得把产品押上去。

```figure
continuous-batching
```

## 动手构建

`code/main.py` 实现了一个 toy Hogwild! simulator：

- 两个 worker processes，每个都是 deterministic “LLM”，会以已知概率生成几类 token（work-token、observe-token、coordinate-token）。
- 一个 shared cache（只是 token list），两个 workers 都读取和写入它。
- 一个简单 coordination logic：当一个 worker 发现另一个 worker 已经在某个 category 中产生了足够 work tokens，它就选择另一个 category。

Simulator 会运行固定 step budget，并报告：

- 生成的 total work-tokens。
- Total wall time（worker steps 数量）。
- 相对单 worker 的 effective speedup。
- 哪个 worker 写了哪个 token 的 trace。

### Step 1：Shared cache

一个两个 workers 都 append 的 list。真实实现中使用简单锁（Python `threading.Lock`）；这里用 counter 模拟。

### Step 2：Worker loop

每个 worker 在每一步：

- 读取当前 shared cache。
- 根据 cache 中已有内容，决定要写入哪一类 token。
- 写入一个 token。

### Step 3：Coordination heuristic

如果 category X 在 cache 中已经有 K 个 tokens，而 worker 本来想写的 category 也是 X，那么 worker 切换到 category Y。这是对 reasoning-model 行为的 toy stand-in：注意到 “这部分已经 covered”，于是做别的事。

### Step 4：Measured speedup

用 N=1 worker 和 N=2 workers 运行 simulator，使用相同 total step budget。统计产生的 work-tokens。N=2 应该产生大约 1.5-1.8x 的 work-tokens，因为 coordination-driven task division 发挥了作用。

### Step 5：Stress the coordination

降低 coordination heuristic 的 sensitivity。再次运行。观察缺乏良好 coordination 时，N=2 会冗余地产生相同 tokens，speedup 会跌到 1 以下。这对应论文中的观察：这个技巧只有在 workers 具备自我协调的 reasoning capacity 时才有效。

## 投入使用

截至 2026 年 4 月，Hogwild! 的 production integration 仍是 research-grade。Yandex/HSE/IST 的 reference implementation 基于 PyTorch，面向 DeepSeek-R1 和 QwQ models 的 single-node multi-process setups。

务实 adoption path：

1. Profile 你的 reasoning-task workload。测量有多少 tokens 是 exploratory（多种 strategies、case analyses、search），有多少是 linear。
2. 如果 exploration 占主导，运行 two-worker Hogwild! experiment。测量 wall-time improvement。
3. 如果 improvement 低于 1.3x，你处在 coordination-dominated regime。退回 single-worker。
4. 如果 improvement 超过 1.5x，推进到 N=4 并再次测量。Diminishing returns 通常会在 N=4-8 附近出现。

和 speculative decoding 组合：每个 Hogwild! worker 都可以独立使用 spec decode。这两个 speedups 大致相乘，把 3x spec decode 和 1.8x Hogwild! 组合起来，可以得到相对 naive single-worker decoding 的约 5.4x 有效提升。

## 交付成果

本课会产出 `outputs/skill-parallel-inference-router.md`。给定 reasoning workload profile（token budget、task parallelism profile、model family、deployment target），它会在 voting、tree-of-thought、multi-agent、Hogwild! 和 speculative decoding strategies 之间做路由。

## 练习

1. 用默认设置运行 `code/main.py`。确认 N=2 Hogwild! configuration 在相同 wall time 内产生的 work-tokens 多于 N=1 baseline。

2. 降低 coordination heuristic 的强度（设置 `coordination_weight=0.1`）。重新运行。展示 speedup collapse。解释原因：workers 无法协调时会重复劳动。

3. 计算一个 50k-token reasoning task 的 expected Hogwild! speedup，条件是 `p=0.8, c=500`，N=4 workers。再对一个 1k-token chat task 做同样计算，条件是 `p=0.3, c=200`，N=4。为什么一个是收益，一个是损失？

4. 阅读 Hogwild! paper 的 Section 4（preliminary evaluation）。找出作者报告的两个 failure modes。描述一个更好的 coordination prompt 可能如何缓解它们。

5. 在 toy 中结合 Hogwild! 和 speculative decoding：每个 worker 内部使用 2-token spec-decode。报告 multiplicative speedup。当两个 workers 都想扩展同一个 shared-cache prefix 时，会出现什么 bookkeeping problem？

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|----------|----------|
| Hogwild! | “Parallel workers, shared cache” | 同一个 LLM 的 N 个实例并发运行，并共享一个 KV cache；通过 self-prompting 涌现 coordination |
| Shared KV cache | “Coordination medium” | 一个单独增长的 KV buffer，所有 workers 都读取和写入；让 workers 之间可以即时看到 tokens |
| Emergent coordination | “不需要 training” | 具备 reasoning capability 的 LLM 可以读取 shared cache 并分工，不需要任何 fine-tuning 或显式 protocol |
| Coordination overhead (c) | “花在 orienting 上的 tokens” | 每个 worker 读取扩展 cache 并决定下一步的成本；必须相对 total decode time 保持较小 |
| Parallelizable fraction (p) | “能并行的部分” | Task-level parallelism：总工作中不是内在 sequential 的比例 |
| RoPE enables Hogwild! | “Rotary positions 是 shift-invariant” | 因为 positions 是 rotations，写入 shared cache 不需要重新计算 prior tokens |
| Voting ensemble | “Run N, pick the majority” | 最简单的 parallel inference topology；适合 classification，不太适合 long-form reasoning |
| Tree of thought | “Branch and prune” | 探索多个 reasoning branches 并剪枝的策略；需要显式 coordination logic |
| Multi-agent framework | “Assign sub-tasks” | 每个 agent 有一个 role，由 coordinator 编排；protocol overhead 很重 |

## 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Hogwild! 论文，包含 QwQ 和 DeepSeek-R1 上的 preliminary evaluation
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — 原始 Hogwild!，也是命名来源
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE，让 shared-cache inference 可行的性质
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — tree-of-thought reasoning strategy，Hogwild! 与它正交
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative decoding，Hogwild! 可以与之组合的 within-sequence parallelism
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — 论文实验的 single source of truth
