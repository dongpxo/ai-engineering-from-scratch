# LLM 的群体优化（PSO、ACO）

> Bio-inspired optimization 正在 LLM 时代回潮。**LMPSO**（arXiv:2504.09247）使用 PSO，其中每个 particle 的 velocity 是一个 prompt，LLM 生成下一个 candidate；它很适合 structured-sequence outputs（数学表达式、程序）。**Model Swarms**（arXiv:2410.11163）把每个 LLM expert 看成 model-weight manifold 上的一个 PSO particle，并报告在 9 个 datasets、12 个 baselines 上有 **13.3% average gain**，每轮只用 200 个 instances。**SwarmPrompt**（ICAART 2025）把 PSO + Grey Wolf 混合用于 prompt optimization。**AMRO-S**（arXiv:2603.12933）是 ACO-inspired pheromone specialists，用于 multi-agent LLM routing——**4.7x speedup**、可解释 routing evidence，以及把 inference 与 learning 解耦的 quality-gated asynchronous update。本课会在 prompt parameter space 上实现 PSO，在 agent routing 上实现 ACO，衡量这些经典算法为什么适合 LLM 时代，以及什么时候不适合。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## 问题

你有一个 prompt，在任务 eval 上得分 62%。你想改进它。朴素做法是 gradient-free manual tweaking，但这很难 scale。Reinforcement learning 需要 reward signals 和足够多的 rollouts 来训练。对 prompts 做 backprop 并不现实——prompt 是离散字符串，不是可微参数。

经典 bio-inspired optimization——用于 continuous search spaces 的 PSO、用于 path selection 的 ACO——正是为这种场景设计的：gradient-free、population-based、单次 evaluation 成本低。把它们和 LLMs 配对，让 LLM 负责 gradient-free search step，你会得到一个意外实用的 optimizer。

同样的模式也适用于 multi-agent systems 中的 agent *routing*。ACO-style pheromone trail 会记录哪个 agent 在哪类 task-type 上效果最好，让 router 利用这条 trail，并让 pheromones 随时间衰减，以便 routes 可以被重新发现。

## 概念

### PSO refresher（Kennedy & Eberhart 1995）

Particle Swarm Optimization：continuous search space 中的一组 particles。每个 particle 有 position `x_i` 和 velocity `v_i`。每轮迭代：

```text
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

其中 `p_best` 是 particle 自己的历史最佳，`g_best` 是 swarm 的全局最佳，`w, c1, c2` 是 inertia + cognitive + social weights，`r1, r2` 是 random factors。

### LLM outputs 上的 PSO — LMPSO

arXiv:2504.09247 把 PSO 适配到 LLM-generated structured outputs（数学表达式、程序）。每个 particle 是一个 candidate output。Velocity 是一个 *prompt*，描述如何把当前 output 朝 personal/global best 修改。LLM 根据 velocity prompt 生成新的 output。Velocity 的 “inertia” 可以是类似 “make small incremental changes” 的 prompt。

它在这些条件下效果很好：

- Output 是 structured（可解析、可评估）。
- Fitness 是 automatic（test runs、arithmetic evaluation）。
- Population 较小（约 10-30 个 particles），所以总 LLM calls 保持可控。

如果 fitness 需要 human review，它就不适合——每轮迭代成本会变得过高。

### Model Swarms

arXiv:2410.11163 把 PSO 从 output layer 移到 *model* layer。每个 “particle” 是一个 expert LLM（parameters）。Swarm 会通过 gradient-free update，把 parameters 朝 collective best 移动。论文报告：在 9 个 datasets 上，相比 12 个 baselines，平均提升 13.3%，每轮只用 200 个 instances。

关键洞察是：LLM expert models 本来就在共享 parameter manifold 中彼此接近（adapter weights、LoRA deltas）。在这个 low-dimensional subspace 上做 PSO 便宜且有效。

### ACO refresher（Dorigo 1992）

Ant Colony Optimization：ants 在 graph 上移动；每条 path 都有 pheromone trail。Ant 的移动概率会按 pheromone strength 加权。完成 task 的 ants 会按 solution quality 沉积 pheromone。Pheromone 会随时间衰减。

### AMRO-S — 用于 agent routing 的 ACO

arXiv:2603.12933 使用 ACO 做 multi-agent routing。每个 task-type 是一个 “destination”；每个 agent 是一条可能 route。产出好结果的 routes 会增强 pheromones。关键贡献：

- **Interpretable routing evidence。** Pheromone strength 是人类可读的 signal。
- **Quality-gated asynchronous update。** 只有 quality checks 通过后才更新 pheromones，把 inference 和 learning 解耦。
- **4.7x speedup**，在 multi-agent routing benchmark 上达成。

Quality gate 很重要：没有它，fast-but-wrong agents 会积累 pheromone，系统会锁死在坏 routes 上。

### 什么时候对 LLMs 使用 PSO / ACO

**使用 PSO，当：**

- Search space 是 continuous，或可以映射到 continuous parameters（prompt embeddings、LoRA weights、numeric generation parameters）。
- Fitness 便宜且 automatic。
- Population 可以较小（10-30）。

**使用 ACO，当：**

- 你有 routing 或 path-selection problem。
- Decisions 会随时间强化（相同 task types 会反复出现）。
- 你需要 routing decisions 的可解释 evidence。

**不要使用二者，当：**

- Fitness 需要 human review（每轮太贵）。
- Search space 是 PSO 无法覆盖的 discrete combinatorial space（改用 genetic algorithms）。
- Real-time decisions 有严格 latency（PSO/ACO 的收敛速度相对 single-pass heuristics 太慢）。

### 为什么 bio-inspired 方法仍然能赢

Gradient-based methods 需要 differentiable signals。LLM outputs 和 routing decisions 并不天然可微。Pseudo-gradient methods（reinforcement-learned routers、DPO-style prompt tuners）可行，但需要昂贵训练。

PSO 和 ACO 只需要一个 *evaluator* function。如果你能给 candidate output 或 routing decision 打分，就能在这个空间上优化。这让适用门槛低得多。

### 实用限制

- **Population budget。** N particles × T iterations × per-eval cost。按 LLM evals 每次约 ~$0.02 计算，一个 20-particle PSO 跑 50 轮约 ~$20。要相应规划。
- **Exploration vs exploitation。** Pheromone decay rate 和 PSO inertia 是 tradeoff；decay 太快会忘记 solutions，太慢会卡在早期 local optima。
- **Catastrophic drift。** 如果 fitness landscape 发生变化（新的 data distribution），两个算法都可能先收敛再发散。要监控 best-fitness stability。

## Build It

`code/main.py` 实现：

- `LMPSO`——在 numeric prompt parameters（temperature、top_k weights）上做 PSO。每个 particle 的 “LLM generation” 被模拟为一个 scripted fitness function。算法运行 30 轮并展示 g_best convergence。
- `AMRO_S`——ACO-style routing。3 个 agents、4 个 task types、pheromone matrix、100 个 routed tasks。打印（task_type → agent choices）随时间变化的分布，以展示 trail formation。
- Comparison：在同一条 task stream 上比较 random routing 和 ACO routing。衡量 quality 和 latency。

运行：

```bash
python3 code/main.py
```

Expected output：

- LMPSO：g_best fitness 从随机值提升到接近最优，持续 30 轮。
- AMRO-S：pheromone table 稳定到每个 task-type 对应的正确 agent；ACO routing 在 quality 上比 random 高约 30-40%，并且也降低 latency（更少 retries）。

## Use It

`outputs/skill-swarm-optimizer.md` 会帮助你在 LLM / agent optimization problems 中选择 PSO、ACO、genetic algorithms 和 gradient-based optimizers。

## Ship It

- **Start small。** 10-20 个 particles，20-50 轮。只有当 convergence curve 显示明确收益时再扩大。
- **Log pheromones or g_best per iteration。** 没有 trail 时调试 swarm optimizers 会很痛苦。
- **Quality-gate updates。** 尤其是 ACO routing：fast-and-wrong agents 不能积累 pheromone。
- **Reset decay on distribution shift。** 当 eval distribution 改变时，老化的 pheromones 是 stale 的；reset，或临时把 decay rate 加倍。
- **Cap the per-iteration cost。** 输出 cost-per-iteration metric。每轮花 $500 但只提升 0.5% 的 PSO 不能发货。

## 练习

1. 运行 `code/main.py`。观察 LMPSO convergence。把 population size 分别改成 5、10、20、50。time-to-converge 在哪个 size 后开始饱和？
2. 实现一个 “catastrophic drift” 实验：在第 30 轮后改变 fitness function。PSO 适应得有多快？重置 `p_best` 有帮助吗？
3. 给 AMRO-S 添加 quality gate：只有 eval score > 0.7 的 runs 才沉积 pheromone。与未 gate 的版本相比，这会如何改变 convergence？
4. 阅读 LMPSO（arXiv:2504.09247）。把论文中的 “velocity as a prompt” 映射回你的 numeric velocity。模拟中丢失了什么，又保留了什么？
5. 阅读 AMRO-S（arXiv:2603.12933）。实现 decoupled “inference fast-path” 和 asynchronous pheromone update。在 sustained load 下，这会如何改变 system latency？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PSO | “Particle Swarm Optimization” | Kennedy-Eberhart 1995。Population-based gradient-free optimizer。 |
| ACO | “Ant Colony Optimization” | Dorigo 1992。通过 pheromone trails 做 path/route optimization。 |
| LMPSO | “PSO with LLM generation” | arXiv:2504.09247。Velocity 是 prompt；LLM 产出 candidates。 |
| Model Swarms | “PSO on expert weights” | arXiv:2410.11163。在 model parameter subspace 上做 gradient-free update。 |
| AMRO-S | “ACO for agent routing” | arXiv:2603.12933。Task-type × agent 上的 pheromone matrix。 |
| p_best / g_best | “Personal / global best” | 单个 particle 和整个 swarm 目前找到的最佳解。 |
| Pheromone | “Routing memory” | Edge 上的强度；随时间衰减；按 quality 沉积。 |
| Quality-gated update | “Only learn from good runs” | Pheromone deposit 以 quality check 通过为条件。 |
| Catastrophic drift | “Distribution shift” | Fitness landscape 改变；旧的 p_best 和 pheromones 变 stale。 |

## 延伸阅读

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — 1995 年 PSO paper。
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — 1992 年 ACO foundations。
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — 用于 structured LLM outputs 的 PSO。
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — model-weight subspace 上的 PSO。
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — 带 quality gate 的 pheromone-driven routing。
