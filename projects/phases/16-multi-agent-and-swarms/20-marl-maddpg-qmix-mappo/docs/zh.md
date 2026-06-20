# MARL：MADDPG、QMIX、MAPPO

> 这是 multi-agent coordination 的 reinforcement-learning 传承，到 2026 年仍然影响着 LLM-agent systems。**MADDPG**（Lowe et al., NeurIPS 2017, arXiv:1706.02275）提出了 Centralized Training, Decentralized Execution（CTDE）：训练时每个 critic 都能看到所有 agents 的 states 和 actions；测试时只运行本地 actors。它适用于 cooperative、competitive 和 mixed settings。**QMIX**（Rashid et al., ICML 2018, arXiv:1803.11485）是 value-decomposition 方法，使用 monotonic mixing network；per-agent Qs 组合成 joint Q，因此 `argmax` 可以干净地分发到各个 agent 上，并在 StarCraft Multi-Agent Challenge（SMAC）上占据主导。**MAPPO**（Yu et al., NeurIPS 2022, arXiv:2103.01955）是带 centralized value function 的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上只需极少调参就 “surprisingly effective”。这些方法支撑了必须去中心化执行的 agent teams 的 policy training。MAPPO 是 **2026 年 cooperative-MARL 的默认 baseline**。本课会在一个小型 grid-world toy 中构建三者，让你先把这三个想法练成肌肉记忆，再接触 LLM-agent training。

**Type:** Learn
**Languages:** Python（stdlib，小型 NumPy-free implementations）
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## 问题

LLM-agent systems 越来越多地训练用于 agent 间协调的 policies：什么时候 defer，什么时候 act，应该调用哪个 peer。告诉你如何训练这类 policies 的文献是 Multi-Agent Reinforcement Learning（MARL）；它早于 LLM 浪潮，并且已经形成了一小组占主导地位的算法。

如果没有模式词汇，直接读 MARL 论文会很痛苦。Centralized training with decentralized execution（CTDE）、value decomposition、centralized critics 不是 buzzwords，而是针对具体问题的具体答案：

- Independent RL（每个 agent 独自学习）在每个 agent 看来都是 non-stationary。糟糕。
- Centralized RL（一个 agent 控制全部）无法扩展，也违反 execution constraints。
- CTDE 两边都要：训练时使用 global information，部署时使用 local policies。

## 概念

### 论文常用的三个环境

- **Particle World（multi-agent particle env）。** 简单的 2D physics，包含 cooperative/competitive tasks。MADDPG 的原始 testbed。
- **StarCraft Multi-Agent Challenge（SMAC）。** Cooperative micro-management，partial observation。QMIX 的 testbed。Discrete actions，continuous states。
- **Google Research Football、Hanabi、MPE。** MAPPO baselines。

不同环境有不同的 action/observation 类型。算法选择也会随之变化。

### MADDPG（2017）：CTDE pattern

每个 agent `i` 有一个 actor `mu_i(o_i)`，它把自己的 observation 映射到 action。每个 agent 也有一个 critic `Q_i(x, a_1, ..., a_n)`，训练时它能看到所有 observations 和所有 actions。Actor 会根据 critic 的评价用 policy gradient 更新。

```text
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么要 CTDE：训练时我们知道每个人的 actions；用这些信息可以降低每个 critic 的 variance。部署时，每个 agent 只看到 `o_i`，并调用 `mu_i(o_i)`。

Failure mode：critics 会随着 agent 数量 N 增长（输入包含所有 actions）。如果没有近似方法，超过约 10 个 agents 后就很难扩展。

### QMIX（2018）：value decomposition

只适用于 cooperative。Global reward 是 per-agent Q-values 的某个 monotone function：

```text
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

Monotonicity 保证 `argmax_a Q_tot` 可以通过每个 agent 独立选择 `argmax_{a_i} Q_i` 来计算。这正是 decentralized execution 需要的性质。训练时，一个 mixing network 会从 per-agent Qs 生成 `Q_tot`。

为什么 QMIX 在 SMAC 上有效：cooperative StarCraft micro-management 有 homogeneous agents、local obs、global reward，正好适合 value decomposition。

Failure mode：monotonicity constraint 很强；有些任务的 reward structure 不能做 monotone decomposition（例如某个 agent 为团队牺牲）。QTRAN、QPLEX 等扩展放松了这个限制。

### MAPPO（2022）：被低估的默认选择

Multi-Agent PPO：带 centralized value function 的 PPO。每个 agent 有自己的 policy；所有 agents 共享（或各自拥有）能看到 full state 的 value functions。Yu et al. 2022 在五个 benchmark 上把 MAPPO 与 MADDPG、QMIX 及其扩展做对比，发现：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上匹配或超过 off-policy MARL methods。
- 所需 hyperparameter tuning 很少。
- 训练稳定，并且在多个 seeds 上可复现。

在这篇论文之前，社区低估了 on-policy MARL。到 2026 年，MAPPO 是 cooperative MARL 的默认 baseline；任何新方法都必须先打赢它。

### 为什么 LLM-agent engineers 应该关心

三个直接用途：

1. **Router training。** 一个 meta-agent 选择哪个 sub-agent 处理任务。这是一个有 N 个 decentralized sub-agents 和一个 centralized router 的 MARL 问题。MAPPO 适合。
2. **Role emergence。** 在 generative-agent simulations 中，训练 agents 随时间形成互补角色，本质上是一个伪装过的 MARL 问题。QMIX-style value decomposition 会在构造上强制 complementarity。
3. **Multi-agent tool use。** 当 agents 共享 tools 并竞争 budget 时，通过 CTDE 训练它们可以得到可部署的 local policies，并且这些 policies 会尊重 resource constraints。

实践提醒：到 2026 年，大多数 production LLM-agent systems 是 prompt policies，而不是 train policies。只有当你同时具备（a）大量 interaction data，（b）清晰 reward signal，（c）愿意投入 training infrastructure 时，MARL 才会进场。

### CTDE 作为 RL 之外的设计模式

即使不训练，CTDE 也是一个有用的架构模式：

- 在 *design* 阶段，假设能看到完整团队信息。
- 在 *runtime* 阶段，强制 decentralized execution：每个 agent 只能看到 `o_i`。

这个模式会迫使你把 per-agent state 显式化，并提前思考 partial observability。很多 production multi-agent systems 默默假设所有地方都共享状态；CTDE discipline 可以防止这种问题。

### Non-stationarity problem

当多个 agents 同时学习时，每个 agent 的环境（包含其他 agents 的 policies）都是 non-stationary。经典 single-agent RL 证明会失效。本课里的 MARL algorithms 都在处理这个问题：

- MADDPG：global critic 看到所有 actions，因此它的 value estimate 更 stationary。
- QMIX：value decomposition 把学习移动到 joint-Q space，其中 optimality 定义更明确。
- MAPPO：centralized value function 会抑制其他 agents policy changes 带来的 variance。

在 LLM-agent systems 中，non-stationarity 的表现是：“我的 agent 上个月还好好的，现在 upstream 的另一个 agent 一改，我的就乱了。” 用 CTDE 训练 MARL 是 principled fix；prompt-level fixes 更快，但不那么 durable。

### 本课不覆盖什么

训练真实 networks 是 Phase 09 的主题。本课构建 scripted-policy 版本，用来展示 CTDE、value-decomposition 和 centralized-value patterns，不做 gradient updates。目标是在你使用完整 MARL library（PyMARL、MARLlib、RLlib multi-agent）之前，先把模式内化。

## Build It

`code/main.py` 在一个小型 2-agent cooperative grid-world 上实现三个 pattern demonstrations：

- Environment：2 个 agents 在 4x4 grid 上，一个 reward pellet。Reward = 任一 agent 到达 pellet 时为 1；任务结束。
- `IndependentAgents`：每个 agent 把其他 agents 当作环境。Baseline。
- `MADDPGStyle`：centralized critic 计算 joint value；actor policies 从中更新。Scripted policy improvement。
- `QMIXStyle`：带 monotone mixer 的 value decomposition。
- `MAPPOStyle`：centralized value function；policies 根据 shared baseline 更新。

四者运行相同 episodes，并报告 average steps-to-goal。CTDE variants 会比 independent baseline 收敛到更短路径。

运行：

```bash
python3 code/main.py
```

Expected output：independent agents 平均大约需要 6 steps；CTDE variants 收敛到约 3.5 steps（4x4 grid 的最优值是 3）。即使使用 scripted policies，pattern difference 也会显现。

## Use It

`outputs/skill-marl-picker.md` 是一个 skill，会根据给定 multi-agent task 选择 MARL algorithm：cooperative vs competitive、homogeneous vs heterogeneous、action-space type、scale、reward signal。

## Ship It

MARL 在 production 中很少见。真正使用时：

- **从 MAPPO 开始。** 2022 年论文已经把它确立为 baseline；先复现它，可以省下几周追逐更花哨方法的时间。
- **记录每个 agent 的 observation 和 action stream。** 没有 per-agent traces，调试 MARL 基本无望。
- **分离 training code 和 execution code。** CTDE 是一种纪律；让 execution path 真的只能看到 `o_i`。
- **Reward shaping warning。** MARL 对 reward design 极其敏感。Shaping 里一个 coordination bug，agents 就会学会 exploit 它。要跑 adversarial tests。
- **对于 LLM agents**，先考虑 prompt-level policies。只有当 interaction data + reward signal + infrastructure 都具备时，才投入 MARL training。

## 练习

1. 运行 `code/main.py`。测量 independent 和 MAPPO-style agents 之间的 steps-to-goal 差距。在 6x6 grid 上，这个差距会变大还是变小？
2. 实现一个 competitive variant：两个 agents，一个 pellet，只有第一个到达者获得 reward。哪个 pattern 能干净地处理 competition？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）第 3 节。用自己的话把准确的 critic update rule 写成 symbolic pseudocode。
4. 阅读 MAPPO（arXiv:2103.01955）。作者为什么认为 centralized value + PPO 在他们的 benchmarks 上胜过 off-policy MARL？列出三个最强主张。
5. 把 CTDE 作为设计模式应用到一个假想 LLM-agent system（例如 research agent + summarizer + coder）。哪些 joint information 在 design time 可用，但 runtime 不可用？

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| MARL | “Multi-Agent RL” | 面向 multi-agent systems 的 reinforcement learning。 |
| CTDE | “Centralized Training, Decentralized Execution” | 用 global info 训练；用 local policies 部署。 |
| MADDPG | “Multi-Agent DDPG” | CTDE；每个 agent 的 critic 都能看到所有 observations + actions。 |
| QMIX | “Value decomposition” | Per-agent Qs 的 monotonic mixing。Cooperative。 |
| MAPPO | “Multi-Agent PPO” | 带 centralized value function 的 PPO。2026 默认 baseline。 |
| Value decomposition | “Sum of individual Qs” | 把 joint Q 表示为 per-agent Qs 的 monotone function。 |
| Non-stationarity | “Moving targets” | 其他 agents 学习时，每个 agent 的环境都会变化。MARL 的核心问题。 |
| On-policy / off-policy | “Learn from current / replay” | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | “StarCraft Multi-Agent Challenge” | Cooperative micromanagement benchmark；QMIX 的主场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — 对 MAPPO 结果的易读解释
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
