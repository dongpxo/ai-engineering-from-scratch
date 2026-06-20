# 直接偏好优化家族

> Rafailov 等人（2023）证明，RLHF 的最优解可以用偏好数据写成闭式形式，因此可以跳过显式奖励模型，直接优化策略。这个洞见催生了一个家族：IPO、KTO、SimPO、ORPO、BPO，它们分别修补 DPO 的一种失效模式。到 2026 年，直接对齐算法在前沿模型 post-training 中比 PPO 更常见。但第 2 课的过度优化曲线仍然适用：DAA 并没有逃离 Goodhart，它只是改变了 Goodhart 咬住系统的位置。

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 的闭式形式。
- 说明 IPO、KTO、SimPO、ORPO、BPO 分别修补 DPO 的哪一种失效模式。
- 区分“隐式奖励差距”和“偏好强度”，并解释 IPO 的 identity mapping 为什么重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明了：即使没有显式奖励模型，DAA 仍会过度优化。

## 问题

RLHF 目标函数（第 1 课）：

```text
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知最优解：

```text
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此，奖励可以由最优策略和参考策略的比值隐式定义：

```text
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

把它代入 Bradley-Terry 偏好似然之后，partition function `Z(x)` 会相消，因为它只依赖 `x`。剩下的是一个只含策略参数的 loss：不再需要奖励模型。这就是 DPO。

麻烦在于：这个推导假设最优解可达，偏好数据在分布内，并且参考策略是真实的 mode anchor。这些假设没有一个会完全成立。家族中的每个成员都在修补一个不同的被破坏假设。

## 概念

### DPO（Rafailov et al., 2023）

```text
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 没有上界。一个很小的偏好也可能产生任意大的 gap。
- 这个 loss 会把 chosen 和 rejected 的 log-prob 往相反方向推。只要 rejected 掉得更快，它就可以让 chosen 的绝对 log-prob 也下降。这就是 Degraded Chosen Response 现象。
- 分布外偏好（罕见样本对罕见样本）会产生任意的隐式奖励。

### IPO（Azar et al., 2024）

Identity Preference Optimization 用 preference probability 上的 identity mapping 替换 log-sigmoid。loss 变成一个有界目标上的平方误差：

```text
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

margin 被 `1/(2 beta)` 限制住。偏好强度和隐式奖励差距成比例，不会爆炸。

### KTO（Ethayarajh et al., 2024）

Kahneman-Tversky Optimization 完全放弃 pairwise structure。给定单个带标签输出，以及一个二值的“desirable”或“undesirable”信号，它会映射到一个 prospect-theory utility：

```text
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

并对 gains 和 losses 使用不同权重（loss aversion）。收益是：你可以使用未配对数据，而这类数据多得多。

### SimPO（Meng et al., 2024）

Simple Preference Optimization 让训练信号与生成过程对齐。它完全移除参考策略，并按长度归一化 log-likelihood：

```text
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

其中 `gamma` 是用于稳定训练的 margin。长度归一化去除了利用 DPO 长度偏置失效模式的动机：在 DPO 中，更长的 `y_w` 会天然给出更大的 log-prob gap。

### ORPO（Hong et al., 2024）

Odds-Ratio Preference Optimization 在标准 SFT negative log-likelihood 上加入一个偏好项：

```text
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

不需要参考策略：SFT 项就是 regularizer。它可以从 base model 到 aligned model 单阶段训练，不需要单独的 SFT checkpoint。

### BPO（ICLR 2026 submission, OpenReview id=b97EwMUWu7）

BPO 识别出 Degraded Chosen Responses 问题：DPO 会保留排序 `y_w > y_l`，但 `y_w` 的绝对 log-prob 可能下降。BPO 添加一行修正，对 chosen response 的下移进行惩罚。论文报告在 Llama-3.1-8B-Instruct 的 math reasoning 上，相比 DPO 提升了 +10.1% accuracy。

### 通用结论：DAA 仍会过度优化

Rafailov 等人的 “Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms”（NeurIPS 2024）在多个数据集和 KL budgets 下，用 DPO、IPO、SLiC 训练策略。gold-reward-vs-KL 曲线具有和 Gao 等人相同的 peak-and-collapse 形状。训练期间，隐式奖励会查询分布外样本；KL regularization 无法稳定这一点。

DAA 并没有逃离 Goodhart。它们把咬合面从“奖励模型被过度优化”变成了“参考策略比值被过度优化”。通用修复手段也一样：更好的数据、ensembles、early stopping。

### 如何选择（2026）

- 如果你有大量 paired preference data：使用带保守 `beta` 的 DPO；如果长度偏置明显，考虑 SimPO。
- 如果你有 unpaired binary feedback：使用 KTO。
- 如果你想从 base model 开始做单阶段 pipeline：使用 ORPO。
- 如果你在 DPO 日志中看到 chosen log-probs 退化：使用 BPO。
- 如果 preference strengths 差异很大，并且 DPO 正在 saturate：使用 IPO。

每个 lab 都会在一组 battery 上跑这五种方法，然后按任务选择赢家。没有理由假设 math reasoning 和 safety 的最优 loss 是同一个。

```figure
dpo-margin
```

## Use It

`code/main.py` 会在一个 toy preference dataset 上比较六种 loss（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度会随 pair 改变。每个 loss 都会在同一份 500-pair sample 上优化一个小型 softmax policy。脚本会绘制每种方法的最终 win rate、chosen-log-prob drift 和 implicit-reward spread。

## Ship It

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计信息（paired vs unpaired、variable vs uniform preference strength、length distribution）以及目标（single-stage 或 SFT-then-preference），它会推荐一种 preference loss，并说明它防护的是哪种失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终 chosen-log-prob drop。BPO 应该保留更高的 chosen absolute probability：验证这一点。

2. 修改 preference data，让所有 pairs 具有相同强度。六种方法中哪一种最稳健？哪一种会退化？解释 IPO 在这里的优势。

3. 让 rejected responses 的平均长度变成 chosen 的 2 倍。在其他条件不变的情况下，数值展示 DPO 的 length exploitation，以及 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 会过度优化。复现一个 single-point 版本：绘制 chosen-minus-rejected KL divergence，并观察大 `beta` 下 DPO 的过度优化。

5. 阅读 BPO paper abstract（OpenReview b97EwMUWu7）。写下 BPO 添加到 DPO 上的那一行 correction。和 `code/main.py` 中的实现核对。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DPO | “没有奖励模型的 RLHF” | 从 RLHF 闭式最优解推导出的 loss；只含策略参数 |
| Implicit reward | “那个 log-ratio” | `beta * log(pi(y\|x) / pi_ref(y\|x))`，也就是 DPO 隐含的奖励 |
| IPO | “有界 DPO” | 用 identity 替换 log-sigmoid；隐式奖励 gap 被 `1/(2 beta)` 截断 |
| KTO | “未配对 DPO” | 带 loss aversion 的单标签 prospect-theory utility |
| SimPO | “无参考策略 DPO” | 长度归一化 log-likelihood + margin；没有 reference policy |
| ORPO | “单阶段 DPO” | NLL + odds-ratio preference term；一轮训练从 base 到 aligned |
| BPO | “保留 chosen 的 DPO” | DPO 加上对 chosen response 绝对 log-prob 下降的惩罚 |
| Degraded Chosen | “chosen 也掉了” | DPO 让 chosen log-prob 下降，只要 rejected 下降得更快 |
| DAA | “直接对齐算法” | 任何跳过显式 RM 的 preference-loss 方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
