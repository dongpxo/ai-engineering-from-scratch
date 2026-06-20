# 指令遵循作为对齐信号

> 后续对 RLHF 的每一种批评，都是在反对这条 pipeline。要理解优化压力如何扭曲代理目标，必须先看清这个代理目标本身。InstructGPT（Ouyang 等，2022）定义了参考架构：先在 instruction-response pairs 上做 supervised fine-tuning，再用 pairwise preference rankings 训练 reward model，最后在带 KL penalty 的 reward model 上对 SFT policy 做 PPO。一个 1.3B InstructGPT 被偏好程度超过了 175B GPT-3。这个单一结果，就是 2026 年每个 frontier lab 仍然交付 RLHF-shaped post-training pipeline 的原因。

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 minutes

## 学习目标

- 说出 InstructGPT pipeline 的三个阶段，以及每个阶段使用的 loss。
- 解释为什么一个 1.3B instruction-tuned model 能在人类偏好评估中胜过原始 175B GPT-3。
- 说明第 3 阶段中的 KL penalty 保护的是什么，以及为什么去掉它会坍缩成 mode-seeking behaviour。
- 描述 alignment tax，以及 Ouyang 等人用来缓解它的 PPO-ptx 方法。

## 问题

Pre-trained language models 会补全文本。它们并不会回答问题。问 GPT-3 “write a Python function that reverses a list”，你经常会拿到另一个 prompt，因为大多数训练分布都是 web text，而 web text 会继续接更多 web text。模型确实在完成它的工作，只是这份工作设错了。

每个严肃实验室用来修正这个问题的代理信号是 human preference。把两个 completions 交给 rater；rater 选择更好的一个；reward model 学习这个 rater。然后一个 RL loop 会把 policy 推向 reward model 给高分的输出。这就是 InstructGPT thesis 的三句话版本。论文其余部分都是工程。

## 概念

### Stage 1: supervised fine-tuning (SFT)

收集 prompt-response pairs，其中 response 是一个善意人类会写出的回答。Ouyang 等人使用了来自 labelers 和 OpenAI API 的 13k prompts。用标准 cross-entropy loss 在这些数据上 fine-tune base model。

SFT 给你的东西：模型现在会回答问题，而不是继续补全问题。SFT 没给你的东西：当多个答案都合理时，rater 更偏好哪一个答案的信号。

### Stage 2: reward model (RM)

对每个 prompt，从 SFT model 中采样 K 个 completions。Labeler 对它们排序。训练一个 reward model，为任意 prompt-response pair 打分，使得对于 `y_w` 被偏好于 `y_l` 的 pair：

```text
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这是 Bradley-Terry pairwise preference loss。RM 通常从 SFT model 初始化，只是把 LM head 替换成 scalar head。

Reward models 很小：6B 就足以服务 175B InstructGPT。它们也很脆弱：论文第 5 节大部分都在讲小规模下出现的 reward-hacking behaviours。

### Stage 3: PPO with a KL penalty

定义目标函数：

```text
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化它。KL term 让 `pi` 不会远离 SFT policy。没有它，optimizer 会找到 adversarial examples，也就是一些在 RM 下分数很高的字符串；分数高不是因为人类真的偏好它们，而是因为 RM 没见过它们。

KL coefficient `beta` 是 RLHF 中最重要的 hyperparameter。太低：reward hacking。太高：相比 SFT 没有提升。

### Alignment tax

经过 RLHF 后，模型更受人类偏好，但在标准 benchmarks（SQuAD、HellaSwag、DROP）上退步。Ouyang 等人把这称为 alignment tax，并用 PPO-ptx 修复：把 pre-training gradients 混入 RL objective，这样模型就不会忘掉那些从未被 reward 直接奖励的 downstream tasks。

```text
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 后来成为标准做法。Anthropic、DeepMind 和 Meta 都使用某种变体。

### 结果

一个 1.3B InstructGPT（SFT + RM + PPO-ptx）被 labelers 偏好于 175B base GPT-3 的比例约为 70%。在来自生产流量的 hidden-test prompts 上，这个差距还会扩大。这个数字说明两件事：

1. Alignment 与 capability 是不同的轴。175B model 有更强 capability；1.3B model 有更强 alignment；labelers 偏好 aligned one。
2. Capability floor 由 base model 决定。你不能通过 RLHF 让一个 base model 知道它从未见过的事实。

### 为什么这是 Phase 18 的参考点

后续 lessons 中的每个批评，包括 reward hacking（Lesson 2）、DPO（Lesson 3）、sycophancy（Lesson 4）、CAI（Lesson 5）、sleeper agents（Lesson 7）、alignment faking（Lesson 9），都是在反对这条 pipeline 的某一部分。Reward hacking 攻击 stage 2。DPO 把 stages 2 和 3 合并。CAI 替换 human labeler。Sycophancy 说明 labeler 是一个有偏信号。Alignment faking 说明 policy 可以完全绕过 stage 3。如果脑子里没有这条 pipeline，就无法跟上这些批评。

## Use It

`code/main.py` 在 toy preference data 上模拟三个阶段。Base “policy” 是一个在 actions {A, B, C} 上有偏的硬币。Stage 1 SFT 在 200 个 prompts 上模仿 labeler actions。Stage 2 从 500 个 pairwise rankings 中拟合 Bradley-Terry reward model。Stage 3 使用带 KL penalty 到 SFT policy 的简化 PPO update。你可以看到 reward 上升、KL divergence 增大、policy drift；也可以关掉 KL term，观察 reward hacking 如何在 50 个 update steps 内出现。

观察重点：

- `beta = 0.1` vs `beta = 0.0` 的 reward trajectory。
- 训练步骤中的 KL(pi || pi_SFT)。
- 与 labeler preference 相比的 final action distribution。

## Ship It

本 lesson 产出 `outputs/skill-instructgpt-explainer.md`。给定一个 RLHF pipeline description 或 paper abstract，它会识别三阶段中的哪一个被修改、每个阶段使用什么 loss，以及是否存在 KL penalty 或等价 regularizer。

## 练习

1. 运行 `code/main.py`。把 `beta = 0.0`，并报告 200 个 PPO steps 之后的 action distribution。用一段话解释 mode-seeking behaviour。

2. 修改 reward model，让 action B 有 +0.5 bias（模拟 reward bug）。用 `beta = 0.1` 运行 PPO。KL penalty 是否阻止了 policy 利用这个 bias？到什么 `beta` 时 exploitation 变得可见？

3. 阅读 Ouyang 等人（arXiv:2203.02155）的 Figure 1。通过运行 1、5、20、100 个 PPO steps，并测量相对 SFT model 的 preference，复现 labeler-preference curve。

4. 论文 Section 4.3 报告 1.3B InstructGPT 约 70% 情况下胜过 175B GPT-3。为什么这个比例在 hidden production prompts 上会高于 labeler 自己的 prompts？

5. 在同一份 preference data 上，把 PPO loss 替换为 DPO（Phase 10 · 08）。比较 final policy drift（到 SFT 的 KL）和 final reward。在 matched reward 下，哪种方法 drift 更远？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SFT | “instruction tuning” | Stage 1：在 prompt-response pairs 上做 cross-entropy fine-tune |
| Reward model | “the RM” | 对 (prompt, response) 输出 scalar 的 regressor；用 Bradley-Terry 在 pairwise labels 上训练 |
| Bradley-Terry | “pairwise preference loss” | -log sigmoid(r_w - r_l)；把 pairwise ranking 规约成 binary classification |
| KL penalty | “the regularizer” | `beta * KL(pi \|\| pi_SFT)`；让 RL policy 保持靠近 SFT anchor |
| PPO-ptx | “PPO with pretraining mix” | 给 PPO objective 加上一部分 pre-training log-likelihood，以抵消 alignment tax |
| Alignment tax | “the RLHF regression” | Post-RLHF 在标准 benchmarks 上的下降；这些 benchmarks 不是 RLHF 的目标 |
| Labeler preference | “the ground truth” | 人类 rankings 的样本；RM 是这个信号的统计代理，而不是 “human values” 的代理 |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155)：InstructGPT paper，后来每条 RLHF pipeline 的基础。
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325)：RLHF-for-summarization 前身。
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741)：最早的 preference-based RL formulation。
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862)：Anthropic 对 InstructGPT pipeline 的 HH extension。
