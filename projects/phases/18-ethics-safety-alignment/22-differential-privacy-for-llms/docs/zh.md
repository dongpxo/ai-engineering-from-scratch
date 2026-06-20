# 面向 LLM 的差分隐私

> DP-SGD 仍然是标准方法：向 gradient updates 注入噪声，提供形式化的 (epsilon, delta) guarantees。Compute、memory 和 utility 的 overhead 都很大；parameter-efficient DP fine-tuning（LoRA + DP-SGD）是 2025 年常见配置（ACM 2025）。两组证据彼此紧张：canary-based membership inference（Duan 等人，2024）报告它对 language models 的成功有限；training-data extraction（Carlini 等人，2021；Nasr 等人，2025）则能恢复大量逐字 memorization。调和解释（arXiv:2503.06808，2025 年 3 月）：差异来自测量对象不同，即 inserted canaries vs “most extractable” data。新的 canary designs 使不依赖 shadow models 的 loss-based MIA 成为可能，并给出了第一个在真实数据、现实 DP guarantees 下训练 LLM 的非平凡 DP audit。替代路线：PMixED（arXiv:2403.15638），在 inference time 通过 next-token distributions 上的 mixture of experts 实现 private prediction；以及 DP synthetic data generation（Google Research 2024）。新兴攻击：Differential Privacy Reversal via LLM Feedback，也就是 confidence-score leakage。

**Type:** Build
**Languages:** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**Prerequisites:** Phase 01 · 09 (information theory), Phase 10 · 01 (large-model training)
**Time:** ~60 minutes

## 学习目标

- 定义 (epsilon, delta)-differential privacy，并说出 DP-SGD recipe。
- 解释 2024-2025 年的张力：canary MIA 和 training-data extraction 给出了不同图景。
- 描述 PMixED，以及为什么 inference-time private prediction 是 DP training 的替代路线。
- 描述 Differential Privacy Reversal via LLM Feedback attack。

## 问题

LLMs 会记忆。Carlini 等人 2021 年展示了 production language models 会按需逐字复现 training text。DP 是形式化防御：训练过程应让输出对任何单个 training example 都 provably insensitive。2024-2025 年的证据表明，DP-SGD 是必要的，但部署中的 ε values 未必匹配 threat model。

## 概念

### (ε, δ)-differential privacy

如果对于任意两个只相差一个 example 的 datasets，以及任意 event S，一个 randomized algorithm M 满足：

P(M(D) in S) <= e^ε * P(M(D') in S) + δ.

那么 M 是 (ε, δ)-DP。

解释：output distribution 足够接近（由 ε 参数化），因此任何单个个体的贡献都不能被可靠推断，除了概率 δ 所允许的例外。

### DP-SGD

Abadi 等人 2016。标准 recipe：

1. 采样一个 mini-batch。
2. 计算 per-example gradients。
3. 把每个 per-example gradient clip 到阈值 C。
4. 对 clipped gradients 求和，并加入 std 为 σ * C 的 Gaussian noise。
5. 使用 noisy sum 更新参数。

Privacy cost 由 accountant 追踪（Moments Accountant、Rényi DP accountant）。LLM literature 中报告的 ε values 会随 threat model、data sensitivity 和 utility target 大幅变化；不存在 universally “safe” default ε。已发表例子在一些 LLM training settings 中大致覆盖 ε ≈ 1–10，但这些只是 illustrative，不是推荐默认值。更低的 ε 通常需要更多 noise，并可能增加 utility loss。

### LoRA + DP-SGD

对 frontier model 做 full DP-SGD 代价过高。LoRA（Hu 等人 2022）把 gradient updates 限制在一个小 adapter 上，减少 per-example gradient storage。LoRA + DP-SGD 是 2025 年常见配置。DP guarantees 适用于 adapter；base model 保持 fixed。

### 2024-2025 年的张力

两条证据线：

- **Canary MIA（Duan 等人 2024）。** 向 training data 插入 unique canaries，测量 membership-inference attacker 是否能识别它们。报告显示在 language models 上成功有限。这暗示 MIA 很难。
- **Training-data extraction（Carlini 2021，Nasr 等人 2025）。** 用 prefix prompt 模型，测量它是否恢复 training 中的逐字文本。报告显示 memorization 很明显。这暗示在相关意义上 MIA 很容易。

2025 年 3 月的调和解释（arXiv:2503.06808）：二者测量的是不同东西。MIA 在 inserted canaries 上问 “example e 是否在 D 里？” Extraction 问 “我能从 D 中恢复什么？” 对 privacy 来说，真正重要的是 “most extractable” example；canaries 会低估这一点，因为它们没有被优化成 extractable。

新的 canary designs。无需 shadow models 的 loss-based MIA。第一个针对真实数据、现实 DP guarantees 下 LLM 的非平凡 DP audit。

### DP training 的替代路线

- **PMixED（arXiv:2403.15638）。** 在 inference time 做 private prediction。Next-token distributions 上的 mixture of experts；每个 expert 看到一个 training data shard；aggregation 加噪以实现 DP。完全避开 DP training。
- **DP synthetic data generation（Google Research 2024）。** 用 DP-SGD 做 LoRA fine-tune，采样 synthetic data，再用 synthetic data 训练 downstream classifier。

二者都绕开了 full DP training 的 utility cost，但代价是采用不同 threat model。

### Differential Privacy Reversal via LLM Feedback

2025 年新兴攻击。把 DP-trained model 的 confidence scores 当作 oracle，用来 re-identify individuals。即使 outputs 不泄漏，confidence distributions 也可能泄漏。

防御方式：不要暴露 confidences，或者在暴露前 truncate/quantize 它们。这是 (ε, δ)-DP training 之外的额外要求。

### 它在 Phase 18 中的位置

Lessons 20-21 是 bias/fairness。Lesson 22 是 privacy。Lesson 23 是通过 watermarking 做 provenance。Lesson 27 覆盖 regulatory data-provenance layer。

## Use It

`code/main.py` 会在一个 toy binary-classification dataset 上模拟 DP-SGD。你可以 sweep noise multiplier σ 和 clipping norm C，追踪 (ε, δ) budget 以及 accuracy cost。一个 “canary attack” 会插入 unique training example，并测量 log-loss test 在 DP 前后是否能检测它。

## Ship It

本 lesson 产出 `outputs/skill-dp-audit.md`。给定一个关于 language model deployment 的 DP claim，它会 audit：(ε, δ) values、使用的 accountant、MIA evaluation protocol，以及 confidence-exposure vectors 是否已被评估。

## 练习

1. 运行 `code/main.py`。Sweep σ in {0.5, 1.0, 2.0}，并报告 (ε, δ)-accuracy trade-off。找出 utility collapse 的位置。

2. 实现 canary insertion 和 log-loss test。测量 σ = 1.0 时，DP-SGD 前后的 detection rate。

3. 阅读 Nasr 等人 2025 年关于 training-data extraction 的论文。为什么 extraction success 不会在 moderate ε 下 collapse？这对 MIA-as-evaluation 意味着什么？

4. 设计一个使用 PMixED（arXiv:2403.15638）的 deployment，使其完全在 inference time 运行。PMixED 处理了 DP-SGD 没有处理的哪种 threat model？

5. 勾勒 DP Reversal via LLM Feedback attack。设计一个限制 confidence-score leakage 的 countermeasure，并估计它的 deployment cost。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DP | “(ε, δ)-differential privacy” | 形式化 privacy：当 neighbouring dataset 改变时，output distribution 仍然接近 |
| DP-SGD | “noise-injected SGD” | Gradient clipping + Gaussian noise addition；标准 DP training |
| LoRA + DP-SGD | “efficient private fine-tune” | 在 low-rank adapters 上做 DP-SGD；2025 年标准配置 |
| MIA | “membership inference” | 判断某个 example 是否出现在 training data 中的 attack |
| Canary | “inserted watermark example” | 用来测量 DP leakage 的 unique training example |
| PMixED | “private inference mixture” | 在 next-token distributions 上通过 mixture-of-experts 实现 inference-time DP |
| DP Reversal | “confidence leakage attack” | 使用模型 confidence 作为 re-identification oracle 的 attack |

## 延伸阅读

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133)：标准 DP training algorithm。
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805)：canonical extraction paper。
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841)：limited-success MIA。
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808)：对上述张力的调和解释。
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638)：inference-time private prediction。
