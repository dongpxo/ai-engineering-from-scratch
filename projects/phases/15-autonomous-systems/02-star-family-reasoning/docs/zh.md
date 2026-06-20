# STaR、V-STaR、Quiet-STaR：自学推理

> 最小的自我改进循环藏在 rationale 里。模型生成一条思维链，保留那些最终答案正确的样本，再用它们微调自己。这就是 STaR。V-STaR 加入 verifier，让推理时的候选选择更可靠。Quiet-STaR 则把 rationale 下沉到每个 token。三者都有效。也都不是魔法：这个循环会保留任何碰巧得到正确答案的捷径。

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60 minutes

## 学习目标

- 解释 STaR 如何用“答案正确”的模型生成 rationale 来 bootstrap 推理能力。
- 区分 STaR、V-STaR 和 Quiet-STaR 的训练信号、推理成本和失败模式。
- 说明 rationalization 为什么能扩大训练集，也为什么可能制造不真实的推理轨迹。
- 分析 verifier 如何改善 best-of-N 推理选择，以及它为什么仍可能偏爱自信但错误的 rationale。
- 设计能暴露 shortcut rationale 的 OOD 评估或过程监督替代方案。

## 问题

教模型推理，最直接的方法是收集人工写出的推理轨迹。这个过程昂贵、缓慢，而且受限于人类愿意写出多少高质量 chain-of-thought。

STaR（Self-Taught Reasoner，Zelikman 等，2022）问了一个问题：如果让模型自己写 rationale，再用已知答案给它们打分，会怎样？循环是：

1. 采样一条 reasoning trace 加答案。
2. 如果最终答案正确，保留这条 trace。
3. 用保留下来的 traces 微调模型。
4. 重复。

它确实有效。GSM8K 和 CommonsenseQA 都在没有新增人工标注的情况下得到提升。但这个循环有一个内置偏差：任何得到正确答案的 rationale 都会被保留，不管推理本身是否可靠。V-STaR（Hosseini 等，2024）用一个学习到的 verifier 来修补这个问题；Quiet-STaR（Zelikman 等，2024）把这个思想推广到每个 token 的内部 rationale。

## 概念

### STaR：在有效样本上 bootstrap

从一个具有弱推理能力的 base model 开始。对每个训练问题，采样一条 rationale 加答案。如果答案匹配标签，就保留这个（problem, rationale, answer）三元组。用保留集微调模型。重复。

其中有一个关键变体。如果模型永远答不对某个问题，这个循环就无法从它上面学习。STaR 增加了 **rationalization**：对模型失败的问题，把正确答案作为提示注入，再重新 prompt 模型生成一条能通向这个答案的 rationale。被 rationalize 出来的 rationales 也会进入训练集。

原始论文（Zelikman 等，2022）的结果：一个 GPT-J base model 经过多轮带 rationalization 的 STaR 后，在 GSM8K 上从 5.8% 提升到 10.7%，绝对提升约 5 个百分点。在 CommonsenseQA 上，STaR 训练的 GPT-J 6B 达到 72.5%，接近用人工标注 rationale 微调的 GPT-3 175B（约 73%）——后者模型规模约大 30 倍。

### V-STaR：用 DPO 训练 verifier

STaR 会丢弃错误 rationales。Hosseini 等（2024）观察到，这些也是数据：每一对（rationale, “是否正确”）都可以用来训练 verifier。他们用 Direct Preference Optimization 同时利用正确和错误解法来构建 ranker。推理时，采样 N 条 rationales，再选 verifier 排名最高的一条。

报告中的提升：在 GSM8K 和 MATH 上，比之前的 self-improvement baseline 高出 4 到 17 个百分点；大部分收益来自推理时用 verifier 做候选选择，而不是额外微调 generator。

### Quiet-STaR：每个 token 的内部 rationale

Zelikman 等（2024）问：如果模型不仅在问题和答案之间生成 rationale，而是在每个 token 位置都学习生成一段短的内部 rationale，会怎样？Quiet-STaR 训练模型在预测每个 token 前发出一个隐藏的“thought”，然后通过学习到的权重，把 thought-aware prediction 和 baseline prediction 混合起来。

结果：Mistral 7B 在没有任务特定微调的情况下，GSM8K zero-shot 从 5.9% 绝对提升到 10.9%，CommonsenseQA 从 36.3% 提升到 47.2%。模型学会了“什么时候该思考”：困难 token 会获得更长的内部 rationale；简单 token 几乎不需要。

### 为什么三者共享同一个安全担忧

三种方法都用最终答案作为梯度信号。通过错误推理得到正确答案的 rationale——利用捷径、猜测，或依赖不能泛化的模式——也会被正向强化。在同分布问题上，这个捷径可能有效；在分布外问题上，它会静默失效。

V-STaR 的 verifier 通过学习排序 rationales 来缓解这个问题，但 verifier 也训练在同一组标签上。它可能学会偏爱格式漂亮但推理错误的答案，而不是诚实的不确定。更安全的设计是把 STaR 风格数据与两件事结合起来：（a）过程监督 reward model，也就是奖励中间步骤而不只奖励最终答案；（b）能打破简单捷径的 held-out OOD 评估。

### 对比

| Method | Training signal | Inference cost | Data waste | Known failure mode |
|---|---|---|---|---|
| STaR | 答案正确就保留（rationale, answer） | 1x | 丢弃所有错误 rationales | shortcut rationales |
| STaR + rationalization | 上述信号 + 带正确答案提示的重试 | 1x | 更少 | rationalized rationales 可能不可信 |
| V-STaR | STaR + 用两类样本 DPO 训练 verifier | Nx（best-of-N） | 最少 | verifier 可能强化自信的错误 |
| Quiet-STaR | 每 token rationale + mixing weight | 1.5-3x | 最少 | 仍然是 answer-conditioned gradient |

### 它在 2026 技术栈里的位置

STaR 已经不新了。但这个模式在 2025-2026 年反复出现。可验证数学问题上的 RL（DeepSeek-R1、Kimi-k1.5、o1）就是放大版的 STaR answer-conditioned gradient。过程 reward model（Lightman 等，2023；OpenAI 的 “Let’s verify step by step”）是过程监督版本的替代信号。AlphaEvolve（Lesson 3）是代码领域的 STaR，只是用程序 evaluator 代替标签。Darwin Godel Machine（Lesson 4）则是 agent scaffold 自身的 STaR。

理解 STaR，会让这些系统一下子连起来。它是最小可行的自我改进循环。

```figure
reflection-loop
```

## Use It

`code/main.py` 会在一个玩具算术任务上运行模拟的 STaR 循环。你可以观察：

- 准确率如何随着 bootstrap 轮次上升。
- 捷径如何混入训练：模拟器包含一个 “lazy” rationale 类，它有 40% 的概率得到正确答案，但泛化很差。观察 STaR 是否会保留它们。
- verifier（V-STaR 风格）如何帮助推理时选择候选，但无法彻底清理训练过程中引入的捷径。

## Ship It

`outputs/skill-star-loop-reviewer.md` 帮你在真正训练之前，审计一个 proposed self-taught-reasoning pipeline。

## 练习

1. 运行模拟器。把 shortcut frequency 先设为 0，再设为 0.4。即使两个设置在训练分布上都超过 90%，最终准确率会相差多少？

2. 给模拟器添加一个 held-out OOD 测试。从另一个分布采样问题，并同时评估 bootstrapped model 在同分布和 OOD 集合上的表现。量化这个差距。

3. 阅读 Quiet-STaR 论文（arXiv:2403.09629）第 3 节。分别用三句话解释 “end-of-thought” token 和 mixing-weight head。

4. 比较 STaR 的 keep-if-correct 过滤器和一种过程监督替代方案：后者独立奖励每个 rationale step。指出两者的标注成本差异和可能的质量差异。

5. 设计一个能在部署模型里抓到 shortcut rationales 的评估。它不必完美，但必须能打破 STaR 循环最容易强化的简单捷径。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| STaR | “Self-Taught Reasoner” | 在模型生成且最终答案正确的 rationales 上微调；重复 |
| Rationalization | “Hinted retry” | 对 base model 失败的问题注入正确答案，并重新 prompt 生成 rationale |
| V-STaR | “Verifier STaR” | 在正确和错误 rationales 上用 DPO 训练 verifier，并用于推理时选择 |
| Quiet-STaR | “Per-token rationales” | 在每个 token 位置生成隐藏 thoughts；再与 baseline prediction 混合 |
| Answer-conditioned gradient | “Outcome-based signal” | 训练循环奖励最终答案，而不是推理步骤 |
| Process reward model | “Step-level verifier” | 在每步正确性上训练的 reward model，而不是只看 outcome——与 STaR 形成对照 |
| Shortcut rationale | “Right answer, wrong reasoning” | 通过不能泛化的模式到达标签的 rationale；STaR 会保留它 |

## 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)：原始论文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457)：为推理时选择增加 DPO verifier。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)：每 token 内部 rationale。
- [Lightman et al. (2023). Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)：process reward models，也就是替代梯度信号。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)：可验证任务上的 RL，可以视作扩展到 frontier training 的 STaR。
