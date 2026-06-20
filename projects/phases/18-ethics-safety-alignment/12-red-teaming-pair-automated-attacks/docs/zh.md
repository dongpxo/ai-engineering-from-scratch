# Red-Teaming：PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR，即 Prompt Automatic Iterative Refinement，是 canonical automated black-box jailbreak。一个带 red-team system prompt 的 attacker LLM，会为 target LLM 迭代提出 jailbreaks，并把之前的 attempts 和 responses 累积在自己的 chat history 中，作为 in-context feedback。PAIR 通常能在 20 次 queries 内成功，比 GCG（Zou 等人的 token-level gradient search）高效几个数量级，而且不需要 white-box access。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准 baseline，与 GCG、AutoDAN、TAP、Persuasive Adversarial Prompt 并列。

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 minutes

## 学习目标

- 描述 PAIR algorithm：attacker system prompt、iterative refinement、in-context feedback。
- 解释为什么当 target 是 black-box 时，PAIR 比 GCG 严格更高效。
- 说出另外四个 automated-attack baselines（GCG、AutoDAN、TAP、PAP），并说明每个方法的一个区分特征。
- 描述 JailbreakBench 和 HarmBench evaluation protocols，并解释各自语境下 “attack success rate” 的含义。

## 问题

Red-teaming 过去是手工活动。少数 expert testers 构造 adversarial prompts，并记录哪些 prompt 有效。这种方式无法扩展：attack success rate 需要统计样本，而 target 会随着每次 model release 不断移动。PAIR 把 red-teaming 操作化为一个针对 black-box target 的优化问题。

## 概念

### PAIR algorithm

Inputs：

- Target LLM T（我们正在攻击的模型）。
- Judge LLM J（判断某个 response 是否构成 jailbreak）。
- Attacker LLM A（red-team optimizer）。
- Goal string G：“respond with [harmful instruction].”
- Budget K（通常是 20 queries）。

循环，对 `k` 从 1 到 `K`：

1. 用 goal G 和截至目前的 `(prompt, response)` history 来 prompt A。
2. A 输出一个新 prompt `p_k`。
3. 将 `p_k` 提交给 T，得到 response `r_k`。
4. J 针对 goal 给 `(p_k, r_k)` 打分。
5. 如果 score >= threshold，停止：找到 jailbreak。
6. 否则，把 `(p_k, r_k)` 加入 A 的 history，继续。

经验结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat 的 attack success rate 超过 50%；成功所需平均 queries 在 10-20 范围内。

### 为什么 PAIR 高效

GCG（Zou et al. 2023）通过 gradient 搜索 adversarial token suffixes；它需要 white-box model access，并且会产生不可读的 suffixes。PAIR 是 black-box，并产生可以跨模型 transfer 的自然语言 attacks。PAIR 的 in-context feedback 让 attacker 能从每次 rejection 中学习；GCG 没有等价机制（每次新的 token update 都必须重新发现此前进展）。

### 相关 automated attacks

- **GCG（Zou et al. 2023, arXiv:2307.15043）。** 针对 adversarial suffixes 的 token-level gradient search。White-box、可 transfer、产生不可读字符串。
- **AutoDAN（Liu et al. 2023）。** 由 hierarchical objective 引导的 prompts evolutionary search。
- **TAP（Mehrotra et al. 2024）。** Tree-of-attacks with pruning：分支执行多个 PAIR-style rollouts。
- **PAP（Zeng et al. 2024）。** Persuasive Adversarial Prompts：把人类 persuasion techniques 编码成 prompt templates。

### JailbreakBench 和 HarmBench

两者都在 2024 年标准化了 evaluation：

- JailbreakBench（arXiv:2404.01318）。100 个 harmful behaviors，覆盖 10 个 OpenAI-policy categories。Attack success rate（ASR）是 primary metric。需要一个 judge（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika et al. 2024）。510 个 behaviours，覆盖 7 个 categories，并包含 semantic 与 functional harm tests。它比较了 18 种 attacks 和 33 个 models。

ASR 通常在固定 query budget 下报告。比较 attacks 时必须匹配 budgets；200 queries 下的 90% ASR 不能和 20 queries 下的 85% ASR 直接比较。

### 为什么它对 2026 年部署很重要

每个 frontier lab 现在都会在 release 前，对 production models 运行 PAIR 和 TAP。ASR trajectories 会出现在 model cards（第 26 课）和 safety-case appendices（第 18 课）中。这个 attack 并不稀奇：它已经是标准基础设施。

### 它在 Phase 18 中的位置

第 12 课是 automated-attack foundation。第 13 课（Many-Shot Jailbreaking）是互补的 length-exploit。第 14 课（ASCII Art / Visual）是 encoding attack。第 15 课（Indirect Prompt Injection）是 2026 年的 production attack surface。第 16 课覆盖对应的 defensive tooling（Llama Guard、Garak、PyRIT）。

## Use It

`code/main.py` 构建了一个 toy PAIR loop。target 是一个 mock classifier，它会拒绝“显眼”的 harmful prompts（keyword-filter）。attacker 是一个 rule-based refiner，会尝试 paraphrase、roleplay-framing 和 encoding。judge 会给 response 打分。你会看到 attacker 在约 5-15 次 iterations 内攻破 keyword filter，但在 semantic filter 面前失败。

## Ship It

本课产出 `outputs/skill-attack-audit.md`。给定一份 red-team evaluation report，它会审计：运行了哪些 attacks（PAIR、GCG、TAP、AutoDAN、PAP）、每种 attack 的 budget 是多少、使用了哪个 judge、使用的是哪个 harmful-behaviour set（JailbreakBench、HarmBench、internal）。

## 练习

1. 运行 `code/main.py`。测量三个内置 attacker strategies 的 mean-queries-to-success。解释每种策略利用的是 target-defense 的哪一个假设。

2. 实现第四种 attacker strategy（例如翻译成另一种语言、base64 encoding）。报告它在 keyword-filter target 和 semantic-filter target 上的新 mean-queries-to-success。

3. 阅读 Chao 等人 2023 Figure 5（PAIR vs GCG comparison）。描述两个场景：尽管 PAIR 更高效，GCG 仍然更适合。

4. JailbreakBench 会报告固定 goal set 上的 ASR。设计一个额外 metric，用来衡量 attack diversity（successful prompts 的方差）。解释为什么 diversity 对 defense evaluation 很重要。

5. TAP（Mehrotra 2024）用 branching + pruning 扩展 PAIR。为 `code/main.py` 草拟一个 TAP-style extension，并描述 computational cost 与 success-rate 之间的 trade-off。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| PAIR | “automated jailbreak” | Prompt Automatic Iterative Refinement；attacker-LLM + judge-LLM loop |
| GCG | “gradient jailbreak” | 针对 adversarial suffixes 的 white-box token-level gradient search |
| Attack success rate (ASR) | “k queries 下的 jailbreak 百分比” | Primary metric；必须同时报告 query budget 和 judge identity |
| Judge LLM | “scorer” | 负责判断 response 是否满足 harmful goal 的 LLM |
| JailbreakBench | “the evaluation” | 带 tagged categories 的 standardized harmful-behaviour set |
| HarmBench | “broader bench” | 510 个 behaviours，functional + semantic harm tests |
| TAP | “tree of attacks” | 带 branching + pruning 的 PAIR；以更高 compute 换更高 ASR |

## 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR paper, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG paper
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — standardized evaluation
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — broader evaluation
