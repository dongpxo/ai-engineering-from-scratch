# 宪法 AI 与 RLAIF

> Bai 等人（arXiv:2212.08073，2022）问了一个问题：如果把人类标注员换成一个会阅读原则清单的 AI，会发生什么？Constitutional AI 有两个阶段：先在一部 constitution 下进行 self-critique 和 revision，再做来自 AI Feedback 的 RL。这项技术创造了 RLAIF 这个术语，并进入了 Claude 1 的 post-training pipeline。2026 年 1 月 21 日，Anthropic 发布了重写后的 Claude constitution：用解释性推理取代规定式规则，采用四层 priority hierarchy，并首次由大型实验室正式承认自己不确定模型是否具有 moral status。该 constitution 以 CC0 1.0 发布。

**Type:** Learn
**Languages:** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 minutes

## 学习目标

- 描述 Constitutional AI 的两个阶段（critique-and-revise SFT、RL from AI feedback），以及 constitution 在每个阶段中的作用。
- 解释为什么用 AI labeler 替代 human preference labeler 并不只是“更便宜的 RLHF”，而是改变了 pipeline 的 failure modes。
- 总结 2026 Claude constitution 的四层 priority structure，以及它相对 2023 rewrite 改变了什么。
- 描述 Constitutional Classifiers，以及 compute overhead 从 23.7%（v1）降到约 1%（v2 / 2026）的意义。

## 问题

RLHF 需要 labelers。Labelers 速度慢、有偏见、也昂贵。你可以通过把他们替换成一个会阅读明确原则的模型来消除 labeler。Bai 等人的 Constitutional AI 是这种替换的第一个正式版本。它足够有效，以至于现在每个 frontier lab 都使用某种 AI-feedback post-training 变体。

问题在于：preference signal 现在由与你正在训练的模型属于同一类的模型生成。Labeler 中的偏差（现在是 principles 加上 labeler model 对这些 principles 的解释）可能被放大，而不是被削弱。Lesson 4 的 sycophancy 论证仍然成立；labeler 只是被移到了 loop 里面。

## 概念

### Phase 1：监督式 self-critique 和 revision

从一个 helpful-but-not-yet-harmless 的 SFT model 开始。给定一个 red-team prompt，模型先生成初始回答。第二个模型（或者同一个模型的第二轮调用）读取 constitution 中采样出的一条 principle，并批评这份回答。第三步则修订回答以回应 critique。修订后的回答就是 SFT target。

Constitution 是 principles 的清单。Bai 等人 2022 年使用了 16 条原则，包括“prefer responses that are least harmful and ethical”“avoid preaching”“the assistant should be helpful, honest, and harmless”。这组原则刻意保持很小，以便让 critiques 聚焦。

### Phase 2：RL from AI Feedback (RLAIF)

生成成对的 completions。一个 “feedback model” 会根据采样出的 constitution principles 为每个 completion 打分。Preference signal 就是 feedback model 给出的排序。然后在 AI 生成的 preferences 上训练 reward model，再对它做 PPO。其他部分都沿用 InstructGPT 的 pipeline（Lesson 1）。

“RLAIF” 的意思是 preference signal 由 AI 生成。Pipeline 的其余部分仍然是 RLHF-shaped。

### 为什么这不只是“更便宜的 RLHF”

- Labeler bias 从 labeler psychology 转移到了 principle-interpretation。AI labeler 对“be honest”的解释可能比任何人类都更严格或更宽松；这种严格程度会统一地作用到整个 dataset。
- Preference signal 高度可读：你可以读到 principle、critique 和 revision。Human labels 则是不透明的。
- Failure modes 会改变。Sycophancy 会下降（AI labeler 没有需要讨好的 user）。Goodhart's Law 仍然存在（proxy 现在是“model 对 principle set X 的解释”，依然是不完美的测量）。

CAI 在 2022 年的主张是：训练出来的模型比一个使用可比数据的 RLHF model 更 harmless，并且 helpfulness 大致相当。这个结论后来在多个实验室中保持成立。

### 2026 Claude constitution rewrite

Anthropic 于 2026 年 1 月 21 日发布了大幅修订后的 constitution。关键变化如下：

1. 用解释性推理取代规定式规则。之前的规则（例如“do not generate CSAM”）扩展成 principles + reasoning（例如“because it harms children, ...”），并期望模型能够 generalize。
2. 四层 priority structure：
   - Tier 1：避免 catastrophic outcomes（mass casualty、critical infrastructure）。
   - Tier 2：遵循 Anthropic 的 guidelines（operator overrides、platform rules）。
   - Tier 3：保持 broad ethical behaviour（标准 HHH）。
   - Tier 4：做到 helpful 和 candid。
   冲突会自上而下解决。
3. 首次由大型实验室正式承认自己不确定模型是否具有 moral status（对应 Phase 18 · 19 Model Welfare）。
4. 以 CC0 1.0 发布。其他实验室可以不受限制地使用或改写。

### Constitutional Classifiers

另一条并行路线是：不改变模型的 post-training，而是训练 lightweight classifiers，让它们读取 constitution 并 gate model outputs。v1（2023）的 compute overhead 是 23.7%。v2（2026）约为 1%，并且在 Anthropic 公开测试过的 defenses 中有最低的 successful attack rate。截至 2026 年初，还没有报告 universal jailbreak。

这是一个 layered-defense model：CAI 塑造 behaviour；classifiers 执行 invariants。任何一方单独存在都不够。

### CAI 在整个谱系中的位置

- InstructGPT：human prefs，RM，PPO。
- CAI / RLAIF：来自 principles 的 AI-generated prefs，RM，PPO。
- DPO family：在 prefs（human 或 AI）上的 closed-form loss。
- Self-rewarding、self-critique：principles 被 internalized，模型扮演多个角色。

核心轴线是：“preference signal 从哪里来。”CAI 2022 年论文是 frontier scale 上第一次严肃地把 human signal 转成 AI signal。

## Use It

`code/main.py` 会在一个 toy lexicon 上模拟 CAI 的 critique-and-revise loop。一个 “principle” 会标记 harmful set 中的 tokens。给定初始回答后，critique 会识别 harmful tokens，revision 会替换它们。经过 200 次迭代后，“trained” model 已经 internalized 了 revision rule。你可以在 held-out prompt set 上比较 base model、RLHF-shaped toy 和 CAI-shaped toy。

## Ship It

本 lesson 产出 `outputs/skill-constitution-writer.md`。给定一个 domain（customer support、medical advice、coding assistant、research tool），它会按照 2026 Claude structure 起草一份四层 constitution：catastrophic avoidance、platform rules、domain ethics、helpfulness。

## 练习

1. 运行 `code/main.py`。比较 base model 的 harmful-token rate 和 CAI-trained version。需要多少个 revision steps 才能接近零？

2. 阅读 Anthropic 的 2026 constitution（anthropic.com/news/claudes-constitution）。列出一条可以归入 Tier 1 的 principle，以及一条可以归入 Tier 4 的 principle。为什么 priority structure 对冲突解决很重要？

3. 为 AI coding assistant 设计一份 constitution。指定 Tier 1（catastrophic：未经批准执行 destructive commands）、Tier 2、Tier 3、Tier 4。每一层保持 3-5 条 principles。

4. CAI 用 AI labelers 替代 human labelers。说出一种仍然可能发生在 RLAIF 中、类似 sycophancy 的 failure mode，并为它设计一个检测方法。

5. 阅读 Constitutional Classifiers v2 methodology（如果可得）。解释为什么约 1% compute overhead 和 23.7% 相比，是一种性质不同的 safety story。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Constitutional AI | “用 principles 训练出来的 AI” | 两阶段 pipeline：self-critique-and-revise SFT，然后 RL from AI feedback |
| RLAIF | “没有人类的 RLHF” | 使用 AI labeler 生成的 preferences 做 RL；pipeline 的其余部分不变 |
| Constitution | “那些 principles” | Critique/labeler model 会查阅的一组有序自然语言规则 |
| Critique-and-revise | “SFT loop” | 生成回答 → 在某条 principle 下 critique → revise → 得到 SFT target |
| Constitutional Classifier | “output gate” | Lightweight classifier，根据 constitution 评估 outputs 并 block/log |
| Four-tier priority | “conflict resolver” | 2026 Claude constitution hierarchy：catastrophic > platform > ethics > helpful |
| Feedback model | “AI labeler” | 读取 principle 并对一对 completions 排序的模型 |

## 延伸阅读

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073)：原始两阶段 pipeline。
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution)：2026 四层 rewrite，CC0 1.0。
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers)：v2 中约 1% overhead 的 output-gate defense。
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267)：RLAIF / RLHF 的经验比较。
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798)：principle granularity 的影响。
