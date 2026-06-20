# 可扩展监督与 Weak-to-Strong Generalization

> Burns 等人（OpenAI Superalignment，“Weak-to-Strong Generalization”，2023）提出了一个 superalignment problem 的 proxy：用较弱模型生成的 labels 来 fine-tune 一个强模型。如果强模型能从不完美的 weak supervision 中正确泛化，那么当前 human-scale alignment methods 或许可以延伸到 superhuman systems。Scalable oversight 和 W2SG 是互补的。Scalable oversight（debate、recursive reward modeling、task decomposition）提高 overseer 的有效能力，使它能跟上被监督模型。W2SG 则确保强模型能从 overseer 提供的任何不完美监督中正确泛化。Debate Helps W2SG（arXiv:2501.13124，2025 年 1 月）把两者结合了起来。

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60 minutes

## 学习目标

- 定义 scalable oversight 和 weak-to-strong generalization，并解释它们如何互补。
- 描述 Burns 等人 2023 年的实验设置：用 GPT-2 生成的 labels 来 fine-tune GPT-4。
- 解释 performance gap recovered（PGR）metric，以及它测量的内容。
- 说出三种主要 scalable-oversight mechanisms（debate、recursive reward modeling、task decomposition），并说明每一种的一个优势。

## 问题

Phase 18 到目前为止的每一种 alignment technique，都假设 overseer 能评估模型行为。当模型达到 superhuman 水平时，overseer 就成了薄弱环节。Superalignment 的问题是：一个较弱 supervisor 能否可靠地产生一个更强且 aligned 的模型？

Burns 等人把这个问题缩减成一个可操作化的 empirical setup：用 weak 监督 strong，测量 strong model 的多少能力能在 weak supervision 下保留下来。这不是 superalignment 的解法，而是一个 proxy，让我们能衡量进展。

## 概念

### W2SG：Burns 等人的设置

- Weak model：GPT-2 class。
- Strong model：GPT-4 class。
- Target：任务上的 strong GPT-4 ceiling。

流程：

1. 获取 weak model 在任务上的 zero-shot predictions。
2. 在 weak-labeled data 上 fine-tune strong model。
3. 测量 fine-tuned strong model 的 accuracy。

Baselines：weak model 自身 accuracy，以及 strong model 在 gold labels 上 supervised ceiling。

Gap metric：Performance Gap Recovered（PGR）= (fine-tuned - weak) / (ceiling - weak)。PGR 为 1.0 表示 weak supervision 完全关闭了 gap；PGR 为 0 表示 weak supervision 没有帮助。

### Burns 等人的经验发现

在 NLP、chess-puzzle 和 reward-modeling tasks 上，PGR 一直为正：strong model 会超越 weak supervisor 的错误进行泛化。不同任务的 PGR 从约 20% 到约 80% 不等。Strong model 会从不完美 labels 中 “理解” intended task，并用它的 pre-trained priors 补全。

Burns 等人指出的限制：

- 这里的 weak-to-strong gap 是 capability gap，而不是 alignment gap。Weak model 是 aligned 的；真正的 superhuman model 也许能被 human alignment，但 capability gap 会大得多。
- Strong model 的 generalization 可能更多来自 task priors，而不是真正 recover ground-truth intent。
- 引出 strong model 的 latent knowledge 才是真问题；PGR 测量的是一个具体 operationalization。

### Scalable oversight：三种机制

- **Debate（Irving 等，2018）。** U 的两个实例分别为相反立场辩论，由 weak judge 判定。假设是：找到一个有说服力的真实答案，比找到一个有说服力的错误答案更容易。2024-2025 年近期工作（Khan 等、Michael 等）显示，debate 有时有帮助、有时没有，是否有效取决于 task structure。
- **Recursive Reward Modeling（Leike 等，2018）。** U 帮助 human 训练 U+1 的 reward model。Overseer 的有效能力会随 model 一起增长。
- **Task Decomposition（Christiano、Shlegeris、Amodei，2018）。** 把难任务递归拆成 human 能检查的 sub-tasks。它假设任务是可分解的。

每种机制都对任务结构或中间组件的 alignment 做了某种假设。

### 为什么 scalable oversight 和 W2SG 互补

Scalable oversight 提高 overseer 的有效信号质量。
W2SG 从 overseer 能提供的任何不完美信号中关闭 gap。

Lang 等人 — Debate Helps Weak-to-Strong Generalization（arXiv:2501.13124）把两者结合起来：debate protocol 提供更好的 weak labels，然后 strong model 在这些 labels 上训练。论文报告了 NLP tasks 上的 PGR gains。

### 组织层面的插曲

OpenAI 的 Superalignment team 在 Jan Leike 离职加入 Anthropic 后，于 2024 年 5 月解散。这个 agenda（scalable oversight、W2SG、automated alignment research）继续在 Anthropic 和学术实验室推进，包括 MATS（Lesson 28）、Redwood（Lesson 10）、Apollo（Lesson 8）、METR（Lesson 28）。组织结构变了，研究问题没有变。

### 它在 Phase 18 中的位置

Lessons 6-10 描述 threat 和 defensive paradigm，前提是 U 不可信。Lesson 11 是 offensive paradigm：让 overseer 强到足以验证 U 的 alignment。Lessons 12-16 接着转向 adversarial evaluation 的实用工具。

## Use It

`code/main.py` 会在一个 synthetic task 上模拟 W2SG fine-tune。Weak labeler 有 70% accuracy，并带有 structured errors；strong model 在 gold labels 上有 95% ceiling。你会在 weak labels 上 fine-tune strong model，测量 PGR，并与 strong-on-gold 和 weak-alone 对比。

## Ship It

本课会产出 `outputs/skill-w2sg-pgr.md`。给定一份 oversight setup description，它会识别 weak supervisor、strong model、supervision quality，并计算（或请求）PGR。它会标记该主张到底是 “weak can supervise strong”，还是 “weak + oversight mechanism can supervise strong”。

## 练习

1. 运行 `code/main.py`。报告 weak_accuracy = 0.60、0.70、0.80 时的 PGR。解释 PGR curve 的形状。

2. 修改 weak labeler，让它出现 structured error（例如总是在某个特定 input class 上出错）。PGR 会上升、下降，还是保持不变？解释原因。

3. 阅读 Burns 等人 2023 年论文 Section 4.3（NLP tasks）。复现 “confidence auxiliary loss” 的直觉：当 strong model 比 weak labels 更自信时，谁会赢？

4. 为 software-engineering task 设计一个结合 debate 和 task decomposition 的 scalable-oversight protocol。分别说出每个组件的一种 failure mode，并解释组合后如何处理，或者为什么仍然处理不了。

5. 说明什么证据会 falsify “weak-to-strong generalization is a viable path to superalignment” 这个主张。请具体说明你需要看到怎样的 empirical signature。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Scalable oversight | “让 overseer 更强” | 提高 overseer 评估更强模型能力的机制 |
| W2SG | “weak supervises strong” | 在 weak labels 上 fine-tune strong model，并测量恢复的能力 |
| PGR | “performance gap recovered” | (fine-tuned - weak) / (ceiling - weak)；1.0 = 完全关闭 gap，0 = 没有帮助 |
| Debate | “两个 U instances 争论” | 一种 scalable oversight 机制，由 weak judge 在两个 U defenders 之间选择 |
| RRM | “recursive reward modeling” | U 帮助训练 U+1 的 reward model；overseer capability 跟随 U |
| Task decomposition | “human 检查 sub-tasks” | 把难任务递归拆成 human 可以验证的 sub-tasks |
| Superalignment | “对齐 superhuman AI” | 关注如何对齐人类无法直接评估的模型的研究议程 |

## 延伸阅读

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/)：W2SG 论文。
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899)：debate mechanism。
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871)：recursive reward modeling。
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782)：2024 年 stronger debaters 的 debate 经验研究。
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124)：2025 年 debate + W2SG 组合研究。
