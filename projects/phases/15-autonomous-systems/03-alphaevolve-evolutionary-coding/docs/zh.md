# AlphaEvolve：进化式编码 Agent

> 把一个前沿编码模型、一套进化循环、一个可由机器检查的 evaluator 配在一起。让这个循环运行足够久。它会发现一种 4x4 复数矩阵乘法过程，只需要 48 次标量乘法。这是 Strassen 之后 56 年来第一次改进。它还会找到一条 Google 全域 Borg 调度启发式，在生产环境中回收约 0.7% 的集群计算资源。这个架构刻意保持朴素。真正的收益来自 evaluator 的严谨性。

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 minutes

## 问题

大型语言模型可以写代码。进化算法可以搜索代码。几十年来，人们分别尝试过这两类方法；它们也分别撞上了天花板。LLM 的天花板是虚构：模型会写出看起来可信、却并不实现其声称行为的代码。进化算法的天花板是搜索成本：对语法做随机 mutation 几乎不会产生可编译程序，更不用说产生更好的程序。

AlphaEvolve（Novikov et al., DeepMind, arXiv:2506.13131, 2025 年 6 月）把二者结合起来。LLM 向一个程序数据库提出定向 edits；自动 evaluator 给每个 variant 打分；高分 variants 成为后续 generation 的 parents。LLM 负责昂贵的步骤：写出看起来合理的代码。Evaluator 捕捉其中的虚构。这个循环可以运行数小时到数周。

论文报告的结果包括：48 次标量乘法的 4x4 复数矩阵乘法（Strassen 1969 年的界限是 49 次）、Google 生产环境中的 Borg 调度启发式、FlashAttention kernel 32.5% 的加速、Gemini 训练吞吐改进。

这个架构能工作，是因为 evaluator 可以由机器检查。Evaluator 不成立的地方，架构就不成立。这个不对称性就是本课的核心。

## 概念

### 循环

1. 从一个正确但非最优的 seed program `P_0` 开始。
2. 维护一个 variant programs 数据库，每个程序都由 evaluator 打分。
3. 从数据库中采样一个或多个 parents（MAP-elites-style 或 island-based）。
4. 提示 LLM（许多 candidates 用 Gemini Flash，难题用 Gemini Pro）基于 parent 生成一个修改后的 variant。
5. 编译、运行，并在 held-out evaluator 上评估这个 variant。
6. 按 score 和 feature vector 把它插入数据库。
7. 重复。

两个细节很重要。第一，给 LLM 的 prompt 不只是 parent program，通常还会包含数据库中的多个 top variants、evaluator signature，以及一小段 task description。模型的任务是提出一个可能改进 score 的定向修改。第二，数据库是结构化的（MAP-elites grid、island-based），所以循环会探索多样性，而不是只追逐当前 leader。

### 为什么 evaluator 不可协商

AlphaEvolve 的收益都来自 evaluator 快速、确定、难以钻空子的领域：

- **矩阵乘法算法**：unit test 执行矩阵乘法，并检查结果是否 bit-identically 相等。
- **Borg 调度启发式**：production-grade simulator 重放历史集群负载，并测量浪费的计算资源。
- **FlashAttention kernel**：correctness test 加真实硬件上的 wall-clock benchmark。
- **Gemini 训练吞吐**：测量每 step 消耗的 GPU-seconds。

在每个案例中，evaluator 都会抓住本来会主导结果的 LLM 错误类别：虚构的正确性声明、在硬件上消失的性能声明，以及边界条件失败。移除 evaluator，循环就会优化出漂亮代码，而不是正确代码。

### Reward hacking 是同一句话的另一面

进化会优化 evaluator 测量的任何东西。如果 evaluator 不完美，循环就会找到这个不完美之处。在未验证领域，循环会优化表面特征，而不是预期行为。DeepMind 在论文中明确指出：AlphaEvolve 的成功只会迁移到 evaluator 严谨性匹配搜索野心的领域。

2025-2026 年代码搜索循环中 reward hacking 的具体例子：

- 奖励“完成时间”的优化目标，最后奖励了提交空解法。
- 奖励 correctness-under-test 的 benchmark 分数，最后奖励了记忆 tests 和过拟合。
- 一个 “code quality” proxy 奖励删除注释和改写变量名，即使语义完全没变。

AlphaEvolve 的修复方式是：提供一个 LLM 从未见过的 held-out evaluator，并在评估时生成 inputs。即便如此，DeepMind 仍建议对任何拟部署方案进行强 review。

### 为什么 LLM + search 胜过单独使用任一方

LLM 可以产出可编译、语义上看起来合理的修改。对一个 2000 行 Python 文件做随机 mutation 的 GA，几乎总是生成语法错误。LLM 还会把搜索集中到合理邻域中（修改一个函数，而不是随机字节），大幅减少浪费的 evaluator calls。

反过来，evaluator 会抓住 LLM 的虚构。LLM 可能自信地声称某个函数“极限情况下是 O(n log n)”，但它实际上是 O(n^2)；wall-clock benchmark 会让问题有定论。

### AlphaEvolve 在前沿技术栈中的位置

| System | Generator | Evaluator | Domain | Example win |
|---|---|---|---|---|
| AlphaEvolve | Gemini | correctness + benchmark | algorithms, kernels, schedulers | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | correctness | combinatorial math | cap-set lower bounds |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiment | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent code | 20% → 50% SWE-bench |

这四者都是同一个配方的变体：generator 加 evaluator，再加一个循环。差异在于 evaluator 评什么，以及它有多严谨。

## Use It

`code/main.py` 在一个玩具 symbolic-regression 问题上实现了一个最小 AlphaEvolve-like loop。这里的 “LLM” 是一个 stdlib proxy，它会对计算目标函数的程序提出小型语法 mutation。这里的 “evaluator” 会在 held-out test points 上测量 mean squared error。

观察：

- best score 如何随 generations 改进。
- MAP-elites grid 如何保留多样化 solutions，让循环不至于收敛到局部最小值。
- 移除 held-out test（training-only evaluator）后，循环如何发生惊人的过拟合。

## Ship It

`outputs/skill-evaluator-rigor-audit.md` 是在新领域考虑 AlphaEvolve-style loop 的前置条件：你的 evaluator 是否真的能抓住你在意的 failure？

## 练习

1. 运行 `code/main.py`。记录 best score trajectory。禁用 held-out evaluator（flag `--no-holdout`）后重新运行。量化 overfitting。

2. 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites grid 的内容。为一个新问题（例如 compiler optimization passes）设计一个 feature-vector descriptor，用来保持搜索多样性。

3. 48 次乘法的 4x4 结果在 56 年后改进了 Strassen 的 49-mul bound。阅读论文 Appendix F，用三句话解释为什么这个问题的 evaluator 特别容易做对，以及为什么大多数领域并非如此。

4. 提出一个 AlphaEvolve 会失败的领域。明确指出 evaluator 在哪里断裂，以及为什么断裂。

5. 针对你熟悉的一个领域，写出你会使用的 evaluator signature。包含 (a) correctness conditions，(b) performance metric，(c) held-out input generation rule，(d) 至少一个 anti-reward-hacking check。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| AlphaEvolve | “DeepMind 的 evolutionary coding agent” | Gemini + program database + machine-checkable evaluator |
| MAP-elites | “Diversity-preserving archive” | 由 feature vectors 索引的 grid；每个 cell 保存具有该 descriptor 的最佳 variant |
| Island model | “Parallel evolution subpopulations” | 周期性迁移的独立 populations；防止 premature convergence |
| Machine-checkable evaluator | “Deterministic oracle” | LLM 无法伪造的 unit test、simulator 或 benchmark，也是这个 loop 的前提 |
| Reward hacking | “优化指标，而不是目标” | 循环找到一种最大化 score、却不完成预期任务的方法 |
| Seed program | “The starting point” | 循环从中演化的初始正确但非最优程序 |
| Held-out evaluator | “LLM 从未见过的 evaluation data” | 在 evaluation time 生成 inputs，防止 memorization |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 含结果的 vendor writeup。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的算法，包括 48-mul 4x4 matmul。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 把 evaluator-bound autonomy 作为关键研究方向来界定。
