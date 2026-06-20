# 公平性标准：群体、个体、反事实

> 公平性文献大体由三类标准组织起来。Group fairness：demographic parity、equalized odds、conditional use accuracy equality，也就是在受保护群体之间让平均率相等。Individual fairness（Dwork 等，2012）：相似个体获得相似决策；对 decision map 施加 Lipschitz condition。Counterfactual fairness（Kusner 等，2017）：如果在反事实中改变 sensitive attributes 后决策不变，那么该决策对个体是公平的。2024 年理论结果（NeurIPS 2024）：CF-vs-accuracy 存在内在 trade-off；一种 model-agnostic method 可以把 optimal-but-unfair predictor 转换成 counterfactually fair predictor，并带有 bounded accuracy loss。Backtracking counterfactuals（arXiv:2401.13935，2024 年 1 月）：一种新范式，避免要求对 legally protected attributes 进行 interventions。哲学调和（ICLR Blogposts 2024）：在 causal graphs 下，满足某些 group fairness measures 会蕴含 counterfactual fairness。

**Type:** Learn
**Languages:** Python (stdlib, three-criteria comparison)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60 minutes

## 学习目标

- 说出三种 group-fairness criteria（demographic parity、equalized odds、conditional use accuracy equality）以及一个 impossibility result。
- 通过 Dwork 等人 2012 年的 Lipschitz formulation 描述 individual fairness。
- 描述 counterfactual fairness 及其对 causal graph 的依赖。
- 解释 backtracking counterfactuals，以及为什么它们能绕开 protected attribute intervention 问题。

## 问题

Lesson 20 讨论的是如何测量 bias。Lesson 21 讨论的是测量应该服务于哪一种 fairness standard。这三类标准给出结构上不同的规范：一个模型可以 group-fair 但 individual-unfair，也可以 counterfactually fair 但 group-unfair。选择标准是 policy decision；没有哪一种标准在所有场景下都是最优的。

## 概念

### Group fairness

- **Demographic parity。** 对所有群体，`P(Y=1 | A=a) = P(Y=1 | A=a')`。也就是 acceptance rates 相等。
- **Equalized odds。** `P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')`。也就是不同群体的 TPR 和 FPR 相等。
- **Conditional use accuracy equality。** `P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')`。也就是不同群体的 predictive value 相等。

Impossibility（Chouldechova、Kleinberg-Mullainathan-Raghavan 2017）：在 base rates 不相等时，这三种标准不能同时满足。

### Individual fairness

Dwork 等人 2012。对于任务特定的 similarity metric `d`，如果存在某个 Lipschitz constant `L`，使得 decision map `f` 满足 `|f(x) - f(x')| <= L * d(x, x')`，那么 `f` 在该 metric 下是 individually fair 的。相似个体获得相似决策。

这要求先定义 `d`。这不是统计问题，而是 policy question。

### Counterfactual fairness

Kusner 等人 2017。对于个体 `i`，如果在人口的 causal model 下，反事实改变 `i` 的 sensitive attributes 后决策仍然不变，那么这个决策对 `i` 是 counterfactually fair 的。

它需要一个 causal DAG。DAG 是 modeling choice。Counterfactual fairness 的正当性只和这个 DAG 一样强。

### CF-vs-accuracy trade-off

NeurIPS 2024 的理论结果：counterfactual fairness 和 predictive accuracy 之间存在内在 trade-off。一种 model-agnostic method 可以把 optimal-but-unfair predictor 转换成 counterfactually fair predictor，并让 accuracy cost 有界。Accuracy cost 取决于 optimal unfair predictor 中 sensitive-attribute coefficient 的大小。

### Backtracking counterfactuals

arXiv:2401.13935（2024 年 1 月）。传统 counterfactuals 要求对 sensitive attribute 进行 interventions，例如 “如果这个人是另一个性别，决策会不会变化”。在法律上，这很有问题：classification law 中不能对 protected attributes 做 intervention。

Backtracking counterfactuals 反转方向：不是干预属性，而是问这个人实际 features 的哪种组合会产生 counterfactual outcome。这绕开了法律上的反对意见。

### 哲学调和

ICLR Blogposts 2024。如果手里有 causal graph，那么满足某些 group-fairness measures 会蕴含 counterfactual fairness。这三类标准并不是彼此正交的；它们是同一个底层 causal structure 的不同侧面。

这并不会解决 impossibility theorems（base rates 不相等仍然会阻止同时满足 group fairness）。但它说明，“group” 和 “individual / counterfactual” 之间看似对立，部分是因为我们没有明确 causal model。

### 它在 Phase 18 中的位置

Lesson 20 是 bias measurement。Lesson 21 是 fairness definition。Lesson 22 是 privacy（differential privacy）。Lesson 23 是 watermarking。这些是和 allocation 相邻的课程，用来补充 Lessons 7-11 那条和 deception 相邻的主线。

## Use It

`code/main.py` 会构建一个带 sensitive attribute 和 unequal base rates 的 toy binary-classification dataset。在一个简单 classifier 上计算 demographic parity、equalized odds 和 conditional use accuracy equality。观察三种 metrics 如何互相不一致。再应用一个针对 demographic parity 的 re-weighting，观察它对另外两个指标的代价。

## Ship It

本课会产出 `outputs/skill-fairness-criterion.md`。给定一个 fairness claim 或 policy，它会识别被声称采用的是哪种 criterion，模型在声明的 unequal base rates 下能否满足其他 criteria，以及这个主张依赖什么 causal DAG。

## 练习

1. 运行 `code/main.py`。报告 default data 上的三个 group metrics。应用 targeted at demographic parity 的 re-weighting 后重新报告。

2. 用 non-sensitive features 上的 L2 distance 实现 Dwork 等人 2012 年的 individual-fairness metric。报告在 `L=1` 时有多少 pairs 违反 Lipschitz 条件。

3. 阅读 Kusner 等人 2017。为 resume scoring 构造一个简单的 two-feature causal DAG，并指出它蕴含的 counterfactual-fairness condition。

4. 2024 年 backtracking-counterfactuals 论文避免对 protected attributes 做 intervention。描述一个这对 legal compliance 很重要的场景。

5. ICLR 2024 的调和观点认为 group fairness 和 counterfactual fairness 是同一结构的不同侧面。从 `code/main.py` 的三种 criteria 中任选两种，说明什么 causal assumption 会让它们等价。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Demographic parity | “equal rates” | 各群体之间 `P(Y=1 | A=a)` 相等 |
| Equalized odds | “equal TPR/FPR” | 各群体之间 true-positive 和 false-positive rates 相等 |
| Conditional use accuracy | “equal PPV/NPV” | 各群体之间 predictive values 相等 |
| Individual fairness | “Lipschitz condition” | 相似个体获得相似决策 |
| Counterfactual fairness | “causal alteration invariance” | 在反事实属性改变下，决策保持不变 |
| Backtracking counterfactual | “explain via actuals” | 从 outcome 反向推理 counterfactual，而不是从 attribute 正向干预 |
| Impossibility theorem | “the three conflict” | Chouldechova / KMR 2017：base rates 不相等时，group criteria 互斥 |

## 延伸阅读

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913)：individual fairness。
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856)：counterfactual fairness。
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056)：impossibility。
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935)：protected-attribute interventions 的新范式。
