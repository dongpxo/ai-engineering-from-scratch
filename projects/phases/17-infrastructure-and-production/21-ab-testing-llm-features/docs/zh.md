# A/B 测试 LLM 功能：GrowthBook、Statsig 与凭感觉上线的问题

> 传统 A/B testing 不是为非确定性的 LLM 设计的。关键区别是：evals 回答“模型能不能完成任务？”A/B tests 回答“用户在不在乎？”两者都需要；只靠 vibe checks 上线的时代结束了。2026 年值得测试的对象：prompt engineering（措辞）、model selection（GPT-4 vs GPT-3.5 vs OSS；accuracy vs cost vs latency）、generation parameters（temperature、top-p）。真实案例：chatbot reward-model variant 带来 +70% conversation length 和 +30% retention；Nextdoor AI subject-line experiments 在 reward-function refinement 后带来 +1% CTR；Khan Academy Khanmigo 围绕 latency-vs-math-accuracy 轴迭代。平台分野：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）—— sequential testing、CUPED、all-in-one。**GrowthBook** —— open-source、warehouse-native、Bayesian + Frequentist + Sequential engines、CUPED、SRM checks、Benjamini-Hochberg + Bonferroni corrections。选择取决于你是否偏好 warehouse-SQL，以及“被 OpenAI 收购”这件事对组织是否重要。

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## 学习目标

- 区分 evals（“模型能不能完成任务”）和 A/B tests（“用户在不在乎”）。
- 列举三个可测试的轴（prompt、model、parameters），并为每个轴选择指标。
- 解释 CUPED、sequential testing 和 Benjamini-Hochberg multiple-comparison corrections。
- 根据 warehouse-SQL 姿态和 corporate acquisition stance，在 Statsig 与 GrowthBook 之间做选择。

## 问题

你手工调好了一个 system prompt。它感觉更好。你上线了它。Conversion 的变化只是噪声。你开始怪 metric。或者你上线了一个新模型，conversion 没动——是模型退化了，还是变化太小以至于检测不到？你不知道，因为上线前没有做 A/B。

Evals 回答模型是否能在带标签的数据集上完成任务。它们不回答用户是否更喜欢输出。只有受控的线上实验能回答这个问题，而且实验必须有足够 power、控制非确定性，并修正多重比较。

## 概念

### Evals vs A/B tests

**Evals** —— offline、labeled set、judge（rubric、LLM-as-judge 或 human）。回答：“在这个固定分布上，输出是否正确 / 有帮助 / 安全？”

**A/B test** —— online、live users、randomized。回答：“新 variant 是否推动了真正重要的 user-level metric？”

两者都需要。Evals 在暴露给用户之前捕捉 regressions；A/B 在暴露之后确认 product impact。

### 要测试什么

1. **Prompt engineering** —— wording、system-prompt structure、examples。Metric：task success、user retention、cost/request。
2. **Model selection** —— GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。Metric：accuracy（task）+ cost/request + latency P99。Multi-objective。
3. **Generation parameters** —— temperature、top-p、max_tokens。Metric：task-specific（output diversity vs determinism）。

### CUPED：方差降低

Controlled-experiments Using Pre-Experiment Data。在比较 post-period 之前，先回归掉 pre-period variance。典型 variance reduction：30-70%。Effective sample size 免费上升。

实现层面：Statsig 和 GrowthBook 都支持。

### Sequential testing

经典 A/B 假设 fixed sample size。Sequential tests（“peek-and-decide”）在重复查看结果时控制 false-positive rate。Always-valid sequential procedures（mSPRT、Howard's confidence sequences）允许你在明显赢家出现时提前停止。

### Multiple-comparison corrections

跑 20 个 95% confidence 的 A/B tests，会由于 chance 产生一个 false positive。Bonferroni correction 会收紧每个 test 的 α；Benjamini-Hochberg 控制 false-discovery rate。GrowthBook 两者都实现了。

### SRM：sample ratio mismatch

Assignment hash 把用户随机分配到 variants。如果 50/50 split 实际得到 47/53，就说明有东西坏了——SRM check 会标记它。两个平台都支持。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $1.1B 收购。Hosted, SaaS。
- Sequential testing、CUPED、held-out populations。
- All-in-one：feature flags + experimentation + observability。
- Best fit：团队已经想要 bundled product，并且不在乎 OpenAI ownership。

**GrowthBook**：
- Open-source（MIT）；warehouse-native（直接读取 Snowflake/BigQuery/Redshift）。
- 多种 engines：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH corrections。
- Self-host 或 managed cloud。
- Best fit：warehouse-SQL shop，data team 控制 metric layer，并且想要 OSS。

### 非确定性会让 power 更复杂

同一个 prompt 会产生变化的输出。传统 power calculations 假设 IID observations。遇到 LLM non-determinism 时，effective sample size 会低于 nominal。把所需 sample size 乘以大约 1.3-1.5x，作为 safety margin。

### 真实案例结果

- Chatbot reward model variant：+70% conversation length，+30% retention。
- Nextdoor subject lines：reward-function refinement 后 +1% CTR。
- Khan Academy Khanmigo：围绕 latency-vs-math-accuracy trade 做迭代。

### 反模式：凭感觉上线

每个 senior engineer 都能说出一个因为“感觉更好”而上线、却没有做 A/B 的 feature。它们大多数都让产品指标退化了，只是团队几个月都没有发现。A/B 是 forcing function。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：open-source MIT；Bayesian + Frequentist + Sequential。
- CUPED variance reduction：30-70%。
- LLM non-determinism → +30-50% sample-size buffer。

## Use It

`code/main.py` 会模拟一个 sequential A/B test，包含 fixed 和 sequential boundaries。它展示 sequential 如何让你提前停止。

## Ship It

本课产出 `outputs/skill-ab-plan.md`。给定 feature change、workload、baseline，它会选择 platform、gates 和 sample size。

## 练习

1. 运行 `code/main.py`。当 baseline conversion 为 3%、预期 lift 为 5% 时，要达到 80% power，需要多少 sample size？
2. 为一个 healthcare-regulated on-prem customer 选择 Statsig 或 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 在 cost-per-resolved-ticket 上表现的 A/B。Primary metric、guardrail metric、secondary 分别是什么？
4. 你的 canary 通过了，但 A/B 显示 conversion -1.2%。你会 ship 吗？写出 escalation criteria。
5. 对 pre-period variance 为 post 的 60% 的场景应用 CUPED。计算 effective-sample-size boost。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Eval | “offline test” | 对 labeled set 做模型能力评估 |
| A/B test | “experiment” | 面向用户的 live randomized comparison |
| CUPED | “variance reduction” | 使用 pre-period regression 降低 variance |
| Sequential test | “peek-ok test” | 允许 early stop 的 always-valid procedure |
| Multiple comparison | “the family error” | 同时运行许多 tests 会抬高 false positives |
| Bonferroni | “tight correction” | 用 tests 数量除 α |
| Benjamini-Hochberg | “BH FDR” | False-discovery-rate control，比 Bonferroni 不那么保守 |
| SRM | “bad split” | Sample ratio mismatch；assignment bug |
| Statsig | “OpenAI owned” | Commercial all-in-one，2025 年被收购 |
| GrowthBook | “the OSS one” | MIT warehouse-native platform |
| mSPRT | “sequential probability ratio test” | 经典 sequential procedure |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
