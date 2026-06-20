# Benchmarks：SWE-bench、GAIA、AgentBench

> 2026 年有三个 benchmarks 锚定 agent evaluation。SWE-bench 测代码修补。GAIA 测 generalist tool use。AgentBench 测多环境推理。你需要知道它们的组成、污染故事，以及它们没有测量什么。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06（Tool Use）
**Time:** ~60 minutes

## 学习目标

- 命名 SWE-bench 的 test harness（FAIL_TO_PASS），并解释为什么它用单元测试把关。
- 解释为什么 SWE-bench Verified（OpenAI，500 tasks）存在，以及它剔除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 困难；三个难度等级。
- 命名 AgentBench 的八个 environments，以及它对 open-source LLMs 的主要 blocker。
- 总结 SWE-bench+ 的 contamination finding 及其影响。

## 问题

Leaderboards 会告诉你哪个模型在某个 benchmark 上赢了。它们不会告诉你：

- Benchmark 是否被污染（solutions 在训练数据里，test leakage）。
- Benchmark 是否测量你真正关心的东西（代码、浏览，还是 generalist）。
- Evaluator 是否稳健（AST matching、state checks、human review）。

引用数字之前，先理解这三个锚点 benchmarks 和它们的 failure modes。

## 概念

### SWE-bench（Jimenez 等，ICLR 2024 oral）

- 来自 12 个热门 Python repos 的 2,294 个真实 GitHub issues。
- Agent 获得：pre-fix commit 下的 codebase + 自然语言 issue description。
- Agent 产出：一个 patch。
- Evaluator：应用 patch，运行 repo 的 test suite。Patch 必须让 FAIL_TO_PASS tests（之前失败、现在通过）翻转，同时不能破坏 PASS_TO_PASS tests。

SWE-agent（Yang 等，2024）发布时达到 12.5%，重点在 agent-computer interfaces（file editor commands、模型能理解的 search syntax）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工筛选的 500-task subset。它移除了模糊 issues、不可靠 tests，以及 fix 不明确的 tasks。它是 “你的 agent 是否真的能交付真实 patches？” 的主要 benchmark。

### Contamination

- 超过 94% 的 SWE-bench issues 早于大多数 model cutoffs。
- **SWE-bench+** 发现，在成功 patches 中，32.67% 的 issue text 泄露了解法（模型在描述里看到了 fix），31.08% 因 test coverage 较弱而可疑。
- Verified 更干净，但不是完全无污染。

实际影响：一个在 SWE-bench 上得分 50% 的模型，在 SWE-bench+ 上可能只有 35%。如果你声称 SWE-bench performance，请始终同时报告两者。

### GAIA（Mialon 等，2023 年 11 月）

- 466 个问题；其中 300 个保留给 huggingface.co/gaia-benchmark 的 private leaderboard。
- 设计哲学：“对人类概念上简单（92%），但对 AI 很难（带 plugins 的 GPT-4：15%）。”
- 测试 reasoning、multi-modality、web、tool use。
- 三个难度等级；Level 3 需要跨 modalities 的长工具链。

GAIA 用来衡量 “generalist capability”。不要把它和代码专用 benchmarks 混在一起。

### AgentBench（Liu 等，ICLR 2024）

- 8 个 environments，覆盖 code（Bash、DB、KG）、games（Alfworld、LTP）、web（WebShop、Mind2Web）以及 open-ended generation。
- 多轮，每个 split 约 4k-13k turns。
- 主要发现：long-term reasoning、decision-making 和 instruction following，是 OSS LLMs 追上 commercial models 的 blockers。

### 这些 benchmarks 不测什么

- 真实世界 operational cost（tokens、wall-clock）。
- 对抗条件下的 safety behavior。
- 你的 domain performance（用自己的 evals，第 30 课）。
- Tail failures（benchmarks 看平均值；生产 operators 关心最差的 1%）。

### Benchmarking 常见错误

- **Single-number fixation.** SWE-bench 50% 不如 P50/P75/P95 cost + step distribution 有信息量。
- **Contaminated claims.** 报告 SWE-bench 却不提 Verified 或 SWE-bench+，会产生误导。
- **Benchmark-as-development-target.** 为 benchmark 优化会偏离生产实用性。

## Build It

`code/main.py` 实现了一个玩具版 SWE-bench-like harness：

- Synthetic bug-fix tasks（3 个 tasks）。
- 一个 scripted “agent”，会提出 patches。
- 一个 test runner，检查 FAIL_TO_PASS（bug 现在修好了）和 PASS_TO_PASS（没有破坏原本通过的测试）。
- 一个基于 question decomposition depth 的 GAIA-style difficulty classifier。

运行它：

```bash
python3 code/main.py
```

输出会展示每个 task 和每个 difficulty 的 resolution rate，并把 evaluator rules 具体化。

## Use It

- **SWE-bench Verified** 用于 code agents。始终报告 Verified scores。
- **GAIA** 用于 generalist agents。使用 private leaderboard split。
- **AgentBench** 用于 multi-environment comparison。
- **Custom evals**（第 30 课）用于你产品的真实形状。

## Ship It

`outputs/skill-benchmark-harness.md` 会为任意 codebase-task pair 构建一个 SWE-bench-style harness，并使用 FAIL_TO_PASS / PASS_TO_PASS gating。

## 练习

1. 把 toy harness 移植到一个真实 repo（选你自己的一个）。为已知 bugs 写 3 个 FAIL_TO_PASS tests。
2. 添加 step-count metric。在你的 3 个 tasks 上，每次 resolution 需要多少 agent steps？
3. 阅读 SWE-bench+ paper。实现一个 solution-leakage check（把 issue text 和 diff 做 pattern-match）。
4. 从 public split 下载一个 GAIA question。追踪一个 GPT-4-class agent 会怎么做。它需要哪些 tools？
5. 阅读 AgentBench 的 per-environment breakdown。哪个 environment 映射到你的 product surface？那里 “SOTA” 是什么样子？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SWE-bench | “Code agent benchmark” | 2,294 个 GitHub issues；patch 必须翻转 FAIL_TO_PASS tests |
| SWE-bench Verified | “Clean SWE-bench” | OpenAI 人工筛选的 500 个 tasks |
| FAIL_TO_PASS | “Fix gate” | 之前失败、patch 后必须通过的 tests |
| PASS_TO_PASS | “No-regression gate” | 原本通过、patch 后仍必须通过的 tests |
| GAIA | “Generalist benchmark” | 466 个对人类简单、对 AI 困难的 multi-tool questions |
| AgentBench | “Multi-env benchmark” | 8 个 environments；long-horizon multi-turn |
| Contamination | “Training-set leak” | Benchmark tasks 出现在模型训练中 |
| SWE-bench+ | “Contamination audit” | 在成功 SWE-bench patches 中发现 32.67% solution leakage |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770)：original benchmark。
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)：curated subset。
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983)：generalist benchmark。
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688)：multi-environment suite。
