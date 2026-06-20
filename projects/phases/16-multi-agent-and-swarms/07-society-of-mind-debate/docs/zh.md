# Society of Mind 与 Multi-Agent Debate

> Minsky 1986 年的前提——智能是由 specialists 组成的 society——每十年都会被重新发现一次。2023 年，Du 等人把它变成了一个具体算法：多个 LLM instances 提出答案，阅读彼此的答案，critique，然后 update。经过 N 轮之后，它们会收敛到一个 consensus，在六个 reasoning 和 factuality tasks 上超过 zero-shot CoT 和 reflection。有两个发现很重要：**multiple agents** 和 **multiple rounds** 都独立贡献收益。Society 胜过 single-agent monologue；multi-round exchange 胜过 one-shot voting。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题

Self-consistency——从一个模型采样多次，然后取 majority answer——是你能加上的最便宜 reasoning improvement。它有效，但很快饱和。你可以把 samples 翻倍，却看不到新的显著提升。

Debate 打破了这种饱和。它不是从一个模型取 N 个彼此独立的 samples，而是让 N 个 agents 阅读彼此的 reasoning 并修正。Samples 之间的相关性会下降（它们不再是 i.i.d.），而 convergence point 往往会在 i.i.d. voting 自信但错误的地方变成正确答案。

## 概念

### Du et al. 2023 算法

来自 arXiv:2305.14325（ICML 2024）：

1. N 个 agents 各自为问题生成一个 initial answer。
2. 对 round r = 2..R：每个 agent 都会看到其他 agents 在 round r-1 的答案，并被要求“considering these, give your updated answer.”
3. R 轮之后，对 final answers 做 majority-vote。

论文在 MMLU、GSM8K、biographies、MATH 和 factuality benchmarks 上测试。Debate 持续超过 CoT 和 Self-Reflection。

### 两个独立旋钮

同一篇论文中的 ablations：

- **只增加 agent count**（1 round，对 N 个答案 majority vote）在大多数任务上超过 single-agent，但会 plateau。
- **只增加 round count**（1 个 agent 看自己的 prior reasoning）几乎没有帮助——这也是 reflection 的已知弱点。
- **两者同时增加** 才会产生大幅提升。多个 agents 之间的 multi-round exchange 驱动了收益。

### 为什么有效

两个机制：

1. **暴露于 disagreement。** 当一个 agent 看到另一个 agent 带有不同结论的 reasoning chain 时，它必须 justify 或 update。无论哪种方式，round r+1 的 context 都比 round r 更丰富。
2. **降低 correlated error。** 在 self-consistency 中，所有 samples 都来自同一个模型，所以 errors 相关——你会平均出一个自信但错误的答案。不同模型或不同 seeds 会 decorrelate。不同的 *debated views* 会进一步 decorrelate。

### Heterogeneous debate

A-HMAD 和相关后续工作为不同 agents 使用 *different base models*。Llama + Claude + GPT 互相 debate 可以减少 monoculture collapse（Lesson 26），因为一个 model family 的 correlated errors 不会被其他家族共享。

缺点是：一个弱模型参与 debate 时，可能把 consensus 拖向它的错误答案（见 “Should we be going MAD?”，arXiv:2311.17371）。

### NLSOM：129-agent extension

Zhuge 等人（“Mindstorms in Natural Language-Based Societies of Mind,” arXiv:2305.17066）把这个想法扩展到 129-member societies。结果是：specialization 和 self-organization 会随着规模涌现，并且系统在 visual question answering 等任务上超过 single-agent。

### Failure modes

- **Sycophancy cascade。** 所有 agents 都服从听起来最自信的 agent。Debate 坍缩成最响亮声音的独白。用 adversarial roles prompting（“one agent must argue the counter-position”）会有帮助。
- **Topic drift。** 多轮 debate 会偏离原始问题。缓解方法：每一轮都重新注入 question。
- **Compute blowup。** N agents × R rounds = N·R 次 LLM calls，而且每次调用的 context 都在增长。一个 5-agent、5-round debate 是 25 次调用，并且 context 越来越长。每个问题的成本可能超过单次 CoT call 的 10 倍。

## Build It

`code/main.py` 会在一个数学问题上运行 3-agent × 3-round debate，每个 agent 都从一个不同的（可能错误的）答案开始。Agents 是 scripted 的——每个 agent 都会根据 scripted confidence 对 neighbors' answers 加权平均来“update”。你能在 round-by-round log 里看到 convergence。

这个 demo 展示两个关键效果：

- 单轮 exchange 会让 agents 更接近正确答案。
- 超过 round 2 的额外 rounds 会出现 diminishing returns（与 Du et al. 的 plateau 相符）。

运行：

```bash
python3 code/main.py
```

## Use It

`outputs/skill-debate-configurator.md` 会为新任务配置 debate：agents 数量、rounds 数量、heterogeneity（same model vs mixed）、role assignment（symmetric vs one-adversarial）。它还会在运行前估算 token cost。

## Ship It

如果你要 ship debate：

- **把 rounds 限制在 3。** Du et al. 显示 3 轮捕获了大部分收益。更多通常是成本，不是质量。
- **把 agents 限制在 5。** 超过 5 后，context bloat 和成本会占主导。
- **默认 heterogeneous。** Pool 中至少有两个不同 base models。
- **Adversarial slot。** 一个 agent 被 prompt 成无论如何都要 disagree。它能打破 sycophancy。
- **记录每一轮。** 隐藏 intermediate rounds 的 debate systems 无法 debug 或 audit。

## 练习

1. 运行 `code/main.py`，然后把 round count 设为 5，观察 diminishing returns。额外 convergence 在第几轮停止？
2. 增加第四个带 adversarial role 的 agent：始终反对当前 majority。这会破坏还是改善 convergence？
3. Plot（print）每一轮的 agreement score（处于 majority answer 的 agents 占比）。它什么时候达到 1.0？这是否等价于“correct”？
4. 阅读 Du et al. 第 4 节 ablations。用这份代码复现 “agents-only” vs “rounds-only” vs “both” 结果。
5. 阅读 “Should we be going MAD?”（arXiv:2311.17371），列出 round-robin 之外的两个 debate variants，例如 judge-led、chain-of-debate、adversarial。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Society of Mind | “Minsky's idea” | 智能是 interacting specialists；1986 年 framing 现在通过 LLM debate 被 operationalized。 |
| Multi-agent debate | “Agents argue” | N 个 agents 提出答案、互相 critique、经过 R 轮 revise，然后 majority-vote。 |
| Consensus | “They agree” | 不是 epistemic truth，只是 fraction-on-majority-answer。可能自信地错。 |
| Rounds | “Exchange steps” | 一轮 = 每个 agent 读其他 agents 的答案并 update 一次。 |
| Heterogeneous debate | “Mix model families” | 使用不同 base models 来 decorrelate errors。 |
| Sycophancy cascade | “Everyone agrees with the loud one” | Debate failure：agents 无论正确性如何，都服从最自信的 agent。 |
| NLSOM | “129-agent society” | Natural-language society of mind；Zhuge 等人的 scaled version。 |
| Correlated error | “Same model, same bug” | Self-consistency 饱和的原因；不同 views 的 debate 会 decorrelate。 |

## 延伸阅读

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — the reference paper, ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmarks debate variants
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al.'s code, demos, and ablation details
