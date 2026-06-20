# 自主编码 Agent 版图（2026）

> SWE-bench Verified 在不到三年里从 4% 走到 80.9%。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline autonomous 上得分 59.8%——围绕模型的 scaffolding 现在和模型本身一样重要。OpenHands（原 OpenDevin）是最活跃的 MIT 许可平台，它的 CodeAct loop 会直接在 sandbox 中执行 Python actions，而不是 JSON tool calls。头部数字掩盖了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要改 1–2 行，而 SWE-bench Pro（10+ 行任务）上，同样的 frontier models 只有 23–59%。

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题

“哪个 coding agent 最好”是错误问题。正确问题是：在一个匹配我实际工作的 task distribution 上，使用我会在生产中运行的 scaffolding，我能得到怎样的 end-to-end reliability？

2022 到 2026 年间，这个领域学到了一件事：scaffolding——retrieval layer、planner、sandbox、edit-verify loop、feedback format——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分是 43.2%；同一个模型放进 Cline 的 autonomous scaffold 后得分是 59.8%。同样权重，绝对差值 16.6 个点。Base model 是组件；loop 才是产品。

伴随而来的问题是，benchmark saturation 会隐藏退步。SWE-bench Verified 已经接近饱和，而且 easy-task tail（500 个任务中有 161 个只需要 ≤2 行改动）会把顶部分数拉高。现实世界质量更适合用 SWE-bench Pro（10+ 行改动）这样的分布衡量；在这个分布上，同样的领先系统仍然只有 23–59%。

## 概念

### 用一段话理解 SWE-bench

SWE-bench（Jimenez 等）取真实 GitHub issues，带有 ground-truth patches，并要求 agent 产出一个能让 test suite 通过的 patch。SWE-bench Verified（OpenAI，2024）是一个由人类筛选的 500 任务子集，移除了 ambiguous 和 broken tasks。SWE-bench Pro 是更难的后继版本——任务需要 10+ 行改动，目前 frontier agents 的成绩在 23–59%。

### 2022 → 2026 曲线真正说明了什么

- **2022**：research models 在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin-style scaffolding 约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推进到 40–55% 区间。
- **2026**：Claude Sonnet 4.5 和 frontier competitors 在 SWE-bench Verified 上达到 70–80%+。Epoch AI 的 leaderboard 会实时追踪这一点。

这个斜率来自三个复合来源：更好的 base models、更好的 scaffolding（CodeAct、reflection、verifier loops），以及更好的 benchmarks（Verified 移除了噪声）。

### CodeAct vs JSON tool calls

OpenHands（All-Hands-AI，arXiv:2407.16741，原 OpenDevin）押注了一个具体架构：模型不是输出由 host 解码和执行的 JSON tool calls，而是输出 Python code，再由 Jupyter-style kernel 在 sandbox 中运行。Agent 可以在一个 action 里遍历文件、串联工具，并捕获自己的异常。

取舍如下：

- **JSON tool calls**：每个 action 是一轮；容易 audit；compositionality 有限；默认更安全，因为每次调用都经过显式 validator。
- **CodeAct**：一个 action 可以是一整个程序；可组合性强；需要 hardened sandbox（OpenHands 使用 Docker isolation）；failure modes 包括 sandbox runtime 允许的任何事情。

两种架构都已经进入生产。CodeAct 在 open platforms（OpenHands、smolagents）中占主导。JSON tool calls 仍然主导 managed services（Anthropic Managed Agents、OpenAI Assistants），因为 provider 控制 executor。

### 2026 版图中的 scaffolds

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct in Docker | 最活跃的 open platform；event-stream 可 replay |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 第一个 end-to-end SWE-bench scaffold |
| Aider | Apache-2 | edit-via-diff in local repo | 最小 scaffold，regression stability 强 |
| Cline | Apache-2 | VS Code agent with tool policy | Sonnet 4.5 上得分最高的 open scaffold |
| Devin (Cognition) | Proprietary | Managed VM + planner | 第一个“AI software engineer”产品类别 |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 会详细覆盖 agent loop |

### 为什么 scaffolding 占主导

一次 coding run 是 long-horizon trajectory（Lesson 1）。Reliability 会跨步骤复合。Scaffolding 能带来收益的地方有三个：

1. **Retrieval**：找到该读哪些文件是沉默的瓶颈。SWE-agent 的 ACI、OpenHands 的 file-index、Aider 的 repo-map 都在处理这个问题。
2. **Verifier loop**：运行测试、读取 stack traces、重新尝试，在 SWE-bench 上能带来 10+ 个点的差异。
3. **Failure containment**：出错后能 rollback 的 sandbox 会阻止伤害复合。有没有 verifier loop，会让同一个模型看起来像两个不同产品。

### Benchmark saturation 和真实分布

OpenHands 作者和 Epoch AI 都指出，SWE-bench Verified 有一个 easy tail：500 个任务中有 161 个只需要 1–2 行改动。高分部分由这个 tail 驱动。SWE-bench Pro 限制为 10+ 行改动；即使是 frontier systems，分数也回落到 23–59% 区间。你的生产分布几乎肯定更接近 Pro，而不是 Verified。

选择 agent 的含义是：从自己的 bug backlog 中跑一个类似 Pro 的子集。真正重要的分数，是代表你实际 ship 内容的任务上的分数。

## Use It

`code/main.py` 会在固定 mini-task distribution 上比较两个 toy agent scaffolds：

1. 一个 **JSON tool-call** scaffold，每轮只采取一个 action。
2. 一个 **CodeAct** scaffold，每个 action 可以发出一小段 Python snippet。

两者都使用 stub “model”（deterministic rules），所以比较隔离了 scaffold 和 model quality。输出会显示 CodeAct scaffold 用更少 turns 解决更多任务，但代价是每个 action 的 blast radius 更大。

## Ship It

`outputs/skill-scaffold-audit.md` 帮你在采用某个 coding-agent scaffold 之前审计它：retrieval quality、verifier presence、sandbox isolation，以及 benchmark-to-distribution fit。

## 练习

1. 运行 `code/main.py`。同一组任务上，每个 scaffold 分别用了多少 turns？各自的 per-action blast radius 是多少？

2. 阅读 OpenHands paper（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上优于 JSON tool calls。找出论文承认的一个 failure mode，并用一句话说明这个 failure mode 什么时候会在生产中占主导。

3. 从你的 bug backlog 中选择一个需要跨两个文件改动 10+ 行的任务。估计 frontier model 在 (a) JSON tool calls 和 (b) CodeAct 下的 end-to-end success probability。解释差距。

4. SWE-bench Verified 有 161 个单文件、1–2 行任务。构造一个排除它们的 score。Leaderboard 会怎样重排？

5. 阅读 “Introducing SWE-bench Verified”（OpenAI）。解释用于移除 ambiguous tasks 的具体方法，并说出一种这个 curation 会漏掉的类别。

## 关键术语

| Term | 常见说法 | 实际含义 |
|---|---|---|
| SWE-bench | “Coding benchmark” | 带有 ground-truth patches 和 test suites 的真实 GitHub issues |
| SWE-bench Verified | “Cleaned subset” | 500 个由人类筛选的任务，仍然存在 easier-tail |
| SWE-bench Pro | “Harder subset” | 10+ 行改动；frontier 成绩在 23–59% |
| CodeAct | “Code-as-action” | Agent 输出 Python；Jupyter-style kernel 在 sandbox 中执行 |
| JSON tool call | “Function calling” | 每个 action 都是执行前经过验证的 structured JSON payload |
| Scaffold | “Agent framework” | 围绕 base model 的 retrieval + planner + executor + verifier loop |
| ACI (Agent-Computer Interface) | “SWE-agent's format” | 为 LLM ergonomics 设计的 command set，而不是人类 shell |
| Verifier loop | “Test-and-retry” | 运行测试、读取输出、修改 patch；最大的非模型 reliability gain |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — the original benchmark and methodology.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — how the curated subset was built.
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct architecture and event-stream design.
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — live-tracked scores.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon coding-agent reliability framing.
