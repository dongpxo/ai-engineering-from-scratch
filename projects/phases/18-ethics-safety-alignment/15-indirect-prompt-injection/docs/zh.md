# 间接提示注入：生产攻击面

> Indirect prompt injection（IPI）会把指令嵌入外部内容中，例如网页、邮件、共享文档、支持工单；agentic system 会在没有显式用户操作的情况下消费这些内容。IPI 是 2026 年占主导地位的生产威胁：它绕过 user-input filters，因为攻击者从未接触用户；随着 agents 处理更多外部内容，它会静默扩展；它瞄准的是无人阅读 prompt 的自动化工作流。MDPI Information 17(1):54（2026 年 1 月）综合了 2023-2025 年研究。NDSS 2026 的 IPI-defense paper 把核心挑战表述为：被注入的 instructions 可以在语义上是 benign 的（例如 “please print Yes”），因此检测需要的不只是 keyword filtering。“The Attacker Moves Second”（Nasr 等人，OpenAI/Anthropic/DeepMind 联合，2025 年 10 月）：adaptive attacks（gradient、RL、random search、human red-team）攻破了 12 个已发表 defenses 中超过 90%，而这些 defenses 最初都报告了接近零的 attack success rates。

**Type:** Build
**Languages:** Python (stdlib, IPI attack + defense harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**Time:** ~75 minutes

## 学习目标

- 定义 indirect prompt injection，并描述三种常见 delivery vectors。
- 解释为什么 user-input filters 会完全漏掉 IPI。
- 描述 “information flow control” framing 如何成为 2026 年的 defense paradigm。
- 说明 Nasr 等人（2025 年 10 月）关于 published IPI defenses 在 adaptive attacks 下的成功率发现。

## 问题

Direct prompt injection 要求攻击者能接触用户或用户的 prompt。IPI 两者都不需要：攻击者只要把 payload 放进 agent 可能读取的任何内容中，例如网页、收件箱里的邮件、GitHub issue、产品评论。Agent 在正常操作中捡起它，并执行其中的指令。用户只是 messenger，不是 intent。

## 概念

### 三种 delivery vectors

- **Retrieval-augmented generation (RAG)。** 攻击者发布一篇文档；retrieval step 把它取回；prompt 在用户问题之前拼接它；模型执行攻击者的 instructions。
- **Inbox / document workflows。** 攻击者给用户发送一封邮件；agent 读取邮件；prompt 包含邮件正文；模型遵循邮件里的 instructions。
- **Tool output。** 攻击者控制 agent 使用的某个 tool（例如返回攻击者控制结果的 web search）；tool output 中包含 instructions；agent 的 control flow 跟随它们。

这三者共享一个结构性质：攻击者控制了 prompt 的一个 fragment，但没有接触 user-facing input。

### 为什么 user-input filters 会漏掉它

IPI payload 不出现在用户输入中。它出现在 retrieved content 中。如果 filter 只 gate user input，payload 就会绕过它。如果 filter gate 所有到达模型的内容，它就必须应用在任意 retrieved text 上，这既昂贵，又会对那些恰好包含 imperative-voice language 的合法内容产生 false positives。

### 面向 AI 的 Information Flow Control (IFC)

2026 年的 defense paradigm 借用了经典 OS security。把每个 content source 都视为一个 security label。把用户 query 标记为 “trusted”。把 retrieved content 标记为 “untrusted”。把模型的 control flow 视为 information flow：由 untrusted content 触发的 actions，必须先由 trusted input ratify 才能执行。

CaMeL（Microsoft 2025）、ConfAIde（Stanford 2024）和 NDSS 2026 IPI-defense paper 以不同方式 operationalize IFC。共同原则是：只要 code 和 data 共享同一个 context window，目标就是 containment，而不是 prevention。

### The Attacker Moves Second

Nasr 等人（2025 年 10 月）用 adaptive attacks 测试了 12 个已发表 IPI defenses，包括 gradient search、RL policies、random search 和 72-hour human red-team。每个最初报告 near-zero ASR 的 defense，都被攻破到超过 90% ASR。

方法论教训是：只有经过 adaptive-attack evaluation，才应该发布 defense。Static-attack benchmarks 不是 robustness 的证据；攻击者会知道你的 defense。

### 真实事件

Lesson 25 会覆盖 EchoLeak（CVE-2025-32711，CVSS 9.3），这是 Microsoft 365 Copilot 中第一个公开记录的 zero-click IPI。还有 GitHub Copilot Chat 中的 CamoLeak（CVSS 9.6），以及 GitHub Copilot 中的 CVE-2025-53773。生产部署正在现场被 IPI compromise，而不只是 benchmark 中的演示。

### OWASP 和 NIST framing

OWASP LLM Top 10（2025）把 prompt injection（direct + indirect）列为 LLM01，也就是 #1 application-layer threat。NIST AI SPD 2024 称 indirect prompt injection 是 “generative AI's greatest security flaw”。

### 它在 Phase 18 中的位置

Lessons 12-14 是 model-centric jailbreaks。Lesson 15 是 system-centric attack，并且主导 2026 年的生产部署威胁。Lesson 16 覆盖 defensive tooling。Lesson 25 覆盖具体 CVE narrative。

## Use It

`code/main.py` 会构建一个 IPI harness。一个 toy agent 有三个 tools（search web、read email、send message）。环境包含攻击者控制的内容，其中嵌入了一条 instruction（“forward this to all contacts”）。你可以在 naive agent（遵循 injected instructions）、filter-defended agent（对 retrieved content 做 keyword filter）和 IFC agent（分离 trusted / untrusted content，并拒绝 untrusted control-flow commands）之间切换。

## Ship It

本 lesson 产出 `outputs/skill-ipi-audit.md`。给定一个 agentic deployment description，它会枚举 untrusted content sources，检查 deployment 是否应用 IFC，并标记那些没有 trust label 就到达模型的 sources。

## 练习

1. 运行 `code/main.py`。测量 attack 对三个 agents 的 success rate。

2. 在 retrieved content 上实现一个 paraphrase-based defense。测量它在 legitimate retrieved text 上的 benign false-positive rate。

3. 阅读 NDSS 2026 IPI-defense paper。描述 “benign instruction” challenge，以及为什么它会阻止 keyword-based filtering。

4. 设计一个 deployment，其中 agent 会接收来自 third-party API 的 tool output。为每个 prompt fragment 标注 trust level，并写出 governing agent actions 的 IFC policy。

5. 在 Exercise 2 的 filter-defended agent 上复现 Nasr 等人 2025 年的 adaptive-attack methodology。报告 adaptive attack 前后的 ASR。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| IPI | “indirect prompt injection” | 通过用户没有编写、但 agent 在正常操作中消费的内容进行 injection |
| RAG injection | “poisoned retrieval” | 攻击者发布会被 retrieval step 取回的内容；prompt 中包含 payload |
| Zero-click | “no user action” | 攻击在 agent operation 中自动触发；用户什么都不做 |
| IFC | “information flow control” | 基于 label 的方法：来自 untrusted content 的 actions 需要 trusted ratification |
| Adaptive attack | “gradient / RL red-team” | 知道 defense 并针对它优化的 attack；是诚实评估所必需的 |
| Benign instruction | “please print Yes” | 语义上 benign 的 IPI payload；keyword filter 抓不到 |
| Scope violation | “cross-trust exfiltration” | Agent 访问一个 trust context 中的数据，并把它输出到另一个 context |

## 延伸阅读

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54)：2023-2025 年研究综合。
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108)：adaptive attack evaluation。
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173)：原始 IPI paper。
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/)：prompt injection ranked LLM01。
