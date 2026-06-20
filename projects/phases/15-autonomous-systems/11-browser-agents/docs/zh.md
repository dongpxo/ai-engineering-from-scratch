# 浏览器 Agent 与长程 Web 任务

> ChatGPT agent（2025 年 7 月）把 Operator 和 deep research 合并成一个 browser/terminal agent，并在 BrowseComp 上达到 68.9% 的 SOTA。OpenAI 在 2025 年 8 月 31 日关闭 Operator：产品层发生整合。Anthropic 收购 Vercept 后，把 Claude Sonnet 在 OSWorld 上的成绩从不到 15% 推到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修正了原始 WebArena 中 11.3 个百分点的 false-negative rate，并发布了 258-task Hard subset。这些数字是真实的。attack surface 也同样真实：OpenAI 的 preparedness 负责人公开表示，针对 browser agents 的 indirect prompt injection “不是一个可以被完全打补丁修好的 bug”。2025-2026 年已有记录的攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键 hijacks。

**Type:** Learn
**Languages:** Python（stdlib，indirect prompt-injection attack surface model）
**Prerequisites:** Phase 15 · 10（Permission modes），Phase 15 · 01（Long-horizon agents）
**Time:** ~45 minutes

## 问题

浏览器 agent 是一种长程 agent：它读取不可信内容，并执行有后果的动作。agent 访问的每个页面，都是用户没有亲自写下的输入。每个页面上的每个表单，都是潜在 command channel。2025-2026 年的攻击语料说明这不是假设：Tainted Memories 让攻击者通过精心构造的页面，把恶意指令绑定到 agent 的 memory；HashJack 把命令藏在 agent 会访问的 URL fragments 里；Perplexity Comet 的 hijack 只需要一次点击。

防御图景并不舒服。OpenAI 的 preparedness 负责人把那句安静的话大声说了出来：indirect prompt injection “不是一个可以被完全打补丁修好的 bug”。原因是攻击存在于 agent 的 reading-vs-acting boundary，而这个边界在架构上是模糊的：模型读到的每一个 token，原则上都可能被读成一条 instruction。

本课会命名这个 attack surface，命名 benchmark landscape（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小 indirect-prompt-injection 场景，这样你就能推理 Lessons 14 和 18 中的真实防御。

## 概念

### 2026 landscape：每个系统一段话

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。统一了 Operator（browsing）和 Deep Research（multi-hour research）。2025 年 8 月 31 日关闭 standalone Operator。在 BrowseComp 上达到 68.9% SOTA；在 OSWorld 和 WebArena-Verified 上也有强成绩。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept，聚焦 computer-use capabilities。把 Claude Sonnet 在 OSWorld 上从 <15% 推到 72.5%。Claude Computer Use 以 tool API 形式发货。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use integration 提供 computer-use controls；FSF v3（2026 年 4 月，Lesson 20）专门跟踪 ML R&D domain 中的 autonomy。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复一个有充分文档记录的问题：原始 WebArena 有约 11.3% false-negative rate（被标记为失败、但实际已经完成的任务）。Verified release 使用 human-curated success criteria 重新评分，并新增 258-task Hard subset（ICLR 2026 paper，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| Benchmark | What it measures | Horizon |
|---|---|---|
| BrowseComp | 在时间压力下从 open web 找到特定事实 | minutes |
| OSWorld | Agent 操作完整桌面（mouse、keyboard、shell） | tens of minutes |
| WebArena-Verified | simulated sites 中的 transactional web tasks | minutes |
| Hard subset | 带 multi-page state transitions 的 WebArena-Verified tasks | tens of minutes |

这些 benchmark 测的是不同轴。BrowseComp 高分说明 agent 能找事实；它不说明 agent 能订机票。OSWorld 分数更接近“它能不能在我的桌面上工作”。WebArena-Verified 更接近“它能不能完成一个 flow”。任何生产决策都需要选择与 task distribution 匹配的 benchmark。

### Attack surface，逐项命名

1. **Indirect prompt injection.** 不可信页面内容包含 instructions。agent 读取它们。agent 执行它们。公开例子包括：2024 Kai Greshake et al.、2025 Tainted Memories paper、2026 HashJack（Cato Networks）。
2. **URL fragment / query injection.** 被 crawl 的 URL 的 `#fragment` 或 query string 包含 commands。它从未被可见渲染；但仍在 agent 的 context 里。
3. **Memory-binding attacks.** 页面指示 agent 写入一条 persistent memory（Lesson 12 覆盖 durable state）。下一次 session 中，这条 memory 在没有可见触发器的情况下发射 payload。
4. **Authenticated sessions 上的 CSRF-shaped attacks.** Tainted Memories 类：agent 已经在某处登录；攻击者页面发出 state-changing requests，agent 使用用户 cookies 执行它们。
5. **One-click hijack.** 一个视觉上无害的按钮，携带 agent 会跟随的 payload。Comet 类。
6. **Agent host surface 中的 Content-Security-Policy holes.** rendering 和 tool layers 本身也可能是攻击向量；browser-in-a-browser-agent stack 很宽。

### 为什么“不能完全 patch”

这种攻击与 agent 的能力同构。agent 必须读取不可信内容才能完成工作。agent 读取的任何内容都可能包含 instructions。agent 遵循的任何 instructions 都可能与用户真实请求不对齐。防御（trust boundaries、classifiers、tool allowlists、consequential actions 上的 HITL）会提高攻击成本，并降低 blast radius。它们不能关闭整个攻击类别。

这和 Lob's theorem（Lesson 8）的推理模式一样：agent 不能证明下一个 token 是安全的；它只能建立一个系统，让不安全 tokens 更容易被检测。

### 实际会发货的防御姿态

- **Read / write boundary.** 读取永远不产生后果。写入（提交表单、发布内容、调用有 side effects 的 tool）如果由 trust boundary 外的内容触发，就需要 fresh human approval。
- **每个 task 的 tool allowlist.** agent 可以 browse；除非明确为该任务启用了 wire transfer tool，否则它不能发起转账。Lesson 13 覆盖 budgets。
- **Session isolation.** Browser agent sessions 只使用 scoped credentials。没有 production auth，没有个人邮箱。保留每个 HTTP request 的 logs 以供 audit。
- **Content sanitizer.** Fetched HTML 在拼接进 model context 前，会剥离 known-bad patterns。（减少简单攻击；挡不住复杂 payload。）
- **Consequential actions 上的 HITL.** Propose-then-commit pattern（Lesson 15）。
- **Memory 上的 canary tokens.** 如果某条 memory entry 被触发，用户能看见它（Lesson 14）。

## Use It

`code/main.py` 建模了一个 tiny browser-agent run，目标是三个 synthetic pages。一个页面是 benign，一个在 visible text 中包含 direct prompt-injection blob，一个包含 URL-fragment injection（不可见，但在 agent context 内）。脚本展示：(a) naïve agent 会做什么，(b) read/write boundary 能抓住什么，(c) sanitizer 能抓住什么，(d) 两者都抓不住什么。

## Ship It

`outputs/skill-browser-agent-trust-boundary.md` 用来限定一个拟议 browser-agent deployment 的范围：它接触哪些 trust zones，它被授权写入什么，以及第一次运行前必须具备哪些防御。

## 练习

1. 运行 `code/main.py`。找出 sanitizer 能抓住但 read/write boundary 抓不住的攻击，以及只有 read/write boundary 能抓住的攻击。

2. 扩展 sanitizer，让它检测一类 HashJack-style URL-fragment injection。在带有合法 fragments 的 benign URLs 上测量 false-positive rate。

3. 选择一个你熟悉的真实 browser-agent workflow（例如“订机票”）。列出每一个 read 和每一个 write。标记哪些 writes 需要 HITL，并说明原因。

4. 阅读 WebArena-Verified ICLR 2026 paper。找出原始 WebArena scoring 不可靠的一类任务，并解释 Verified subset 如何解决它。

5. 为 browser-agent 场景设计一个 memory canary。你会存什么、存在哪里、什么会触发 alarm？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Indirect prompt injection | “Bad page text” | agent 读取的页面中，不可信内容包含 agent 会执行的 instructions |
| Tainted Memories | “Memory attack” | agent 把攻击者提供的 instruction 写入 durable memory；下一次 session 触发 |
| HashJack | “URL fragment attack” | payload 藏在 URL fragment / query string 中，进入 agent context 但不可见渲染 |
| One-click hijack | “Bad button” | 可见 affordance 搭载 agent 会执行的后续 payload |
| BrowseComp | “Web search benchmark” | 在 open web 上寻找特定事实；minute-scale horizon |
| OSWorld | “Desktop benchmark” | 完整 OS control；multi-step GUI tasks |
| WebArena-Verified | “Fixed web-task benchmark” | ServiceNow 重新评分的 WebArena，带 Hard subset |
| Read/write boundary | “Side-effect gate” | 读取永远不产生后果；如果内容来自 trust 外部，写入需要 fresh approval |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和 deep research 的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator lineage，以及成为 ChatGPT agent 的 architecture。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始 benchmark。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 fixed-subset paper。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含 computer-use agents 的 attack-surface discussion。
