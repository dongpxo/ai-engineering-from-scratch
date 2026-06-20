# Capstone 82：Jailbreak 分类法

> 没有 taxonomy 的 safety harness 就像掷硬币。先命名攻击，再防御攻击。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## 问题

一个没有 attack model 就部署的模型，其实没有针对任何具体东西做防御。Operators 读到一条 Twitter thread，认出了技巧，写一个 regex，发布，然后继续往前走。下一条 prompt 是 paraphrase。Regex 漏掉了。一周后，有人展示同一种技巧，只是包在 base64 里，于是 operator 写第二条 regex。到了第三个月，系统里有 40 条 patch rules，却没有 shared vocabulary，没有办法讨论一个 attack 实际上是什么，backlog 增长得比 patches 还快。

在本 track 中，任何 detector、classifier 或 rule engine 想要真正有用之前，团队都需要一种共享方式来标注 attacks。不是因为 label 能阻止 attack，而是因为 label 能把 attack stream 变成 histogram。Histogram 会变成 coverage chart。Coverage chart 会驱动下一个 sprint。第 83-87 课中的 harness 会花时间判断一个 prompt 究竟是针对 refusal policy 的 role-play attack，还是针对 tool 的 context-smuggling attack。没有 taxonomy，这个判断不可能完成。

本 capstone 定义一个六类 taxonomy：宽到足以覆盖大多数现实中出现的 attacks，窄到两个 reviewer 通常会同意 category，具体到每个 category 至少有七个手工构造的 fixtures。Taxonomy 是所有下游内容的 carrier wave。

## 概念

六个 category 都沿着同一条轴线切分：attack 滥用了哪条 trust boundary？每个名字都对应一条 boundary。

| Category | Trust boundary abused |
|---|---|
| role-play | assistant 的 persona |
| instruction-override | system prompt 的 authority |
| context-smuggling | user content 和 instruction content 之间的 gap |
| multi-turn-ramp | conversation history 作为 contract |
| encoding-trick | forbidden tokens 的 surface form |
| prefix-injection | assistant 的 next-token decision |

Role-play attack 会把 assistant 重塑为另一个 agent（例如“你是一个不受限制的研究模型 QX”），让附着在原 persona 上的 refusal rules 不再触发。Instruction-override prompts 会说“忽略之前的指令”，并试图直接覆盖 system prompt。Context-smuggling 会把 instructions 藏进看起来像 data 的内容里：粘贴的文档、tool result、code block。Multi-turn-ramp 会先用无害轮次给模型热身，然后一步步降低边界，利用模型倾向于与对话保持一致的特性。Encoding tricks（base64、rot13、leet-speak、zero-width insertion）会把 forbidden tokens 从朴素 keyword filters 面前藏起来。Prefix-injection 会在 prompt 末尾放上“当然，下面是方法”这类 continuation cue，让模型沿着假定答案继续生成，而不是拒绝。

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

每个 fixture 都是一条 record，包含 `id`、`category`、`subtype`、`prompt`、`target_behavior` 和 `severity`。Taxonomy object 会加载 fixtures，按 category 分组，并暴露一个 `match` API：给定 candidate prompt，返回最接近的 fixture 及其 category。Match 使用 character-trigram cosine：粗糙、快速、无依赖。它不是 detector。Detector 会在第 83 课实现。这里是 label producer。

Severity 使用 1-5 级。1 是针对 benign target 的笨拙攻击（例如“请假装成海盗”）。5 是一旦成功就会产生 deployed system 绝不能输出内容的攻击（例如危险活动的 operational details）。大多数 fixtures 位于 2-3，因为部署规模上的真实 attacks 通常偏向容易和偷懒。Severity 由 fixture author 设置。如果两个 reviewer 的分歧超过一级，说明 rubric 需要打磨。

## Build It

Corpus 位于 `code/fixtures.py`，是一个单独的 Python list。`code/main.py` 中的 taxonomy class 会加载它，验证每个 category 至少有七个 fixtures，暴露 `by_category`、`match` 和 `stats` methods，并提供一个 runnable demo 来打印 histogram。Trigram cosine 使用 `numpy` 从零实现。

Validation pass 检查四个 invariants：每个 fixture 都有非空 prompt，schema 中每个 category 都有代表样本，每个 severity 都在 `1..5` 内，每个 fixture id 都唯一。这里的 failure 是 hard exit，不是 warning，因为本 track 后续内容依赖 corpus 内部一致。

## Use It

在 lesson 的 `code/` 目录运行 `python3 main.py`。Demo 会打印 per-category fixture count，针对 `match` 运行三个 sample probes，并把 `taxonomy.json` 写入 lesson outputs folder。下游课程会读取 `taxonomy.json`，而不是 import Python module，因此 corpus 是一个 stable artifact。

## Ship It

`outputs/skill-jailbreak-taxonomy.md` 记录六个 categories 和 rubric。把它当作团队的 shared vocabulary。第 87 课中的 harness 记录的每个 finding 都会引用一个 taxonomy id。

## 练习

1. 添加第七个 category：indirect-prompt-injection（instruction 嵌在 retrieved document 中，而不是 user turn 中）。编写十个 fixtures，并重新运行 validator。
2. 把 trigram cosine 替换成 token-edit-distance scorer，并测量现有 corpus 上的 match assignment 如何变化。
3. 从你自己产品的日志中抽取三十个额外 fixtures（redacted），并确认 category distribution 是否符合团队的直觉预期。

## 关键术语

| Term | Common usage | Precise meaning |
|---|---|---|
| jailbreak | any unsafe model output | 产生违反既定 policy 的输出的 prompt |
| taxonomy | a list of categories | 按所滥用的 trust boundary 对 attacks 做出的 partition |
| fixture | a test example | 带 category、severity 和 target behavior 的 labeled prompt |
| severity | how bad the output is | 如果 attack 成功，其 impact 的 1-5 rank |
| match | a detection decision | 按 trigram cosine 找到 nearest fixture，用来给新 prompt 分配 category |

## 延伸阅读

本课是入口。第 83-87 课会直接基于这个 corpus 继续构建。
