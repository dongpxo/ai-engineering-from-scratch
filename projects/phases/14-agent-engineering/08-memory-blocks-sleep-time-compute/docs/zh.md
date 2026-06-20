# Memory Blocks 与 Sleep-Time Compute（Letta）

> MemGPT 在 2024 年演进为 Letta。2026 年的版本增加了两个关键思想：模型可以直接编辑的离散功能性 memory blocks，以及一个在主 agent 空闲时异步整合记忆的 sleep-time agent。这就是把记忆扩展到单次对话之外的方法。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## 学习目标

- 命名 Letta 使用的三层记忆（core、recall、archival），并说明每一层的作用。
- 解释 memory-block 模式：Human block、Persona block 和用户定义 blocks 都是一等 typed objects。
- 描述什么是 sleep-time compute，为什么它位于关键路径之外，以及为什么它可以运行比主 agent 更强的模型。
- 实现一个脚本化双 agent 循环：primary agent 负责响应，sleep-time agent 在 turns 之间整合 blocks。

## 问题

MemGPT（Lesson 07）解决了虚拟记忆的控制流。随后出现了三个生产问题：

1. **Latency.** 每个记忆操作都在关键路径上。如果 agent 必须在用户等待时修剪、总结或调和记忆，尾延迟会爆炸。
2. **Memory rot.** 写入会持续累积。相互矛盾的事实会留下来。检索会被陈旧内容淹没。
3. **Structure loss.** 扁平的 archival store 无法表达“Human block 永远在 prompt 中；Persona block 永远在 prompt 中；Task block 按 session 交换”。

Letta（letta.com）是 2026 年的重写版本。Memory blocks 让结构显式化；sleep-time compute 把整合工作移到关键路径之外。

## 概念

### 三层记忆

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | Always visible | Inside the main prompt | Agent tool call + sleep-time rewrites |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | Arbitrary facts | Vector + KV + graph | Agent tool call + sleep-time ingest |

Core 是 MemGPT 的核心记忆。Recall 是对话缓冲区以及被逐出的尾部内容。Archival 是外部存储。这个拆分清理了 MemGPT 两层设计中的职责重叠。

### Memory blocks

Block 是 core tier 中一个 typed、persistent、editable 的 section。最初的 MemGPT 论文定义了两个：

- **Human block**：关于用户的事实（姓名、角色、偏好、目标）。
- **Persona block**：agent 的自我概念（身份、语气、约束）。

Letta 将它泛化为任意用户定义 blocks：用于当前目标的 `Task` block、用于代码库事实的 `Project` block、用于硬约束的 `Safety` block。每个 block 都有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道什么时候应该编辑它）。

Blocks 可以通过工具面被编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)`：压缩接近上限的 block。

### Sleep-time compute

Letta 在 2025 年加入的能力：在后台运行第二个 agent，并让它脱离关键路径。Sleep-time agents 会处理对话 transcripts 和代码库上下文，把 `learned_context` 写入共享 blocks，并整合或废弃 archival records。

由此得到几个性质：

- **No latency cost.** Primary responses 不等待 memory ops。
- **Stronger model allowed.** Sleep-time agent 可以使用更贵、更慢的模型，因为它不受延迟约束。
- **Natural consolidation window.** 在用户没有等待时去重、总结、废弃矛盾事实。

这个形状很像人类的工作方式：你先完成任务，睡一觉，长期记忆在夜里沉淀下来。

### Letta V1 与 native reasoning

Letta V1（`letta_v1_agent`，2026）弃用了 `send_message`/heartbeat 和 inline `Thought:` tokens，转向 native reasoning。Responses API（OpenAI）以及带 extended thinking 的 Messages API（Anthropic）会在单独 channel 上输出 reasoning，并跨 turns 传递（生产中跨 providers 时会加密）。控制循环仍然是 ReAct。Thought trace 是结构性的，而不是 prompt-shaped。

### 这个模式会在哪里出错

- **Block bloat.** 无限 `block_append` 很快会打到上限。要在一次写入会越过 cap 之前接入 block summarizer。
- **Silent drift.** Sleep-time agent 重写了一个 block，而 primary agent 没有注意到。给 blocks 加版本，并在 trace 中展示 diff。
- **Poisoned consolidation.** Sleep-time agent 把攻击者可触达的内容处理进了 core。Lesson 27 同样适用于 sleep-time surface。

## Build It

`code/main.py` 实现了：

- `Block`：id、label、value、limit、description。
- `BlockStore`：CRUD 加 `near_limit(label)` helper。
- 两个脚本化 agents：`PrimaryAgent` 服务一个 turn，`SleepTimeAgent` 在 turns 之间做整合。
- 一条 trace，展示三轮对话中的 block writes，以及一次 sleep-time pass 如何总结 block 并废弃陈旧事实。

运行：

```bash
python3 code/main.py
```

transcript 会展示这种拆分：primary turns 很快，并产生原始写入；sleep pass 负责压缩和清理。

## Use It

- **Letta**（letta.com）：参考实现。可以 self-host，也可以使用 managed cloud。
- **Claude Agent SDK skills** 作为 block-shaped knowledge：skill 是一块命名、版本化、可检索的指令 block，agent 会按需加载。
- **Custom builds**：适合想控制存储后端的团队。使用 Letta API contract，这样以后可以迁移。

## Ship It

`outputs/skill-memory-blocks.md` 会为任意 runtime 生成一个 Letta-shaped block system，并带 sleep-time hooks、安全规则和 citation wiring。

## 练习

1. 添加一个 `block_summarize` 工具：当 `near_limit` 返回 true 时，用模型生成的 summary 替换 block value。哪个触发阈值能同时最小化 summarization calls 和 block overflow？
2. 在 archival 上实现 sleep-time dedup：两个 text 有超过 90% token overlap 的 records 合并为一个。只在 sleep pass 中做，永远不要放到关键路径上。
3. 给 blocks 加版本。每次写入都记录旧值和 diff。暴露 `block_history(label)`，让 operators 可以调试“为什么 agent 忘了 X”。
4. 把 sleep-time agents 当作不可信 writers。它们触碰 Persona 或 Safety block 时，提交前要求第二个 agent review。
5. 把示例移植到 Letta API（`letta_v1_agent`）。Block schema 有什么变化？Native reasoning 如何改变 trace shape？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory block | "Editable prompt section" | Core memory 中 typed、persistent、LLM-editable 的 segment |
| Human block | "User memory" | 关于用户的事实，会固定在 core 中 |
| Persona block | "Agent identity" | 自我概念、语气、约束，会固定在 core 中 |
| Sleep-time compute | "Async memory work" | 第二个 agent 在关键路径之外做 consolidation |
| Core / Recall / Archival | "Tiers" | 三层记忆拆分：always-visible / conversation / external |
| Block limit | "Cap" | 每个 block 的字符上限；迫使系统做 summarization |
| Native reasoning | "Thinking channel" | Provider-level reasoning output，而不是 prompt-level `Thought:` |
| Learned context | "Sleep output" | Sleep-time agent 写入共享 blocks 的事实 |

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：block pattern。
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute)：async consolidation。
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent)：native reasoning rewrite。
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：起源论文。
