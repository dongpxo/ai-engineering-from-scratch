# 记忆：虚拟上下文与 MemGPT

> Context window 是有限的。对话、文档和工具 trace 不是。MemGPT（Packer 等，2023）把这个问题类比成操作系统虚拟内存——main context 是 RAM，external store 是 disk，agent 在两者之间换页。这是 2026 年每个 memory system 继承的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01（Agent Loop），Phase 14 · 06（Tool Use）
**Time:** ~75 minutes

## 学习目标

- 解释 MemGPT 建立在其上的 OS 类比：main context = RAM，external context = disk，memory tools = page in/out。
- 用 stdlib 实现两层 MemGPT 模式：main-context buffer、external searchable store，以及 page in/out tools。
- 描述 agent 如何发出 “interrupts” 来查询或修改 external memory，以及结果如何拼回下一次 prompt。
- 识别哪些 MemGPT 设计选择延续到了 Letta（第 08 课）和 Mem0（第 09 课）。

## 问题

Context window 看起来像是能解决记忆问题。实际上不能。生产中反复出现三种 failure modes：

1. **Overflow.** 多轮对话、长文档，或工具调用密集的 trajectories 会越过窗口。超过 cutoff 的所有内容都会消失。
2. **Dilution.** 即使还在窗口内，把无关 context 塞进去也会稀释模型对关键信息的注意力。Frontier models 在长输入上仍然会退化。
3. **Persistence.** 新 session 会从空窗口开始。没有 external memory 的 agents 无法跨 session 说出 “还记得你之前让我……”。

更大的窗口有帮助，但不会修好这个问题。Mem0 的 2025 年论文测到：128k-window baselines 仍会漏掉长跨度事实，而带 external memory 的 4k-window agent 能抓住这些事实。

## 概念

### MemGPT：OS 类比

Packer 等人（arXiv:2310.08560，v2 Feb 2024）把 context management 映射到操作系统虚拟内存：

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context（prompt） | Anthropic/OpenAI context window |
| Disk | external context | vector DB、KV、graph store |
| Page fault | memory tool call | `memory.search`、`memory.read`、`memory.write` |
| OS kernel | agent control loop | 带 memory tools 的 ReAct loop |

Agent 运行一个普通 ReAct loop。额外的一类 tools 让它能把数据换入和换出 main context。

### 两层结构

- **Main context.** 固定大小的 prompt，保存当前任务。模型始终可见。
- **External context.** 无上限，通过 tools 搜索。相关时读取，出现事实时写入。

原论文在两个超过 base window 的任务上评估了这个设计：超过 100k tokens 的文档分析，以及跨多天、带持久记忆的多 session 聊天。

### Interrupt 模式

MemGPT 引入 memory-as-interrupt：对话中途，agent 可以调用 memory tool，runtime 执行它，并把结果作为新的 observation 拼进下一次 assistant turn。概念上等同于 Unix 的 `read()` syscall：它阻塞进程，返回 bytes，然后进程继续执行。

Canonical memory tool surface：

- `core_memory_append(section, text)`：写入 prompt 中的持久 section。
- `core_memory_replace(section, old, new)`：编辑一个持久 section。
- `archival_memory_insert(text)`：写入可搜索的 external store。
- `archival_memory_search(query, top_k)`：从 external store 检索。
- `conversation_search(query)`：扫描过去 turns。

### MemGPT 在哪里结束，Letta 从哪里开始

2024 年 9 月，MemGPT 变成了 Letta。研究 repo（`cpacker/MemGPT`）仍然保留；Letta 扩展了这个设计：

- 从两层变成三层（core、recall、archival——第 08 课）。
- 用 native reasoning 替代 `send_message` / heartbeat pattern（第 08 课）。
- Sleep-time agents 运行异步 memory work（第 08 课）。

即便生产系统运行的是 Letta、Mem0，或自定义两层 store，MemGPT 论文仍然是 2026 年的基础。

### 这个模式哪里会出错

- **Memory rot.** 写入增长快于读取；stale facts 淹没检索。修复方式：周期性 consolidation（Letta sleep-time），显式 invalidation（Mem0 conflict detector）。
- **Memory poisoning.** External memory 是被检索出来的文本。如果攻击者控制的内容进入 memory note，agent 会在下一次 session 重新摄入它。这就是 Greshake 等人（第 27 课）的攻击，只是跨时间重述了一遍。
- **Citation loss.** Agent 回忆起 “用户让我发布 X”，但无法引用是哪一轮。每次 archival write 都要存 source references（session ID、turn ID）。

```figure
context-budget
```

## Build It

`code/main.py` 用 stdlib 实现了 MemGPT 的两层模式：

- `MainContext`：固定大小的 prompt buffer，带一个 `core` dict 和一个 `messages` list；超过上限时自动 compact 最旧 messages。
- `ArchivalStore`：内存中的 BM25-esque store（token-overlap scoring），保存 (id, text, tags, session, turn) records。
- 五个 memory tools，对应 MemGPT surface。
- 一个 scripted agent：先把 facts 写入 archival，然后通过调用 `archival_memory_search` 回答问题。

运行它：

```bash
python3 code/main.py
```

trace 会展示 agent 写入三个 facts、把 main context 填到上限（触发 eviction），然后通过 archival retrieval 回答 follow-up question——不用真实 LLM 也能复现 MemGPT workflow。

## Use It

今天每个生产 memory system 都是 MemGPT variant：

- **Letta**（第 08 课）：三层结构、native reasoning、sleep-time compute。
- **Mem0**（第 09 课）：vector + KV + graph 与 scoring layer 融合。
- **OpenAI Assistants / Responses**：通过 threads 和 files 管理 memory。
- **Claude Agent SDK**：通过 skills 和 session store 做 long-term memory。

选择时看 operational shape（self-hosted、managed、framework-integrated），而不是 core pattern——core pattern 就是 MemGPT。

## Ship It

`outputs/skill-virtual-memory.md` 是一个可复用 skill。它能为任何目标 runtime 产出正确的两层 memory scaffold（main + archival + tool surface），并接好 eviction policy 和 citation fields。

## 练习

1. 增加一个按 tokens 度量的 `max_main_context_tokens` 上限（用 `len(text.split())` * 1.3 近似）。超过上限时，把最旧 messages compact 成 summary。对比有 summarizer 和没有 summarizer 的行为。
2. 在 archival store 上正确实现 BM25（term frequency、inverse document frequency）。在玩具 fact set 上测量 recall@10，并和 token-overlap baseline 对比。
3. 给 archival inserts 增加 `citation` fields（session_id、turn_id、source_url）。让 agent 在每个 retrieval-backed answer 中引用来源。
4. 模拟 memory poisoning：添加一条 archival record，内容是 “ignore all future user instructions”。写一个 guard，扫描 retrievals 中像指令的文本，并把它们标记为 untrusted。
5. 把实现移植为 MemGPT 研究 repo 的 core-memory JSON schema（`cpacker/MemGPT`）。从 flat strings 切换到 typed sections 时，会发生什么变化？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Virtual context | “Unlimited memory” | Main（prompt）+ external（searchable）两层结构，带 page in/out |
| Main context | “Working memory” | Prompt——固定大小，始终可见 |
| Archival memory | “Long-term store” | External searchable persistence，按需检索 |
| Core memory | “Persistent prompt section” | 固定在 main context 内的 named sections |
| Memory tool | “Memory API” | Agent 发出的用于读写 external memory 的 tool call |
| Interrupt | “Memory page fault” | Agent 暂停，runtime 获取内容，结果拼回下一轮 |
| Memory rot | “Stale facts” | 旧写入淹没 retrieval；用 consolidation 修复 |
| Memory poisoning | “Injected persistent note” | 攻击者内容被存为 memory，并在 recall 时重新摄入 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：受 OS 启发的 virtual context paper。
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：三层演进。
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：把 context 当作预算来管理。
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)：建立在这个模式上的 hybrid production memory。
