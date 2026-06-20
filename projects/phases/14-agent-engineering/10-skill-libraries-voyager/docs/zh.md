# Skill Libraries 与终身学习（Voyager）

> Voyager（Wang 等人，TMLR 2024）把可执行代码当作 skill。Skills 有名称、可检索、可组合，并能通过环境反馈被改进。这是 Claude Agent SDK skills、skillkit，以及 2026 年 skill-library 模式的参考架构。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## 学习目标

- 命名 Voyager 的三个组件——automatic curriculum、skill library、iterative prompting——并说明每个组件的职责。
- 解释为什么 Voyager 把 action space 设计为代码，而不是 primitive commands。
- 用 stdlib 实现一个 skill library，支持注册、检索、组合，以及由失败驱动的 refinement。
- 把 Voyager 的模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态。

## 问题

每个 session 都从零重建能力的 agents，会犯三类错误：

1. **浪费 tokens。** 每个任务都会重新诱导同样的 reasoning。
2. **丢失进展。** Session A 中学到的修正，无法迁移到 session B。
3. **无法处理长程组合。** 复杂任务需要能力层级；one-shot prompts 表达不了这种层级。

Voyager 的答案是：把每个可复用能力当作一段有名称的代码，存入 library；它可以通过相似度被检索、与其他 skills 组合，并通过执行反馈被改进。

## 概念

### 三个组件

Voyager（arXiv:2305.16291）围绕三个模块组织 agent：

1. **Automatic curriculum.** 一个 curiosity-driven proposer 会根据 agent 当前 skill set 和环境状态选择下一个任务。探索是 bottom-up 的。
2. **Skill library.** 每个 skill 都是可执行代码。任务成功时会添加新 skill。Skills 通过 query-to-description similarity 被检索。
3. **Iterative prompting mechanism.** 失败时，agent 会接收执行错误、环境反馈和 self-verification 输出，然后改进 skill。

Minecraft 评测（Wang 等人，2024）：相比 baselines，Voyager 获得 3.3 倍更多 unique items，制作 stone tools 快 8.5 倍，制作 iron tools 快 6.4 倍，地图探索距离长 2.3 倍。这些数字是 Minecraft-specific 的，但模式可以迁移。

### Action space = code

大多数 agents 输出 primitive commands。Voyager 输出 JavaScript functions。一个 skill 是：

```javascript
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

它由 sub-skills 组合而成。它按 description 和 embedding 建索引。它被当作程序检索，而不是当作 prompt 检索。

这就是 2026 年 Claude Agent SDK skill 的形状：一段有名称、可检索的代码，以及 agent 按需加载的说明。

### Skill retrieval

新任务是 “make a diamond pickaxe”。Agent 会：

1. 对任务描述做 embedding。
2. 在 skill library 中查询 top-k 相似 skills。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用检索到的 primitives + 新逻辑组合出新 skill。

这也是 MCP resources（Phase 13）和 Agent SDK skills 实现的模式：在知识/代码表面上做 retrieval，并把范围限定到当前任务。

### Iterative refinement

Voyager 的反馈循环：

1. Agent 写出一个 skill。
2. Skill 在环境中运行。
3. 返回三类信号之一：`success`、`error`（带 stack trace）、`self-verification failure`。
4. Agent 把这个信号作为上下文，重写 skill。
5. 循环直到成功或达到 max rounds。

这是把 Self-Refine（Lesson 05）应用到带环境验证的代码生成上。CRITIC（Lesson 05）也是同一个模式，只是 verifier 换成了外部工具。

### Curriculum 与探索

Voyager 的 curriculum module 会根据 agent 已经拥有和还没有完成的内容，提出类似 “build a shelter near the lake” 的任务。Proposer 使用环境状态 + skill inventory，选择一个略高于当前能力的任务——这就是探索的甜蜜点。

在生产 agents 中，这会转化为一个 “what's missing” operator：给定当前 skill library 和某个 domain，我们还没有覆盖哪些 skills？团队通常会把它手动实现为 curriculum review。

### 这个模式容易在哪里出错

- **Skill library rot.** 同一个 skill 被用略有差异的 descriptions 添加 10 次。写入时加入 deduplication；检索时只返回一个。
- **Composed-skill drift.** Parent skill 依赖一个后来被 refinement 的 child。给 skills 做版本管理；pin 到 v1 的 parent 不应自动捡起 v3。
- **Retrieval quality.** 当 library 超过几百个 skills 时，只对 skill descriptions 做 vector retrieval 会退化。用 tag filters 和 hard constraints 补强，比如 “only skills with `category=tooling`”。

## Build It

`code/main.py` 实现了一个 stdlib skill library：

- `Skill`：name、description、code（字符串形式）、version、tags、dependencies。
- `SkillLibrary`：register、search（token overlap）、compose（deps 的 topological sort）和 refine（更新时 bump version）。
- 一个脚本化 agent：注册三个 primitive skills，组合第四个，遇到一次失败，然后 refine。

运行它：

```bash
python3 code/main.py
```

Trace 会展示 library writes、retrieval、composition、一次失败执行，以及 v2 refinement——也就是 Voyager 循环的端到端形状。

## Use It

- **Claude Agent SDK skills**（Anthropic）：2026 年参考形态；每个 skill 都有 description、code 和 instructions，并会在 agent session 中按需加载。
- **skillkit**（npm: skillkit）：面向 32+ AI coding agents 的跨 agent skill management。
- **Custom skill libraries**：面向具体 domain，例如 data agents 的 SQL skills、infra agents 的 Terraform skills。Voyager 模式可以缩小到这些场景。
- **OpenAI Agents SDK `tools`**：低配版本；每个 tool 都是一个轻量 skill。

## Ship It

`outputs/skill-skill-library.md` 会为任意目标 runtime 生成一个 Voyager-shaped skill library，内置 registration、retrieval、versioning 和 refinement。

## 练习

1. 给 `compose()` 添加 dependency-cycle detector。当 skill A 依赖 B，而 B 又依赖 A 时，会发生什么？应该 error 还是 warning？
2. 实现 per-skill version pinning。当 parent skill 组合 child `crafting@1` 时，对 `crafting@2` 的 refinement 不能静默升级 parent。
3. 用 sentence-transformers embeddings（或 stdlib BM25 实现）替换 token-overlap retrieval。在一个 50-skill toy library 上测量 retrieval@5。
4. 添加一个 “curriculum” agent：给定当前 library 和 domain description，提出 5 个缺失 skills。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill docs。把 toy library 移植到 SDK 的 skill schema。Discoverability 会发生什么变化？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Skill | “Reusable capability” | 有名称的代码块 + description，可通过相似度检索 |
| Skill library | “Agent memory of how-to” | 持久化 skill store，可搜索、可组合 |
| Curriculum | “Task proposer” | 由当前能力缺口驱动的 bottom-up goal generator |
| Composition | “Skill DAG” | Skills 调用 skills；执行前做拓扑排序 |
| Iterative refinement | “Self-correcting loop” | 环境反馈 + errors + self-verification 回流到下一版 |
| Action-space-as-code | “Programmatic actions” | 输出 functions，而不是 primitive commands，用于 temporally extended behavior |
| Dedup on write | “Skill collapse” | 近重复 descriptions 合并到一个 canonical skill |

## 延伸阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291)：原始 skill-library paper。
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：skills 作为 2026 年产品化形态。
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)：skills 和 subagents 的实践。
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)：Voyager 底层使用的 refinement loop。
