# 工具接口：为什么 Agent 需要结构化输入输出

> 语言模型生成 token。程序执行动作。两者之间的缺口，就是工具接口：一个让模型请求动作、让宿主执行动作的合约。2026 年的每一套技术栈——OpenAI、Anthropic、Gemini 的 function calling；MCP 的 `tools/call`；A2A 的 task parts——本质上都是同一个四步循环的不同编码方式。本课会给这个循环命名，并展示运行它所需的最小机制。

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## 学习目标

- 解释为什么一个只能生成文本的 LLM，不能独自对真实世界采取行动。
- 画出四步工具调用循环（describe → decide → execute → observe），并指出每一步由谁负责。
- 把一个工具描述写成三部分：名称、JSON Schema 输入，以及确定性的执行器函数。
- 区分纯工具和有副作用的工具，并说明这种区分为什么关系到安全性。

## 问题

LLM 输出的是下一个 token 的概率分布。这就是它的全部输出表面。如果你问聊天模型“Bengaluru 现在天气怎么样”，它可以写出一个看似合理的句子，但它不能真正拨进天气 API。这个句子可能碰巧正确，也可能已经过期三天。

工具接口的目的，就是补上这个缺口。宿主程序——你的 agent runtime、Claude Desktop、ChatGPT、Cursor，或者一个自定义脚本——向模型公布一组可调用工具。模型在判断需要采取动作时，会输出一个结构化 payload，说明要调用哪个工具以及使用哪些参数。宿主解析这个 payload，真正运行工具，再把结果喂回模型。这个循环会持续到模型判断不再需要更多调用为止。

这个合约的第一个版本，是 OpenAI 在 2023 年 6 月推出的 `functions` 参数。Anthropic 随后在 Claude 2.1 中加入了 `tool_use` blocks。Gemini 几个月后加入了 `functionDeclarations`。现在每个 provider 都暴露了同样的形状：输入一组带 JSON Schema 类型的工具列表，输出一个 JSON payload 形式的工具调用。Model Context Protocol（2024 年 11 月）把这个合约泛化，让同一个工具注册表可以服务每个模型。A2A（2026 年 4 月，v1.0）又在同一个 primitive 上叠加了 agent-to-agent delegation。

四步循环是所有这些接口底下的不变量。Phase 13 后面的所有内容，都是对它的展开。

## 概念

### 第一步：describe

宿主用三个字段声明每个工具。

- **Name。** 稳定、机器可读的标识符。用 `get_weather`，不要用 “weather thing”。
- **Description。** 一段自然语言简介。“Use when the user asks about current conditions for a specific city. Do not use for historical data.”
- **Input schema。** 一个 JSON Schema object（draft 2020-12），描述工具的参数。

模型会接收这份列表。现代 provider 会用 provider-specific template，把这些声明序列化进 system prompt，因此作为调用方，你只需要处理结构化形式。

### 第二步：decide

给定用户消息和可用工具，模型会在三种行为之间选择。

1. **直接用文本回答。** 不调用工具。
2. **调用一个或多个工具。** 输出结构化 call objects。在 `parallel_tool_calls: true` 下（OpenAI 和 Gemini 默认开启，Anthropic 需要 opt-in），模型可以在一轮中输出多个调用。
3. **拒绝。** Strict-mode structured outputs 可以输出一个带类型的 `refusal` block，而不是工具调用。

一个工具调用 payload 有三个稳定字段：call `id`、工具 `name`，以及 JSON `arguments` object。id 的作用是让宿主把后续结果和具体调用对应起来；当并行调用乱序返回时，这一点很重要。

### 第三步：execute

宿主收到调用后，会根据声明的 schema 校验参数，并运行执行器。无效参数意味着模型幻觉了一个字段，或使用了错误类型——这是弱模型上非常常见的失败模式。生产环境宿主遇到无效参数时通常有三种选择：快速失败并把错误暴露给模型；用受约束的 parser 修复 JSON；或带上 validation error 重新请求模型。

执行器本身就是普通代码。Python、TypeScript、shell command、database query 都可以。它会产出一个结果，通常是字符串，但也可以是任意 JSON value，或一个结构化 content block（MCP 中的 text、image、resource reference）。结果必须可以序列化。

### 第四步：observe

宿主把工具结果追加到对话里（作为带匹配 `id` 的 `tool` role message），然后再次调用模型。模型现在在上下文里看到了工具输出，可以生成最终答案，或请求更多调用。这个过程会持续到模型停止输出调用，或宿主触发迭代次数的安全上限。

### 信任边界

从安全角度看，工具有两种重要类型。

- **Pure。** 只读、确定性、没有副作用。`get_weather`、`search_docs`、`get_current_time`。可以安全地推测性调用。
- **Consequential。** 会改变状态、花钱、接触用户数据。`send_email`、`delete_file`、`execute_trade`。必须加门禁。

Meta 2026 年用于 agent security 的 “Rule of Two” 规定，单个 turn 最多只能同时包含以下三项中的两项：untrusted input、sensitive data、consequential action。工具接口就是执行这条规则的地方——通过拒绝调用、要求用户确认，或升级 scope。完整安全章节见 Phase 13 · 15，agent 级权限策略见 Phase 14 · 09。

### 循环在哪里运行

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

无论在哪里，都是同样的四步。列名会变，结构不变。

### 为什么不直接提示模型输出 JSON？

“让模型用 JSON 回复”是 function calling 出现前的常见模式。在 frontier models 上，它仍然会有约 5% 到 15% 的失败率；在小模型上失败更多。失败模式包括缺少大括号、尾随逗号、幻觉字段和错误类型。随后你就需要 JSON repair pass、retry，或 constrained decoder。

原生 function calling 更好，原因有三点。第一，provider 会用准确的调用形状对模型端到端训练，因此 strict mode 下有效 JSON 率可以升到 98% 到 99%。第二，call payload 位于自己的协议槽位里，而不是混在自由文本里，因此工具调用不会泄漏到用户可见回复中。第三，provider 会用 constrained decoding 强制 schema compliance（OpenAI 的 strict mode、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`）。输出保证可以通过校验。

Phase 13 · 02 会并排讲解三个 provider API。Phase 13 · 04 会深入 structured outputs。

### 断路器

当模型停止输出调用，或宿主达到最大 turn count 时，循环终止。生产环境宿主通常把上限设在 5 到 20 turns 之间。超过这个范围，你几乎可以确定自己进入了模型无法退出的循环。Claude Code 默认是 20；OpenAI Assistants 是 10；Cursor 的 agent mode 是 25。

另一个选择——无界循环——每六个月就会以“agent overnight 花掉 400 美元 API 调用费”的复盘形式出现一次。没有上限就不要发布。

Phase 14 · 12 会深入讲解错误恢复和 self-healing；Phase 17 覆盖生产环境 rate limits。

### Phase 13 接下来会怎么走

- Lessons 02 through 05 会打磨 provider-level tool-call surface。
- Lessons 06 through 14 会把这个循环泛化成 MCP。
- Lessons 15 through 18 会防御 hostile servers、adversarial users 和 unauthenticated remote auth surfaces。
- Lessons 19 through 22 会把这个模式扩展到 agent-to-agent collaboration、observability、routing 和 packaging。
- Lesson 23 会交付一个使用所有 primitive 的完整生态系统。

后续每一课都是这个四步循环的展开。把它当作不变量记住。

## Use It

`code/main.py` 会在没有 LLM 的情况下运行这个四步循环。一个假的 “decider” 函数通过对用户消息做 pattern matching 来模拟模型；执行器、schema validator 和 observe-step harness 都是真实的。运行它，观察完整 request/response choreography 以及可打印的中间状态；在后续课程中，你可以把这个 fake decider 替换成任意真实 provider。

值得关注的地方：

- 工具注册表为每个工具保存三个字段：name、description、schema，以及 executor reference。
- validator 是一个最小 JSON Schema 子集（types、required、enum、min/max），只用 stdlib 写成。Phase 13 · 04 会交付更完整的版本。
- 循环把迭代次数限制在五次。生产 agents 正需要这种 circuit breaker。

## Ship It

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一个工具定义草案（name + description + schema + executor outline），这个 skill 会审计它是否适合工具循环：名称是否机器稳定，description 是否是完整的使用简介，schema 是否正确使用 JSON Schema 2020-12，以及 pure-vs-consequential 分类是否明确。

## Exercises

1. 在 `code/main.py` 中加入第四个工具 `get_stock_price(ticker)`。把它的 description 写成：“Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries.” 运行 harness，并确认 fake decider 会把提到 tickers 的查询路由到新工具。

2. 故意破坏 schema validator。传入一个 `arguments` object 缺少 required field 的调用，并确认宿主会在执行前拒绝它。然后传入一个带额外未知字段的调用。做出判断：宿主应该拒绝还是忽略？用安全论证说明你的选择。

3. 把 harness 里的每个工具分类为 pure 或 consequential。给需要的 registry entries 加上 `consequential: true` flag，并修改循环，让它在选中 consequential tool 时打印一行 “would confirm with user”。这是每个生产环境宿主都需要的 confirmation gate 形状。

4. 在纸上画出四步循环，并把上面的 provider-column table 按你最喜欢的 client（Claude Desktop、Cursor、ChatGPT，或自定义 stack）补完整。和 Phase 13 · 06 中的 MCP-specific variant 交叉对照。

5. 从头到尾阅读 OpenAI 的 function-calling guide。找出一个位于 request 中、但不在这里介绍的四步循环里的字段。解释它增加了什么，以及为什么它是方便项而不是必需项。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | “模型可以调用的东西” | name + JSON-Schema-typed input + executor function 组成的三元组 |
| Function calling | “原生工具使用” | provider-level API support，用于输出结构化工具调用而不是散文 |
| Tool call | “模型请求采取行动” | 模型输出的 JSON payload，包含 `id`、`name`、`arguments` |
| Tool result | “工具返回了什么” | 执行器输出，包装在带匹配 id 的 `tool` role message 中 |
| Parallel tool calls | “一次调用很多个” | 一个模型 turn 中的多个 call objects，可通过 id 独立排序 |
| Strict mode | “保证 JSON” | constrained decoding，强制模型输出通过声明 schema 的校验 |
| Pure tool | “只读工具” | 没有副作用；可以安全重跑 |
| Consequential tool | “动作工具” | 改变外部状态；需要 gate、audit 或 user confirmation |
| Four-step loop | “工具调用周期” | describe → decide → execute → observe |
| Host | “Agent runtime” | 持有工具注册表、调用模型并运行执行器的程序 |

## Further Reading

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明和调用形状的 canonical reference
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` block format
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 和 parallel-call semantics
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — provider-agnostic 工具接口泛化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代工具 API 使用的 schema dialect
