# CrewAI：Role-Based Crews 与 Flows

> CrewAI 是 2026 年的 role-based multi-agent framework。四个 primitives：Agent、Task、Crew、Process。两个顶层形状：Crews（autonomous、role-based collaboration）和 Flows（event-driven、deterministic）。文档说得很直接：“for any production-ready application, start with a Flow.”

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## 学习目标

- 说出 CrewAI 的四个 primitives（Agent、Task、Crew、Process），以及每个 primitive 负责什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus process，并为每种 workload 选择合适的 process。
- 区分 Crews（autonomous role-based）和 Flows（event-driven deterministic），并解释文档中的 production recommendation。
- 使用 `@tool` decorator 和 `BaseTool` subclass 接入 tools；理解 structured outputs 与 free text 的取舍。
- 说出 CrewAI 的四种 memory types，以及各自什么时候值得使用。
- 实现一个 stdlib three-agent crew（researcher、writer、editor），产出一份 brief。
- 识别 CrewAI 的三种 failure modes：prompt-bloat、manager-LLM tax、brittle handoffs。

## 问题

采用 multi-agent frameworks 的团队常常撞上同一堵墙。“Autonomous collaboration” 在 demo 里听起来很棒。然后客户提交了 bug，你需要 deterministic replay。或者 finance 问每次 LLM-routed crew 的运行成本。或者 on-call 需要知道凌晨 3 点是哪一个 agent 卡住了。

Free-form LLM-routed crews 无法清晰回答这些问题。Pure DAGs 都能回答，但会丢掉 brainstorming agent 所需要的探索形状。

CrewAI 的拆分很诚实地承认了这个 tradeoff。Crews 用于 collaborative、role-based、exploratory work。Flows 用于 event-driven、code-owned、auditable production。同一个 framework，两种形状，按 surface 选择。

## 概念

### 四个 primitives

CrewAI 的 surface 很小。记住这个，剩下的就是配置。

- **Agent.** `role + goal + backstory + tools + (optional) llm`。Backstory 很关键。它会塑造 tone、judgment，以及 agent 何时停止。Tools 是 agent 可以调用的 functions（后面会讲）。
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。一个可复用的 work unit。`expected_output` 是 contract。`context` 列出上游 tasks，这些 tasks 的 outputs 会被传入。`output_pydantic` 强制 structured shape。
- **Crew.** Container。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose`、`manager_llm` settings。
- **Process.** Execution strategy。Sequential、Hierarchical、Consensus（planned）。决定一次 run 的形状。

Agents 不能直接看见彼此。Tasks 引用 agents。Crew 编排 tasks。Process 决定谁来选择下一个 task。这就是完整 mental model。

> **Validated against** CrewAI 0.86（2026-05）。较新的版本可能会重命名或合并 process types；在依赖某个具体形状之前，先查看 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential.** Tasks 按声明顺序运行。Task N 的 output 可以作为 `context` 传给 task N+1。成本最低，最可预测。适合顺序固定的 workload。
- **Hierarchical.** 一个 manager Agent（额外的 LLM call）在 specialists 之间路由。CrewAI 会根据你的 `manager_llm` config 或默认设置生成 manager。Manager 每轮选择下一个 task，也可以拒绝或重新路由。适合有四个或更多 specialists，且顺序确实取决于前序 output 的情况。
- **Consensus.** 计划中，当前 public API 尚未实现。文档保留这个名称，用于未来 voting-based process。今天不要依赖它。

Hierarchical 会在每个 specialist call 之外，每轮额外增加一个 LLM call（manager）。一个五步 run 的 token cost 可能变成三倍。只有在你真的需要 routing 时才付这个成本。

### Crews vs Flows

这是 2026 年文档最先强调的 framing。

- **Crew.** LLM-driven autonomy。Framework 在 runtime 选择形状。适合：research、brainstorming、first drafts，以及 path 本身就是答案一部分的场景。难以 replay。难以测试。很适合快速 prototype。
- **Flow.** 你拥有的 event-driven graph。`@start` 标记入口。`@listen(topic)` 标记某个 step 会在另一个 step 发出 topic 后触发。每个 step 都是 plain Python（内部可以调用 Crew）。适合：production。Observable、testable、deterministic。

文档在 2026 年给出的 production recommendation：从 Flow 开始。当 autonomy 值得付出成本时，把 Crews 作为 Flow steps 内部的 `Crew.kickoff()` 调用折进去。Flow 给你 audit trail，Crew 给你 exploration。要组合，而不是二选一。

### Tool integration

有三种方式可以给 Agent 一个 tool。选择满足需求的最简单方式。

1. **`@tool` decorator.** Pure functions 会变成 tools。Signature 是 schema；docstring 是 LLM 看到的 description。最适合一次性 helpers。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.** Class-based tool，带 explicit args schema、async support、retries。当 tool 有 state（client、cache）或需要 structured args 时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.** CrewAI 提供 first-party adapters：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次 import 就能接入。

Structured outputs 使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会根据 model 验证 LLM response，并进行 coercion 或 retry。把它和紧凑的 `expected_output` string 搭配使用。Free-text outputs 很适合 drafts；structured outputs 才是 downstream Flows 可以消费的东西。

### Memory hooks

CrewAI 开箱提供四种 memory types。它们可以组合：一个 Crew 可以同时启用四种 memory。

> **Validated against** CrewAI 0.86（2026-05）。近期 releases 会把所有东西路由到统一的 `Memory` system，这个 system 包装了下面四种 stores。下方 conceptual model 仍然成立，但更新版本中的 public class surface 可能会收敛成单个 `Memory` entry-point；请查看 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 获取当前 API。

- **Short-term.** 单次 run 内的 conversation buffer。Run 结束后清空。
- **Long-term.** 跨 runs 持久化。存储在 vector DB 中（默认 Chroma，可替换）。通过与当前 task 的 similarity 检索。
- **Entity.** Per-entity facts。例如 “Customer X is on the enterprise plan.” 按 entity keyed，而不是按 similarity。跨 runs 保留。
- **Contextual.** Assembly-time retrieval。在 Agent 需要 memory 的那一刻拉取 relevant memory，而不是预先加载。

在 Crew 上通过 `memory=True` 或 per-type config 启用。底层依赖你配置的 embeddings provider（默认 OpenAI，可替换为 local）。Memory 是 CrewAI 相比更薄 frameworks 真正有价值的地方之一；纯 LangGraph 需要你自己把这些都接起来。

### CrewAI 什么时候合适

- 三到六个 agents，带命名 roles 和 collaborative workflow。Drafting、reviewing、planning、brainstorming。
- LLM 对下一步的判断本身就是价值的一部分的 routing（Hierarchical）。
- 团队更愿意阅读 `role + goal + backstory`，而不是 graph definition 的场景。

### CrewAI 什么时候不合适

- 带 strict ordering 的 deterministic DAGs。使用 LangGraph（Lesson 13）。Graph shape 才是正确抽象；CrewAI 的 role framing 会变成阻力。
- Sub-second latency budgets。Hierarchical 会增加 round trips。即使是 Sequential，也会串行化包含 backstories 和 prior outputs 的 prompts。
- Single-agent loops。跳过 framework；一个 agent loop（Lesson 1）加 tool registry 更短。

Lesson 17（Agent Framework Tradeoffs）会用矩阵展开这一点。简短版本：CrewAI 位于 “collaborative role-based” 这个角落。

### Dependency shape

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star count：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（snapshot as of 2026-05）。AWS Bedrock integration 有文档；vendor benchmarks 报告在 QA workloads 上相比 LangGraph 有显著 speedup，但 methodology（dataset、hardware、evaluation metric）没有公开，所以 framework-vendor numbers 只能视作方向性参考。

### 这个模式会在哪里出错

- **Prompt-bloat from backstories.** 每个 agent 一段 2000 词 backstory，五个 agents 的 crew 在第一次 tool call 前就烧完 context budget。Backstories 控制在 200 词以内。跨 agents 复用短语；不要把 house style 重复五遍。
- **Manager-LLM token tax.** Hierarchical process 会在每个 specialist call 前增加一个 manager LLM call。五个 task 的 crew 变成六次 LLM calls，而 manager call 会携带完整 task list 和 prior outputs。除非 routing 依赖 output，否则切到 Sequential。
- **Brittle handoffs.** Task N 的 `expected_output` 是 “an outline”。Task N+1 把它当成 `context`，并试图解析三个 sections。LLM 产出了四个。下游 Agent 开始即兴发挥。用 Task N 上的 `output_pydantic` 修复，让 Task N+1 读取 typed object，而不是 free text。
- **Crew-as-prod.** Free-form Crew 没有 Flow wrapper 就被发到 production。Output variability 很高；replay 不可能；on-call 无法 diff bad run 和 good run。用 Flow 包起来。

## Build It

`code/main.py` 实现了两个形状的 stdlib 版本，以及一个 three-agent crew。

形状：

- `Agent`、`Task` dataclasses，对应 CrewAI surface。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行 tasks，并把 outputs 作为 `context` 串起来。
- `HierarchicalCrew.kickoff(topic)` 添加一个 manager Agent，每轮选择下一个 specialist，在 “done” 时停止。
- `Flow` 带 `@start` 和 `@listen(topic)` decorators、一个 tiny event loop，以及一条 trace。
- `tool(name)` decorator，模拟 CrewAI 的 `@tool` 形状。
- `Memory` 带 `short_term`、`long_term`、`entity` stores；mocked similarity 使用 numpy。
- Mock LLM responses 是根据 role 加 input prefix 硬编码的 strings。无网络。Deterministic。

具体 demo：researcher、writer、editor crew 会围绕 “agent engineering 2026” 产出一份 brief。Researcher 拉取（mocked）sources。Writer 起草。Editor 收紧。相同 crew 通过 Flow 再跑一次，展示 deterministic shape。

运行：

```bash
python3 code/main.py
```

Trace 覆盖：sequential crew 通过 `context` 串接 outputs，hierarchical crew 带 manager picks（researcher、writer、editor，然后 “done”），flow 使用显式 topics（`researched`、`drafted`、`edited`）运行同样三步，tool calls 通过 `@tool` 路由，以及 long-term memory 跨两次 kickoffs 存活。

Crew trace 是 fluid 的；manager 理论上可以重新排序。Flow trace 是固定的。这个选择就是本课重点。

## Use It

- **CrewAI Flow** 用于 production。即便这个 Flow 只有一个 step，里面调用 `Crew.kickoff()`。Flow 提供 audit boundary。
- **CrewAI Crew (Sequential)** 用于顺序清晰的 collaborative work，尤其是 first drafts 和 review loops。
- **CrewAI Crew (Hierarchical)** 用于 routing 取决于 output，且有四个或更多 specialists 的情况。
- **LangGraph**（Lesson 13）用于 explicit state machines、durable resume、strict ordering。
- **AutoGen v0.4**（Lesson 14）用于 actor-model concurrency 和 fault isolation。
- **OpenAI Agents SDK**（Lesson 16）用于 OpenAI-first products，带 handoffs 和 guardrails。
- **Claude Agent SDK**（Lesson 17）用于 Claude-first products，带 subagents 和 session store。

## Ship It

`outputs/skill-crew-or-flow.md` 会为一个 task 选择 Crew 或 Flow，并 scaffold 最小实现。它会 hard reject：没有 backstory 的 Crew、没有 explicit topics 的 Flow、少于三个 specialists 的 Hierarchical。

## Pitfalls

- **Backstory as flavor.** 它会塑造 outputs。每个 agent 测试三个 variants；variance 是真实存在的。选一个，freeze。
- **Skipping `expected_output`.** 如果每个 task 没有 contract，下游 tasks 会拿到 LLM 产出的任意形状。Crew 会运行；audit 会失败。
- **Memory always-on.** Long-term 每次 run 都写入。Vector DB 增长。Retrieval 变吵。只在 fact 会持久存在的 tasks 上 scope writes。
- **Manager prompt drift.** Hierarchical 的 manager prompt 是隐式的。如果 routing 变怪，在 verbose mode dump 出来读。
- **Tool side effects in Crews.** Crew 可能会比预期更多次调用某个 tool。POST、DELETE、payment 应该放在 Flow step，永远不要放进 Crew tool。

## 练习

1. 把 Sequential crew 转成 Flow。数一数 variability 降低了哪些触点。也记下 readability 在哪里下降了。
2. 给 crew 添加 entity memory：关于某个 customer 的 facts 会跨 kickoffs 保留。验证 retrieval 拉取的是正确 entity。
3. 实现一个 Hierarchical process：在 writer 的 output 至少有三段之前，manager 拒绝路由到 editor。Trace 这次 retry。
4. 为一个（mocked）web search 接入 `BaseTool` subclass。比较它和 `@tool` decorator 版本的 trace shape。
5. 给 editor task 添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让 writer task 先输出一次 malformed JSON；在 trace 中验证 CrewAI 的 retry behavior。
6. 阅读 CrewAI docs intro。把 toy 移植到真实 `crewai` API。Stdlib 版本跳过了哪些 guarantees？
7. 把 AgentOps 或 Langfuse（Lesson 24）接到一次真实 run。Stdlib 版本遗漏了哪些 traces？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent | “Persona” | Role + goal + backstory + tools |
| Task | “Unit of work” | Description + expected output + assignee + optional structured output |
| Crew | “Agent team” | Agents + Tasks + Process 的 container |
| Process | “Execution strategy” | Sequential / Hierarchical / Consensus（planned） |
| Flow | “Deterministic workflow” | Event-driven、code-owned、testable |
| Backstory | “Persona prompt” | Agent 的 tone 和 judgment shaper |
| `@tool` | “Function tool” | 把 function 变成 Agent 可调用 tool 的 decorator |
| `BaseTool` | “Class tool” | 带 args schema、retries、async support 的 class-based tool |
| Entity memory | “Per-entity facts” | Scoped to customer / account / issue 的 memory |
| Long-term memory | “Cross-run memory” | Kickoffs 之间持久存在的 vector-backed memory |
| Contextual memory | “Just-in-time retrieval” | Agent 需要时才拉取的 memory |
| Manager LLM | “Router agent” | Hierarchical process 中选择下一个 task 的额外 LLM |
| `expected_output` | “Task contract” | 告诉 Agent（和 audit）应该返回什么形状的 string |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：concepts and the recommended production path
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：event-driven shape、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、built-in toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：short-term、long-term、entity、contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：when multi-agent helps and when it does not
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：state-machine alternative
