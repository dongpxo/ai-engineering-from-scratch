# Tool Use 与 Function Calling

> Toolformer（Schick et al., 2023）开启了 self-supervised tool annotation。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）设定了 2026 年的标准：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。Single-turn 已经基本解决。Memory、dynamic decision-making 和 long-horizon tool chains 还没有。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## 学习目标

- 解释 Toolformer 的 self-supervised training signal：只有当 tool execution 能降低 next-token loss 时，才保留 tool annotation。
- 说出 BFCL V4 的五个 evaluation categories，并解释每一类衡量什么。
- 用 stdlib 实现一个 tool registry，包含 schema validation、argument coercion 和 execution sandboxing。
- 诊断 2026 年三个未解决问题：long-horizon tool chaining、dynamic decision-making 和 memory。

## 问题

早期 tool use 问的是：模型能不能预测一个正确的 function call？现代 tool use 问的是：模型能不能在 40 步中串联工具、带着 memory、在 partial observability 下工作、从 tool failures 中恢复，并且不幻觉出不存在的工具？

Toolformer 建立了 baseline：模型可以通过 self-supervision 学会什么时候调用工具。BFCL V4 定义了 2026 年的 evaluation target。它们之间的差距，就是 production agents 生存的空间。

## 概念

### Toolformer（Schick et al., NeurIPS 2023）

想法：让模型用 candidate API calls 标注自己的 pretraining corpus。对每个 candidate 执行调用。只有当包含 tool result 后能降低下一个 token 的 loss 时，才保留这条 annotation。然后在过滤后的 corpus 上 fine-tune。

覆盖的 tools：calculator、QA system、search engines、translator、calendar。Self-supervision signal 完全取决于工具是否帮助预测文本，不需要 human labels。

规模结论：tool use 会在规模足够大时涌现。较小模型会因为 tool annotations 受损；较大模型会获益。这就是为什么 2026 年 frontier models 内置了很强的 tool use，而大多数 7B models 仍然需要显式 tool-use fine-tuning 才可靠。

### Berkeley Function Calling Leaderboard V4（Patil et al., ICML 2025）

BFCL 是 2026 年事实上的 evaluation。V4 构成：

- **Agentic (40%)**：完整 agent trajectories，包括 memory、multi-turn、dynamic decisions。
- **Multi-Turn (30%)**：带 tool chains 的交互式对话。
- **Live (10%)**：用户提交的真实 prompts（分布更难）。
- **Non-Live (10%)**：合成测试用例。
- **Hallucination (10%)**：检测什么时候不应该调用任何工具。

V3 引入了 state-based evaluation：在一串 tool sequence 之后，检查 API 的真实状态（例如 “file 是否创建了？”），而不是匹配 tool calls 的 AST。V4 增加了 web search、memory 和 format sensitivity categories。

2026 年关键发现：single-turn function calling 已接近解决。失败集中在 memory（跨 turns 携带上下文）、dynamic decision-making（根据先前结果选择工具）、long-horizon chains（20+ 步后 drift），以及 hallucination detection（没有合适工具时拒绝调用）。

### Tool schema

每个 provider 都有 schema。细节不同，但共享同一种形状：

```text
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。Descriptions 是承重结构，因为模型会读它们来选择正确工具。糟糕的 tool descriptions 是 wrong-tool-picked failures 的第一根因。

### Argument validation

不要信任任何 tool call。需要验证：

1. **Type coercion.** 模型可能在 schema 要求 int 时返回字符串 `"5"`。如果无歧义，可以 coerce；否则 reject。
2. **Enum validation.** 如果 schema 写着 `status in {"open", "closed"}`，但模型输出 `"in_progress"`，要 reject，并给出描述性错误。
3. **Required fields.** 缺少 required field -> 立刻把 error observation 喂回模型，而不是让程序崩溃。
4. **Format validation.** Dates、emails、URLs 要用具体 parser 验证，不要只用 regex。

每一次 validation failure 都应该返回结构化 observation，这样模型才能用正确形状重试。

### Parallel tool calls

现代 providers 支持在一个 assistant turn 中发出 parallel tool calls。循环如下：

1. 模型发出 3 个 tool calls，每个都有不同的 `tool_use_id`。
2. Runtime 执行它们（如果彼此独立，可以并行执行）。
3. 每个 result 都以 `tool_result` block 返回，并通过 `tool_use_id` 关联。

工程规则：把 correlation IDs 当作承重结构。调换它们，就会把错误结果路由给错误工具调用。

### Sandboxing

Tool execution 是 sandbox boundary。细节见 Lesson 09。简短版本：每个 tool 都应该声明 read/write surface、network access、timeout、memory cap。通用 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

```figure
tool-routing
```

## Build It

`code/main.py` 实现了一个 production-shape tool registry：

- JSON Schema subset validator（只用 stdlib）。
- 带 description、input schema、timeout 和 executor 的 tool registration。
- Argument coercion 和 enum validation。
- 带 correlation IDs 的 parallel tool dispatch。
- 作为 structured strings 返回的 error observations。

运行：

```bash
python3 code/main.py
```

Trace 会展示一个 mini agent 在一个 turn 中调用三个工具，其中一个故意 malformed call 会被拒绝，并返回模型可以处理的描述性错误。

## Use It

每个 provider 都有自己的 tool schema：Anthropic、OpenAI、Gemini、Bedrock。如果需要 multi-provider，使用 translation layer（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是参考 benchmark；如果 tool use 是产品核心，上线前应该对你的 agent 跑它。

## Ship It

`outputs/skill-tool-registry.md` 会为给定 task domain 生成 tool catalog、schema 和 registry。它包含 description-quality checks：每个 tool 的 description 是否告诉了模型什么时候使用它？

## 练习

1. 添加一个 “no-op” tool，让模型可以显式拒绝使用任何其他工具。用一个 BFCL-like hallucination test 来测量效果。
2. 为 int-as-string 和 float-as-string 实现 argument coercion。Coercion 从哪里开始会掩盖真实 bug？
3. 添加 per-tool timeout 和 circuit breaker（连续失败 3 次后，在 60 秒内拒绝该 tool）。这会如何改变模型的恢复方式？
4. 阅读 BFCL V4 描述。选择一个 category（例如 “multi-turn”），让你的 agent 跑 10 个 example prompts。报告 pass rate。
5. 把 stdlib validator 移植到 Pydantic 或 Zod。Pydantic/Zod 捕获了哪些 toy validator 漏掉的问题？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Function calling | “Tool use” | 带 validated schema 的 structured-output tool invocation |
| Toolformer | “Self-supervised tool annotation” | Schick 2023：保留那些结果会降低 next-token loss 的 tool calls |
| BFCL | “Berkeley Function Calling Leaderboard” | 2026 benchmark：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | “Function signature for the model” | name、description、arguments 的 JSON Schema |
| tool_use_id | “Correlation ID” | 连接 tool call 和 result；parallel dispatch 中不可或缺 |
| Hallucination detection | “Know when not to call” | V4 category：没有合适工具时拒绝调用 |
| Argument coercion | “String-to-int repair” | 针对可预测 schema mismatch 的窄修复；有歧义就 reject |
| Sandboxing | “Tool execution boundary” | 每个 tool 的 read/write surface、network、timeout、memory cap |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 eval benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — production tool schema in the Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — function tool type and Guardrails
