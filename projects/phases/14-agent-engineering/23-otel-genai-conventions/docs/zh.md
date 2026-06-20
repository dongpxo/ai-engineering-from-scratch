# OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 agent telemetry 的标准 schema。Span names、attributes 和 content-capture rules 在不同 vendors 之间趋同，因此 agent traces 在 Datadog、Grafana、Jaeger 和 Honeycomb 中表达同一件事。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## 学习目标

- 说出 GenAI span categories：model/client、agent、tool。
- 区分 `invoke_agent` CLIENT spans 和 INTERNAL spans，并说明各自在什么时候适用。
- 列出顶层 GenAI attributes：provider name、request model、data-source ID。
- 解释 content-capture contract：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、external-reference recommendation。

## 问题

每个 vendor 都会发明自己的 span names。Ops teams 最后不得不为每个 framework 构建一套 dashboard。OpenTelemetry 的 GenAI SIG 通过定义一个全生态系统都能对齐的标准，修复这个问题。

## 概念

### Span categories

1. **Model / client spans.** 覆盖 raw LLM calls。由 provider SDKs（Anthropic、OpenAI、Bedrock）和 framework model adapters 发出。
2. **Agent spans.** `create_agent`（agent 被构造时）和 `invoke_agent`（agent 运行时）。
3. **Tool spans.** 每次 tool invocation 一个 span；通过 parent-child relation 连接到 agent span。

### Agent span naming

- Span name：如果有名字，使用 `invoke_agent {gen_ai.agent.name}`；否则 fallback 到 `invoke_agent`。
- Span kind：
  - **CLIENT**：用于 remote agent services（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL**：用于 in-process agent frameworks（LangChain、CrewAI、本地 ReAct）。

### Key attributes

- `gen_ai.provider.name`：`anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model`：model ID。
- `gen_ai.response.model`：解析后的 model（可能因为 routing 与 request 不同）。
- `gen_ai.agent.name`：agent identifier。
- `gen_ai.operation.name`：`chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id`：用于 RAG，表示查询了哪个 corpus 或 store。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 都有 technology-specific conventions。

### Content capture

默认规则：instrumentations SHOULD NOT 默认捕获 inputs/outputs。Capture 需要通过这些字段 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的 production pattern：把 content 存在外部（S3、你的 log store），在 spans 上记录 references（pointer IDs，而不是 prose）。这就是 Lesson 27 的 content-poisoning defense 接到 observability 上的形状。

### Stability

截至 2026 年 3 月，大多数 conventions 仍是 experimental。通过下面的环境变量 opt in 到 stable preview：

```bash
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 会把 GenAI attributes 原生映射到它的 LLM Observability schema。其他 backends（Grafana、Honeycomb、Jaeger）支持 raw attributes。

### 这个 pattern 容易出错的地方

- **Capturing full prompts in spans.** PII、secrets、customer data 会进入 ops 可读的 traces。应该存到外部。
- **No `gen_ai.provider.name`.** 缺少 attribution 时，multi-provider dashboards 会坏掉。
- **Spans without parent links.** 出现 orphaned tool spans。始终要 propagate context。
- **Not setting stability opt-in.** Backend upgrade 后 attributes 可能被重命名。

## Build It

`code/main.py` 实现了一个符合 GenAI conventions 的 stdlib span emitter：

- 带 GenAI attribute schema 的 `Span`。
- 带 `start_span` 和 nested contexts 的 `Tracer`。
- 一个 scripted agent run，会发出：`create_agent`、`invoke_agent`（INTERNAL）、per-tool spans、用于 LLM calls 的 `chat` spans。
- 一个 content-capture mode：把 prompts 存到外部，并在 spans 上记录 IDs。

运行：

```bash
python3 code/main.py
```

输出：一棵包含所有 required GenAI attributes 的 span tree，以及一个展示 opt-in content references 的 “external store”。

## Use It

- **Datadog LLM Observability**（v1.37+）：原生映射 attributes。
- **Langfuse / Phoenix / Opik**（Lesson 24）：auto-instrument 生态系统。
- **Jaeger / Honeycomb / Grafana Tempo**：raw OTel traces；从 GenAI attributes 构建 dashboards。
- **Self-hosted**：运行带 GenAI processor 的 OTel Collector。

## Ship It

`outputs/skill-otel-genai.md` 会把 OTel GenAI spans 接入现有 agent，并设置 content-capture defaults 和 external-reference storage。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ per-tool spans instrument 你的 Lesson 01 ReAct loop。发送到一个 Jaeger instance。
2. 添加 “references only” 模式的 content capture：prompts 存到 SQLite，span attributes 只携带 row IDs。
3. 阅读 `gen_ai.data_source.id` 的 spec。把它接入你的 Lesson 09 Mem0 search。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，并确认 collector 没有重命名你的 attributes。
5. 只依靠 GenAI attributes 构建一个 dashboard：“哪些 tool errors 和哪些 models 相关？”

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI SIG | “OpenTelemetry GenAI group” | 定义 schema 的 OTel working group |
| invoke_agent | “Agent span” | 表示一次 agent run 的 span name |
| CLIENT span | “Remote call” | 调用 remote agent service 的 span |
| INTERNAL span | “In-process” | in-process agent run 的 span |
| gen_ai.provider.name | “Provider” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG source” | 某次 retrieval hit 来自哪个 corpus/store |
| Content capture | “Prompt logging” | Opt-in 捕获 messages；production 中应外部存储 |
| Stability opt-in | “Preview mode” | 用来 pin experimental conventions 的 env var |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI spans by default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel spans built in
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
