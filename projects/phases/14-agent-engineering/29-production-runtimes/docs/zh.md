# 生产运行时：Queue、Event、Cron

> 生产 agents 运行在六种 runtime shapes 上：request-response、streaming、durable execution、queue-based background、event-driven、scheduled。先选择运行时形状，再选择框架。Observability 在每种形状里都是承重结构。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## 学习目标

- 命名六种 production runtime shapes，并把每一种匹配到对应 framework / product pattern。
- 解释为什么 durable execution（LangGraph）对 long-horizon tasks 很重要。
- 描述 event-driven runtime，以及 Claude Managed Agents 适合什么时候使用。
- 解释为什么 multi-step agents 中 observability 是 load-bearing。

## 问题

生产 agents 的失败方式，是 Jupyter notebook 暴露不出来的：第 37 步网络超时、用户在 voice call 中途挂断、cron job 在机器重启时死掉、background worker 内存耗尽。Runtime shape 决定了哪些失败可以存活下来。

## 概念

### Request-response

- 同步 HTTP。用户等待任务完成。
- 只适合短任务（<30s）。
- Stacks：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- Observability：标准 HTTP access logs + OTel spans。

### Streaming

- 用 SSE 或 WebSocket 做 progressive output。
- LiveKit 把这种模式扩展到 voice/video 的 WebRTC（Lesson 22）。
- Stacks：任意支持 streaming 的框架 + 能处理 SSE/WS 的 frontend。
- Observability：per-chunk timing、first-token latency、tail latency。

### Durable execution

- 每一步后 checkpoint state；失败后自动 resume。
- AutoGen v0.4 actor model 把失败隔离到单个 agent（Lesson 14）。
- LangGraph 的核心差异点（Lesson 13）。
- 当 step count 未知且 recovery cost 很高时，这是必需品。

### Queue-based / background

- Job 进入 queue，workers 拉取执行，结果通过 webhooks 或 pub/sub 回流。
- 对 long-horizon agents 是必需的（Anthropic computer use announcement 中提到每个任务几十到几百步）。
- Stacks：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、custom。
- Observability：queue depth、per-job latency distribution、DLQ size。

### Event-driven

- Agents 订阅 triggers：新邮件、PR opened、cron fire。
- Claude Managed Agents 开箱覆盖这种场景（Lesson 17）。
- CrewAI Flows（Lesson 15）用于组织 event-driven deterministic workflows。
- Observability：trigger source、event-to-start latency、agent latency。

### Scheduled

- Cron-shaped agents 会周期性运行。
- 和 durable execution 结合起来，这样失败的 nightly run 可以在下一 tick resume。
- Stacks：Kubernetes CronJob + durable framework；hosted（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows**：用于 event-driven production。
- **Agno**：用于 Python microservices 的 stateless FastAPI。
- **Mastra**：server adapters（Express、Hono、Fastify、Koa），用于嵌入。
- **Pipecat Cloud / LiveKit Cloud**：用于 managed voice（Lesson 22）。
- **Claude Managed Agents**：用于 hosted long-running async。

### Observability 是承重结构

没有 OpenTelemetry GenAI spans（Lesson 23）加上 Langfuse/Phoenix/Opik backend（Lesson 24），你无法调试一个在第 40 步失败的 multi-step agent。这对生产不是可选项。它决定了你是在 “快速调试”，还是在 “从头 replay 并临时加更多 logging”。

### Production runtimes 容易在哪里失败

- **Wrong shape choice.** 给一个 5 分钟任务选择 request-response。用户挂断；workers 堆积；retries 叠加。
- **No DLQ.** Queue workers 没有 dead-letter。失败 jobs 消失。
- **Opaque background work.** Background agent 运行时没有 trace export。直到用户报告前，失败都不可见。
- **Skipping durable state.** 任何超过 30 秒、且你不能接受从头重启的 run，都需要 durable execution。

## Build It

`code/main.py` 是一个 stdlib multi-shape demo：

- Request-response endpoint（普通函数）。
- Streaming handler（generator）。
- 带 DLQ 的 queue-based worker。
- Event trigger registry。
- Cron-shaped scheduler。

运行它：

```bash
python3 code/main.py
```

输出会展示五段 traces，说明同一个任务在每种形状下的行为。同一套 agent logic，不同的 outer shells。第六种形状 durable execution 已经在 Lesson 13 通过 LangGraph checkpointing 单独覆盖，所以这里故意不重复实现。

## Use It

- **Request-response**：用于 chat-style UX。
- **Streaming**：用于 progressive responses。
- **Durable**：用于 long-horizon tasks。
- **Queue**：用于 batch / async / long-running。
- **Event**：用于 agent reactivity。
- **Cron**：用于 housekeeping（memory consolidation、evals、cost reports）。

## Ship It

`outputs/skill-runtime-shape.md` 会为一个任务选择 runtime shape，并接好 observability requirements。

## 练习

1. 把 Lesson 01 的 ReAct loop 移植到你技术栈里的全部六种形状。哪种 shape 适合哪个 product surface？
2. 给 queue-based demo 添加 DLQ。模拟 10% job failure；暴露 DLQ size。
3. 写一个 cron-triggered eval agent，每晚针对当天 top 20 traces 运行。
4. 实现带 backpressure 的 streaming：如果 client 很慢，就暂停 agent。这会如何影响 turn budget？
5. 阅读 Claude Managed Agents docs。什么时候你会把 self-hosted long-horizon agent 迁到 managed？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Request-response | “Synchronous” | 用户等待；只适合短任务 |
| Streaming | “SSE / WS” | Progressive output；UX 更好；latency 可按 chunk 观测 |
| Durable execution | “Resume from failure” | Checkpointed state；从最后一步 restart |
| Queue-based | “Background jobs” | Producer / worker pool / DLQ |
| Event-driven | “Trigger-based” | Agent 响应外部 events |
| DLQ | “Dead-letter queue” | 失败 jobs 的停车场 |
| Claude Managed Agents | “Hosted harness” | Anthropic-hosted long-running async，带 caching + compaction |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：durable execution details。
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview)：hosted long-running async。
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use)：“dozens-to-hundreds of steps per task”。
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：actor-model fault isolation。
