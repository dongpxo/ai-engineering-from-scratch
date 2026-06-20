# Agno 与 Mastra：生产运行时

> Agno（Python）与 Mastra（TypeScript）是 2026 年的一组生产运行时搭配。Agno 目标是微秒级 agent 实例化和无状态 FastAPI 后端。Mastra 则基于 Vercel AI SDK substrate，提供 agents、tools、workflows、统一模型路由和 composite storage。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## 学习目标

- 识别 Agno 的性能目标，以及这些目标在什么时候重要。
- 命名 Mastra 的三个 primitives：Agents、Tools、Workflows，并说明支持的 server adapters。
- 解释为什么 stateless session-scoped FastAPI backend 是推荐的 Agno 生产路径。
- 针对给定 stack，在 Agno 与 Mastra 之间选择（Python-first vs TypeScript-first）。

## 问题

LangGraph、AutoGen、CrewAI 都偏 framework-heavy。那些想要“就给我 agent loop，够快，而且待在我的 runtime 里”的团队，会转向 Agno（Python）或 Mastra（TypeScript）。两者都用一部分 framework-owned primitives 换取原始速度，以及与周边技术栈更紧密的贴合。

## 概念

### Agno

- Python runtime，前身是 Phi-data。
- “No graphs, chains, or convoluted patterns — just pure python.”
- 文档中的性能目标：约 2μs agent instantiation、约 3.75 KiB memory per agent、约 23 个 model providers。
- 生产路径：stateless session-scoped FastAPI backend。每个 request 启动一个 fresh agent；session state 存在 DB 中。
- 原生 multimodal（text、image、audio、video、file）和 agentic RAG。

当你每秒有数千个 short-lived agents（chat fan-in、evaluation pipelines）时，这些速度目标很重要。当一个 agent 要跑 10 分钟时，它们就没那么重要。

### Mastra

- TypeScript，建立在 Vercel AI SDK 之上。
- 三个 primitives：**Agents**、**Tools**（Zod-typed）、**Workflows**。
- Unified Model Router：跨 94 个 providers 的 3,300+ models（2026 年 3 月）。
- Composite storage：memory、workflows、observability 可以分别接到不同 backends；规模化 observability 推荐 ClickHouse。
- Apache 2.0，但源码里的 `ee/` directories 使用 source-available enterprise license。
- 支持 Express、Hono、Fastify、Koa 的 server adapters；一等支持 Next.js 和 Astro integration。
- 附带 Mastra Studio（localhost:4111）用于 debugging。
- 1.0 发布时（2026 年 1 月）有 22k+ GitHub stars，300k+ weekly npm downloads。

### 定位

两者都不是在试图成为 LangGraph。它们竞争的是：

- **Language fit.** Python-first teams 选 Agno；TypeScript-first 选 Mastra。
- **Runtime ergonomics.** Agno = 接近零 overhead；Mastra = 与 Vercel ecosystem 集成。
- **Observability.** 两者都能与 Langfuse/Phoenix/Opik（Lesson 24）集成，但 Mastra Studio 是 first-party。

### 什么时候选择哪个

- **Agno**：Python backend、大量 short-lived agents、强性能要求、FastAPI shop。
- **Mastra**：TypeScript backend、Next.js / Vercel deploy、统一 multi-provider model routing、Zod-typed tools。
- **LangGraph**（Lesson 13）：当 durable state 和显式 graph reasoning 比原始速度更重要时。
- **OpenAI / Claude Agent SDK**：当你想要 provider 的产品化形态时（Lessons 16-17）。

### 这个模式会在哪里出错

- **Perf-for-perf's-sake.** 因为“2μs”听起来很厉害而选择 Agno，但实际 workload 是每个 request 一次很慢的 agent call。Overhead 并不是瓶颈。
- **Ecosystem lock-in.** Mastra 的 Vercel-flavored integration 在 Vercel 上是优势，在其他地方可能是负担。
- **Enterprise license confusion.** Mastra 的 `ee/` directories 是 source-available，不是 Apache 2.0。如果你计划 fork，要读清楚 licenses。

## Build It

这一课主要是比较型内容：单个 code artifact 很难公平呈现两个 frameworks。见 `code/main.py` 中的 side-by-side toy：一个最小的“运行 agent、stream output、persist session”流程实现了两遍（一遍 Agno-shaped，一遍 Mastra-shaped）。

运行：

```bash
python3 code/main.py
```

你会看到两条结构不同但功能等价的 traces。

## Use It

- **Agno**：需要速度和 FastAPI shape 的 Python backend。
- **Mastra**：拥有许多 providers 和 workflow primitives 的 TypeScript backend。
- 两者都提供 first-party observability hooks，也都能与 Langfuse 集成。

## Ship It

`outputs/skill-runtime-picker.md` 会根据 stack、latency budget 和 operational shape，在 Agno、Mastra、LangGraph 或 provider SDK 之间做选择。

## 练习

1. 阅读 Agno docs。把 stdlib ReAct loop（Lesson 01）移植到 Agno。什么消失了？什么保留了？
2. 阅读 Mastra docs。把同一个 loop 移植到 Mastra。Tool typing 中发生了什么变化（Zod vs nothing）？
3. Benchmark：在你的 stack 上测量 agent instantiation latency。Agno 的 2μs 对你的 workload 重要吗？
4. 设计一次 migration：如果你一直在 Python 中运行 CrewAI，迁到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` license terms。哪些限制会影响一个 open-source fork？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | 面向 94 个 providers 中 3,300+ models 的单一 client |
| Composite storage | "Multiple backends" | Memory/workflows/observability 分别接到不同 store |
| Mastra Studio | "Local debugger" | 用于 introspecting agents 的 localhost:4111 UI |
| Source-available | "Not OSS" | 允许阅读源码但限制商业使用的 license |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework)：performance targets、FastAPI integration。
- [Mastra docs](https://mastra.ai/docs)：primitives、server adapters、Model Router。
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：stateful-graph alternative。
- [Comet Opik](https://www.comet.com/site/products/opik/)：Mastra integrations 引用的 observability comparisons。
