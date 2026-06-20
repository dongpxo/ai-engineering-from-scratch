# Parallel / Swarm / Networked 架构

> 和 supervisor 相比：这里没有中心决策者。Agents 读取共享 event bus，异步领取工作，再把结果写回去。LangGraph 明确支持面向去中心化、动态环境的 “Swarm Architecture”。Matrix（arXiv:2511.21686）把 control flow 和 data flow 都表示成通过分布式队列传递的序列化消息，以消除 orchestrator bottleneck。取舍也很明确：用 determinism 和 traceability 换 scalability。Swarm 适合有大量独立子问题的任务；不适合需要单一连贯计划的任务。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题

Supervisor 可以扩展到几个 workers。那几百个呢？Supervisor 本身会变成瓶颈：谁做什么的每个决定都要流经一个 agent。一个缓慢的 plan step 会卡住整个系统。

Swarm architectures 把设计反过来。不是由中心 planner 派发工作，而是 workers 从共享 queue 中领取工作。“协调”被烘焙进 event bus semantics。没有 orchestrator；系统会一直扩展，直到 queue 成为瓶颈。

## 概念

### 形状

```text
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

没有 orchestrator。每个 worker 都重复同一个循环：拉取任务、处理任务、写入结果（并可选地 enqueue follow-ups）。

### 什么时候适合 swarm

- **大量独立任务。** Scraping、transforming、classifying。任务彼此不依赖。
- **工作时长差异很大。** 如果有些任务耗时 100ms，另一些耗时 10s，swarm 会自动平衡负载——快 worker 会继续拉下一个任务。Supervisor 必须提前预测时长。
- **吞吐优先于确定性。** 你关心的是总完成时间，而不是严格顺序。

### 什么时候 swarm 会失败

- **有序 workflows。** 如果 step 3 需要 step 2 的输出，swarm 可能让 step 3 在 step 2 完成前触发。
- **全局计划任务。** 复杂研究问题受益于 planner。研究型 swarm 会产出独立事实，而不是一份连贯报告。
- **Debugging。** 没有中心日志且工作异步时，复现 bug 的成本很高。

### Matrix（arXiv:2511.21686）

Matrix 是 2025 年把 swarm 推到自然终点的论文：control flow 和 data flow 都是 distributed queues 上的序列化消息。没有中心 coordinator。Fault tolerance 来自 message durability。Scalability 是 message broker 的问题，而不是系统本身的问题。

贡献：一种 programming model，其中 multi-agent coordination 变成“这个 agent 订阅哪个 message topic？”，而不是“supervisor 下一步选择哪个 agent？”这让系统看起来像 pub/sub event mesh。

### LangGraph 的 Swarm Architecture

LangGraph 2025 docs 明确把 “Swarm Architecture” 描述为 multi-agent patterns 之一：agents 是 nodes，但 edges 形成带 cycles 的 directed graph，并且任何 node 都可以从 pool 中被激活。Worker 根据条件从可用工作中选择，而不是由 supervisor 指派。

### 失败模式：starvation 和 hot-spotting

如果所有 workers 都拉取最快可用的任务，long-running tasks 可能直到最后才被选择。经典 queue starvation。

缓解方法：

- 带显式 aging 的 priority queues（随等待时间提高 priority）。
- Worker specialization：某些 workers 只接 “long” tasks。
- Back-pressure：限制进入 queue 的 fast tasks 数量。

### 与 content-based routing 的连接

Swarm 很自然地和 content-based routing（Lesson 22）配对。不是使用一个 generic queue，而是每种 message type 一个 queue。Specialist workers 只订阅自己的类型。这是 message-bus architectures 能扩展到数千 agents 的基础。

## Build It

`code/main.py` 实现了一个由 4 个 worker threads 组成的 swarm，它们从共享 `queue.Queue` 拉取任务。任务有不同 duration（有些快，有些慢）。Demo 对比：

- **Sequential baseline：** 一个 worker 串行处理所有任务。
- **Fixed assignment：** 每个任务预先分配给特定 worker（supervisor-style）。
- **Swarm：** Workers 从共享 queue 拉取任务。

Swarm 会自动平衡负载；fixed assignment 会在某个 assigned task 很慢时，让快 workers 闲置。

运行：

```bash
python3 code/main.py
```

输出会展示 per-worker task counts（swarm 分配不均但近似最优）和 wall-clock times。

## Use It

`outputs/skill-swarm-fit.md` 会评估一个任务应该使用 swarm 还是 supervisor。输入包括：task independence、duration variance、ordering requirements、debuggability needs。

## Ship It

Checklist：

- **Priority queue with aging。** 防止 long-task starvation。
- **Worker idempotency。** 如果 worker 中途 crash，一个任务可能会被拉取多次。Workers 必须是 idempotent。
- **Durable queue。** 生产中使用 Kafka、Redis Streams 或 database-backed queue。`queue.Queue` 只在内存中。
- **Observability per task。** 每个 task 都有 trace ID；每个 worker 都带着它记录 start/end。
- **Back-pressure。** 如果 queue 增长速度快于 workers 消耗速度，就放慢 producer。

## 练习

1. 运行 `code/main.py`。在 variable-duration workload 上，swarm 比 sequential 快多少？比 fixed assignment 快多少？
2. 添加 priority queue 变体（使用 `queue.PriorityQueue`）。按任务的 “importance” 字段分配 priority。观察在持续负载下，低优先级任务是否会 starve。
3. 实现一个 hot-spot detector：当任意 worker 处理的任务数达到最慢 worker 的 3× 时记录日志。这说明 task-duration distribution 有什么特征？
4. 阅读 Matrix 论文（arXiv:2511.21686）的 abstract 和第 3 节。指出 Matrix 接受的一个具体取舍：它获得了什么 scalability gain，又放弃了什么（traceability、determinism）。
5. 把 swarm demo 改成使用由 `(task_type, payload)` tuples 组成的 `queue.Queue`，并让 workers 只订阅特定类型。任务异构时，哪些 routing rules 是合理的？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Swarm architecture | “Decentralized agents” | Workers 从共享 queue 拉取任务；没有中心 orchestrator |
| Event bus | “Agents subscribe to topics” | 根据 type 或 content 把任务路由给 workers 的 message broker |
| Starvation | “任务永远不运行” | 因为更高优先级工作持续到达，低优先级任务一直无法被选择 |
| Hot-spotting | “一个 worker 被淹没” | 负载不均衡，一个 worker 拿到大多数任务 |
| Back-pressure | “放慢 producer” | 当 queue 填满时，向上游发出停止生产的信号机制 |
| Idempotent worker | “可安全重跑” | 一个任务被处理两次也产生相同结果；因为 workers 可能中途 crash，所以必须具备 |
| Durable queue | “跨 crash 存活” | 由磁盘或 replicated storage 支撑的 queue；worker crash 时任务不会丢失 |
| Matrix framework | “完整 message-passing swarm” | Data flow 和 control flow 都是 distributed queues 上的序列化消息 |

## 延伸阅读

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents)：明确的 swarm 支持。
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686)：完整 message-passing swarm。
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system)：为什么一个具体生产系统明确选择 supervisor 而不是 swarm。
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/)：event-driven actor rewrite，比 v0.2 的 GroupChat 更接近 swarm。
