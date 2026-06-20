# FIPA-ACL 与言语行为的遗产

> 在 MCP 之前，在 A2A 之前，就有 FIPA-ACL。2000 年，IEEE Foundation for Intelligent Physical Agents 批准了一种 agent communication language：二十个 performatives、两种 content languages，以及一组 interaction protocols，包括 contract net、subscribe/notify、request-when。它淡出工业界，是因为 ontology 开销对 Web 来说太重；但 LLM 复兴 multi-agent systems 时，正在悄悄重新实现同一批想法，只是没有 formal semantics：JSON contracts 替代 performatives，natural language 替代 ontologies。本课会认真阅读 FIPA-ACL，让你看清 2026 年的 protocol decisions 哪些是重新发明，哪些是真正的新东西，以及当前浪潮会在哪里重新发现 2000 年代已经解决过的问题。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

2026 年的 agent-protocol landscape 很拥挤：MCP 用于 tools，A2A 用于 agents，ACP 用于 enterprise audit，ANP 用于 decentralized trust，NLIP 用于 natural-language content，此外还有 CA-MCP 和二十多个研究提案。每个 spec 都宣称自己是 foundational。

诚实地看，大多数协议都在重新发现一个二十年前非常具体的决策树。Austin（1962）和 Searle（1969）的 speech-act theory 给了我们“utterances are actions”这个想法。KQML（1993）把它变成 wire protocol。FIPA-ACL（2000 年批准）给出了参考标准化版本：二十个 performatives、SL0/SL1 content languages，以及 contract-net 和 subscribe-notify 等 interaction protocols。JADE 和 JACK 是 Java reference platforms。这个努力在 2010 年前后淡出，因为 ontology 开销太重，而 Web 技术栈赢了。

当你看 MCP 的 `tools/call`、A2A 的 task lifecycle，或 CA-MCP 的 shared context store 时，你看到的是 FIPA 决策的一个更柔软、更 JSON-native 的重述。了解这段遗产会告诉你两件事：哪些新的“创新”其实是在重新发明，以及新 specs 将重新遇到哪些旧 failure modes。

## 概念

### 一段话解释 speech acts

Austin 注意到，有些句子不是在描述世界，而是在改变世界。“I promise.” “I request.” “I declare.” 他把这些称为 performative utterances。Searle 将它们形式化为五类：assertive、directive、commissive、expressive、declarative。KQML（Finin et al., 1993）把这套理论 operationalize 到 software agents：一条 message 是 performative（动作）加 content（动作关于什么）。FIPA-ACL 修补了 KQML 的缺口，并围绕约二十个 performatives 标准化。

### 二十个 FIPA performatives（部分列表）

| Performative | Intent |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我请求你做 X” |
| `query-if` | “P 为真吗？” |
| `query-ref` | “X 的值是什么？” |
| `propose` | “我提议我们做 X” |
| `accept-proposal` | “我接受这个提案” |
| `reject-proposal` | “我拒绝这个提案” |
| `agree` | “我同意做 X” |
| `refuse` | “我拒绝做 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “你的消息无法解析” |
| `cfp` | “对 X 发起 call for proposals” |
| `subscribe` | “当 X 变化时通知我” |
| `cancel` | “取消正在进行的 X” |
| `failure` | “我尝试了 X，但失败了” |

完整列表在 `fipa00037.pdf`（FIPA ACL Message Structure）里。重点不是背下来，而是看到：其中每一个都对应一个 LLM protocol 迟早会重新加入的 primitive。

### Canonical FIPA-ACL message

```text
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载 protocol envelope；一个字段（`content`）承载 payload。其余字段正是你每次给 JSON protocol 补 retries、threading、ontology 时都会重新发明的东西。

### 两个 legacy platforms

**JADE**（Java Agent DEvelopment framework，1999-2020s）是最常用的 FIPA-compliant runtime。Agents 继承 base class，交换 ACL messages，运行在 containers 中，并用 “behaviors” 协调。它的 interaction-protocol library 内置了 contract-net、subscribe-notify、request-when、propose-accept。

**JACK**（Agent Oriented Software，商业产品）强调在 FIPA messages 之上做 BDI（Belief-Desire-Intention）reasoning。形式化程度更高，采用范围更小。

当 Web stack 吞掉 multi-agent use cases 后，两者都衰落了。MCP 和 A2A 是 2026 年的 runtime “containers”。

### 为什么 FIPA 淡出

- **Ontology overhead。** FIPA 要求共享 ontology 来解析 `content`。达成 ontologies 共识是以年计的标准化过程。Web 直接用了 HTTP + JSON。
- **没人真正使用的 formal semantics。** SL（Semantic Language）提供严谨的 truth conditions，但多数 production systems 使用 free-form content，并忽略形式化。
- **Tooling lock-in。** JADE 只有 Java；JACK 是商业产品。Polyglot teams 绕开了二者。
- **Internet 赢了技术栈。** REST，然后 JSON-RPC，再到 gRPC，取代了 ACL 的 transport。

### LLM 复兴是 FIPA-lite

比较一个 FIPA `request` 和一个 MCP `tools/call`：

```text
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同样的 envelope，不同的语法。二者都携带：谁、发给谁、intent、payload、correlation id。二者相对彼此都不是革命，只是在同一设计上做了不同取舍。

Liu et al. 2025 的 survey（“A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP”，arXiv:2505.02279）明确指出了这条 lineage：MCP 对应 tool-use speech acts，A2A 对应 agent-peer speech acts，ACP 对应 audit-trail speech acts，ANP 对应 decentralized-identity extensions。新的 specs 是 JSON syntax 和更松语义下的 ACL descendants。

### 直说这笔 trade-off

**FIPA 给了你、而 modern specs 放弃的东西：**

- Formal semantics：你可以证明 `inform` 表示 sender believes the content。
- Canonical catalog of performatives：你不需要重新争论“我们是否应该有 `cancel`？”
- 几十年的 interaction-protocol patterns：contract-net、subscribe-notify、propose-accept，并且有已知 correctness properties。

**Modern specs 给了你、而 FIPA 没给的东西：**

- 与所有现代工具兼容的 JSON-native payloads。
- LLM 可以解释的 natural-language content，不需要 hand-coded ontology。
- Web-stack transport（HTTP、SSE、WebSocket）。
- 通过 self-describing documents 进行 capability discovery（MCP `listTools`、A2A Agent Card）。

用更松的 intent semantics 换更容易的实现。这就是准确的 trade。

### 值得移植的 interaction protocols

FIPA 发布了大约 15 个 interaction protocols。三个值得带进 LLM multi-agent systems：

1. **Contract Net Protocol（CNP）。** Manager 发出 `cfp`（call for proposals）；bidders 用 `propose` 回复；manager 接受或拒绝。这是 canonical task-market pattern（Phase 16 · 16 Negotiation）。
2. **Subscribe/Notify。** Subscriber 发送 `subscribe`；topic 变化时 publisher 发送 `inform`。这就是 2026 年的每个 event-bus。
3. **Request-When。** “当条件 Y 成立时做 X。” 带 pre-conditions 的 delayed-action。2026 年的 analog 是 durable workflow engines 中的 deferred tasks（Phase 16 · 22 Production Scaling）。

每一种都能干净映射到 modern message queues、HTTP + polling，或 SSE streaming。

### 放弃 ontology 后会坏在哪里

没有 shared ontology，agents 会从 natural-language content 中推断 meaning。2026 年已记录的 failure mode 是 **semantic drift**：两个 agents 对同一个词（`"customer"`）使用略有不同的概念，receiver agent 基于错误解释采取行动，而没有 schema validator 能抓住它。FIPA 的 ontology requirement 会在 parse time 拒绝这条 message。

不走 full ontology 的缓解方式：

- 在 `content` 上使用 JSON Schema：在线路层拒绝 structural errors。
- Typed artifacts（A2A）：拒绝错误 modality。
- 在 envelope 中放入 explicit performative：即使 content 是 natural language，也让 intent 不含糊。

### 2026 specs 到 speech-act heritage 的映射

| Modern spec | FIPA analog | What it keeps | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | explicit intent, correlation id | formal semantics, ontology |
| MCP `resources/read` | `query-ref` | explicit intent, correlation id | formal semantics |
| A2A Task lifecycle | contract-net + request-when | async lifecycle, state transitions | formal completeness guarantees |
| A2A streaming events | subscribe/notify | async push | typed-predicate subscription |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | multi-writer shared memory | logical consistency model |
| NLIP | natural-language content | LLM-native | schema |

从上到下读这张表，模式很清楚：保留结构 primitive，放弃形式化，让 LLMs 填补歧义。

## Build It

`code/main.py` 实现了一个 pure-stdlib FIPA-ACL translator。它会 encode 和 decode canonical ACL envelope，并展示每一种 MCP / A2A message shape 如何归约到同样的七个字段。Demo 会：

- 把五条 MCP-style 和 A2A-style messages 编码为 FIPA-ACL。
- 把 FIPA-ACL 解码回 modern equivalent。
- 用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个 manager 和三个 bidders 之间运行一个 toy Contract Net negotiation。

运行：

```bash
python3 code/main.py
```

输出会以 side-by-side trace 展示每条 modern message 的 2026 JSON form 和 FIPA-ACL form，然后展示 contract-net bid 的 round-trip。同样的 protocol primitives 经受住了 round-trip；只有 syntax 不同。

## Use It

`outputs/skill-fipa-mapper.md` 是一个读取任意 agent-protocol spec 并生成 FIPA-ACL mapping 的 skill。采用新 protocol 前使用它来回答：“这是真正的新东西，还是 JSON syntax 下的 `inform`？”

## Ship It

不要把 FIPA-ACL 带回来。带回它的 checklist：

- 每条 message 的 intent primitive（performative）是什么？
- request-response 和 cancellation 是否有 correlation id？
- 是否有 explicit content language（JSON-RPC、plain text、structured typed artifact）？
- interaction protocols 是 first-class，还是你正在从零重新实现 contract-net？
- 当两个 agents 对 content meaning 有分歧时会发生什么（semantic drift）？

把这五个问题写进任何新 protocol 的文档，再把它放进 production。

## 练习

1. 运行 `code/main.py`。观察 round-trip encoding。识别 `tools/call`、`resources/read` 和 A2A task creation 分别对应哪个 FIPA performative。
2. 给 contract-net demo 扩展一个 `cancel` performative，让 manager 可以在 bidding 过程中撤回 task。`cancel` 解决了什么 retries alone 解决不了的 failure case？
3. 阅读 FIPA ACL Message Structure（http://www.fipa.org/specs/fipa00037/）第 4.1-4.3 节。选择本课未覆盖的一个 performative，并描述它的 modern JSON-RPC analog。
4. 阅读 Liu et al., arXiv:2505.02279。对 MCP、A2A、ACP、ANP 分别列出它们保留和放弃的 FIPA performative families。
5. 为你自己的系统中 `request` performative 的 `content` 字段设计一个 minimal JSON-Schema。这个 schema 给了你哪些 pure natural-language 没有的东西，又带来了什么成本？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Speech act | “An utterance that does something” | Austin/Searle：utterances as actions。ACL 的理论源头。 |
| FIPA | “That old XML thing” | IEEE Foundation for Intelligent Physical Agents。2000 年标准化 ACL。 |
| ACL | “Agent Communication Language” | FIPA 的 envelope format：performative + content + metadata。 |
| Performative | “The verb” | Message 的 intent class：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA's predecessor” | Knowledge Query and Manipulation Language（1993）。更简单、更窄。 |
| Ontology | “Shared vocabulary” | 对 content language 所讨论概念的 formal definition。 |
| SL0 / SL1 | “FIPA content languages” | Semantic Language levels 0 and 1，formal content language family。 |
| Contract Net | “Task market” | Manager 发出 cfp；bidders propose；manager accept。Canonical interaction protocol。 |
| Interaction protocol | “Pattern of messages” | 一串具有已知 correctness 的 performatives：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 把 modern specs 连接到 FIPA heritage 的 2025 年 canonical survey
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的 envelope format
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整 performative catalog
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request` / `query-ref` 的现代 tool-use equivalent
- [A2A specification](https://a2a-protocol.org/latest/specification/) — contract-net 和 subscribe-notify 的现代 agent-peer equivalent
