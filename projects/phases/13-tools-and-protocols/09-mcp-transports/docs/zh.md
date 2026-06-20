# MCP 传输：stdio vs Streamable HTTP vs SSE 迁移

> stdio 只适合本地，离开本机就不成立。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE transport 已经废弃，并会在 2026 年中移除。选错 transport 会带来一次迁移；选对 transport，会得到一个可远程托管、支持 session continuity，并能防 DNS rebinding 的 MCP server。

**Type:** Learn
**Languages:** Python (stdlib, Streamable HTTP endpoint skeleton)
**Prerequisites:** Phase 13 · 07, 08 (MCP server and client)
**Time:** ~45 minutes

## 学习目标

- 根据部署形态（本地 vs 远程、单进程 vs fleet）在 stdio 和 Streamable HTTP 之间做选择。
- 实现 Streamable HTTP 的单 endpoint 模式：POST 处理 request，GET 处理 session stream。
- 强制执行 `Origin` 校验和 session id 语义，防御 DNS rebinding。
- 在 2026 年中移除 deadline 之前，把 legacy HTTP+SSE server 迁移到 Streamable HTTP。

## 问题

第一个 MCP 远程 transport（2024-11）是 HTTP+SSE：两个 endpoint，一个接收 client 的 POST，另一个是 Server-Sent Events channel，用于 server-to-client stream。它能工作，但很笨重：每个 session 需要两个 endpoint，在某些 CDN 前面会破坏 cache，并且强依赖长连接 SSE，而一些 WAF 会非常激进地终止这类连接。

2025-03-26 版规范用 Streamable HTTP 取代了它：一个 endpoint，POST 用于 client request，GET 用于建立 session stream，两者共享 `Mcp-Session-Id` header。从那之后新建或迁移的 server 都使用 Streamable HTTP。旧 SSE 模式正在废弃：Atlassian Rovo 在 2026-06-30 移除它，Keboola 在 2026-04-01 移除它，大多数剩余企业 server 会在 2026 年底前移除。

而 stdio 对本地 server 仍然重要。Claude Desktop、VS Code 和每个 IDE 形态的 client 都会通过 stdio 启动 server。正确的心智模型是：stdio 用于“这台机器”，Streamable HTTP 用于“跨网络”。不要混用。

## 概念

### stdio

- 子进程 transport。Client 启动 server，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。以换行分隔。
- 没有 session id；进程身份就是 session。
- 不需要 auth（子进程继承父进程的信任边界）。
- 永远不要用于远程 server。否则你需要 SSH 或 socat 做 tunnel，而那时就应该直接使用 Streamable HTTP。

### Streamable HTTP

单 endpoint `/mcp`（或任意路径）。支持三种 HTTP method：

- **POST /mcp。** Client 发送 JSON-RPC message。Server 返回单个 JSON response，或者返回一个包含一个或多个 response 的 SSE stream（适合 batched response，以及和该 request 相关的 notification）。
- **GET /mcp。** Client 打开一个长连接 SSE channel。Server 用它发送 server-to-client request（sampling、notification、elicitation）。
- **DELETE /mcp。** Client 显式终止 session。

Session 由 `Mcp-Session-Id` header 标识。Server 在第一次 response 上设置它，client 在后续每个 request 中回显它。Session id 必须是密码学随机的（128+ bits）；出于安全原因，client 自选的 id 会被拒绝。

### 单 endpoint vs 两个 endpoint

旧规范中的双 endpoint 模式在 2026 年仍然可以调用，规范称它为“legacy compatible”。但所有新 server 都应该使用单 endpoint。官方 SDK 产出的是单 endpoint；只有在连接尚未迁移的 remote 时，才使用 legacy 模式。

### `Origin` 校验与 DNS rebinding

浏览器今天不是 MCP client，但攻击者可以构造一个网页，让浏览器向 `localhost:1234/mcp` 发 POST，而用户的本地 MCP server 可能正监听在那里。如果 server 不检查 `Origin`，浏览器的 same-origin policy 不会救你，因为 `Origin: http://evil.com` 是有效的跨源请求。

2025-11-25 版规范要求 server 拒绝 `Origin` 不在 allowlist 中的 request。Allowlist 通常包含 MCP client host（`https://claude.ai`、`vscode-webview://*`）以及本地 UI 使用的 localhost 变体。

### Session id 生命周期

1. Client 发送第一个 request，不带 `Mcp-Session-Id`。
2. Server 分配一个随机 id，并在 response header 中设置 `Mcp-Session-Id`。
3. Client 在所有后续 request 中回显该 header，并在用于 stream 的 `GET /mcp` 中也带上它。
4. Server 可以撤销 session；client 后续 request 会收到 404，必须重新 initialize。
5. Client 可以显式 DELETE session，完成干净关闭。

### Keepalive 与 reconnect

SSE 连接会断。Client 通过使用同一个 `Mcp-Session-Id` 重新发起 GET 来恢复。Server 必须把中断期间错过的 event 排队（直到合理窗口），并通过 client 回显的 `last-event-id` header replay。

Phase 13 · 13 会覆盖 Tasks；Tasks 让长时间运行的工作即使遇到完整 session reconnect 也能存活。

### 向后兼容探测

一个想同时支持旧 server 和新 server 的 client：

1. POST 到 `/mcp`。
2. 如果 response 是带 JSON 或 SSE 的 `200 OK`，这是 Streamable HTTP。
3. 如果 response 是 `200 OK`，`Content-Type: text/event-stream`，并且有一个指向 secondary endpoint 的 `Location` header，这是 legacy HTTP+SSE；跟随 `Location`。

### Cloudflare、ngrok 与托管

2026 年，生产远程 MCP server 运行在 Cloudflare Workers（配合 MCP Agents SDK）、Vercel Functions，或容器化 Node/Python 上。关键点是：你的托管环境必须支持 SSE GET 所需的长连接 HTTP。Vercel 的 free tier 限制为 10 秒，不适合。Cloudflare Workers 支持无限期 stream。

### Gateway composition

当你用一个 gateway（Phase 13 · 17）挡在多个 MCP server 前面时，gateway 本身是一个单一 Streamable HTTP endpoint，负责重写 session id 并 multiplex upstream。工具在 gateway 层合并；client 看到的是一个逻辑 server。

### Transport failure modes

- **stdio SIGPIPE。** 子进程在写入中途死亡会引发 SIGPIPE；server 应该干净退出。Client 应该检测 EOF，并把 session 标记为 dead。
- **HTTP 502 / 504。** Cloudflare、nginx 和其它 proxy 在 upstream failure 时会发出这些状态码。Streamable HTTP client 应该在短暂 backoff 后重试一次。
- **SSE connection drop。** TCP RST、proxy timeout 或 client 网络变化会关闭 stream。Client 使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **Session revocation。** Server 使某个 session id 失效；client 在下一次 request 看到 404。Client 必须重新 handshake。
- **Clock skew。** Client 上的 resource TTL 计算偏离 server。Client 应该把 server timestamp 视为权威。

### 什么时候绕过 Streamable HTTP

一些企业会在内部网络中把 MCP server 部署在 gRPC 或 message queue transport 后面。这不是标准做法；MCP 规范没有正式定义这些 transport。Gateway 可以对 MCP client 暴露 Streamable HTTP surface，同时在内部使用 gRPC。外部 surface 保持规范兼容；gateway 负责翻译。

## Use It

`code/main.py` 使用 `http.server`（stdlib）实现了一个最小 Streamable HTTP endpoint。它在 `/mcp` 上处理 POST、GET 和 DELETE，在第一次 response 上设置 `Mcp-Session-Id`，校验 `Origin`，并拒绝来自非 allowlisted origin 的 request。Handler 复用了 Lesson 07 notes server 的 dispatch 逻辑。

要观察的点：

- POST handler 读取 JSON-RPC body，dispatch，然后写出 JSON response（这是单 response 变体；SSE 变体在结构上类似）。
- `Origin` 检查会拒绝默认的 `http://evil.example` probe，但接受 `http://localhost`。
- Session id 是随机 128-bit hex string；server 在内存中保存每个 session 的状态。

## Ship It

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（legacy）MCP server，这个 skill 会产出一份迁移到 Streamable HTTP 的计划，覆盖 session-id continuity、Origin checks，以及向后兼容的 probe support。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize`，观察 response header 中的 `Mcp-Session-Id`。再 POST 第二个 request，回显该 header，并验证 session continuity。

2. 增加一个 GET handler，打开 SSE stream。每五秒发送一个 `notifications/progress` event。使用同一个 session id 重新 GET 来 reconnect，并确认 server 接受它。

3. 实现 `last-event-id` replay 逻辑。Reconnect 时，replay 自该 id 之后生成的所有 event。

4. 扩展 `Origin` 校验，支持 wildcard pattern（`https://*.example.com`），并确认它接受 `https://app.example.com`，但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方 registry 选择一个 legacy HTTP+SSE server（有好几个），草拟迁移方案：endpoint handling、session id generation 和 header semantics 分别要怎么变。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| stdio transport | “本地子进程” | 通过 stdin/stdout 承载 JSON-RPC，以换行分隔 |
| Streamable HTTP | “远程 transport” | 单 endpoint POST + GET + 可选 SSE，来自 2025-03-26 版规范 |
| HTTP+SSE | “Legacy” | 正在 2026 年中移除的双 endpoint 模型 |
| `Mcp-Session-Id` | “Session header” | Server 分配的随机 id，后续每个 request 都要回显 |
| `Origin` allowlist | “DNS-rebinding 防御” | 拒绝 Origin 未获批准的 request |
| Single endpoint | “一个 URL” | `/mcp` 处理所有 session 操作的 POST / GET / DELETE |
| `last-event-id` | “SSE replay” | 用来恢复断开的 stream，避免漏掉 event 的 header |
| Backwards-compat probe | “新旧检测” | Client 通过 response 形状自动选择 transport |
| Long-lived HTTP | “SSE streaming” | Server 在一个 TCP connection 上推送数分钟或数小时的 event |
| Session revocation | “强制重新 init” | Server 使 session id 失效；client 必须重新 handshake |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的权威参考。
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的规范修订版。
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers-hosted Streamable HTTP pattern。
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 不同部署形态的比较。
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移 deadline 示例。
