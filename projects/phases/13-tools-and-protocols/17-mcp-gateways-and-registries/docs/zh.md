# MCP Gateways and Registries：企业控制平面

> 企业不能允许每个开发者随意安装随机 MCP servers。Gateway 会集中处理 auth、RBAC、audit、rate limiting、caching 和 tool-poisoning detection，然后把合并后的 tool surface 暴露成一个 MCP endpoint。Official MCP Registry（Anthropic + GitHub + PulseMCP + Microsoft，namespace-verified）是 canonical upstream。本课会说明 gateway 放在哪里，走过一个最小实现，并浏览 2026 年的 vendor landscape。

**Type:** Learn
**Languages:** Python (stdlib, minimal gateway)
**Prerequisites:** Phase 13 · 15 (tool poisoning), Phase 13 · 16 (OAuth 2.1)
**Time:** ~45 minutes

## 学习目标

- 解释 MCP gateway 位于哪里（MCP clients 和多个 backend MCP servers 之间）。
- 实现 gateway 的五项职责：auth、RBAC、audit、rate limit、policy。
- 在 gateway 层强制执行 pinned-tool-hash manifest。
- 区分 Official MCP Registry 和 metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一家 Fortune 500 公司有 30 个已批准的 MCP servers、5000 名开发者、合规与审计要求，还有一个希望集中管理 policy 的安全团队。让每个开发者在自己的 IDE 里安装任意 servers，这根本不可行。

Gateway pattern：

1. Gateway 作为一个单一 Streamable HTTP endpoint 运行，开发者连接到它。
2. Gateway 持有每个 backend MCP server 的 credentials。
3. 每个开发者 request 都通过 gateway 自己的 OAuth 完成 authentication 和 scoping。
4. Gateway 把 call 路由到 backend server，并应用 policy。
5. 所有 calls 都记录到 audit log。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway——这些都在 2025-2026 年发布了 gateways 或 gateway features。

与此同时，Official MCP Registry 作为 canonical upstream 发布：curated、namespace-verified、reverse-DNS-named servers，gateway 可以从中拉取。Metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）会聚合多个来源的 servers。

## 概念

### Gateway 的五项职责

1. **Auth。** 使用 OAuth 2.1 识别开发者；映射到 user roles。
2. **RBAC。** Per-user policy：哪些 servers、哪些 tools、哪些 scopes。
3. **Audit。** 记录每次 call 的 who、what、when、result。
4. **Rate limit。** Per-user / per-tool / per-server caps，防止滥用。
5. **Policy。** 拒绝 poisoned descriptions，执行 Rule of Two，redact PII。

### Gateway as a single endpoint

对开发者来说，gateway 看起来像一个 MCP server。在内部，它会路由到 N 个 backends。Session ids（Phase 13 · 09）会在边界处被重写。

### Credential vaulting

开发者永远看不到 backend tokens。Gateway 持有这些 tokens（或代理到持有它们的 identity provider）。一个在 gateway 上拥有 `notes:read` 权限的开发者，可能会通过 gateway 自己的 backend credentials 间接访问 notes MCP server——但只能在绑定这类 transitive access 的 policy 之下访问。

### Gateway 层的 tool-hash pinning

Gateway 持有一份 approved tool descriptions 的 manifest（SHA256 hashes）。Discovery 时，它拉取每个 backend 的 `tools/list`，把 hashes 和 manifest 对比，并移除任何 description 已经变化的 tool。这就是 Phase 13 · 15 的 rug-pull defense，只是集中应用在 gateway 层。

### Policy-as-code

高级 gateways 会用 OPA/Rego、Kyverno 或 Styra 表达 policy。像 “user `alice` may call `github.open_pr` only on repos in org `acme`” 这样的规则会以声明式方式编码。简单 gateways 使用手写 Python。两种形状都有效。

### Session-aware routing

当一个用户 session 混合了多个 servers 时，gateway 会 multiplex：开发者的单一 MCP session 持有 N 个 backend sessions，每个 server 一个。来自任意 backend 的 notifications 都会通过 gateway 路由回开发者 session。

### Namespace merging

Gateways 会合并所有 backends 的 tool namespaces，通常在冲突时加 prefix。`github.open_pr`、`notes.search`。这样 routing 就不会有歧义。

### Registries

- **Official MCP Registry (`registry.modelcontextprotocol.io`)。** 在 Anthropic、GitHub、PulseMCP、Microsoft 的托管下发布。Namespace-verified（reverse-DNS：`io.github.user/server`）。预先做了基本质量过滤。
- **Glama。** Search-centric metaregistry，聚合许多来源。
- **MCPMarket。** 偏商业目录，带 vendor listings。
- **MCP.so。** Community directory；开放提交。
- **Smithery。** Package-manager-style installation flow。
- **LobeHub。** 集成在 LobeChat app 里的 UI registry。

Enterprise gateways 默认从 Official Registry 拉取，允许管理员从 metaregistries 中 curated additions，并拒绝任何 unpinned 内容。

### Reverse-DNS naming

Official Registry 要求 public servers 使用 reverse-DNS names：`io.github.alice/notes`。Namespaces 可以防止 squatting，并让 trust delegation 更清晰。

### Vendor survey, April 2026

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | Edge-hosted；OAuth integrated；free tier |
| Kong AI Gateway | K8s-native；fine-grained policy；logs to OpenTelemetry |
| IBM ContextForge | Enterprise IAM；compliance；audit export |
| TrueFoundry | DevOps-leaning；metrics-first |
| MintMCP | Developer-platform oriented |
| Envoy AI Gateway | Open-source；customizable filters |

Phase 17（production infrastructure）会更深入讲 gateway operations。

## Use It

`code/main.py` 提供一个约 150 行的 minimal gateway：通过 fake Bearer token 认证用户，持有 per-user RBAC policy，把 requests 路由到两个 backend MCP servers，把每次 call 写入 audit log，强制执行 rate limit，并拒绝任何 description hash 与 pinned manifest 不匹配的 backend tool。

值得关注的地方：

- `RBAC` dict 以 `user_id` 为 key，包含允许的 `server_tool` entries。
- `AUDIT_LOG` 是 append-only events list。
- Rate limit 使用 per-user token bucket。
- Pinned manifest 是一个 `server::tool -> hash` dict。

## Ship It

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个 enterprise MCP plan（users、backends、compliance），这个 skill 会生成一份 gateway configuration spec。

## Exercises

1. 运行 `code/main.py`。以 allowed user 发起一次 call；再以 disallowed user 发起一次；再发起一次 rate-limit-exceeded burst。验证三条 flows。

2. 添加一个 policy，在把结果返回给 client 前 redact PII。用一个简单 regex pass 处理 SSN-shaped strings；同时注明缺口（emails、phone numbers）。

3. 扩展 audit log，让它 emit OpenTelemetry GenAI spans。Phase 13 · 20 会覆盖具体 attributes。

4. 为一个 50 人开发团队设计 RBAC policy，它有五个 backends（notes、github、postgres、jira、slack）。谁对每个 backend 拿 read-only？谁拿 write？

5. 从头到尾阅读 Cloudflare enterprise MCP post。找出 Cloudflare 发布了一个这个 stdlib gateway 没有的 feature。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | “MCP proxy” | 位于 clients 和 backends 之间的集中式 server |
| Credential vaulting | “Backend tokens stay server-side” | 开发者永远看不到 upstream tokens |
| Session-aware routing | “Multi-backend session” | Gateway 为每个 developer session multiplex N 个 backend sessions |
| Tool-hash pinning | “Approved manifest” | 每个 approved tool description 的 SHA256；集中阻断 rug-pulls |
| RBAC | “Per-user policy” | 针对 tools 和 servers 的 role-based access control |
| Policy-as-code | “Declarative rules” | 在 gateway 执行的 OPA/Rego、Kyverno、Styra policies |
| Audit log | “Who, what, when” | 面向 compliance 的 append-only event log |
| Rate limit | “Per-user token bucket” | 防止滥用的 per-minute caps |
| Official MCP Registry | “Canonical upstream” | `registry.modelcontextprotocol.io`，namespace-verified |
| Reverse-DNS naming | “Registry namespace” | `io.github.user/server` convention |

## Further Reading

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — canonical upstream, namespace-verified
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — gateway pattern with OAuth and policy
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — open-source reference gateway
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — feature comparison article
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — enterprise gateway from IBM
