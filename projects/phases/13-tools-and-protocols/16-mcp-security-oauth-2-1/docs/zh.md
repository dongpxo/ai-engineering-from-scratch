# MCP 安全 II：OAuth 2.1、Resource Indicators、增量 Scope

> 远程 MCP server 需要 authorization，而不只是 authentication。2025-11-25 版规范与 OAuth 2.1 + PKCE + resource indicators（RFC 8707）+ protected-resource metadata（RFC 9728）对齐。SEP-835 增加了增量 scope consent：server 在 403 WWW-Authenticate 上要求 step-up authorization。本课把 step-up flow 实现成状态机，让你能看清每一次跳转。

**Type:** Build
**Languages:** Python (stdlib, OAuth state machine simulator)
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security I)
**Time:** ~75 minutes

## 学习目标

- 区分 resource server 和 authorization server 的职责。
- 走通受 PKCE 保护的 OAuth 2.1 authorization code flow。
- 使用 `resource`（RFC 8707）和 protected-resource metadata（RFC 9728）防止 confused-deputy attack。
- 实现 step-up authorization：server 返回带 WWW-Authenticate 的 403，请求更高 scope；client 重新提示用户授权并重试。

## 问题

早期 MCP（2025 年之前）的远程 server 常用临时 API key，甚至没有 auth。2025-11-25 版规范用完整 OAuth 2.1 profile 补上了这个缺口。

三个真实世界需求：

- **普通远程 server。** 用户安装一个远程 MCP server，用它访问自己的 Notion / GitHub / Gmail。OAuth 2.1 with PKCE 是正确形状。
- **Scope escalation。** 一个 notes server 已被授予 `notes:read`，后来某个具体动作需要 `notes:write`。不必重做完整流程，step-up（SEP-835）只请求额外 scope。
- **Confused deputy prevention。** Client 持有一个 audience 限定到 Server A 的 token。Server A 是恶意的，试图把这个 token 拿去 Server B 使用。Resource indicators（RFC 8707）会把 token 固定到预期 audience。

OAuth 2.1 不是新东西。新的部分是 MCP 的 profile：明确要求的 flow（只允许 authorization code + PKCE；不使用 implicit flow，默认不使用 client credentials）、每次 token request 都必须带 resource indicators，以及发布 protected-resource metadata，让 client 知道该去哪里授权。

## 概念

### 角色

- **Client。** MCP client（Claude Desktop、Cursor 等）。
- **Resource server。** MCP server（notes、GitHub、Postgres 或其它）。
- **Authorization server。** 签发 token。它可以和 resource server 是同一个服务，也可以是单独的 IdP（Auth0、Keycloak、Cognito）。

在 MCP profile 中，resource server 和 authorization server 可以是同一个 host，但应该用 URL 区分。

### Authorization code + PKCE

流程如下：

1. Client 生成 `code_verifier`（随机值）和 `code_challenge`（SHA256）。
2. Client 把用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意授权。Authorization server 重定向到 `redirect_uri?code=...`。
4. Client POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. Authorization server 用存储的 challenge 校验 verifier 的 hash，并签发 access token。
6. Client 使用 token：每个发往 resource server 的 request 都带 `Authorization: Bearer ...`。

PKCE 防止 authorization-code interception attack。Resource indicators 防止 token 在其它地方有效。

### Protected-resource metadata（RFC 9728）

Resource server 发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Client 从 resource server 发现 authorization server。这样可以减少配置：client 只需要 resource URL。

### Resource indicators（RFC 8707）

Token request 里的 `resource` 参数会固定 token 的预期 audience。签发出的 token 包含 `aud: "https://notes.example.com"`。另一个 MCP server 收到这个 token 时会检查 `aud` 并拒绝它。

### Scope model

Scope 是用空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 表示管理能力（谨慎使用）
- `profile:read` 表示身份信息

Scope selection 应遵循 least privilege：现在需要什么就请求什么，需要更多时再 step up。

### Step-up authorization（SEP-835）

用户授予了 `notes:read`。后来他们要求 agent 删除一条 note。Server 返回：

```text
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Client 看到 `insufficient_scope` 错误后，提示用户为额外 scope 打开 consent dialog，为它执行一个小型 OAuth flow，然后用新 token 重试 request。

### Token audience validation

每个 request 中，server 都检查 `token.aud == self.resource_url`。不匹配就是 401。这会阻止跨 server 复用 token。

### 短寿命 token 与 rotation

Access token 应该是短寿命的（默认 1 小时）。Refresh token 每次 refresh 都轮换。Client 在后台处理 silent refresh。

### No token passthrough

Sampling server（Phase 13 · 11）绝不能把 client 的 token 透传给其它服务。Sampling request 就是边界。

### Confused deputy prevention

Token 绑定到 `aud`。Client 绑定到 `client_id`。每个 request 都要同时校验两者。规范明确禁止旧的 “pass-the-token” 模式；这种模式在 MCP 之前的远程工具生态中很常见。

### Client ID discovery

每个 MCP client 会在固定 URL 发布自己的 metadata。Authorization server 可以获取 client 的 metadata document，发现 redirect URI 和联系信息。这移除了手动 client registration。

### Gateway 与 OAuth

Phase 13 · 17 展示企业 gateway 如何处理 OAuth：gateway 持有 upstream server 的凭据，发给 client 的 token 由 gateway 签发，upstream token 永远不会离开 gateway。这会翻转信任模型：用户只向 gateway authenticate 一次；gateway 处理 N 个 server authorization。

## Use It

`code/main.py` 把完整 OAuth 2.1 step-up flow 模拟成一个状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带 resource indicator 的 authorization code flow。
- Protected-resource metadata endpoint。
- 带 audience check 的 token validation。
- 遇到 `insufficient_scope` 时 step-up。

本课没有 HTTP server；状态机在内存中运行，所以你可以追踪每一次跳转。Phase 13 · 17 的 gateway 课会把它接到真实 transport 上。

## Ship It

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带 tools 的远程 MCP server，这个 skill 会设计 scope set、pinning rules 和 step-up policy。

## 练习

1. 运行 `code/main.py`。追踪两个 scope 的 step-up flow。注意 step-up 时哪些 hop 会重复。

2. 添加 refresh-token rotation：每次 refresh 都签发新的 refresh token，并使旧 token 失效。模拟被盗 refresh token 在 rotation 后被使用，并确认它失败。

3. 使用 stdlib `http.server` 把 protected-resource metadata endpoint 实现成真实 HTTP response。对齐 Lesson 09 的 `/mcp` endpoint。

4. 为 GitHub MCP server 设计 scope hierarchy：read repo、write PR、approve PR、merge PR、admin。每个层级之间使用 step-up。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 的用法和 RFC 示例不同的一个字段。（提示：它和 `scopes_supported` 有关。）

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| OAuth 2.1 | “Modern OAuth” | 整合后的 RFC，要求 PKCE，并禁止 implicit flow |
| PKCE | “Proof-of-possession” | Code verifier + challenge，用来防止 authorization-code interception |
| Resource indicator | “Token audience” | RFC 8707 的 `resource` 参数，把 token 固定到一个 server |
| Protected-resource metadata | “Discovery doc” | RFC 9728 的 `.well-known/oauth-protected-resource` |
| Step-up authorization | “Incremental consent” | SEP-835 中按需添加 scope 的 flow |
| `insufficient_scope` | “403 with WWW-Authenticate” | Server 发出的信号，要求为更大 scope 重新 consent |
| Confused deputy | “Token reuse across services” | 可信 token 持有者不恰当地转发 token 的攻击 |
| Short-lived token | “Access token TTL” | 很快过期的 bearer；通过 refresh token 续期 |
| Scope hierarchy | “Least privilege stack” | 分层 scope set，各层之间通过 step-up 过渡 |
| Client ID metadata | “Client discovery doc” | Client 发布自身 OAuth metadata 的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth profile 的权威参考。
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更 walkthrough。
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning RFC。
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — discovery-document RFC。
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — practical step-up-flow walk-through。
