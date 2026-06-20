# 生产环境 MCP Auth：注册、JWKS 刷新与 Audience-Pinned Tokens

> Lesson 16 在内存里搭起了 OAuth 2.1 状态机。到 2026 年，你交付给真实组织的每个 MCP server 都会站在生产级 auth 后面：client enrollment 要能扩展到无界 client 数量（优先使用 Client ID Metadata Documents，dynamic client registration 作为向后兼容 fallback）、authorization-server metadata discovery（RFC 8414 *或* OpenID Connect Discovery）、不会让凌晨 3 点 token validation 崩掉的 JWKS cache refresh，以及拒绝跨 resource replay 的 audience-pinned tokens。本课用三个角色建模完整表面：authorization server、resource server（MCP server）和 client，让你能追踪从 discovery 到已验证 tool call 的每一跳。
>
> **Spec note (2025-11-25):** 2025 年 11 月版 MCP authorization spec 把 Dynamic Client Registration 从 `SHOULD` 降级为 `MAY`，并把 **Client ID Metadata Documents (CIMD)** 设为推荐的默认 enrollment 机制。本课会按 spec 的优先级顺序讲两者；代码保留 DCR 用于 walk-through，因为它能在单进程里完全自包含。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## 学习目标

- 通过 RFC 8414 metadata 发现 authorization server，并验证它的合约。
- 实现 RFC 7591 dynamic client registration，让 MCP clients 无需管理员介入即可 enroll。
- 按计划缓存并刷新 JWKS keys，让 signature verification 能熬过 key roll-over。
- 使用 RFC 8707 resource indicators 把 tokens 绑定到单个 MCP resource，并拒绝 confused-deputy reuse。
- 清晰分离三个角色：authorization server、resource server、client，让每个角色只执行属于自己的检查。
- 读取 IdP capability matrix，并在 IdP 无法满足 MCP auth profile 时拒绝部署。

## 问题

Lesson 16 的模拟器在内存中运行 OAuth 2.1。生产环境有三个内存模拟器看不到的运维缺口。

第一个缺口是 enrollment。真实组织会运行数百个 MCP servers 和数千个 MCP clients。运维人员不会把每个 Cursor 用户都手工注册成 OAuth client。2025-11-25 spec 给 clients 一个解决优先级：如果已有 pre-registered `client_id` 就使用它；否则使用 **Client ID Metadata Document**（client 用一个自己控制的 HTTPS URL 标识自己，authorization server *拉取* metadata）；否则 fallback 到 **RFC 7591 dynamic client registration**（client *推送* 一个 `POST /register` 并当场收到 `client_id`）；否则提示用户。CIMD 是推荐默认值，因为它完全移除了 per-server registration，同时保留 DNS-rooted trust model；DCR 被保留用于向后兼容。两者都从 authorization server metadata 中发现入口点：CIMD 使用 `client_id_metadata_document_supported`，DCR 使用 `registration_endpoint`。

第二个缺口是 key rotation。JWT validation 依赖 authorization server 的 signing keys，这些 keys 会以 JSON Web Key Set (JWKS) 发布。Authorization server 会按计划轮换这些 keys（通常每小时，有时在 incident response 下更快）。如果 MCP server 只在启动时获取一次 JWKS，那么它会一直正常校验，直到 rotation window 到来——随后每个请求都会失败，直到重启。生产环境会把 JWKS 接成带缓存的值，并配一个 refresh job，在旧 keys 过期前覆盖缓存；还要在 cache miss 时做 fallback fetch，以处理由比缓存更新的 key 签名的 token 到达的情况。

第三个缺口是 audience binding。Lesson 16 引入了 RFC 8707 resource indicators。在生产环境中，这个 indicator 会变成每个请求上的硬性 claim check。MCP server 会把 `token.aud` 和自己的 canonical resource URL 比较，不匹配则以 HTTP 401 拒绝。这是抵御跨 resource replay 的唯一防线：上游 MCP server（或持有某个 server 专用 token 的恶意 client）不能把这个 token 重放到同一 trust mesh 里的另一个 server。

本课会把每个缺口映射到具体表面。Metadata document 是一个 HTTP endpoint。JWKS cache refresh 是一个 scheduled job 加 key-value cache。JWT validation 是 resource server 在 dispatch 任意 tool 前运行的 routine。保持三个角色分离，每个角色只执行自己拥有的检查：authorization server 负责签发和轮换 keys，resource server 负责缓存和验证，client 负责 discovery 和 enrollment。

## 概念

### RFC 8414：OAuth Authorization Server Metadata

位于 `/.well-known/oauth-authorization-server` 的 document 会描述 client 需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

拿到 MCP resource URL 的 client 会链式 discovery：RFC 9728 中的 `oauth-protected-resource`（resource server 的 document）命名 issuer，然后 `oauth-authorization-server`（本 RFC）命名每个 endpoint。Client 永远不硬编码 authorization URL。

在把某个 IdP 信任为 MCP IdP 前，你要验证的合约：

- `code_challenge_methods_supported` 包含 `S256`（按 RFC 7636 使用 PKCE）。Spec 很明确：如果这个字段**缺失**，authorization server 不支持 PKCE，client **MUST** 拒绝继续。
- `grant_types_supported` 包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 至少公布一个 enrollment path：`client_id_metadata_document_supported: true`（CIMD，首选）**或** `registration_endpoint`（RFC 7591 DCR，fallback）。任意一个都满足合约；你不再硬性要求 DCR。
- 对于 OAuth 2.1，`response_types_supported` 恰好是 `["code"]`。

如果缺少 `S256`，MCP server 会拒绝针对这个 IdP 部署——PKCE 没有 degraded mode。如果**两个** enrollment path 都没有公布，并且你也没有 pre-registered `client_id`，你也无法 enroll；这是 deployment manifest 错了，不是代码错了。

### RFC 9728（回顾）：Protected Resource Metadata

Lesson 16 已经讲过 RFC 9728。生产环境里的增量是：这个 document 是 client 查找“这个 MCP server 信任哪些 authorization servers”的唯一位置。一个 MCP server 可以接受来自多个 IdPs 的 tokens（一个给员工，一个给合作方）。RFC 9728 声明这组 IdPs；RFC 8414 记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（推荐默认值）

CIMD 把 registration 从 *push* 反转为 *pull*。Client 不再要求 authorization server mint 一个 `client_id`，而是使用它控制的 HTTPS URL **作为** `client_id`。这个 URL 会解析到一个 JSON metadata document；authorization server 会在 OAuth flow 中按需获取它。信任根基在 DNS：如果 server operator 信任 `app.example.com`，它就信任由 `https://app.example.com/client.json` 提供的 client。没有 registration round-trip，没有会耗尽的 `client_id` namespace，也没有需要同步的 per-server state。

Client 托管的 metadata document：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Document 中的 `client_id` 值 **MUST** 等于它被服务的 URL（authorization server 会验证这一点；不匹配会被拒绝）。Authorization server 会在自己的 RFC 8414 metadata 中用 `client_id_metadata_document_supported: true` 宣告支持。

Spec 对两个安全事实说得很直白：

- **SSRF。** Authorization server 会获取攻击者提供的 URL。它必须防御 server-side request forgery（不能 fetch internal/admin endpoints）。
- **localhost impersonation。** CIMD 本身不能阻止本地攻击者声称一个合法 client 的 metadata URL，并绑定任意 `localhost` redirect。Authorization server **MUST** 在 consent 阶段清楚展示 redirect URI hostname，并且 **SHOULD** 对 `localhost`-only redirects 发出警告。

因为 CIMD 不需要 server-side state，所以不需要像 DCR 那样搭 registrar。Client side 是只读的：从一个静态 HTTPS endpoint 提供 metadata document，让 authorization server 去拉取。

### RFC 7591：Dynamic Client Registration（fallback / 向后兼容）

DCR 现在是 `MAY`，保留给 2025-11-25 之前的部署，以及尚未支持 CIMD 的 IdPs 做向后兼容。没有它（并且没有 CIMD 或 pre-registration）时，每个 MCP client（Cursor、Claude Desktop、自定义 agent）都需要和 IdP admin 进行 out-of-band exchange。有了 DCR，client 会提交：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server 返回 `client_id` 和用于后续 updates 的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP clients 的正确默认值。它们只拿到 `client_id`，没有可被 exfiltrate 的 `client_secret`。PKCE 提供 public clients 需要的 proof-of-possession。

三个生产陷阱：

- Registration endpoint 必须按 source IP rate-limit。否则恶意 actor 可以脚本化创建数百万个假 registrations，耗尽 `client_id` namespace。在 registrar 处理请求前先跑 rate-limit check。
- 某些 enterprise IdPs 要求 `software_statement`（一个为 client 背书的 signed JWT）。本课 mock 跳过了它；生产环境要接一个 verification step，拒绝除了 localhost redirect URIs 之外的所有 unsigned registrations。
- `registration_access_token` 必须以 hash 存储，不能明文存储。这个 token 被盗，意味着攻击者可以重写 client 的 redirect URIs。

### RFC 8707（回顾）：Resource Indicators

Lesson 16 已经建立了形状。生产规则是：每个 token request 都包含 `resource=<canonical-mcp-url>`，并且 MCP server 在每次调用时验证 `token.aud` 是否匹配自己的 resource URL。Canonical URI 是 server 的*最具体*标识符：使用小写 scheme 和 host，不带 fragment，通常不带 trailing slash。Path component **不会**按规则剥离——当它需要用来标识单个 MCP server 时，spec 会保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效 canonical URIs。每个 server 选一个，并把 `aud` 精确 pin 到它。（本课 mock 为了简洁使用 `https://notes.example.com` 这样的 bare-host audiences；如果一个 deployment 在同一 origin 下共托管多个 MCP servers，就要通过 path 区分它们。）

### RFC 7636（回顾）：PKCE

PKCE 在 OAuth 2.1 中是必需的。本课的 authorization-code flow 始终携带 `code_challenge` 和 `code_verifier`。如果 token request 没有 verifier，或 verifier hash 后不能匹配存储的 challenge，server 会拒绝。

### MCP Spec 2025-11-25 Auth Profile

MCP spec（2025-11-25）很精确地规定了 MCP server 的 authorization layer 必须做什么：

- 实现 RFC 9728 protected-resource metadata，并通过 401 上的 `WWW-Authenticate: Bearer resource_metadata="..."` header **或** well-known URI `/.well-known/oauth-protected-resource` 提供它的位置（SEP-985 让 header 变成 optional，并提供 well-known fallback）。Metadata 的 `authorization_servers` 字段 **MUST** 命名至少一个 server。
- 在**每个**请求上只通过 `Authorization: Bearer ...` 接受 tokens——永远不要放在 query string 中，也不要只在 session start 时验证。
- 每个请求都验证 `aud`、`iss`、`exp` 和 required scopes。Server **MUST** 验证 token 是专门签发给它的（audience）；缺失或不匹配的 `aud` 会被拒绝，永远不能当成 wildcard。
- 在 401/403 上返回 `WWW-Authenticate: Bearer`，其中带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（metadata document 的 URL，*不是*裸 resource），并在 `insufficient_scope`（403）时带 `scope="..."`。注意：参数名是 `resource_metadata`，它是 discovery pointer；challenge 中没有 `resource` 参数。
- Authorization-server discovery 接受 RFC 8414 OAuth metadata **或** OpenID Connect Discovery 1.0；clients 必须按优先级顺序尝试两个 well-known suffixes。
- Client（不是 server）防御 **mix-up attacks**：它在 redirect 前记录预期 `issuer`，并在兑换 code 前验证 `iss` authorization-response parameter（RFC 9207）。PKCE 本身不能阻止 mix-up，因为 client 会把自己的 `code_verifier` 交给被引导到的 token endpoint。

OAuth 2.1 draft 是底层 substrate；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是表面；MCP spec 是 profile。

### IdP capability matrix

不是每个 IdP 都支持完整 MCP profile。下表记录的是截至 2025-11-25 spec 的事实性 capability statements。它是一个 *deployment gate*，不是推荐清单。

CIMD 随 2025-11-25 spec 发布，底层 OAuth draft 也只是在 2025 年 10 月被 adopted，因此 vendor support 仍在到来——把下面的 “CIMD” 理解为“今天大概处于什么状态，请在你的 tenant 中验证”，不是永久判断。

| IdP category | AS metadata (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | Notes |
|---|---|---|---|---|---|---|
| Self-hosted (Keycloak) | yes | emerging | yes | yes (since 24.x) | yes | 本课 MCP profile 的 reference IdP；DCR path 可以 end-to-end 跑通，CIMD 正跟进新 spec。 |
| Enterprise SSO (Microsoft Entra ID) | yes | emerging | yes (premium tiers) | yes | yes | DCR 可用性因 tenant tier 而异；部署前要在目标 tenant 中验证。 |
| Enterprise SSO (Okta) | yes | emerging | yes (Okta CIC / Auth0) | yes | yes | DCR 可在 Auth0（现在是 Okta CIC）使用；classic Okta orgs 需要管理员 pre-registration。 |
| Social login IdPs (generic) | varies | no | rarely | rarely | yes | 大多数 social IdPs 把 clients 当成静态 partners；没有 self-service enrollment。只把它们当 identity source，在上层叠自己的 MCP-aware authorization server。 |
| Custom / homegrown | depends | depends | depends | depends | depends | 如果你自己交付，就交付完整 profile，并优先使用 CIMD。跳过 PKCE 或 audience binding 会破坏 MCP auth contract。 |

Deployment manifest 的拒绝规则：如果选中的 IdP 没有在 `code_challenge_methods_supported` 中列出 `S256`，MCP server 拒绝启动——PKCE 没有 degraded mode。Enrollment 是更软的 gate：你需要*一个*可用路径（pre-registered `client_id`、`client_id_metadata_document_supported: true`，或 `registration_endpoint`）。DCR 单独缺失不再触发拒绝，因为 CIMD 或 pre-registration 可以覆盖它。

### JWKS refresh pattern（AS rotate，resource server refresh）

把两个动词分开，因为混淆它们是真实生产 bug：

- **Rotate** 是 *authorization server* 做的事：mint 一个新的 signing key，把它发布到 JWKS，稍后 retire 旧 key。Resource server 不参与，也不能做这件事——它不持有 IdP 的 private keys。
- **Refresh** 是 *resource server* 做的事：重新 `GET` 已发布的 JWKS 到自己的 cache。这是 resource server 唯一会做的 JWKS 动作。

生产失败模式是 stale cache。解决方案是 scheduled refresh job 加 key-value cache。Resource server 跑一个 job（cron、timer，或你的 runtime 提供的任意机制），按固定间隔 fetch `<issuer>/.well-known/jwks.json`，并覆盖 `cache[issuer] = {keys, fetched_at}`。Validator 从这个 cache 读取。如果某个 token 的 `kid` 在 cache 中缺失，它会触发**一次**同步 refresh 作为 fallback，然后再检查一次。这同时处理两种情况：scheduled refresh，以及 key-overlap windows 中由全新 key 签发的 token 在下一次 scheduled refresh 前到达。

Fallback **必须是 re-fetch，绝不能是 rotate**。如果你把 cache-miss path 接成 rotate-and-mint，会破坏两件事：(1) mint 出来的新 key 的 `kid` *仍然*不匹配 token，所以 lookup 依旧失败；(2) 攻击者可以用随机 `kid` 的 tokens 迫使系统创建无界数量的 keys，形成自伤式 DoS。Re-fetch 是幂等的，所以一个伪造 `kid` 最多浪费一次 fetch。

Cache shape：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两个 keys 是 steady state。Authorization servers 通过先引入下一个 key（`k_2026_04`），再 retire 前一个 key（`k_2026_03`）来 rotate，因此旧 key 签发的 tokens 在过期前仍然有效。Cache 持有 union；validator 按 `kid` 选择。

### Validation routine

MCP server 会在 dispatch 任意 tool 前运行 validation。`code/main.py` 使用的形状如下：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 会 decode JWT，从 JWKS cache 中解析 signing key（miss 时刷新一次），验证 signature，然后检查 `iss` 是否在 allow-list 中、`aud` 是否匹配这个 server 的 canonical resource、`exp`，以及 required scope——遇到第一个失败就返回 `WWW-Authenticate` challenge。把它保留为 resource server 上的单一 routine，意味着每个 entry point（每个 tool call、每个 transport）都会经过相同检查；没有任何路径能绕过 validation 直接到达 tool。

### Audience-replay walkthrough（access-token privilege restriction）

Server A（`notes.example.com`）和 Server B（`tasks.example.com`）都注册到同一个 authorization server。Server A 被攻破。攻击者拿到用户的 notes token，并把它 replay 到 Server B。

Server B 的 validator：

1. Decode JWT，按 `kid` fetch JWKS，验证 signature。
2. 检查 `iss` 是否在自己的 protected-resource metadata 的 `authorization_servers` 中。（通过——同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，并带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

Audience claim 是协议层抵御这个攻击的唯一防线。为了性能跳过它，是最常见的生产错误；validator 必须在每个请求上运行，而不是只在 session start 运行。Spec 把这叫做 **access-token privilege restriction**：MCP server `MUST` reject 任何没有在 audience 中命名它的 token。

> **Naming note.** Spec 把 *confused deputy* 这个术语保留给一个相关但不同的问题：MCP server 作为 OAuth **proxy** 访问第三方 API，使用静态 client ID，在没有获得 per-client user consent 的情况下转发 token。Audience binding 修复的是上面的 replay；confused-deputy 的修复是 per-client consent **加上**永远不要把 inbound token 传给 upstream APIs（MCP server `MUST` 获取自己单独的 upstream token）。

### Mix-up attacks（server 无法提供的 client-side defense）

一个 client 在生命周期里会和许多 authorization servers 通信。恶意 AS 可以试图让 client 把诚实 AS 的 authorization code 兑换到攻击者的 token endpoint。Audience binding 在这里无能为力——攻击发生在任何 token 存在之前。防御位于 client 中（RFC 9207）：

1. Redirect 前，client 从已验证的 AS metadata 中记录预期 `issuer`。
2. 在 authorization response 上，client 把返回的 `iss` parameter 与记录的 issuer 比较（简单字符串比较，不做 normalization），然后才把 code 发给任何地方。
3. 不匹配（或当 AS 宣告 `authorization_response_iss_parameter_supported` 时 `iss` 缺失）→ reject，并且连 `error` fields 都不要展示。

PKCE 本身不能阻止 mix-up，因为 client 会把自己的 `code_verifier` 交给被引导到的任意 token endpoint。这就是为什么 spec 会把 issuer 与 PKCE verifier 和 `state` 一起按请求记录。

### Failure modes

- **Stale JWKS。** AS rotate key 后，validator 拒绝有效 tokens。修复方式是上面的 cron-refresh + cache-miss-refetch pattern。永远不要缓存 JWKS 却不配 refresh job。
- **Rotate-as-fall-back。** 把 cache-miss path 接到 rotate-and-mint 而不是 re-fetch 是真实 bug：它永远产生不了缺失的 `kid`，还会把攻击者控制的 `kid` values 变成 key-creation DoS。Fallback 必须是幂等的 `refresh-jwks`。
- **Missing `aud` claim。** 某些 IdPs 默认会省略 `aud`，除非 token request 中出现 `resource`。Validator 必须拒绝缺失 `aud` 的 tokens，不能把缺失当 wildcard。
- **Mix-up via missing `iss` check。** 不把 RFC 9207 `iss` authorization-response parameter 和 redirect 前记录的 issuer 做校验的 client，可能会被引导到攻击者 token endpoint 兑换诚实 AS 的 code。这是 client-side failure；resource server 无法补偿。
- **Scope upgrade race。** 同一个用户的两个并发 step-up flows 都可能成功，并产生两个 scopes 不同的 access tokens。Validator 必须使用请求中呈现的 token，而不是查询“这个用户当前的 scope”——后者会产生 TOCTOU window。
- **Registration token theft。** 泄漏的 `registration_access_token` 会让攻击者重写 redirect URIs。把这些 tokens hash-at-rest；每次 update 都要求 client 出示明文；怀疑泄漏时 rotate。
- **`iss` not pinned。** 接受任意 `iss` 的 validator 会让攻击者搭一个自己的 authorization server，给目标 audience 注册 client 并签发 tokens。Protected-resource metadata 的 `authorization_servers` list 就是 allow-list；必须执行它。

## Use It

`code/main.py` 用 stdlib Python 和三个角色走完整生产流程：`AuthorizationServer`、`ResourceServer` 和 `Client`。流程如下：

1. Authorization server 在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 metadata。
2. MCP client 调用 metadata endpoint，检查 enrollment options（CIMD 的 `client_id_metadata_document_supported`，DCR 的 `registration_endpoint`）和 `S256` PKCE support。
3. Walk-through 走 DCR fallback path：client 向 `/register` post（RFC 7591）并收到 `client_id`。（CIMD client 会改为呈现自己的 HTTPS `client_id` URL，并跳过这一步。）
4. MCP client 运行带 PKCE 保护的 authorization code flow（RFC 7636），并带上 `resource` indicator（RFC 8707）。
5. MCP client 使用 `Authorization: Bearer ...` 调用 MCP server 上的 tool。
6. MCP server 运行 `validate`，从 JWKS cache 中解析 signing key。
7. IdP rotate 一个 key；scheduled refresh 会把 JWKS 重新拉进 cache。
8. 下一次调用无需重启，就能用 refreshed keys 通过 validation；overlap window 中，旧 token 仍然可以验证。
9. 针对不同 MCP resource 的 audience-replay attempt 会得到 401，带 `audience mismatch` 和 `resource_metadata` pointer。

这里的 JWT 使用 HS256 和 shared secret（这样课程只需 stdlib 就能运行）。生产环境使用 RS256 或 EdDSA，并配合上面的 JWKS pattern；validation 逻辑除此之外相同。因为 IdP 和 resource server 位于同一进程，`refresh_jwks` 会直接读取 authorization server 的 key list；在线上它是一次到 `jwks_uri` 的 HTTP `GET`。

## Ship It

本课产出 `outputs/skill-mcp-auth.md`。给定一个 MCP server config 和一组 IdP capabilities，这个 skill 会输出要搭建的 auth surface：protected-resource metadata、要使用的 enrollment path（CIMD、pre-registration 或 DCR fallback）、JWKS refresh schedule、scope mapping，以及当 IdP 不支持完整 RFC profile 时要应用的 refusal rules。

## Exercises

1. 运行 `code/main.py`。追踪流程。注意 IdP 如何在 step 6 rotate 一个 key，scheduled `refresh_jwks` 如何重新拉取 published set，以及旧 token（overlap window）和新 token 如何都能无需重启通过 validation。

2. 向 protected-resource metadata 的 `authorization_servers` list 添加一个新 IdP。签发一个由新 IdP 签名的 token，并确认 validator 接受它。签发一个由未列出的 IdP 签名的 token，并确认 validator 以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 给 `register_client` 添加一个 rate-limit check，并让它在 registrar 接受请求前运行。使用一个按 source IP keyed 的小 dict，保存 per-source-IP token bucket。

4. 阅读 RFC 7591，找出本课 `/register` handler 没有校验的两个字段。加上校验。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 添加一个 Client ID Metadata Document path。提供一个 `client.json`，其 `client_id` 等于它自己的 URL，并让 authorization server fetch 和 verify 它（如果 `client_id` ≠ URL 就拒绝）。确认 CIMD client 可以在不调用 `register_client` 的情况下 enroll。

6. 证明 DoS 修复。向 validator 发送一个带随机 `kid` 的 token，确认 `refresh_jwks` 最多运行一次，并且 authorization server 的 key count 不会增长。然后故意把 fallback 重新接成 rotate-and-mint，观察 key count 如何随每个 bogus token 增长——之后恢复 re-fetch。

7. 实现 mix-up section 中的 client-side RFC 9207 `iss` check：在 authorization request 前记录预期 issuer，然后拒绝 `iss` 不匹配的 authorization response。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ASM | “OAuth metadata document” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | “Client metadata URL” | Client ID Metadata Document：用作 `client_id` 的 HTTPS URL；AS 会拉取其中的 JSON。自 2025-11-25 起是推荐默认值 |
| DCR | “Self-service client registration” | RFC 7591 `POST /register` flow；在 2025-11-25 中降级为 `MAY` fallback |
| JWKS | “用于 JWT validation 的 public keys” | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| Rotate vs refresh | “更新 keys” | *Rotate* = AS mint/retire signing keys；*refresh* = resource server 重新拉取已发布集合。Resource servers 永远只 refresh |
| Resource indicator | “Audience parameter” | RFC 8707 `resource` parameter，把 token pin 到一个 server |
| `aud` claim | “Audience” | Validator 会和 canonical resource URL 比较的 JWT claim |
| Audience replay | “Token replay” | 发给 Server A 的 token 被呈现给 Server B；通过 audience validation 防御（spec: access-token privilege restriction） |
| Confused deputy | “Proxy token misuse” | 使用静态 client ID 的 MCP proxy 在没有 per-client consent 的情况下转发 token；和 audience replay 不同 |
| Mix-up attack | “Wrong token endpoint” | Client 被引导到攻击者 endpoint 兑换诚实 AS 的 code；通过 client-side RFC 9207 `iss` 防御 |
| `iss` allow-list | “Trusted authorization servers” | Protected-resource metadata 的 `authorization_servers` 中命名的集合 |
| `resource_metadata` | “去哪里找 PRM doc” | 401/403 上命名 RFC 9728 metadata URL 的 `WWW-Authenticate` 参数 |
| Public client | “Native or browser client” | 没有 `client_secret` 的 OAuth client；PKCE 负责补偿 |
| `WWW-Authenticate` | “401/403 response header” | 携带 `Bearer error=...` directives，驱动 client recovery |

## Further Reading

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP auth profile
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 的变化（CIMD、XAA、DCR demotion）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — 选择 CIMD 而非 DCR 的 rationale
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（fallback path）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — public-client proof-of-possession
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — resource server discovery
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防御 mix-up attacks 的 `iss` parameter
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — consolidated OAuth substrate
