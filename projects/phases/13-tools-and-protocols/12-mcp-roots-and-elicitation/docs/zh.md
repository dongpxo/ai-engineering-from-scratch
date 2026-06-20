# Roots 与 Elicitation：范围控制与中途用户输入

> 硬编码路径会在用户打开另一个项目时立刻崩掉。预填的 tool arguments 会在用户描述不足时失效。Roots 把 server 限定在一组由用户控制的 URIs 内；elicitation 则能在 tool-call 运行中途暂停，通过表单或 URL 向用户索取 structured input。两个 client primitives，修复两个常见 MCP failure modes。SEP-1036（URL-mode elicitation，2025-11-25）到 2026 年上半年结束前仍处于 experimental 状态；依赖它之前先检查 SDK versions。

**Type:** Build
**Languages:** Python (stdlib, roots + elicitation demo)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 学习目标

- 声明 `roots`，并响应 `notifications/roots/list_changed`。
- 把 server file operations 限制在 declared root set 内部的 URIs。
- 使用 `elicitation/create`，在 tool-call 中途向用户请求 confirmation 或 structured input。
- 在 form-mode 和 URL-mode elicitation 之间做选择（后者是 experimental；已标注 drift-risk）。

## 问题

一个 notes MCP server 在生产中会撞到两个具体失败。

**Broken path assumption。** Server 针对 `~/notes` 编写。另一台机器上的用户把 notes 放在 `~/Documents/Notes`，于是 tool call 会静默失败（找不到文件），更糟的是写到了错误的位置。

**Missing argument the user would know。** 用户说：“delete the old TPS report note”。模型调用 `notes_delete(title: "TPS report")`，但有三条匹配 notes，分别来自 2023、2024 和 2025。Tool 不能猜。返回 “ambiguous” 很烦；对三条全部执行则是灾难。

Roots 解决第一个问题：client 在 `initialize` 时声明 server 可以触碰的 URI 集合。Elicitation 解决第二个问题：server 暂停 tool call，并发送 `elicitation/create`，要求用户选择其中一个。

## 概念

### Roots

Client 在 `initialize` 时声明 root list：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

Server 随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Servers 必须把 roots 视为边界：任何超出 root set 的文件读写都要被拒绝。这不是由 client 强制执行的（server 仍然是用户信任的一段代码），但符合 spec 的 servers 会遵守它。

当用户添加或移除 root 时，client 会发送 `notifications/roots/list_changed`。Server 重新调用 `roots/list`，并更新自己的边界。

### 为什么 roots 是 client primitive

Roots 由 client 声明，因为它们代表用户的 consent model。用户告诉 Claude Desktop：“给这个 notes server 访问这两个目录的权限。”Server 不能扩大这个 scope。

### Elicitation：默认的 form mode

`elicitation/create` 接收一个 form schema 和一段 natural-language prompt：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

Client 渲染一个表单，收集用户答案，然后返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

可能的 action 有三种：`accept`（用户填写了）、`decline`（用户关闭了）、`cancel`（用户中止整个 tool call）。

Form schemas 是 flat 的，v1 不支持 nested objects。SDKs 通常会拒绝比单层更复杂的 schema。

### Elicitation：URL mode（SEP-1036，experimental）

这是 2025-11-25 新增的模式。Server 不发送 schema，而是发送 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

Client 在 browser 中打开 URL，等待完成，并在用户回来时返回。它适合 OAuth flows、payment authorization 和 document signing 等 form 不够表达的场景。

Drift-risk note：SEP-1036 response shape 仍在稳定中；有些 SDKs 返回 callback URL，有些返回 completion token。在生产中使用 URL mode 前，先阅读你所用 SDK 的 release notes。

### 什么时候 elicitation 是正确工具

- Destructive actions 之前的用户 confirmation（destructive hint + elicitation）。
- Disambiguation（从 N 个 matches 中选一个）。
- First-run setup（API keys、directories、preferences）。
- OAuth-style flows（URL mode）。

### 什么时候 elicitation 是错误工具

- 填充那些模型本可以用普通对话询问的 required arguments。使用 normal re-prompt，而不是 elicitation dialog。
- 高频调用。Elicitation 会打断对话；不要在 loop 中触发它。
- Server 可以事后 validate 的任何内容。Validate，返回 error，让模型用文本询问用户。

### Human-in-the-loop bridge

Elicitation 加上 sampling，共同构成 MCP 的 “human-in-the-loop” model。Server 的 agent loop 可以为用户输入（elicitation）暂停，也可以为模型推理（sampling）暂停。Phase 13 · 11 已覆盖 sampling；本课覆盖 elicitation。把二者结合起来，就能获得完整的 mid-loop control。

## Use It

`code/main.py` 对 notes server 做了如下扩展：

- `roots/list` response，server 会在 root-list-changed notifications 后重新查询它。
- `notes_delete` tool，当多个 notes 匹配时使用 `elicitation/create` 做 disambiguation。
- `notes_setup` tool，使用 URL-mode elicitation 打开 first-run config page（模拟）。
- Boundary check，拒绝对 declared roots 之外 URI 的操作。

Demo 会运行三个 scenarios：happy path（单个 match）、disambiguation（三个 matches，触发 elicitation）、out-of-root-write（被拒绝）。

## Ship It

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户 confirmation 或 disambiguation 的 tool，这个 skill 会设计 elicitation form schema 和 message template。

## 练习

1. 运行 `code/main.py`。触发 disambiguation path；确认模拟用户答案被路由回 tool。

2. 添加一个新的 `notes_archive` tool，让它每次都要求 elicitation confirmation（destructive hint）。检查 UX：这和模型用文本重新询问相比如何？

3. 为 first-run OAuth flow 实现 URL-mode elicitation。记录 drift risk，并添加 SDK-version guard。

4. 扩展 `roots/list` handling：notification 到达时，server 应该 atomic re-read，并重新扫描那些可能已经越界的 open file handles。

5. 阅读 GitHub 上的 SEP-1036 issue discussion thread。找出一个会影响 servers 如何处理 URL-mode callbacks 的 open question。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Root | “Consent boundary” | Client 允许 server 触碰的 URI |
| `roots/list` | “Server asks for scope” | Client 返回当前 root set |
| `notifications/roots/list_changed` | “User changed scope” | Client 表示 root set 已发生变化 |
| Elicitation | “Ask the user mid-call” | Server 发起的 structured user input 请求 |
| `elicitation/create` | “The method” | 用于 elicitation requests 的 JSON-RPC method |
| Form mode | “Schema-driven form” | 被 client UI 渲染为表单的 flat JSON Schema |
| URL mode | “Browser redirect” | SEP-1036 experimental；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | “User response outcomes” | Server 需要处理的三个 branches |
| Disambiguation | “Pick one” | Tool 有 N 个 candidates 时的常见 elicitation use case |
| Flat form | “Top-level properties only” | Elicitation schemas 不能嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — canonical roots reference
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — canonical elicitation reference
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 additions walk-through
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL-mode elicitation proposal（experimental，drift-risk）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
