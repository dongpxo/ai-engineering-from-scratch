# MCP Apps：通过 `ui://` 暴露交互式 UI Resources

> 纯文本工具输出会限制 agents 能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）让工具可以返回 sandboxed interactive HTML，并在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中 inline 渲染。Dashboard、form、map、3D scene，全都通过同一个扩展完成。本课会走读 `ui://` resource scheme、`text/html;profile=mcp-app` MIME、iframe-sandbox postMessage protocol，以及允许 server 渲染 HTML 所带来的安全面。

**Type:** Build
**Languages:** Python (stdlib, UI resource emitter), HTML (sample app)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## 学习目标

- 从 tool call 返回一个 `ui://` resource，并设置正确的 MIME 和 metadata。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现 iframe sandbox postMessage JSON-RPC，用于 UI-to-host communication。
- 应用 CSP 和 permissions-policy defaults，防御 UI-originated attacks。

## 问题

一个 2025 年风格的 `visualize_timeline` 工具可以返回 “Here are 14 notes organized chronologically: ...”。那只是一段文字。用户真正想要的是可交互的 timeline。MCP Apps 之前，选项只有：client-specific widget APIs（Claude artifacts、OpenAI Custom GPT HTML），或者完全没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了这个合约。Tool result 包含一个 `resource`，它的 URI 是 `ui://...`，MIME 是 `text/html;profile=mcp-app`。Host 会在 sandboxed iframe 中渲染它，并应用受限 CSP；除非显式授权，否则没有网络访问。Iframe 内的 UI 通过一个很小的 postMessage JSON-RPC dialect 向 host 发送消息。

每个兼容 client（Claude Desktop、ChatGPT、Goose、VS Code）都会以同样方式渲染同一个 `ui://` resource。一个 server，一个 HTML bundle，通用 UI。

## 概念

### `ui://` resource scheme

一个工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

随后 host 会对 `ui://notes/timeline` URI 调用 `resources/read`，并得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox

Host 会把 HTML 渲染到一个 sandboxed `<iframe>` 中，并带有：

- `sandbox="allow-scripts allow-same-origin"`（或根据 server declaration 更严格）。
- 通过 response headers 应用 server-declared CSP。
- 没有 host origin 的 cookies，也没有 host origin 的 localStorage。
- 网络访问被限制在 CSP 的 `connectSrc` 内。

### postMessage protocol

Iframe 通过 `window.postMessage` 和 host 通信。它是一个很小的 JSON-RPC 2.0 dialect：

始终把 `targetOrigin` pin 到 peer 的精确 origin；接收侧在处理任何 payload 前，都要用 allowlist 校验 `event.origin`。这个通道的任意一侧都不要使用 `"*"`——body 中携带 tool calls 和 resource reads。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可调用的 host-side methods：

- `host.callTool(name, arguments)` — 调用一个 server tool。
- `host.readResource(uri)` — 读取一个 MCP resource。
- `host.getPrompt(name, arguments)` — 获取一个 prompt template。
- `host.close()` — 关闭 UI。

每次调用仍然会经过 MCP protocol，并继承 server 的 permissions。

### Permissions

`_meta.ui.permissions` list 用来请求额外能力：

- `camera` — 访问用户摄像头（用于 scan-a-document UIs）。
- `microphone` — 语音输入。
- `geolocation` — 位置。
- `network:*` — 比单独的 `connectSrc` 更宽的网络访问。

每个 permission 都是用户在 UI 渲染前会看到的一次 prompt。

### 安全风险

Iframe 中的 HTML 仍然是 HTML。新的攻击面包括：

- **Prompt-injection via UI。** 恶意 server UI 可以显示看起来像 system message 的文字来欺骗用户。Host rendering 应该明显区分 server UI 和 host UI。
- **Exfiltration via `connectSrc`。** 如果 CSP 允许 `connect-src: *`，UI 可以把数据发送到任何地方。默认值应该严格。
- **Clickjacking。** UI 覆盖 host chrome。Hosts 必须防止 z-index manipulation，并强制 opacity rules。
- **Steal focus。** UI 抢走 keyboard focus，并捕获下一条消息。Hosts 必须拦截。

Phase 13 · 15 会在 MCP security 中深入覆盖这些问题；本课先引入它们。

### `ui/initialize` handshake

Iframe 加载后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

Host 会返回 capabilities 和 session token。UI 在后续每次 host call 中都使用该 session token。

### AppRenderer / AppFrame SDK primitives

ext-apps SDK 暴露两个便利 primitives：

- `AppRenderer`（server side）— 包装 React / Vue / Solid component，并输出带正确 MIME 和 metadata 的 `ui://` resource。
- `AppFrame`（client side）— 接收 resource、mount iframe，并调解 postMessage。

你可以使用这些 primitive，也可以手写 HTML 和 JSON-RPC。

### 生态状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的 client support：

- **Claude Desktop。** 自 2026 年 1 月起 full support。
- **ChatGPT。** 通过 Apps SDK full support（底层是同一个 MCP Apps protocol）。
- **Cursor。** Beta；需要在 settings 中启用。
- **VS Code。** 仅 Insider builds。
- **Goose。** Full support。
- **Zed、Windsurf。** Roadmapped。

生产中的 servers：dashboards、map visualizations、data tables、chart builders、sandbox IDE previews。

## Use It

`code/main.py` 会扩展 notes server，加入一个 `visualize_timeline` 工具。它返回 `ui://notes/timeline` resource，并提供一个 `resources/read` handler：读取该 URI 时返回一个小而完整的 HTML bundle，里面有 SVG timeline。HTML 只用 stdlib template——没有 build system。postMessage 因为 stdlib 不能驱动 browser，所以只在 JS 注释中勾勒。

值得关注的地方：

- Tool response 上的 `_meta.ui` 携带 resourceUri、CSP 和 permissions。
- HTML 不需要网络访问；所有数据都 inline。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（已记录，但在这个 stdlib demo 中是 inert）。

## Ship It

本课产出 `outputs/skill-mcp-apps-spec.md`。给定一个会受益于交互式 UI 的工具，这个 skill 会产出完整的 MCP Apps contract：`ui://` URI、CSP、permissions、postMessage entrypoints 和安全 checklist。

## 练习

1. 运行 `code/main.py`，检查输出的 HTML。直接在浏览器中打开 HTML，确认 SVG 会渲染。然后草拟 UI 用来调用 `host.callTool("notes_update", ...)` 的 postMessage contract。

2. 收紧 CSP：移除 `'unsafe-inline'`，改用 nonce-based script policy。HTML generation code 需要做哪些变化？

3. 添加第二个 UI resource `ui://notes/editor`，提供一个 form 用于原地编辑 note。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的 attack surface。恶意 server 可以在哪里注入内容？Iframe sandbox 能防住什么，防不住什么？

5. 阅读 SEP-1724 spec，找出 MCP Apps SDK 中一个本 toy implementation 没有用到的 capability。（提示：component-level state sync。）

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP Apps | “Interactive UI resources” | 2026-01-26 发布的 SEP-1724 extension |
| `ui://` | “App URI scheme” | UI bundles 使用的 resource scheme |
| `text/html;profile=mcp-app` | “The MIME” | MCP App HTML 的 content-type |
| Iframe sandbox | “Render container” | 使用 CSP 和 permissions 对 UI 进行 browser sandboxing |
| postMessage JSON-RPC | “UI-to-host wire” | 用于 host calls 的 JSON-RPC-over-postMessage 小 dialect |
| `_meta.ui` | “Tool-UI binding” | 把 tool result 链接到 UI resource 的 metadata |
| CSP | “Content-Security-Policy” | 声明 scripts、network、styles 的允许来源 |
| AppRenderer | “Server SDK primitive” | 把 framework component 转换成 `ui://` resource |
| AppFrame | “Client SDK primitive” | Mount iframe 并调解 postMessage 的 helper |
| `ui/initialize` | “Handshake” | UI 发给 host 的第一条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation 和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — formal spec document
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — high-level documentation
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月 launch post
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc-style SDK reference
