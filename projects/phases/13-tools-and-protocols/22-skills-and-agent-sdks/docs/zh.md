# Skills and Agent SDKs：Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说明“有哪些工具”。Skills 说明“如何完成一个任务”。2026 年的技术栈会把两者叠在一起。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以带 progressive disclosure 的 SKILL.md 形式发布。OpenAI 的 Apps SDK 是 MCP 加 widget metadata。AGENTS.md（现在已在 60,000+ repos 中使用）位于 repo root，作为 project-level agent context。本课会命名三者各自覆盖的范围，并构建一个可以跨 agents 迁移的最小 SKILL.md + AGENTS.md bundle。

**Type:** Learn
**Languages:** Python (stdlib, SKILL.md parser and loader)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 学习目标

- 区分三层：AGENTS.md（project context）、SKILL.md（reusable know-how）、MCP（tools）。
- 编写带 YAML frontmatter 和 progressive disclosure 的 SKILL.md。
- 以 filesystem-style 把 skills 加载进 agent runtime。
- 把一个 skill 与 MCP server 和 AGENTS.md 组合起来，让同一个 package 可以在 Claude Code、Cursor 和 Codex 中工作。

## 问题

一位工程师把 release notes 写作 workflow 提炼成一个多步骤 prompt：“读取最新 merge 的 PR。按 area 分组。总结每个 PR。按团队风格写 changelog entry。发布到 Slack draft。”他们把它放进团队的 Notion 文档。

现在他们想在 Claude Code、Cursor 和 Codex CLI 中使用这个 workflow。每个 agent 加载指令的方式都不同：Claude Code slash-commands、Cursor rules、Codex `.codex.md`。工程师把 workflow 复制三次，并维护三份副本。

AGENTS.md 和 SKILL.md 合在一起可以修复这个问题：

- **AGENTS.md** 位于 repo root。每个兼容 agent 都会在 session start 时读取它。“这个项目如何工作？有哪些 conventions？哪些 commands 跑 tests？”
- **SKILL.md** 是一个 portable bundle：YAML frontmatter（name、description）+ markdown body + optional resources。支持 skills 的 agents 会按需按名称加载它们。
- **MCP**（Phase 13 · 06-14）处理 skill 需要调用的 tools。

三层，一个 portable artifact。

## 概念

### AGENTS.md（agents.md）

AGENTS.md 于 2025 年末发布，到 2026 年 4 月已有 60,000+ repos 采用。它是 repo root 下的一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agents 会在 session start 时读取它，并用它来校准自己在该项目中的行为。2026 年的每个 coding agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md format

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明 skill 的 identity。Body 是 skill 加载时展示给模型的 prompt。

### Progressive disclosure

Skills 可以引用 sub-resources，agent 只在需要时 fetch 它们。例如：

```text
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 写着 “see style-guide.md for the style rules.” Agent 只会在 skill 正在运行时拉取 style-guide.md。这样可以避免把模型未必需要的细节塞满 prompt。

### Filesystem discovery

Agent runtimes 会扫描已知目录中的 SKILL.md files：

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载时使用 folder name 和 frontmatter `name`。Claude Code、Anthropic Claude Agent SDK，以及 SkillKit（cross-agent）都遵循这个 pattern。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）会在 session start 时加载 skills，并把它们作为 runtime 内部可调用的 “agents” 暴露出来。Agent loop 会在用户调用某个 skill 时 dispatch 到该 skill。

### OpenAI Apps SDK

OpenAI Apps SDK 于 2025 年 10 月发布，直接构建在 MCP 之上。它把 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个 developer surface。一个 Apps SDK app 包含：

- 一个 MCP server（tools、resources、prompts）。
- 加上 ChatGPT UI 的 widget metadata。
- 再加上可选的 MCP Apps `ui://` resource，用于 interactive surfaces。

同一个 protocol，更丰富的 UX。

### 通过 SkillKit 实现跨 agent 可移植性

SkillKit 和类似 cross-agent distribution layers，会把单一 SKILL.md 翻译成 32+ AI agents 的 native format（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）。一个 source of truth，多个 consumers。

### 三层 stack

| Layer | File | Loaded when | Purpose |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | project-level conventions |
| SKILL.md | skills directory | skill invoked | reusable workflow |
| MCP server | external process | tools needed | callable actions |

三者可以组合：agent 在 session start 时读取 AGENTS.md；用户调用一个 skill；skill 的指令包含 MCP tool calls；agent 通过 MCP client dispatch。

## Use It

`code/main.py` 提供了一个 stdlib SKILL.md parser 和 loader。它会发现 `./skills/` 下的 skills，解析 YAML frontmatter 和 markdown body，并生成一个以 skill name 为 key 的 dict。然后它模拟一个 agent loop，通过名称调用 `release-notes-writer`。

值得关注的地方：

- YAML frontmatter 使用一个最小 stdlib parser 解析（没有 `pyyaml` 依赖）。
- Skill body 会原样保存；agent 在调用时把它 prepend 到 system prompt。
- Progressive disclosure 通过 `read_subresource` function 演示：只在需要时拉取被引用的文件。

## Ship It

本课产出 `outputs/skill-agent-bundle.md`。给定一个 workflow，这个 skill 会产出组合好的 SKILL.md + AGENTS.md + MCP-server-blueprint bundle，可跨 agents 移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个 skill，并确认 loader 能发现它。

2. 为本课程 repo 编写一个 AGENTS.md。包含 testing commands、style conventions 和 Phase 13 mental model。

3. 把你团队内部文档里的一个 multi-step workflow 移植成 SKILL.md。验证它能在 Claude Code 中加载。

4. 手动把这个 skill 翻译成 Cursor 和 Codex 的 native rule formats。计算不同 formats 之间的 diff——这就是 SkillKit 自动化的 translation surface。

5. 阅读 Anthropic Agent Skills blog post。找出 Claude Agent SDK 中一个本课 loader 没有覆盖的 feature。（提示：agent sub-invocation。）

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| SKILL.md | “The skill file” | YAML frontmatter 加 markdown body，由 agent runtime 加载 |
| AGENTS.md | “Repo-root agent context” | Session start 时读取的 project-level conventions file |
| Progressive disclosure | “Lazy-load sub-resources” | Skill body 引用文件，只在需要时拉取 |
| Frontmatter | “YAML block at top” | `---` delimiters 中的 metadata（name、description） |
| Claude Agent SDK | “Anthropic's skill runtime” | `@anthropic-ai/claude-agent-sdk`，加载 skills 并 route |
| OpenAI Apps SDK | “MCP + widget meta” | OpenAI 基于 MCP 加 ChatGPT UI hooks 构建的 developer surface |
| Skill discovery | “Filesystem scan” | 遍历已知目录寻找 SKILL.md，并按 name 建 key |
| Cross-agent portability | “One skill many agents” | 通过 SkillKit-style tools 把一个 SKILL.md 翻译到 32+ agents |
| Agent Skill | “Portable know-how” | MCP tool 概念之外的可复用 task template |
| Apps SDK | “MCP plus ChatGPT UI” | Connectors 和 Custom GPTs 在 MCP 上的统一 |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月 launch
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md format reference
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — ChatGPT 的 MCP-based developer platform
- [agents.md](https://agents.md/) — AGENTS.md format 和 adoption list
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — official skill examples
