# Capstone 01 — 终端原生编码 Agent

> 到 2026 年，coding agent 的形态已经基本稳定：TUI harness、stateful plan、sandboxed tool surface，以及一个会 plan、act、observe、recover 的 loop。Claude Code、Cursor 3 和 OpenCode 从远处看都长得差不多。这个 capstone 要求你端到端构建一个这样的系统：CLI 输入，pull request 输出，并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 对比。你会学到，难点不是 model call，而是 tool loop、sandbox，以及 50-turn run 里的 cost ceiling。

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## 问题

Coding agents 在 2026 年成为主导性的 AI application category。Claude Code（Anthropic）、带有 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids 和 Google Jules 都交付了同一种 architecture 的变体：terminal harness、permissioned tool surface、sandbox，以及围绕 frontier model 构建的 plan-act-observe loop。Frontier 很窄：Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%；但 engineering craft 很宽。大多数 failure modes 不是 model mistakes，而是 tool-loop instability、context poisoning、runaway token cost，以及 destructive filesystem operations。

你不能从外部推理这些 agents。你必须自己构建一个，看着 loop 在第 47 turn 因为 ripgrep 返回 8MB matches 而崩溃，然后重建 truncation layer。这就是这个 capstone 的目的。

## 概念

Harness 有四个 surface。**Plan** 维护一个 TodoWrite-style state object，由 model 在每一轮重写。**Act** 分发 tool calls（read、edit、run、search、git）。**Observe** 捕获 stdout / stderr / exit codes，进行 truncation，然后把 summary 喂回模型。**Recover** 处理 tool errors，避免撑爆 context window 或陷入无限循环。2026 年的形态还增加了一个东西：**hooks**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`：这些是 configurable extension points，operator 可以在其中注入 policy、telemetry 和 guardrails。

Sandbox 是 E2B 或 Daytona。每个 task 都在 fresh devcontainer 中运行，里面挂载一个 read-write git worktree。Harness 绝不触碰 host filesystem。Worktree 会在 success 或 failure 后被销毁。Cost control 在三层执行：per-turn token ceiling、per-session dollar budget，以及 hard turn limit（通常是 50）。Observability layer 使用带有 GenAI semantic conventions 的 OpenTelemetry spans，并发送到 self-hosted Langfuse。

## Architecture

```text
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Stack

- Harness runtime：Bun 1.2 + Ink 5（React-in-terminal）
- Model access：OpenRouter unified API，接入 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难任务）
- Tool transport：Model Context Protocol StreamableHTTP（MCP 2026 revision）
- Sandbox：E2B sandboxes（JS SDK）或 Daytona devcontainers
- Code search：ripgrep subprocess、面向 17 种语言的 tree-sitter parsers（pre-compiled）
- Isolation：每个 task 使用 `git worktree add`，success / failure 后 cleanup
- Eval harness：SWE-bench Pro（verified subset）+ Terminal-Bench 2.0 + 你自己的 30-task holdout
- Observability：OpenTelemetry SDK，使用 `gen_ai.*` semconv -> self-hosted Langfuse
- PR posting：GitHub App，使用 fine-grained token，scope 限制在目标 repo

## Build It

1. **TUI and command loop。** 用 Ink scaffold 一个 Bun project。接受 `agent run <repo> "<task>"`。打印 split view：plan pane（top）、tool-call stream（middle）、token budget（bottom）。添加 Ctrl-C cancel，并在退出前触发 `SessionEnd` hook。

2. **Plan state。** 定义 typed TodoWrite schema（pending / in_progress / done items，带 notes）。Model 每一轮以 tool call 的形式重写完整 state，不要让它 incremental mutate。把 plan 持久化到 `.agent/state.json`，这样 crash 后可以 resume。

3. **Tool surface。** 定义六个 tools：`read_file`、`edit_file`（带 diff preview）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带 timeout）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，让 harness 与 transport 解耦。每个 tool 返回 truncated output（每次调用 cap at 4k tokens）。

4. **Sandbox wrapping。** 每个 task 启动一个 E2B sandbox。执行 `git worktree add -b agent/$TASK_ID` 创建 fresh branch。所有 tool calls 都在 sandbox 内执行。Host filesystem 不可达。

5. **Hooks。** 实现 2026 年全部八类 hook。至少接入四个 user-authored hooks：(a) `PreToolUse` destructive-command guard，阻止 worktree 外的 `rm -rf`；(b) `PostToolUse` token accounting；(c) `SessionStart` budget initialization；(d) `Stop` 写出 final trace bundle。

6. **Eval loop。** Clone SWE-bench Pro Python 的 30-issue subset。用你的 harness 跑每个 issue。基于 pass@1、turns-per-task 和 $-per-task 与 mini-swe-agent（minimal baseline）对比。把结果写入 `eval/results.jsonl`。

7. **Cost control。** Hard cutoffs：50 turns、200k context、每个 task $5。`PreCompact` hook 在 150k mark 将旧 turns 总结成 prior-state block，为新的 observations 腾出空间，同时保留 plan。

8. **PR posting。** 成功时最后一步是 `git push` + GitHub API call，打开一个 PR，并在 body 中写入 plan 和 diff summary。

## Use It

```console
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Ship It

交付物 skill 位于 `outputs/skill-terminal-coding-agent.md`。给定一个 repo path 和 task description，它会在 sandbox 中运行完整的 plan-act-observe loop，并返回 PR URL 和 trace bundle。本 capstone 的 rubric：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Your harness vs mini-swe-agent on 30 matched Python tasks |
| 20 | Architecture clarity | Plan/act/observe separation, hook surface, tool schema — reviewed against Live-SWE-agent layout |
| 20 | Safety | Sandbox escape tests, permission prompts, destructive-command guard passes red-team |
| 20 | Observability | Trace completeness (100% of tool calls spanned), token accounting per turn |
| 15 | Developer UX | Cold-start < 2s, crash recovery resumes plan, Ctrl-C cancels mid-tool cleanly |
| **100** | | |

## 练习

1. 把 backing model 从 Claude Sonnet 4.7 换成运行在 vLLM 上的 Qwen3-Coder-30B。比较 pass@1 和 $-per-task。报告 open model 表现不足的位置。

2. 添加一个 `reviewer` sub-agent，在 PR posting 前读取 diff，并可以请求 revision loop。测量 false-positive reviews 是否会把 SWE-bench pass rate 拉低到 single-agent baseline 之下（提示：通常会）。

3. Stress-test sandbox：写一个尝试 `curl` external URL 的 task，以及一个尝试写入 worktree 外部的 task。确认两者都被 PreToolUse hook 阻止。记录这些 attempts。

4. 使用较小模型（Haiku 4.5）实现 `PreCompact` summarization。测量 3x compaction 时 plan fidelity 损失多少。

5. 把 MCP StreamableHTTP transport 换成 stdio。Benchmark cold-start 和 per-call latency。为 local-only use 选择胜者。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Harness | “The agent loop” | 包围 model 的代码：分发 tools、维护 plan state、执行 budgets |
| Hook | “Agent event listener” | Harness 在八类 lifecycle events 中运行的 user-authored script |
| Worktree | “Git sandbox” | 独立路径上的 linked git checkout；可丢弃且不触碰 main clone |
| TodoWrite | “Plan state” | Typed pending/in-progress/done item list，由 model 每轮重写 |
| StreamableHTTP | “MCP transport” | 2026 MCP revision：带 bidirectional streaming 的 long-lived HTTP connection；替代 SSE |
| Token ceiling | “Context budget” | 对 input+output tokens 的 per-turn 或 per-session cap；触发 compaction 或 termination |
| pass@1 | “Single-attempt pass rate” | SWE-bench tasks 在第一次运行中被解决的比例，不 retry，也不窥探 test set |

## 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的 reference harness
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 product notes
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 用于 SWE-bench harness comparison 的 minimal baseline
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 达到 79.2% SWE-bench Verified
- [OpenCode](https://opencode.ai) — open harness，112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 本 capstone 面向的 evaluation
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — tool calls 和 token usage 的 span schema
