# Capstone 09 — 代码迁移 Agent（Repo-Level Language / Runtime Upgrade）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 migrator 设定了 2026 年的标准。Moderne 的 OpenRewrite 可以在规模化场景中执行 deterministic AST rewrites。Grit 用 codemod-style DSL 解决同类问题。Production pattern 会把二者结合起来：用 deterministic substrate 处理安全 rewrites，用 agent layer 处理 ambiguous cases，为每个 branch 提供 sandbox build，并用 test harness 在 PR 打开前把状态翻绿。本 capstone 的目标是迁移 50 个真实 repos，并发布 pass rate 和 failure taxonomy。

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## 问题

Large-scale code migration 是 2026 coding agents 最干净的 production applications 之一。Ground truth 很明显（migration 后 test suite 是否通过？），reward 很真实（Java-8 fleet migration 是 headcount-scale project），benchmark 也是公开的（MigrationBench 50-repo subset）。Moderne 的 OpenRewrite 处理 deterministic side。Agent layer 处理 OpenRewrite recipes 无法覆盖的一切：ambiguous rewrites、build-system drift、long-tail syntax、transitive dependency breakage。

你会构建一个 agent：输入 Java 8 repo（或 Python 2 repo），输出 green-CI migrated branch。你会测量 pass rate、test-coverage preservation、cost per repo，并构建 failure taxonomy。与 deterministic-only baseline 的 side-by-side 对比会告诉你 agent 的价值到底在哪里。

## 概念

Pipeline 有两层。**Deterministic substrate**（Java 用 OpenRewrite，Python 用 libcst）会安全地执行大部分 mechanical rewrites：imports、method signatures、null-safety edits、try-with-resources、deprecated API replacements。它速度快，并产出 auditable diffs。**Agent layer**（OpenAI Agents SDK 或 LangGraph，基于 Claude Opus 4.7 和 GPT-5.4-Codex）处理 recipes 无法覆盖的 cases：build-file upgrades（Maven/Gradle/pyproject）、transitive dependency conflicts、test flakes、custom annotations。

每个 repo 都会获得一个 Daytona sandbox，并预装 target runtime。Agent 迭代执行：run build、classify failures、apply fix、rerun。Hard limits：每个 repo 30 分钟、$8、20 个 agent turns。如果所有 tests pass，且 coverage delta 不为负，就打开 branch PR。否则，这个 repo 会带 evidence 归档到某个 failure class 下。

Failure taxonomy 是 deliverable。跨 50 个 repos，到底哪里坏了？Transitive deps？Custom annotations？Build tool version？与 migration 无关的 test flakes？每个 class 都有 count 和 exemplar diff。未来的 recipe authors 可以优先处理 top three。

## 架构

```text
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- Deterministic substrate：OpenRewrite（Java）或 libcst（Python）
- Agent：OpenAI Agents SDK 或 LangGraph，基于 Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox：Daytona devcontainers per branch，预装 target runtime（Java 17 / Python 3.12）
- Build systems：Maven、Gradle、uv（Python）
- Benchmarks：Amazon MigrationBench 50-repo subset（Java 8 到 17）、Google App Engine Py2-to-Py3 repos
- Test harness：parallel runner；Java coverage 用 Jacoco，Python coverage 用 coverage.py
- Observability：Langfuse + 每个 repo 的 trace bundle，包含每个 diff chunk
- Dashboard：failure-taxonomy dashboard，展示 per-class counts 和 exemplar diffs

## Build It

1. **Recipe pass。** 先运行 OpenRewrite（Java）或 libcst（Python）recipes。捕获 70-80% mechanical migrations。提交为 “recipe” commit。

2. **Build trial。** Daytona sandbox：安装 target runtime，运行 build。如果 green，跳到 tests。如果 red，交给 agent。

3. **Agent loop。** LangGraph 带 tools：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。Agent 对 failure 分类（dep、syntax、test、build-tool），并应用 targeted fix。然后 rerun。

4. **Budget caps。** 每个 repo 30 分钟 wall-clock、$8 cost、20 个 agent turns。任何 breach 都会 halt，并以当前 diff 归档到 “budget_exhausted”。

5. **Test + coverage gate。** Build 变 green 后，运行 test suite。把 coverage 与 base repo 比较。如果 coverage 下降超过 2%，归档到 “coverage_regression”。

6. **PR open。** 成功后 push branch，打开 PR，并附上 diff、applied recipes summary，以及哪些 commits 由 agent authored。

7. **Failure taxonomy。** 对每个 failed repo 打上 class：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建 dashboard。

8. **50-repo run。** 在 MigrationBench subset 上执行。报告 per-class pass rate、cost-per-repo、coverage-preservation，以及 compare-vs-deterministic-only baseline。

## Use It

```console
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Ship It

`outputs/skill-migration-agent.md` 是 deliverable。给定一个 repo，它会先执行 deterministic recipes，再执行 agent loop，以产出 green migrated branch；如果失败，则把 repo 归档到 taxonomy class 下。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | MigrationBench pass rate | 50-repo subset pass@1 |
| 20 | Test-coverage preservation | 相对 base 的 mean coverage delta |
| 20 | Cost per migrated repo | Passing runs 上的 $/repo |
| 20 | Agent / deterministic-tool integration | OpenRewrite 处理的 fixes 与 agent authored fixes 的比例 |
| 15 | Failure analysis write-up | 带 exemplars 的 taxonomy completeness |
| **100** | | |

## 练习

1. 只用 OpenRewrite（不启用 agent）运行 migrate pipeline。把 pass rate 与完整 pipeline 比较。识别只有 agent 才能带来差异的 cases。

2. 实现一个 “lint-clean” check：migration 后运行 style linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint errors，则让 PR fail。测量 coverage-preserved-but-style-regressed rate。

3. 添加一个 “minimal-diff” optimizer：agent branch 通过 tests 后，用第二轮 pass 修剪不必要 changes。报告 diff-size reduction。

4. 扩展到第三种 migration：Node 18 到 Node 22。复用 sandbox wrapping；把 recipe layer 换成 custom codemod。

5. 把 time-to-first-green-build（TTFGB）作为 UX metric 来测。目标：p50 低于 10 分钟。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Deterministic substrate | “Recipe engine” | OpenRewrite / libcst：带 safety guarantees 的 declarative AST rewrites |
| Codemod | “Code-modifying program” | 机械修改 source code 的 rewrite rule |
| Build drift | “Tool version skew” | Maven / Gradle / uv 在 major versions 之间的细微行为变化 |
| Failure class | “Taxonomy bucket” | Repo 没有迁移成功的标记原因：dep、syntax、test、build-tool、budget |
| Coverage delta | “Coverage preservation” | 从 base 到 migrated branch 的 test coverage % 变化 |
| Agent turn | “Tool-call round” | Agent loop 中一次 plan -> act -> observe cycle |
| Budget exhaustion | “Hit the ceiling” | Repo 在通过前耗尽 30-min / $8 / 20-turn limit |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — canonical 2026 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — deterministic substrate reference
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — alternate codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK reference
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — alternate migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python deterministic substrate
- [Daytona sandboxes](https://daytona.io) — reference per-branch sandbox
