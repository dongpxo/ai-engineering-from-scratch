# Benchmarks：WebArena 与 OSWorld

> WebArena 在四个自托管应用上测试 web-agent 能力。OSWorld 在 Ubuntu、Windows、macOS 上测试 desktop-agent 能力。发布时（2023-2024），两者都显示出 best-in-class agents 和人类之间的巨大差距。差距正在缩小；失败模式没有变。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## 学习目标

- 描述 WebArena 的四个自托管应用，并解释为什么 execution-based evaluation 很重要。
- 解释为什么 OSWorld 使用真实 OS screenshots，而不是 accessibility APIs。
- 命名 OSWorld 的两个主要失败模式：GUI grounding 和 operational knowledge。
- 总结 OSWorld-G 和 OSWorld-Human 在基础 benchmark 之上增加了什么。

## 问题

通用 agents 可以调用工具。但它们能不能在浏览器里连续点击 20 次，完成一次购物 checkout？能不能只靠键盘和鼠标配置一台 Linux 机器？WebArena 和 OSWorld 回答的就是这些问题。

## 概念

### WebArena（Zhou 等人，ICLR 2024）

- 812 个 long-horizon tasks，覆盖四个自托管 web apps：购物网站、论坛、GitLab-like dev tool、business CMS。
- 还包括 utilities：map、calculator、scratchpad。
- Evaluation 通过 gym APIs 做 execution-based 判断：订单是否下成功，issue 是否关闭，CMS 页面是否更新？
- 发布时：最佳 GPT-4 agent 成功率 14.41%，人类 78.24%。

自托管设定很重要。目标 apps 被固定并可复现，因此 benchmark 不会因为外部网站变化而变得 flaky。

### Extensions

- **VisualWebArena**：visually grounded tasks，成功依赖于理解图像；screenshots 是 first-class observations。
- **TheAgentCompany**（2024 年 12 月）：加入 terminal + coding；更像真实 remote-work environment。

### OSWorld（Xie 等人，NeurIPS 2024）

- 369 个真实 computer tasks，覆盖 Ubuntu、Windows、macOS。
- 对真实应用进行 free-form keyboard 和 mouse control。
- 观察输入是 1920×1080 screenshots。
- 发布时：最佳模型 12.24%，人类 72.36%。

### 主要失败模式

1. **GUI grounding.** Pixel → element mapping。模型很难在 1920×1080 中可靠定位 UI elements。
2. **Operational knowledge.** 设置在哪个菜单、哪个 keyboard shortcut、哪个 preference pane。人类花多年积累的知识长尾。

### Follow-ups

- **OSWorld-G**：564-sample grounding suite + Jedi training set。把 grounding 和 planning 拆开，让你能分别测量它们。
- **OSWorld-Human**：人工整理的 gold action trajectories。它显示 top agents 使用的步骤数比必要步骤多 1.4-2.7 倍，也就是 trajectory-efficiency gap。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（Lesson 21）都在 WebArena 和 OSWorld 形状的 workloads 上训练。Benchmarks 是目标；生产模型是交付出来的答案。

### Benchmarking 容易在哪里出错

- **Screenshot-only evals.** OSWorld 是 screenshot-driven 的；如果你在 OSWorld 上评估一个使用 DOM 或 accessibility APIs 的 agent，就错过了 grounding challenge。
- **Ignoring trajectory length.** 只看 success-rate，会漏掉 OSWorld-Human 暴露的 1.4-2.7 倍步骤低效。
- **Stale self-hosted apps.** WebArena 的 apps 固定了特定版本；如果不重新 curated 就升级版本，会破坏可比性。

## Build It

`code/main.py` 实现了一个 toy web-agent harness：

- 一个最小 “shopping app” state machine：list_items、add_to_cart、checkout。
- 3 个 tasks 的 gold trajectories。
- 一个会尝试每个任务的 scripted agent。
- Execution-based evaluator（state check）和 trajectory-efficiency metric（steps vs gold）。

运行它：

```bash
python3 code/main.py
```

输出是 per-task success rate 和 trajectory efficiency，模拟 OSWorld-Human 的方法论。

## Use It

- **WebArena Verified**：在内部集群上自托管，用于 continuous evaluation。
- **OSWorld**：在 VM fleet 中评估 desktop agents。
- **Computer-use agents**（Lesson 21）：Claude、OpenAI CUA、Gemini 都在类似 workloads 上训练。
- **Your own product flows**：为最重要的 20 个任务捕获 gold trajectories；每周让 agents 跑一遍。

## Ship It

`outputs/skill-web-desktop-harness.md` 会构建一个 web/desktop agent harness，带 execution-based eval 和 trajectory efficiency metric。

## 练习

1. 给 toy harness 增加第二个 app（论坛）。写 3 个 tasks 和 gold trajectories。
2. 增加 per-task trajectory-efficiency reporting。在你的 toy 中，agent 是 gold 的 1 倍、2 倍，还是 3 倍？
3. 实现一个 “distractor” tool，也就是 gold trajectory 从不使用的工具。Scripted agent 会被诱惑吗？
4. 阅读 OSWorld-G。你会如何在自己的 evals 中区分 grounding failures 和 planning failures？
5. 阅读 WebArena 的 apps README。升级某个 pinned app version 时会破坏什么？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| WebArena | “Web agent benchmark” | 4 个自托管 apps 上的 812 个 tasks；gym-style evaluation |
| VisualWebArena | “Visual WebArena” | Visually grounded WebArena；screenshots 是 observations |
| OSWorld | “Desktop agent benchmark” | 真实 Ubuntu/Windows/macOS 上的 369 个 tasks |
| GUI grounding | “Pixel-to-element mapping” | 模型在 1920x1080 中定位 UI elements |
| Operational knowledge | “OS know-how” | 哪个菜单、哪个快捷键、哪个 preference pane |
| OSWorld-G | “Grounding suite” | 564 个 grounding-only samples + training set |
| OSWorld-Human | “Gold trajectories” | 人工专家 action sequences，用于测量 efficiency |
| Trajectory efficiency | “Steps over gold” | Agent step count 除以人类最小步骤数 |

## 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854)：四应用 web benchmark。
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972)：跨 OS desktop benchmark。
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use)：Claude 的 benchmark-shaped capability。
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/)：OSWorld 和 WebArena numbers。
