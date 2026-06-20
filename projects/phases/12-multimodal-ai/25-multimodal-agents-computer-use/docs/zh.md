# 多模态 Agents 与 Computer-Use（Capstone）

> 2026 年 frontier product 是一个 multimodal agent：它读取 screenshots、点击 buttons、导航 web UIs、填写 forms，并端到端完成 workflows。SeeClick 和 CogAgent（2024）证明了 GUI-grounding primitive。Ferret-UI 加入了 mobile。ChartAgent 引入了面向 charts 的 visual tool-use。VisualWebArena 和 AgentVista（2026）是 frontier 正在追逐的 benchmarks，而且即使 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista hard tasks 上也只有约 30% 分数。这个 capstone 把 Phase 12 的所有线索汇总起来：perception（high-res VLM）、reasoning（带 tool use 的 LLM）、grounding（coordinate output）、long-horizon memory 和 evaluation。

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## Learning Objectives

- 设计一个 multimodal agent loop：perceive → reason → act → observe → repeat。
- 构建 GUI grounding output schema（click coordinates、type text、scroll、drag），让 VLM 能以 JSON 输出。
- 比较 screenshot-only agents、accessibility-tree agents 和 hybrid agents。
- 在一个小型 VisualWebArena slice 上设置 multimodal agent benchmark evaluation。

## 问题

一个 booking-site workflow："find me a flight to Tokyo for April 15, aisle seat under $800, book it."

Multimodal agent 需要：

1. 截取 browser screenshot。
2. 把 screenshot + URL + goal 解析成 plan。
3. 输出 structured action：click (at x,y)、type "Tokyo"（at element E）、scroll down、select（radio button）。
4. 把 action 应用到 browser。
5. 观察新状态（next screenshot）。
6. 重复直到 task 完成。

每一步都是一次 multimodal VLM call。VLM output 必须是 parseable JSON。错误会跨 steps 累积，因此 recovery 很重要。

## 概念

### GUI grounding：核心 primitive

GUI grounding 是：给定 screenshot 和 natural language instruction，输出要点击的 (x, y) coordinate（或其他 action）。

SeeClick（arXiv:2401.10935）是第一个 scale 上的 open result：在 synthetic + real GUI data 上 fine-tune VLM，把 coordinates 作为 plain text tokens 输出。有效。

CogAgent（arXiv:2312.08914）加入了 1120x1120 high-resolution encoding，用于 dense UIs。分数：web navigation 上约 84%。

Ferret-UI（arXiv:2404.05719）专注 mobile UIs，并与 iOS accessibility data 集成。

Output format 通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助于 recovery：如果 screenshots 之间坐标漂移，semantic hint 可以让系统重新 ground。

### Action schemas

典型 action schema 有 6-10 种 action types：

- `click`：(x, y)
- `type`：(text, x?, y?)
- `scroll`：(direction, amount)
- `drag`：(x0, y0, x1, y1)
- `select`：(option_index)
- `hover`：(x, y)
- `navigate`：(url)
- `wait`：(ms)
- `done`：(success, explanation)

Agent 每一步输出一个 action。Browser wrapper 执行它，并返回新状态。

### Screenshot-only vs accessibility-tree

两种 input modes：

- Screenshot-only：完整 image，没有 structural info。最通用；任何 app 都能工作。
- Accessibility tree：structured DOM / iOS accessibility info。Grounding 可靠得多；在 tree 可用的地方工作。
- Hybrid：两者都有，tree 作为 atomic actions 的可靠 grounder，screenshot 提供 semantic context。

Production agents 在可能时使用 hybrid。Browser automation（Selenium + accessibility）总是有 tree；desktop apps 有时有。

### Long-horizon memory

一个 20-step workflow 会产生 20 张 screenshots。VLM context 很快会填满。三种 compression strategies：

- Summary-chain：每 5 steps 总结发生了什么，丢弃旧 screenshots。
- Skip-frame：保留第一张、最后一张，以及每 3 张中的一张。
- Tool-recorded log：执行 actions，保留已完成操作的 text log；不重新查看旧 screenshots。

Claude 的 computer-use API 使用 log pattern。它更简单，也更可靠。

### Visual tool use

ChartAgent（arXiv:2510.04514）为 chart understanding 引入 visual tool use：crop、zoom、OCR、调用 external detection。Agent 可以输出 "crop to region (100, 200, 300, 400) then call OCR" 作为 tool call。Tool 返回 text；VLM 继续 reasoning。

这个 pattern 可以泛化：set-of-mark prompting、region annotation 和 external detection tools 都符合相同的 "output a tool call, receive a structured response" schema。

### 2026 benchmarks

- ScreenSpot-Pro。约 1k web screenshots 上的 GUI grounding。Open SOTA Qwen2.5-VL-72B 约 85%。Frontier 约 90%。
- VisualWebArena。End-to-end web tasks（shop、forum、classifieds）。Open SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista（arXiv:2602.23166）。2026 年最难 benchmark。跨 12 个 domains 的 realistic workflows。Frontier models 得分 27-40%；open models 10-20%。
- WebArena / WebShop。较老的 benchmarks；已被 frontier 饱和。

### 为什么它仍然很难

Agent performance bottlenecks：

1. Fine scale 的 visual grounding。"Click the small X" 在 mobile resolution 下经常失败。
2. Long-horizon planning。10 个 actions 之后，agent 会偏离 goal。
3. Error recovery。当 click 失败（点错 button）时，检测 + recovery 很少出现在 training data 中。
4. Cross-page context。在 tabs 或长表单之间跳转会丢失 state。

Research directions：memory architectures、explicit replanning、multimodal verification（用 screenshot match 判断 action success）。

### Capstone build-it

Capstone task：构建一个 computer-use agent，它能够：

1. 读取 booking-site mock page 的 HTML + screenshot。
2. 规划 multi-step sequence：search → select → fill form → submit。
3. 输出与 action schema 匹配的 JSON actions。
4. 在固定 10-task slice 上 evaluate。

本课提供 scaffold code，方便扩展成真实 browser。

## Use It

`code/main.py` 是 capstone scaffold：

- Action schema JSON definition（10 actions）。
- 作为 dict 的 mock browser state。
- Agent loop skeleton：接收 state、输出 action、apply、loop。
- 10-task mini-benchmark（synthetic pages），用于测量 end-to-end success rate。
- Error-recovery hook，用于 action fails 时。

## Ship It

本课产出 `outputs/skill-multimodal-agent-designer.md`。给定一个 computer-use product（domain、action set、evaluation target），它会设计完整 agent loop、memory strategy、grounding mode 和 expected benchmark score。

## Exercises

1. 用 `screenshot_region` tool（crop + zoom）扩展 action schema。哪些 tasks 会受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最难的 task category，以及为什么 frontier models 仍然失败。

3. Long-horizon memory compression：设计一个 summary-chain，live 中保留 ≤4 张 screenshots，log 数量不限。

4. 构建 error-recovery hook：当 action failure（button not found）发生时，agent 下一步做什么？

5. 在 10 个 web tasks 上比较 screenshot-only Claude 4.7 与 hybrid screenshot + accessibility-tree Qwen2.5-VL。哪种在哪些 tasks 上胜出？

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GUI grounding | "Click coordinates" | 模型为 screenshot 上的 instruction target 输出 (x,y) |
| Action schema | "Tool definitions" | 有效 actions（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | "Structured DOM" | 来自 browser/iOS APIs 的 machine-readable UI hierarchy |
| Hybrid agent | "Screenshot + tree" | 同时使用 image 和 structured info；比任一单独输入都更可靠 |
| Visual tool use | "Zoom/crop/detect" | Agent 在 plan 中途调用 external vision tools（OCR、detection） |
| Summary-chain | "Memory compression" | 用周期性 text summaries 替代冗长 screenshot history |
| VisualWebArena | "E2E web bench" | 2024 年的 end-to-end web tasks benchmark |
| AgentVista | "2026 hard bench" | 覆盖 12 个 domains 的 realistic workflows；即使 Gemini 3 Pro 也约 30% |

## Further Reading

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
