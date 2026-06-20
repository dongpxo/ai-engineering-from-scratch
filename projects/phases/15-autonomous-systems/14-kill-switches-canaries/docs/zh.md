# Kill Switches、Circuit Breakers 与 Canary Tokens

> Kill switch 是一个保存在 agent edit surface 之外的 boolean——Redis key、feature flag、signed config——用来完全禁用 agent。Circuit breaker 更细粒度：它在特定模式上触发（比如连续五次相同 tool call），暂停出问题的路径，并升级给人类。Canary token 继承自经典 deception：一个 fake credential 或 honeypot record，agent 没有任何正当理由接触它，一旦访问就触发告警。基于 eBPF 的 datapaths（例如 Cilium）可以在 kernel layer 把 quarantined pod 的 egress 重写到 forensic honeypot；公开的 Cilium benchmarks 报告在负载下 P99 datapath latency 低于 1 毫秒（你的 propagation budget 取决于 policy update 如何抵达 node，而不是 datapath 本身）。会适应移动 baseline 的 statistical detectors（EWMA、CUSUM）会悄悄接受 drift——要把它们和不会弯曲的 hard constitutional limits 叠起来用。

**Type:** Learn
**Languages:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## 问题

Cost governors（Lesson 13）限制 agent 能花多少钱。它们不限制 agent 在预算内能做什么。一个带有 $50 velocity limit 的 agent 仍然可以外泄 secret、发布错误帖子，或者删除资源——危险动作在 token 上往往很便宜。

本课覆盖位于 cost layer 旁边的三类 detectors：

1. **Kill switch**：保存在 agent 触达范围之外的 boolean off-button。
2. **Circuit breaker**：检测 action pattern，并暂停特定路径。
3. **Canary token**：诱饵；没有正当理由接触它的 agent 一旦碰到它，就会暴露自己。

三者都是 pre-LLM engineering。经典 deception、rate-limit breakers 和 feature-flag kills 都早于自主 agent。新的地方是 attack surface：agents 会读取不可信内容（Lesson 11）、编辑自己的 memory，并能把许多看起来安全的 actions 组合成不安全的结果。这里命名的 detectors 有效，是因为它们不信任 agent 的 self-report。

## 概念

### Kill switches

Kill switch 是一个 agent 可以读取、但不能写入的 boolean。实现选择包括：

- **托管服务里的 feature flag。** LaunchDarkly、Statsig、Unleash。更新会在数秒内传播。
- **Agent 轮询的 Redis key。** 简单；要求 agent process 在每一轮都检查。
- **Object storage 中的 signed config。** Agent 启动时验证 signature；拒绝 unsigned states。
- **OS-level signal 或 container-lifecycle kill。** Docker `kill`、Kubernetes `kubectl delete pod`、systemd stop。

正确的 kill switch 具备这些性质：

- Agent 不能把它设成 `off`。（它存在于 agent credentials 无写权限的系统里。）
- 每个 consequential action 都会检查它，而不只是在启动时检查。
- 当它关闭时，agent 不做任何 externally observable 的事，包括向 agent 能触达的系统写 log。
- 重新启用是明确的人类动作，而不是自动 timeout。

### Circuit breakers

Circuit breaker 暂停的是特定模式，而不是整个 agent。经典形态（来自 2007 年 Nygard 的书，今天仍然适用）：

- **Closed**：action 被允许。
- **Open**：action 被阻止。
- **Half-open**：cool-down 后允许 1–3 次 probe attempts（默认 1 次）；成功会关闭 breaker，任何剩余失败都会重新打开它。

与 agent 相关的 triggers：

- 连续五次相同 tool calls（重复循环）。
- 同一个 tool 在不同输入上连续失败五次（系统性失败）。
- Tool call rate 超过阈值（Lesson 13 velocity）。
- 读取 out-of-trust content 后调用了特定 tool（例如 `file.delete`）（Lesson 11）。

### Canary tokens

Canary tokens（也叫 honeytokens）是 agent 永远不该接触的 inputs。访问就是告警。

Agent 场景中的例子：

- 环境变量里一个没有真实权限的 fake `AWS_SECRET_ACCESS_KEY`。任何 exfiltration attempt 都会立刻可归因。
- 一个带有已知 sentinel value 的 fake database record。对该记录的任何 read 或 update 都会触发 alert。
- Workspace 里的一个 fake file。读取或编辑这个文件都可疑。
- Session start 时注入的 fake memory entry。如果它出现在后续输出里，说明 memory 被 exfiltrated。

Canary design 要贴合 workflow。一个 agent 有正当理由接触的 canary 只会制造噪声。

### 为什么要叠加 statistical 和 hard limits

Statistical detectors（EWMA、CUSUM、tool-call rates 上的 z-score）会适应移动 baseline。这对合法 drift 是特性，对 adversarial drift 是 bug。有耐心的攻击者可以慢慢移动 baseline，让告警永远不响。

Hard constitutional limits（Lesson 17）不会弯曲。把 “Never email `secrets@`” 写成 constitutional rule，更简单、可审计，也不会被 slow drift 攻破。Layering pattern 是：statistical detectors 捕获大多数噪声，hard limits 捕获那些绕过 detector 的攻击。

### 通过 eBPF datapath redirect 做 quarantine

Cilium 的 eBPF-based network policy 可以在 kernel layer 把 quarantined pod 的 egress 重写到 forensic honeypot。Cilium 的公开 benchmarks 报告 TCP_RR 上 P99 datapath latency 低于 1 毫秒；在你的 cluster 里，触发 quarantine 的 wall-clock budget 主要由 policy update 从 control plane 传播到 node 的方式决定（如果 operator 已预加载，在 Kubernetes 上几百毫秒是现实的；tail latency 取决于 cluster size 和 reconcile cadence）。这给你两件事：agent 不会外泄任何真实内容，并且 agent 在 honeypot 中的行为会被完整观察。同一模式、不同层级，就像 canary token：agent 在被 containment 后尝试做什么，会暴露它的 intent。

### 没有 detector 能捕获什么

- **由允许 primitives 组合出来的 semantic attacks。** 每个独立 action 都通过了每个 detector；组合才是问题。这就是 Lesson 10 的 Auto Mode composite failure。
- **把 detector 本身包含进 payload 的攻击。** 如果 agent 读到一个页面说“行动前先禁用 canary”，而 agent 又有这个能力，那么 detector 已经被破坏。Canary tokens 应该存在于 agent 无法修改的系统里。

## Use It

`code/main.py` 会模拟一段短 agent trajectory 穿过三个 detectors。一个保存在 external dict 中的 kill switch；一个在五次相同 tool calls 后触发的 circuit breaker；一个读取即触发 alert 的 canary file。它会喂入一段 synthetic trajectory：合法 actions、重复循环、canary probe，以及一个由 kill switch 触发、agent actions 被 halted 的场景。

## Ship It

`outputs/skill-tripwire-design.md` 会审查某个 agent deployment 提出的 detector stack，并标出 gaps（缺 kill switch、缺 canary、circuit breaker threshold 太松）。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker 在 turn 5（第五次相同调用）触发，canary 在 turn 9（读取 fake-key）触发。

2. 添加一个 statistical detector：tool-call rate 上的 EWMA z-score。喂入一段缓慢 drift 的 trajectory，并展示 detector 从不触发。然后添加一个 hard limit（10 分钟内不超过 50 次 tool calls），并展示同一条 trajectory 会触发 hard limit。

3. 为一个 browser agent（Lesson 11）设计 canary token set。列出至少三个 canaries，以及每个会检测什么。

4. 阅读 Cilium network-policy docs。具体描述一个 egress-redirect quarantine flow：哪个 policy selector、哪个 pod、哪个 egress rewrite、哪个 alert。什么因素决定从“决定 quarantine”到“第一个被重定向 packet”的 wall-clock latency？

5. 为一个被 kill-switched 的 agent 定义 re-enable procedure。谁可以重新启用？必须记录什么？重新启用前 agent 必须发生什么变化？

## 关键术语

| Term | 常见说法 | 实际含义 |
|---|---|---|
| Kill switch | “Off button” | 位于 agent edit surface 之外的 boolean；每个 consequential action 都会检查 |
| Circuit breaker | “Pattern pause” | 针对重复、失败率或 rate-limit 的 action-specific trip |
| Canary token | “Honeytoken” | Agent 没有正当理由接触的诱饵；访问即触发 alert |
| Honeypot | “Forensic sandbox” | 被 quarantine 的 agent 会被观察的 redirected traffic / workspace |
| EWMA | “Moving average” | 指数加权；会适应 drift（既是特性也是 bug） |
| CUSUM | “Cumulative sum” | 检测 baseline 的持续偏移 |
| Hard limit | “Constitutional rule” | 不会自适应；无论历史如何都保持常量 |
| Constitutional limit | “Always-true rule” | 绑定 Lesson 17 的 constitution；agent 无法编辑 |

## 延伸阅读

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — kill-switch and circuit-breaker framing for autonomous agents.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — production governance patterns.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — detection-and-response requirements.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod-level egress redirect and forensic honeypot patterns.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — hardcoded prohibitions as "constitutional limits".
