# 生产环境中的 EAGLE-3 Speculative Decoding

> Speculative decoding 会把一个快速草稿模型和目标模型配对。草稿模型先提出 K 个 token；目标模型用一次 forward 验证；被接受的 token 就近似免费。到 2026 年，EAGLE-3 是生产级变体：它不是在原始 token 上训练草稿模型，而是在目标模型的 hidden states 上训练 draft head，把通用聊天场景的接受率 alpha 推到 0.6-0.8 区间。真正的问题不是“草稿有多快”，而是“我的流量上 alpha 是多少？”如果 alpha 低于约 0.55，在高并发下 speculative decoding 会变成负收益，因为每个被拒绝的草稿都会带来第二次目标模型 forward。本课教你先测 alpha，再开开关。

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## 学习目标

- 说出 speculative decoding 的三代演进，并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型改变了什么。
- 定义接受率 alpha，根据 alpha 和 K（draft length）计算期望加速，并找出目标并发下的盈亏平衡 alpha。
- 解释为什么 speculative decoding 在 2026 年的 vLLM 中是 opt-in（不是默认开启），以及为什么不测 alpha 就直接开启是生产反模式。
- 写出一份测量计划：用哪个 benchmark、哪种 prompt 分布、哪个并发点，以及用哪个指标做发布闸门。

## 问题

Decode 是 memory-bound 的。在 H100 上运行 Llama 3.3 70B FP8 时，每解码一个 token 大约要读取 140 GB/s 的权重并输出一个 token。Decode 期间 GPU compute 几乎闲着，瓶颈是 HBM bandwidth，而不是 matmul throughput。

Speculative decoding 利用了这个空隙。先用廉价草稿模型生成 K 个候选 token，然后让目标模型用一次 forward 验证全部 K 个 token。每个通过验证的 token 实际上都是免费的（被摊进一次 batch-of-K forward，而目标模型本来也得做这些 forward）。

经典草稿模型方法会使用同一模型家族里的小模型（比如用 Llama 3.2 1B 为 Llama 3.3 70B 起草）。它能工作，但接受率一般，因为小模型分布会偏离目标模型。EAGLE、EAGLE-2、EAGLE-3 则直接在目标模型内部状态上训练轻量 draft head，因此草稿分布能更紧密地跟随目标模型。这就是为什么 alpha 会从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8。

关键限制是：EAGLE-3 在 2026 年的 vLLM 中是 opt-in。你必须显式设置 `speculative_config`。没有 flag，就没有加速。很多团队在真实流量上没有先测 alpha 就开启它，最后看到的是尾延迟变差，而不是变好。

## 概念

### Speculative decoding 实际买到了什么

没有 spec decode 时，每个 token 的成本是一次目标模型 forward。使用 draft length K 和接受率 alpha 做 spec decode 时，每次目标模型 forward 的期望 token 数是 `1 + K * alpha`。加速比是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft-plus-verify 开销。对于 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。真实世界的数据通常集中在 2-3x，因为生产流量上的 alpha 很少这么高，而且 epsilon 会在高 batch size 下增长。

### 为什么 alpha 是唯一真正重要的指标

被拒绝的 token 不会消失，它们会迫使系统为第一个被拒绝 token 做第二次目标模型 forward。当一个 workload 的 alpha 掉到 0.4 时，你要同时支付草稿开销、验证开销和重采样开销。在高并发（比如 256 并发）下，decode batch 已经足够大，“单独目标模型”和“带验证的目标模型”之间的内存带宽空隙会变小。在大多数 2026 硬件上，alpha 低于 0.55 时，spec decode 通常是负收益。

Alpha 会随 workload 变化。在 ShareGPT 风格的通用聊天上，用 ShareGPT 训练的 EAGLE-3 可以达到 0.6-0.8。在领域流量（代码、医疗、法律）上，用通用数据训练的 draft head 会掉到 0.4-0.6。训练领域专用 draft head 可以恢复 alpha，而且相比目标模型 finetuning，这只是一个轻量、快速的训练任务。

### EAGLE 各代速览

- **经典草稿模型**：同一模型家族中的小模型。Alpha 0.3-0.5。基础设施简单：加载两个模型，每次目标模型 forward 前草稿模型运行 K 次 forward。
- **EAGLE-1（2024）**：在目标模型 hidden states（最后一层）上训练的单个 draft head。Alpha 约 0.5-0.6。在目标模型之上增加少量参数开销。
- **EAGLE-2（2025）**：自适应 draft length 和基于 tree 的 drafts（一次目标模型 pass 验证多个分支）。Alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3（2025-2026）**：draft head 在多个目标模型层上训练（不只最后一层），对齐更好。通用聊天 alpha 约 0.6-0.8。

### 2026 年生产 recipe

1. 先用普通目标模型上线。测量目标并发下的 baseline TTFT、ITL 和 throughput。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 draft。重新跑 benchmark。
3. 记录接受率 alpha。vLLM V1 会以 `spec_decode_metrics.accepted_tokens_per_request` 上报它。除以请求的 draft length 即可得到 alpha。
4. 如果生产流量分布上的 alpha < 0.55，关闭 spec decode，或者训练领域专用 EAGLE-3 draft。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

Spec decode 会降低平均 ITL。但如果没有调好，P99 可能变差。被拒绝的 drafts 会触发两段式序列（draft + verify-fail + reroll）。在 full batch 下，这两段 pass 会串行化。要看 P99 ITL，不要只看 P50。

### EAGLE-3 已经部署在哪里

Google 在 2025 年的 AI Overviews 中部署了 speculative decoding（质量相同，响应更快）。vLLM V1 将 `speculative_config` 作为文档化接口；V1 中的 N-gram GPU speculative decoding 是兼容 chunked prefill 的变体。SGLang 支持 EAGLE-3，并把它作为 prefix-heavy workloads 的推荐草稿路径。

### 一行盈亏平衡数学

期望加速：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1`，可解得 alpha：`alpha_breakeven = verify_overhead / K`。对于典型的 verify_overhead ~0.15 和 K=5：`alpha_breakeven = 0.03`。但这是原始 decode 数学。在高并发下，验证开销会上升，而且 decode batch 已经把内存读取摊到了多个 sequence 上，所以实践中的有效 `alpha_breakeven` 会爬升到约 0.45-0.55。

### 什么时候不要用 speculative decoding

- Batch-1 离线生成，且 latency 不重要。直接用普通目标模型。
- 非常短的输出（少于 50 个 token）。草稿开销和验证成本会占主导。
- 没有领域专用 draft head 的专业领域。Alpha 太低。
- vLLM v0.18.0 + draft-model spec decode + `--enable-chunked-prefill`。这个组合无法编译。文档化的例外是 V1 中的 N-gram GPU spec decode。

## Use It

`code/main.py` 会模拟带 speculative decoding 和不带 speculative decoding 的 decode loop，并 sweep 一系列 alpha 值和 draft length K。它会打印盈亏平衡 alpha、实测加速和尾部行为。用几组不同的 `(alpha, K)` 组合运行它，你会清楚看到 speculative decoding 在什么位置停止赚钱。

## Ship It

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它会生成分阶段 EAGLE-3 rollout 计划：benchmark baseline、启用配置、测量 alpha、用 alpha >= 0.55 做闸门，并观察 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，要达到 2x 加速需要什么 alpha？3x 呢？这个结果对 verify_overhead 有多敏感？
2. 假设生产流量由 70% 通用聊天和 30% 代码组成。用 ShareGPT 训练的 EAGLE-3 在通用聊天上 alpha 为 0.7，在代码上 alpha 为 0.4。混合 alpha 是多少？Spec decode 是否仍然净正收益？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（draft model、EAGLE、N-gram），以及哪一种兼容 chunked prefill。
4. 启用 EAGLE-3 后，你看到平均 ITL 降低 25%，但 P99 ITL 上升 15%。诊断原因，并提出一个缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 draft head 内存成本。它与运行 Llama 3.2 1B 作为经典草稿模型相比如何？

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Speculative decoding | “draft plus verify” | 用廉价模型提出 K 个 token，并用一次目标模型 forward 验证全部 K 个 token |
| Acceptance rate alpha | “spec accept rate” | 被目标模型接受的草稿 token 比例；唯一真正重要的指标 |
| Draft length K | “spec k” | 每次目标模型 forward 前草稿模型提出多少 token；典型值为 4-8 |
| Verify overhead epsilon | “spec overhead” | 相比普通目标模型 forward，验证和重采样带来的额外成本；随 batch 增长 |
| EAGLE-3 | “latest EAGLE” | 2025-2026 变体；在多个目标模型层上训练 draft head；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | “vLLM spec config” | vLLM V1 中显式 opt-in 的配置；默认不开启意味着没有加速 |
| N-gram spec decode | “N-gram draft” | 在 GPU 侧用 prompt 中的 N-gram 查找做草稿；兼容 chunked prefill |
| Break-even alpha | “no-op alpha” | Spec decode 零加速时的 alpha；生产并发下尤其要关注 |
| Rejected-draft two-pass | “reroll cost” | 草稿被拒绝时出现的两次目标模型 forward；会推高 P99 尾部 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — `speculative_config` 和 V1 中 chunked-prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确字段集合。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE draft-head 公式化。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应 drafts 和 trees。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 使用 speculative decoding 的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产 rollout checklist。
