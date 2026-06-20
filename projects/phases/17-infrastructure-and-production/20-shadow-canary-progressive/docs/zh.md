# 面向 LLM 的影子流量、金丝雀发布与渐进式部署

> LLM rollout 把软件部署里最难的部分叠在了一起：没有能完全覆盖的 unit tests、failure modes 分散、signals 还会延迟出现。顺序是：(1) shadow mode：把 production requests 复制给 candidate model，记录并比较，但不影响用户；能抓住明显的 distribution 问题，但不是 quality guarantee；(2) canary rollout：按 10% → 25% → 50% → 75% → 100% 渐进转移流量，每一步都有 gates；追踪 latency percentiles、cost/request、error/refusal rate、output length distribution、user-feedback rate；(3) 稳定性确认之后，再对差异明显的 alternatives 做 A/B testing。Non-determinism 无法彻底消除：即使输入完全相同，因为 GPU FP non-associativity 和 batch-size variance，运行之间也可能有最高 15% 的 accuracy variation。Cost 是变量，不是常量：一个好 20% 的 model 可能每次调用贵 3x。Rollback speed 决定生死：如果 rollback 需要 redeploy，你已经太慢了。Policy 放在 config/flags；model 放在 registry 并 pin digest；rollback = 翻转 policy + 恢复 threshold + 在几秒内 pin 回旧 model。

**Type:** Learn
**Languages:** Python（stdlib，玩具版 canary-progression simulator）
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 minutes

## 学习目标

- 区分 shadow mode（zero-impact compare）、canary（live traffic progressive）和 A/B（stability-confirmed comparison）。
- 列出五个 LLM-specific canary metrics：latency、cost/request、error/refusal、output-length distribution、user feedback。
- 解释为什么 LLM non-determinism（最高 15%）会改变 rollout 中 “stable” 的含义。
- 设计一个以秒为单位完成的 rollback path（policy flip），而不是以小时为单位（redeploy）。

## 问题

你发布了一个新模型。Offline evals 显示 accuracy 提升 3%。你直接在 production 打开它。24 小时内，成本上涨 40%，用户 thumbs-down 增加 8%，三个 customer tickets 都在说“回答很奇怪”。你开始 rollback。Redeploy 花了 3 小时。周末没了。

这一切都可以避免。Shadow mode 会在任何用户看到之前抓住 40% 的 cost spike。Canary 会在 thumbs-down 刚移动时停在 10%。Policy-flag rollback 只需要 30 秒。这套纪律填补的是 “offline evals 看起来不错” 和 “真实用户真的满意” 之间的缺口。

## 概念

### Shadow mode

Candidate 收到与 production 相同的 requests；outputs 被记录，但不会返回给用户。Zero user impact。记录：

- Output content（与 production diff）。
- Token counts（cost delta）。
- Latency。
- Refusal 和 error。

能抓住：cost blow-ups、length regressions、明显的 refusal changes、hard errors。抓不住：用户会感知到的 quality delta。Shadow 是 smoke test，不是 quality test。

### Canary rollout

带 gates 的渐进式流量转移。典型 progression：1% → 10% → 25% → 50% → 75% → 100%。每一步用 5 个 metrics 设 gate：

1. **Latency percentiles** — P50、P95、P99。Breach：canary 的 P99 > baseline 的 1.5x。
2. **Cost per request** — blended $。Breach：比 baseline 高 >20%。
3. **Error / refusal rate** — 5xx 加显式 refusals。Breach：baseline 的 2x。
4. **Output length distribution** — mean + P99。Breach：出现 distributional shift。
5. **User-feedback rate** — thumbs-down / ticket filings。Breach：baseline 的 1.5x。

### Non-determinism 是新的 variance

相同 inputs 会产生不完全相同的 outputs。原因：

- GPU FP non-associativity（floating-point reduction order 会随 batch 变化）。
- Batch-size variance（同一个 prompt 放在 batch of 128 和 batch of 16 里不同）。
- Sampling（temperature > 0）。

实测：同一组 eval sets、相同 inputs，run-to-run accuracy variation 最高可达 15%。“Stable” 在 rollout 中的意思不是与 baseline 完全相同，而是 metrics 落在 expected variance 之内。Gates 要设在 noise floor 之上。

### Cost 是变量

一个好 20% 的 model 可能每次调用贵 3x。Cost/request 是五个 gates 之一。发布一个 “better” model，但破坏 unit economics，本身就是 rollback case。

### Rollback 是武器

- Policy flag（feature flag system）：在 config 中翻转 percentage；耗时几秒。
- Model pinning（registry digest）：pinned model 不会自动升级。
- Rollback = revert flag + set pinned digest to previous。几秒，不是几小时。

如果你的 stack 需要 redeploy 才能 rollback，先修它，再 rollout。

### Tooling

**Argo Rollouts** / **Flagger** — Kubernetes progressive delivery controllers。与 Istio/Linkerd weighted routing 集成。

**Istio weighted routing** — service-mesh-level traffic split。

**KServe / Seldon Core** — 带内置 canary primitives 的 model serving。

**Feature flags** — LaunchDarkly、Flagsmith、Unleash。Policy-level flip，无需 redeploy。

### Metrics cadence

Canary gates 每 5-15 分钟检查一次，具体取决于 traffic volume。1% traffic 且 10 req/min 时，每个 window 只有 50-150 个 data points；对 latency 足够，但对 user feedback 很 noisy。10% 会带来约 10x 数据。Progressions 应该在每一步暂停足够久，积累足够 samples。

### A/B step 是可选的

如果新模型明显不同（behavior 不同、cost curve 不同、tone 不同），canary 通过后在 50% 做 A/B test。如果它只是改进版，canary gates 通过后可以直接到 100%。

### 你应该记住的数字

- Canary progression：1% → 10% → 25% → 50% → 75% → 100%。
- Non-determinism ceiling：同一 inputs 的 run-to-run variance 最高可达 15%。
- 五个 canary metrics：latency、cost、error/refusal、output length、user feedback。
- Cost gate：比 baseline 高 >20% 就是 breach。
- Rollback：秒级，不是小时级。

## Use It

`code/main.py` 会模拟带 injected regressions 的 canary rollout。它会报告 rollout 停在哪个 stage，以及触发了哪个 gate。

## Ship It

本课产出 `outputs/skill-rollout-runbook.md`。给定 candidate model、baseline 和 risk tolerance，它会设计 shadow→canary→100% plan。

## 练习

1. 运行 `code/main.py`。注入 25% cost regression。Canary 会停在哪个 stage？
2. 你的新 model offline accuracy gain 是 3%，但 cost/request 是 +18%。能不能 ship？取决于 policy — 分别写出两条路径。
3. 设计一个 end-to-end 低于 60 秒的 rollback。列出所需 infrastructure。
4. Non-determinism 在你的 eval 上显示为 ±7%。设置 canary gates，避免 false alarm。你会用什么 multipliers？
5. Shadow mode 在 canary 前抓住了 40% cost spike。写出会在 shadow 阶段触发的 alert rule。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Shadow mode | “duplicate to new” | 零影响地发送给 candidate 并记录。 |
| Canary | “progressive traffic” | 带 gates 的渐进式、用户可见 rollout。 |
| Gates | “rollout checks” | 阻止 progression 的 metric thresholds。 |
| Non-determinism | “LLM variance” | 不可消除的 run-to-run differences。 |
| Policy flag | “flag flip rollback” | Config-level rollback，秒级而不是小时级。 |
| Model pin | “registry digest” | 指向某个 model version 的 immutable reference。 |
| Argo Rollouts | “K8s progressive” | Kubernetes-native canary/rollback controller。 |
| KServe | “inference K8s” | 带 canary primitives 的 model serving。 |
| Istio weighted | “mesh split” | Service-mesh traffic splitter。 |

## 延伸阅读

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
