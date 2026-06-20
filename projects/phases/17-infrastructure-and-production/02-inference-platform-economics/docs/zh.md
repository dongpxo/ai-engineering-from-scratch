# 推理平台经济学：Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场已经不再只是租 GPU 时间。它分裂成三类：定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API-first 市场（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日把每块 GPU 的价格上调 1 美元/小时；每天处理 10T+ tokens、估值 40 亿美元，说明 volume-driven 模式确实跑通了。Baseten 在 2026 年 1 月完成 3 亿美元 E 轮，估值 50 亿美元。竞争定位规则很简单：Fireworks 优化 latency，Together 优化 catalog breadth，Baseten 优化 enterprise polish，Modal 优化 Python-native DX，Replicate 优化 multimodal reach，Anyscale 优化 distributed Python。这节课给你一张可以直接交给 founder 的矩阵。

**Type:** Learn
**Languages:** Python（stdlib，玩具版 per-call economics comparator）
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 说出三个市场分段（custom silicon、GPU platforms、API-first），并把每个 vendor 映射到对应分段。
- 解释为什么 “per-token” API pricing model 会向 serving engine 的成本曲线收敛，而不是向硬件成本曲线收敛。
- 在至少三个 vendor 之间计算有效 request 成本，并说明 per-minute（Baseten、Modal）什么时候胜过 per-token。
- 判断给定 workload 的默认平台选择：serverless bursty、steady high-throughput、fine-tuned variants、multimodal。

## 问题

你已经评估过 hyperscaler 的 managed platform。现在你决定需要一个更窄、更快的 provider：Fireworks 用来压低 latency，Together 用来覆盖更宽的模型目录，Baseten 用来部署 fine-tuned custom model。问题是你现在有六个真实选择，而且它们的 pricing page 完全对不上。Fireworks 展示 $/M tokens；Baseten 展示 $/minute；Modal 展示 $/second；Replicate 展示 $/prediction。如果不建模 workload，你没法把它们正面对比。

更麻烦的是，每个 pricing page 背后的商业模型也不一样。Fireworks 在共享 GPU 上跑自己的 custom engine（FireAttention）；per-token rate 反映的是它们的利用率曲线。Baseten 给你 Truss + dedicated GPUs；per-minute 反映的是独占性。Modal 是真正的 Python serverless：per-second billing，冷启动低于数秒。同样输出一个 LLM response，却对应三种不同的成本函数。

这节课会建模这六个平台，并告诉你什么时候谁会赢。

## 概念

### 三个分段

**Custom silicon。** Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在同一个模型上，相比 GPU-based cluster，decode 通常快 5-10 倍。per-token 价格更高（2025 年末 Groq 在 Llama-70B 上约 $0.99/M），但在 latency-sensitive use case 中很难被击败。Groq 是 voice agent 和 real-time translation 的 production 选择。

**GPU platforms。** Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年的 H100、H200、B200）或偶尔 AMD 上。它们位于 “raw GPU rental”（RunPod、Lambda）和 “hyperscaler managed service”（Bedrock）之间，是中间经济层。

**API-first marketplaces。** Replicate、DeepInfra、OpenRouter、Fal。模型目录宽，按 prediction 或 second 付费，强调 time-to-first-call。

### Fireworks：latency-optimized GPU platform

- FireAttention engine（custom）；宣传口径是在等价配置上比 vLLM latency 低 4 倍。
- Batch tier 对非交互 workload 大约是 serverless rate 的 50%。
- Fine-tuned model 与 base model 同价服务。这相对于那些对你的 LoRA 收 premium 的 provider，是一个真实差异点。
- 2026 年中：on-demand GPU rental 从 2026 年 5 月 1 日起上调 1 美元/小时。大规模用量可谈 volume pricing。
- 财务信号：40 亿美元估值，每天处理 10T+ tokens。

### Together：breadth-optimized

- 200+ models，包括 upstream 发布后几天内上线的 open-source release。
- 在等价 LLM model 上比 Replicate 便宜 50-70%。“AI Native Cloud” 的定位靠的是 volume 和 catalog。
- Inference + fine-tuning + training 统一在一个 API 中。

### Baseten：enterprise-polish-optimized

- Truss framework：把 model packaging、dependencies、secrets、serving config 放进一个 manifest。
- GPU 覆盖从 T4 到 B200。per-minute billing，并带有合理的 cold-start mitigation。
- SOC 2 Type II，HIPAA-ready。常见于 fintech 和 healthcare 场景。
- 50 亿美元估值，2026 年 1 月 E 轮（CapitalG、IVP、NVIDIA 参与，3 亿美元）。

### Modal：Python-native-optimized

- 用纯 Python 写 Infrastructure-as-code。给函数加上 `@modal.function(gpu="A100")`，一条命令部署。
- Per-second billing。预热后冷启动 2-4 秒；小模型低于 1 秒。
- 2025 年以 11 亿美元估值完成 8700 万美元 B 轮。在独立调研中 developer experience 得分最高。

### Replicate：multimodal breadth

- Pay-per-prediction。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS plugins）。
- 在 LLM per-token rate 上竞争力较弱，但胜在 multimodal variety。

### Anyscale：Ray-native

- 基于 Ray；RayTurbo 是 Anyscale 的 proprietary inference engine（与 vLLM 竞争）。
- 最适合 distributed Python workload，其中 inference step 只是更大计算图里的一个节点。
- Managed Ray clusters；与 Ray AIR 和 Ray Serve 紧密集成。

### Per-token 与 per-minute：什么时候谁赢

当 workload latency-insensitive 且 bursty 时，per-token 合理：你只为实际使用量付费。当 utilization 高且可预测时，per-minute 合理：一旦你能把 GPU 跑满，就会打败 per-token。

粗略规则：如果 workload 超过 dedicated GPU 约 30% 的 sustained utilization，per-minute（Baseten、Modal）开始胜过 per-token（Fireworks、Together）。低于这个水平，per-token 更划算，因为你不用为 idle 付费。

### Custom engine 才是真 moat

每个高于 vLLM 和 SGLang 的平台都会声称自己有 custom engine：FireAttention、RayTurbo、Baseten 的 inference stack。Custom-engine claim 多少带有 marketing shade。更诚实的说法是：vLLM + SGLang 代表了生产级开源推理的大约 80%；平台层的差异化更多来自 DX、attribution 和 SLA。

### 你应该记住的数字

- Fireworks GPU rental：从 2026 年 5 月 1 日起上调 1 美元/小时。
- Fireworks claim：在等价配置上 latency 比 vLLM 低 4 倍。
- Together：LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：50 亿美元（2026 年 1 月 E 轮，3 亿美元 round）。
- Modal 估值：11 亿美元（2025 年 B 轮）。
- Per-minute 在约 30% 以上 sustained utilization 时胜过 per-token。

```figure
cost-per-token
```

## Use It

`code/main.py` 会在一个 synthetic workload 上比较六个 vendor 的 pricing model，报告 $/day 和有效 $/M tokens。运行它，找到 per-token 与 per-minute 之间的 break-even。

## Ship It

本课产出 `outputs/skill-inference-platform-picker.md`。给定 workload profile、SLA 和 budget，它会选择 primary inference platform，并指出 runner-up。

## 练习

1. 运行 `code/main.py`。对于一块 H100 上的 70B model，Baseten（per-minute）在什么 sustained utilization 下会胜过 Fireworks（per-token）？自己推导 crossover，并与经验法则比较。
2. 你的产品同时提供 image generation、chat 和 speech-to-text。为每个 modality 选择平台，并说出把它们统一起来的 gateway pattern。
3. Fireworks 把你的 primary model 价格上调 1 美元/小时。如果 40% traffic 转到 batch tier（50% off），请建模 blended cost impact。
4. 一个 regulated customer 要求 SOC 2 Type II + HIPAA + dedicated GPUs。哪三个平台可行？其中哪一个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together on-demand、Baseten dedicated 和 Replicate API 上每 1,000 predictions 的成本。10 predictions/day 时谁最便宜？10,000 时呢？

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Custom silicon | “non-GPU chips” | Groq LPU、Cerebras WSE、SambaNova RDU，针对 decode 优化。 |
| FireAttention | “Fireworks engine” | Custom attention kernel；宣传口径是 latency 比 vLLM 低 4 倍。 |
| Truss | “Baseten's format” | Model packaging manifest；dependencies + secrets + serving config。 |
| Per-token | “API pricing” | 按消耗 token 计费；不为 idle 付费。 |
| Per-minute | “dedicated pricing” | 按 wall-clock GPU time 计费；在高 utilization 下胜出。 |
| Per-prediction | “Replicate pricing” | 按 model invocation 计费；图像/视频中常见。 |
| RayTurbo | “Anyscale engine” | Ray 上的 proprietary inference；在 Ray clusters 上与 vLLM 竞争。 |
| Batch tier | “50% off” | 降价的非交互队列；Fireworks、OpenAI 都常见。 |
| Fine-tuned at base rate | “Fireworks LoRA” | LoRA-served requests 按 base model rate 计费（差异点）。 |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — per-token rates、batch tier、GPU rental。
- [Baseten Pricing](https://www.baseten.co/pricing/) — per-minute rates、committed capacity、enterprise tiers。
- [Modal Pricing](https://modal.com/pricing) — per-second GPU rates 和 free tier。
- [Together AI Pricing](https://www.together.ai/pricing) — model catalog 和 per-token rates。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 和 managed Ray pricing。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — comparative assessment。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — vendor landscape。
