# Capstone 07 — 端到端微调流水线（Data 到 SFT 到 DPO 到 Serve）

> 一个 8B model，用你自己的 data 训练，用你自己的 preferences 做 DPO alignment，quantized，speculative-decoded，并以可测量的 $/1M tokens served。2026 年 open stack 是 Axolotl v0.8、TRL 0.15、用于 iteration 的 Unsloth、用于 quantization 的 GPTQ/AWQ/GGUF，以及带 EAGLE-3 serving 的 vLLM 0.7。本 capstone 的目标是 reproducibly 跑完整条 pipeline——YAML in，served endpoint out——并按 2026 Model Openness Framework 发布 model card。

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 hours

## 问题

到 2026 年，每个严肃的 AI team 都会随时保有一条 fine-tuning pipeline。不是因为他们要发布 frontier base model，而是因为 downstream adaptation——domain SFT、基于 labeled preferences 的 DPO、用于 speculative decoding 的 distilled drafts、用 EAGLE-3 serving——才是 measurable wins 所在。Axolotl v0.8 处理 multi-GPU SFT configs。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让 single-GPU iteration 变快。vLLM 0.7 with EAGLE-3 在不损失质量的情况下把 decode throughput 推高 2-3x。Tooling 已经可用；craft 在 YAMLs、data hygiene 和 eval discipline 里。

你会把一个 8B base（Llama 3.3、Qwen3 或 Gemma 3）用 task-specific data 跑过 SFT 再跑 DPO，quantize 后用于 serving，并用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 测量收益。你会按 2026 Model Openness Framework 产出 model card。重点是 reproducibility：一个命令可以端到端重跑完整 pipeline。

## 概念

Pipeline 有五个阶段。**Data**：dedup（MinHash / Datatrove）、quality filter（Nemotron-CC style classifier）、PII scrub、针对 public benchmark contamination 的 split-hygiene check。**SFT**：Axolotl YAML、8xH100 上的 ZeRO-3、cosine schedule、packed sequences、2-3 epochs。**DPO or GRPO**：TRL config、1 epoch、preference pairs 可以来自 human-labeled 或 model-judged、beta tuning。**Quantize**：GPTQ + AWQ + GGUF，用于 deployment flexibility。**Serve**：vLLM 0.7 with EAGLE-3 speculative heads（或 SGLang with SpecForge）、K8s deployment、基于 queue-wait 的 HPA。

Ablations 是 deliverable：在三个 task-specific benchmarks 上比较 SFT-only vs SFT+DPO vs SFT+GRPO。Serving metrics：batch 1 / 8 / 32 的 tokens/s、EAGLE-3 acceptance rate、$/1M tokens。Safety eval：Llama Guard 4 pass rate。Model card：bias evaluations、reproducibility seeds、data licensing。

## 架构

```text
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- Data：Datatrove 用于 dedup，Nemotron-CC classifier 用于 quality，Presidio 用于 PII
- Base：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8 with ZeRO-3、Flash Attention 3、packed sequences
- Preference tuning：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于 single-GPU iteration
- Quantization：GPTQ（Marlin）、AWQ、通过 llama.cpp 生成 GGUF
- Serving：vLLM 0.7 with EAGLE-3 speculative decoding（或 SGLang 0.4 + SpecForge）
- Eval：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- Safety eval：Llama Guard 4、ShieldGemma-2
- Infrastructure：Kubernetes + NVIDIA device plugin，基于 queue-wait metric 的 HPA
- Observability：Training 用 W&B，inference 用 Langfuse

## Build It

1. **Data pipeline。** 对 raw corpus 运行 Datatrove dedup。应用 Nemotron-CC-style quality classifier。用 Presidio scrub PII。用显式 seed 写出 train/val splits。

2. **Contamination check。** 对每个 validation split，与 MMLU-Pro、MT-Bench-v2、RewardBench-2 test sets 计算 MinHash。拒绝任何 overlap。

3. **Axolotl SFT。** YAML 包含 ZeRO-3、FA3、sequence packing。在 8xH100 上训练 2-3 epochs。Log 到 W&B。

4. **TRL DPO / GRPO。** 取 SFT checkpoint，在 preference pairs 上跑一轮 DPO（或在 math/code 上用 verifiable reward 跑 GRPO）。Sweep beta。

5. **Quantize。** 产出三种 quants：GPTQ-INT4-Marlin、AWQ-INT4、用于 llama.cpp 的 GGUF-Q4_K_M。记录 size 和 nominal throughput。

6. **Serve with speculative decoding。** 配置 vLLM 0.7，并使用通过 Red Hat Speculators 训练的 EAGLE-3 draft heads。测量 batch 1 / 8 / 32 下的 acceptance rate 和 tail latency。报告与 Anthropic / OpenAI 在同一 eval 上的 $/1M tokens 对比。

7. **Eval matrix。** 在 base、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。产出表格。

8. **Safety eval。** 在 dev set 上计算 Llama Guard 4 pass rate。使用 ShieldGemma-2 output filter。

9. **Model card。** MOF 2026 template：data、training、eval、safety、license、reproducibility section，其中包含 YAMLs 和 commit SHAs。

## Use It

```console
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Ship It

`outputs/skill-finetuning-pipeline.md` 描述 deliverable。一个命令让 data 依次经过 SFT、DPO、quant、serve、eval，并 emit model card + served endpoint。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Eval delta vs base | 在 target tasks（MMLU-Pro、MT-Bench-v2、task-specific）上测 gain |
| 20 | Pipeline reproducibility | 一个命令用相同 seeds 端到端重跑 |
| 20 | Data hygiene | Dedup rate、PII scrub coverage、contamination check green |
| 20 | Serving efficiency | bs=1/8/32 下的 tokens/s、EAGLE-3 acceptance rate、$/1M tokens |
| 15 | Model card + safety eval | 2026 MOF completeness + Llama Guard 4 pass rate |
| **100** | | |

## 练习

1. 在同一个 task-specific benchmark 上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种 preference method 胜出，以及胜出多少。

2. 把 Llama 3.3 8B 换成 Qwen3 14B。在 matched quality 下测量 $/1M tokens。

3. 测量 domain data vs generic ShareGPT 上的 EAGLE-3 acceptance rate。报告 delta，并说明它对 latency budgets 意味着什么。

4. 注入 1% contamination（把 MMLU-Pro answers 泄漏进 training data）并重新运行 eval。观察 MMLU-Pro accuracy 不真实地跳升。构建一个能捕获它的 contamination-check CI gate。

5. 添加 LoRA SFT 作为 full fine-tune 的替代方案。在 memory 低 10x 的情况下测量 quality gap。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Axolotl | “SFT trainer” | 用 YAML 驱动的统一 trainer，支持 SFT、DPO 和 distillation |
| TRL | “Preference tuner” | Hugging Face 用于 LLMs 上 DPO、GRPO、PPO 的 library |
| GRPO | “Group-relative policy optimization” | DeepSeek R1 使用的 RL recipe，配合 verifiable rewards |
| EAGLE-3 | “Speculative decoding draft” | 可提前预测 N tokens 的 draft heads；vLLM 用 target model 验证 |
| MOF | “Model Openness Framework” | 2026 年用于按 data、code、license 评估 model releases 的标准 |
| Contamination check | “Split hygiene” | 基于 MinHash 检测 test-set leakage 是否进入 training |
| Acceptance rate | “EAGLE / MTP metric” | Drafted tokens 中被 target model 接受的比例 |

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — reference SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO 和 GRPO reference implementations
- [Unsloth](https://github.com/unslothai/unsloth) — single-GPU iteration reference
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO methodology
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — reference serving stack
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — alternate speculative-decoding trainer
- [Model Openness Framework 2026](https://isocpp.org/) — open-release grading standard
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — canonical eval runner
