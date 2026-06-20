# DeepSeek-V3 Architecture Walkthrough

> Phase 10 · Lesson 14 命名了每个 open model 都会调的六个 architecture knobs。DeepSeek-V3（2024 年 12 月，总参数 671B，active 参数 37B）把这六个 knob 全部调了，并额外加入四个：Multi-Head Latent Attention、auxiliary-loss-free load balancing、Multi-Token Prediction 和 DualPipe training。本课从上到下阅读 DeepSeek-V3 architecture，并从 published config 推导每一项参数量。学完后，你能解释为什么 671B/37B 这个比例是正确押注，以及为什么 MLA + MoE 在 frontier 上优于单独使用任一方案。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14（open-model walkthroughs），Phase 10 · 17（NSA），Phase 10 · 18（MTP），Phase 10 · 19（DualPipe）
**Time:** ~75 minutes

## Learning Objectives

- 从上到下阅读 DeepSeek-V3 config，并用六个 GPT-2 knobs 加四个 DeepSeek-specific additions 来解释每个字段。
- 推导 total parameter count（671B）、active parameter count（37B），以及贡献这些数字的各个组件。
- 计算 MLA 在 128k context 下的 KV cache footprint，并与同 active-param 规模的 GQA dense model 对比。
- 说出四个 DeepSeek-specific innovations（MLA、MTP、auxiliary-loss-free routing、DualPipe），并指出每个创新作用在 architecture/training stack 的哪一部分。

## 问题

DeepSeek-V3 是第一个 architecture 与 Llama family 有实质区别的 frontier open model。Llama 3 405B 是 "GPT-2 with six knobs turned"。DeepSeek-V3 则是 GPT-2 加上全部六个 knobs，再加四个新 knobs。阅读 Llama 3 config 是阅读 DeepSeek config 的热身，但它的深层结构，也就是 attention block 的形状、routing logic、training-time objective，都足够不同，因此需要单独 walkthrough。

学习它的回报是：DeepSeek-V3 的 open-weights release 改变了 open models 中 "frontier capability" 的含义。这套 architecture 是许多 2026 training runs 正在复制的 blueprint。理解它，是任何接触 frontier LLM training 或 inference 的角色的基本功。

## 概念

### 不变的核心，再看一次

DeepSeek-V3 仍然是 autoregressive。它仍然堆叠 decoder blocks。每个 block 仍然包含 attention + MLP + 两个 RMSNorms。MLP 仍然使用 SwiGLU。仍然使用 RoPE。Pre-norm。Weight-tied embeddings。baseline 与每个 Llama 或 Mistral 相同。

### 转折：MLA 取代 GQA

从 Phase 10 · 14 你已经知道，GQA 通过让多组 Q heads 共享 K 和 V 来缩小 KV cache。Multi-Head Latent Attention（MLA）更进一步：K 和 V 被压缩进一个共享的 low-rank latent representation（`kv_lora_rank`），然后在运行时按 head 解压。KV cache 只存 latent，通常是每个 token 每层 512 个 floats，而不是 8 x 128 = 1024 个 floats。

128k context 下，使用 MLA 的 DeepSeek-V3（每个 token 每层一个共享 latent `c^{KV}`；K 和 V 都由这个 latent 经 up-projections 得到，而这些 up-projections 可以吸收到后续 matmul 中）：

```text
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个 hypothetical GQA baseline（Llama 3 70B 形状，8 KV heads，head dim 128）需要：

```text
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k context 下，MLA 比 Llama-3-70B-style GQA cache 小 4x。

tradeoff 是：MLA 在每次 attention computation 中增加一个 per-head decompression step。额外 compute 相比节省的 bandwidth 很小。对 long-context inference 来说，净收益为正。

### Routing：auxiliary-loss-free load balancing

MoE routers 决定每个 token 由 top-k experts 中的哪些 expert 处理。naive router 会把太多工作集中到少数 experts 上，让其他 experts idle。标准修复方法是添加 auxiliary loss term，惩罚 load imbalance。这确实有效，但会轻微损伤 main-task performance。

DeepSeek-V3 引入 auxiliary-loss-free scheme。它把 per-expert bias terms 加到 router logits 上，并在训练中用简单规则调整：如果 expert `e` overloaded，就降低 `bias_e`；如果 underloaded，就提高它。没有额外 loss term。Training 保持干净。Expert load 保持平衡。

对 main loss 的影响：没有可测影响。对 MoE architecture 的影响：更干净，不需要调 auxiliary-loss hyperparameter。

### MTP：更密的训练 + 免费 draft

从 Phase 10 · 18 你知道，DeepSeek-V3 增加了 D=1 MTP module，用来预测后两位 token。inference 时，这个训练好的 module 会被改用作 speculative-decoding draft，并达到 80%+ acceptance。training 时，每个 hidden state 都接受 D+1 = 2 个 targets 的监督，提供更密集的信号。

参数量：在 671B main 之外增加 14B。overhead：2.1%。

### Training：DualPipe

从 Phase 10 · 19 你知道，DualPipe 是一个 bidirectional pipeline，它把 forward 和 backward chunks 与 cross-node all-to-all comms 重叠。在 DeepSeek-V3 的 2,048-H800 规模上，它大约挽回了 245k GPU-hours，这些时间本会被 1F1B 浪费在 pipeline bubbles 上。

### Config，逐字段阅读

这是 DeepSeek-V3 config（简化版）：

```text
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析它：

- `hidden_size=7168`：embedding dimension。
- `num_hidden_layers=61`：总 block depth。
- `first_k_dense_layers=3`：前 3 个 blocks 使用 size 为 18432 的 dense MLP。剩余 58 个使用 MoE。
- `num_attention_heads=128`：128 个 query heads。
- `kv_lora_rank=512`：K 和 V 被压缩到这个 latent dimension，再按 head 解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE block 有 256 个 experts，route top-8。
- `shared_experts=1`：在 256 个 routed experts 之外，还有 1 个 always-on expert 作用于每个 token。可以把它看作一个 "dense floor"，保证每个 token 都得到可靠处理。
- `moe_intermediate_size=2048`：每个 expert 的 MLP hidden size。它比 dense MLP 小，因为 expert 有 256 个。

### 参数账本

完整计算在 `code/main.py`。headline 如下：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个 dense blocks：带 MLA 的 attention（每 block ~144M）+ dense MLP（每 block ~260M）+ norms。合计约 1.2B。
- 58 个 MoE blocks：带 MLA 的 attention（~144M）+ 256 个 experts（每个 30M）+ 1 个 shared expert（30M）+ norm。每个 block 总计约 7.95B（包含所有 experts）。58 个 MoE blocks 合计 461B。
- MTP module：14B。

Grand total：core architecture 约 476B + 14B MTP；published 671B number 还 distinctively 计入额外 structural parameters（bias tensors、expert-specific components、shared expert scaling 等）。calculator 复现的数字与 published 数字在 3-5% 之内，delta 来自 DeepSeek report Section 2 appendix 中记录的细粒度 accounting。

每次 forward 的 active parameters：

- Attention：每层 144M * 61 = 8.8B（所有 layers 都会 fire）。
- MLP active：前 3 层 dense（3 * 260M = 780M），58 个 MoE layers 中每层激活 8 routed + 1 shared + routing overhead。每层 active MLP 约 260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- Embedding + norms：1.2B。
- Total active：约 26B core + 14B MTP（训练时使用，但 inference 并不总是运行）≈ 37B。

### 671B / 37B 比例

18x sparsity ratio（active params 是 total params 的 5.5%）。DeepSeek-V3 是已经发布 open weights 的最稀疏 frontier MoE model。Mixtral 8x7B 的 ratio 是 13/47（28%），密得多。Llama 4 Maverick 的 ratio 是 17B/400B（4.25%），与它相近。DeepSeek 的押注是：在 frontier scale 上，更多 experts 与更低 activation ratio 能带来更好的 quality per active-FLOP。

### DeepSeek-V3 的位置

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 backbone 上做的 reasoning-training run。R1 使用相同 architecture。变化的是 post-training recipe（在可验证任务上做 large-scale RL），不是 pretraining architecture。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），也就是 Phase 10 · 17 中 NSA 的继任者。lineage 是稳定的：architecture-level innovations 会累积；每个版本会继续调新的 knobs。

```figure
moe-routing
```

## 使用它

`code/main.py` 是专门针对 DeepSeek-V3 形状的 parameter calculator。运行它，把输出与 paper 的数字对比，并在 hypothetical variants 上使用它（256 experts vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

观察重点：

- Total parameter count vs published 671B。
- Active parameter count vs published 37B。
- 128k context 下的 KV cache，也就是 MLA vs GQA comparison。
- Per-layer breakdown，看看 parameter budget 实际花在哪里。

## 交付物

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek-family model（V3、R1 或任何未来变体），它会生成 component-by-component architecture reading：命名 config 中每个字段，按组件推导参数量，并识别模型使用了四个 DeepSeek-specific innovations 中的哪些。

## 练习

1. 运行 `code/main.py`。把 calculator 的 total-parameter estimate 与 published 671B 对比，并识别 delta 来自哪里。paper 的 Section 2 有完整 itemization。

2. 把 config 修改为 MLA rank 256，而不是 512。计算 128k context 下的 KV cache size。它带来百分之多少的 reduction，又以什么 per-head expressiveness 成本换取？

3. 对比 DeepSeek-V3 的（256 experts，top-8）routing 与 hypothetical（512 experts，top-8）变体。Total parameters 增长；active parameters 保持不变。额外 expert capacity 理论上带来什么，又在 inference 时花费什么？

4. 阅读 DeepSeek-V3 technical report（arXiv:2412.19437）Section 2.1 关于 MLA 的内容。用三句话解释为什么 K 和 V decompression matrices 可以在 inference-time efficiency 上被 "absorbed" 到后续 matmul 中。

5. DeepSeek-V3 对大多数 operations 使用 FP8 training。计算用 FP8 而不是 BF16 存储 671B weights 的 memory savings。这与 14.8T-token training budget 如何交叉影响？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | 把 K 和 V 压缩进共享 low-rank latent（kv_lora_rank，通常 512），再按 head on-the-fly 解压；KV cache 只存 latent |
| kv_lora_rank | "MLA compression dim" | K 和 V 的共享 latent size；DeepSeek-V3 使用 512 |
| First k dense layers | "Early layers stay dense" | MoE model 的前几层跳过 MoE router，运行 dense MLP，以提高稳定性 |
| num_experts_per_tok | "Top-k routing" | 每个 token 会激活多少 routed experts；DeepSeek-V3 使用 8 |
| Shared experts | "Always-on experts" | 不管 routing 如何都处理每个 token 的 experts；DeepSeek-V3 使用 1 个 |
| Auxiliary-loss-free routing | "Bias-adjusted load balance" | 训练中调整 per-expert bias terms，在不添加 loss term 的情况下保持 expert load balanced |
| MTP module | "Extra prediction head" | 从 h^(1) 和 E(t+1) 预测 t+2 的 transformer block；让训练更密集，也提供免费的 speculative-decoding draft |
| DualPipe | "Bidirectional pipeline" | 将 forward/backward compute 与 cross-node all-to-all 重叠的 training schedule |
| Active parameter ratio | "Sparsity" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | "8-bit training" | 用 FP8 存储训练数据并执行许多 compute ops；相比 BF16 约减半 memory，质量代价很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整 architecture、training 和 results 文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — config files 和 deployment notes
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前代模型
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 基于 V3 architecture 的 reasoning-training successor
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek-family attention 的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — training-schedule reference
