# Multi-Token Prediction（MTP）

> 从 GPT-2 到 Llama 3，每个 autoregressive LLM 都在每个位置上训练一个 loss：预测下一个 token。DeepSeek-V3 在每个位置上增加了第二个 loss：预测再下一个 token。在 671B 模型上额外加入的 14B 参数，通过 gradient flow 被 distill 回主模型；训练好的 MTP heads 又在 inference 时被改用作 speculative-decoding drafters，acceptance 超过 80%。1.8× generation throughput 几乎白送。这一课构建 DeepSeek technical report 中的 sequential MTP module，计算 loss 和 shared-head parameter layout，并解释为什么 MTP 保持 causal chain，而 Gloeckle et al. 最初的 parallel MTP 打破了它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标

- 陈述 MTP training objective，并推导跨 prediction depths 的 joint loss
- 解释 Gloeckle et al. 的 parallel MTP heads（2024）与 DeepSeek-V3 的 sequential MTP modules 之间的差异，以及为什么 sequential design 能保留 causal chain
- 计算在 pre-training run 中加入 MTP modules 带来的 parameter 和 memory overhead
- 从零实现一个 MTP module：shared embedding、per-depth transformer block、projection 和 shared output head

## 问题

Next-token prediction 是标准 LLM training objective。每个 hidden state 都被监督去预测一件事：紧随其后的 token。这是一个出人意料地弱的信号。Sequence 中的大部分信息会跨越不止一个 token：structure、coherence、factuality、arithmetic flow。模型必须通过 trillions of tokens 上的许多 one-token signals 积累学到这些。

MTP 问的是：如果每个 hidden state 同时被监督去预测多个 future tokens，会怎样？Gloeckle et al.（Meta, 2024）证明这有帮助。他们的实现是在 backbone 顶部放几个独立 output heads，每个 head 预测不同 offset。并行、简单，但这些 heads 看到的是同一个 hidden state，没有任何 hierarchical refinement；而且 predictions 不形成 causal chain，所以无法用于 speculative decoding。

DeepSeek-V3（2024 年 12 月）把 MTP 重新设计成 sequential modules，在每个 prediction depth 上都保留 causal chain。模型从 `h_i^(0)` 预测 `t+1`，然后从新的 hidden state `h_i^(1)` 预测 `t+2`，这个新 hidden state 结合了 `h_i^(0)` 与 `E(t+1)` embedding，依此类推。每个 depth 都有自己的小 transformer block。Shared embedding 和 shared output head 让 parameter overhead 保持适中。在 DeepSeek-V3 的规模上，MTP modules 在 671B main-model weights 之外额外增加 14B parameters。这个 2% overhead 换来了更密集的 training signals，并且 inference 时得到一个现成的 speculative-decoding draft。

这一课从零构建一个单独的 MTP module 和 D-depth loss。数学很整洁。实现只有 150 行。

## 概念

### Sequential MTP recipe

DeepSeek-V3 在主模型之上添加 `D` 个 MTP modules。每个 module `k`（`k = 1..D`）预测 depth `k` 处的 token，也就是在给定到位置 `i` 为止的 prefix 时预测 `t_{i+k}`。

Module `k` 包含：

- 一个 transformer block `T_k`，带有自己的 attention 和 MLP。
- 一个 projection matrix `M_k`，用于结合 previous-depth hidden state 与 next-depth ground-truth token 的 embedding。
- Shared embedding `E`（与主模型相同）。
- Shared output head `Out`（与主模型相同）。

训练时，对于到位置 `i` 为止的 prefix，per-depth hidden state 是：

```text
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Per-depth prediction 是：

```text
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Per-depth loss 是对 ground-truth `t_{i+k}` 的 cross-entropy：

```text
L_k = CE(logits_{i+k}, t_{i+k})
```

跨 depths 的 joint loss：

```text
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个小 weighting factor。DeepSeek-V3 在训练前 10% 使用 0.3，之后使用 0.1。总训练 loss 是 `L_main + L_MTP`。

### 为什么是 sequential，而不是 parallel

Gloeckle 最初的 parallel MTP 有 D 个 output heads，每个 head 都直接应用到 `h_i^(0)`。每个 head 从同一个 backbone hidden state 预测 `t_{i+k}`。这种方式可以训练，但 predictions 之间没有互相 conditioning。你不能用 `head_1` 的输出来帮助 `head_2`，因为 heads 是并行触发的。

DeepSeek-V3 的 sequential design 从 `h_i^(k-1)` 加上真实 next-token embedding `E(t_{i+k})` 构建 `h_i^(k)`。这保留了 causal chain：为了预测 `t_{i+k+1}`，depth `k+1` 的 module 会看到 `t_{i+k}` 位置的内容。这在结构上等同于 autoregressive decoder 消费自己的输出，因此 MTP modules 可以直接作为 speculative-decoding drafters。

Inference 时：把 `h_i^(k-1)` 和 drafted `t_{i+k}` 喂给 module `k+1`，得到对 `t_{i+k+1}` 的预测。重复。这正是 EAGLE-style draft，只是把训练好的 MTP module 当成 draft network。DeepSeek-V3 报告第一个 MTP module 的 acceptance 超过 80%，speedup 约 1.8×。

### Parameter accounting

对 hidden size 为 `h`、vocabulary size 为 `V` 的模型：

- Main model：数十亿参数，加上一个大小为 `V * h` 的 output head。
- Shared output head：复用主模型 head。没有额外参数。
- Shared embedding：复用主模型 embedding。没有额外参数。
- 每个 MTP module：
  - Projection `M_k`：`(2h) * h = 2h^2`。
  - Transformer block `T_k`：attention（MHA 为 `4h^2`）加上 MLP（SwiGLU ratio 8/3 时通常为 `8h^2`）。每个 block 大约 `12h^2`。

每个 module 的总额外参数：`~14h^2`。对 DeepSeek-V3 的 `h = 7168`，D = 1 module 时，纸面上是 `~14 * 7168^2 = ~720M` parameters。DeepSeek-V3 报告 14B，差异主要来自 MTP module 里的 expert layers 也采用了 MoE。

### Speculative-decoding payoff

Pre-training 期间，MTP modules 会让训练慢约 10%（更多 forward compute，额外 loss）。回报有两层：

1. 更密集的 training signal。每个 hidden state 看到 D+1 个 supervision targets。DeepSeek-V3 ablations 中，MMLU、GSM8K、MATH、HumanEval 都有稳定的几个百分点提升。

2. Inference 时得到免费的 speculative decoding draft。MTP module 已经被训练来预测接下来的几个 tokens。把它改用作 draft network，可以获得 80%+ acceptance rates。在这个水平上，N=3 或 N=5 的 spec decoding 会带来 1.8× throughput。10% 的 training-time cost 会在你第一次大规模 inference 时回本。

### 与 EAGLE 的关系

EAGLE 在 pre-training 之后单独训练一个小 draft model。MTP 把 draft 烘进 pre-training。两种方法的 accept rates 接近，但 pipeline 不同：

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No（需要重新训练） |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | 只有 speculative decoding | 更密集 training signal + speedup |

## Build It

`code/main.py` 端到端构建一个 MTP module：shared embedding、projection、transformer block、shared output head。然后它会在一个短 synthetic sequence 上计算 per-depth cross-entropy loss，并按 component 打印 parameter count。Toy vocabulary 只有 32 个 tokens，让数字更容易读。

### Step 1：shared embedding table

一个 `vocab_size x hidden` table 同时被主模型和每个 depth 的每个 MTP module 使用。不是第二份副本，而是字面意义上的同一个 tensor。

### Step 2：per-depth combination

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真实 DeepSeek-V3 会把两个 RMSNormed vectors concat 成 `[2h]`，再用一个 `h x 2h` matrix 投影。Toy 版本为了保持 stdlib 简洁，用 vector addition 代替。

### Step 3：depth k 的 transformer block

Self-attention 加 MLP。在 toy 中，一个 one-layer linear attention block 和一个 SwiGLU MLP 让结构可见，同时避免引入 numpy。

### Step 4：shared output head

复用主模型 output projection。得到 vocabulary 上的 logits。

### Step 5：per-depth loss

对 softmax(logits) 与 offset `k` 处的 ground-truth token 做 cross-entropy。用 `lambda / D` scaling factor 跨 depths 聚合。

### Step 6：parameter accounting

打印 total parameter count、shared（embedding、head）count，以及 per-module extra count。展示 MTP extra 与 main-model size 的比例。

## Use It

MTP 已集成到 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列中。Inference 时：

- DeepSeek 自己的 serving stack 会开箱即用地把 MTP modules 当作 speculative decoders。
- 截至 2026 年 4 月，vLLM 和 SGLang 已有 DeepSeek-V3 MTP 的 integration paths。
- AMD 的 ROCm SGLang tutorial 展示了一个具体的 MTP speculative-decoding config，并在 V3 checkpoint 上测得 1.8× speedup。

什么时候在新的 pre-training run 中使用 MTP：

- 你控制完整 pre-training pipeline，并希望存下更密集的 training signal。
- 你知道自己会大规模 serve 这个模型，并想免费得到 speculative decoding。
- 你的 hidden size 至少是 4096。在 1B 规模上，overhead 带来的伤害通常大于收益。

什么时候不使用：

- Fine-tuning 一个已有 pre-trained dense model。MTP module 没有被训练过。
- 你想要 clean baseline 做对比的 research models。MTP 会改变 architecture。

## Ship It

这一课产出 `outputs/skill-mtp-planner.md`。给定一个 pre-training run specification（model size、data、compute），它会返回一个集成 MTP 的计划：depth 数量 D、`lambda` schedule、memory overhead，以及 inference-time speculative-decoding wiring。

## 练习

1. 运行 `code/main.py`。展示随着 synthetic signal 增强，per-depth loss 会单调下降。把 synthetic 改成 fixed pattern，并验证 depth-1 和 depth-2 losses 都会 converge。

2. 计算一个 dense 70B model（hidden 8192，80 layers）加 D=1 MTP module 的 parameter overhead。与 DeepSeek-V3 报告的 14B overhead 比较。解释为什么 DeepSeek 的数字更高：MTP transformer block 继承同样的 MoE 结构，放大了 per-module parameter count。

3. 在 toy 中实现 D=2：添加第二个 MTP module，它接收 h^(1) 并预测 `t_{i+2}`。验证 joint loss 和 parameter accounting 与 DeepSeek paper 的 equations 19-21 一致。

4. 把 toy 切换为 parallel MTP（Gloeckle-style）：在 main hidden state 之上添加 D 个 output heads，每个预测不同 offset。在同一个 synthetic signal 上比较每个 depth 的 losses 与 sequential version。对 k > 1，sequential version 的 depth-k loss 应该更低，因为它会 condition on intermediate predictions。

5. 把训练好的 MTP module 当作 EAGLE-style draft：inference 时调用 module k 来提出 `t_{i+k}`。在 held-out sequence 上，用主模型 predictions 作为 verifier，测量这些 draft tokens 的 acceptance rate。如果 toy 上达到 50%+，你就复现了 MTP-as-draft 的经验性质。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| MTP module | "Extra loss block" | 一个小 transformer block 加 projection，用于预测主模型当前位置之后 `k` 个位置的 token |
| Prediction depth | "Which offset" | 整数 `k`，表示 module `k` 从到位置 `i` 为止的 prefix 预测 `t_{i+k}` |
| Parallel MTP | "Gloeckle-style" | 在同一个 backbone hidden state 上放 D 个独立 heads，没有 conditional chain |
| Sequential MTP | "DeepSeek-V3 style" | 每个 module 都 condition on previous depth 的 hidden state 加 next token embedding，保留 causal chain |
| Shared output head | "Reuse the main head" | MTP modules 调用主模型 LM head，而不是单独的 output projection |
| Shared embedding | "Reuse the main table" | 同一个 vocabulary embedding table 到处复用，没有 duplicate parameters |
| Projection matrix M_k | "Combine hidden + next-token" | 一个 `h x 2h` linear layer，把 previous hidden state 和 target-token embedding 折叠成 next depth 的输入 |
| Joint loss L_MTP | "Averaged extra losses" | Per-depth cross-entropy losses 的 arithmetic mean，再乘以 `lambda` |
| Acceptance rate at depth 1 | "MTP draft 对的频率" | D=1 MTP module 的 top-1 prediction 等于主模型 top-1 prediction 的比例；DeepSeek-V3 上超过 80% |
| Lambda weighting | "Extra-loss importance" | Per-depth scaling factor；DeepSeek-V3 训练开始时为 0.3，之后为 0.1 |

## 延伸阅读

- [DeepSeek-AI - DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) - 完整 sequential MTP 描述（Section 2.2），包括 joint-loss equations 和 inference 时的 1.8× speedup。
- [Gloeckle et al. - Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) - DeepSeek 设计所改进的 parallel MTP baseline。
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) - 685B total（671B main + 14B MTP）和 deployment notes。
- [Leviathan et al. - Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) - MTP 所属的 speculative-decoding framework。
- [Li et al. - EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) - EAGLE 的 2025 draft architecture，也是 MTP 竞争的 counterpart。
