# Speculative Decoding 和 EAGLE-3

> Phase 7 · Lesson 16 已经证明了数学：Leviathan rejection rule 会精确保留 verifier 的分布。这一课从 training stack 的角度看 2026 年生产级 speculative decoding。EAGLE-3 把 draft model 从一个廉价近似，变成了一个专门训练的小网络：它在 verifier 自己的 hidden states 上训练，并加入 training-time test loop，让训练分布与推理分布对齐。结果：端到端 3× 到 6.5× 加速，chat 场景的 per-token acceptance rate 超过 0.9，并且没有 distributional tradeoff。2026 年每个 production inference stack 都默认启用它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## 学习目标

- 用一句话陈述 Leviathan theorem，并证明 speculative loop 产生的 samples 与 verifier 的分布完全一致
- 梳理 vanilla spec-decoding（Leviathan 2023）到 EAGLE、EAGLE-2、EAGLE-3 两年间的演进，并说出每一步消除了哪个具体限制
- 根据 acceptance rate `α` 和 draft-to-verifier cost ratio `c` 计算 expected speedup，并为每种 regime 选择最佳 draft length `N`
- 从零实现完整 speculative loop：draft、verify、从 residual 中 reject-sample、在 rejection 时回滚 KV cache、在 full acceptance 时发出 bonus token

## 问题

在 H100 上，一个 70B 模型做 autoregressive decoding 可能只有每秒 35 tokens。GPU 远没有被打满。Memory bandwidth 才是上限：每生成一个 token，都要从 HBM 读取 70B 权重，做一步算术，然后产出一个 float。Compute units 大多处于空闲状态。

Speculative decoding 把这个问题变成了一个你真的可以解决的 throughput problem。廉价 draft 用 `N` 次小 forward passes 提出 `N` 个 tokens。Verifier 在 prefix 加上全部 `N` 个 drafts 上运行一次。如果 verifier 在位置 `i` 的分布与 draft 一致（用一个我们马上会精确定义的统计意义），就 accept；否则 reject，并从 residual distribution 中采样一个 correction。一次大模型 forward 最多产出 `N+1` 个 accepted tokens，而不是一个。

关键定理来自 Leviathan, Kalman, Matias（ICML 2023）：输出分布与直接从 verifier sampling 得到的分布完全一致。不是近似一致。是完全一致。这就是 speculative decoding 能进 production 的全部原因：它是纯 latency optimization，没有质量 tradeoff。

Phase 7 · Lesson 16 给你的是数学。这一课给你的是 training stack。好的 draft 比便宜的 draft 多带来 2× speedup。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024-2025）把 "draft = 同一模型家族的小版本" 变成了精确的工程学科。2026 年 production inference servers 默认使用 EAGLE-3。

## 概念

### Invariant：Leviathan rejection sampling

令 `p(t)` 表示 draft 在某个 prefix 下对 next token 的分布，`q(t)` 表示 verifier 的分布。采样一个 draft token `d ~ p`。以 `min(1, q(d) / p(d))` 的概率 accept。若 reject，则从 residual distribution `(q - p)_+ / ||(q - p)_+||_1` 中采样。最终 samples 会按 `q` 分布。这与 `p` 有多差无关：`p` 越差，reject 越频繁，但输出仍然精确。

把 `N` 次这样的调用首尾相接，并用一次 verifier forward 处理 `prefix + d_1 + ... + d_N`。Verifier 同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右扫描。第一次在位置 `j` reject 时，从 `residual(q_j, p_j)` 采样并停止。若全部 accept，则从 `q_{N+1}` 采样一个 bonus token。

### 什么决定 speedup

令 `α` 为每个 drafted token 的 expected acceptance rate。令 `c = cost(draft) / cost(verifier)` 为 cost ratio。每次 verifier forward 的 expected accepted tokens 数量为：

```text
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个 accepted token 的 expected total wall time 是 `(N * c + 1) / E[accepted]`。对 `N` 最小化这个式子，就能得到 sweet spot。对于 `α = 0.8, c = 0.05`：最佳 `N` 大约是 5-7，speedup 为 3.2×。对于 `α = 0.95, c = 0.02`：最佳 `N` 大约是 8-10，speedup 接近 5×。

最大的杠杆是 `α`。在固定 `N = 5` 下，从 `α = 0.6`（vanilla draft）提升到 `α = 0.9`（EAGLE-3），每次 verifier forward 的 expected accepted tokens 会从 2.2 提升到 4.1。同一个 verifier 获得了接近 2× 的 throughput。

### 两年演进

**Vanilla speculative（Leviathan, 2023）。** Draft model 是同一 family 中独立训练的小 LLM。接线简单，`α ≈ 0.6`，speedup 最多约 2×。

**EAGLE-1（Li et al., 2024）。** Draft 是一个 tiny transformer，通常一到两层，输入是 verifier 的 last-layer hidden state，并直接预测 next token。因为 draft 能看到 verifier 的 feature representation，它的分布更接近 verifier。`α` 提升到 0.7-0.8。

**EAGLE-2（Li et al., 2024）。** 加入 dynamic draft tree：不再提出单条 `N` tokens 序列，而是提出一个小 candidate tree，用 verifier 在一次 forward pass（tree attention）中为每个 candidate 打分，并沿着最高概率 path 前进。Draft length 变成每一步自适应。Accepted-path token 的 `α` 超过 0.85。

**EAGLE-3（Li et al., 2025, NeurIPS）。** 又做了两个改变。第一，完全去掉 feature-prediction loss。EAGLE-1/2 会训练 draft 去匹配 verifier hidden states，这会限制数据规模带来的收益。EAGLE-3 直接在 token prediction 上训练。第二，training-time test（TTT）：draft 训练期间，把 draft 自己之前的 predictions 作为输入喂回去多步运行，与 inference 时的操作方式一致。这会对齐 train 和 test distributions，并阻止 error accumulation。测得 speedup：chat 场景最高 6.5×，SGLang 在 H100 batch 64 上吞吐提升 38%。

### KV cache rollback

Verification 会在一次 pass 中把 verifier 的 KV cache 扩展 `N` 个 entries。如果 rejection 发生在位置 `j`，位置 `j-1` 之后的 cache contents 就都错了。常见实现有两种：写入 scratch buffer，accept 后再 commit（vLLM、TensorRT-LLM）；或者维护一个 physical KV cache 和一个 logical length，在 reject 时 truncate。无论哪种，rollback cost 都只是每 layer、每 head 的若干 bytes，相比 forward-pass cost 可以忽略。

对 EAGLE-2 tree search 来说，verifier 使用一个尊重 tree topology 的 non-causal mask 运行 attention。工程细节比较繁琐，但计算本身就是一次带 custom mask 的标准 flash-attention call。

### 2026 年的 Draft architectures

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3× | None（复用现有小模型） |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3× | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4× | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokens |
| Lookahead | No draft（Jacobi iteration） | N/A | 1.3-1.6× | None |

2026 年 production 中：vLLM 和 SGLang 在可用时默认使用 EAGLE-3，否则使用 EAGLE-2。TensorRT-LLM 对 Meta 和 NVIDIA public models 有最快的 Medusa path。llama.cpp 为 CPU deployments 提供 vanilla draft。

## Build It

见 `code/main.py`。这是完整的 Leviathan speculative loop，包含所有组件：draft-of-N、verifier parallel pass、per-position rejection、residual sampling、bonus token、KV rollback，以及经验验证输出分布与直接从 `q` 采样一致。

### Step 1：rejection rule

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Step 2：residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Step 3：一个完整 speculative step

`spec_step` 函数从 `p` 中 draft `N` 个 tokens，然后用一次 parallel `q` evaluation 验证所有 tokens。它会对每个 drafted token 应用 rejection rule，并在第一次 rejection 时从 residual 中采样 correction。如果全部 accept，则从 `q_{N+1}` 发出一个 bonus token。

### Step 4：KV rollback bookkeeping

Simulator 会为每个 worker 跟踪一个 logical `kv_length`。Accept `k` 个 drafts 时，`kv_length += k`。如果在位置 `j` reject，cache 已经写过了 `j`，但 logical length 会被设置为 `prefix_length + j + 1`，也就是 correction token 之后一位。后续读取会 truncate 到 logical length。

### Step 5：Leviathan check

运行 50,000 次 speculative steps。统计 accepted tokens 的 empirical distribution。与 50,000 次从 `q` 直接采样的结果比较。Chi-square statistic 应该显著低于 critical value。这个 theorem 在实践中也会通过。

### Step 6：speedup vs. α

用不同 amplitude 把 `p` 从 `q` 扰开，以此 sweep draft quality。测量 `α`，然后绘制 expected tokens per verifier call 关于 `α` 和 `N` 的函数。代码会打印一个表，展示 EAGLE-3-class draft quality（`α ≈ 0.9`）如何解锁每次 verifier call 4-5 个 tokens。

## Use It

带 EAGLE-3 的 production-level `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

根据 EAGLE-3 paper，SGLang 在 H100 上使用 EAGLE-3、batch 64 时，相比 batch-64 vanilla decoding 吞吐约提升 1.38×。

什么时候使用 speculative decoding：

- 任何 p50 latency 比 peak throughput 更重要的 interactive chat workload。
- Code generation 和 structured output（JSON、SQL）。`α` 会超过 0.9，因为目标分布高度可预测。
- Long-form generation（数千 tokens）。Amortized speedup 会持续发挥作用。

什么时候不使用：

- 很小的模型（< 3B）。Draft 并没有比 verifier 便宜太多。
- Tiny batch-1 CPU deployments。Draft model 的 memory overhead 可能不值得。
- Very-high-temperature creative sampling，此时 `α` 会崩掉。

## Ship It

这一课产出 `outputs/skill-eagle3-tuner.md`。给定一个 inference workload（model、batch size、target latency、task profile），它会推荐 speculative-decoding strategy 和 tuning parameters（draft family、`N`、tree depth、temperature-aware switching）。

## 练习

1. 运行 `code/main.py`。确认 Leviathan distribution check 的 chi-square statistic 在 50,000 samples 上低于 95% critical value。

2. 在 `α` 固定为 0.9、`c` 固定为 0.04 的情况下，将 `N` 从 1 sweep 到 10。绘制 expected tokens per verifier call 和 actual wall time per token。找出使 wall time 最小的 `N`。解释曲线形状。

3. 修改代码，模拟 EAGLE-2 tree search：每一步 draft 提出形状为 `[2, 2, 2]` 的 tree（8 条 candidate paths）。Verifier 运行一次，最高概率 accepted path 胜出。计算每个 leaf 的 `α` 和 total tokens per verifier call。与等价 compute 下的 linear-chain spec-decoding 比较。

4. 为两个 concurrent sequences 实现 batched KV rollback simulator。Sequence A 的所有 drafts 都被 accept；Sequence B 在位置 2 reject。展示正确的 `kv_length` 会按 sequence 更新，并且没有浪费 work。

5. 阅读 EAGLE-3 paper 的 Section 4（Training-Time Test）。用两句话解释为什么没有 TTT 的 naive draft training 会受到 exposure bias 影响，以及为什么训练时喂入 draft 自己的 predictions 能修复这个问题。把它与 seq2seq 中的 scheduled-sampling literature 联系起来。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Leviathan rule | "min(1, q over p)" | 以 `min(1, q(d)/p(d))` 的概率做 Bernoulli accept/reject；如果 rejection 时从 residual 中采样，它会精确保留 verifier distribution |
| Residual distribution | "(q minus p) plus, normalized" | `(q - p)_+` 先 clamp 到零以上再 renormalize；这是 rejection 时应采样的正确分布 |
| Acceptance rate α | "draft 对的频率" | Rejection rule 下 per-token Bernoulli success probability 的期望；支配所有 speedup math |
| EAGLE-1 | "hidden-state draft" | 以 verifier 的 last-layer hidden state 为条件的 tiny transformer draft（Li et al., 2024） |
| EAGLE-2 | "dynamic draft tree" | EAGLE-1 加一个 candidate continuations tree，并用一次 verifier pass 中的 tree attention 打分 |
| EAGLE-3 | "training-time test" | 去掉 feature-prediction loss，直接做 token prediction training，并在训练中让 draft 喂入自己的 outputs |
| Training-time test (TTT) | "exposure bias fix" | 训练期间 autoregressively 运行 draft，让 train 和 test input distributions 匹配；这是 scheduled sampling 的直接类比 |
| KV rollback | "撤销 rejected drafts" | Rejection 后把 verifier KV cache 重置到 accepted-prefix length 的 bookkeeping |
| Bonus token | "免费的那个" | 当所有 `N` 个 drafts 都 accept 时，从 `q_{N+1}` 额外采样一个 token，不增加 verifier cost |
| Tree attention | "一次验证很多 candidates" | 使用尊重 draft tree topology 的 non-causal mask 的 attention；一次 forward pass 计算 tree 中每个 node 的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias - Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) - foundational paper 和 equivalence theorem。
- [Chen et al. - Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) - 同期独立提出，证明非常清晰。
- [Li et al. - EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) - EAGLE-1，hidden-state-conditioned draft。
- [Li et al. - EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) - dynamic tree search。
- [Li et al. - EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) - 2026 年 production default。
- [Cai et al. - Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) - 另一种 draft-free approach。
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) - 将所有策略接起来的 canonical production reference。
