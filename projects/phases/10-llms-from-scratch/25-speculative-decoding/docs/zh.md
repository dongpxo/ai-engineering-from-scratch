# Speculative Decoding 和 EAGLE

> Frontier LLM 每生成一个 token，都需要对数十亿参数做一次完整 forward pass。这个 forward pass 严重 over-provisioned：大多数时候，一个小得多的模型就能正确猜出接下来的 3-5 个 tokens，大模型只需要 *verify* 这个猜测。猜对时，你用一次 forward 的代价拿到了 5 个 tokens。Speculative decoding（Leviathan et al. 2023）让这件事变得精确，EAGLE-3（2025）把 acceptance rates 推到每次 verify 约 4.5 个 tokens，在输出分布一致的前提下获得 4-5x speedup。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 问题

在 H100 上，一个 70B 级模型的 decode throughput 通常是每秒 40-80 tokens。每个 token 都需要一次完整 forward pass，从 HBM 读取全部模型权重。你不能在不改变输出的情况下把模型变小。你也不能在超过 memory 后继续增加 batch size。你被卡住了，除非你能让模型每次 forward pass 输出不止一个 token。

Autoregressive generation 看起来天然是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里有一个并发机会。如果你有一个廉价 predictor 说 "接下来 4 个 tokens 可能是 [a, b, c, d]"，你就可以在 **一次大模型 forward pass** 中验证全部 5 个位置，并 accept 最长 matching prefix。

Leviathan, Kalai, Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的 accept/reject rule 让这件事变得精确，同时保留 target model 的 sampling distribution。同样的输出分布，快 2-4×。

## 概念

### 双模型设置

- **Target model** `M_p`：你真正想从中采样的大、慢、高质量模型。Distribution: `p(x)`。
- **Draft model** `M_q`：一个小、快、质量较低的模型。Distribution: `q(x)`。通常小 5-30×。

每一步：

1. Draft model 自回归提出 `K` 个 tokens：`x_1, x_2, ..., x_K ~ q`。
2. Target model 在所有 `K+1` 个位置上并行运行一次 forward pass，为每个 proposed token 产生 `p(x_k)`。
3. 用下面的 modified rejection-sampling rule 从左到右 accept/reject 每个 token。Accept 最长 matching prefix。
4. 如果有任意 token 被 reject，就从 corrected distribution 中采样 replacement 并停止。否则从 `p(· | x_1...x_K)` 采样一个 bonus token。

如果 draft 与 target 完全匹配，你每次 target-forward 会得到 K+1 个 tokens。如果 draft 在位置 1 就错了，你只得到 1 个 token。

### Exactness Rule

Speculative decoding **被证明与从 p 中采样在分布上等价**。Rejection rule：

```text
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差值的 positive part。当 draft 与 target 一致（`p ≈ q`）时，acceptance 几乎为 1。当两者不一致时，residual distribution 会被构造出来，使整体 sample 仍然精确服从 `p`。

**Greedy case。** 对 temperature=0 sampling，只要检查 `argmax(p) == x_t`。如果是，accept；如果不是，输出 `argmax(p)` 并停止。

### Expected Speedup

如果 draft model 的 token-level acceptance rate 是 `α`，每次 target-forward pass 产生的 expected tokens 数量为：

```text
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens per forward。一次 target forward 的成本大约是 `cost_q * K + cost_p`（K 个 draft steps 加一次 target verify）。如果 `cost_p >> cost_q * K`，throughput speedup ratio 就是 `3.36× / 1 = 3.36×`。

真正的参数只有 `α`，它完全取决于 draft-target alignment。好的 draft 就是一切。

### 训练 Draft：Distillation

随机小模型是很差的 draft。标准 recipe 是从 target distill：

1. 选择一个小 architecture（70B target 对应约 1B，7B target 对应约 500M）。
2. 在大文本语料上运行 target model，存储它的 next-token distributions。
3. 用 KL divergence 训练 draft，使它匹配 target distribution（不是 ground-truth tokens）。

结果：`α` 在 coding 上通常为 0.6-0.8，在 natural-language chat 上为 0.7-0.85。Production 中 speedups 为 2-3×。

### EAGLE：Tree Drafting + Feature Reuse

Li, Wei, Zhang, Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准 speculative decoding 的两个低效点：

1. Draft 做 K 个串行 steps，每一步都是 full-stack。但 draft 可以复用最近一次 verify 中 target 的 features（hidden states）。Target 已经计算了丰富 representations，而 draft 正在从零重新推导它们。
2. Draft 输出一条 linear chain。如果 draft 能输出一棵 candidate *tree*（每个 node 多个 guesses），target 的一次 forward pass 就可以通过 tree attention mask 并行验证多条 candidate paths，并选择最长 accepted branch。

EAGLE-1 的变化：

- Draft input = target 在位置 t 的 final hidden state，而不是 raw tokens。
- Draft architecture = 1 个 transformer decoder layer（不是单独的小模型）。
- Output = tree，depth 4-6，每个 depth 有 K = 4-8 个 candidates。

EAGLE-2（2024）加入 dynamic tree topology：draft 不确定时 tree 变宽，确定时保持窄。这会在不增加 verify cost 的情况下提高 `α_effective`。

EAGLE-3（Li et al. 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）去掉固定 top-layer feature dependency，并用新的 "test-time simulation" loss 训练 draft，也就是让 draft 匹配 target test-time distribution 的 outputs，而不是 teacher-forced training distribution。Acceptance rate 从 EAGLE-2 的 0.75 提升到 EAGLE-3 的 0.82，mean tokens/verify 从 3.0 提升到 4.5。

### Tree Attention Verification

当 draft 输出一棵 tree 时，target model 会使用 **tree attention mask** 在一次 forward pass 中验证它。Tree attention mask 是一种编码 tree topology 的 causal mask，而不是纯线性 mask。每个 token 只能 attend to 它在 tree 中的 ancestors。Verify pass 仍然是一次 forward、一次 matmul；topological mask 只多花少量 KV entries。

```text
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的 first-token candidates，`c, d, e, f` 是 second-token candidates，那么全部六个位置都能在一次 forward pass 中被验证。输出是任意 accepted path 上的最长 prefix。

### 什么时候赢，什么时候不赢

**会赢：**

- 可预测文本的 chat / completion（code、common English、structured output）。`α` 高。
- Decode 阶段 GPU compute 没用满的 settings（memory-bound phase）。Tree drafting 会利用可用 FLOPs。

**会输 / 没收益：**

- 高随机性的 outputs（高 temperature creative writing）。`α` 会掉到接近 `1/|vocab|`。
- Very high concurrency 的 batch serving。Batching 已经填满 FLOPs，tree verification 空间很小。
- 很小的 target models，此时 draft 并没有小很多。

Production teams 通常报告：chat wall-clock speedup 为 2-3×，code generation 为 3-5×，creative writing 接近零。

```figure
speculative-decoding
```

## Build It

`code/main.py`：

- 一个 reference `speculative_decode(target, draft, prompt, K, temperature)`，实现 exact rejection rule，并验证它能保留 target distribution（与 plain target sampling 相比 empirical KL < 0.01）。
- 一个 EAGLE-style tree drafter，用 top-p branching 构建 depth-K tree。
- 一个 tree attention mask builder，为 verifier 产生正确的 causal pattern。
- 一个 acceptance-rate harness，在 tiny LM 上运行两种方法（从 GPT-2-medium target distill 一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted -> sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Use It

- **vLLM** 和 **SGLang** 提供 first-class speculative decoding。Flags: `--speculative_model`, `--num_speculative_tokens`。EAGLE-2/3 通过 `--spec_decoding_algorithm eagle` flag 支持。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE trees。
- **Reference draft models**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B drafting），`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B drafting）。
- **Medusa heads**（Cai et al. 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不使用 draft model，而是在 target 自身上添加 K 个并行 prediction heads。部署更简单，acceptance 略低于 EAGLE。

## Ship It

这一课产出 `outputs/skill-speculative-tuning.md`：一个会 profile target model workload，并选择 draft model、K（draft length）、tree width、temperature，以及何时 fallback 到 plain decode 的 skill。

## 练习

1. 实现 exact rejection rule 并进行经验验证。分别通过 `speculative_decode` 和 plain target sampling 运行 10K samples；计算两个 output distributions 的 TV distance。应该 < 0.01。

2. 计算 speedup formula。给定固定 `α` 和 `K`，绘制每次 target-forward 的 expected tokens。为 α ∈ {0.5, 0.7, 0.9} 找出 optimal K。

3. 训练一个 tiny draft。使用 124M GPT-2 target，并在 100M tokens 上用 KL loss distill 一个 30M GPT-2 draft。测量 held-out text 上的 `α`。预期：0.6-0.7。

4. 实现 EAGLE-style tree drafting。不要输出 chain，而是让 draft 在每个 depth 输出 top-3 branches。构建 tree attention mask。验证 target 会 accept 最长 correct branch。

5. 测量 failure modes。在 temperature=1.5（high stochasticity）下运行 speculative decode。展示 α 会崩掉，并且因为 draft overhead，算法会比 plain decode 更慢。

## 关键术语

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Target model | "The big model" | 你想从中采样的慢速高质量模型（p distribution） |
| Draft model | "The speculator" | 小而快的 predictor（q distribution）；通常小 5-30x |
| K / draft length | "Look-ahead" | 每次 verify pass 之前 speculated tokens 的数量 |
| α / acceptance rate | "Hit rate" | Draft proposal 被 accept 的 per-token probability |
| Exact rejection rule | "The accept test" | `r < p/q` 比较，会保留 target distribution |
| Residual distribution | "Corrected p-q" | `(p - q)+ / ||(p - q)+||_1`，rejection 时要采样的分布 |
| Tree drafting | "Branching speculation" | Draft 输出 candidate tree，并用 tree-structured attention mask 在一次 pass 中验证 |
| Tree attention mask | "Topological mask" | 编码 tree topology 的 causal mask，让每个 node 只能 attend to 它的 ancestors |
| Medusa heads | "Parallel heads" | Target 自身上的 K 个额外 prediction heads；不需要单独 draft model |
| EAGLE feature reuse | "Hidden-state draft" | Draft input 是 target 的 last hidden state，而不是 raw tokens，因此 draft 更小 |
| Test-time simulation loss | "EAGLE-3 training" | 训练 draft 匹配 target test-time distribution 的 outputs，而不是 teacher forcing |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 - "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) - exact rejection rule 和 theoretical speedup analysis。
- [Chen, Borgeaud, Irving et al., 2023 - "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) - DeepMind 同期 speculative-sampling 论文。
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 - "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) - draft model 的 parallel-heads 替代方案。
- [Li, Wei, Zhang, Zhang, 2024 - "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) - feature reuse 和 tree drafting。
- [Li et al., 2024 - "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) - dynamic tree topology。
- [Li et al., 2025 - "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) - train-time test-time matching。
- [Fu, Haotian, Peng et al., 2024 - "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) - Jacobi/lookahead decoding，一种不需要 speculator 的替代方案。
