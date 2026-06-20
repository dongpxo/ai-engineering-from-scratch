# Speculative Decoding — Draft, Verify, Repeat

> 自回归解码是串行的。每个 token 都要等待前一个 token。Speculative decoding 打断这条链：便宜模型先 draft N 个 token，昂贵模型一次 forward 验证全部 N 个。如果 draft 正确，你只付出一次大模型 forward，就完成了 N 次生成。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## 问题

一个 70B LLM 在 H100 上采样一个 token 大约需要 30 ms。一个 3B draft model 大约需要 3 ms。如果让 3B draft model 往前草拟 5 个 token，然后让 70B *一次* 验证全部 5 个，总耗时就是 `5×3 + 30 = 45 ms`，最多接受 5 个 token；而直接生成需要 `5×30 = 150 ms`。这就是 speculative decoding 的完整主张：用少量额外 GPU 内存（draft model）换取 2-4 倍更低的 decode latency。

这个技巧必须保持分布不变。Leviathan et al. (2023) 和 Chen et al. 同期提出的 speculative sampling 保证输出序列与大模型单独生成时 **同分布**。没有质量折衷。只是更快。

到 2026 年，四类 draft-verifier 组合主导 inference：

1. **Vanilla speculative（Leviathan 2023）。** 独立 draft model（例如 Llama 3 1B）+ verifier（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 在 verifier 上挂多个 decoding heads，并行预测位置 `t+1..t+k`。不需要单独 draft model。
3. **EAGLE family（Li 2024, 2025）。** 复用 verifier hidden states 的轻量 draft；接受率比 vanilla 更接近 verifier；典型加速 3-4 倍。
4. **Lookahead decoding（Fu 2024）。** Jacobi iteration；完全不需要 draft model。属于 self-speculation。小众但零依赖。

2026 年的每个生产 inference stack 默认都带 speculative decoding。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少都支持 vanilla + EAGLE-2。

## 概念

### 核心算法

给定 verifier `M_q` 和更便宜的 draft `M_p`：

1. 令 `x_1..x_k` 为已经解码的 prefix。
2. **Draft**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，并记录 draft probabilities `p_1..p_N`。
3. **Parallel verify**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行一次 `M_q`，得到位置 `k+1..k+N+1` 的 verifier probabilities `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个 draft token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 如果在位置 `j` 第一次拒绝：从归一化的 “residual” distribution `(q_j - p_j)_+` 采样 `t_j`。`j` 后面的所有 draft 都丢弃。
6. 如果全部 `N` 个都被接受：从 `q_{N+1}` 额外采样一个 token `t_{N+1}`（免费的 bonus token）。

Residual distribution 技巧是保持输出精确等价于 `M_q` 从零采样的数学洞察。

### 什么决定加速比

令 `α` = 每个 draft token 的期望接受率。令 `c` = draft 与 verifier 的成本比例。每一步：

- Naive generation 每生成一个 token 都调用一次大模型。
- Speculative 在 `α` 较高时，每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 调用一次大模型。

经验法则：当 `α = 0.75` 且 `N = 5` 时，大模型调用减少约 3 倍。Draft cost 是 5 次便宜调用。总 wall-clock 大约下降 2.5 倍。

**α 取决于：**

- Draft 对 verifier 的近似有多好。同一模型家族、同一训练数据会显著提高 α。
- 解码策略。Greedy draft 对 greedy verifier：α 高。Temperature sampling：更难匹配，接受率下降。
- 任务类型。代码和结构化输出更容易接受（可预测）；自由创意写作接受率更低。

### Medusa — 没有 draft model 的 draft

Medusa 用 verifier 上的额外输出头替代 draft model。在位置 `t`：

```text
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个 head 都输出自己的 logits。Inference 时，从每个 head 采样得到 candidate sequence，然后用 tree-attention scheme 一次 forward 验证所有 candidate continuation。

优点：没有第二个模型。缺点：增加可训练参数；需要一次 supervised fine-tuning stage（约 1B tokens）；接受率比使用好 draft 的 vanilla speculative 略低。

### EAGLE — 通过复用 hidden states 得到更好的 draft

EAGLE-1/2/3（Li et al., 2024-2025）让 draft model 成为一个很小的 transformer（通常 1 层），输入 verifier 的最后一层 hidden states。因为 draft 看到的是 verifier 的 feature representation，它的预测会与 verifier 的输出分布强相关。接受率会从约 0.6（vanilla）爬升到 0.85+。

EAGLE-3（2025）加入了对 candidate continuations 的 tree search。vLLM 和 SGLang 把 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认 speculative pathway。

### KV cache dance

Verification 会一次把 `N` 个 draft token 喂给 verifier。这会把 verifier 的 KV cache 扩展 `N` 个 entry。如果某些 draft 被拒绝，你必须把 cache 回滚到已接受 prefix 的长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）会用 scratch KV buffers 处理这个问题。先写入，接受时提交。概念上不难，但实现细节很容易出错。

## 动手构建

见 `code/main.py`。我们会实现核心 speculative-sampling 算法（rejection step + residual distribution），包括：

- 一个“大模型”：在手写分布上做 deterministic-softmax（这样可以分析性验证接受数学）。
- 一个“draft model”：大模型分布的扰动版本。
- 一个 acceptance / rejection loop，产生与 direct sampling 相同的边际分布。

### Step 1：rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是 verifier 对 drafted token 的概率。`p_prob` 是 draft model 的概率。Leviathan 定理说，这个 Bernoulli decision，加上拒绝时从 residual 采样，会精确保持 verifier 的分布。

### Step 2：residual distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 中减去 `p`，把负值 clamp 到零，再重新归一化。任何 rejection 都从这个分布采样。

### Step 3：一个 speculative step

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

接受五个 draft → 一个 bonus → 一次 verifier pass 产生六个 token。

### Step 4：测量接受率

在不同 draft-quality level 下运行 10,000 个 speculative steps。绘制 acceptance rate 与 draft 和 verifier 分布之间 KL divergence 的关系。你应该会看到清晰的单调关系。

### Step 5：验证分布等价

经验验证：speculative loop 产生的 token histogram 应该与直接从 verifier 采样产生的 histogram 匹配。这就是 Leviathan theorem 的实践版本。Chi-square test 会确认两者差异在采样误差之内。

## 使用它

生产：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa path。`faster-whisper` 为 Whisper-large 封装了带小 draft 的 speculative decoding。

**选择 draft：**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft（1B/3B Llama family） | 快速原型，不训练 | 1.8-2.3× |
| Medusa heads | 你可以 fine-tune verifier | 2-3× |
| EAGLE-2 / 3 | 生产，最高速度 | 3-4× |
| Lookahead | 无 draft、无训练、无额外参数 | 1.3-1.6× |

**什么时候不要 spec-decode：**

- 单序列生成 1-5 个 token。开销会主导。
- 非常创意 / 高温采样（α 会下降）。
- 内存受限部署（draft model 会增加 VRAM）。

## 交付它

见 `outputs/skill-spec-decode-picker.md`。这个 skill 会为新的 inference workload 选择 speculative decoding 策略（vanilla / Medusa / EAGLE / lookahead）和调参（N、draft temperature）。

## 练习

1. **Easy.** 运行 `code/main.py`。确认在 50,000 个 token 上，speculative token distribution 与 verifier direct-sample distribution 匹配，chi-square p > 0.05。
2. **Medium.** 对 `α = 0.5, 0.7, 0.85`，绘制 speedup（每次大模型 forward 产生的 token 数）随 `N` 的变化。找出每个 α 的最佳 `N`。（提示：每次 verify call 的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **Hard.** 实现一个 tiny Medusa：取 Lesson 14 的 capstone GPT，添加 3 个额外 LM heads，分别预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用 joint multi-head loss 训练。与通过截断同一模型得到的 vanilla draft 比较接受率。
4. **Hard.** 实现 rollback：从一个 10-token prefix KV cache 开始，喂入 5 个 draft token，模拟位置 3 的 rejection。验证下一轮迭代时，你的 cache 读取正确匹配 “prefix + first 2 accepted drafts”。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Draft model | “便宜的那个” | 提出 candidate tokens 的小模型；通常比 verifier 便宜 10-50 倍。 |
| Verifier | “大的那个” | 要保持其分布的目标模型；每个 speculative step 运行一次。 |
| Acceptance rate (α) | “draft 有多常正确” | Verifier 接受 draft 的逐 token 概率。典型值 0.7-0.9。 |
| Residual distribution | “rejection fallback” | 归一化的 `(q - p)_+`；拒绝时从这里采样以保持 verifier 分布。 |
| Bonus token | “免费的那个” | 当全部 N 个 draft 都被接受时，从 verifier 的 next-step distribution 再采一个。 |
| Medusa | “无 draft 的 speculative” | Verifier 上的多个 LM heads 并行预测位置 t+1..t+k。 |
| EAGLE | “Hidden-state draft” | 以 verifier 最后一层 hidden states 为条件的小 transformer draft。 |
| Lookahead decoding | “Jacobi iteration” | 使用 fixed-point iteration 的 self-speculation；没有 draft model。 |
| Tree attention | “一次验证多个 candidates” | 分支验证，同时考虑多个 draft continuation。 |
| KV rollback | “撤销被拒绝的 draft” | Scratch KV buffer；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法与等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期提出；清晰的 Bernoulli-rejection 证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；tree-attention verification。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；hidden-state-conditioned draft。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；dynamic tree depth。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，无 draft 方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 生产标准参考，接入了全部四类策略。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 参考代码。
