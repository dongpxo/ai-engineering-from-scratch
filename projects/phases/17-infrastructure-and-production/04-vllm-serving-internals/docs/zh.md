# vLLM 服务内部机制：PagedAttention、连续批处理、分块预填充

> vLLM 在 2026 年的主导地位不是来自单个技巧，而是三个会叠加收益的默认机制。PagedAttention 始终开启。连续批处理会在 decode 迭代之间把新请求注入活跃 batch。分块预填充会把长 prompt 切成片段，让 decode tokens 不会饥饿。三者一起开启后，一张 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下可达到 2,200-2,400 tok/s，比 vLLM 自身默认值约高 25%，是 naive PyTorch loop 的 3-4 倍。本课会把 scheduler 和 attention kernel 读到你能画图解释的程度，并以 `code/main.py` 中的 toy continuous batcher 收尾：它会像 vLLM 一样调度 prefill 和 decode。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## 学习目标

- 把 PagedAttention 解释成一个 KV cache allocator：blocks、block tables，以及为什么 production load 下 fragmentation 能保持在 4% 以下。
- 在迭代层面画出 continuous batching：完成的 sequences 如何离开 batch，新 sequences 如何不等 batch drain 就加入。
- 用一句话描述 chunked prefill，并说出它保护的是哪个 latency metric（提示：是 TTFT tail，不是 mean throughput）。
- 说出 2026 年 vLLM v0.18.0 中会让团队在一次性开启所有优化时踩坑的 gotcha。

## 问题

Naive PyTorch serve loop 一次只跑一个请求：tokenize、prefill、decode 到 EOS，然后返回。一个用户时这能工作；一百个用户时，就变成一队非常有耐心的人。显而易见的修复是 static batching，但它会把窗口内每个请求都 pad 到最长 prompt，把每次 decode 都 pad 到最长预期输出，并让整个 batch 等最慢的 sequence。你为永远不会用到的 padding 付费，而快请求要等慢请求。

vLLM 同时解决三个问题。PagedAttention 阻止 KV cache fragmentation 像经典连续分配那样吃掉 60-80% GPU memory。Continuous batching 允许请求在每次 decode iteration 之间进入和离开 batch，所以 batch 里总是充满真实工作。Chunked prefill 把 32k-token prompt 拆成约 512-token 的切片，与 decode 交错执行，所以一个长 prompt 不会冻结 GPU 上的所有 decode token。

2026 年的 production default 是三者全开。你需要理解每个机制的作用，因为 failure modes 都在 scheduler 上，而不是 model 上。

## 概念

### 把 PagedAttention 看作虚拟内存系统

KV cache 对每个 sequence 的大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对 Llama 3.3 70B、8192 tokens 来说，BF16 下每个 sequence 大约是 1.25 GB。如果你为每个请求预留 8192 个 slots，但平均请求只使用 1500 tokens，你会浪费约 82% 已预留的 HBM。经典 batching 会支付这笔浪费。

PagedAttention 借用了操作系统虚拟内存的思路。每个 sequence 的 KV cache 不是连续的，而是以固定大小 blocks 分配（默认 16 tokens）。每个 sequence 都有一个 block table，把逻辑 token 位置映射到物理 block IDs。当 sequence 增长超过已分配 blocks 时，再追加一个 block。当它结束时，blocks 回到 pool。

Fragmentation 从 60-80%（经典方式）降到 4% 以下（PagedAttention）。你不会用某个 flag 启用 PagedAttention，它是 vLLM 唯一提供的 allocator。真正的旋钮是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载 weights 和 activations 之后，为 KV blocks 预留多少比例的 HBM。

### 迭代层面的 Continuous batching

旧式 “dynamic batching” 会等一个窗口（例如 10 ms）填满 batch，然后运行 prefill + decode + decode + decode，直到每个 sequence 都结束。快 sequences 提前离开后只能空等 GPU 完成慢 sequences。

Continuous batching 在每个 decode step 之间工作。把正在运行的 sequences 集合称为 `RUNNING` list。每次迭代：

1. `RUNNING` 中刚命中 EOS 或 max_tokens 的 sequence 会被移除。
2. Scheduler 查看 waiting queue。如果还有空闲 KV blocks，就接纳新的 sequences（prefill 或 resumed）。
3. Forward pass 在当前 `RUNNING` 上运行，对每个 sequence 输出一个新 token。

Batch size 不会被 pad 到固定数字。处在不同输出位置的 sequences 共享一次 fused forward。在 2026 年的 vLLM 中，这叫做 `V1 scheduler`。关键不变量是：scheduler 每次 decode iteration 运行一次，而不是每个 request 运行一次。

### Chunked prefill 保护 TTFT tail

Prefill 是 compute-bound 的。Llama 3.3 70B 上的 32k-token prompt 在一张 H100 上需要约 800 ms 纯 prefill 时间。Prefill 运行时，batch 中所有其他 sequence 的 decode tokens 都要等待。在 serving loop 中，一个长 prompt 的 first-token latency（TTFT）会变成数十个其他用户的 inter-token latency（ITL）抖动。

Chunked prefill 把 prefill 拆成固定大小 chunks（默认 512 tokens），并把每个 chunk 作为一个调度单元。两个 chunks 之间，scheduler 可以推进 decode sequences 一个 token。你用少量绝对 prefill latency 损失（每个 chunk 几毫秒）换来更低的 decode-time jitter。公开 benchmark 中，mixed load 下 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个默认机制会彼此作用

这三个特性都假设另外两个存在。PagedAttention 给 scheduler 一个细粒度 KV resource 来做权衡。Continuous batching 需要这种细粒度资源，否则接纳新 sequence 会迫使全局 reshuffle。Chunked prefill 是 scheduler 在同一个 `RUNNING` list 上做出的决策，它只是另一个 scheduler policy，不是独立系统。

你不需要记住每个 flag。你需要知道 scheduler 优化什么：在 KV-block budget 下最大化 goodput，并受 chunked prefill slicing 约束。

### 2026 年 v0.18.0 gotcha

在 vLLM v0.18.0 中，不能把 `--enable-chunked-prefill` 与 draft-model speculative decoding（`--speculative-model`）组合使用。文档中例外的是 V1 scheduler 中的 N-gram GPU speculative decoding。没有读 release notes 就把每个 flag 都打开的团队，会在启动时遇到 run-time error，而不是一个软性的性能回退。如果你的 speculative gain 值得开启 chunked prefill，那就重新审视这个选择：2026 年正确答案常常是没有 chunked prefill 的 EAGLE-3，而不是一个无法编译的 draft model + chunked prefill。

### 需要记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同模型，默认 vLLM（无 chunked prefill）：约 1,800 tok/s。
- 同模型，naive PyTorch forward loop：约 600 tok/s。
- PagedAttention 在 production load 下的 KV fragmentation waste：<4%。
- Mixed load 下 P99 ITL：有 chunked prefill 约 15 ms，无 chunked prefill 约 50 ms。

### Scheduler 看起来是什么样

```python
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个 loop 的 stdlib Python 版本，使用 fake token counts 和 fake forward latency。运行它能看到 chunked prefill 如何让 decode sequences 在长 prefill 期间保持活跃。

```figure
tensor-parallel
```

## Use It

`code/main.py` 会模拟一个 vLLM-style scheduler，并提供可切换的 features。运行它可以看到：

- `NAIVE` mode：一次一个请求，没有 batching。
- `STATIC` mode：pad 并等待，经典 batching。
- `CONTINUOUS` mode：iteration-level admission 和 release。
- `CONTINUOUS + CHUNKED` mode：prefill slices 与 decode 交错。

输出会展示 total throughput（每虚拟秒 tokens 数）、TTFT mean 和 P99 ITL。`CONTINUOUS + CHUNKED` 这一行应该在 mixed traffic 上占优。

## Ship It

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定一个 serving config（batch size、KV memory utilization、chunked prefill size、speculative config），它会生成 scheduler diagnosis，指出三个默认机制中哪一个是 bottleneck，以及应该调哪个参数。

## 练习

1. 运行 `code/main.py`。在 mixed short and long requests workload 上比较 `STATIC` 和 `CONTINUOUS`。Throughput gap 来自哪里：prefill efficiency、decode efficiency，还是 tail latency？
2. 修改 toy scheduler，加入 `--max-num-batched-tokens`。H100 运行 Llama 3.3 70B FP8 时，正确值是多少？（提示：它是 KV block size 和 free blocks 数量的函数，不是 raw HBM 的函数。）
3. 重新阅读 vLLM v0.18.0 release notes。哪些 flag 组合是互斥的？列出来。
4. 对一个 1,000 requests 的 trace 计算 KV cache fragmentation waste：mean 1,500 output tokens、std 600 tokens，在（a）按 8192 max 做 contiguous per-request allocation，（b）PagedAttention with 16-token blocks 两种情况下分别是多少。
5. 用一段话解释为什么 chunked prefill 会改善 P99 ITL，但单独看不会改善 throughput。实践中的 throughput win 来自哪里？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PagedAttention | “the KV trick” | KV cache 的固定大小 block allocator；fragmentation <4% |
| Block table | “the page table” | 每个 sequence 从逻辑 token 位置到物理 KV block 的映射 |
| Continuous batching | “dynamic batching, but right” | 每次 decode iteration 都做 admit/release 决策 |
| Chunked prefill | “prefill splitting” | 把长 prefill 拆成 512-token slices，并与 decode 交错 |
| TTFT | “first token time” | Prefill + queue + network；长 prompts 下由 prefill 主导 |
| ITL | “inter-token latency” | 连续 decode tokens 之间的时间；由 batch size 主导 |
| Goodput | “throughput that meets SLO” | 每个 request 仍满足 TTFT 和 ITL targets 时的 tokens/sec |
| V1 scheduler | “the new scheduler” | vLLM 的 2026 scheduler；N-gram spec decode 是与 chunked prefill 兼容的路径 |
| `--gpu-memory-utilization` | “the memory knob” | 在 weights 和 activations 之后为 KV blocks 预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)：关于 chunked-prefill 与 speculative-decoding 兼容性的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html)：2026 release cadence 与 version-specific behavior。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)：仍然定义 allocator 心智模型的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)：fragmentation analysis 与 scheduler design。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm)：带 flame graphs 的详细 V1 scheduler walkthrough。
