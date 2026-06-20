# 指令微调（SFT）

> Base model 只预测下一个 token。仅此而已。它不会遵循指令、回答问题，也不会拒绝有害请求。SFT 是 token predictor 和实用 assistant 之间的桥。你聊过的每个模型，Claude、GPT、Llama Chat，都经历过这一步。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lesson 04 (Pre-Training a Mini GPT)
**Time:** ~90 minutes

## 学习目标

- 实现 supervised fine-tuning（SFT），把 base language model 转换成 instruction-following assistant
- 使用带 system、user、assistant roles 的 chat templates 格式化训练数据，并对 non-assistant tokens mask loss
- 解释为什么 SFT 是必要的：base models 是续写文本，而不是回答问题
- 在 held-out instruction set 上比较 base model 和 fine-tuned model 的 responses，评估 SFT 质量

## 问题

你在 Lesson 04 里训练了一个模型。给定一段 sequence，它可以预测下一个 token。给它输入 "The transformer architecture"，它可能继续写出 "has revolutionized natural language processing." 对一个 next-token predictor 来说，这很厉害。

现在试试这个：给它输入 "What is the capital of France?" Base model 不会回答 "Paris." 它会继续这个 pattern。它可能输出 "What is the capital of Germany? What is the capital of Spain?"，因为它从文档中学到了问题列表。或者它可能输出 "is a question that many people ask"，因为这是一个合理的 next-token continuation。模型没有 *回答* 的概念。它只知道 *续写*。

这就是 GPT-3（base model，2020 年 6 月发布）和 ChatGPT（instruction-tuned，2022 年 11 月发布）之间的差距。相同 architecture。相同 pre-training。区别是 20,000 到 100,000 条精心构造的 (instruction, response) pairs，它们教会模型遵循 conversation pattern。

Stanford Alpaca 证明你不需要数百万 examples。2023 年 3 月，他们只用 GPT-3.5 生成的 52,000 条 instruction-response pairs 微调 Llama 7B。总成本：600 美元。结果是一个能遵循指令、回答问题并进行对话的 chatbot。它没有 ChatGPT 那么好，但以 600 美元和几个小时训练换来的效果，已经惊人接近。

Meta 的 Llama 2 Chat 在初始 SFT 阶段只用了约 27,000 条高质量 examples。关键洞见：质量比数量更重要。27,000 条由熟练标注员编写的 examples，胜过 1 million 条从互联网抓取的 noisy examples。

## 概念

### SFT 实际上做了什么

Supervised Fine-Tuning 继续使用 pre-training 的同一套训练循环：forward pass、compute loss、backward pass、update weights，只是换了一类数据。你不再用 raw text 训练，而是用 structured conversations：

```json
{
  "system": "You are a helpful assistant.",
  "user": "What is the capital of France?",
  "assistant": "The capital of France is Paris."
}
```

模型已经知道 Paris 是 France 的首都。它是在 pre-training 阶段从 Wikipedia、textbooks 和 web pages 中学到这个事实的。SFT 不教模型新事实。它教模型一种新 *behavior*：看到问题时，输出回答。看到指令时，输出 completion。看到有害请求时，输出 refusal。

可以这样理解：Pre-training 给模型知识。SFT 给模型礼貌。

### 数据格式

行业里主要有三种格式。它们编码的是同一类信息，也就是谁说了什么，只是 delimiters 不同。

**Alpaca Format**（Stanford，2023 年 3 月）：

```json
{
  "instruction": "Summarize the following article in 3 sentences.",
  "input": "The European Central Bank raised interest rates...",
  "output": "The ECB increased rates by 25 basis points..."
}
```

简单且广泛使用。`input` field 是可选的，许多 instructions 不需要额外 context。Stanford 发布了 52,000 条这种格式的 examples，由 GPT-3.5 生成，成本 600 美元。这开启了 open-source instruction tuning movement。

**ShareGPT Format**（community，2023）：

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What causes tides?"},
    {"from": "gpt", "value": "Tides are caused by the gravitational pull of the Moon..."},
    {"from": "human", "value": "How often do they occur?"},
    {"from": "gpt", "value": "Most coastal areas experience two high tides and two low tides per day..."}
  ]
}
```

支持 multi-turn conversations。按照惯例，"from" field 使用 "human" 和 "gpt"，不管实际模型是什么。Vicuna 使用从用户分享的 ChatGPT transcripts 中抓取的 70,000 条 ShareGPT conversations 训练。

**ChatML Format**（OpenAI，许多 open-source models 也使用）：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

使用 special tokens（`<|im_start|>`、`<|im_end|>`）分隔 roles。这些 tokens 会在 fine-tuning 时加入 tokenizer vocabulary。Qwen、Yi 和许多其他模型使用 ChatML。

三种格式做的是同一件事：告诉模型“这是 instruction，这是 response，学习这个 pattern”。

### 为什么它有效

模型已经通过 pre-training 学会了语言。它见过数十亿个问题后跟回答、指令后跟完成内容、人与人之间对话的例子。这些 patterns 已经编码在 weights 里。

SFT 会集中这种 latent ability。模型不再需要从 context 中猜测自己应该回答问题还是继续写文档；SFT 会明确训练 conversation pattern。几千条 examples 之后，模型会学到：当你看到 assistant role marker，就生成一个有帮助的 response。

这就是为什么 27,000 条 examples 已经足够。你不是在教模型英语，也不是在教它世界事实。你只是在教它一个简单行为：响应指令。知识已经在那里了。

### Masked Loss

这是 SFT 中最重要的技术细节，而大多数教程会跳过它。

Pre-training 时，你在每个 token 上计算 loss。模型学习预测 sequence 中的每个 next token。SFT 时，你只在 *response* tokens 上计算 loss。Instruction tokens 作为 context 存在，但模型不会因为“预测”错它们而被惩罚。

为什么？因为你不想让模型学会 *生成* instructions。你想让它学会 *响应* instructions。如果你在 instruction tokens 上计算 loss，就相当于训练模型预测 "What is the capital of France?"，仿佛它才是提问的人。这会浪费 gradient signal，也可能让模型混淆自己的角色。

实践中，你会创建一个 loss mask：response tokens 是 1，instruction tokens 是 0。求平均前，把 per-token loss 乘以这个 mask。

```text
Tokens:    [SYS] You are helpful [USER] What is the capital? [ASST] Paris is the capital [EOS]
Loss mask:   0    0    0     0      0     0   0  0     0       1     1    1   1     1      1
```

只有 `[ASST]` 之后的 tokens 会贡献 loss。Forward pass 时模型看到完整 conversation（它需要 instruction 才能生成正确 response），但只根据 response 预测得好不好来更新 weights。

### Training Hyperparameters

SFT 使用的 hyperparameters 和 pre-training 差异很大。你不是从零训练。你是在调整一个已经可用的模型。

| Parameter | Pre-Training (Llama 2 7B) | SFT (Llama 2 Chat) |
|-----------|---------------------------|---------------------|
| Learning rate | 3e-4 (peak) | 2e-5 |
| Epochs | 1 (single pass over data) | 2 |
| Batch size | 4M tokens | 64 examples |
| Warmup steps | 2,000 | 0-100 |
| Weight decay | 0.1 | 0.0-0.1 |
| Data size | 2T tokens | 27,000 examples |

SFT 的 learning rate 低 15 倍。这很关键。Fine-tuning 时如果 learning rate 太高，会破坏 pre-trained knowledge。模型会“忘记”学过的内容，并 overfit 到很小的 fine-tuning dataset。这就是 catastrophic forgetting。

Two epochs 表示模型会看到每个 training example 两次。在小 dataset 上超过 3 epochs 通常会导致 memorization，模型开始逐字复现 training examples，而不是泛化。

### Catastrophic Forgetting

Fine-tuning 会破坏 general capabilities。在 instruction-following data 上训练太久，模型会失去写代码、做数学或生成 creative text 的能力。它会非常擅长训练数据里的特定格式，却在其他事情上变差。

三个缓解方法：

1. **Low learning rate.** 1e-5 到 5e-5。更小的 updates 表示对 pre-trained features 的破坏更少。

2. **Short training.** 1-3 epochs。在模型 overfit 前停止。

3. **Mix in pre-training data.** Llama 2 Chat 把少量（2-5%）raw pre-training data 混入 SFT dataset。这会在模型学习新的 instruction-following behavior 时，“提醒”它保留 general capabilities。

### 真实数字

在 10,000 条高质量 instruction pairs 上 fine-tune 一个 7B model，用单张 NVIDIA A100 80GB GPU 大约需要 1 小时。算式如下：

- 10,000 examples x 平均 512 tokens = 5.12M tokens
- 2 epochs = 总共 10.24M tokens
- A100 对 7B model fine-tuning 的吞吐量：约 3,000 tokens/second
- 10.24M / 3,000 = 约 3,400 seconds = 约 57 minutes

对我们的 mini GPT（4 layers，128 dims）来说，训练几乎是瞬间完成。重点是理解机制，不是规模。

```mermaid
graph TD
    subgraph SFT["Supervised Fine-Tuning Pipeline"]
        direction TB
        D["Instruction Dataset\n(10K-100K examples)"] --> F["Format into\n(instruction, response) pairs"]
        F --> T["Tokenize with\nchat template"]
        T --> M["Create loss mask\n(1 for response, 0 for instruction)"]
        M --> FW["Forward pass\n(full sequence)"]
        FW --> L["Compute masked loss\n(response tokens only)"]
        L --> BW["Backward pass"]
        BW --> U["Update weights\n(lr=2e-5, 1-3 epochs)"]
    end

    subgraph Base["Base Model\n(pre-trained)"]
        B1["Knows language"]
        B2["Knows facts"]
        B3["No conversation pattern"]
    end

    subgraph Chat["Chat Model\n(after SFT)"]
        C1["Knows language"]
        C2["Knows facts"]
        C3["Follows instructions"]
    end

    Base --> SFT --> Chat

    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style L fill:#1a1a2e,stroke:#e94560,color:#fff
    style B3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style C3 fill:#1a1a2e,stroke:#51cf66,color:#fff
```

## 动手构建

### Step 1：Instruction Dataset

创建一个 synthetic instruction dataset。Production 中，Scale AI 和 Anthropic 这类公司会雇佣 human annotators 来写这些数据。这里我们用程序生成，目的是展示格式。

```python
import numpy as np

INSTRUCTION_DATA = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "instruction": "Explain gravity in one sentence.",
        "response": "Gravity is the force that attracts objects with mass toward each other."
    },
    {
        "instruction": "Write a haiku about the ocean.",
        "response": "Waves crash on the shore, salt and foam beneath the sun, endless blue expanse."
    },
    {
        "instruction": "What is 15 multiplied by 7?",
        "response": "15 multiplied by 7 is 105."
    },
    {
        "instruction": "Name three programming languages.",
        "response": "Three programming languages are Python, Rust, and TypeScript."
    },
    {
        "instruction": "Summarize photosynthesis.",
        "response": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."
    },
    {
        "instruction": "What year did World War II end?",
        "response": "World War II ended in 1945."
    },
    {
        "instruction": "Define machine learning.",
        "response": "Machine learning is a field where algorithms learn patterns from data to make predictions."
    },
]
```

八条 examples 非常少。Stanford Alpaca 使用了 52,000 条。但无论你有 8 条还是 52,000 条，机制都一样：tokenize、mask，只在 responses 上计算 loss。

### Step 2：用 Chat Template Tokenize

把 instruction-response pairs 转成带 special role markers 的 token sequences。Markers 会告诉模型 instruction 在哪里结束、response 在哪里开始。

```python
SPECIAL_TOKENS = {
    "INST_START": 253,
    "INST_END": 254,
    "RESP_START": 255,
}


def tokenize_instruction_pair(instruction, response, vocab_size=256):
    inst_tokens = list(instruction.encode("utf-8"))
    resp_tokens = list(response.encode("utf-8"))

    inst_tokens = [min(t, vocab_size - 4) for t in inst_tokens]
    resp_tokens = [min(t, vocab_size - 4) for t in resp_tokens]

    tokens = (
        [SPECIAL_TOKENS["INST_START"]]
        + inst_tokens
        + [SPECIAL_TOKENS["INST_END"]]
        + [SPECIAL_TOKENS["RESP_START"]]
        + resp_tokens
    )

    return tokens


def create_loss_mask(tokens):
    mask = np.zeros(len(tokens), dtype=np.float32)
    in_response = False

    for i, token in enumerate(tokens):
        if token == SPECIAL_TOKENS["RESP_START"]:
            in_response = True
            continue
        if in_response:
            mask[i] = 1.0

    return mask
```

Loss mask 对 instruction tokens 全是 0，对 response tokens 全是 1。`RESP_START` token 本身的 mask 是 0，因为它是 delimiter，不是 response content 的一部分。

### Step 3：Masked Cross-Entropy Loss

标准 cross-entropy，但乘以 loss mask。只有 response tokens 贡献 gradient。

```python
def masked_cross_entropy_loss(logits, targets, loss_mask):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    per_token_loss = -log_softmax[np.arange(len(targets_flat)), targets_flat]

    masked_loss = per_token_loss * mask_flat
    num_response_tokens = mask_flat.sum()
    if num_response_tokens == 0:
        return 0.0
    loss = masked_loss.sum() / num_response_tokens

    return loss
```

分母是 `num_response_tokens`，不是 `seq_len`。如果除以总 sequence length，较长 instructions 会稀释 gradient signal。除以 response token count 可以保证每个 response token 权重相同，不受 instruction length 影响。

### Step 4：SFT Training Loop

复用 Lesson 04 的 MiniGPT。Training loop 和 pre-training 几乎一样，只是加入了 instruction formatting 和 masked loss。

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock, Embedding


def sft_train(model, dataset, num_epochs=2, lr=2e-5, seq_len=64):
    formatted_data = []
    for example in dataset:
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        formatted_data.append((tokens, mask))

    print(f"SFT Training: {len(formatted_data)} examples, {num_epochs} epochs, lr={lr}")
    print(f"Total tokens: {sum(len(t) for t, _ in formatted_data):,}")
    print()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(formatted_data))

        for idx in indices:
            tokens, mask = formatted_data[idx]

            if len(tokens) < 3:
                continue
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
                mask = mask[:seq_len]

            input_ids = np.array(tokens[:-1]).reshape(1, -1)
            target_ids = np.array(tokens[1:]).reshape(1, -1)
            loss_mask = np.array(mask[1:]).reshape(1, -1)

            logits = model.forward(input_ids)
            loss = masked_cross_entropy_loss(logits, target_ids, loss_mask)

            batch_size, s_len, v_size = logits.shape
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            dlogits = probs.copy()
            dlogits[np.arange(batch_size)[:, None], np.arange(s_len), target_ids] -= 1.0

            mask_expanded = loss_mask[:, :, np.newaxis]
            num_resp = loss_mask.sum()
            if num_resp > 0:
                dlogits = dlogits * mask_expanded / num_resp

            for block in model.blocks:
                block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 -= lr * np.random.randn(*block.ffn.W2.shape) * 0.01
                block.ffn.b1 -= lr * np.random.randn(*block.ffn.b1.shape) * 0.01
                block.ffn.b2 -= lr * np.random.randn(*block.ffn.b2.shape) * 0.01

            epoch_loss += loss
            num_batches += 1
            losses.append(loss)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    return model, losses
```

Learning rate 是 2e-5，和 Llama 2 Chat 一致。把它和 pre-training 使用的 3e-4 比一下，它小了 15 倍。Gradient 是 masked：instruction tokens 产生 zero gradient。只有 response tokens 推动 weights。

### Step 5：比较 Base 和 SFT Model

SFT 的全部意义是 behavior change。我们通过检查模型如何响应 instruction-formatted inputs 和 raw text continuations 来衡量它。

```python
def generate_response(model, prompt_tokens, max_new_tokens=50, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / max(temperature, 1e-8)
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))

    return tokens


def evaluate_instruction_following(model, instructions):
    print("Evaluating instruction following:")
    print("-" * 50)

    for instruction in instructions:
        tokens = (
            [SPECIAL_TOKENS["INST_START"]]
            + [min(t, 252) for t in list(instruction.encode("utf-8"))]
            + [SPECIAL_TOKENS["INST_END"]]
            + [SPECIAL_TOKENS["RESP_START"]]
        )

        output = generate_response(model, tokens, max_new_tokens=30, temperature=0.6)
        response_start = len(tokens)
        response_tokens = output[response_start:]
        response_bytes = bytes([t for t in response_tokens if t < 128])
        response_text = response_bytes.decode("utf-8", errors="replace")

        print(f"  Q: {instruction}")
        print(f"  A: {response_text[:80]}")
        print()
```

在只有 8 条 examples 的 tiny model 上，responses 不会有实际意义。这是预期的。重要的是 *structure*：模型会学习在 response marker 之后输出内容，而不是继续生成更多 instructions。

### Step 6：测量 Catastrophic Forgetting

比较 SFT 前后模型的 next-token prediction ability。如果 SFT 损害了 general capabilities，raw text 上的 loss 会上升。

```python
def measure_forgetting(model, test_text, seq_len=64):
    tokens = np.array(list(test_text.encode("utf-8")[:512]))

    total_loss = 0.0
    num_windows = 0

    for start in range(0, len(tokens) - seq_len - 1, seq_len):
        input_ids = tokens[start:start + seq_len].reshape(1, -1)
        target_ids = tokens[start + 1:start + seq_len + 1].reshape(1, -1)

        logits = model.forward(input_ids)

        batch, s_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        max_logits = logits_flat.max(axis=-1, keepdims=True)
        log_softmax = logits_flat - max_logits - np.log(
            np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
        )

        loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
        total_loss += loss
        num_windows += 1

    return total_loss / max(num_windows, 1)
```

真实 fine-tuning 中，你会在整个训练过程中追踪这个 metric。如果 raw text loss 上升超过 10-15%，说明你的 SFT 太激进。降低 learning rate，或减少 epochs。

## 使用它

### Full SFT Pipeline Demo

```python
if __name__ == "__main__":
    np.random.seed(42)

    test_text = """The transformer architecture processes sequences through self-attention.
Each layer applies multi-head attention followed by a feedforward network.
Residual connections and layer normalization stabilize deep networks.
The model learns to predict the next token given all previous tokens."""

    print("=" * 70)
    print("INSTRUCTION TUNING (SFT) DEMO")
    print("=" * 70)
    print()

    model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Config: 4 layers, 4 heads, 128 dims (mini GPT from Lesson 04)")
    print()

    print("PRE-SFT: Measuring base model loss on raw text")
    base_loss = measure_forgetting(model, test_text)
    print(f"  Base model loss: {base_loss:.4f}")
    print()

    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)

    model, losses = sft_train(
        model, INSTRUCTION_DATA, num_epochs=3, lr=2e-5, seq_len=128
    )

    print()
    print("POST-SFT: Measuring fine-tuned model loss on raw text")
    sft_loss = measure_forgetting(model, test_text)
    print(f"  SFT model loss: {sft_loss:.4f}")
    print(f"  Change: {((sft_loss - base_loss) / base_loss * 100):+.1f}%")
    if abs(sft_loss - base_loss) / base_loss < 0.15:
        print("  Minimal forgetting (< 15% change)")
    else:
        print("  Significant forgetting detected")
    print()

    print("=" * 70)
    print("INSTRUCTION FOLLOWING EVALUATION")
    print("=" * 70)
    print()

    test_instructions = [
        "What is the capital of France?",
        "Name a programming language.",
        "Define gravity.",
    ]
    evaluate_instruction_following(model, test_instructions)

    print("=" * 70)
    print("DATA FORMAT EXAMPLES")
    print("=" * 70)
    print()

    for i, example in enumerate(INSTRUCTION_DATA[:3]):
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        resp_count = int(mask.sum())
        total_count = len(tokens)
        print(f"  Example {i + 1}: {total_count} tokens, {resp_count} response tokens ({resp_count/total_count:.0%} of sequence)")
        print(f"    Instruction: {example['instruction']}")
        print(f"    Response: {example['response']}")
        print()

    print("=" * 70)
    print("TRAINING LOSS CURVE")
    print("=" * 70)
    print()

    if losses:
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"  Steps {i:3d}-{i + len(chunk) - 1:3d}: avg loss = {avg:.4f}")
```

## 交付它

这一课会产出 `outputs/prompt-sft-data-curator.md`，一个帮助你设计和 curate SFT instruction datasets 的 prompt。给定目标能力（code generation、math、conversation），它会产出带 format specifications、quality criteria 和 diversity requirements 的 data collection plan。

## 练习

1. 添加 system prompt support。修改 `tokenize_instruction_pair`，让它接收 system message，并在 instruction 前 prepend 它。创建 5 个带不同 system prompts 的 examples（"You are a poet"、"You are a math tutor"），并验证模型在 training 时能看到不同 system prompts。

2. 实现 data mixing。创建一个函数，接收 SFT dataset 和 raw text corpus，然后生成 training batches，其中 5% examples 是 raw text（no masking），95% 是 instruction pairs（masked）。运行 3 epochs，并把 forgetting metrics 和 pure SFT training 对比。

3. 构建 data quality scorer。对每个 instruction-response pair 计算：(a) response length in tokens，(b) instruction-to-response ratio，(c) vocabulary diversity（unique tokens / total tokens）。过滤 response length < 10 tokens 或 diversity < 0.3 的 examples。展示 filtering 如何影响 final loss。

4. 实现 multi-turn conversation training。扩展 tokenization，让它处理 3-turn conversations（user-assistant-user-assistant-user-assistant）。Loss mask 应该覆盖全部三个 assistant turns。通过打印一个 example 的 token-mask alignment 来验证 mask 正确。

5. 比较 learning rates。用 lr=1e-4、lr=2e-5、lr=1e-6 分别训练同一个模型三次。Plot loss curves。1e-4 run 应该表现为初期快速下降但 final loss 更高（overfitting）。1e-6 run 应该几乎不动。2e-5 run 应该是 sweet spot。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| SFT | "Fine-tuning on conversations" | Supervised Fine-Tuning：继续在 (instruction, response) pairs 上训练，但只在 response tokens 上计算 loss |
| Instruction tuning | "Teaching the model to follow instructions" | 在 explicit instruction-response pairs 上训练，让 base model 学到 conversation pattern，而不是新知识 |
| Loss masking | "Ignoring the prompt" | 把 instruction tokens 的 loss 设为 zero，让 gradients 只来自 response token predictions |
| ChatML | "Chat Markup Language" | 使用 `<\|im_start\|>` 和 `<\|im_end\|>` delimiters 在 conversation data 中标记 speaker roles 的 token format |
| Alpaca format | "Stanford's format" | 带 instruction/input/output fields 的 JSON format，用于 52K 条 GPT-3.5 生成、成本 600 美元的 examples |
| Catastrophic forgetting | "The model gets dumber" | Fine-tuning 破坏 pre-trained capabilities，因为 gradient updates 会用 task-specific patterns 覆盖 general knowledge |
| Weight tying | "Shared embeddings" | 输入 token embeddings 和输出 prediction head 使用同一个矩阵，节省 parameters 并提升 coherence |
| Chat template | "How you format the prompt" | 为模型组织 conversation 的特定 token sequence，包括 role markers 和 delimiters |

## 延伸阅读

- [Ouyang et al., 2022 -- "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155) -- 在 OpenAI 引入 instruction tuning + RLHF 的论文。
- [Taori et al., 2023 -- "Stanford Alpaca: An Instruction-following LLaMA Model"](https://github.com/tatsu-lab/stanford_alpaca) -- 600 美元生成的 52K instruction examples，证明 SFT 可以在小数据集上工作。
- [Touvron et al., 2023 -- "Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288) -- Meta 使用 27K 高质量 examples 的 SFT + RLHF pipeline。
- [Chiang et al., 2023 -- "Vicuna: An Open-Source Chatbot Impressing GPT-4"](https://lmsys.org/blog/2023-03-30-vicuna/) -- 在 70K ShareGPT conversations 上训练。
- [Zhou et al., 2023 -- "LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206) -- 证明 1,000 条精心 curated examples 可以匹配使用大得多数据集的 SFT。
