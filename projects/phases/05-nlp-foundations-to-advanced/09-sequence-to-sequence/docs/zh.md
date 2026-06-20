# 序列到序列模型

> 两个 RNN 假装自己是翻译器。它们撞上的瓶颈，就是 attention 存在的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08（CNNs + RNNs for Text），Phase 3 · 11（PyTorch Intro）
**Time:** ~75 minutes

## 问题

分类会把一个长度可变的序列映射到单个标签。翻译会把一个长度可变的序列映射到另一个长度可变的序列。输入和输出位于不同词表中，甚至可能是不同语言，而且长度不保证一一对应。

seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的配方解决了这个问题：两个 RNN。一个读取源句子，并产出固定大小的 context vector。另一个读取这个向量，然后逐 token 生成目标句子。它和第 08 课里你写过的是同一批代码，只是用另一种方式粘在一起。

这件事值得学习，原因有两个。第一，context-vector bottleneck 是 NLP 中最有教学价值的失败案例。它解释了 attention 和 transformer 擅长的所有事情。第二，训练配方（teacher forcing、scheduled sampling、推理时 beam search）仍然适用于包括 LLM 在内的每一种现代生成系统。

## 概念

**Encoder。** 一个读取源句子的 RNN。它的最终 hidden state 就是 **context vector**，也就是整个输入的固定大小摘要。理论上，除了源句子本身，它什么都不该丢。

**Decoder。** 另一个 RNN，用 context vector 初始化。每一步，它把上一步生成的 token 作为输入，并输出目标词表上的概率分布。采样或取 argmax 来选择下一个 token。再把它喂回去。重复，直到生成 `<EOS>` token 或达到最大长度。

**训练。** 在 decoder 的每一步计算 cross-entropy loss，并对整个序列求和。然后通过两个网络做标准 backprop through time。

**Teacher forcing。** 训练时，decoder 在第 `t` 步的输入是位置 `t-1` 上的 *ground-truth* token，而不是 decoder 自己上一步的预测。这样能稳定训练；没有它，早期错误会级联，模型永远学不起来。推理时，你必须使用模型自己的预测，所以训练和推理之间总会有分布差异。这个差异叫做 **exposure bias**。

**瓶颈。** Encoder 学到的关于源句子的所有东西，都必须被挤进那一个 context vector。长句子会丢细节。稀有词会被模糊掉。语序重排（`chat noir` vs. `black cat`）必须靠记忆，而不是计算。

Attention（第 10 课）通过让 decoder 查看 *每一个* encoder hidden state，而不只是最后一个 hidden state，直接修复这个问题。这就是它的完整卖点。

```figure
seq2seq
```

## Build It

### Step 1：一个 encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的 shape 是 `[batch, seq_len, hidden_dim]`，也就是每个输入位置一个 hidden state。`hidden` 的 shape 是 `[1, batch, hidden_dim]`，也就是最后一步。第 08 课说过“对 outputs 做 pooling 来分类”。这里我们保留最后的 hidden state 作为 context vector，并忽略每一步的 outputs。

### Step 2：一个 decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder 会一次调用一步。输入：一批单 token 和当前 hidden state。输出：下一个 token 的词表 logits，以及更新后的 hidden state。

### Step 3：带 teacher forcing 的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

这里有两个旋钮值得点名。`ignore_index=0` 会跳过 padding token 上的 loss。`teacher_forcing_ratio` 是每一步使用真实 token 而不是模型预测的概率。训练开始时使用 1.0（完整 teacher forcing），然后在训练过程中退火到约 0.5，用来缩小 exposure-bias gap。

### Step 4：推理循环（greedy）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding 会在每一步选择概率最高的 token。它可能会跑偏：一旦你提交了一个 token，就不能撤回。**Beam search** 会保留 top-`k` 个仍在扩展中的部分序列，并在最后选择得分最高的完整序列。标准 beam width 是 3-5。

### Step 5：演示瓶颈

在玩具 copy task 上训练模型：source `[a, b, c, d, e]`，target `[a, b, c, d, e]`。不断增加序列长度。观察准确率。

```text
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU hidden state 无法无损记住一个 40-token 输入。信息存在于 encoder 的每一步里，但 decoder 只能看到最后一个 state。Attention 会直接修复这一点。

## Use It

PyTorch 提供了 `nn.Transformer`，也有基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供了完整的 encoder-decoder 模型（BART、T5、mBART、NLLB），这些模型已经在数十亿 token 上训练过。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代 encoder-decoder 已经把 RNN 换成了 transformer。高层形状（encoder、decoder、逐 token 生成）与 2014 年 seq2seq 论文完全相同。不同的是每个 block 内部的机制。

### 什么时候仍然该用基于 RNN 的 seq2seq

对新项目来说，几乎永远不该用。具体例外：

- 流式翻译：输入一次消费一个 token，并且需要有界内存。
- 端侧文本生成：transformer 的内存成本太高。
- 教学。理解 encoder-decoder bottleneck，是理解 transformer 为什么获胜的最快路径。

### Exposure bias 及其缓解方式

- **Scheduled sampling。** 训练时退火 teacher forcing ratio，让模型学会从自己的错误中恢复。
- **Minimum risk training。** 用句子级 BLEU 分数训练，而不是 token 级 cross-entropy。更接近你真正想优化的东西。
- **Reinforcement learning fine-tuning。** 用某个指标奖励序列生成器。现代 LLM RLHF 也使用这个思路。

这三者仍然适用于基于 transformer 的生成。

## Ship It

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **（Easy）** 实现玩具 copy task。训练一个 GRU seq2seq，输入输出成对样本中 target 等于 source。测量长度 5、10、20 上的准确率。复现瓶颈。
2. **（Medium）** 添加 beam width 为 3 的 beam search decoding。在一个小型 parallel corpus 上，对比 greedy 和 beam search 的 BLEU。记录 beam search 在哪里获胜（通常是最后几个 token），以及哪里没有差别。
3. **（Hard）** 在一个 1 万 pair 的 paraphrase 数据集上 fine-tune `facebook/bart-base`。在 held-out inputs 上，把 fine-tuned 模型的 beam-4 输出与 base model 对比。报告 BLEU，并挑选 10 个定性样例。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Encoder | 输入 RNN | 读取 source。产生每一步的 hidden states 和最终 context vector。 |
| Decoder | 输出 RNN | 用 context vector 初始化。一次生成一个 target token。 |
| Context vector | 摘要 | Encoder 的最终 hidden state。固定大小。Attention 要解决的瓶颈。 |
| Teacher forcing | 使用真实 token | 训练时喂入 ground-truth previous token。稳定学习过程。 |
| Exposure bias | 训练/测试差异 | 模型训练时只看真实 token，从未练习过如何从自己的错误中恢复。 |
| Beam search | 更好的解码 | 每一步保留 top-k 个部分序列，而不是 greedy 地立刻提交一个选择。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)：原始 seq2seq 论文。只有四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)：引入了 GRU 和 encoder-decoder 表述。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)：attention 论文。读完这一课后应该立刻读它。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)：可以运行的 seq2seq + attention 代码。
