# 用于文本的 CNNs 和 RNNs

> Convolutions 学 n-grams。Recurrences 负责记忆。二者都已被 attention 超越。二者在受限硬件上仍然重要。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## 问题

TF-IDF 和 Word2Vec 产生的是 flat vectors，会忽略 word order。基于它们构建的 classifier 无法区分 `dog bites man` 和 `man bites dog`。Word order 有时就是信号本身。

在 transformers 到来之前，有两类 architectures 填补了这个空白。

**Convolutional nets for text（TextCNN）.** 在 word embeddings 的 sequences 上应用 1D convolutions。宽度为 3 的 filter 是一个可学习的 trigram detector：它跨过三个 words 并输出一个 score。堆叠不同宽度（2、3、4、5）来检测 multi-scale patterns。用 max-pool 得到 fixed-size representation。扁平、并行、快速。

**Recurrent nets（RNN、LSTM、GRU）.** 一次处理一个 token，并维护一个向前携带信息的 hidden state。Sequential，有记忆，支持 flexible input lengths。它们在 2014 到 2017 年主导了 sequence modeling，然后 attention 出现了。

本课会构建这两类模型，然后指出那个推动 attention 出现的 failure。

## 概念

**TextCNN**（Kim, 2014）。Tokens 会先被 embedded。宽度为 `k` 的 1D convolution 会让 filter 滑过连续的 `k`-grams embeddings，产生 feature map。对这个 map 做 global max-pooling，会选出最强 activation。把多个 filter widths 的 max-pooled outputs 拼接起来。送进 classifier head。

为什么它有效。一个 filter 就是一个可学习的 n-gram。Max-pooling 是 position-invariant 的，所以 "not good" 在 review 开头或中间都会触发同一个 feature。三个 filter widths，每个 100 个 filters，就给你 300 个 learned n-gram detectors。Training 是并行的；没有 sequential dependency。

**RNN.** 在每个 time step `t`，hidden state 为 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 会跨时间共享。Time `T` 的 hidden state 是整个 prefix 的 summary。对于 classification，可以在 `h_1 ... h_T` 上做 pooling（max、mean 或 last）。

Plain RNNs 会遭遇 vanishing gradients。**LSTM** 加入 gates，决定遗忘什么、存储什么、输出什么，从而稳定 long sequences 中的 gradients。**GRU** 把 LSTM 简化成两个 gates；参数更少，表现相近。

**Bidirectional RNNs** 运行一个 forward RNN 和一个 backward RNN，再拼接 hidden states。每个 token 的 representation 都能看到左右两侧 context。对 tagging tasks 来说这是必需的。

```figure
rnn-unroll
```

## Build It

### Step 1: PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 会把 `[batch, seq_len, embed_dim]` reshapes 成 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 会把中间轴当作 channels。无论 input length 如何，pooled output 都是 fixed-size。

### Step 2: LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

在 sequence 上做 max-pool，不要只取 last-state pool。对于 classification，max-pooling 通常比取最后一个 hidden state 更好，因为 long sequence 末尾的信息容易支配 last state。

### Step 3: vanishing gradient demo（直觉）

没有 gating 的 plain RNN 无法学习 long-range dependencies。考虑一个 toy task：预测 token `A` 是否曾出现在 sequence 中。如果 `A` 在位置 1，而 sequence 长度是 100，loss 的 gradient 必须穿过 recurrent weight 的 99 次乘法才能回到开头。如果 weight 小于 1，gradient 会 vanish。如果大于 1，它会 explode。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTMs 用 **cell state** 修复这个问题。Cell state 穿过 network 时只有 additive interactions（forget gate 会以乘法方式缩放它，但 gradients 仍然可以沿着 "highway" 流动）。GRUs 用更少参数做类似的事。二者都能让 100+ step sequences 的 training 稳定下来。

### Step 4: 为什么这仍然不够

即便有 LSTMs，仍然存在三个问题。

1. **Sequential bottleneck.** 在长度为 1000 的 sequence 上训练 RNN，需要 1000 个串行 forward/backward steps。无法沿时间维并行。
2. **Encoder-decoder setups 中的 fixed-size context vector.** Decoder 只能看到 encoder 的 final hidden state，而它压缩了整个 input。Long inputs 会丢失细节。Lesson 09 会直接讲这个问题。
3. **Distant-dependency accuracy ceiling.** LSTMs 胜过 plain RNNs，但仍然难以把具体信息传播到 200+ steps 之后。

Attention 解决了这三个问题。Transformers 完全抛弃 recurrence。Lesson 10 是转折点。

## Use It

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都是 production-ready。Training code 很标准。

Hugging Face 提供 pretrained embeddings，你可以把它们接成 input layer：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Use-when-it-fits-the-constraint checklist。

- **Edge / on-device inference.** 带 GloVe embeddings 的 TextCNN 比 transformer 小 10-100 倍。如果 deploy target 是手机，这就是该用的 stack。
- **Streaming / online classification.** RNN 一次处理一个 token；transformers 需要完整 sequence。对于 real-time incoming text，LSTMs 仍然会赢。
- **Tiny models for baselines.** 在新 task 上快速 iterate。TextCNN 可以在 CPU 上 5 分钟训练完。
- **Sequence labeling with limited data.** BiLSTM-CRF（lesson 06）对于 1k-10k labeled sentences，仍然是 production-grade NER architecture。

其它情况都交给 transformer。

## Ship It

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **Easy.** 在 3-class toy dataset（你自己 invent data）上训练 TextCNN。验证 filter widths（2、3、4）在 average F1 上优于单一 width（3）。
2. **Medium.** 为 LSTM classifier 实现 max-pool、mean-pool 和 last-state pooling。在小型 dataset 上比较；记录哪种 pooling 胜出，并假设原因。
3. **Hard.** 构建一个 BiLSTM-CRF NER tagger（结合 lesson 06 和本课）。在 CoNLL-2003 上训练。与 lesson 06 中的 CRF-alone baseline 和 BERT fine-tune 比较。报告 training time、memory 和 F1。

## 关键术语

| Term | 人们通常怎么说 | 它实际是什么意思 |
|------|----------------|------------------|
| TextCNN | CNN for text | 在 word embeddings 上堆叠 1D convolutions，并使用 global max-pool。Kim (2014)。 |
| RNN | Recurrent net | 每个 time step 更新 hidden state：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | Gated RNN | 加入 input / forget / output gates 和 cell state。能在 long sequences 中稳定训练。 |
| GRU | Simpler LSTM | 两个 gates 而不是三个。Accuracy 相近，参数更少。 |
| Bidirectional | Both directions | Forward + backward RNN 拼接。每个 token 都能看到自身 context 的两侧。 |
| Vanishing gradient | Training signal dies | Plain RNNs 中反复乘以 <1 的 weights，会让 early-step gradients 实际变成 zero。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN paper。八页，可读性很好。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM paper。意外地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTMs 变得人人看得懂的那些 diagrams。
