# OCR 与文档理解

> OCR 是一个三阶段 pipeline：检测文本框，识别字符，然后恢复 layout。每个现代 OCR system 都是在重排这些阶段，或者把它们合并起来。

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06（Detection），Phase 7 Lesson 02（Self-Attention）
**Time:** ~45 minutes

## 学习目标

- 跟踪 classical OCR pipeline（detect -> recognise -> layout）和现代 end-to-end alternatives（Donut、Qwen-VL-OCR）
- 实现用于 sequence-to-sequence OCR training 的 CTC（Connectionist Temporal Classification）loss
- 使用 PaddleOCR 或 EasyOCR，在不训练模型的情况下完成 production document parsing
- 区分 OCR、layout parsing 和 document understanding，并为每个任务选择正确工具

## 问题

到处都是充满文字的图像：receipts、invoices、IDs、scanned books、forms、whiteboards、signs、screenshots。从这些图像中提取 structured data，而不只是字符本身，还要知道“这是总金额”，是应用视觉中价值最高的问题之一。

这个领域分成三层技能：

1. **OCR proper**：把 pixels 变成 text。
2. **Layout parsing**：把 OCR output 分组成 regions（title、body、table、header）。
3. **Document understanding**：从 layout 中提取 structured fields（`invoice_total = $42.50`）。

每一层都有 classical 和 modern approaches。而“我想从一张图里拿到文字”和“我需要这张收据里的总金额”之间的距离，比大多数团队意识到的更大。

## 概念

### Classical pipeline

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection** 产生 line-level 或 word-level quadrilaterals。
- **Recognition** 把每个 region crop 到固定高度，运行 CNN + BiLSTM + CTC，输出 character sequence。
- **Layout** 重建 reading order（Latin 文本通常 top-to-bottom、left-to-right；Arabic、Japanese 会不同）。

### 用一段话理解 CTC

OCR recognition 要从固定长度 feature map 生成可变长度 sequence。CTC（Graves et al., 2006）让你可以在没有 character-level alignment 的情况下训练它。模型在每个 time step 输出一个对（vocab + blank）的 distribution；CTC loss 会对所有可能 alignments 求边缘化，只要这些 alignments 在合并重复字符并移除 blanks 后能还原成 target text。

```text
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 在 2015 年能工作的原因，也仍然在 2026 年训练大多数 production OCR models。

### 现代 end-to-end models

- **Donut**（Kim et al., 2022）：ViT encoder + text decoder；读取 image 并直接输出 JSON。没有 text detector，也没有 layout module。
- **TrOCR**：用于 line-level OCR 的 ViT + transformer decoder。
- **Qwen-VL-OCR / InternVL**：为 OCR tasks fine-tuned 的完整 vision-language models；在 2026 年复杂 documents 上准确率最好。
- **PaddleOCR**：成熟 production package 中的 classical DB + CRNN pipeline；仍然是 open-source workhorse。

End-to-end models 需要更多 data 和 compute，但跳过了 multi-stage pipelines 的 error accumulation。

### Layout parsing

对 structured documents，运行一个 layout detector（LayoutLMv3、DocLayNet），给每个 region 打标签：Title、Paragraph、Figure、Table、Footnote。Reading order 就变成“按 layout order 遍历 regions 并 concatenate”。

对 forms，使用 **Key-Value extraction** models（visually-rich documents 用 Donut，plain scans 用 LayoutLMv3）。它们接收 image + detected text + positions，并预测 structured key-value pairs。

### Evaluation metrics

- **Character Error Rate（CER）**：Levenshtein distance / reference length。越低越好。Production target：clean scans 上 < 2%。
- **Word Error Rate（WER）**：同样的指标，但在 word level。
- **Structured fields 上的 F1**：用于 key-value tasks；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 上的 edit distance**：用于 end-to-end document parsing；Donut paper 引入了 normalised tree edit distance。

## Build It

### Step 1: CTC loss + greedy decoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 会在可用时使用高效的 CuDNN implementation。Greedy decoder 比 beam search 简单，而且通常只比它差不到 1% CER。

### Step 2: Tiny CRNN recogniser

用于 line OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

输入高度固定（CNN 会把 height max-pool 到 1）。Width 是 CTC 的 time dimension。

### Step 3: Synthetic OCR

生成黑字白底的 digit strings，用于端到端 smoke test。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR dataset 会加入 fonts、noise、rotation、blur 和 colour。上面的 pipeline 是一样的。

### Step 4: Training sketch

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个 trivial synthetic data 上，loss 应该在 200 steps 内从 ~3 降到 ~0.2。

## Use It

三条 production paths：

- **PaddleOCR**：成熟、快速、多语言。One-line usage：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**：Python-native、多语言、PyTorch backbone。
- **Tesseract**：classical；当 models 在老式 scanned documents 上表现不佳时仍然有用。

对于 end-to-end document parsing，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于 receipts、invoices 和结构重复的 forms，fine-tune Donut。对于任意 documents 或带 reasoning 的 OCR，像 Qwen-VL-OCR 这样的 VLM 是当前默认选择。

## Ship It

这一课产出：

- `outputs/prompt-ocr-stack-picker.md`：一个 prompt，会根据 document type、language 和 structure 选择 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md`：一个 skill，从零写 greedy 和 beam-search CTC decoders，包括 length normalisation。

## Exercises

1. **（Easy）** 在 5-digit random numeric strings 上训练 TinyCRNN 500 steps。报告 held-out set 上的 CER。
2. **（Medium）** 用 beam search（beam_width=5）替换 greedy decoding。报告 CER delta。Beam search 在哪些 inputs 上胜出？
3. **（Hard）** 在 20 张 receipts 上使用 PaddleOCR，提取 line items，并针对手工标注的 `{item_name, price}` pairs 计算 F1。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| OCR | “Text from pixels” | 把 image regions 转成 character sequences |
| CTC | “Alignment-free loss” | 在没有 per-timestep labels 的情况下训练 sequence model 的 loss；会对 alignments 求边缘化 |
| CRNN | “Classic OCR model” | Conv feature extractor + BiLSTM + CTC；2015 baseline，至今仍用于 production |
| Donut | “End-to-end OCR” | ViT encoder + text decoder；直接从 image 输出 JSON |
| Layout parsing | “Find regions” | 检测并标注 document 中的 Title/Table/Figure/Paragraph regions |
| Reading order | “Text sequence” | 把识别出的 regions 排成句子的顺序；对 Latin 很简单，对 mixed layouts 不简单 |
| CER / WER | “Error rates” | character 或 word 粒度上的 Levenshtein distance / reference length |
| VLM-OCR | “LLM that reads” | 为 OCR tasks 训练或 prompt 的 vision-language model；当前复杂 documents 上的 SOTA |

## Further Reading

- [CRNN（Shi et al., 2015）](https://arxiv.org/abs/1507.05717)：原始 CNN+RNN+CTC architecture
- [CTC（Graves et al., 2006）](https://www.cs.toronto.edu/~graves/icml_2006.pdf)：原始 CTC paper，算法思想很密集
- [Donut（Kim et al., 2022）](https://arxiv.org/abs/2111.15664)：OCR-free document understanding transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：open-source production OCR stack
