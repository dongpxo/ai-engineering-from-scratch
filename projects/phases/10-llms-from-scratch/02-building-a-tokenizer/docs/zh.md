# 从零构建一个 Tokenizer

> Lesson 01 给了你一个玩具。这一课给你一件真正能上场的工具。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## 学习目标

- 构建一个 production-grade BPE tokenizer，能处理 Unicode、whitespace normalization 和 special tokens
- 实现 byte-level fallback，让 tokenizer 可以编码任何输入，包括 emoji、CJK 和代码，而且不会产生 unknown token
- 添加 pre-tokenization regex patterns，在应用 BPE merges 之前按词边界切分文本
- 在 corpus 上训练自定义 tokenizer，并在 multilingual text 上和 tiktoken 比较 compression ratio

## 问题

Lesson 01 里的 BPE tokenizer 可以处理英文文本。现在把日语扔给它。或者 emoji。或者混着 tabs 和 spaces 的 Python 代码。

它会崩。

不是因为 BPE 错了，而是因为实现还不完整。一个 production tokenizer 要处理任何 encoding 的 raw bytes，在切分前 normalize Unicode，管理永远不能被 merge 的 special tokens，把 pre-tokenization 和 subword splitting 串起来，而且速度还要快到不会拖慢处理 15 trillion tokens 的 training pipeline。

GPT-2 的 tokenizer 有 50,257 个 tokens。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些不是玩具数字。这些 vocabulary 背后的 merge tables 是在数百 GB 文本上训练出来的，而周边机制，例如 normalization、pre-tokenization、special token injection、chat template formatting，才是把只能处理 "hello world" 的 tokenizer 和能处理整个互联网的 tokenizer 区分开的东西。

你要构建的就是这套机制。

## 概念

### 完整 Pipeline

Production tokenizer 不是单个算法。它是一条由五个阶段组成的 pipeline，每个阶段解决不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有明确职责：

| Stage | 它做什么 | 为什么重要 |
|-------|----------|------------|
| Normalize | NFKC Unicode，可选 lowercase，可选 strip accents | "fi" 连字（U+FB01）会变成 "fi"（两个字符）。没有这一步，同一个词会得到不同 tokens。 |
| Pre-Tokenize | 在 BPE 前把文本切成 chunks | 防止 BPE 跨词边界 merge。"the cat" 不应该产生一个 "e c" token。 |
| BPE Merge | 把学到的 merge rules 应用到 byte sequences | 核心压缩步骤。把 raw bytes 变成 subword tokens。 |
| Special Tokens | 注入 [BOS]、[EOS]、[PAD]、chat template markers | 这些 tokens 有固定 IDs。它们永远不参与 BPE merges。模型需要它们表示结构。 |
| ID Mapping | 把 token strings 转成 integer IDs | 模型看到的是 integers，不是 strings。 |

### Byte-Level BPE

Lesson 01 的 tokenizer 在 UTF-8 bytes 上工作。这是正确选择。但我们跳过了一个重要问题：如果这些 bytes 不是合法 UTF-8，会发生什么？

Byte-level BPE 的解法是把每一个可能的 byte value（0-255）都当作合法 token。你的 base vocabulary 正好是 256 个 entries。任何文件，无论是文本、二进制还是损坏文件，都可以被 tokenized，而且不会产生 unknown token。

GPT-2 加了一个技巧：把每个 byte 映射到一个可打印的 Unicode character，让 vocabulary 对人类可读。Byte 0x20（space）在它们的 mapping 里会变成字符 "G"。这只是外观处理。算法本身并不关心。

真正的力量在于：byte-level BPE 可以处理地球上的每一种语言。中文字符每个是 3 个 UTF-8 bytes。日语可以是 3-4 个 bytes。阿拉伯文、天城文、emoji，全都只是 byte sequences。BPE algorithm 会像在 English ASCII bytes 中寻找 patterns 一样，在这些 byte sequences 里寻找 patterns。

### Pre-Tokenization

在 BPE 接触文本之前，你需要先把文本切成 chunks。这可以防止 merge algorithm 生成跨越词边界的 tokens。

GPT-2 使用一个 regex pattern 来切分文本：

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个 pattern 会切分 contractions（"don't" 变成 "don" + "'t"）、带可选 leading spaces 的 words、numbers、punctuation 和 whitespace。Leading space 会被保留并附着在 word 上，所以 "the cat" 会变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳过 regex。它把 raw byte stream 当作一条长 sequence，让 BPE algorithm 自己找 boundaries。这更简单，但也给了 BPE 更多自由去创建 cross-word tokens。

这个选择很重要。GPT-2 的 regex 会阻止 tokenizer 学到一个 token，它由某个词末尾的 "the" 和下一个词开头的 "the" merge 而成。SentencePiece 允许这种情况，有时会得到更高效的压缩，但 tokens 的可解释性会下降。

### Special Tokens

每个 production tokenizer 都会为结构标记预留 token IDs：

| Token | Purpose | Used By |
|-------|---------|---------|
| `[BOS]` / `<s>` | sequence 开始 | Llama 3, GPT |
| `[EOS]` / `</s>` | sequence 结束 | All models |
| `[PAD]` | batch alignment 的 padding | BERT, T5 |
| `[UNK]` | unknown token（byte-level BPE 会消除它） | BERT, WordPiece |
| `<\|im_start\|>` | chat message boundary start | ChatGPT, Qwen |
| `<\|im_end\|>` | chat message boundary end | ChatGPT, Qwen |
| `<\|user\|>` | user turn marker | Llama 3 |
| `<\|assistant\|>` | assistant turn marker | Llama 3 |

Special tokens 永远不会被 BPE 切分。它们会在 merge algorithm 运行前被精确匹配，替换成固定 ID；周围的文本照常 tokenized。

### Chat Templates

这里是大多数人困惑、大多数实现出错的地方。

当你把 messages 发给 chat model 时，API 接收的是一个 messages list：

```json
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是 JSON。它看到的是一条扁平 token sequence。Chat template 会用 special tokens 把 messages 转换成这条扁平 sequence。每个模型的格式都不同：

```text
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板错了，模型就会输出垃圾。它是在某一种精确格式上训练的。任何偏差，比如缺一个 newline、换错 token、多一个 space，都会把输入推到 training distribution 之外。

### 速度

Python 对 production tokenization 来说太慢了。

tiktoken（OpenAI）用 Rust 编写，并提供 Python bindings。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些实现比 pure Python 快 10-100 倍。

换个角度看：如果用每秒 1 million tokens 的速度（快 Python）为 Llama 3 pre-training tokenize 15 trillion tokens，需要 174 天。如果用每秒 100 million tokens 的速度（Rust），只需要 1.7 天。

你会用 Python 构建它，是为了理解算法。在生产中，你会使用 compiled implementation，只接触 Python wrapper。

```figure
weight-tying
```

## 动手构建

### Step 1：Byte-Level Encoding

这是基础。把任何 string 转成 byte sequence，把每个 byte 映射成可打印字符用于展示，并实现反向过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用 multilingual text 测试 byte counts：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 bytes。"你好" 是 6 bytes（每个字符 3 bytes）。火焰 emoji 是 4 bytes。Byte-level tokenizer 不关心它是什么语言。Bytes 就是 bytes。

### Step 2：Regex Pre-Tokenizer

使用 GPT-2 regex pattern 把文本切成 chunks。每个 chunk 会被 BPE 独立 tokenized。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` module 支持 Unicode property escapes（`\p{L}` 表示 letters，`\p{N}` 表示 numbers）。标准库 `re` module 不支持，所以我们 fallback 到 ASCII character classes。Production multilingual tokenizers 应该安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Leading space 会留在 word 上。Contractions 会在 apostrophe 处分裂。Punctuation 变成独立 chunk。BPE 永远不会跨越这些 boundaries merge tokens。

### Step 3：Byte Sequences 上的 BPE

这是 Lesson 01 的核心算法，但现在会在 pre-tokenized chunks 上独立运行。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Step 4：Special Token Handling

Special tokens 需要精确匹配和固定 IDs。它们会完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Step 5：完整 Tokenizer Class

把所有东西串起来：normalize、按 special tokens 切分、pre-tokenize、BPE merge、映射到 IDs。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Step 6：Multilingual Test

真正的测试：把 English、Chinese、emoji 和 code 都扔进去。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 bytes。Emoji 产生 4 bytes。它们都不会让 tokenizer 崩掉，也不会产生 unknown tokens。这就是 byte-level BPE 的力量。

## 使用它

### 比较真实 Tokenizers

加载 Llama 3、GPT-4 和 Mistral 的真实 tokenizers。看看它们如何处理同一个 multilingual paragraph。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

同一段文本会得到不同 token counts。Llama 3 有 128K vocabulary，对常见 patterns 的 merge 更激进。GPT-4 的 100K 处在中间。Mistral 的 32K 会产生更多 tokens，但 embedding layer 更小。

Tradeoff 总是一样的：更大的 vocabulary 意味着更短的 sequences，但也意味着更多 parameters。

## 交付它

这一课会产出一个用于构建和调试 production tokenizers 的 prompt。见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **Easy:** 添加一个 `get_token_bytes(id)` method，用来展示任意 token ID 的 raw bytes。用它检查最常见的 merged tokens 实际代表什么。
2. **Medium:** 实现 Llama-style pre-tokenizer：按 whitespace 和 digits 切分，但保留 leading spaces。在同一个 corpus 上，把它和 GPT-2 regex approach 学到的 vocabulary 做比较。
3. **Hard:** 添加一个 chat template method，接收 `{"role": ..., "content": ...}` messages list，并为 Llama 3 chat format 生成正确 token sequence。用 HuggingFace implementation 对照测试。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Byte-level BPE | "Tokenizer that works on bytes" | 以 256 个 byte values 为 base vocabulary 的 BPE，可以处理任意输入而不产生 unknown tokens |
| Pre-tokenization | "Splitting before BPE" | 用 regex 或规则在 BPE 前切分，防止 BPE 跨 word boundaries merge |
| NFKC normalization | "Unicode cleanup" | Canonical decomposition 后接 compatibility composition；"fi" 连字变成 "fi"，全角 "A" 变成 "A" |
| Chat template | "How messages become tokens" | 把 role/content messages list 转成 flat token sequence 的精确格式；模型特定，必须匹配训练格式 |
| Special tokens | "Control tokens" | 绕过 BPE 的 reserved token IDs；[BOS]、[EOS]、[PAD]、chat markers；在 merge 前精确匹配 |
| Fertility | "Tokens per word" | 输出 tokens 与输入 words 的比例；GPT-4 英文约 1.3，韩文约 2-3，越高意味着越浪费 context |
| tiktoken | "OpenAI tokenizer" | 带 Python bindings 的 Rust BPE implementation，比 pure Python 快 10-100 倍 |
| Merge table | "The vocabulary" | 训练期间学到的有序 byte-pair merges list；这就是 tokenizer 学到的知识 |

## 延伸阅读

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4 使用的 Rust BPE implementation。
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- 支持 BPE、WordPiece、Unigram 的 Rust tokenizer library。
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K vocabulary 与 tokenizer training 细节。
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- language-agnostic tokenization。
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 最初的 byte-to-Unicode mapping。
