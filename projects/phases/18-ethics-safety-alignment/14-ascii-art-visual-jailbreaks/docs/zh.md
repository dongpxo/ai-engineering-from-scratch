# ASCII Art 与视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，“ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs”（ACL 2024，arXiv:2402.11753）。在有害请求中 mask safety-relevant tokens，把它们替换成同一字母的 ASCII-art rendering，然后发送 cloaked prompt。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳健识别 ASCII-art tokens。该攻击绕过了 PPL（perplexity filters）、Paraphrase defenses 和 Retokenization。相关方向：ViTC benchmark 测量模型对 non-semantic visual prompts 的识别能力；StructuralSleight 把问题推广到 Uncommon Text-Encoded Structures（trees、graphs、nested JSON）这一类 encoding attacks。

**Type:** Build
**Languages:** Python (stdlib，ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## 学习目标

- 描述 ArtPrompt attack：word-identification step、ASCII-art substitution、final cloaked prompt。
- 解释为什么标准防御（PPL、Paraphrase、Retokenization）会在 ArtPrompt 上失效。
- 定义 ViTC，并描述它测量什么。
- 描述 StructuralSleight 如何把问题推广到任意 Uncommon Text-Encoded Structures。

## 问题

通过 paraphrase 和 roleplay 的攻击（第 12 课），以及通过 long context 的攻击（第 13 课），都作用在 text-level pattern 上。ArtPrompt 作用在 recognition level 上：模型不是解析 forbidden token，而是解析一个由字符渲染出的图像。Safety filter 看到的是无害的标点。模型看到的是一个词。

## 概念

### ArtPrompt，两个步骤

Step 1. Word Identification。给定一个 harmful request，attacker 使用 LLM 识别 safety-relevant words（例如 “how to make a bomb” 中的 “bomb”）。

Step 2. Cloaked Prompt Generation。把每个识别出的词替换成它的 ASCII-art rendering（用 7x5 或 7x7 的字符块形成字母形状）。模型收到的是一块由标点和空格构成的网格；足够强的模型能把它识别成词，而 safety filter 只看到这个网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。在他们的 benchmark subset 上，attack success rate 超过 75%。

### 为什么标准防御会失败

- **PPL（perplexity filter）。** ASCII art 的 perplexity 很高，但所有新颖输入也都会很高。能挡住 ArtPrompt 的 threshold 往往也会挡住合法的 structured input。
- **Paraphrase。** 改写 prompt 会破坏 ASCII art。实践中，paraphrase LLMs 经常会保留或重建这段 art。
- **Retokenization。** 换一种方式切分 tokens，并不会改变模型的“视觉”正在识别 letter shapes 这个事实。

底层问题是：safety filters 往往处在 token-level 或 semantic-level；ArtPrompt 则工作在 visual recognition level。

### ViTC benchmark

ViTC 测量模型对 non-semantic visual prompts 的识别能力。它评估模型读取 ASCII-art、wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC accuracy 相关：模型越擅长读 visual text，ArtPrompt 在它身上越有效。这是一个 capability-safety tradeoff。

### StructuralSleight

StructuralSleight 将 ArtPrompt 推广为 Uncommon Text-Encoded Structures（UTES）。Trees、graphs、nested JSON、CSV-in-JSON、diff-style code blocks 都属于这类结构。如果某种结构在 safety training data 中很少见，但模型可以解析它，它就可能隐藏有害内容。

防御含义是：safety 必须能泛化到模型可以解析的各种 structured representations。这个集合很大，而且还在增长。

### Image-modality analog

Visual LLMs（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了 attack surface。带真实图像的 ArtPrompt-style attacks 比 ASCII-art analogs 更强，因为 image encoders 会产生更丰富的 signal。

### 它在 Phase 18 中的位置

第 12-14 课描述了三个正交攻击向量：iterative refinement（PAIR）、context length（MSJ）和 encoding（ArtPrompt/StructuralSleight）。第 15 课从 model-centric attacks 转向 system-boundary attacks（indirect prompt injection）。第 16 课描述 defensive tooling response。

## Use It

`code/main.py` 构建了一个玩具型 ArtPrompt。你可以用 ASCII-art glyphs cloak harmful query 中的特定 words，验证 cloaked string 会通过 keyword filter，并且可以（可选）用一个简单 recognizer 把 cloaked string decode 回来。

## Ship It

本课产出 `outputs/skill-encoding-audit.md`。给定一份 jailbreak-defense report，它会枚举已覆盖的 encoding attack families（ASCII art、base64、leet-speak、UTF-8 homoglyph、UTES），以及捕捉每一类攻击的 defense layer。

## 练习

1. 运行 `code/main.py`。验证 cloaked string 会通过一个简单 keyword filter。报告所需的 character-level change。

2. 为同一个 target word 实现第二种 encoding：base64。比较它与 ArtPrompt 的 filter-bypass rate 和 recovery difficulty。

3. 阅读 Jiang 等人 2024 年论文第 4.3 节（five-model results）。提出一个理由，解释为什么 Claude 在同一 benchmark 上的 ArtPrompt-resistance 高于 Gemini。

4. 设计一个 pre-generation defense，用来检测 prompt 中 ASCII-art-shaped regions。在合法 code、tables 和 mathematical notation 上测量 false-positive rate。

5. StructuralSleight 列出了 10 种 encoding structures。草拟一个能处理全部 10 种结构的 generalized defense，并估算每个 defended prompt 的 compute cost。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| ArtPrompt | “ASCII-art attack” | 两步 jailbreak：用 ASCII-art renderings mask safety words |
| Cloaking | “hide the word” | 把 forbidden token 替换成模型能读出、filter 读不出的视觉表示 |
| UTES | “uncommon structure” | Uncommon Text-Encoded Structure；用 tree、graph、nested JSON 等结构 smuggle content |
| ViTC | “visual-text capability” | 测量模型读取 non-semantic visual encoding 能力的 benchmark |
| Perplexity filter | “PPL defense” | 拒绝 high-perplexity prompts；会失败，因为合法 structured input 也常常 high-perplexity |
| Retokenization | “tokenizer shift defense” | 用不同 tokenizer 预处理 prompt；会失败，因为 recognition 是视觉层面的 |
| Homoglyph | “lookalike characters” | 看起来和 Latin letters 一样的 Unicode characters；可绕过 substring checks |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753)：ASCII-art jailbreak paper。
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754)：UTES generalization。
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419)：complementary iterative attack。
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking)：complementary length attack。
