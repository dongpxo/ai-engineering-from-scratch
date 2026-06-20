# Open-Weight VLM Recipes：真正重要的是什么

> 2024-2026 年的 open-weight VLM 文献像一片 ablation tables 的森林。Apple 的 MM1 测试了 image encoder、connector 和 data mix 的 13 种组合。Allen AI 的 Molmo 证明详细 human captions 胜过 GPT-4V distillation。Cambrian-1 跑了 20+ 个 encoder 对比。Idefics2 把五轴设计空间形式化。Prismatic VLMs 在受控 benchmark 上比较了 27 种 training recipes。在所有噪声里，有一小组结果跨论文成立：image encoder 比 connector architecture 更重要，data mixture 比两者都重要，详细 human captions 胜过蒸馏出来的 synthetic data。本课替你读这些表。

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## 学习目标

- 说出 VLM 五轴设计空间：image encoder、connector、LLM、data mix、resolution schedule
- 阅读 MM1 / Idefics2 / Cambrian-1 的 ablation table，并预测哪个旋钮会推动某个 benchmark
- 在给定 compute budget 和 task mix 的情况下，为一个新 VLM 选择 recipe（encoder、connector、data、resolution）
- 解释为什么在相同 token count 下，详细 human captions 胜过 GPT-4V distillation

## 问题

Open-weight VLM 已经有数百个。大多数“好模型”和“state-of-the-art”之间的差距，不在 architecture。它在 data、resolution schedule 和 encoder choice。知道模型表现不足时先拧哪个旋钮，能帮你避免一个 500 万 GPU 小时级别的错误。

2023 年那一波模型（LLaVA-1.5、InstructBLIP、MiniGPT-4）依赖 caption-pair pretraining + LLaVA-Instruct-150k。它们是不错的 baseline，但在 MMMU 上大约止步于 35%。

2024 年那一波模型（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了穷举式 ablations。结果既意外，也非常实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon et al., 2024）命名了这些轴：

1. Image encoder。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。Encoders 在 patch size、resolution 和 pretraining objective 上不同。
2. Connector。MLP（2-4 layers）、Q-Former（32 queries + cross-attn）、Perceiver Resampler（64 queries）、C-Abstractor（convolutional + bilinear pooling）。
3. Language model。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM size 是主要参数成本。
4. Training data。Caption pairs（CC3M、LAION）、interleaved（OBELICS、MMC4）、instruction（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. Resolution schedule。固定 224/336/448、AnyRes、native dynamic。训练时可以 ramp，也可以 constant。

每个 production VLM 都会在每条轴上做选择。MMMU 分数的大部分方差由第 1、4、5 轴解释，而不是由你选择了哪种 connector 解释。

### Axis 1：encoder > connector

MM1 第 3.2 节显示：从 CLIP ViT-L/14 换到 SigLIP SO400m/14，MMMU 增加 3+ 分。把 connector 从 MLP 换到 Perceiver Resampler，增加不到 1 分。Idefics2 复现了这个结果：SigLIP > CLIP；在相同 token count 下，Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的 "Cambrian Vision Encoders Match-Up"（Tong et al., 2024）在一个 vision-centric benchmark（CV-Bench）上跑了 20+ 个 encoders。Leaderboard 顶部是 DINOv2 和 SigLIP 的混合；CLIP 位于中游；ImageBind 和 ViT-MAE 更低。从 CLIP ViT-L 到 DINOv2 ViT-g/14，在 CV-Bench 上有约 5-7 分差距。

2026 年 open VLM 的默认 encoder 是 SigLIP 2 SO400m/14，用于 semantic + dense features；如果需要 segmentation/grounding，有时会与 DINOv2 ViT-g/14 features 拼接（Cambrian 的 "Spatial Vision Aggregator" 就这么做）。

### Axis 2：connector design 差异不大

MM1、Idefics2、Prismatic 和 MM-Interleaved 得出同一个结论：在固定 visual-token count 下，connector architecture 几乎不重要。同样 token budget 下，基于 mean-pooled patches 的 2-layer MLP，表现距离 32-query Q-Former 不到 1 分。

真正重要的是 token count。更多 visual tokens = 更多 LLM compute = 性能更好，直到某个点后收益递减。每张图 64 tokens 对 OCR 来说太少。576-1024 tokens 是多数 open VLM 的甜点区。2048+ 只对 documents 和 charts 有明显帮助。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 不管 image resolution 如何，都把 tokens 限制在 32-64；MLP 会发出全部 patch tokens。对 high-res inputs，Q-Former 节省 LLM context；对 low-res，差异基本是噪声。

### Axis 3：LLM size 决定上限

把 LLM 从 7B 翻倍到 13B，在每篇 VLM 论文里都稳定给 MMMU 增加 2-4 分。到了 70B，多数 benchmarks 接近饱和。VLM 的 multimodal reasoning ceiling 就是 LLM 的 text reasoning ceiling；visual encoder 只能喂信息，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro、ScreenSpot-Pro 上碾压：language brain 巨大。一个 7B VLM 不能靠巧妙 connector design 替代 70B VLM。

### Axis 4：data，详细 human captions 胜过 distillation

Molmo + PixMo（Deitke et al., 2024）是 2024 年每个人都该读的结果。Allen AI 让 human annotators 用 1-3 分钟的 dense speech-to-text 方式描述图像，得到 712K 张 densely-captioned images。训练数据里没有任何 GPT-4V distillation。

Molmo-72B 在 11/11 个 benchmarks 上超过 Llama-3.2-90B-Vision。差距不来自 architecture，而来自 caption quality。详细 human captions 每张图包含的信息量比短 web captions 多 5-10 倍，并且在 GPT-4V distillation 会 hallucinate 的地方保持事实 grounded。

ShareGPT4V（Chen et al., 2023）和 Cauldron（Idefics2）也采用同样路线：混合 human + GPT-4V captions。趋势很清楚：对 2026 年 frontier 来说，caption density > caption quantity > distillation convenience。

### Axis 5：resolution 及其 schedule

Idefics2 的 ablations：384 → 448 增加 1-2 分。448 → 980 并配合 image splitting（AnyRes），在 OCR benchmarks 上再增加 3-5 分。Flat resolution training 会在中等准确率处 plateau；resolution ramping（从 224 开始，最后到 448 或 native）训练更快，最终分数更高。

Cambrian-1 跑了 resolution vs tokens 的取舍：在固定 compute 下，你可以用低 resolution 换更多 tokens，也可以用高 resolution 换更少 tokens。OCR 场景高 resolution 赢；general scene understanding 则是 lower-res-more-tokens 赢。

2026 年的 production recipe：Stage 1 使用 384 fixed 训练，Stage 2 对 OCR-heavy tasks 使用最高 1280 的 dynamic resolution。

### Prismatic 的受控比较

Prismatic VLMs（Karamcheti et al., 2024）是控制所有变量的那篇论文。同一个 13B LLM、同一份 instruction data、同一套 evaluation；每次只改变一条轴。结果：

- Per-image visual-token count 解释约 60% 方差。
- Encoder choice 解释约 20%。
- Connector architecture 解释约 5%。
- 其他因素（data mix、scheduler、LR）解释剩余约 15%。

这是一个粗略分解，但它是文献里对“我应该先 ablate 什么”最干净的回答。

### 2026 年的 picker

根据这些证据，2026 年新项目默认 open-VLM recipe 是：

- Encoder：native resolution + NaFlex 的 SigLIP 2 SO400m/14；如果需要 dense features 做 segmentation/grounding，再拼接 DINOv2 ViT-g/14。
- Connector：patch tokens 上的 2-layer MLP。除非 token-constrained，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2；7B 用于成本，70B 用于质量，按目标 latency 选择。
- Data：PixMo + ShareGPT4V + Cauldron，再补充 task-specific instruction data。
- Resolution：dynamic（long side 最小 256、最大 1280 pixels）。
- Schedule：Stage 1 alignment（projector-only），Stage 2 full fine-tune，Stage 3 task-specific fine-tune。

这些 defaults 的每一个，都能追溯到本课末尾引用论文中的实测 ablation。

## 投入使用

`code/main.py` 是一个 ablation table parser 和 recipe picker。它编码了 MM1 和 Idefics2 的 ablation tables（压缩版），并支持查询：

- “给定 budget X 和 task Y，哪个 recipe 胜出？”
- “如果我在 7B Llama 上把 SigLIP 换成 CLIP，预期 MMMU delta 是多少？”
- “为了得到 80% confidence answer，我应该先 ablate 哪条轴？”

输出是一组 ranked recipes，包含 expected benchmark deltas 和一个 "ablate first" recommendation。

## 交付成果

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定 target task mix、compute budget 和 latency target，它会输出完整 recipe（encoder、connector、LLM、data mix、resolution schedule），并引用支持每个选择的 ablation。它能阻止工程师在每个新 VLM 项目开始时重新发明 Idefics2 ablation table。

## 练习

1. 阅读 MM1 第 3.2 节。对一个固定 2B LLM，在 50M images budget 下哪个 encoder 胜出？如果换成 13B LLM，答案会反转吗？为什么？

2. Cambrian-1 发现 DINOv2 + SigLIP 拼接在 vision-centric benchmarks 上优于任一单独 encoder，但在 MMMU 上没有额外信号。预测哪些 benchmarks 会提升，哪些会持平。

3. 你的目标是一个运行在 2B LLM 上的 mobile UI agent。选择 encoder、connector、resolution 和 data mix。用具体 ablation table 为每个选择辩护。

4. Molmo 发布 4B 和 72B models。4B 能与 closed 7B VLMs 竞争；72B 在 11/11 benchmarks 上超过 Llama-3.2-90B-Vision。这说明 LLM-size plateau hypothesis 的什么问题？

5. 设计一个 ablation table，在 7B VLM 上隔离 data-mix quality 和 encoder quality。最少需要多少次 training runs？提出四个 axis settings。

## 关键术语

| Term | 人们常说 | 它真正的意思 |
|------|----------|-------------|
| Ablation | "Turning one knob" | 训练多次 run，每次只改变一条 design-space axis，其余全部保持不变 |
| Connector | "Bridge" / "projector" | 把 vision encoder output 映射到 LLM token space 的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | "Dense caption" | 人类写的多句描述（通常 80-300 tokens），比 web alt text 更丰富 |
| Distillation | "GPT-4V captions" | 由更强 proprietary VLM 生成的 training data；方便，但容易继承 hallucination |
| AnyRes / dynamic res | "High-res path" | 通过 tiling 或 M-RoPE 输入大于 encoder native resolution 的图像 |
| Resolution ramp | "Curriculum" | 从低分辨率开始并逐步提高的训练 schedule，可加速 alignment learning |
| Vision-centric bench | "CV-Bench / BLINK" | 强调细粒度 visual perception，而不是 language-heavy reasoning 的 evaluation |
| PixMo | "Molmo's data" | Allen AI 的 712K densely-captioned image dataset；由 human speech 转写为 dense captions |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
