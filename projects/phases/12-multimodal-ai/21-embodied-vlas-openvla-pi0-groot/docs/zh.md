# Embodied VLAs：RT-2、OpenVLA、π0、GR00T

> 第一次让模型从网站上读食谱，并由厨房机器人执行，是 RT-2（Google DeepMind，2023 年 7 月）。RT-2 把 actions 离散化为 text tokens，在 web data 和 robot-action data 上对 VLM 做 co-fine-tuning，并证明 web-scale vision-language knowledge 可以迁移到 robotic control。OpenVLA（2024 年 6 月）发布了 open 7B reference。Physical Intelligence 的 π0 系列（2024-2025）加入了 flow-matching action experts。NVIDIA 的 GR00T N1（2025 年 3 月）为 humanoid robots 大规模交付了 dual-system（System 1 / System 2）控制。VLA primitive，也就是 vision-language-action，一个能看、读、行动的单模型，是本阶段的理解模型与 Phase 15 autonomous systems 之间的桥。

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 minutes

## 学习目标

- 描述 action tokenization：discrete bin encoding（RT-2）、FAST efficient action tokens、continuous flow-matching actions（π0）
- 解释为什么在 web + robot data 上 co-fine-tuning 能保留 general-knowledge transfer，并迁移到新任务
- 在同一个 robot task 上比较 OpenVLA（open 7B Llama+VLM）、π0（flow-matching）和 GR00T N1（dual-system）
- 说出 Open X-Embodiment dataset，以及它作为 RT-X training corpus 的作用

## 问题

一个能根据自然语言指令做家务的机器人，从 1970 年代开始就是研究目标。2020 年代的答案是 vision-language-action（VLA）model。它使用和 VQA 相同的 VLM architecture，但输出不是文本，而是 actions（joint torques、end-effector poses、discrete commands）。

VLA 特有挑战：

1. Action spaces 是 continuous（joint angles、forces）且 high-dimensional（7-DOF arm + 3-DOF gripper = 10 dims at 30 Hz）。
2. Robot-specific training data 很稀缺。Open X-Embodiment 有约 1M trajectories；web text-image 是 5B+。
3. Control frequency 很关键。30 Hz control loop 意味着每个 action 只有 33ms budget。
4. Safety。一个错误 action 可能损坏硬件、伤害人，或破坏财产。

## 概念

### Action tokenization（RT-2）

RT-2 的技巧是：把每个 joint target 表示成一个 quantized text token。把归一化的 [-1, 1] 范围离散化成 256 个 bins，并把每个 bin 映射到一个 vocabulary ID。一个 10-DOF action 在每个 control step 会变成 10 个 tokens。

对 PaLM-X VLM 做 co-fine-tune，混合数据包括：

- Web image-text pairs（captioning、VQA）。
- Robot demonstrations，其中 action 表示为 tokens。

模型看到的是：“pick up the red cube”（language）→ image（vision）→ 10-token action sequence（discretized joint targets）。Web pretraining 保留 general-knowledge transfer：RT-2 能遵循 “move towards the fast-moving object”，即使 training data 里没有 “fast-moving”。

RT-2 论文中的 inference 是 3-5 Hz，受限于 VLM autoregressive decode。

### OpenVLA：open 7B reference

OpenVLA（Kim et al.，2024 年 6 月）是 open-weights 的 RT-2 等价物。它使用 7B Llama backbone、DINOv2 + SigLIP dual vision encoder，并在 256 bins 上做 action tokenization。

它在 Open X-Embodiment 上训练，数据包含 22 个 robots、970k trajectories。它也提供 LoRA fine-tuning support，用于适配新机器人。

Inference：A100 上配合 quantization 约 4-5 Hz。足够 slow manipulation，但不够 high-frequency control。

### FAST tokenizer：更快的 action decode

Pertsch et al.（2024）指出 discrete-bin tokenization 效率低；多数 actions 都聚集在 bin-space 的一小片区域里。FAST（Frequency-domain Action Sequence Tokenizer）通过 DCT 压缩 action sequences，并量化 coefficients。

一个 30-step action trajectory 会变成约 10 个 FAST tokens，而不是 300 个 discrete-bin tokens。Inference 在不损失质量的情况下加速 3-5 倍。

### π0 和 flow-matching actions

Physical Intelligence 的 π0（Black et al.，2024 年 10 月）用 flow-matching action expert 替换 discrete action tokens：

- 一个小 action transformer 读取 VLM hidden states，并通过 rectified flow 输出 continuous 50-step action sequence。
- Action head 使用 flow-matching loss 训练；VLM pretraining 保持不变。
- Inference：完整 action sequence 约 5 个 denoising steps 发出，实际达到 50 Hz control。

π0 的主张是：在一大组 manipulation tasks 上超过 OpenVLA 和 Octo。Continuous-action formulation 保留了 discretization 会破坏的平滑性。

π0.5 和 π0-FAST 是增量升级。π0-FAST 把 FAST tokenization 与 flow matching 结合起来。

### GR00T N1：面向 humanoids 的 dual-system

NVIDIA 的 GR00T N1（2025 年 3 月）是为 humanoid robots（>30 DOF，full-body）构建的：

- System 2：大型 VLM，读取 scene + instruction，并以约 1 Hz 产生 high-level subgoals。
- System 1：小型 action-head transformer，根据 subgoals 产生 low-level 50-100 Hz joint commands。

这个切分对应 Kahneman 的 fast-and-slow thinking：System 2 规划，System 1 行动。好处是：慢速的 VLM-sized planning 不会阻塞快速 control；System 1 保持小巧以满足 latency。

GR00T N1.7（2025 年末）改进了 data scaling。GR00T 使用来自 Omniverse 的 sim-to-real data 做 fine-tune。

### Open X-Embodiment

这是 training data。RT-X（2023 年 10 月）汇集了 22 个 datasets，覆盖 22 个 robots、1M trajectories。Open X-Embodiment 是大家都使用的 corpus：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个 sample：robot state、camera views、instruction、action sequence。
- Training hygiene：统一 action space、归一化 joint ranges、resize cameras。

OpenVLA 和 π0 都在 Open X-Embodiment 上训练。到某个具体 robot 的 domain gap，通过在 100-1000 条 task-specific demos 上做 LoRA fine-tuning 来弥合。

### Co-fine-tuning vs robot-only

Co-fine-tuning 把 web VQA data 与 robot trajectories 混在一起。比例很重要：VQA 太多，模型会忘记 actions；robot data 太多，模型会丢失 general knowledge。

RT-2 的比例约为 1:1。OpenVLA 约为 0.5:1 web-to-robot。π0 类似。精确比例是一个需要按 dataset size 调整的 hyperparameter。

Robot-only training 会产出 task-specific models，并在 out-of-distribution instructions 上失败。Co-fine-tuning 的差别就在于：它能从 “pick up the red cube（demo 里出现过）” 扩展到 “pick up the third largest object from the left（新说法）”。

### Safety 和 action limits

每个 production VLA 都会带上：

- Hard joint limits（不能超过规格扭矩）。
- Velocity limits（soft clipping）。
- Workspace bounds（end-effector 不能离开桌面）。
- Human-in-the-loop approval for novel tasks。

这些都位于 VLA 之外，作为 control-layer checks。VLA 的输出是建议，不是命令。

## 投入使用

`code/main.py`：

- 实现 256-bin action tokenization 和 de-tokenization。
- 草拟一个基于 DCT + quantization 的 FAST tokenizer。
- 比较 discrete-bin、FAST、continuous-flow 在每个 action step 上的 token-count。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的 lineage summary。

## 交付成果

本课产出 `outputs/skill-vla-action-format-picker.md`。给定一个 robot task（manipulation、navigation、humanoid whole-body），它会在 discrete-bin + RT-2、FAST + OpenVLA、flow-matching + π0、dual-system + GR00T 之间做选择。

## 练习

1. 一个 10-DOF arm，control rate 是 30 Hz。256 bins 的 discrete-bin tokenization 每秒会发出多少 tokens？一个 7B VLM 跟得上吗？

2. FAST tokenization 把 30-step trajectories 压缩到约 10 tokens。如果 trajectory 有 high-frequency motion（例如 drumming），用户会失去什么？

3. π0 的 flow-matching head 用约 5 steps denoise。把它的 throughput 与 OpenVLA 的 4-5 Hz autoregressive decode 比较。

4. GR00T 的 System 1 / System 2 切分对应 Kahneman。提出另一种切分（System 3？），可能帮助 bipedal walking。

5. 阅读 Open X-Embodiment 第 4 节关于 dataset curation 的内容。说出防止 domain leakage 的三条 curation rules。

## 关键术语

| Term | 人们常说 | 它真正的意思 |
|------|----------|-------------|
| VLA | "Vision-language-action" | 接收 image + instruction，并输出 action commands 的模型 |
| Action tokenization | "Discrete bins" | 把 continuous joint targets 量化为每维 256 bins，每个 bin 是一个 vocab ID |
| FAST tokenizer | "Frequency action tokens" | DCT + quantize，把 30-step trajectories 压缩到约 10 tokens |
| Co-fine-tune | "Mix web + robot" | 把 web VQA data 与 robot demos 一起训练，以保留 general knowledge |
| Flow-matching action head | "π0 continuous output" | 小 transformer，通过 rectified flow 输出 50-step action sequence |
| System 1 / System 2 | "Dual-system control" | 大型 VLM 慢速规划，小型 action head 快速行动；GR00T 模式 |
| Open X-Embodiment | "RT-X dataset" | 1M-trajectory cross-robot dataset；训练 corpus |

## 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
