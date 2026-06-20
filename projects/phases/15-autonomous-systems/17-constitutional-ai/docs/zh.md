# Constitutional AI 与规则覆盖

> Anthropic 在 2026 年 1 月 22 日发布的 Claude Constitution 长达 79 页，并采用 CC0 授权。它从 rule-based alignment 转向 reason-based alignment，并建立了四层优先级层级：（1）安全与支持人类监督，（2）伦理，（3）Anthropic guidelines，（4）有用性。行为被拆分为 hardcoded prohibitions（例如生物武器能力提升、CSAM），运营方和用户都不能覆盖；以及 soft-coded defaults，运营方可以在定义好的边界内调整。2022 年原始论文（Bai 等）通过 self-critique 和基于 constitution 的 RLAIF 来训练 harmlessness。诚实的保留意见是：reason-based alignment 依赖模型把原则泛化到未预见场景。Anthropic 自己 2023 年的参与式实验显示，公众来源原则和企业原则之间约有 50% 分歧；2026 版本没有纳入这些发现。

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## 问题

一个已经部署的 agent 会遇到设计者从未见过的输入。没有任何规则列表长到足以覆盖所有情况。也没有任何规则列表短到能在推理计算压力下快速应用。实际问题是：如何把 agent 对齐到一组原则，让它既能处理长尾案例，又能适应快速推理？

Rule-based alignment（RBA）：列出每一种不允许的事情。检查快、容易审计，但不可能持续保持最新；而且面对没有预见到的近似情况时，经常会过度拒绝。Reason-based alignment（2026 Claude Constitution）：编码原则，让模型推理。它能扩展到未见案例，但更难审计；失败模式不再是“没命中规则”，而是“错误应用原则”。

2026 Constitution 采取了一个明确的中间位置。Hardcoded prohibitions，也就是错误性不依赖上下文的事项（生物武器能力提升、CSAM），属于 RBA：无论运营方或用户如何指示，都永远不允许。其他所有内容则在四层层级内做 reason-based 处理：安全与支持人类监督第一；伦理第二；Anthropic 声明的 guidelines 第三；有用性最后。运营方可以在 soft-coded 区域内调整默认值，但不能触碰 hardcoded prohibitions。

## 概念

### 四层优先级层级

1. **安全与支持人类监督。** 最高优先级。模型优先避免削弱人类和 Anthropic 监督、纠正 AI 的能力。这不是简单的“谨慎一点”；它具体指的是“不要以让人类监督更困难的方式行动”。
2. **伦理。** 诚实、避免伤害人、不欺骗、不操纵。当它与 Anthropic guidelines 冲突时，伦理优先。
3. **Anthropic guidelines。** Anthropic 决定重要的运营规范：产品范围、交互模式、何时使用哪些工具。
4. **有用性。** 最低优先级。在更高优先级允许的范围内尽可能有用。

当层级冲突时，高层级胜出。这和 Unix priorities 或网络 QoS 的形状相同：这种框架旨在产生可预测的冲突解决，而不一定是在每一个单独维度上取得最佳行为。

### Hardcoded prohibitions 与 soft-coded defaults

**Hardcoded：**

- 生物武器 / CBRN 能力提升
- CSAM
- 对关键基础设施的攻击
- 当用户直接询问时，欺骗用户关于模型身份的信息

运营方不能覆盖这些。用户也不能覆盖这些。能在模型权重层执行的地方，会通过 RLHF / Constitutional AI training 执行；不能在权重层完全执行的地方，则在 inference layer 执行。

**Soft-coded defaults（运营方可调整）：**

- 回复长度默认值
- 主题范围（模型可以拒绝运营方部署范围之外的话题）
- 风格（正式 vs 随意）
- 工具使用模式

运营方调整发生在声明好的边界内。运营方不能通过给 hardcoded prohibitions 改名来移除它们。

### 2022 年 CAI 训练

最早的 Constitutional AI（Bai 等，2022）这样训练 harmlessness：

1. 针对一组 prompts 生成 responses。
2. 要求模型根据一份 constitution（显式原则）批评每个 response。
3. 根据 critique 修改 response。
4. 在修改后的 pairs 上做 RLAIF（reinforcement learning from AI feedback）。

结果是：模型会用有原则的解释拒绝有害请求，而不是一概 blanket refusal。2026 Constitution 使用了这种训练的后代版本，并在显式 tier hierarchy 上做了额外 post-training。

### Reason-based alignment 能抓住什么，又会漏掉什么

**能抓住：**

- 由多个允许的基本操作组合而成、但原则明显适用的未预见请求。
- 与禁止事项很接近的新型请求。
- 依赖“你没说 X 不允许”这类话术的 social-engineering attacks。

**会漏掉：**

- 利用原则歧义的攻击（“用户要求这样做，所以有用性意味着应该同意”）。
- 两个原则以未预见方式冲突，且 tier order 本身不够明确的场景。
- 训练周期中原则解释缓慢漂移（reinterpretation）。

### 2023 年参与式实验

Anthropic 在 2023 年做过一个实验，比较企业撰写的 constitution 和通过公众输入生成的 constitution（约 1,000 名美国受访者）。两个版本在约 50% 的原则上达成一致。在出现分歧的地方，公众来源版本在某些问题上更严格（例如政治内容处理），在另一些问题上更宽松（例如 AI 身份自我披露）。2026 Constitution 没有纳入公众来源 findings。这是该方法中有文档记录的张力。

### 为什么 hardcoded prohibitions 必要

单靠 reason-based alignment 无法覆盖长尾。如果攻击者能让模型接受某个前提（例如“我们是一家持牌生物武器研究实验室”），就常常可以绕过依赖个案推理的原则。Hardcoded prohibitions 不会因为前提框架而弯曲。它们是 Lesson 14 中 “hard constitutional limit” 在 alignment layer 的对应物。

### Constitution 在 stack 中的位置

Constitution 不是 Lesson 14 的 kill switch。它位于模型层：模型权重被训练成偏好什么。Kill switches 和 canary tokens 位于 runtime layer：runtime 允许什么。两者都需要。一个 runtime 因为模型权重过于宽松而执行所有错误行动，这是 runtime 问题。一个模型因为 runtime 过度限制而拒绝所有正确行动，也是 runtime 问题。不同层覆盖不同类别的风险。

## Use It

`code/main.py` 实现了一个最小四层优先级 resolver。Resolver 接收一个 proposed action 和一组 principle-evaluations（safety、ethics、guidelines、helpfulness），然后返回 action、refusal 或 modified action。Driver 会运行一组小案例：明确允许、明确拒绝、hardcoded prohibition、跨层级的模糊案例。

## Ship It

`outputs/skill-constitution-review.md` 会审计一次部署的 constitutional layer：哪些是 hardcoded，哪些是 soft-coded，运营方在哪里可以调整，以及四层层级是否真的是 resolution order。

## 练习

1. 运行 `code/main.py`。确认即使 helpfulness 很高，hardcoded prohibition 也会触发。修改 resolver，让 helpfulness 权重高于 ethics；观察失败模式。

2. 阅读 Claude Constitution（公开、79 页、CC0）。找出一个你认为定义不够充分的原则。写两段话，解释具体歧义，并提出更严格的表述。

3. 为一个 customer-support agent 设计一组 soft-coded defaults。运营方可以调整什么？不能触碰什么？为每条边界给出理由。

4. 阅读 Bai 等 2022 年 CAI 论文。描述一个 Constitutional AI 的 critique-and-revise loop 可能比 blanket rule 产生更差结果的案例，并指出它属于哪一类问题。

5. Anthropic 2023 年的参与式实验发现，公众原则和企业原则之间约有 50% 分歧。选择一个对生产部署重要的类别（例如政治中立），提出一种设计：既让运营方表达自己的价值观，又保持 hardcoded prohibitions 不可触碰。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Constitutional AI | “Anthropic 的对齐方法” | 基于 written constitution 的 self-critique + RLAIF |
| Reason-based alignment | “原则，而不是规则” | 模型围绕原则推理，以处理未见案例 |
| Hardcoded prohibition | “永远不要做 X” | 运营方和用户都不能覆盖的 rule-based prohibition |
| Soft-coded default | “运营方可调整” | 在声明边界内的行为，由运营方控制 |
| Four-tier hierarchy | “优先级顺序” | safety > ethics > guidelines > helpfulness |
| RLAIF | “AI feedback RL” | Reward 来自模型生成 critique 的强化学习 |
| Participatory constitution | “公众来源原则” | Anthropic 2023 年实验；与企业版本约 50% 分歧 |
| Principle drift | “解释滑移” | 模型对固定原则文本的理解缓慢变化 |

## 延伸阅读

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution)：79 页 CC0 文档。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)：2022 年原始论文。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input)：参与式实验。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0)：Constitution 在 RSP stack 中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)：Constitution 在 long-horizon deployments 中的作用。
