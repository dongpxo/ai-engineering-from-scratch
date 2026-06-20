# CAIS、CAISI 与社会尺度风险

> Center for AI Safety（CAIS，旧金山，2022 年由 Hendrycks 和 Zhang 创立）发布了 four-risk framework：malicious use、AI races、organizational risks、rogue AIs；也发布了 2023 年 5 月关于 extinction risk 的声明，数百名教授和公司领导签署。CAIS 在 2026 年的发布包括：用于 frontier-model evaluation 的 AI Dashboard、Remote Labor Index（与 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers newsletter。另一个不同实体：NIST Center for AI Standards and Innovation（CAISI），面向美国政府，负责 voluntary agreements 和 unclassified capability evaluations，重点关注 cyber、bio 与 chemical-weapons risks。CAIS 把 organizational risk 标为四个顶层风险之一：safety culture、rigorous audits、multi-layered defenses 和 information security 都是基础，但经常被部署速度压过。California SB-53 如果签署，将成为美国第一个州级 catastrophic-risk regulation。

**Type:** Learn
**Languages:** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time:** ~45 minutes

## 问题

第 19 和 20 课覆盖了 lab-internal scaling policies。第 21 课覆盖了独立 capability evaluation。本课覆盖第三个视角：塑造 public discussion 和 catastrophic AI risk regulatory baseline 的 civil society 与 government organizations。

有两个不同实体很重要。CAIS 是一家 non-profit research org，发布 AI risk 框架并协调 public statements。CAISI 是 NIST 下属的美国政府中心，负责与 labs 的 voluntary agreements 和 unclassified capability evaluations。名字押韵，但使命不重叠。实践者应该同时了解二者。

实际内容是：CAIS 的 four-risk framework 是文献中引用最广泛的 societal-scale-risk taxonomy。Safety culture 和 organizational risk 是这四类之一，也最直接受实践者控制。SB-53（California）如果签署，将成为美国第一个州级 catastrophic-risk regulation；该法案的 framing 很重要，因为在美国科技政策中，州级 regulation 历史上常常先于联邦行动。

## 概念

### CAIS — Center for AI Safety

- 成立：2022 年，旧金山，由 Dan Hendrycks 和同事创立（“Zhang” 指早期合作者，不是现任 co-founder；当前 leadership 见 CAIS 网站）。
- 状态：501(c)(3) non-profit。
- 2023 年重要输出：关于 extinction risk 的 statement，由数百名 researchers 和 CEOs 共同签署。声明称：“Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war.”
- 2026 年输出：用于 frontier-model evaluation 的 AI Dashboard、Remote Labor Index（与 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers newsletter。

### Four-risk framework

CAIS 的框架把 catastrophic AI risk 分成四个顶层类别：

1. **Malicious use**：坏行为者使用 AI 造成伤害（bioweapons synthesis、disinformation、cyberattacks）。
2. **AI races**：labs、companies 或 nations 之间的竞争压力，把部署推过安全边界。
3. **Organizational risks**：lab 内部动态（safety-culture failures、insufficient audit、under-resourced security）导致糟糕部署。
4. **Rogue AIs**：足够强的 AI 追求与 human welfare 冲突的目标。

这不是唯一 taxonomy，但它是最常被引用的。几个类别并不互斥：一个由组织在 race 中为了速度牺牲 audit 而生产出来的 rogue AI，可以同时属于四类。

### Organizational risk 在哪里

在四个类别中，organizational risk 对实践者来说最可行动。一个 lab 的 safety culture、audit rigor、defense layering 和 information security，会决定第 10-18 课的控制措施是真的落地了，还是只停留在没人验证的 checklist items 上。

具体的 organizational-risk levers：

- **Safety culture**：团队成员是否能在不付出职业代价的情况下升级 concern？CAIS surveys 发现这是其他 levers 的强预测因子。
- **Rigorous audits**：external 和 internal 都需要。只有 internal audits 往往会产出过于乐观的报告。
- **Multi-layered defenses**：没有单一层是充分的（这是 Phase 15 的主线）。
- **Information security**：model weights 泄露、eval data 泄露、monitor-bypass techniques 泄露。第 19 课的 RAND SL-4 是一个具体标准。

### CAISI — Center for AI Standards and Innovation

- 在 NIST 内运作。
- 与 frontier labs 运行 voluntary agreements。
- 发布 focused on cyber、bio 和 chemical-weapons risks 的 unclassified capability evaluations。
- 与 CAIS 不同；缩写很像；阅读时检查 URL（nist.gov）来确认你读的是哪一个。

CAISI 的角色是 METR private lab engagements（第 21 课）的 public、government-facing counterpart。CAISI reports 是 unclassified；METR reports 经常受 NDA 限制。实践者同时阅读两者，可以获得更完整的图景。

### California SB-53

California Senate bill（2025-2026 session）处理 frontier models 带来的 catastrophic risk。草案中的关键条款包括：

- 触发州级义务的 specific capability thresholds。
- AI lab employees 的 whistleblower protections。
- Catastrophic failures 的 incident reporting requirements。

如果签署，它将成为美国第一个州级 catastrophic-risk regulation。不管是否签署，该法案的 framing 都会影响其他州 legislature 如何处理这个问题。加州实践者应该追踪法案状态；其他地区的实践者也应阅读它，以理解美国州级 regulation 可能长什么样。

### Societal-scale risk 不是单层问题

Phase 15 的主线 defense in depth，也适用于社会层面。没有任何单个组织、regulation 或 framework 能关闭 catastrophic risk。只有当以下部分共同运作时，生态系统才有效：

- Labs 发布 scaling policies（第 19、20 课）。
- External evaluators 产出 measurements（第 21 课）。
- Civil society 追踪并公开传播（CAIS）。
- Government 运行 voluntary programs 和 baseline regulation（CAISI、SB-53）。
- Practitioners 构建 multi-layered controls（第 10-18 课）。

这是本阶段的最终综合：前面每一课都是一个 stack layer；整个 stack 的完整性比任何单层的强度更重要。

## Use It

`code/main.py` 实现了一个小型 risk-inventory tool。给定一个 proposed deployment，它会按 four-risk categories 给 deployment 打标签，并返回 mitigation checklist。它是阅读框架的辅助工具，不是 human judgment 的替代品。

## Ship It

`outputs/skill-societal-risk-review.md` 会审查某个 deployment 的 societal-scale-risk posture：它触及四类中的哪些，已有 mitigations 是什么，organizational-risk exposure 有多大。

## 练习

1. 运行 `code/main.py`。输入三个不同 scale 的 synthetic deployments。确认 four-risk tags 符合你的预期；找出一个 tool under-tag 或 over-tag 的案例。

2. 完整阅读 CAIS four-risk paper。选择一个 risk category，用两段话写出你认为该类别在 2026 年最重要的发展。

3. 阅读 California SB-53 的当前 draft。找出一个你认为会强化 catastrophic-risk posture 的条款，以及一个你认为会削弱它的条款。分别说明理由。

4. 选择一个你了解的 production AI deployment（你自己的或公开的）。按 organizational-risk sub-levers 给它评分：safety culture、audit rigor、multi-layered defenses、information security。哪一项最弱？把它补齐到合理水平要付出什么成本？

5. 草拟一个 2028 版 four-risk framework，反映额外一年 capability 和额外一年 deployment experience。你会添加、删除或重组什么？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| CAIS | “Center for AI Safety” | Non-profit；four-risk framework；2023 extinction statement |
| CAISI | “US government AI safety” | NIST Center；voluntary agreements；unclassified evals |
| Four-risk framework | “CAIS 的 taxonomy” | malicious use、AI races、organizational risks、rogue AIs |
| Malicious use | “坏行为者使用 AI” | Bioweapons、disinformation、cyberattacks |
| AI races | “竞争压力” | Labs/companies/nations 把部署推过安全边界 |
| Organizational risk | “Lab internal failure” | Safety culture、audit、defenses、infosec |
| Rogue AI | “Misaligned agent” | 有能力的 AI 追求与 human welfare 冲突的目标 |
| California SB-53 | “State-level regulation” | 2025-2026 bill；如果签署，将成为美国第一个州级 catastrophic-risk regulation |

## 延伸阅读

- [Center for AI Safety](https://safe.ai/)：four-risk framework 的 institutional home。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk)：four-risk paper。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk)：简短 joint statement。
- [NIST CAISI](https://www.nist.gov/caisi)：government-facing AI standards and innovation center。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)：把 lab-level commitments 连接到 societal-scale framing。
