# 结构化输出：JSON Schema、Pydantic、Zod 与约束解码

> “礼貌地要求模型返回 JSON” 即使在前沿模型上也会有 5% 到 15% 的失败率。结构化输出用约束解码补上这个缺口：模型从字面上就被阻止生成任何违反 schema 的 token。OpenAI 的 strict mode、Anthropic 带 schema 类型的 tool use、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`，以及 Zod 的 `.parse`，都是同一个思想的五种表面形式。本课会构建 schema validator 和 strict-mode contract，学习者之后每条生产级抽取 pipeline 都会用到它们。

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 学习目标

- 使用合适的约束（enum、min/max、required、pattern）为抽取目标编写 JSON Schema 2020-12。
- 解释 strict mode 和约束解码为什么提供了不同于“生成后再校验”的保证。
- 区分三种失败模式：解析错误、schema 违规、模型拒答。
- 交付一条带类型化修复和类型化拒答处理的抽取 pipeline。

## 问题

一个读取采购订单邮件的 agent，需要把自由文本转换成 `{customer, line_items, total_usd}`。有三种做法。

**做法一：提示模型返回 JSON。** “Reply in JSON with fields customer, line_items, total_usd.” 在前沿模型上有 85% 到 95% 的时间可用。它有六类失败：缺少大括号、尾随逗号、类型错误、幻觉字段、在 token limit 处截断，以及泄漏 “Here is your JSON:” 这样的散文。

**做法二：生成后校验。** 让模型自由生成，解析结果，用 schema 校验，失败后重试。可靠但昂贵，因为每次重试都要付费；截断 bug 每出现一次，就会多花一个 turn。

**做法三：约束解码。** Provider 在解码阶段强制执行 schema。无效 token 会从采样分布中被 mask 掉。输出保证可以解析，也保证通过校验。失败会收敛为一种模式：拒答（模型判断输入不适合该 schema）。

到 2026 年，每个前沿 provider 都提供了某种形式的做法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，如果模型拒绝回答，response 中会带 `refusal`。
- **Anthropic。** 对 `tool_use` inputs 做 schema enforcement；没有 `stop_reason: "refusal"` 这种信号，`end_turn` 且没有 tool call 就是信号。
- **Gemini。** 在 request 级别提供 `responseSchema`；到 2026 年，Gemini 对选定类型提供 token-level grammar constraints。
- **Pydantic AI。** `output_type=InvoiceModel` 会产出类型为 `InvoiceModel` 的 structured `RunResult`。
- **Zod（TypeScript）。** Runtime parser，用 Zod schema 校验 provider output；可以和 OpenAI 的 `beta.chat.completions.parse` 搭配使用。

共同点是：只声明一次 schema，然后端到端执行它。

## 概念

### JSON Schema 2020-12：通用语言

每个 provider 都接受 JSON Schema 2020-12。最常用的构造包括：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到 subschema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用到每个数组元素的 subschema。
- `additionalProperties`：`false` 会禁止额外字段（默认值随模式而变）。

OpenAI strict mode 额外增加三个要求：每个 property 都必须列在 `required` 里；每一层都必须有 `additionalProperties: false`；不能有未解析的 `$ref`。如果违反这些要求，API 会在 request 阶段返回 400。

### Pydantic：Python 绑定

Pydantic v2 可以通过 `model_json_schema()` 从 dataclass 形状的 model 生成 JSON Schema。Pydantic AI 把这一层包起来，所以你可以写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent framework 会在边界处把 schema 翻译成 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出会作为类型化的 `Invoice` instance 返回。Validation error 会以带类型化 error path 的 `ValidationError` 抛出。

### Zod：TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 中的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，会把它翻译成 API 的 JSON Schema payload。

### 拒答

Strict mode 不能强迫模型回答。如果输入无法适配 schema（“这封邮件是一首诗，不是发票”），模型会输出包含原因的 `refusal` 字段。你的代码必须把它作为一等 outcome 处理，而不是当成失败。拒答也很适合作为安全信号：如果要求模型从受保护内容邮件中抽取信用卡号，模型会返回带安全原因的 refusal。

### 开放实现中的约束解码

Open-weights 实现通常使用三种技术。

1. **基于 grammar 的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建 deterministic finite automaton；每一步都 mask 掉会违反 FSM 的 token logits。
2. **带 JSON parser 的 logit masking**：用 streaming JSON parser 和模型同步前进；每一步计算 valid-next-token set。
3. **带 verifier 的 speculative decoding**：廉价 draft model 提出 token，verifier 强制执行 schema。

商业 provider 会在幕后选择其中一种。2026 年的最新状态是：对短结构化输出，它比普通生成更快；对长输出，速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是合法 JSON。在 strict mode 下不会发生。在非 strict provider 上仍可能发生。
2. **Schema 违规。** 输出能解析，但违反 schema。在 strict mode 下不会发生。在 strict 之外很常见。
3. **拒答。** 模型拒绝回答。必须作为类型化 outcome 处理。

### 重试策略

当你不在 strict mode 中时（Anthropic tool use、非 strict OpenAI、旧版 Gemini），恢复模式是：

```text
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常一次重试就够。三次重试能兜住弱模型的偶发失败。超过三次通常说明 schema 设计有问题：模型对某些输入无法满足它，需要修正 prompt 或 schema。

### 小模型支持

约束解码也适用于小模型。一个带 grammar enforcement 的 3B 参数开放模型，在结构化任务上会超过一个只靠原始 prompt 的 70B 参数模型。这是结构化输出对生产环境重要的主要原因：它把可靠性和模型规模解耦。

## Use It

`code/main.py` 提供了一个只用 stdlib 写成的最小 JSON Schema 2020-12 validator（types、required、enum、min/max、pattern、items、additionalProperties）。它包装一个 `Invoice` schema，并把一个 fake LLM output 送进 validator，演示解析错误、schema 违规和拒答路径。生产环境里可以把 fake output 换成任意 provider 的真实 response。

值得关注的地方：

- Validator 返回类型化的 `[ValidationError]` list，里面包含 path 和 message。这就是你应该暴露给 retry prompt 的形状。
- 拒答分支不会重试。它会记录并返回一个类型化 refusal。Phase 14 · 09 会把 refusals 作为安全信号使用。
- `additionalProperties: false` 检查会在 adversarial test input 上触发，展示 strict mode 为什么能关上幻觉字段这扇门。

## Ship It

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本抽取目标（发票、支持工单、简历等），该 skill 会产出一个兼容 strict mode 的 JSON Schema 2020-12，以及一个镜像它的 Pydantic model，并预留类型化拒答和重试处理 stub。

## 练习

1. 运行 `code/main.py`。加入第四个测试用例，让它的 `total_usd` 是负数。确认 validator 会通过 `minimum` constraint path 拒绝它。

2. 扩展 validator，让它支持带 discriminator 的 `oneOf`。常见场景：`line_item` 可以是 product 或 service，通过 `kind` 标记。Strict mode 在这里有细微规则；请查看 OpenAI 的 structured outputs guide。

3. 用 Pydantic BaseModel 写出同一个 Invoice schema，并把 `model_json_schema()` 输出和手写 schema 对比。找出一个 Pydantic 默认会设置、但手写版本省略的字段。

4. 测量拒答率。构造十个不应该被抽取的输入（一段歌词、一个数学证明、一封空邮件），用真实 provider 的 strict mode 跑它们。统计 refusals 和 hallucinated outputs。这就是 refusal-aware retries 的 ground truth。

5. 从头到尾阅读 OpenAI 的 structured outputs guide。找出一个 strict mode 明确禁止、但普通 JSON Schema 允许的构造。然后设计一个非必要地使用该禁止构造的 schema，并把它重构成 strict-compatible。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “schema 规范” | 每个现代 provider 都支持的 IETF-draft schema dialect |
| Strict mode | “保证 schema” | OpenAI flag，通过约束解码强制执行 schema |
| Constrained decoding | “Logit masking” | 解码阶段的 enforcement，会 mask 掉无效 next-token |
| Refusal | “模型拒绝” | 输入无法适配 schema 时的类型化 outcome |
| Parse error | “无效 JSON” | 输出不能解析为 JSON；strict mode 下不可能发生 |
| Schema violation | “形状错误” | 已解析，但违反 type / required / enum / range |
| `additionalProperties: false` | “不允许额外字段” | 禁止未知字段；OpenAI strict mode 要求使用 |
| Pydantic BaseModel | “类型化输出” | 会生成并校验 JSON Schema 的 Python class |
| Zod schema | “TypeScript 输出类型” | 用于校验 provider output 的 TS runtime schema |
| Grammar enforcement | “Open-weights 约束解码” | 基于 FSM 的 logit masking，例如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、refusals 和 schema requirements
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月的发布文章，解释 decoding guarantee
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 会序列化到各 provider 的类型化 `output_type` bindings
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — canonical spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — enterprise deployment notes 和 strict-mode caveats
