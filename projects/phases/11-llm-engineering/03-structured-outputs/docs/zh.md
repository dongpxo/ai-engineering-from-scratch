# 结构化输出：JSON、Schema 校验与约束解码

> 你的 LLM 返回的是字符串。你的应用需要的是 JSON。这个落差让生产系统崩溃的次数，比模型幻觉还多。结构化输出就是自然语言和类型化数据之间的桥。做对了，LLM 就会变成可靠的 API；做错了，你就是在凌晨 3 点用正则解析自由文本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 minutes
**Related:** Phase 5 · 20 (Structured Outputs & Constrained Decoding) 讲解 decoder 层面的理论（FSM/CFG logit processors、Outlines、XGrammar）。本课聚焦生产 SDK 表面（OpenAI `response_format`、Anthropic tool use、Instructor）。如果你想先理解 API 下面发生了什么，请先读 Phase 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON mode 与 schema-constrained outputs
- 构建一个 Pydantic 校验层，拒绝格式错误的 LLM 输出，并带着错误反馈重试
- 解释约束解码如何在 token 层强制生成合法 JSON，而不是事后修补
- 设计可靠的抽取 prompt，把非结构化文本稳定转换成类型化数据结构

## 问题

你向 LLM 提问："Extract the product name, price, and availability from this text." 它回答：

```text
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是一个完全正确的回答。但对你的应用来说，它完全没用。库存系统需要的是 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要一个 JSON object，里面有特定 key、特定类型和特定值约束。你不需要一句自然语言描述。

最天真的方案：在 prompt 里加一句 "Respond in JSON"。这在 90% 的情况下可行。剩下 10% 里，模型会把 JSON 包进 markdown code fence，或者加上 "Here's the JSON:" 这样的前言，或者因为提前闭合括号而生成语法错误的 JSON。你的 JSON parser 崩了。你的 pipeline 断了。你加上 try/except 和 retry loop。重试有时又生成不同数据。于是你在 parsing 问题之外，又多了 consistency 问题。

这不是 prompt engineering 问题。这是 decoding 问题。模型从左到右生成 token。每个位置上，它都从 10 万以上的 vocabulary 里挑选最可能的下一个 token。在任意给定位置，绝大多数 token 都会产生非法 JSON。如果模型刚输出了 `{"price":`，下一个 token 必须是数字、引号（表示 string）、`null`、`true`、`false` 或负号。其他任何 token 都会产生非法 JSON。没有约束时，模型可能会挑一个看起来合理的英文词，但语法上是灾难。

## 核心概念

### 结构化输出的光谱

结构化输出控制有四个层级，越往后越可靠。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**Prompt-based**（"Respond in valid JSON"）：没有强制约束。模型通常会配合，但偶尔不会。可靠性：约 90%。失败模式：markdown fence、前言文本、截断输出、结构错误。

**JSON mode**：API 保证输出是合法 JSON。OpenAI 的 `response_format: { type: "json_object" }` 会启用它。输出一定可以 parse。但它不一定符合你期望的 schema：可能有多余 key、错误类型、缺失字段。

**Schema mode**：API 接收 JSON Schema，并保证输出匹配它。到 2026 年，主流 provider 都原生支持这一层：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可以通过 `tool_choice="required"`）、Anthropic 带 `input_schema` 的 tool use、Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出会包含你指定的精确 key、类型和约束。

**Constrained decoding**：在 generation 的每个 token 位置，decoder 会 mask 掉所有会产生非法输出的 token。如果 schema 要求一个 number，而模型准备输出一个字母，这个 token 的概率会被设为 0。模型只能生成会通向合法输出的 token。这就是 OpenAI structured output mode 以及 Outlines、Guidance 等库在底层实现的机制。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或校验层）输出必须长什么样的方式。每个主流 structured output system 都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 schema 表示：输出必须是一个 object，包含 string 类型的 `product`、非负 number 类型的 `price`、boolean 类型的 `in_stock`，以及一个可选的 string array `categories`。任何不匹配的输出都会被拒绝。

Schema 能处理难点：nested objects、带类型 item 的 arrays、enums（把 string 限制到特定值）、pattern matching（字符串上的 regex）以及 combinators（`oneOf`、`anyOf`、`allOf`，用于多态输出）。

### Pydantic 模式

在 Python 里，你通常不会手写 JSON Schema。你会定义一个 Pydantic model，再让它生成 schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会生成和上面类似的 JSON Schema。Instructor library（以及 OpenAI SDK）可以直接接收 Pydantic models：传入 model class，拿回 validated instance。如果 LLM 输出不匹配，Instructor 会自动 retry。

### Function Calling / Tool Use

这是同一个问题的另一种接口。你不直接要求模型生成 JSON，而是定义带类型参数的 "tools"（functions）。模型输出一个带结构化 arguments 的 function call。OpenAI 称它为 "function calling"。Anthropic 称它为 "tool use"。结果一样：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个 function，而不是只填参数时，tool use 更合适。如果你有 10 种不同的 extraction schema，并且模型必须根据输入选择正确 schema，tool use 会同时给你 schema selection 和 structured output。

### 常见失败模式

即使有 schema enforcement，structured outputs 也可能以微妙方式失败。

**Hallucinated values**：输出匹配 schema，但包含编造的数据。文本里明明写的是 $348，模型却输出 `{"price": 299.99}`。Schema validation 抓不到这个问题：类型正确，值错误。

**Enum confusion**：你把字段限制为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`，语义上正确，但不在允许集合中。好的 constrained decoding 会阻止这种情况。Prompt-based 方法不会。

**Nested object depth**：深度嵌套的 schema（4 层以上）更容易出错。每增加一层嵌套，模型就多一个迷失结构的位置。

**Array length**：模型可能在 array 中输出过多或过少 item。Schema 支持 `minItems` 和 `maxItems`，但不是所有 provider 都在 decoding 层强制它们。

**Optional field omission**：模型会省略技术上可选、但对你的 use case 很重要的字段。即使数据有时缺失，也可以把它们设为 required，强制模型显式输出 `null`。

## 动手构建

### Step 1: JSON Schema Validator

从零构建一个 validator，检查 Python object 是否匹配 JSON Schema。这就是在输出侧运行、验证 compliance 的逻辑。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### Step 2: Pydantic-Style Model to Schema

构建一个最小 class-to-schema converter。定义 Python class，然后自动生成它的 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### Step 3: Constrained Token Filter

模拟 constrained decoding。给定一个 partial JSON string 和一个 schema，判断当前位置哪些 token categories 是合法的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### Step 4: Extraction Pipeline

把所有组件组合成 extraction pipeline：定义 schema，模拟 LLM 生成 structured output，校验输出，并处理 retries。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### Step 5: Run the Full Pipeline

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 实战使用

### OpenAI Structured Outputs

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的 structured output mode 在内部使用 constrained decoding。模型生成的每个 token 都被保证会产生匹配 Pydantic schema 的输出。不需要 retries。不需要 validation。约束已经烘进 decoding 过程里。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过 tool use 实现 structured output。模型会发出一个 tool call，里面带有匹配 `input_schema` 的 structured arguments。结果相同，API 表面不同。

### Instructor Library

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 会包装任意 LLM client，并加上 validation 失败后的自动 retry。如果第一次尝试没通过 validation，它会把错误作为 context 发回给模型，请模型修正输出。它适用于任何 provider，不只 OpenAI。

## 交付

本课产出 `outputs/prompt-structured-extractor.md`：一个可复用的 prompt template。给它 JSON Schema 和非结构化文本，它会抽取结构化数据并返回 validated JSON。

它还产出 `outputs/skill-structured-outputs.md`：一个决策框架，用来根据 provider、可靠性要求和 schema complexity 选择合适的 structured output 策略。

## 练习

1. 扩展 schema validator，让它支持 `oneOf`（数据必须恰好匹配多个 schemas 中的一个）。这可以处理 polymorphic outputs，例如某个字段既可能是 `Product` object，也可能是形状不同的 `Service` object。

2. 构建一个 "schema diff" tool，比较两个 schemas，并识别 breaking changes（移除了 required fields、改变了 types）和 non-breaking changes（添加 optional fields、放宽 constraints）。这对 production 中 extraction schemas 的版本管理非常关键。

3. 实现一个更真实的 constrained decoding simulator。给定 JSON Schema 和一个 100 token 的 vocabulary（letters、digits、punctuation、keywords），一步步走过 generation，在每个位置 mask invalid tokens。测量每一步 vocabulary 中有多少比例是 valid 的。

4. 构建 extraction eval suite。创建 50 条产品描述，并手工标注 JSON outputs。让你的 extraction pipeline 跑完这 50 条，测量 exact match、field-level accuracy 和 type compliance。找出最难正确抽取的字段。

5. 给 extraction pipeline 增加 "confidence scores"。为每个抽取字段估计模型置信度（基于 token probabilities，或重复抽取 3 次并测量一致性）。把低置信字段标记给人工 review。

## 关键术语

| Term | 人们通常怎么说 | 它实际是什么意思 |
|------|----------------|------------------|
| JSON mode | "Returns JSON" | API flag，保证输出是语法合法的 JSON，但不强制任何特定 schema |
| Structured output | "Typed JSON" | 匹配特定 JSON Schema 的输出，包含正确 keys、types 和 constraints |
| Constrained decoding | "Guided generation" | 在每个 token 位置 mask 会产生非法输出的 tokens，保证 100% schema compliance |
| JSON Schema | "A JSON template" | 描述 JSON 数据结构、类型和约束的声明式语言（OpenAPI、JSON Forms 等都使用它） |
| Pydantic | "Python dataclasses+" | Python library，用类型校验定义 data models，FastAPI 和 Instructor 用它生成 JSON Schemas |
| Function calling | "Tool use" | LLM 输出结构化 function invocation（name + typed arguments），而不是自由文本；OpenAI 和 Anthropic 都支持 |
| Instructor | "Pydantic for LLMs" | 包装 LLM clients，使其返回 validated Pydantic instances，并在 validation failure 时自动 retry 的 Python library |
| Token masking | "Filtering the vocabulary" | 在 generation 时把特定 token probabilities 设为 0，让模型无法生成它们 |
| Schema compliance | "Matches the shape" | 输出包含每个 required field、正确 types、约束范围内的 values，并且没有不允许的 extra fields |
| Retry loop | "Try again until it works" | 把 validation errors 发回给模型并要求它修正输出；Instructor 会自动执行，直到达到可配置上限 |

## 延伸阅读

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- OpenAI API 中基于 JSON Schema 的 constrained decoding 官方文档
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) -- Outlines paper，讲解如何把 JSON Schemas 编译成 finite state machines 来实现 token-level constraints
- [Instructor documentation](https://python.useinstructor.com/) -- 使用 Pydantic validation 和 retries，从任意 LLM 获取 structured outputs 的标准 library
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) -- Claude 如何通过带 JSON Schema `input_schema` 的 tool use 实现 structured output
- [JSON Schema specification](https://json-schema.org/) -- 每个主流 structured output system 都使用的 schema language 完整规范
- [Outlines library](https://github.com/outlines-dev/outlines) -- 开源 constrained generation：把 regex 和 JSON Schema 编译成 finite state machines
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100) -- 当前最先进的 grammar engine；pushdown-automaton compilation，以约 100 ns / token 的速度 mask tokens
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094) -- LMQL paper，把 constrained decoding 表述为带类型和取值约束的 query language
- [Microsoft Guidance (framework docs)](https://github.com/guidance-ai/guidance) -- template-driven constrained generation；是 Outlines 和 XGrammar 的 vendor-agnostic 补充
