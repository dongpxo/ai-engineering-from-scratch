# 函数调用与工具使用

> LLM 什么也做不了。它们生成文本。这就是全部能力。它们不能查天气、查询数据库、发送邮件、运行代码或读取文件。你见过的每一个“AI agent”，本质上都是 LLM 生成一段 JSON，说明要调用哪个函数，然后由你的代码真正执行调用。模型是大脑，工具是双手，函数调用就是连接两者的神经系统。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 03 (Structured Outputs)
**Time:** ~75 minutes
**Related:** Phase 11 · 14 (Model Context Protocol) — 当一个工具需要跨 host 共享时，就应该从内联 function calling 升级到 MCP server。本课覆盖内联场景；MCP 覆盖协议场景。

## 学习目标

- 实现一个函数调用循环：定义工具 schema，解析模型的 tool-call JSON，执行函数，并返回结果
- 设计带有清晰描述和类型化参数的工具 schema，让模型能够可靠调用
- 构建一个多轮 agent 循环，通过串联多个函数调用回答复杂查询
- 处理函数调用边界情况：并行工具调用、错误传播，以及防止无限工具循环

## 问题

你构建了一个聊天机器人。用户问：“What's the weather in Tokyo right now?”

模型回答：“I don't have access to real-time weather data, but based on the season, Tokyo is likely around 15 degrees Celsius...”

这是一种披着免责声明外衣的幻觉。模型不知道天气。它永远不会知道。天气每小时都在变化，而模型训练数据已经是几个月前的了。

正确答案需要调用 OpenWeatherMap API，获取当前温度，然后返回真实数字。模型不能调用 API。你的代码可以。缺失的一环是：一个结构化协议，让模型能够说“我需要用这些参数调用天气 API”，再让你的代码执行调用并把结果喂回模型。

这就是函数调用。模型输出结构化 JSON，描述要调用哪个函数，以及使用哪些参数。你的应用执行函数。结果回到对话里。模型再使用这个结果生成最终答案。

没有函数调用，LLM 是百科全书。有了函数调用，它们才成为 agent。

## 概念

### 函数调用循环

每一次工具使用交互都遵循同一个 5 步循环。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

第 1 步：用户发送消息。第 2 步：模型收到消息，以及工具定义（描述可用函数的 JSON Schema）。第 3 步：模型不直接回复文本，而是输出一个 tool call，也就是包含函数名和参数的结构化 JSON 对象。第 4 步：你的代码执行函数并捕获结果。第 5 步：结果回到模型，模型现在拥有真实数据，可以生成最终答案。

模型永远不会执行任何东西。它只决定调用什么，以及用什么参数调用。你的代码才是执行器。

### 工具定义：JSON Schema 合约

每个工具都由一个 JSON Schema 定义。它告诉模型这个函数做什么、接收哪些参数，以及这些参数必须是什么类型。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description` 字段至关重要。模型会读取它们，决定何时以及如何使用工具。像“gets weather”这样的模糊描述，会比“Get current weather for a city. Returns temperature in Celsius and conditions.”产生更差的工具选择。description 本身就是工具选择提示词。

### Provider 对比

每个主流 provider 都支持函数调用，但 API 表面不一样。

| Provider | API Parameter | Tool Call Format | Parallel Calls | Forced Calling |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | 支持（单轮多个） | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | 支持（多个 block） | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | 支持 | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Llama 4 原生 `tools`；其他常用 Hermes 或 ChatML | 混合 | 取决于模型 | 基于提示，或在支持时使用 `tool_choice` |

到 2026 年，三家闭源 provider 已经收敛到几乎相同的、基于 JSON Schema 的格式。Llama 4 也提供了原生 `tools` 字段，与 OpenAI 的形状一致。开放权重微调模型仍然差异很大，其中 Hermes 格式（NousResearch）是第三方微调中最常见的形式。对于跨 host 共享的工具，优先使用 MCP（Phase 11 · 14），而不是内联 function calling；server 对所有 host 都是同一个。

### Tool Choice：Auto、Required、Specific

你可以控制模型什么时候使用工具。

**Auto**（默认）：模型自行决定是调用工具还是直接回答。“What's 2+2?”会直接回答。“What's the weather?”会调用工具。

**Required**：模型必须至少调用一个工具。当你知道用户意图一定需要工具时使用它。这样可以防止模型猜测，而不是查找真实数据。

**Specific function**：强制模型调用某个特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 会保证调用天气工具，不管查询内容是什么。这适合路由场景：上游逻辑已经判断出需要哪个工具。

### 并行函数调用

GPT-4o 和 Claude 可以在单轮中调用多个函数。用户问：“What's the weather in Tokyo and New York?” 模型会同时输出两个 tool call：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码执行两个调用（理想情况下并发执行），返回两个结果，然后模型合成一个回答。这会把往返次数从 2 次减少到 1 次。对每次查询需要 5 到 10 个工具调用的 agent，并行调用可以降低 60% 到 80% 的延迟。

### Structured Outputs vs Function Calling

第 03 课覆盖了结构化输出。函数调用使用同样的 JSON Schema 机制，但目的不同。

**Structured outputs**：强制模型生成特定形状的数据。输出本身就是最终产物。例如：从文本中抽取产品信息为 `{name, price, in_stock}`。

**Function calling**：模型声明它想执行某个动作。输出是一个中间步骤。例如：`get_weather(city="Tokyo")` 表示模型正在请求一个动作，而不是产出最终答案。

当你想要数据抽取时，使用 structured outputs。当你想让模型与外部系统交互时，使用 function calling。

### 安全：不可协商的规则

函数调用是你能赋予 LLM 的最危险能力。模型会选择要执行什么。如果你的工具集包含数据库查询，模型会构造查询。如果包含 shell 命令，模型会编写命令。

**规则 1：永远不要把模型生成的 SQL 直接传给数据库。** 模型可以而且会生成 DROP TABLE、UNION 注入，或返回所有行的查询。始终参数化。始终验证。始终使用操作 allowlist。

**规则 2：对函数使用 allowlist。** 模型只能调用你显式定义的函数。永远不要构建一个通用的“按名称执行任意函数”工具。如果你有 50 个内部函数，只暴露用户需要的那 5 个。

**规则 3：验证参数。** 模型可能传入一个 city 名称：`"; DROP TABLE users; --"`。执行前必须根据预期类型、范围和格式验证每个参数。

**规则 4：清理工具结果。** 如果工具返回敏感数据（API key、PII、内部错误），在发回模型之前先过滤。模型会逐字把工具结果包含进它的回复里。

**规则 5：限制工具调用速率。** 陷入循环的模型可能调用工具数百次。设置最大值（每次对话 10 到 20 次调用是合理的）。打断无限循环。

### 错误处理

工具会失败。API 会超时。数据库会宕机。文件可能不存在。模型需要知道工具什么时候失败，以及为什么失败。

把错误作为结构化工具结果返回，而不是抛出异常：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型会读取这个结果，调整参数，然后重试。模型很擅长从结构化错误消息中自我修正。它们不擅长从空响应或泛泛的“something went wrong”中恢复。

### MCP：Model Context Protocol

MCP 是 Anthropic 的开放工具互操作标准。与其让每个应用都定义自己的工具，不如使用 MCP 提供的通用协议：工具由 MCP server 提供，由 MCP client（例如 Claude Code、Cursor 或你的应用）消费。

一个 MCP server 可以把工具暴露给任何兼容 client。Postgres MCP server 让任何兼容 MCP 的 agent 访问数据库。GitHub MCP server 让任何 agent 访问仓库。工具定义一次，到处使用。

MCP 对函数调用的意义，就像 HTTP 对网络的意义。它标准化传输层，让工具变得可移植。

## Build It

### 第 1 步：定义工具注册表

构建一个注册表，用来存储工具定义和它们的实现。每个工具都有一个 JSON Schema 定义（模型看到的内容）和一个 Python 函数（你的代码执行的内容）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### 第 2 步：实现 5 个工具

构建计算器、天气查询、网页搜索模拟器、文件读取器和代码运行器。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### 第 3 步：注册所有工具

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### 第 4 步：构建函数调用循环

这是核心引擎。它模拟模型决定调用哪个工具，执行工具，并把结果喂回。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### 第 5 步：参数验证

构建一个验证器，在执行前根据 JSON Schema 检查 tool call 参数。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### 第 6 步：运行演示

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## Use It

### OpenAI Function Calling

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI 会把 tool call 返回在 `response.choices[0].message.tool_calls` 中。每个调用都有一个 `id`，你返回结果时必须包含它。模型会用这个 ID 匹配结果和调用。GPT-4o 可以在单个响应中返回多个 tool call，要逐个迭代并全部执行。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 会把 tool call 作为 `type: "tool_use"` 的 content block 返回。工具结果放在一个带有 `type: "tool_result"` 的 user message 里。注意关键差异：Anthropic 使用 `input_schema` 定义工具参数，而 OpenAI 使用 `parameters`。

### MCP 集成

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP 会把工具实现和工具消费解耦。Postgres server 懂 SQL。GitHub server 懂 API。你的 agent 只需要发现并调用工具，不需要为每个集成编写 provider-specific 代码。

## Ship It

本课产出 `outputs/prompt-tool-designer.md`：一个可复用的提示模板，用于设计工具定义。给它一段你希望工具完成什么的描述，它会生成完整的 JSON Schema 定义，包括描述、类型和约束。

本课还产出 `outputs/skill-function-calling-patterns.md`：一个在生产环境实现函数调用的决策框架，覆盖工具设计、错误处理、安全，以及 provider-specific 模式。

## 练习

1. **添加第 6 个工具：数据库查询。** 实现一个模拟 SQL 工具，使用内存表。工具接收表名和过滤条件（不是原始 SQL）。验证表名在 allowlist 中，并且过滤操作符只允许 `=`、`>`、`<`、`>=`、`<=`。把匹配行作为 JSON 返回。

2. **实现带错误反馈的重试。** 当 tool call 失败时（例如 city not found），把错误消息喂回模型决策函数，让它修正参数。记录每次调用需要多少次重试。每个 tool call 最多重试 3 次。

3. **构建多步骤 agent。** 有些查询需要串联工具调用：“Read the config file and tell me what model is configured, then search the web for that model's pricing.” 实现一个循环，一直运行到模型判断不再需要工具，并把累积结果传入每个决策步骤。限制为 10 次迭代，防止无限循环。

4. **测量工具选择准确率。** 创建 30 条测试查询，并标注预期工具名称。对全部 30 条运行你的决策函数，测量它选择正确工具的比例。找出哪些查询最容易在工具之间造成混淆。

5. **实现 tool call 缓存。** 如果 60 秒内同一个工具以完全相同参数再次被调用，就返回缓存结果，而不是重新执行。使用以 `(tool_name, frozenset(args.items()))` 为 key 的字典。测量一个包含 20 条查询的对话里的缓存命中率。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Function calling | “Tool use” | 模型输出结构化 JSON，描述要用特定参数调用的函数；执行函数的是你的代码，不是模型 |
| Tool definition | “Function schema” | 一个 JSON Schema 对象，描述工具名称、用途、参数和类型；模型读取它来决定何时以及如何使用工具 |
| Tool choice | “Calling mode” | 控制模型必须调用工具（required）、可以调用工具（auto），或必须调用指定工具（named） |
| Parallel calling | “Multi-tool” | 模型在单轮中输出多个 tool call，从而减少往返次数；GPT-4o 和 Claude 都支持 |
| Tool result | “Function output” | 执行工具后得到的返回值，会作为消息发回模型，使模型能在回复中使用真实数据 |
| Argument validation | “Input checking” | 在执行工具前，验证模型生成的参数是否符合预期类型、范围和约束 |
| MCP | “Tool protocol” | Model Context Protocol；Anthropic 的开放标准，通过任何兼容 client 都能发现和调用的 server 暴露工具 |
| Agent loop | “ReAct loop” | 模型决定工具、代码执行工具、结果喂回模型的迭代循环，直到模型有足够信息回答 |
| Tool poisoning | “Prompt injection via tools” | 一种攻击：工具结果中包含操纵模型行为的指令；必须清理所有工具输出 |
| Rate limiting | “Call budget” | 设置每次对话的最大工具调用数，防止无限循环和失控 API 成本 |

## 延伸阅读

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) -- GPT-4o 工具使用的权威参考，包括并行调用、强制调用和结构化参数。
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) -- Claude 的工具使用实现，包含 `input_schema`、多工具响应和 `tool_choice` 配置。
- [Model Context Protocol Specification](https://modelcontextprotocol.io) -- AI 应用之间实现工具互操作的开放标准，包含 server/client 架构。
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761) -- 关于训练 LLM 判断何时以及如何调用外部工具的基础论文。
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334) -- 针对 1,645 个 API 微调 LLM，以提升 API 调用准确率并减少幻觉。
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) -- 实时基准，对比 GPT-4o、Claude、Gemini 和开放模型的函数调用准确率。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) -- Thought-Action-Observation 循环，也就是每个工具调用外层的 agent loop；本课结束处，正是第 14 阶段接续处。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) -- 五种可组合模式（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer），都建立在单一 tool-use 原语之上。
