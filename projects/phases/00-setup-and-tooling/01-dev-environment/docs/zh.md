# 开发环境

> 你的工具会塑造你的思考方式。一次性搭好，并且搭对。

**Type:** Build
**Languages:** Python, Node.js, Rust
**Prerequisites:** None
**Time:** ~45 minutes

## 学习目标

- 从零配置 Python 3.11+、Node.js 20+ 和 Rust toolchains
- 配置 virtual environments 和 package managers，让 builds 可复现
- 使用 CUDA/MPS 验证 GPU access，并运行一个测试 tensor operation
- 理解四层 stack：system、packages、runtimes、AI libraries

## 问题

你将用 Python、TypeScript、Rust 和 Julia，跨 200+ lessons 学习 AI engineering。如果你的环境是坏的，每一课都会变成和 tooling 打架，而不是学习。

大多数人会跳过环境配置。然后他们会花几个小时调试 import errors、version conflicts 和缺失的 CUDA drivers。我们要把这件事认真做一次。

## 概念

一个 AI engineering environment 有四层：

```mermaid
graph TD
    A["4. AI/ML Libraries\nPyTorch, JAX, transformers, etc."] --> B["3. Language Runtimes\nPython 3.11+, Node 20+, Rust, Julia"]
    B --> C["2. Package Managers\nuv, pnpm, cargo, juliaup"]
    C --> D["1. System Foundation\nOS, shell, git, editor, GPU drivers"]
```

我们从下往上安装。每一层都依赖它下面那一层。

## Build It

### Step 1: System Foundation

检查你的系统，并安装基础工具。

```bash
# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows (use WSL2)
wsl --install -d Ubuntu-24.04
```

### Step 2: Python with uv

我们使用 `uv`。它比 pip 快 10-100 倍，并且会自动处理 virtual environments。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

uv pip install numpy matplotlib jupyter
```

验证：

```python
import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"Vector: {a}, dot product with itself: {np.dot(a, a)}")
```

### Step 3: Node.js with pnpm

用于 TypeScript lessons（agents、MCP servers、web apps）。

```bash
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"
```

### Step 4: Rust

用于 performance-critical lessons（inference、systems）。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version
```

### Step 5: Julia (Optional)

用于 Julia 擅长的 math-heavy lessons。

```bash
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

### Step 6: GPU Setup (If You Have One)

```bash
# NVIDIA
nvidia-smi

# Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

没有 GPU？没关系。大多数 lessons 都可以在 CPU 上完成。对训练负载较重的 lessons，可以使用 Google Colab 或 cloud GPUs。

### Step 7: Verify Everything

运行 verification script：

```bash
python phases/00-setup-and-tooling/01-dev-environment/code/verify.py
```

## Use It

你的环境现在已经可以支持本课程中的每一课。下面是各语言的使用位置：

| Language | Used In | Package Manager |
|----------|---------|-----------------|
| Python | Phases 1-12 (ML, DL, NLP, Vision, Audio, LLMs) | uv |
| TypeScript | Phases 13-17 (Tools, Agents, Swarms, Infra) | pnpm |
| Rust | Phases 12, 15-17 (Performance-critical systems) | cargo |
| Julia | Phase 1 (Math foundations) | Pkg |

## Ship It

本课产出一个 verification script，任何人都可以运行它来检查自己的 setup。

查看 `outputs/prompt-env-check.md`，那里有一个 prompt，可以帮助 AI assistants 诊断 environment issues。

## 练习

1. 运行 verification script，并修复任何 failures。
2. 为本课程创建一个 Python virtual environment，并安装 PyTorch。
3. 用四种语言分别写一个 “hello world”，并逐个运行。
