# JAX 入门

> PyTorch 会改变 tensors。TensorFlow 会构建 graphs。JAX 会编译 pure functions。最后这一点会改变你思考深度学习的方式。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## 学习目标

- 使用 JAX 的 functional API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写 pure-function 神经网络代码
- 解释 PyTorch 的 eager mutation 和 JAX 的 functional compilation model 之间的关键设计差异
- 应用 jit compilation 和 vmap vectorization，相比 naive Python 加速训练循环
- 在 JAX 中训练一个简单网络，并把显式 state management 与 PyTorch 的面向对象方式进行对比

## 问题

你已经知道如何在 PyTorch 中构建神经网络。定义一个 `nn.Module`，调用 `.backward()`，让 optimizer step。它能工作。数百万人都在使用它。

但 PyTorch 的 DNA 里有一个约束：它会在 Python 中 eager 地、一次一个操作地 trace。每个 `tensor + tensor` 都是一次单独的 kernel launch。每个 training step 都会重新解释同一段 Python 代码。这在普通场景下没问题，直到你需要在 2,048 个 TPU 上训练一个 5400 亿参数模型。那时这些 overhead 会压垮你。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这些不是小规模操作，而是地球上最大的神经网络训练任务之一。它们选择 JAX，是因为 JAX 把你的训练循环当成一个可编译程序，而不是一串 Python 调用。

JAX 是带有三种超能力的 NumPy：automatic differentiation、JIT compilation to XLA，以及 automatic vectorization。你写一个处理单个样本的函数。JAX 给你一个可以处理 batch、计算梯度、编译成机器码并跨多个设备运行的函数。原始函数不用改。

## 概念

### JAX 哲学

JAX 是一个函数式框架。没有 classes，没有 mutable state，没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

这不是风格偏好，而是编译器约束。JIT compilation 需要 pure functions：相同输入永远产生相同输出，没有副作用。这个限制正是 100x 加速成为可能的原因。

### jax.numpy：熟悉的表层

JAX 在 accelerators 上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名。相同的 broadcasting rules。相同的 slicing semantics。但 arrays 位于 GPU/TPU 上，并且每个操作都能被 compiler trace。

一个关键区别：JAX arrays 是 immutable 的。不能写 `a[0] = 5`。要写：`a = a.at[0].set(5)`。这会别扭一周，然后你会想通：immutability 正是让 `grad`、`jit` 和 `vmap` 这些 transformations 能组合起来的原因。

### jax.grad：函数式自动微分

PyTorch 把梯度附着到 tensors 上（`.grad`）。JAX 把梯度附着到函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，并返回一个会计算梯度的新函数。没有 `.backward()` 调用。没有存储在 tensors 上的 computation graph。梯度只是另一个可以调用、组合或 JIT-compile 的函数。

这种方式可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。Jacobians。Hessians。全部通过组合 `grad` 得到。PyTorch 也能做这些（`torch.autograd.functional.hessian`），但那是后加上去的。在 JAX 中，这是基础。

约束是：`grad` 只适用于 pure functions。函数内部不要放 print statements（它们会在 tracing 时运行，而不是 execution 时运行）。不要修改外部 state。不要在没有显式 key management 的情况下生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会 trace 这个函数：它记录发生了哪些操作，但不真正执行它们。然后它把这条 trace 交给 XLA（Accelerated Linear Algebra），也就是 Google 为 TPUs 和 GPUs 构建的 compiler。XLA 会融合操作、消除冗余内存拷贝，并生成优化过的机器码。

之后的调用会完全跳过 Python。编译后的代码会以 C++ 速度在 accelerator 上运行。

JIT 有帮助的场景：

- Training steps（相同计算重复上千次）
- Inference（相同模型，不同输入）
- 任何被多次调用且输入 shape 相似的函数

JIT 会拖后腿的场景：

- Python control flow 依赖值的函数（例如 `if x > 0`，其中 x 是 traced array）
- 一次性计算（编译开销超过运行时间）
- 调试（tracing 会隐藏真实执行过程）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` loops。这些不是可选项，而是编译的代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 会把它提升成处理 batch 的函数：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不要沿着 `params` batch（它是共享的），沿着 `x` 的 axis 0 batch。不需要手写 `for` loop。不需要 reshape。不需要把 batch dimension 一路手动穿过代码。JAX 会找出 batch 维度，并向量化整个计算。

这不是语法糖。`vmap` 会生成 fused vectorized code，比 Python loop 快 10-100 倍。而且它可以和 `jit`、`grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Per-example gradients。一行代码。在 PyTorch 中不借助技巧几乎做不到。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会把函数复制到所有可用设备（GPUs/TPUs）上，并拆分 batch。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 会跨设备同步 gradients。

Google 使用 `pmap`（以及它的继任者 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型是：写单设备版本，包上 `pmap`，完成。

### Pytrees：通用数据结构

JAX 操作的是 “pytrees”：由 lists、tuples、dicts 和 arrays 嵌套组合成的结构。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX transformation（`grad`、`jit`、`vmap`）都知道如何遍历 pytrees。`jax.tree.map(f, tree)` 会把 `f` 应用到每个 leaf。这就是 optimizers 一次更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有参数注册。tree structure 就是模型。

### Functional vs Object-Oriented

PyTorch 把 state 存在 objects 内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式 state 的 pure functions：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Params 会被传入。没有任何东西被存储。没有任何东西被修改。这让每个函数都可测试、可组合、可编译。它也意味着你要自己管理 params，或者使用 Flax、Equinox 这样的库。

### JAX 生态

JAX 给你 primitives。Libraries 给你 ergonomics：

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax 是标准 optimizer library。它把 gradient transformation（Adam、SGD、clipping）和 parameter update 分开，让组合变得很简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 什么时候用 JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

诚实答案是：除非你有明确理由用 JAX，否则用 PyTorch。这些理由包括：能使用 TPU、需要 per-example gradients、需要超大规模 multi-device training，或者你在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要显式 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这很烦。但它保证了跨设备、跨编译的可复现性，这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的性质。

```figure
batchnorm-effect
```

## Build It

### Step 1：Setup and Data

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐藏层分别有 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2：初始化参数

没有 class。只有一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动完成 He-initialization。从一个 seed 拆出三个 PRNG keys。每个 weight 都是 nested dict 中的 immutable array。

### Step 3：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Pure functions。Params in，prediction out。没有 `self`，没有存储 state。`loss_fn` 从零计算 cross-entropy：softmax、log、negative mean。

### Step 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 会在一次 pass 中同时返回 loss value 和 gradients。`@jax.jit` decorator 会把两个函数都编译到 XLA。第一次调用之后，每个 training step 都不会触碰 Python。

### Step 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epochs。约 97% 测试准确率。第一个 epoch 很慢（JIT compilation）。第 2-10 个 epochs 很快。

注意少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个 update 是一次组合函数调用。Gradients 被计算出来，经 Adam 变换，再应用到参数上，全都在 `train_step` 内部完成。

## Use It

### Flax：Google 标准

Flax 是最常见的 JAX neural network library。它把 `nn.Module` 加了回来，但仍使用显式 state management：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构和 PyTorch 相同，但 `params` 与 model 分离。`model.init()` 创建 params。`model.apply(params, x)` 运行 forward pass。Model object 没有 state。

### Equinox：更 Pythonic 的替代方案

Equinox（Patrick Kidger 开发）把 models 表示成 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Model 本身就是 pytree。不需要 `.apply()`。Parameters 就是 model 的 leaves。这更接近 JAX 的思维方式。

### Optax：可组合的 Optimizers

Optax 会把 gradient transformation 与 update 解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping、learning rate warmup、weight decay 都可以作为 transforms 链式组合。每个 transform 查看 gradients，修改它们，再传给下一个。没有 monolithic optimizer class。

## Ship It

**Installation:**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- 第一次 JIT 调用很慢（compilation）。benchmark 前先 warm up。
- 避免在 JIT 内部对 JAX arrays 使用 Python loops。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内部工作。普通 `print()` 不行。
- 用 `jax.profiler` 或 TensorBoard 做 profiling。XLA compilation 可能隐藏 bottlenecks。
- JAX 默认会预分配 75% GPU memory。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以禁用。

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**

- `outputs/prompt-jax-optimizer.md` -- 一个用于选择合适 JAX optimizer 配置的 prompt
- `outputs/skill-jax-patterns.md` -- 一个覆盖 JAX functional patterns 的 skill

## 练习

1. 给 MLP 添加 dropout。在 JAX 中，dropout 需要 PRNG key。把 key 穿过 forward pass，并为每个 dropout layer split key。比较使用和不使用 dropout 的测试准确率。

2. 使用 `jax.vmap` 为一批 32 张 MNIST 图片计算 per-example gradients。计算每个样本的 gradient norm。哪些样本梯度最大，为什么？

3. 把手写 forward function 替换成一个通用的 `mlp_forward(params, x)`，让它适用于任意层数。使用 `jax.tree.leaves` 自动确定深度。

4. 对比带 `@jax.jit` 和不带 `@jax.jit` 的 training step。分别计时 100 步。在你的硬件上加速有多大？第一次调用的 compilation overhead 是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现 gradient clipping。分别在使用和不使用 clipping 的情况下训练。绘制训练过程中的 gradient norm，观察效果。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | “让 JAX 快的东西” | Accelerated Linear Algebra，一个 compiler，会从 computation graph 融合操作并生成优化过的 GPU/TPU kernels |
| JIT | “Just-in-time compilation” | JAX 在第一次调用时 trace 函数、编译到 XLA，并在之后调用中运行编译版本 |
| Pure function | “没有副作用” | 输出只依赖输入的函数，没有全局 state、没有 mutation、没有不带显式 key 的 randomness |
| vmap | “Auto-batching” | 把处理单个样本的函数转换成处理 batch 的函数，无需重写 |
| pmap | “Auto-parallelism” | 把函数复制到多个设备上，并拆分输入 batch |
| Pytree | “Nested dict of arrays” | JAX 可以遍历和 transform 的任意嵌套结构，由 lists、tuples、dicts 和 arrays 组成 |
| Tracing | “记录计算” | JAX 用抽象值执行函数来构建 computation graph，而不计算真实结果 |
| Functional autodiff | “函数的 grad” | 通过 transform functions 来计算导数，而不是把 gradient storage 附着到 tensors 上 |
| Optax | “JAX 的 optimizer library” | 一个可组合的 gradient transformations 库，包括 Adam、SGD、clipping、scheduling，并可以链式组合 |
| Flax | “JAX 的 nn.Module” | Google 的 JAX 神经网络库，在保持 state 显式的同时添加 layer abstractions |

## 延伸阅读

- JAX documentation: https://jax.readthedocs.io/ -- 官方文档，包含优秀的 grad、jit 和 vmap 教程
- “JAX: composable transformations of Python+NumPy programs”（Bradbury 等人，2018）-- 解释设计哲学的原始论文
- Flax documentation: https://flax.readthedocs.io/ -- Google 的 JAX 神经网络库
- Patrick Kidger，“Equinox: neural networks in JAX via callable PyTrees and filtered transformations”（2021）-- Flax 的 Pythonic 替代方案
- DeepMind，“Optax: composable gradient transformation and optimisation” -- 标准 optimizer library
- “You Don't Know JAX”（Colin Raffel，2020）-- 来自 T5 作者之一的实用 JAX gotchas 和 patterns 指南
