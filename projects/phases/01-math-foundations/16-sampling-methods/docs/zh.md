# 采样方法

> 采样是 AI 探索可能性空间的方式。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## 学习目标

- 只使用 uniform random numbers，从零实现 inverse CDF、rejection 和 importance sampling
- 为语言模型 token generation 构建 temperature、top-k 和 top-p（nucleus）sampling
- 解释 reparameterization trick，以及它为什么能让 VAE 中的采样参与 backpropagation
- 运行 Metropolis-Hastings MCMC，从未归一化的 target distribution 中采样

## 问题

一个语言模型处理完你的 prompt 后，会输出一个包含 50,000 个 logits 的向量。词表里的每个 token 对应一个 logit。现在它必须选出一个。怎么选？

如果它总是选择概率最高的 token，每次回答都会一样。Deterministic。无聊。如果它完全均匀随机选择，输出就会变成胡言乱语。答案存在于这两个极端之间，而这个“之间”由 sampling 控制。

Sampling 不只用于文本生成。Reinforcement learning 通过采样 trajectories 来估计 policy gradients。VAEs 通过从 learned distributions 中采样、再穿过 randomness 做 backpropagation 来学习 latent representations。Diffusion models 通过采样 noise 并迭代 denoising 来生成图像。Monte Carlo methods 用来估计没有 closed-form solution 的积分。MCMC algorithms 探索不可能穷举的高维 posterior distributions。

每个 generative AI system 都是一个 sampling system。Sampling strategy 决定输出的质量、多样性和可控性。本课会从零构建所有主要 sampling methods，从 uniform random numbers 开始，一直到驱动现代 LLMs 和 generative models 的技术。

## 概念

### 为什么 Sampling 重要

Sampling 在 AI 和机器学习中扮演四种基础角色：

**Generation。** Language models、diffusion models 和 GANs 都通过 sampling 产生输出。Sampling algorithm 直接控制创造性、一致性和多样性。Temperature、top-k 和 nucleus sampling 是工程师每天都会调的旋钮。

**Training。** Stochastic gradient descent 会采样 mini-batches。Dropout 会采样要停用的 neurons。Data augmentation 会采样 random transformations。Importance sampling 会重新加权 samples，以降低 reinforcement learning（PPO、TRPO）中的 gradient variance。

**Estimation。** ML 中很多量没有 closed-form solution：data distribution 上的 expected loss、energy-based model 的 partition function、Bayesian inference 中的 evidence。Monte Carlo estimation 通过对 samples 求平均来近似这些量。

**Exploration。** MCMC algorithms 在 Bayesian inference 中探索 posterior distributions。Evolutionary strategies 采样 parameter perturbations。Thompson sampling 在 bandits 中平衡 exploration 和 exploitation。

核心挑战是：你只能直接从简单分布（uniform、normal）中采样。对其他所有分布，你都需要一种方法，把简单 samples 转换成来自 target distribution 的 samples。

### Uniform Random Sampling

每种 sampling method 都从这里开始。Uniform random number generator 会产生 [0, 1) 中的值，任意长度相同的子区间都有相同概率。

```text
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个离散 item 中均匀采样，生成 U，然后返回 floor(n * U)。要从连续区间 [a, b] 中采样，计算 a + (b - a) * U。

关键洞察：一个 uniform random number 正好包含从任意分布产生一个 sample 所需的随机性。技巧在于找到正确的 transformation。

### Inverse CDF Method（Inverse Transform Sampling）

Cumulative distribution function（CDF）把值映射到概率：

```text
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF 把概率映射回值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 就服从 target distribution。

```text
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution example：**

```text
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出 closed-form 的 F_inverse 时，这种方法非常完美。Normal distribution 没有 closed-form inverse CDF，所以我们会用其他方法（Box-Muller，或 numerical approximation）。

**离散版本：** 对 discrete distributions，先用 cumulative sum 构建 CDF，生成 U，然后找到第一个 cumulative sum 超过 U 的 index。这就是 Lesson 06 中 `sample_categorical` 的工作方式。

### Rejection Sampling

当你无法反转 CDF，但可以计算 target PDF（即使只差一个常数）时，rejection sampling 就能派上用场。

```text
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Bound M 越紧，acceptance rate 越高。在低维（1-3 维）中，rejection sampling 效果很好。在高维中，acceptance rate 会指数级下降，因为 proposal volume 的大部分都会被拒绝。这就是 rejection sampling 的 curse of dimensionality。

**例子：从 truncated normal 中采样。** 在截断区间内使用 uniform proposal。Envelope M 是该区间内 normal PDF 的最大值。

**例子：从 semicircle 中采样。** 在 bounding rectangle 中均匀提出点。如果点落在 semicircle 内就接受。这也是 Monte Carlo 计算 pi 的方式：acceptance rate 等于面积比 pi/4。

### Importance Sampling

有时候你并不需要来自 target distribution p(x) 的 samples。你需要估计 p(x) 下的 expectation，但手头的 samples 来自另一个 distribution q(x)。

```text
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在 reinforcement learning 中至关重要。在 PPO（Proximal Policy Optimization）里，你用旧 policy pi_old 收集 trajectories，但想优化新的 policy pi_new。Importance weight 是 pi_new(a|s) / pi_old(a|s)。PPO 会 clip 这些 weights，避免新 policy 偏离旧 policy 太远。

Importance sampling estimator 的 variance 取决于 q 和 p 有多相似。如果 q 与 p 很不同，少数 samples 会得到巨大的 weights 并主导估计。Self-normalized importance sampling 会除以 weights 的总和来缓解这个问题：

```text
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation

Monte Carlo estimation 通过对 random samples 求平均来近似积分。Law of large numbers 保证它会收敛。

```text
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么在 grid-based integration 不可能处理的高维场景中，Monte Carlo methods 会占主导。

**估计 pi：**

```text
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计 expectations：**

```text
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo（MCMC）：Metropolis-Hastings

MCMC 会构造一个 Markov chain，使它的 stationary distribution 是 target distribution p(x)。经过足够多 steps 后，chain 中的 samples 就（近似）来自 p(x)。

```text
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于 symmetric proposals（q(x'|x) = q(x|x')），这个 ratio 会简化为 p(x')/p(x)。这就是原始的 Metropolis algorithm。

**为什么它有效。** Acceptance rule 保证 detailed balance：位于 x 并移动到 x' 的概率，等于位于 x' 并移动到 x 的概率。Detailed balance 意味着 p(x) 是这个 chain 的 stationary distribution。

**实践注意事项：**
- Burn-in：丢弃 chain 达到 equilibrium 前的早期 samples
- Thinning：每隔 k 个 sample 保留一个，以降低 autocorrelation
- Proposal scale：太小会导致 chain 移动缓慢（高 acceptance，慢探索）；太大会导致大部分 proposals 被拒绝（低 acceptance，卡在原地）
- Gaussian proposal 在高维中的 optimal acceptance rate 约为 0.234

### Gibbs Sampling

Gibbs sampling 是 multivariate distributions 的一种特殊 MCMC。它不是一次性在所有维度上提出移动，而是每次从 conditional distribution 中更新一个 variable。

```text
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling 要求你能从每个 conditional distribution p(x_i | x_{-i}) 中采样。很多模型中这很直接：
- Bayesian networks：conditionals 来自 graph structure
- Gaussian mixtures：conditionals 是 Gaussian
- Ising models：每个 spin 的 conditional 只依赖它的 neighbors

Acceptance rate 永远是 1（每个 proposal 都会被接受），因为从 exact conditional 中采样会自动满足 detailed balance。

**局限。** 当 variables 高度相关时，Gibbs sampling mixing 很慢，因为一次只更新一个 variable，无法沿着 distribution 做大的 diagonal moves。

### Temperature Sampling（用于 LLMs）

Language models 会为词表中的每个 token 输出 logits z_1, ..., z_V。Softmax 会把它们转换为概率。Temperature 会在 softmax 之前重新缩放 logits：

```text
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么它有效。** 把 logits 除以一个小于 1 的 T，会放大 logits 之间的差异。如果 z_1 = 2 且 z_2 = 1，用 T = 0.5 相除会得到 z_1/T = 4 和 z_2/T = 2，让差距更大。经过 softmax 后，最高 logit token 会拿到大得多的概率质量。

**实践中：**
- T = 0.0：greedy decoding，最适合 factual Q&A
- T = 0.3-0.7：略有创造性，适合 code generation
- T = 0.7-1.0：平衡，适合 general conversation
- T = 1.0-1.5：creative writing、brainstorming
- T > 1.5：越来越随机，通常很少有用

Temperature 不会改变哪些 tokens 是可能的。它改变的是分配给每个 token 的 probability mass。

### Top-k Sampling

Top-k sampling 会把 candidate set 限制为概率最高的 k 个 tokens，然后重新归一化，并从这个受限集合中采样。

```text
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 会阻止模型选择 extremely unlikely tokens（typos、nonsense），这些 token 存在于 vocabulary distribution 的 long tail 中。问题是：k 是固定的，不随上下文变化。当模型很自信时（某个 token 概率为 95%），k = 40 仍然允许 39 个 alternatives。当模型不确定时（概率分散在 1000 个 tokens 上），k = 40 又会截断一些合理选项。

### Top-p（Nucleus）Sampling

Top-p sampling 会动态调整 candidate set size。它不是保留固定数量的 tokens，而是保留 cumulative probability 超过 p 的最小 token 集合。

```text
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型很自信时，nucleus sampling 会保留很少 tokens（可能 2-3 个）。当模型不确定时，它会保留很多 tokens（可能 200 个）。这种 adaptive behavior 是 nucleus sampling 通常比 top-k 生成更好文本的原因。

**常见组合：**
- Temperature 0.7 + top-p 0.9：通用设置
- Temperature 0.0（greedy）：最适合 deterministic tasks
- Temperature 1.0 + top-k 50：Fan et al. (2018) 原论文设置

Top-k 和 top-p 可以组合使用。先应用 top-k，然后在剩余集合上应用 top-p。

### Reparameterization Trick（用于 VAEs）

Variational autoencoders（VAEs）通过把 inputs 编码到 latent space 中的一个 distribution，从这个 distribution 中采样，再把 sample 解码回来进行学习。问题是：你不能直接穿过 sampling operation 做 backpropagation。

```text
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick 会把 randomness 和 parameters 分离：

```text
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

它之所以有效，是因为 N(mu, sigma^2) 和 mu + sigma * N(0, 1) 服从同一个 distribution。关键洞察是：把 randomness 移到不含 parameters 的来源（epsilon）中，然后把 sample 表达为 parameters 的 differentiable transformation。

**在 VAE training loop 中：**
1. Encoder 为每个 input 输出 mu 和 log(sigma^2)
2. Sample epsilon ~ N(0, 1)
3. Compute z = mu + sigma * epsilon
4. Decode z 来 reconstruct input
5. Backpropagate through steps 4, 3, 2, 1（可行，因为 step 3 是 differentiable）

没有 reparameterization trick，VAEs 就无法用 standard backpropagation 训练。这个单一洞察让 VAEs 变得实用。

### Gumbel-Softmax（Differentiable Categorical Sampling）

Reparameterization trick 适用于 continuous distributions（Gaussian）。对于 discrete categorical distributions，我们需要另一种方法。Gumbel-Softmax 提供了 categorical sampling 的 differentiable approximation。

**Gumbel-Max trick（non-differentiable）：**

```text
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（differentiable approximation）：**

```text
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 会产生 discrete sample 的 continuous relaxation。输出是 probability vector（soft one-hot），而不是 hard one-hot。Gradients 会穿过 softmax。训练的 forward pass 中可以使用 "straight-through" estimator：forward pass 用 hard argmax，但 backward pass 使用 soft Gumbel-Softmax gradients。

**Applications：**
- VAEs 中的 discrete latent variables
- Neural architecture search（选择 discrete operations）
- Hard attention mechanisms
- 带 discrete actions 的 reinforcement learning

### Stratified Sampling

Standard Monte Carlo sampling 可能因为随机性在 sample space 中留下空洞。Stratified sampling 会把空间切分成 strata，并从每个 stratum 采样，从而强制均匀覆盖。

```text
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

与 standard Monte Carlo 相比，stratified sampling 的 variance 总是更低或相等：

```text
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Applications：**
- Numerical integration（quasi-Monte Carlo）
- Training data splits（确保每个 fold 中 class balance）
- 带 stratification 的 importance sampling（组合两种技术）
- NeRF（Neural Radiance Fields）沿 camera rays 使用 stratified sampling

### 与 Diffusion Models 的联系

Diffusion models 通过 sampling process 生成图像。Forward process 会在 T 个 steps 中向图像添加 Gaussian noise，直到它变成 pure noise。Reverse process 学习 denoise，一步步恢复原始图像。

```text
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

它与本课方法的联系：
- 每个 denoising step 都使用 reparameterization trick（sample noise，应用 deterministic transform）
- Noise schedule {alpha_t} 控制了一种 temperature annealing
- Training 使用 Monte Carlo estimation 来近似 ELBO（evidence lower bound）
- Diffusion models 中的 ancestral sampling 是一个 Markov chain（每一步只依赖当前 state）

整个 image generation process 都是 iterative sampling：从 noise 开始，每一步都在 learned denoising model 条件下采样出一个稍微更少噪声的版本。

```figure
monte-carlo-pi
```

## Build It

### Step 1：Uniform and inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个 exponential samples，并验证 mean 是 1/lambda。

### Step 2：Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用 rejection sampling 从 truncated normal distribution 中抽样。通过 histogram 验证 samples 的形状。

### Step 3：Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用 uniform proposal 估计 normal distribution 下的 E[X^2]。与已知答案（mu^2 + sigma^2）比较。

### Step 4：Monte Carlo estimation of pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从 bimodal distribution（两个 Gaussians 的 mixture）中采样。Visualize chain 的 trajectory。

### Step 6：Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7：Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示 temperature 如何改变一组 token logits 的 output distribution。

### Step 8：Top-k and top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9：Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

展示 gradients 能穿过 reparameterized sample，但不能穿过 direct sampling。

### Step 10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低 temperature 如何让输出接近 one-hot vector。

带所有 visualizations 的完整实现位于 `code/sampling.py`。

## Use It

使用 NumPy 和 SciPy 时，production versions 是：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC 可以使用专门的 libraries：
- PyMC：使用 NUTS（adaptive HMC）进行完整 Bayesian modeling
- emcee：ensemble MCMC sampler
- NumPyro/JAX：GPU-accelerated MCMC

你已经从零构建了这些方法。现在你知道 library calls 背后在做什么。

## 练习

1. 为 Cauchy distribution 实现 inverse CDF sampling。CDF 是 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个 samples，并将 histogram 与真实 PDF 对比。注意 heavy tails（远离中心的 extreme values）。

2. 使用 rejection sampling，以 Uniform(0, 1) proposal 从 Beta(2, 5) distribution 中生成 samples。将 accepted samples 与真实 Beta PDF 对比。理论 acceptance rate 是多少？

3. 使用 Monte Carlo，用 1,000、10,000 和 100,000 个 samples 估计 sin(x) 从 0 到 pi 的积分。比较每个 sample level 的误差。验证 error 按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings，从 2D distribution p(x, y) proportional to exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制 samples 和 chain trajectory。尝试不同的 proposal standard deviations。

5. 构建一个完整 text generation demo：给定一个包含 10 个 words 的 vocabulary 和对应 logits，分别使用 (a) greedy、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成长度为 20 tokens 的 sequences。比较 5 次运行中 outputs 的 diversity。

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|----------|----------|
| Sampling | “Drawing random values” | 按照某个 probability distribution 生成 values。所有 generative AI 背后的机制 |
| Uniform distribution | “All equally likely” | [a, b] 中每个值都有相同 probability density 1/(b-a)。所有 sampling methods 的起点 |
| Inverse CDF | “Probability transform” | F_inverse(U) 会把 uniform sample 转换成来自任意已知 CDF distribution 的 sample。精确且高效 |
| Rejection sampling | “Propose and accept/reject” | 从简单 proposal 中生成，按 target/proposal ratio 成比例的概率接受。精确但浪费 samples |
| Importance sampling | “Reweight samples” | 用来自 q(x) 的 samples 估计 p(x) 下的 expectations，每个 sample 由 p(x)/q(x) 加权。RL 中 PPO 的核心 |
| Monte Carlo | “Average random samples” | 用 sample averages 近似 integrals。误差 O(1/sqrt(N))，与维度无关 |
| MCMC | “Random walk that converges” | 构造 stationary distribution 为 target 的 Markov chain。Metropolis-Hastings 是 foundational algorithm |
| Metropolis-Hastings | “Accept uphill, sometimes downhill” | 提出 moves，基于 density ratio 接受。Detailed balance 保证收敛到 target distribution |
| Gibbs sampling | “One variable at a time” | 在固定其他变量时，从每个变量的 conditional distribution 更新。Acceptance rate 为 100% |
| Temperature | “Confidence knob” | Softmax 前用 T 除以 logits。T<1 会 sharpen（更自信），T>1 会 flatten（更多样） |
| Top-k sampling | “Keep the k best” | 将除 k 个最高概率 tokens 之外的概率置零，重新归一化后采样。Candidate set size 固定 |
| Nucleus sampling (top-p) | “Keep the probable ones” | 保留 cumulative probability 超过 p 的最小 token 集合。Candidate set size 自适应 |
| Reparameterization trick | “Move randomness outside” | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。让 sampling 可微。VAE training 的关键 |
| Gumbel-Softmax | “Soft categorical sampling” | 使用 Gumbel noise + temperature softmax 对 categorical sampling 做 differentiable approximation |
| Stratified sampling | “Forced coverage” | 将 sample space 切分为 strata，并从每个 stratum 采样。Variance 总是低于 naive Monte Carlo 或相等 |
| Burn-in | “Warm-up period” | 在 chain 到达 stationary distribution 前丢弃的初始 MCMC samples |
| Detailed balance | “Reversibility condition” | p(x) * T(x->y) = p(y) * T(y->x)。这是 p 成为 Markov chain stationary distribution 的充分条件 |
| Diffusion sampling | “Iterative denoising” | 从 noise 开始，通过 learned denoising steps 生成 data。每一步都是 conditional sampling operation |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC foundations 的详细 tutorial
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus（top-p）sampling 论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 提出 reparameterization trick 的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - 将 sampling 与 image generation 连接起来的 DDPM
