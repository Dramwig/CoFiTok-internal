---
created: 2026-07-06
updated: 2026-07-06
tags: [idea, vision, diffusion, tokenizer, coarse-to-fine, generation, noise-prediction, pixel-space]
status: active
---

# CoFiTok：Denoising Token Diffusion

## 0. 一句话定义

**CoFiTok 将 diffusion 的噪声预测过程分解为一个有序的 next-token 过程：每个 token 表示一个压缩后的负噪声分量，经受限合成算子展开为 dense pixel-space noise / velocity component；从前到后，token 序列逐步决定图像的粗结构、物体形状、局部边界和高保真纹理。**

核心不是"latent token 最后经过 decoder 解码成图像"，而是：

$$
\text{tokenized noise prediction} \rightarrow \text{pixel-space denoising update} \rightarrow \text{image}.
$$

最终图像由一串 token 所诱导的 pixel-space denoising increments 累积得到，而不是由一个 VAE-style latent decoder 一次性解码得到。

---

## 1. 研究动机

普通 VAE / latent diffusion 的结构是：

$$
x \xrightarrow{E} z \xrightarrow{\text{diffusion}} \hat{z} \xrightarrow{D} \hat{x}.
$$

它的问题是：图像被压成一个整体 latent，结构、纹理、小物体、文字等信息被缠绕在同一个 latent 空间中；模型无法手动选择"保留多少信息"；最终 pixel-space 质量还受 decoder 上限影响。

**Latent Forcing** [[arXiv:2602.11401](https://arxiv.org/abs/2602.11401)] 提供了一个重要启发：生成质量不仅取决于表示空间，还取决于 **diffusion trajectory 的信息显露顺序**。Latent Forcing 通过联合处理 latent 和 pixel，并使用分别调节的 noise schedules，让 latent 先作为 scratchpad 形成中间计算，再生成高频 pixel features；其目标是获得 latent diffusion 的效率，同时保留 pixel-space generation 的端到端建模优势。

本 idea 的进一步问题是：

> 如果 latent 与 pixel 之间的生成顺序可以重排，那么 diffusion 内部的噪声预测是否也可以被重排为一串 coarse-to-fine 的 token？

也就是从 $\epsilon_\theta(x_t, t, c)$ 一次性预测完整 dense noise field，改为：

$$
\epsilon_\theta(x_t, t, c) = \sum_{k=1}^{K} S_k(z^{(k)}).
$$

其中每个 $z^{(k)}$ 是一个 next denoising token。

---

## 2. 最终核心假设

**Hypothesis**：标准 diffusion 的 reverse trajectory 本身包含一种天然的信息粒度顺序：早期去噪步骤更强地决定全局结构、布局和主体形状，后期去噪步骤逐步补充边缘、纹理、文字、小物体等细节。

因此，可以将 diffusion 的噪声预测显式分解为：

$$
\epsilon = \epsilon^{(1)} + \epsilon^{(2)} + \cdots + \epsilon^{(K)},
$$

其中 $\epsilon^{(k)} = S_k(z^{(k)})$，$z^{(k)}$ 是第 $k$ 个压缩 denoising token，$S_k$ 是受限的 token-to-noise synthesis operator。

直觉上：
- $z^{(1)}$：粗结构 / 大轮廓 / 主体布局；
- $z^{(2)}$：部件结构 / 物体边界；
- $z^{(3)}$：局部形状 / 小目标；
- $z^{(K)}$：纹理 / 文字 / 高频细节。

> **频带只是便于理解的信息粒度类比。** 方法本身不依赖 DCT、DWT、FFT 或任何显式频域变换。

---

## 3. 推荐命名

**CoFiTok: Coarse-to-Fine Denoising Tokens for Pixel-Space Diffusion**

备选：
- **Denoising Token Diffusion: Factorizing Pixel-Space Noise Prediction into Ordered Visual Tokens**
- **CoFiTok: Ordered Denoising-Token Factorization for Prefix-Controllable Pixel-Space Image Generation**

核心关键词：Denoising token · Tokenized noise prediction · Coarse-to-fine reverse trajectory · Prefix-controllable denoising · Restricted synthesis operator · Pixel-space diffusion without VAE decoder bottleneck

---

## 4. 与普通 VAE / Latent Diffusion 的区别

普通 VAE / latent diffusion 是 **latent-to-pixel decoding**：

$$
z \rightarrow D(z) \rightarrow x.
$$

CoFiTok 是 **tokenized pixel-space noise prediction**：

$$
z^{(k)} \rightarrow S_k(z^{(k)}) = \epsilon^{(k)} \rightarrow \text{pixel-space denoising update}.
$$

CoFiTok 不是把低维 latent 交给 decoder 生成图像，而是把 dense noise prediction 分解成多个压缩 token 分量。最终图像仍然通过 diffusion 公式在 pixel-space 中去噪得到：

$$
\hat{x}_0^{(m)} = \frac{x_t - \sigma_t \sum_{k=1}^{m} S_k(z^{(k)})}{\alpha_t}.
$$

当 $m = K$ 时，得到完整去噪结果 $\hat{x}_0 = \hat{x}_0^{(K)}$。

---

## 5. 数学形式

### 5.1 Forward diffusion

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

标准 diffusion 学习 $\epsilon \approx \epsilon_\theta(x_t, t, c)$。

CoFiTok 学习：

$$
\epsilon \approx \hat{\epsilon}^{(K)} = \sum_{k=1}^{K} S_k(z^{(k)}).
$$

### 5.2 Next denoising token

$$
z^{(1)} = T_1(x_t, t, c),
$$

$$
z^{(k)} = T_k\!\left(x_t,\, z^{(1)}, \ldots, z^{(k-1)},\, t, c\right), \qquad k > 1.
$$

其中 $T_k$ 是 next-token predictor（可以是强模型）；$z^{(k)}$ 是压缩 denoising token；token 顺序表示去噪信息从粗到细的显露顺序。

### 5.3 Token-to-noise synthesis

$$
\hat{\epsilon}^{(k)} = S_k(z^{(k)}).
$$

完整噪声预测：

$$
\hat{\epsilon}^{(1:K)} = \sum_{k=1}^{K} S_k(z^{(k)}).
$$

前缀噪声预测（token budget 为 $m$）：

$$
\hat{\epsilon}^{(1:m)} = \sum_{k=1}^{m} S_k(z^{(k)}).
$$

前缀去噪图像：

$$
\hat{x}_0^{(m)} = \frac{x_t - \sigma_t \hat{\epsilon}^{(1:m)}}{\alpha_t}.
$$

---

## 6. 为什么 $S_k$ 不是 decoder（核心设计）

**错误形式**（$S_k$ 看到图像/条件，会退化为 decoder）：

$$
\hat{\epsilon}^{(k)} = S_k(z^{(k)},\, x_t,\, t,\, c,\, z^{(<k)}).
$$

**正确形式**（$S_k$ 只接收当前 token）：

$$
\hat{\epsilon}^{(k)} = S_k(z^{(k)}).
$$

最多允许标量 schedule：

$$
\hat{\epsilon}^{(k)} = \gamma_k(t) \cdot S_k(z^{(k)}),
$$

其中 $\gamma_k(t)$ 是标量系数，不携带语义信息。

---

## 7. $S_k$ 的设计原则

### 只接收当前 token

$$
S_k: z^{(k)} \mapsto \hat{\epsilon}^{(k)}.
$$

**禁止输入**：$x_t$ · text prompt · class condition · time embedding · previous tokens · CLIP/DINO features · U-Net skip connection · learned constant input。

所有样本相关、条件相关、语义相关的信息都必须由 $T_k$ 编码进 $z^{(k)}$。

### 低容量线性合成

推荐形式：

$$
S_k(z^{(k)}) = \gamma_k(t) \cdot \left[\operatorname{Up}_{s_k}\!\left(W_k z^{(k)}\right) * g_k\right],
$$

其中：
- $W_k$：无 bias 的 $1\times1$ projection；
- $\operatorname{Up}_{s_k}$：固定 bilinear upsampling 或 pixel shuffle；
- $g_k$：浅层无 bias 线性卷积；
- $\gamma_k(t)$：标量系数。

**强制**：linear · bias-free · shallow · local · **no attention · no nonlinear activation · no ResBlock · no condition injection · no learned semantic prior**。

### $S_k(0) = 0$

实现方式：所有 linear / conv 去掉 bias，不使用 learned constant，加零输入正则：

$$
\mathcal{L}_{zero} = \sum_k \|S_k(0)\|_2^2.
$$

若 $z^{(k)} = 0$ 时 $S_k$ 仍输出有结构的 noise pattern，说明 $S_k$ 已在携带内容先验。

### 使用 dense token field，而非单向量

推荐 $z^{(k)} \in \mathbb{R}^{h_k \times w_k \times d_k}$，例如：
- early token field：$16 \times 16 \times d$
- middle token field：$32 \times 32 \times d$
- late token field：$64 \times 64 \times d$

---

## 8. 训练目标

### 8.1 完整噪声预测损失

$$
\mathcal{L}_{\epsilon} = \left\| \epsilon - \sum_{k=1}^{K} S_k(z^{(k)}) \right\|_2^2.
$$

### 8.2 前缀去噪损失

令 $x_0^{\star(m)}$ 为第 $m$ 级粒度目标（可由下采样/模糊/perceptual teacher feature 产生，不依赖频域变换）：

$$
\mathcal{L}_{prefix} = \sum_{m=1}^{K} \lambda_m \, d\!\left(\hat{x}_0^{(m)},\; x_0^{\star(m)}\right), \qquad x_0^{\star(K)} = x_0.
$$

### 8.3 单调改进损失

定义 $\ell_m = d(\hat{x}_0^{(m)}, x_0)$，要求 $\ell_{m+1} < \ell_m$：

$$
\mathcal{L}_{mono} = \sum_{m=1}^{K-1} \max\!\left(0,\; \delta_m - (\ell_m - \ell_{m+1})\right).
$$

防止模型把所有信息塞进第一个 token，让后面 token 退化为噪声。

### 8.4 残差解释损失

定义 $r_\epsilon^{(0)} = \epsilon$，第 $m$ 个前缀后 $r_\epsilon^{(m)} = \epsilon - \sum_{k=1}^{m} S_k(z^{(k)})$：

$$
\mathcal{L}_{res} = \sum_{m=1}^{K} \lambda_m \left\| r_\epsilon^{(m)} \right\|_2^2.
$$

### 8.5 子空间去相关

若 $S_k(z) = B_k z$：

$$
\mathcal{L}_{orth} = \sum_{i \neq j} \|B_i^\top B_j\|_F^2.
$$

若 $S_k$ 是局部卷积，则对输出 component 做 batch-level decorrelation：

$$
\mathcal{L}_{decor} = \sum_{i \neq j} \left|\left\langle S_i(z^{(i)}),\, S_j(z^{(j)}) \right\rangle\right|.
$$

### 8.6 能量预算

控制每个 token level 的贡献比例 $\rho_k$（$\sum_k \rho_k = 1$）：

$$
\mathcal{L}_{energy} = \sum_k \left(\frac{E_k}{\sum_j E_j} - \rho_k\right)^2, \qquad E_k = \mathbb{E}\!\left[\|S_k(z^{(k)})\|_2^2\right].
$$

### 8.7 总目标

$$
\mathcal{L} = \mathcal{L}_{\epsilon} + \lambda_{prefix}\mathcal{L}_{prefix} + \lambda_{mono}\mathcal{L}_{mono} + \lambda_{res}\mathcal{L}_{res} + \lambda_{orth}\mathcal{L}_{orth} + \lambda_{energy}\mathcal{L}_{energy} + \lambda_{zero}\mathcal{L}_{zero}.
$$

**MVP 最小集**（第一版只保留）：

$$
\mathcal{L} = \mathcal{L}_{\epsilon} + \lambda_{prefix}\mathcal{L}_{prefix} + \lambda_{mono}\mathcal{L}_{mono}.
$$

---

## 9. 推理过程

```
Input: x_t, t, c

z^(1) = T_1(x_t, t, c)
z^(2) = T_2(x_t, z^(1), t, c)
...
z^(K) = T_K(x_t, z^(<K), t, c)

eps^(k)   = S_k(z^(k))
eps^(1:m) = sum_{k=1}^m eps^(k)
x0_hat^m  = (x_t - sigma_t * eps^(1:m)) / alpha_t

→ 任意时刻停止于 m，得到 budget=m 下的去噪结果
→ 最终图像：x0_hat = x0_hat^(K)
```

**关键约束**：$T_k$ 可以很强；$S_k$ 必须很弱，只看 $z^{(k)}$。

---

## 10. 与 Latent Forcing 的关系

| 维度 | Latent Forcing | CoFiTok |
|------|---------------|---------|
| 核心对象 | latent + pixel 的联合轨迹 | dense noise prediction 的 token 分解 |
| latent 作用 | scratchpad / conditioning signal | 每个 token 是压缩负噪声分量 |
| pixel 生成 | latent 与 pixel 使用不同 schedule | token 展开后直接参与 pixel-space denoising |
| 信息顺序 | latent before pixels | coarse denoising token → fine denoising token |
| VAE decoder 瓶颈 | 目标是减少该问题 | 明确不使用 final latent decoder |

**最核心的区分句**：

> Latent Forcing reorders the diffusion trajectory across latent and pixel variables; CoFiTok tokenizes the noise prediction **inside** the trajectory itself.

---

## 11. 相关工作与区分

### Spectral Image Tokenizer（arXiv:2412.09607）

使用 DWT 得到 coarse-to-fine image spectrum token，支持 partial decoding。

**区分**：CoFiTok 不使用 DWT / FFT / DCT，不把图像转成显式频域 token；"低频到高频"只是直觉，正式方法是 learned noise component factorization。

---

### FlexTok（arXiv:2502.13967）

将 2D 图像映射成 variable-length ordered 1D token sequence，nested dropout 实现任意前缀合法，用于 AR generation；token 形成 coarse-to-fine visual vocabulary。

**区分**：FlexTok 的 token 是图像表示 / tokenizer output；CoFiTok 的 token 是 diffusion negative-noise component，本身参与 pixel-space denoising update。

---

### MRL / M3 / MQT-LLaVA

Matryoshka 系列解决 representation / VLM token budget；token 是静态 embedding。

**区分**：CoFiTok 解决 diffusion noise prediction 的 next-token factorization；token 是 dense noise component 的压缩表示，前缀直接对应 pixel-space denoising result。

---

### AdaTok（arXiv:2606.07185）

Self-budgeting image tokenization，nested tail masking + adaptive token allocation。

**区分**：AdaTok 解决 image tokenization / reconstruction budget；CoFiTok 解决 diffusion noise prediction budget，控制的是每次 denoising 中解释多少 noise component。

---

### ⚠️ Selftok（arXiv:2505.07538）+ D-AR（arXiv:2505.23660）：最高风险

**Selftok** 已经提出用 reverse diffusion process 将图像编码为 AR discrete tokens，每个 token 对应 diffusion time step，可用于纯 AR VLM。

**D-AR** 把 diffusion 改写为 next-token prediction，tokens decoded into pixel-space denoising steps，token 顺序自然 coarse-to-fine。

| 维度 | Selftok / D-AR | CoFiTok |
|------|---------------|---------|
| token 类型 | discrete AR visual tokens | continuous compressed negative-noise components |
| 核心目标 | AR / VLM / RL 兼容 | dense diffusion noise prediction 的有序低容量分解 |
| decode 机制 | token → diffusion step / renderer | $S_k$ 是受限线性合成算子，不是强 decoder |
| 防退化机制 | AR causal schedule | 限制 $S_k$ 容量/输入/子空间 |

**必须强调的区分**：

> Our contribution is not merely mapping diffusion steps to visual tokens, but **factorizing dense noise prediction into ordered compressed components with restricted synthesis operators**, so that each token is itself a negative-noise component rather than a code decoded by a powerful renderer.

---

## 12. 论文最安全的创新表述

❌ 不要写：
- "We are the first to propose coarse-to-fine visual tokens."
- "Each diffusion step is a token."（会被 Selftok / D-AR 直接卡）

✅ 应该写：

> Existing ordered tokenizers represent images as flexible token sequences, and recent diffusion-AR methods map diffusion steps to visual tokens. CoFiTok instead studies whether the **dense noise prediction itself** can be factorized into an ordered sequence of compressed denoising components. Each token is expanded only by a restricted, condition-free synthesis operator, making the token itself responsible for carrying the negative-noise information. This yields prefix-controllable pixel-space denoising **without relying on a final VAE-style decoder**.

---

## 13. 预期贡献

### Contribution 1：Denoising-token factorization

$$
\hat{\epsilon} = \sum_{k=1}^{K} S_k(z^{(k)}),
$$

其中 $z^{(k)}$ 是有序 compressed denoising token。

### Contribution 2：Restricted synthesis operator

设计低容量、无条件、线性/局部的 $S_k$，防止退化成 decoder，使样本相关信息必须由 token 本身携带。

### Contribution 3：Prefix-controllable pixel-space denoising

任意前缀 $z^{(1:m)}$ 对应合法的 partial noise prediction $\hat{\epsilon}^{(1:m)}$，直接得到预算为 $m$ 的去噪图像 $\hat{x}_0^{(m)}$。

### Contribution 4：Ordering diagnostics for denoising components

通过 reverse-order / random-order / simultaneous / unrestricted $S_k$ / deep decoder $S_k$ 等 ablation 验证有序 token factorization 的必要性与 $S_k$ 的非退化性。

---

## 14. 最小可行实验设计

### Stage 1：小数据集验证

- 数据：CIFAR-10 / ImageNet-64 / ImageNet-256 子集
- 模型：small U-Net 或 small DiT；$K = 4, 8, 16$
- $T_k$：shared transformer/U-Net block with step embedding
- $S_k$：bias-free $1\times1$ projection + fixed upsample + shallow linear conv

### Stage 2：证明 prefix 有意义

输出 $\hat{x}_0^{(1)}, \hat{x}_0^{(2)}, \ldots, \hat{x}_0^{(K)}$，预期：
- $m=1$：整体布局、主体轮廓
- $m=K/2$：物体边界和主要部件
- $m=K$：纹理、细节、完整图像

指标：prefix PSNR · LPIPS · rFID · perceptual quality

### Stage 3：证明 $S_k$ 没有退化成 decoder（必做）

| 测试 | 方法 | 预期 |
|------|------|------|
| Zero-token test | $z^{(k)} = 0 \Rightarrow S_k \approx 0$ | 无结构 noise |
| Random-token test | $z^{(k)} \sim \mathcal{N}(0,I)$ | 无语义 noise，不像自然图像 |
| Shuffled-token test | batch 内打乱 $z^{(k)}$ | 结果不匹配原图 |
| Deep-$S_k$ ablation | 替换为深 decoder | prefix ordering 变差，zero/random test 失败 |

### Stage 4：顺序消融

| 方法 | 描述 |
|------|------|
| **Ordered CoFiTok** | 正常顺序（ours） |
| Reverse-order | 反向预测 token |
| Random-order | token 顺序随机 |
| Simultaneous | 一次性预测所有 component |
| Unrestricted $S_k$ | $S_k$ 变成强 decoder |
| No prefix loss | 去掉前缀去噪监督 |
| No monotonic loss | 去掉单调改进约束 |

---

## 15. 预期图表

**Figure 1：方法总览**
- 展示 $x_t \to T_1 \to z^{(1)} \to S_1 \to \epsilon^{(1)}$，依次叠加
- 重点标出：$T_k$ 强，$S_k$ 受限，没有 final VAE decoder

**Figure 2：Prefix denoising visualization**
- $m = 1, 2, 4, 8, K$，每列为一个 prefix denoised image

**Figure 3：$S_k$ 退化诊断**
- zero / random / shuffle test 对比；deep vs restricted $S_k$

**Figure 4：顺序消融曲线**
- $m/K \to$ LPIPS / rFID / prefix error，对比 ordered / random / reverse / simultaneous

**Table 1：相关工作区分表**
- 列：VAE/LDM · Latent Forcing · Spectral · FlexTok · MRL/M3 · Selftok/D-AR · **CoFiTok**

---

## 16. 最大风险与应对

### 风险 1：与 Selftok / D-AR 太近 ⚠️

**应对**：
- 不主打"diffusion step = token"
- 主打"dense noise prediction factorization"
- 强调 $z^{(k)}$ 是 compressed negative-noise component
- 强调 $S_k$ 是受限合成算子，不是 decoder/renderer
- 实验加入 Selftok / D-AR 作为 nearest baseline / concept-level comparison

### 风险 2：$S_k$ 退化为 decoder

**应对**：$S_k$ 只看 $z^{(k)}$；线性/低秩/局部共享；无 bias；无 condition；无 attention；zero/random/shuffle diagnostics；$S_k(0)=0$ 强制。

### 风险 3：所有信息集中到早期 token

**应对**：prefix loss + monotonic improvement loss + energy budget + tail token utilization metric + residual component loss。

### 风险 4：粗到细顺序不自然

**应对**：reverse/random/simultaneous order ablation；prefix visualization；component energy 分析（仅作分析，不作方法）。

---

## 17. 最终推荐论文定位

❌ 不是"又一个 visual tokenizer"  
❌ 不是"又一个 latent diffusion"

✅ **一种 pixel-space diffusion 的噪声预测分解方法**

**最终核心问题**：

> Can dense diffusion noise prediction be factorized into an ordered sequence of compressed denoising tokens?

**最终贡献句**：

> CoFiTok factorizes pixel-space diffusion noise prediction into an ordered sequence of compressed denoising tokens. Each token is expanded by a restricted, condition-free synthesis operator into a dense negative-noise component, and cumulative prefixes yield controllable partial denoising results. This converts diffusion denoising from a monolithic dense prediction into a next-token process without relying on a final VAE-style decoder.

**中文**：

> CoFiTok 将 pixel-space diffusion 的噪声预测从"一次性 dense prediction"改写为"一串有序压缩去噪 token 的累加"。每个 token 本身表示一个负噪声分量，经低容量无条件合成算子展开为 dense noise component；任意 token 前缀都能直接得到对应预算下的 pixel-space 去噪结果。因此，它保留了 next-token 的可控性，同时避免了 latent diffusion 最终依赖 VAE decoder 的瓶颈。

---

## 18. 方法摘要（伪代码）

```
Input:
    noisy image x_t, timestep t, condition c

For k = 1 ... K:
    z_k   = T_k(x_t, z_<k, t, c)   # strong next-token predictor
    eps_k = S_k(z_k)               # restricted synthesis, no condition

Cumulative noise:
    eps_1:m = sum_{k=1}^m eps_k

Prefix denoised image:
    x0_hat_m = (x_t - sigma_t * eps_1:m) / alpha_t

Final denoised image:
    x0_hat = x0_hat_K

KEY CONSTRAINTS on S_k:
    input:    z_k only (no x_t, no t, no text, no z_<k, no skip)
    no bias   no attention   no nonlinear decoder   no condition injection
    S_k(0) = 0
```

> 这个 idea 的价值不在"token 有粗到细顺序"本身，而在于把 diffusion 的 dense noise prediction 显式分解成可前缀截断的 next-token 负噪声序列。论文成败取决于两个点：一是能否证明 ordered noise-token factorization 真的改善生成/可控性；二是能否证明 $S_k$ 没有退化成 decoder。

---

## 参考文献

1. **Latent Forcing** – arXiv:2602.11401. Reordering the diffusion trajectory via separately tuned noise schedules for latent and pixel.
2. **Spectral Image Tokenizer** – arXiv:2412.09607. DWT-based coarse-to-fine image tokenizer with partial decoding.
3. **FlexTok** – arXiv:2502.13967. Variable-length ordered 1D image tokenizer with nested dropout and AR generation.
4. **Matryoshka Representation Learning** – arXiv:2205.13147. Nested representation subsets for adaptive retrieval.
5. **M3 / Matryoshka Multimodal Models** – arXiv:2405.17430. Nested visual tokens for VLM inference-time granularity control.
6. **MQT-LLaVA** – arXiv:2405.19315. Matryoshka Query Transformer for flexible visual token count in VLM.
7. **AdaTok** – arXiv:2606.07185. Self-budgeting image tokenization with quality-preserving dynamic tokens.
8. **Selftok** – arXiv:2505.07538. Discrete visual tokens of autoregression via reverse diffusion process.
9. **D-AR** – arXiv:2505.23660. Diffusion via autoregressive models; tokens decoded into pixel-space denoising steps.
10. **DDPM** – Ho et al., NeurIPS 2020. Standard denoising diffusion probabilistic models.
11. **Flow Matching** – Lipman et al., ICLR 2023. Continuous normalizing flows via conditional flow matching.

---

## 19. 后续扩展方向

> 如果主方法成立，图像就变成一种**可操作的有序 denoising token 表示接口**：
>
> $$x \;\leftrightarrow\; Z = \{z^{(1)}, z^{(2)}, \dots, z^{(K)}\}$$
>
> 生成 = 预测 token；重建 = 累积 token；细化 = 追加 token；编辑 = 修改 token。

### Extension A：Token-space Image Editing

给定一张图的 token 表示 $Z = \{z^{(1)},\dots,z^{(K)}\}$，通过修改某些层的 token 来编辑图像：

$$z^{(j)} \rightarrow \tilde{z}^{(j)} \implies \tilde{\epsilon} = \sum_{k \neq j} S_k(z^{(k)}) + S_j(\tilde{z}^{(j)}).$$

**层级对应的编辑类型**：

| 修改层级 | 预期编辑效果 |
|---------|------------|
| 早期 token（$k$ 小） | 改主体布局、姿态、整体结构、相机视角 |
| 中间 token | 改部件、物体边界、局部形状 |
| 后期 token（$k$ 大） | 改纹理、材质、颜色细节、文字、小装饰 |

**文本驱动 token 编辑**：训练 token editor $G_k$，根据编辑指令 $e$ 生成残差：

$$\Delta z^{(k)} = G_k(Z, e), \qquad \tilde{z}^{(k)} = z^{(k)} + \Delta z^{(k)}.$$

**与普通 latent editing 的本质区别**：普通 latent editing 改的是 entangled 整体 latent，难以局部控制；token editing 改的是结构化有序 component，可以精确选层。

**重要副产品**：层级编辑实验可以反过来**验证主方法的分解有效性**——如果改早期 token 真的改结构、改后期 token 真的改细节，就证明了 token 确实学到了从粗到细的信息顺序。编辑实验不仅是下游应用，更是主方法的解释性验证实验。

---

### Extension B：Appended Refinement Tokens

不重新生成全部 token，而是在已有 $K$ 个 token 后追加少量 refinement tokens $z^{(K+1)}, \dots, z^{(K+R)}$：

$$\hat{\epsilon} = \underbrace{\sum_{k=1}^{K} S_k(z^{(k)})}_{\text{original}} + \underbrace{\sum_{r=1}^{R} S_{K+r}(z^{(K+r)})}_{\text{appended refinement}}.$$

**refinement token 的典型用途**：

| Refinement Token 类型 | 功能 |
|----------------------|------|
| sharpen token | 增强整体清晰度 |
| face refinement token | 补人脸细节 |
| OCR detail token | 补文字 / 图表细节 |
| small-object token | 补充小物体 |
| lighting correction token | 调整光照 |
| style token | 改风格但不动结构 |

**交互式生成流程**：
1. 先用前 $m$ 个 token 低成本出粗图
2. 用户看图决定是否满意
3. 不满意 → 追加 token 细化，不重新运行整个模型
4. 满意 → 停止，按需保存任意粒度版本

---

### Extension C：Visual Memory / Token Diff Updates

将图像存成 token 流 $Z = \{z^{(1)}, \dots, z^{(K)}\}$，场景更新只需传输 token diff：

$$Z_{\text{new}} = Z_{\text{old}} \oplus \Delta Z$$

**应用场景**：
- **流式视觉感知**：粗结构 token 先到，细节 token 后补，天然适配带宽受限传输
- **机器人场景记忆**：新观察只更新被改变层级的 token，保留未变化部分
- **VLM 视觉接口**：先送少量粗粒度 token 做快速全局理解，任务需要细节时追加

---

### 三条后续论文路线

| 路线 | 题目方向 | 核心研究问题 |
|------|---------|------------|
| **A. Token Editing** | Editing Images by Manipulating Ordered Denoising Tokens | 改哪层 token 对应哪种编辑？token editing 比 latent editing 更可控吗？ |
| **B. Token Refinement** | Progressive Image Refinement with Appended Denoising Tokens | 能否先出粗图再追加 token 提升质量？不同任务需要哪种 refinement token？ |
| **C. Visual Memory** | Editable Visual Memory as Ordered Denoising Token Streams | 图像能否存成可更新的 token 流？能否只传输 token diff 来更新图像？ |
