# CoFiTok 必读说明

每次进入本项目时，先阅读并遵守本文档；理解研究背景时同时阅读：

```text
CoFiTok-Denoising Token Diffusion.md
```

项目短名：`CoFiTok`

项目全称：`CoFiTok: Coarse-to-Fine Denoising Tokens for Pixel-Space Diffusion`

中文定位：把 pixel-space diffusion 的 dense noise prediction 从一次性预测改写为一串有序压缩去噪 token 的累加；每个 token 经受限、无条件、低容量的 synthesis operator 展开为 dense negative-noise component，前缀 token 直接对应可控的 partial denoising。

## 当前研究边界

CoFiTok 当前关注的是 `dense diffusion noise prediction` 的有序分解，而不是又一个普通 visual tokenizer、latent tokenizer 或 VAE-style latent diffusion。

最稳妥的核心主张：

> CoFiTok studies whether dense pixel-space diffusion noise prediction can be factorized into an ordered sequence of compressed denoising components. Each token is expanded only by a restricted, condition-free synthesis operator, so token prefixes correspond to controllable partial denoising without relying on a final VAE-style decoder.

写作和实现时保持这些边界：

- `z^(k)` 是 compressed negative-noise component，不是最终图像 latent code。
- `S_k(z^(k))` 只能看当前 token；不要输入 `x_t`、text prompt、class condition、time embedding、previous tokens、CLIP/DINO features、U-Net skip connection 或 learned constant。
- `S_k` 必须弱：linear、bias-free、shallow、local；默认不允许 attention、nonlinear decoder、ResBlock 或 condition injection。
- `S_k(0)=0` 是强约束；zero-token、random-token、shuffled-token 是必做退化诊断。
- Prefix denoising 是主卖点：任意 `z^(1:m)` 都应产生合法 partial noise prediction 和对应 pixel-space denoising result。
- 不要把 novelty 写成“首次提出 coarse-to-fine visual tokens”；Selftok / D-AR 是最近邻风险，必须强调本项目分解的是 dense noise prediction 本身。

## 当前本地布局

本地项目根目录：

```text
C:/Users/zixi-/Desktop/PaperField/CoFiTok
```

当前本地目录作为轻量协调工作区，优先保存：

- `AGENTS.md`：项目定位、边界、路径和操作规则。
- `CoFiTok-Denoising Token Diffusion.md`：研究想法、数学形式、实验路线和风险判断。
- `CoFiTok-internal/`：代码、配置、脚本、测试、内部记录和小型报告的工作区。
- `paper/`：论文工程；当前包含 venue-neutral LaTeX 和 AAAI-27 anonymous-submission build。
- 第三方 baseline 仓库不要放进 `CoFiTok-internal/` 主代码树；按下文隔离到远端 `baselines/repos/`。

不要把大型数据集、模型权重、训练 checkpoint、完整生成结果、batch samples、feature cache 或长日志直接放在本地项目根目录。

## 服务器连接

当前服务器使用 `pro6000`。本机 `~/.ssh/config` 应包含：

```sshconfig
Host pro6000
  HostName connect.bjb1.seetacloud.com
  User root
  Port 29844
  IdentityFile ~/.ssh/id_rsa
```

连接方式：

```bash
ssh pro6000
```

正式实验、训练、批量采样、模型下载和大规模评估默认都在 `pro6000` 上进行。

## 服务器主工作区

服务器项目根目录：

```text
/root/autodl-tmp/CoFiTok
```

代码工作区：

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

数据集目录：

```text
/root/autodl-tmp/CoFiTok/datasets
```

权重、checkpoint、训练输出和可复现实验 run 目录：

```text
/root/autodl-tmp/CoFiTok/checkpoints
```

服务器初始化建议：

```bash
ssh pro6000
mkdir -p /root/autodl-tmp/CoFiTok/CoFiTok-internal
mkdir -p /root/autodl-tmp/CoFiTok/datasets
mkdir -p /root/autodl-tmp/CoFiTok/checkpoints
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
```

不要把 CoFiTok 的新实验写到 `/root`、旧临时目录或其他项目目录下。若需要兼容相对路径，也应让相对路径指向本项目自己的真实目录副本，而不是其他项目的 symlink。

## 资产规则

服务器上的项目级数据资产统一放在：

```text
/root/autodl-tmp/CoFiTok/datasets
```

服务器上的模型基座权重、diffusion checkpoint、LoRA / adapter、训练 checkpoint、生成样本和评估输出统一放在：

```text
/root/autodl-tmp/CoFiTok/checkpoints
```

数据集、权重、生成样本、feature cache 和完整 run 输出不能提交进代码仓库。服务器上发现属于 CoFiTok 的数据或权重 symlink 时，优先复制为本项目自己的真实文件或目录，并确认：

```bash
find /root/autodl-tmp/CoFiTok -path /root/autodl-tmp/CoFiTok/checkpoints/hf_cache -prune -o -type l -print
```

没有输出。

例外：`/root/autodl-tmp/CoFiTok/checkpoints/hf_cache` 如果作为 HuggingFace Hub 标准缓存，内部 `snapshots/` 指向同一 cache 内 `blobs/` 的 symlink 可以保留，不要复制膨胀。

本地大型数据集和模型权重只做归档或离线备份，不作为主要实验路径。需要从本机同步时，优先从这些 hub 取源：

```text
D:/datasets_raw_hub
D:/checkpoints_hub
```

示例传输命令：

```powershell
scp -r D:\datasets_raw_hub\registry\<dataset>\raw pro6000:/root/autodl-tmp/CoFiTok/datasets/<dataset>
scp -r D:\checkpoints_hub\<needed_checkpoint_or_cache> pro6000:/root/autodl-tmp/CoFiTok/checkpoints/
```

AutoDL / pro6000 访问 Hugging Face 或 GitHub 时，优先在远端命令前启用学术加速：

```bash
source /etc/network_turbo
```

如需镜像站，再设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

完成下载或不再需要代理时可关闭：

```bash
unset http_proxy && unset https_proxy
```

## 数据集状态（2026-07-09）

MVP 所需数据均已在 `pro6000` 完成并记录；主论文非分类扩展数据也已补入：

| alias | 当前用途 | 远端目录 | 状态 |
|---|---|---|---|
| `cifar10` | smoke / 快速诊断 | `/root/autodl-tmp/CoFiTok/datasets/cifar10` | 已完成 |
| `tiny_imagenet_200` | MVP 主实验 | `/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200` | 已完成 |
| `imagenet_1k_64x64_hf` | 当前 ImageNet-family 64x64 实验结果来源 | `/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_64x64_hf` | 已完成 |
| `downsampled_imagenet_64` | strict exact source / 后续 exact-source rerun | `/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64` | 已完成 |
| `ffhq_64` | 主论文 P0 非分类人脸数据 / prefix visualization | `/root/autodl-tmp/CoFiTok/datasets/ffhq_64` | 已完成 |
| `afhqv2_64` | 主论文 P0/P1 非分类动物纹理数据 / late-token detail diagnostic | `/root/autodl-tmp/CoFiTok/datasets/afhqv2_64` | 已完成 |
| `imagenet_1k_256x256_hf` | ImageNet-256 HF full raw source | `/root/autodl-tmp/CoFiTok/datasets/imagenet_1k_256x256_hf` | 已完成 |
| `imagenet_256` | P1 full ImageNet-256 train + val | `/root/autodl-tmp/CoFiTok/datasets/imagenet_256` | 已完成 |
| `imagenet_256_10pct` | P1 ImageNet-256 scaling validation，10% train + full val | `/root/autodl-tmp/CoFiTok/datasets/imagenet_256_10pct` | 已完成 |

`downsampled_imagenet_64` 来自 Academic Torrents：

```text
https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b
info hash: 96816a530ee002254d29bf7a61c0c158d3dedc3b
train: 1,281,149 PNG
valid: 49,999 PNG
```

现有论文/审计结果表仍主要使用 `imagenet_1k_64x64_hf`；不要把这些结果静默改名为 `downsampled_imagenet_64`。只有在 strict source 上重跑后，才能用 `downsampled_imagenet_64` 标注对应实验结果。

P1 `imagenet_256` 和 `imagenet_256_10pct` 已在服务器完成 staging。官方 ImageNet-1K raw 已在本地 `D:/datasets_raw_hub/registry/imagenet1k/raw` 存在并通过 MD5 校验。当前服务器 ImageNet-256 实验路径由 HF 256x256 parquet raw 派生，详细记录：

```text
CoFiTok-internal/docs/experiment_conditions/imagenet_256_full_2026-07-09.md
CoFiTok-internal/docs/experiment_conditions/imagenet_256_10pct_2026-07-09.md
```

早期 full ImageNet-256 空间/派生策略记录仍可参考：

```text
CoFiTok-internal/docs/experiment_conditions/imagenet_256_feasibility_2026-07-09.md
```

新增或重下数据后必须在 `CoFiTok-internal/docs/experiment_conditions/` 写入记录，至少包含：

- 数据集 alias、来源 URL 或 loader、下载日期。
- 原始压缩包路径、解压目录、train / val / test split。
- 文件数量、总大小、sha256 或 manifest。
- 图像分辨率、是否 center crop / resize、是否使用 class label。
- 对应实验配置和输出目录。

## 代码仓库规则

`CoFiTok-internal/` 应保存：

- 源码、配置、脚本和测试。
- `docs/records/` 下的实验记录。
- `docs/experiment_conditions/` 下的数据、权重、环境和运行条件记录。
- 小型 markdown / json / csv 报告、manifest、checksum 和论文可引用摘要。

不要提交：

- 完整数据集、完整生成样本、大型 `outputs/`、`runs/`、`wandb/`、hidden cache、feature cache。
- `.pt`、`.pth`、`.ckpt`、`.safetensors`、`.bin`、`.onnx` 等大权重文件。
- 私钥、token、云端凭据或任何个人认证信息。

若后续创建 GitHub 私有仓库，命名保持 `CoFiTok-internal`；具体 `origin` 以实际配置为准，不要在文档里伪造尚未存在的 remote。

## Baseline 对比代码规则

Baseline 目标是公平回答四类威胁：monolithic pixel diffusion、VAE/latent diffusion、ordered/flexible visual tokenizer、diffusion-as-AR / continuous-token AR。不要把 baseline 变成无关 SOTA 堆表。

代码隔离：

| 类型 | 位置 | 规则 |
|---|---|---|
| 外部 GitHub repo clone | `/root/autodl-tmp/CoFiTok/baselines/repos/<alias>` | 不提交进 `CoFiTok-internal`，不默认设 submodule |
| 我们写的 adapter / runner / config bridge | `CoFiTok-internal/baselines/<alias>` 或 `CoFiTok-internal/scripts/baselines/` | 只放轻量 glue code |
| baseline 配置 | `CoFiTok-internal/configs/baselines/<alias>/` | 记录同数据、同分辨率、同评估协议 |
| baseline 输出 | `/root/autodl-tmp/CoFiTok/checkpoints/baselines/<alias>` | checkpoint、samples、logs 不提交 |
| baseline 报告 | `CoFiTok-internal/artifacts/reports/baselines/<alias>` | 只保留小型 json/csv/md 摘要 |

外部 repo 必须 pin：

- repo URL、license、commit hash、clone 日期。
- 环境创建命令、依赖版本、CUDA / PyTorch 版本。
- 原始命令、patch 或本地修改；如需改外部代码，优先保存 patch 到 `CoFiTok-internal/baselines/patches/<alias>/`，不要直接把 fork 混进主包。

Baseline 优先级：

| 优先级 | baseline | 角色 |
|---|---|---|
| P0 | same-backbone dense `epsilon/v` predictor | 最关键公平对照；隔离 factorization 收益 |
| P0 | EDM / DDPM-style pixel diffusion | monolithic pixel-space diffusion |
| P0 | D-AR | 最近邻 diffusion-as-AR 威胁 |
| P1 | FlexTok / MAR | ordered visual tokenizer、continuous-token AR 对照 |
| P1 | TiTok / ReTok | compact/flexible tokenizer 对照，视分辨率和官方代码可用性决定 |
| P2 | LDM / DiT / VAR / Selftok / Spectral / Latent Forcing | 大规模或 related/nearest comparison；不要强行进 MVP 主表 |

公平配置至少记录：

- 数据集 alias、split、分辨率、预处理和是否使用 label。
- train steps / epochs、batch size、optimizer、seed、参数量、NFE、采样步数。
- FID / rFID / LPIPS / PSNR / Prefix-AUC / monotonic violation / zero-random-shuffle diagnostics。
- 评估脚本版本和 real/generated sample count。

现有 CoFiTok 结果中的 `imagenet_1k_64x64_hf` 与 strict `downsampled_imagenet_64` 必须分开报告；baseline 也必须使用同一 alias 才能横向比较。

## 本地运行原则

本地机器只适合阅读、编辑、写作和极小规模 smoke test。

本地可以做：

- 文档整理和实验计划。
- 指标函数、路径拼接、配置解析的 CPU 单元测试。
- toy tensor / tiny U-Net / tiny DiT 的逻辑检查。
- 小型 JSON / CSV / parquet manifest 检查。
- 从 `pro6000` 同步回来的报告、表格、图和少量样例整理。

不要在本地运行大模型训练、pixel diffusion 完整训练、批量采样、ImageNet 级评估、大规模 FID / LPIPS 计算、批量 embedding 抽取或长时间 GPU 作业。

## 本地 uv 环境

如果需要在本机验证 `CoFiTok-internal/` 代码，只在该目录内创建项目专属虚拟环境。不要使用系统 Python、全局 conda 环境，也不要和其他项目共用 `.venv`。

推荐：

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
uv venv .venv --python 3.10
```

如项目已有 `pyproject.toml`，优先使用：

```powershell
uv sync
```

如只有 `requirements.txt`，使用：

```powershell
uv pip install -r requirements.txt
```

本地 `.venv` 只用于轻量验证，不用于正式训练或完整 benchmark。

## 服务器环境原则

跑远端任务前先检查 GPU：

```bash
nvidia-smi
```

不要杀其他项目的 GPU 进程，除非用户明确要求。

优先检查是否可复用服务器现有 PyTorch / CUDA 环境；如果 `pf-vlm` 能满足轻量开发和 smoke test，可先使用：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
```

如果 diffusion 训练需要单独环境，环境名建议使用 `cofitok` 或 `cofitok-diffusion`，并把完整创建命令、CUDA / PyTorch 版本、关键依赖和验证命令写入：

```text
CoFiTok-internal/docs/experiment_conditions/
```

## MVP 实验路线

第一阶段先做小数据集和小模型闭环：

- 数据：先 `cifar10` smoke，再以 `tiny_imagenet_200` 为第一版主实验；`downsampled_imagenet_64` 只作为 MVP 跑通后的升级验证。
- 模型：small U-Net 或 small DiT。
- Token 数：`K = 4, 8, 16`。
- `T_k` 可以是强 next-token predictor；`S_k` 必须是受限 synthesis operator。
- 首批结果必须包含 prefix denoising visualization：`m = 1, 2, 4, 8, K`。

最小训练目标优先：

```text
L = L_epsilon + lambda_prefix * L_prefix + lambda_mono * L_mono
```

扩展项如 residual loss、decorrelation、energy budget、zero-token regularization 可以进入第二阶段，不要一开始把 MVP 做得过重。

## 必做诊断和消融

证明 `S_k` 没有退化成 decoder：

- Zero-token test：`z^(k)=0` 时 `S_k(z^(k))` 应接近零且无结构。
- Random-token test：随机 token 不应生成自然图像语义结构。
- Shuffled-token test：batch 内打乱 token 后结果应不匹配原图。
- Deep-`S_k` ablation：替换为强 decoder 后应暴露退化风险或破坏 prefix ordering 解释。

证明顺序有意义：

- Ordered CoFiTok。
- Reverse-order。
- Random-order。
- Simultaneous prediction。
- Unrestricted `S_k`。
- No prefix loss。
- No monotonic loss。

评估同时报告质量、前缀可控性和成本：

- Prefix PSNR / LPIPS。
- rFID / FID 或小规模替代指标。
- Component energy ratio。
- Tail token utilization。
- Zero / random / shuffle diagnostics。
- 训练吞吐、显存和采样成本。

## 写作规则

不要写：

- “We are the first to propose coarse-to-fine visual tokens.”
- “Each diffusion step is a token.”
- “CoFiTok solves image tokenization broadly.”

优先写：

> CoFiTok factorizes pixel-space diffusion noise prediction into an ordered sequence of compressed denoising tokens. Each token is expanded by a restricted, condition-free synthesis operator into a dense negative-noise component, and cumulative prefixes yield controllable partial denoising results.

相关工作必须明确区分：

- Latent diffusion / VAE decoder bottleneck。
- Latent Forcing。
- Spectral Image Tokenizer / FlexTok / Matryoshka visual token work。
- Selftok / D-AR 等 diffusion-AR 最近邻方法。

## 新会话启动建议

新会话开始后先读：

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok
Get-Content -Raw -Encoding UTF8 .\AGENTS.md
Get-Content -Raw -Encoding UTF8 '.\CoFiTok-Denoising Token Diffusion.md'
Get-ChildItem -Force .\CoFiTok-internal
```

如果需要检查远端：

```powershell
ssh pro6000 "cd /root/autodl-tmp/CoFiTok && pwd && find . -maxdepth 2 -type d | sort"
ssh pro6000 "nvidia-smi"
```

## 当前交接状态（2026-07-11）

- Baseline repo 已隔离、pin 并审计于 `/root/autodl-tmp/CoFiTok/baselines/repos/`；外部代码不进 `CoFiTok-internal/`。
- 最终 matrix 为 `80 completed + 16 completed_eval_only + 24 protocol_blocked`：`CoFiTok-internal/artifacts/reports/paper_comparison_matrix_2026-07-11_final_guard/`。`protocol_blocked` 是明确的跨任务公平性边界，不是缺失实验。
- `endpoint_only_factorized` 仍保留 K-token factorization；`same_backbone_dense` 才是参数匹配的直接 dense 对照。K8 参数差 `-1.35%`，K4 差 `-0.55%`，两者不得混称。
- 八数据集 short-budget 中 CoFiTok path AUC 对 endpoint-only 为 `8/8` 胜；Tiny/ImageNet-64 K8 20K 双种子为 `4/4` 胜，平均 path AUC 下降 `96.18%`，endpoint `$x_0$` MSE 平均变化 `+4.20%`。
- ImageNet-256 K4 20K 双种子 confirmatory 为 `7/7` gate 通过；学到的顺序在全部 24 个 permutation 中均排名第 1，平均 endpoint MSE 相对 direct dense 下降 `6.73%`：`CoFiTok-internal/artifacts/reports/imagenet256_confirmatory_2026-07-11/`。
- D-AR / MAR / ReTok official ImageNet-256 50K 已完成，FID 分别为 `2.6281 / 2.3385 / 2.2189`。MAR 行使用 LTH14 PTH `model_ema`；三者只进 secondary eval-only table，不并入 P0 same-budget 表。
- 最终记录：`CoFiTok-internal/docs/records/2026-07-11_final_baselines_and_paper_evidence_completion.md`；evidence pack：`CoFiTok-internal/artifacts/reports/paper_evidence_report_2026-07-11_final/`；AAAI-27 稿：`paper/venues/aaai27/`。
- AAAI-27 使用双入口：`main_aaai2027.tex` 与 `supplementary_aaai2027.tex`；支撑材料包含 8 数据集 x 15 方法覆盖矩阵、P0 外部生成比较、P1 tokenizer 重建和官方 50K secondary rows。
- AAAI-27 组件消融已补齐：ImageNet-64 HF、5K、seeds 103/139、1,024 图像统一 `t=500` 协议共 `22 train + 34 eval`；报告/曲线见 `CoFiTok-internal/artifacts/reports/aaai27_ablation_2026-07-11/` 与 `CoFiTok-internal/artifacts/figures/ablation_hyperparameters_2026-07-11/`，Experiments 仅保留 Setup/Results/Ablations。
- `paper/latex/` 与 AAAI-27 主文的 Figure 2 均使用四个独立 PNG，由 LaTeX 以 2x2 `minipage` 排版；文件和来源 manifest 位于 `CoFiTok-internal/artifacts/figures/final_visual_claim_2026-07-11/`，不得重新烘焙成单张总图。
- 证据支持 scoped claim：ordered restricted dense-noise factorization 与 prefix-controllable denoising。不支持 broad generation SOTA：CoFiTok lowres `0/8` 最优、Inception-style `1/8` 最优。

后续只做提交包收口或额外 scaling；不得用新运行静默覆盖上述 locked 结果。

## 大规模生成升级分支（2026-07-12）

- `CoFiTok-internal` 已建立 Git：锁定论文证据基线为 `paper-evidence-locked`（初始提交 `88c3aab`），生成系统升级只在 `scale/generative-system` 开发与运行。
- 新训练不得写入旧 run 目录或覆盖 2026-07-11 locked artifacts；统一写入 `/root/autodl-tmp/CoFiTok/checkpoints/generation/`，小型报告再同步回本地。
- 升级目标与门槛见 `CoFiTok-internal/docs/GENERATION_SYSTEM.md`。10% ImageNet-256 仅是 scaling gate，不能代替 full ImageNet-256 长训与 50K sample 正式评估。
- 正式生成路径使用 class-conditional scalable U-Net `T_k`、EMA、bf16、梯度累积、原子 checkpoint、精确恢复和 CFG/DDIM；class/time condition 仍不得进入 `S_k`。
- 公平主对照为相同数据、骨干、优化器、训练步数和采样协议的 `dense_identity` epsilon predictor；D-AR/MAR/ReTok 继续只作官方 eval-only secondary rows。
- 10% ImageNet-256 matched 50K 队列已于 2026-07-12 从 commit `781a014` 启动：runbook PID `28952`，先 CoFiTok 后 dense；运行记录见 `CoFiTok-internal/docs/records/2026-07-12_generation_10pct_50k_launch.md`。2026-07-14 00:59 CST，CoFiTok 已完成 `50,000/50,000`、`3,200,000` images seen，最终 checkpoint、`latest.json` 和 `training_report.json` 均已生成；同一锁定 runbook 已自动进入 dense 训练。实时权威状态必须读取 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_pair_monitor.json` 和两份 run 的 `train_metrics.jsonl` / `training_report.json`。
- CoFiTok step 5,000、10,000 与 40,000 recovery checkpoint 均已原子落盘（各 `1,008,218,658` bytes）；10K checkpoint SHA256 为 `be342592328feac9f684921d57d4f79ff588d4645f7eae331c197ae0c6431672`，40K checkpoint SHA256 为 `f3af056d3a6945567fcef24c951efbd7e168ec97ec7611a64a4feb33dc16121f`，legacy 审计未改写权重本体。实时状态必须读取 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_pair_monitor.json`，静态审计快照保留于 `CoFiTok-internal/artifacts/reports/generation/training_progress_2026-07-12/`。
- 当前 active matched 队列结束前，远端代码必须继续固定在 `781a014`；本地 `scale/generative-system` 后续加固提交只能在 CoFiTok 和 dense 两个 50K 训练都完成后一次性同步，禁止中途造成训练 revision 不一致。
- 固定 10% 队列的纯只读 monitor 已从 `/tmp/monitor_generation_10pct_pair.py` 启动，PID `58348`，权威状态写入 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_pair_monitor.json`；它只读 metrics/report/checkpoint stat，记录进程、磁盘和 GPU，不加载/哈希权重、不使用 GPU、不重启或杀进程。metric 1,800 秒未更新判 `stalled`，未完成且进程消失超过 600 秒判 `failed`，双 50K report 精确完成才 `pass`。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_10pct_readonly_monitor.md`。
- 正式采样、checkpoint mechanism eval 和推理预检共用 `cofitok.generation.load_generation_model`；采样前必须以目标 EMA/model、batch、precision、prefix 和 CFG mode 执行真实前向，并记录 checkpoint SHA、有限性、耗时和 CUDA 峰值显存。实现记录：`CoFiTok-internal/docs/records/2026-07-12_generation_sampling_preflight.md`。
- `781a014` 产生的 10% checkpoint 没有新版 integrity sidecar，且其 CoFiTok token 仍是 full-resolution channel-masked field；该 pair 只作为 immutable legacy evidence 和代码部署前置验证，不再执行 promotion post-eval，也不得授权 full 300K。部署后必须重跑下述 true-compressed 10% matched pair。
- 在线训练审计对当前固定 10% 队列使用 `legacy_compute`：只读计算最新 checkpoint 的 bytes/SHA 并保留兼容 warning；升级后的 full 300K 必须使用 `--integrity-policy required`，缺 sidecar、bytes/SHA 不符或 `latest.json` 未绑定都直接判为 invalid。
- full ImageNet-256 matched 300K 队列必须在 50K/100K/200K/300K 交替运行 CoFiTok 与 dense，并执行 2,048-sample DDIM-50 milestone trend eval；这些 FID/IS 只作早期质量预警，不能替代最终 50K DDIM-250 full gate。实现记录：`CoFiTok-internal/docs/records/2026-07-12_full_generation_milestone_monitoring.md`。
- 本地已实现可恢复的训练后切换入口 `CoFiTok-internal/scripts/deploy_generation_posttraining_pipeline.ps1`：它只在 legacy CoFiTok/dense 两份 10% 50K 报告都严格完成、训练进程退出且远端仍为锁定 revision 时快进部署。部署后的 completion pipeline 必须先训练同一 target revision 的 true-compressed 10% matched 50K pair，再串行执行其 10K promotion gate、full matched 300K、50K 正式评估和 final gate。当前 legacy 队列结束前严禁执行；记录见 `CoFiTok-internal/docs/records/2026-07-12_generation_posttraining_transition_pipeline.md` 与 `CoFiTok-internal/docs/records/2026-07-14_generation_true_compressed_token_layout.md`。
- 训练后部署在 fast-forward 前必须计算“远端 untracked 文件”和“目标 revision tracked 文件”的路径交集，非空即拒绝移动 HEAD。远端测试与全部 runbook syntax check 通过后，必须原子写入 `checkpoints/generation/generation_upgrade_deployment_receipt.json`，绑定 bundle bytes/SHA/heads、10% pair validation SHA、源/目标 revision 与 clean tracked state；最终 completion audit 必须验证该 receipt。
- 训练后部署包必须使用 pinned `781a014` 作为 prerequisite，只打包 `781a014..scale/generative-system HEAD` 的增量提交；本地必须验证 bundle 只 advertised 一个且等于目标 HEAD，远端必须在 fetch 前通过 `git bundle verify` 证明 prerequisite 存在。2026-07-13 rehearsal 的增量包为 `238,044` bytes（完整历史包为 `85,619,213` bytes），远端只读验证通过；当前 10% 队列完成前仍严禁执行实际部署。
- 2026-07-13 已在服务器 `/tmp` 按真实父目录布局建立隔离 checkout，对目标提交 `33c0c4f0d69a1b3f6589a1cc5296e8b866a7c4d6` 完成 Linux rehearsal：禁用 GPU 后 `408 passed`，`artifacts/runbooks/` 下 `40/40` 个 shell 脚本通过 `bash -n`，tracked status 为空。该 rehearsal 没有 fetch/merge 正式仓库；正式远端 HEAD 仍须保持 `781a014` 直到双 50K 报告完成。
- pinned-worktree prevalidation 不得把新版 `validate_generation_training_pair.py` 作为孤立 `/tmp` 单文件运行：2026-07-13 已真实复现 `ModuleNotFoundError`，且仅补 `generation_pair.py` 仍会因目标 `cofitok/__init__.py` 的传递依赖失败。修复后的 helper 先 verify/fetch bundle 对象但不移动 HEAD，再以 `git archive` 从 exact target 提取 validator 和完整 `src/cofitok/` 到临时 `PYTHONPATH`；只有 pair validation、进程退出和 untracked-conflict 检查都通过才允许 merge。
- 上述修复目标 `20558a18973156cf2ee3b7202a7ea348f783d04b` 已完成服务器 rehearsal：增量 bundle `239,737` bytes，pinned HEAD 前后均为 `781a01444fddbf0d48a427ba58bdeed50167b5be`，隔离 Linux checkout `408 passed`，`40/40` runbook 通过 `bash -n`。当前双 50K 队列完成前仍严禁执行实际 merge。
- 训练报告的权威 Git 字段是 `git.revision`，不是 `git.commit`。promotion/final gate 必须验证 CoFiTok 与 dense 同 revision、同 `scale/generative-system` 分支、同 real set/evaluator、精确 10K/50K 样本数、balanced-modulo class schedule、EMA 与 checkpoint/sample-set SHA256；加固记录：`CoFiTok-internal/docs/records/2026-07-12_generation_gate_provenance_hardening.md`。
- 锁定的 `781a014` 训练器会在写 JSONL 后才执行 scheduled validation，因此当前 10% 队列实际做了验证前向但 `train_metrics.jsonl` 中 `validation_event_count=0`；这是已知 legacy observability warning，不是训练失败。升级分支已改为验证后再写日志，full 300K 应有完整 validation events；记录：`CoFiTok-internal/docs/records/2026-07-12_generation_training_progress_auditor.md`。
- 训练后切换的权威训练对校验器是 `CoFiTok-internal/scripts/validate_generation_training_pair.py`；本地部署器会先把它传到远端 `/tmp` 做 fast-forward 前校验，部署后 completion pipeline 再调用 tracked copy。不得另写散落的 shell JSON 字段判断；validator 统一检查完成步数、`git.revision`/branch、dataset、latest checkpoint、matched data/diffusion/runtime/optimization 和 2% 参数差。
- 升级分支的 exact resume 不只恢复 model/EMA/optimizer/scheduler/RNG/sampler，也必须通过 `cofitok.training.metrics.reconcile_metrics_for_resume` 将 checkpoint 之后或被后续恢复覆盖的 JSONL 行归档为 content-addressed orphan artifact，并原子重写严格递增的 canonical metrics；train/eval iterator 必须在状态恢复后创建，validation iterator 按已完成 scheduled events 对齐。未传 `--resume` 时发现任何旧训练状态必须拒绝启动。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_resume_metrics_reconciliation.md`。
- exact resume 还必须在加载 model/EMA/optimizer/scheduler/scaler/RNG state 之前，对当前 fully resolved config 与 checkpoint 内嵌 config 做递归全字段相等校验；micro-batch、gradient accumulation、data/diffusion/model/loss/runtime/optimization 任一变化都必须以 dotted path 报错，禁止静默改变训练轨迹。实现提交 `6bfc41c9e6de83007cde912e692a1c1f3b044f65` 已完成服务器隔离 rehearsal：Linux `410 passed`、`40/40` runbook 通过，正式 HEAD 前后均为 pinned `781a014`。
- full 300K CoFiTok/dense 配置必须永久保护 50K/100K/200K/300K milestone checkpoint，同时滚动保留最近 3 个恢复点；最终 full-training auditor 必须证明四个 milestone 权重及其 integrity sidecar 仍可用，缺少已到达的保护 checkpoint 时不得进入正式 post-eval。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_milestone_checkpoint_retention.md`。
- authoritative true-compressed full direct matched 对照参数为 CoFiTok `62,837,576`、dense `62,824,707`，差 `+0.020484%`；旧 full-resolution-field 配置的 `62,950,800 / +0.200706%` 仅属 legacy 记录。两者每个 diffusion evaluation 都只运行一次共享 U-Net trunk。分段 checkpoint 必须累计 elapsed/peak VRAM，promotion/final gate 必须验证 exact images seen，最终 matched 表必须报告 effective batch、训练 images/hours/img-s/VRAM；D-AR/MAR/ReTok 仍不得混入 compute-matched panel。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_matched_compute_accounting.md` 与 `CoFiTok-internal/docs/records/2026-07-14_generation_true_compressed_token_layout.md`。
- CoFiTok/dense 公平性必须通过统一 `cofitok.generation_pair.generation_pair_contract`：resolved data/diffusion/runtime/optimization 与所有共享 model 字段（U-Net 宽深度、attention、dropout、gradient checkpointing、class count/dropout 等）精确一致，主 epsilon weight 同为正值；仅 token/synthesis/feedback 字段和 CoFiTok factorization-only auxiliary 可不同。rollout consistency 与 EMA-teacher consistency 属双方必须精确一致的共享稳定化损失，可以同时非零；dense 其余 factorization-only `*_weight` 必须为零。训练对 validator 与 promotion/final gate 必须共用该 contract。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_pair_contract.md`。
- 最终 strong-baseline comparison 使用 schema v5：仅 CoFiTok K8 与 dense identity 属于 `matched_training_direct`；两份 full training report、两份正式 50K metrics report 和 final gate 必须以权威路径、bytes、SHA256 绑定并由 completion audit 重新读取。参数量/effective batch/images seen/训练与采样耗时和吞吐/VRAM/checkpoint SHA/sample-set SHA/real-set tree SHA/evaluator identity/runtime-environment SHA、FID/IS/precision/recall 及完整 DDIM/CFG/clipping/random-stream 协议必须从源报告重算；D-AR/MAR/ReTok 只属于 `official_pretrained_contextual`，必须绑定锁定 `official_related_methods_table.json` 的 SHA、alias/method/eval-only status/secondary role/protocol/source metrics/指标，且 `cross_tier_numeric_ranking_allowed=false`。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_baseline_comparison_provenance.md`、`CoFiTok-internal/docs/records/2026-07-13_formal_sampling_protocol_contract.md` 与 `CoFiTok-internal/docs/records/2026-07-13_generation_comparison_source_provenance.md`。
- 正式 preflight/sampling/checkpoint eval 共用的 `load_generation_model` 必须在 `torch.load` 前验证邻接 integrity sidecar 的 filename/bytes/SHA256/format/step，并把 sidecar 路径传播到 sample 与 gate provenance；禁止只重新计算裸 checkpoint SHA 后接受被替换权重。legacy 10% checkpoint 必须先做 byte-preserving migration。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_checkpoint_trust_boundary.md`。
- 正式 10K/50K 采样每个 batch 后必须原子更新与 immutable sampling-manifest SHA 绑定的 `sampling_progress.json`，记录 invocation、累计耗时、吞吐、ETA、完成数、PID/host 和失败/完成状态；只有 progress completed、累计耗时为有限正数且 count/budgets/sample-set SHA 与 sampling report 完全一致时才能运行 FID 或通过 gate，最终比较同时报告采样耗时和吞吐。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_sampling_progress.md`。
- 大规模生成不得仅凭进程退出或单个 gate 宣称完成；最终 pipeline 写入 `pass` 前必须由 `scripts/audit_large_scale_generation_completion.py` 严格验证 pinned 10% matched pair、受控 revision transition receipt、promotion gate、full 300K matched pair、四个 milestone、训练审计、双方法正式 50K 采样、final gate 与 comparison。缺失证据为 `in_progress`，冲突证据为 `failed`，只有完整证据链才是 `complete`。记录：`CoFiTok-internal/docs/records/2026-07-12_large_scale_generation_completion_audit.md`。
- final `large_scale_generation_ready` 质量门槛必须同时满足：50K EMA DDIM-250 同协议样本；所有 distribution metric 有限且在合法数学范围；CoFiTok FID `<=20.0` 且相对 dense 回退不超过 5%；precision/recall 各 `>=0.30` 且相对 dense 下降不超过 `0.05`；endpoint MSE 回退不超过 5%；ordered-prefix、zero-token、shuffle 和 checkpoint-integrity gates 全通过。completion audit 必须拒绝缺失命名 gate、弱化阈值或与正式 generation report 不一致的 FID/precision/recall。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_quality_readiness_gate.md`。
- full 300K 开始前必须用 checkpoint-free 真实训练 benchmark 为 CoFiTok/dense 共同评估 `16x4/32x2/64x1`，保持 effective batch 64；候选须双方完成且峰值显存不超过设备 90%，按双方较慢者的同步 optimizer-step 时间选择。四个 full milestone 的每次续训都必须使用同一 selection，completion audit 校验两份 training report 与 `runtime_selection.json` 一致。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_training_runtime_selection.md`。
- 正式 10K/50K 采样必须为 CoFiTok/dense 共同实测 `batch=16/32/64/128`，要求同 checkpoint step、同 EMA/bf16/CFG 协议、双方输出有限且峰值显存不超过 90%；batch 32 保守基线必须通过，按双方较慢者的 output-images/s 选择同一 batch。正式 sample manifest 的 per-index RNG 必须保持 batch-size invariant；completion audit 绑定 selection revision、双 checkpoint SHA 与两份 generation report，最终表报告 batch/耗时/吞吐。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_sampling_batch_selection.md`。
- sampling batch selector 的每份真实前向 preflight 必须记录 inference-code Git revision；缓存复用除 checkpoint SHA/step 与完整 EMA/bf16/CFG/request 外还必须匹配当前 revision，禁止把旧代码 preflight 复用后仅在 selection 顶层标成新 revision。completion audit 必须读取 selected candidate 内 CoFiTok/dense 两份 preflight revision。实现提交 `b2139e55523a3eed1702c80ad4839dc05658ac2a` 已通过服务器隔离 `412 passed` 与 `40/40` runbook 检查，正式 HEAD 未移动。
- 正式 sampler 自身也必须把 Git revision、branch、tracked-dirty 状态写入 immutable sampling manifest 与 completed report；metrics 必须传播为 `sample_provenance.git`。promotion/final gate 要求 CoFiTok/dense 使用同一 clean sampling code，full completion audit 还必须绑定 full training revision 与 `scale/generative-system`。10K/50K post-eval runbook 在任何真实前向或采样前先拒绝 tracked dirt。实现提交 `73c2e4aa8f7699e29b63ba7f9d1880ef24ef728c` 已通过服务器隔离 `414 passed`、`40/40` runbook，正式 HEAD 未移动。
- torch-fidelity evaluator 必须独立记录 Git revision、branch、tracked-dirty；CoFiTok/dense metrics reports 必须使用同一 clean evaluator code，full 阶段必须等于 full training/sampling revision。`matched_sampling_code_provenance` 与 `matched_evaluator_code_provenance` 都是 final completion audit 的必备命名 gate，删除任一项即失败。实现提交 `dc581b8695f2748ae9c5403452e12c0c5368333e` 已通过服务器隔离 `417 passed`、`40/40` runbook，正式 HEAD 未移动。
- ordered-prefix、zero-token、shuffle、endpoint MSE 的 checkpoint mechanism evaluator 同样必须独立记录并匹配 clean Git revision/branch；full 阶段必须等于 training/sampling/distribution-evaluator revision。`matched_checkpoint_evaluator_code_provenance` 是第三个 final completion audit 必备命名 provenance gate。实现提交 `359ea0038c74c44e0b01a4cd2cbafe4d9b23af1c` 已通过真实 tiny evaluator CLI、服务器隔离 `419 passed` 与 `40/40` runbook，正式 HEAD 未移动。
- 稳定推理统一使用 `cofitok.generation.GenerationSession` + immutable `GenerationRequest`，一次验证/加载 checkpoint 后复用模型与 schedule；class/seed/prefix CLI 与正式 10K/50K sampler 必须共用该 API。输出需携带 checkpoint SHA/step/integrity 和完整请求 provenance，PNG 原子发布；final completion audit 拒绝没有 `GenerationSession@1` 标识的正式 50K evidence。使用说明：`CoFiTok-internal/docs/INFERENCE.md`；记录：`CoFiTok-internal/docs/records/2026-07-12_stable_generation_session.md`。
- 正式 `scripts/generate_samples.py` 必须通过真实 tiny checkpoint -> integrity verify -> DDIM -> PNG -> manifest/progress/report 的 subprocess 集成测试；sampling manifest 的 `actual_timesteps` 从 `GenerationSession.schedule` 读取，禁止重新依赖已移除的局部 schedule。记录：`CoFiTok-internal/docs/records/2026-07-12_formal_sampling_cli_integration.md`。
- 六个权威 post-training runbook 中所有 `python scripts/*.py` 多行命令必须通过 source-to-source CLI contract test：当前 18 个唯一入口均执行真实 `--help`，runbook 使用的每个 `--option` 必须存在于对应 parser；新增未登记入口、移除/改名参数或 import/startup 失败均阻止部署。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_runbook_cli_contracts.md`。
- promotion/final post-eval 必须生成固定 index 的 CoFiTok endpoint、同 index dense endpoint 和独立 `1/2/4/8` prefix-path 三张 PNG；formal endpoint indices 可覆盖 10K/50K 全范围，64-sample prefix diagnostic 只用独立 `0..15`。visual audit 绑定 checkpoint/sample-set/source/panel SHA 与 builder Git revision/branch/clean state，拒绝固定 endpoint 中的 exact duplicates，但明确不能替代 FID/IS/precision/recall；completion audit 只接受与 full-training revision 完全一致的干净构建证据。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_visual_quality_audit.md`。
- 10K promotion eval、full matched 300K 和 final 50K eval 在创建大产物前都必须运行 storage-capacity preflight，按 checkpoint、计划 PNG、评估缓存和安全余量计算 required bytes；容量不足以 exit `78` 非重试停止。completion audit 必须重算三份报告的预算算术、路径、headroom 与 clean deployed revision。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_storage_capacity_preflight.md`。
- full matched 300K 必须启动通用只读 operational monitor：每 300 秒记录 step/metric freshness/checkpoint cadence/process/disk/GPU，拒绝非有限或非单调 metrics，并在每个 CoFiTok/dense milestone 训练段后同步检查；milestone eval 期间只有 runbook 存活时应为 `waiting`，不能误报 stall。最终 completion audit 只接受 clean full revision 下双 300K exact complete、无 health issue 且四个 protected checkpoint stat 齐全的 monitor `pass`。记录：`CoFiTok-internal/docs/records/2026-07-13_full_generation_operational_monitor.md`。
- upgrade-branch full 300K checkpoint 必须把 runtime environment canonical SHA 同时绑定到 payload extra state、integrity sidecar、`latest.json`、run manifest 和 training report；环境覆盖 Python/platform、关键包、Torch/CUDA/cuDNN/driver、GPU capability、backend/env vars 与 `pyproject.toml`/`uv.lock` SHA。resume 在加载任何 model/optimizer/RNG state 前逐字段拒绝漂移；completion audit 要求 CoFiTok/dense 环境 SHA 完全一致。legacy 10% checkpoint 不回写该字段。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_runtime_environment_fingerprint.md`。
- upgrade-branch checkpoint 的同一 pre-deserialization trust boundary 还必须绑定 Git revision/branch/tracked-dirty 到 payload extra state、integrity sidecar 与 `latest.json`；即使 config 和依赖环境相同，换 revision 也不得静默 resume。completion audit 要求双 full training report 与各自 latest checkpoint pointer 同时指向 clean deployed `scale/generative-system` revision；legacy 10% 不伪造该历史字段。实现提交 `b515d28ffa0ff409bbf23c277b62fd19fa9672d9` 已通过本地与服务器 Linux 各 `438/438` tests、服务器 `40/40` runbook syntax 和 CPU step 1 -> 2 exact-resume 演练；payload/sidecar/`latest.json`/report 的 Git 与 runtime environment SHA 全部一致，正式远端 HEAD 前后均为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。
- terminal completion audit 必须重新读取并哈希双方法实际 step-300K exact-resume checkpoint 和 EMA-only deployment artifact，不能只信旧 JSON 报告。checkpoint 本体/sidecar/`latest.json` 的 bytes、SHA、format、runtime environment 和 Git binding 必须一致；导出文件丢失、截断或单字节替换必须使 `reproducible_full_checkpoint_files` 或 `deployable_ema_inference_artifacts` 失败。实现提交 `f4a224b719c60e3f8f621a78b6c6ee434e3b0998` 已通过本地/服务器各 `441/441` tests、服务器 `40/40` runbook syntax 与真实 tiny checkpoint -> EMA export -> tamper rejection；正式远端 HEAD 前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。
- EMA-only deployment artifact schema v2 必须自带源训练 checkpoint SHA、runtime-environment SHA 和 Git revision/branch/dirty，且这些字段要同时绑定 payload、integrity sidecar、export report、loader/session metadata、preflight 和 inference report；无训练 provenance 的 source checkpoint 不允许导出，payload/sidecar 漂移必须拒绝加载。实现提交 `a560d7aa22eb9b5646e88916682ad6c6cadd7494` 已通过本地/服务器各 `444/444` tests、服务器 `40/40` runbook syntax 和 CPU step 1 -> 2 -> EMA export -> preflight -> PNG inference 全链演练；正式远端 HEAD 前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。
- 正式 sampling manifest schema v3 与 completed report schema v6 必须记录实际采样进程的 canonical runtime environment 与 SHA；`--resume` 环境漂移直接拒绝。shared batch selector、metrics、promotion/final gate 和 completion audit 必须要求 CoFiTok/dense 的 Python/PyTorch/CUDA/driver/GPU/backend/project-lock 环境完全一致，并使用命名 gate `matched_sampling_runtime_environment`。环境绑定最初由提交 `e708f9765a06a14841f081c5fc441ffaff973be3` 引入，当前 schema 由 formal sampling contract 提交升级；记录：`CoFiTok-internal/docs/records/2026-07-13_generation_sampling_runtime_environment.md`。
- 正式 torch-fidelity evaluator 必须以 `cofitok_image_tree_sha256_v1` 对 real ImageNet validation 的 root-relative path、size 和完整 bytes 做内容寻址，并用摘要派生 real-feature cache key；metrics schema v2、命名 gate `matched_real_set_provenance`/`matched_evaluator_runtime_environment`、comparison schema v5 和 completion audit 必须贯通同一 real-set 与 evaluator-environment SHA。实现最初由提交 `98d9a32db165c948a4a34f41f04ce5f1e7bf6238` 引入，当前 comparison schema 由 source-provenance hardening 提交升级；记录：`CoFiTok-internal/docs/records/2026-07-13_generation_real_set_evaluator_provenance.md`。
- full ImageNet-256 300K CoFiTok/dense 配置共同使用 `random_horizontal_flip_prob=0.5`；增强在 normalized device batch 上、noise/timestep 抽样前执行，只消费 checkpoint 已保存和恢复的 PyTorch RNG，validation 不增强。active pinned 10% 配置不改写并按默认 `0.0` 保持原协议；fully resolved config identity 与 pair contract 会拒绝 resume 或双方法概率漂移。实现提交 `7bcc72d89e7d2884487d931a63765859c7cb78f1` 已通过本地/服务器各 `459/459` tests、CPU 0.5 flip uninterrupted-vs-resume 轨迹相等和服务器 `40/40` runbook syntax；增量 bundle 为 `302,682` bytes、SHA256 `ed5bc7845ff4158007916ab62286c2fe2b7cca215d2a5dd05d9f49821d4d01f3`，正式远端 HEAD 前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_exact_resume_horizontal_flip.md`。
- production scalable U-Net 的全部 token/epsilon head 必须零初始化，使 CoFiTok K-token 累加与 dense 单头都从严格 `epsilon=0` 开始，消除方法相关的随机初始输出尺度；restricted synthesis 保持非零导数，首个 backward 必须让 head gradient 非零且一次 optimizer step 后离开零状态。strict checkpoint load 会覆盖初始化，因此 active 10% 权重与评估不受影响。实现提交 `452647a76dc5420f74119f32a9bf6c77807cf99f` 已通过本地/服务器各 `461/461` tests、双方法首步可训练性测试和服务器 `40/40` runbook syntax；增量 bundle 为 `316,708` bytes、SHA256 `cb357ba5cec708bde2ac992a636f727a3dd5b4fcc8db648a955b6b6e45270ac7`，正式远端 HEAD 前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_zero_initialized_output_heads.md`。
- 正式采样统一使用 `cofitok_ddim_sampling_v1`：scaling gate 锁定 EMA DDIM-100，full gate 锁定 EMA DDIM-250；两者都要求 CFG 1.5、guidance rescale 0、batched CFG、`eta=0`、`clip_x0=true`、bf16、seed 0、balanced-modulo class schedule 和 batch/resume-invariant per-index random stream。metrics 必须逐字段绑定 immutable manifest，gate 必须使用 `formal_sampling_protocol`，completion audit 与 comparison schema v5 必须独立重算，双方法同时弱化也不得通过。实现提交 `0cd29bc24d53b7460c12da5a4bb4b556ff171945` 已通过本地/服务器各 `472/472` tests 和服务器 `40/40` runbook syntax；pinned `781a014` 到该 HEAD 的增量 bundle 为 `336,661` bytes、SHA256 `9e7048cf7daf53e06d8a5ed76537bb50b621b76f503296bb534970d0da1736cc`，正式远端 HEAD 前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。记录：`CoFiTok-internal/docs/records/2026-07-13_formal_sampling_protocol_contract.md`。
- full 300K 的 checkpoint-free 真实训练 runtime selector 必须在每份 benchmark report 中记录 canonical Python/PyTorch/CUDA/driver/GPU/backend/project-lock environment 与 SHA；缓存复用要求 clean Git/config/horizon/environment 全部一致，所有成功完成的 CoFiTok/dense/candidate benchmark 及后续双 300K training report 必须共享同一 SHA。completion audit 会重算全部 fingerprint 并拒绝 benchmark 或训练环境漂移。实现提交 `35a5b130c033c14f19d89ac384444ab9c0e27e5b` 已通过本地/服务器各 `475/475` tests 和服务器 `40/40` runbook syntax；pinned `781a014` 到该 HEAD 的增量 bundle 为 `334,923` bytes、SHA256 `f20bdbce04d3a39bc3a1c05d3606cff7eef163f530a5c7c4f6097af2edf51ece`，正式远端 HEAD 前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_training_runtime_selection_environment.md`。
- DDIM 递推新增独立闭式 oracle 回归：固定已知 `x0` 构造精确 epsilon predictor，经 cosine 跳步轨迹后，`eta=0` 与 `eta=0.5` 均必须恢复同一目标图像。实现提交 `4d9a1caf0bbd48a24d0495d4caceda18942ee2d6` 的 targeted sampling tests 在本地/服务器均为 `12 passed`、本地全量为 `477 passed`；pinned `781a014` 到该 HEAD 的增量 bundle 为 `337,138` bytes、SHA256 `d9b23e80086711f97d5b9f6154561a0323dc7052efed1dce904f73eb49f6be37`，正式远端 HEAD 仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean。这只证明 sampler 数值递推，不替代 10K promotion、full 300K 或最终 50K 指标。记录：`CoFiTok-internal/docs/records/2026-07-13_ddim_closed_form_correctness.md`。
- full 50K/100K/200K/300K milestone 聚合报告升级为 schema v2：四份 CoFiTok/dense generation-metrics 与 checkpoint-eval 源报告必须位于方法/step 权威路径，并绑定 bytes/SHA256；resume 跳过和 final completion audit 都会重新哈希并独立重算 EMA DDIM-50/CFG 1.5/bf16/fixed-stream 协议、matched delta 与 alert。实现提交 `080bc3ac988fa70fb01e654e35c9b1ec8369b91b` 已在本地和服务器真实 sibling-`paper/` 布局各通过 `481/481` tests，服务器 `40/40` runbook syntax 通过；pinned `781a014` 到该 HEAD 的增量 bundle 为 `347,170` bytes、SHA256 `fa610a0600c632452d5fddb8938608cd5ccc3a71def71edbbbfa15fbdda9110b`，正式远端仍固定在 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_milestone_source_provenance.md`。
- 40K legacy checkpoint 已在不加载模型、不使用 GPU、不改写文件的条件下完成只读审计：step `40,600/50,000`、`status=healthy`、`issues=[]`，retained recovery points 为 30K/35K/40K，40K SHA256 为 `f3af056d3a6945567fcef24c951efbd7e168ec97ec7611a64a4feb33dc16121f`，按 `2.39359s/step` 估计 CoFiTok 半程剩余约 6.25 小时。快照提交 `654bd6d21de86373bc6269abf1b8e55c989fdf20`；当前 pinned `781a014` 到该可部署 HEAD 的 bundle 为 `337,840` bytes、SHA256 `9b92c819f76769d71384696f20c1c1fc54d342744deda61dda14e0d4a6bc0878`。10% legacy 队列只滚动保留最近 3 个恢复点，不得把正常删除的早期 5K-25K 误报为缺失；full run 才永久保护四个正式 milestone。快照：`CoFiTok-internal/artifacts/reports/generation/training_progress_2026-07-12/cofitok_step_040600.json`。
- final comparison source-provenance hardening 提交为 `f4ff6790d12660684aaf55cd801cb86e52242e9e`：comparison schema v5 绑定五份 matched 源报告的 path/bytes/SHA256，并由 completion audit 回读文件、重算 matched FID/IS/precision/recall、evaluator、sampling invocation、成本和派生摘要。提交已在本地与服务器隔离 sibling-`paper/` 布局各通过 `485/485` tests，服务器 `40/40` runbook syntax 通过；pinned `781a014` 到该 HEAD 的 bundle 为 `352,268` bytes、SHA256 `35d14a550734b24e93b5938b91e48fb7f3880931f5615f13a10ca0538aefb495`。正式远端 HEAD 验证前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，双 50K 完成前不得部署该 bundle。
- 正式训练配方契约 `cofitok_generation_training_recipe_v1` 已在提交 `8c72bf00f3d6d819ff004019f12ba509d4da4c13` 落地：在 pair fairness 之外锁定 scaling/full 的数据阶段、256 分辨率、cosine epsilon diffusion、U-Net 容量、class-dropout CFG、bf16/TF32、AdamW/EMA、checkpoint cadence、CoFiTok K8 restricted synthesis 与 denoise-path objective；runtime 只允许实测候选 `16x4/32x2/64x1` 且 effective batch 64。检查同时接入 config preflight、10% predeployment validator、full 300K completed-pair validator 与 terminal completion audit，双方法被同时弱化也会 fail closed。提交已在本地与服务器隔离 sibling-`paper/` 布局各通过 `493/493` tests，服务器 `40/40` runbook syntax 通过；pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `351,154` bytes、SHA256 `a753c35c97eb4e5b3f4a1d54e3b4551cf0961466d66669906db1b4ba0a6ee53c`。正式远端 HEAD 验证前后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，双 50K 完成前不得部署该 bundle。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_training_recipe_contract.md`。
- 正式数据 provenance 已在提交 `ca385aa727b67faae6f26c99643340c46f430460` 落地：训练启动实际重哈希权威 ImageNet manifest 并校验 bytes/train/val 数量，将 canonical dataset identity 绑定到 run manifest、runtime benchmark、checkpoint payload、integrity sidecar、`latest.json` 与 training report；full exact-resume 会在 `torch.load` 前拒绝数据身份漂移，runtime selection schema v2 和 completed-pair schema v2 要求所有候选及双方法共享同一 identity。active pinned 10% 双报告只允许通过显式 legacy flag 同时缺失，混合 legacy/bound pair 仍拒绝。服务器真实捕获的 10% identity 为 `97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741`，full identity 为 `6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659`；本地与服务器隔离 sibling-`paper/` 布局各通过 `501/501` tests，服务器 `40/40` runbook syntax 通过。pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `374,992` bytes、SHA256 `bdbec33c2ce5fad4be291755ed698503762989e0f6ff3e503679dcba30ede6f9`；最终 sanity 为 `pass`，正式远端仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean。双 50K 完成前不得部署该 bundle。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_dataset_provenance.md`。
- promotion/final gate 的共享 authorization contract 已在提交 `a9f9ba96e4061dc306c0a007d4191ba1bd528839` 落地：300K runbook 不再只信 JSON 的 `status/decision`，而是由 `cofitok.generation_gate` 与 `scripts/validate_generation_gate_report.py` 统一复核 schema、唯一必需 gate、阈值方向、FID/endpoint/ordered/zero/shuffle 证据；scaling 固定至少 10K EMA DDIM-100 样本、绝对 FID `<=100`、相对 FID/endpoint 回退 `<=5%`，full 继续固定 50K DDIM-250、FID `<=20`、precision/recall `>=0.30`。completion audit 共用同一 contract，删 gate、弱化阈值或只把状态改成 pass 均失败。本地全量 `510/510` 通过；服务器 isolated 受影响组 `97/97` 与 `40/40` runbook syntax 通过。pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `363,166` bytes、SHA256 `625723e3201d710b381653235c6d21586fd209ef2938d0c2b42b3c1c4a9b39bc`，正式远端验证后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean。双 50K 完成前不得部署。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_gate_authorization_contract.md`。
- scaling gate 到 full 300K 权重的训练授权绑定已在提交 `a42544c1ce9e088ade43beaa4f880ee6245dfbe3` 落地：正式 `imagenet_256` 300K 训练必须显式消费通过共享 contract 验证的 `--authorization-gate`，canonical gate identity、原文件 bytes/SHA、stage 和 decision 同时进入 run manifest、checkpoint payload、integrity sidecar、`latest.json` 与 training report；resume 在反序列化前拒绝 gate 漂移，正式 sampling/EMA export 也会校验 payload-sidecar 绑定。full pair validator schema v3 和 terminal completion audit 要求 CoFiTok/dense 及两份真实 step-300K sidecar 绑定同一实际 scaling gate。提交已通过本地全量 `516/516` tests、服务器隔离受影响组 `106/106` tests 与 `40/40` runbook syntax；pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `390,299` bytes、SHA256 `b9df9e458629c54538a750b7c28776a2cf857fbc7abb82ce016898e4a137a4cb`。正式远端验证后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，双 10% 50K 报告完成前不得部署。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_training_authorization_binding.md`。
- EMA-only deployment artifact 的训练授权 provenance 已在提交 `631be047f7d7d1525e7c5567260fd8ef9006b2e6` 落地：artifact payload/integrity schema v3 将完整 scaling `training_authorization` 与 source checkpoint SHA、runtime environment、Git 一起绑定，export reuse、loader/session、real-forward preflight、inference report 和 terminal completion audit 均要求一致；正式 artifact 即使脱离训练目录也能说明由哪份 promotion gate 授权，CoFiTok/dense 不能混用或事后替换 gate。无授权 tiny/legacy artifact 仍以一致的 `null` 兼容。提交已通过本地全量 `520/520` tests、服务器隔离受影响组 `89/89` tests 与 `40/40` runbook syntax；pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `385,495` bytes、SHA256 `57ed0d743929cecac4c95d5c90522c870a1c3620d199db5e8ae71b7c5ee1c349`。正式远端验证后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，双 10% 50K 报告完成前不得部署。记录：`CoFiTok-internal/docs/records/2026-07-13_inference_artifact_authorization_provenance.md`。
- deployable EMA artifact 的最终质量发布授权已在提交 `f453f5857a6d3a31e955d0f80fe0202d14ac1d84` 落地：artifact payload/integrity schema v4 在 scaling training authorization 之外，强制正式 export 显式消费通过共享 full contract 验证的 `--release-gate`；canonical final-gate identity、原文件 bytes/SHA、full stage、`large_scale_generation_ready` decision 和阈值进入 artifact、sidecar、export report、loader/session、preflight、smoke 与 terminal audit。final gate 失败、弱化、替换或 CoFiTok/dense 使用不同 gate file 时，deployment evidence 同步失败。提交已通过本地全量 `527/527` tests、服务器隔离受影响组 `117/117` tests 与 `40/40` runbook syntax；pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `391,294` bytes、SHA256 `970bb992a5019d6e0839effb78a4f665258e73f9235c9d8eca53a505b65675d9`。正式远端验证后仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，双 10% 50K 报告完成前不得部署。记录：`CoFiTok-internal/docs/records/2026-07-13_inference_artifact_final_release_authorization.md`。
- 训练后部署的 untracked 冲突检查已在提交 `b2b8e8feb67b16126e4e82cd0e88e45576039e38` 改为有界扫描：只枚举目标 revision 相对 pinned revision 的新增路径，以 256 路径块查询 untracked，并单独拒绝父路径文件/symlink 阻塞。真实正式远端 rehearsal 对 `145` 个 target-added paths 耗时 `0.047s`、`conflict_count=0`；本地全量 `530/530` tests、服务器隔离相关组 `29/29` 与 `40/40` runbook syntax 通过。运行证据提交后当前可部署 HEAD 为 `cccc5b43a43760fd29c3cded635401fb1b94cbe3`；pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `398,804` bytes、SHA256 `37518378451d09108242822d0e078f49b3da95a53685f863b35a07f55383a41f`，最终 HEAD 复检耗时 `0.055s`、冲突仍为 `0`。bundle rehearsal 后正式 HEAD 仍为 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean。记录：`CoFiTok-internal/docs/records/2026-07-13_bounded_deployment_conflict_scan.md`。
- 部署 transition evidence 已在提交 `91095b907862b5fe72b7dff2196e2b76b046f3cd` 升级为 receipt schema v2：verified incremental bundle 永久归档到 `checkpoints/generation/deployment/`，pre-merge conflict JSON、完整 pytest JUnit 和由 `git ls-files` 枚举全部 tracked shell 的 syntax JSON 均绑定 path/bytes/SHA；terminal completion audit 会重新读取、哈希并语义验证四份源证据。提交 `1767a5ecbf404d1573fdbd16f758c8ff1303a9e5` 又要求 bundle header 只有一个 prerequisite 且必须等于 pinned `781a014`，pre-fetch helper、receipt 和 terminal audit 三处独立验证；故意使用正确 target head、错误 prerequisite 的服务器负向 rehearsal 以 exit `78` 拒绝。当前实现已通过本地及真实 sibling-`paper/` Linux 各 `542/542` tests，Linux runbook auditor 为 `40 discovered / 40 checked / 0 failed`，正式远端对 `148` 个 target-added paths 的冲突复检为 `0`、耗时 `0.054s`。运行证据提交后当前可部署 HEAD 为 `26925c027e189e5ca892af5f55bf90656cfb93b4`；pinned `781a014` 到该 HEAD 的最终单一增量 bundle 为 `404,049` bytes、SHA256 `da3bd97a181efaff56d3b40c4b0d8901af715413aba02763f0ea3b9fdc5f2f9a`。正式远端仍为 pinned `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_deployment_evidence_provenance.md`。
- full 300K runtime selection 已在实现提交 `3bd4da72d3e58a9787a88db3b29245d6cc7dd615`、证据提交 `b518986984cd1d2a02f3b2ae5c0cba2bc30821af` 冻结：任一正式 run 目录出现训练状态后，selector 只能验证并只读复用原 `runtime_selection.json`，不得重跑 benchmark 或改写选择；lock 绑定双 run 路径、`16x4/32x2/64x1`、effective batch 64、8/2 benchmark horizon、300K horizon、config SHA、clean branch/revision、dataset 与 runtime environment，completion audit 会从最终双 training report 反推并核验每份 candidate config。实现已通过本地与真实 sibling-`paper/` Linux 各 `549/549` tests、服务器 `40/40` runbook syntax；最终 target-added paths `149`、冲突 `0`。当前可部署 HEAD 为 `b518986984cd1d2a02f3b2ae5c0cba2bc30821af`；pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `442,947` bytes、SHA256 `885f8e63510ee63fea1d60104f56dd540ad43ef00062c8ea1d9d29fba3194ce7`。正式远端仍固定在 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，双 10% 50K 完成前不得部署。记录：`CoFiTok-internal/docs/records/2026-07-13_generation_runtime_selection_freeze.md`。
- 正式 10K/50K sampling batch selection 已在实现提交 `459250b0aed0137353263bac0eef4734bf0fe036`、证据提交 `77be40528d7c597f798ca07743ed32727a3e12c6` 冻结：任一 matched 正式 sample output 出现 manifest/progress/PNG 等状态后，selector 只能验证并只读复用原选择，不得重跑 preflight 或改写 batch；lock 绑定双 output path、`16/32/64/128`、baseline 32、90% 显存上限、双 checkpoint path/SHA/step/integrity sidecar、prefix 8/1、EMA/bf16/CFG 1.5/batched、2/5 warmup/measurement、clean Git 与 benchmark root。所有成功 candidate pair 还必须共享同一 runtime environment；completion audit 重算完整 selection，并检查非 selected candidate。实现已通过本地与真实 sibling-`paper/` Linux 各 `558/558` tests、服务器 `40/40` runbook syntax；最终 target-added paths `150`、冲突 `0`。当前可部署 HEAD 为 `77be40528d7c597f798ca07743ed32727a3e12c6`；pinned `781a014` 到该 HEAD 的单一增量 bundle 为 `428,586` bytes、SHA256 `f3a190218e2c48633f3fec8ef9ab78d45bca6f98d21adecb2e310717c9a7808c`。正式远端仍固定在 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，双 10% 50K 完成前不得部署。记录：`CoFiTok-internal/docs/records/2026-07-14_generation_sampling_batch_selection_freeze.md`。
- promotion/final generation gate 源报告绑定已在实现提交 `e34e523223e9e04399b6032638e9b6ad0356dfc3`、证据提交 `ad5010b4abbf966ca72311ef189fdfd467f0aed9` 落地：每份正式 gate 必须绑定 CoFiTok/dense 的 training、distribution metrics 和 checkpoint-mechanism eval 共六份权威报告的绝对路径、bytes 与 SHA256；transition 在跳过 post-eval 前重哈希源文件，source-valid quality `hold` 不会被误重试，terminal completion audit 也会重新验证。实现已通过本地与真实 sibling-`paper/` Linux 各 `563/563` tests、服务器 `40/40` runbook syntax；最终 target-added paths `152`、冲突 `0`。当前可部署 HEAD 为 `ad5010b4abbf966ca72311ef189fdfd467f0aed9`；pinned `781a014` 到该 HEAD 的最终单一增量 bundle 为 `431,910` bytes、SHA256 `c302d4753c8bdf9c7b0c49ef3d7ed65064530314a0443779c956a99c8a76dbf8`。正式远端仍固定在 `781a01444fddbf0d48a427ba58bdeed50167b5be` 且 tracked clean，dense 10% 50K 完成前不得部署。记录：`CoFiTok-internal/docs/records/2026-07-14_generation_gate_source_provenance.md`。
- generation authorization 源新鲜度已在实现提交 `6b4a1a8db70a545bece254433af50acfb945d40e`、证据提交 `acb5f4d6d0979863c9253a2940b9cdd3555b82cc` 收口：gate source verifier 已下沉为 `cofitok.generation_gate_sources`，每次 full 300K start/resume 和 final EMA release authorization 都会在 checkpoint 反序列化前重哈希六份源报告；formal full 50K post-eval 还会在任何 GPU 评估/采样前重新验证 scaling gate 与完整 matched training pair。该阶段已通过本地与真实 sibling-`paper/` Linux 各 `565/565` tests、服务器 `40/40` runbook syntax；target-added paths `154`、冲突 `0`。后续 true-compressed 与 completion-audit 提交已继续推进可部署 HEAD，勿再使用该阶段旧 bundle。记录：`CoFiTok-internal/docs/records/2026-07-14_generation_authorization_source_freshness.md`。
- 第一代 true-compressed generation layout 已在核心提交 `35d21c3c153ea655778a8e490e7410619aa8f1d4`、native-sidecar post-eval 修正 `9cfecade12c9e2bb3786d3ae842f203b56ce993e`、证据提交 `f96eb1270e46324d1972dec6719513967059227b`、waiter/supervisor 恢复性修正 `f9f1d47b8f7ecb8db8b67dda563d4a33bac198f7`/`a064e5432adb08819038402ba37cf3e164d89a06`、部署失败状态与重复 PID 防误判修正 `9d75babd0599fa88de6dc5d0dd056e93d975d2db`/`9036fdb6b9471da30e32ce497f16f3acb86b5a26` 落地：历史 K8 使用 spatial strides `[16,16,8,8,4,4,2,1]` 与真实通道 `[4,4,8,8,8,8,4,2]`。该布局后来被证明在最高频 RGB 子空间 rank-deficient，现只保留为失败证据；当前正式 rank-complete 布局已替换为 strides `[16,16,8,8,4,4,1,1]` 与通道 `[4,4,8,8,8,8,1,2]`，最后两个 full-resolution token 合计三通道。每个 token 的 scalar capacity 仍严格小于 `196,608`-scalar dense RGB epsilon field，但 K8 合计为 `280,576` scalars，即 dense 的 `1.427083x`；因此只能主张 per-token compression，不能主张 aggregate-token compression。旧 `781a014` pair 被 recipe 明确标为 `legacy_scaling`；部署后新 compressed pair 才是 `scaling`，其 post-eval 只读要求 native integrity sidecar，不执行 legacy migration。终局 smoke evidence 在 `d007fd8ce553bd814b3dd0cc7d8aa997b78ca706` 升级为真实文件验证；提交 `c9e9221bc4c8c0024bd7a5783fd4d07fd41389a1` 要求 completion audit 对正式双方法 50K PNG 集合逐文件检查编号/类型/非 symlink，并从物理文件重算 sample-set SHA，同时重验 sampling report/manifest/progress 与 checkpoint、Git、运行协议绑定；提交 `bdf7bd03aee843fba65a826e587990ec29859a3c` 进一步固定并物理重哈希 ImageNet-256 validation 50K tree，拒绝 symlink/非图片文件和哈希期间成员变化，并要求双方 formal metrics 的 real-set digest 精确匹配。真实远端验证得到 `cofitok_image_tree_sha256_v1:19ace4e37bee2fcaeac2b7cbd26ed785aa0e6e01015c90b4955027d163f6e44f`。本地和隔离 Linux 各 `582/582` tests、服务器 `41/41` runbook syntax 通过；target-added paths `164`、冲突 `0`。当前唯一可部署 HEAD 为 `bdf7bd03aee843fba65a826e587990ec29859a3c`；pinned `781a014` 到该 HEAD 的最终 bundle 为 `485,869` bytes、SHA256 `1943f5d75d509bbe9e0ebde9de087f50f49728909a112300105847235bcc62c2`。正式远端仍固定在 pinned revision 且 tracked clean。记录：`CoFiTok-internal/docs/records/2026-07-14_generation_true_compressed_token_layout.md`、`CoFiTok-internal/docs/records/2026-07-19_generation_rank_deficit_recovery.md`、`CoFiTok-internal/docs/records/2026-07-12_generation_completion_supervisor.md`、`CoFiTok-internal/docs/records/2026-07-12_generation_posttraining_transition_pipeline.md`、`CoFiTok-internal/docs/records/2026-07-12_generation_sampling_progress.md`、`CoFiTok-internal/docs/records/2026-07-12_large_scale_generation_completion_audit.md` 与 `CoFiTok-internal/docs/records/2026-07-12_deployable_ema_inference_artifact.md`。
- 2026-07-14 03:06 CST 的权威只读快照：legacy CoFiTok 10% run 已完成 `50,000/50,000`；legacy dense 10% run 为 `3,600/50,000`，latest total loss `0.0292025`、samples seen `230,400`。monitor 为 `running/dense_identity_training`、`issues=[]`，GPU utilization `100%`。一次性部署 waiter PID `221762` 为 `waiting`，目标 `bdf7bd0`；它每 300 秒轮询，48 小时超时，只有双 50K 精确完成且 legacy 进程全部退出后才调用 fail-closed deployer。当前仍不是“全面完成”；后续必须完成 fresh compressed 10% matched pair 与 promotion gate、full ImageNet-256 matched 300K、正式双方法 50K 评估、final quality gate、EMA export 和 terminal completion audit。
- 多周 completion pipeline 必须由 bounded supervisor 启动：只对 `posteval_10pct/full_training/full_posteval/inference_export` 最多做 4 次指数退避恢复，绝不自动重试 preconditions、promotion/final gate、completion audit 或 unknown stage。已有 scaling gate 直接验证，final post-eval 仅在 gate/comparison/visual 三者齐全时跳过；full paired milestone 只有报告、双 protected checkpoint 和 sidecar 同时存在才跳过，已越过 milestone 但权重缺失须立即失败。记录：`CoFiTok-internal/docs/records/2026-07-12_generation_completion_supervisor.md`。
- final gate 通过后必须为 CoFiTok/dense 各导出独立 EMA-only inference artifact；它们使用 `cofitok_generation_inference@1` 类型 sidecar，在反序列化前验证 filename/bytes/SHA/format/step/source-checkpoint SHA，只允许 `weights=ema`。原始 300K training checkpoint 继续作为 exact-resume 权威，不得被导出物替换。completion audit 要求 artifact 小于源 checkpoint、双 preflight 通过并完成 CoFiTok 4 张/dense 2 张短 DDIM smoke PNG。记录：`CoFiTok-internal/docs/records/2026-07-12_deployable_ema_inference_artifact.md`。
- 2026-07-14 11:05 CST 更新：本条覆盖上文 `bdf7bd0` “当前唯一可部署 HEAD”和 03:06 快照。monitor-aware training watchdog 已在最终提交 `fd336a617f03804bf08f5c968ea2a8d6befb6dbc` 收口；它把 read-only pair monitor 变成 fail-closed 训练进程所有者，分别以 exit `86/87/88/89` 表示 monitor terminal failure、无 fresh startup report、report silence 和 monitor process disappearance，并在 Linux 上终止完整训练 process group。Windows 存活探测使用 `OpenProcess/GetExitCodeProcess`，禁止 `os.kill(pid, 0)` 误发 `CTRL_C_EVENT`。记录：`CoFiTok-internal/docs/records/2026-07-14_generation_training_watchdog.md`。
- 最终 prerequisite-aware bundle 为 `/tmp/cofitok-generation-fd336a6.bundle`：只 advertised `fd336a617f03804bf08f5c968ea2a8d6befb6dbc`，唯一 prerequisite 为 pinned `781a01444fddbf0d48a427ba58bdeed50167b5be`，大小 `495,661` bytes，SHA256 `207c271e5d8be691edaceac9e7fc93c16b6337896f2179b744c65050156b1f46`。精确 target 已在服务器隔离 checkout 禁用 GPU 后通过 `590/590` tests（包含 POSIX descendant process-group cleanup）、`41/41` tracked runbook `bash -n`，tracked status clean；正式仓库 rehearsal 前后均为 pinned `781a014`。
- 2026-07-14 11:05 CST 权威运行快照：legacy CoFiTok 为 `50,000/50,000`，legacy dense 为 `17,700/50,000`；monitor 为 `running/dense_identity_training`、`issues=[]`。部署 waiter 已安全替换为 PID `474390`，状态 `waiting`，目标 `fd336a6`，每 300 秒轮询、48 小时超时；旧 waiter PID `221762` 已退出，legacy runbook/training PID 未被触碰。只有双 legacy 50K 报告精确完成、进程全部退出、pair validator 和 untracked-conflict gate 通过后才允许部署并启动 true-compressed completion pipeline。当前仍不是“大规模生成全面完成”。
- 2026-07-15 10:19 CST 更新：本条覆盖上文仍等待 legacy dense 与 `fd336a6` 为最终目标的旧快照。legacy CoFiTok/dense 两份 10% 报告均已严格完成 `50,000/50,000`，`generation_10pct_pair_monitor.json` 为 `pass/complete`。一次性 waiter 随后因 helper 请求 bundle 中不存在的 synthetic `HEAD` ref 而以 Git exit `128` 失败；正式 HEAD 当时仍为 pinned `781a014`，未生成 receipt、未启动后续训练，旧 checkpoint 和报告均未改写。修复与真实 Git 回归测试见 `CoFiTok-internal/docs/records/2026-07-15_generation_bundle_fetch_fix.md`。
- 当前正式生成 target 为 `04a738c637555b2c108e5c468d7931ebdbb59cb9`（`scale/generative-system`）。其 prerequisite-aware bundle 已归档为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/cofitok-generation-upgrade-04a738c637555b2c108e5c468d7931ebdbb59cb9.bundle`，大小 `494,744` bytes，SHA256 `c33429ecaf665f311294d1bb18e32a7e6e4016aa3a27c74763388441909a52a9`，唯一 prerequisite 为 `781a01444fddbf0d48a427ba58bdeed50167b5be`。正式部署 conflict scan 为 `0/169`，远端 receipt 绑定的 pytest 为 `593 passed`、runbook syntax 为 `41/41`，远端 tracked status clean；权威 receipt 是 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_upgrade_deployment_receipt.json`。
- true-compressed completion pipeline 已从 supervisor PID `878331` 启动。权威 supervisor/pipeline 状态分别读取 `generation_completion_supervisor.status.json` 和 `generation_complete_pipeline_after_10pct.status.json`；当前 pipeline 为 `running/scaling_training`，先训练 `/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_compressed_cofitok_k8_50k`，由 monitor `generation_10pct_compressed_pair_monitor.json` 与 `training_watchdog.json` fail-closed 保护，随后才训练 matched dense。后续仍必须依次通过 compressed 10% post-eval/promotion gate、full ImageNet-256 matched 300K、正式双方法 50K 评估、final gate、EMA export 与 terminal completion audit；当前仍不是“大规模生成全面完成”。

## Rank-complete 生成恢复状态（2026-07-20）

- revision `04a738c637555b2c108e5c468d7931ebdbb59cb9` 的旧 true-compressed 10% matched pair 已完成双方法 50K，但正式 promotion gate 为 `fail/hold`：CoFiTok FID `422.5906`、IS `1.3042`、endpoint MSE `0.169522`、ordered rank `3/18`；dense FID `114.8773`、IS `9.2107`、endpoint MSE `0.016330`。协议、公平性、来源和 checkpoint integrity 均通过，失败属于模型质量，不得重命名或覆盖该结果。
- 根因是旧 token layout `[channels 4,4,8,8,8,8,4,2] / [strides 16,16,8,8,4,4,2,1]` 只有 2 个 full-resolution channel，无法在受限线性 `S_k` 下满秩表达 RGB 高频。恢复 layout 为 `[4,4,8,8,8,8,1,2] / [16,16,8,8,4,4,1,1]`，保持逐 token 与总 scalar budget 完全不变，同时把 full-resolution channel 补到 3。
- v2 两个 5K objective probe 已在 revision `05bbb4af63a9f1d9b7f11bc4222d50875e382e1d` 严格完成。denoise-path 为 endpoint MSE `0.0199395`、ordered rank `6/18`、FID-512 `275.1347`；epsilon-band 为 endpoint MSE `0.0200607`、ordered rank `18/18`、FID-512 `281.8623`。rank-complete layout 已把旧 endpoint `0.1695224` 降低约 88%，但两者均未恢复 rank 1，因此不得启动正式 50K。权威 aggregate：`CoFiTok-internal/artifacts/reports/generation/rank_recovery_probe_2026-07-20_v2/rank_recovery_probe.json`；记录：`CoFiTok-internal/docs/records/2026-07-20_generation_rank_recovery_probe_v2_result.md`。
- v2 的 component energy 均坍缩到最后两 token：denoise-path 为 `0.3440/0.6727`，epsilon-band 为 `0.3403/0.6823`；prefix `1/2/4` 确定性 DDIM 样本基本仍是噪声，prefix 8 才出现粗糙语义，属于 endpoint-heavy decoding。raw model 虽把 endpoint 改善到 `0.0186604/0.0186982`，ordered rank 仍为 `7/18` 与 `18/18`，故 EMA lag 不是顺序失败原因。修复后的 `ema_lag_audit.json` 已生成并同步回本地。
- 远端代码已通过受控 probe-only fast-forward 更新到 `a37b2e410aa558e831688009b13012a6a7da616b`；target-specific evidence 位于 `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/probe-a37b2e410aa558e831688009b13012a6a7da616b/`。增量 bundle 为 `34,063` bytes / SHA256 `5701d8450a903736432ce077dc117a0df35ecd888ae3add71aa95540cac85651`，唯一 prerequisite 为 `05bbb4a`；远端隔离 CUDA pytest 为 `621 passed / 0 skipped`，tracked runbook `45/45` syntax 通过，untracked target-path conflict 为 0。该 receipt 只证明 probe hop，不替代 objective 选定后的正式 target-SHA deployment attestation。
- equal-progress + balanced-energy v3 probe 已启动：runbook PID `700981`，权威 monitor 为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_probe_v3_monitor.json`，日志/PID 为 `generation_rank_recovery_probe_v3_2026-07-20.{log,pid}`，输出目录为 `imagenet256_10pct_rankcomplete_equal_progress_k8_probe5k_v3`。2026-07-20 09:47 CST 已到 step `100/5000`，GPU `100%`、显存约 `19.9 GiB`、revision/tracked state 正确、watchdog/monitor 均为 `running` 且 `issues=[]`；energy-budget loss 已从初始化 `0.015625` 降到约 `1.9e-4`。运行期间禁止移动远端 HEAD 或启动其他 GPU 任务。
- v3 保持同一 rank-complete compressed layout，把 denoise progress 改为每 token 等量（power `1.0`），使用已在低分辨率验证的 path 权重 `prefix=0.15/component=0.3`，并以 weight `0.5` 约束八个 component 等能量。配置：`configs/generation/imagenet256_10pct_rankcomplete_equal_progress_k8_probe5k.json`。它不直接训练 shuffled-order 判别，因此 shuffle 仍是独立诊断。
- 本地与远端 tracked `scale/generative-system` HEAD 均为 `a37b2e4`；本地另有未提交的 v2 小型 JSON/PNG 镜像，故只能称 tracked clean，不能称整个工作树无 untracked。fresh 50K 与 full 300K 均从 `16x4/32x2/64x1` 中选择 CoFiTok/dense 共同最快且峰值显存不超过 90% 的组合，effective batch 固定为 64；完成校验必须显式绑定所选 micro-batch/accumulation，记录见 `CoFiTok-internal/docs/records/2026-07-20_generation_selected_runtime_completion_binding.md`。下一次正式 10% pair 必须使用新目录 `imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2`、`imagenet256_10pct_rankcomplete_dense_50k_v2` 与报告目录 `imagenet256_10pct_rankcomplete_matched_50k_v2`，旧 `compressed_*` 目录仅作 immutable failure evidence。
- future v2 10% matched 50K 的 CoFiTok/dense 现统一使用 `random_horizontal_flip_prob=0.5`，与 full 300K recipe 一致；旧失败 pair 和已固定运行的 5K objective probes 仍为 `0.0`，不得静默混称。新 scaling config validator 已通过，双方法参数差仍为 `+0.019242%`，记录：`CoFiTok-internal/docs/records/2026-07-20_generation_scaling_augmentation.md`。
- 5K probe 的生产 EMA 在 step 5,000 时加权平均来源约为 step `2367`、median step `1961`、最近 1,000 update 权重约 `9.52%`；objective 选择必须继续同时比较同一 checkpoint/config/Git 下的 raw model 与 EMA mechanism eval，但正式采样和 gate 仍只允许 EMA。v3 runbook 会在训练后串行完成两种权重的 512-image mechanism eval、EMA DDIM-50 prefix 采样、方向性 FID/IS、三候选 aggregate 与 EMA-lag audit；它永远不得自动启动 50K 或 300K。
- 两个恢复 bundle 已按原字节持久化到 `checkpoints/generation/deployment/`：`05572b1` bundle 为 `11,520,065` bytes / SHA256 `b457808f78fcea5869bbae7742dfc43beaee3d6892fa3ee1506130d4900d07bf`；`05bbb4a` hotfix bundle 为 `8,496` bytes / SHA256 `8b1341da07c995aa1a20d42d561f84e5280967c28e7d15e1c1745857f392aee0`。对应旧 receipt 不得覆盖。
- v3 完成后必须同步 aggregate、checkpoint eval、训练报告和样本图回本地并人工选择 objective；硬要求仍为 ordered rank `1/18`、endpoint 不退化、component 使用合理、zero/shuffle/integrity/provenance 全通过，512-image FID/IS 仅作方向诊断。随后才可把选定 objective 原子写入 scaling/full config 与 recipe，运行 `generation_attest_deployed_revision.sh` 写入按 target SHA 命名的综合证明，再启动 fresh matched 50K。只有正式 10K EMA DDIM-100 promotion gate 全通过，才允许 full ImageNet-256 matched 300K；最终还需双方法 50K DDIM-250、quality gate、EMA export 和 terminal completion audit 才能称“全面完成”。

## Target-energy 生成恢复更新（2026-07-20）

本节覆盖上文“v3 正在运行”和远端 HEAD 为 `a37b2e4` 的旧快照。

- equal-progress + batch-balanced-energy v3 已严格完成 5K、EMA/raw mechanism eval、512 组四-prefix DDIM-50 采样、FID/IS、aggregate 和 monitor。EMA endpoint MSE `0.0224811`、ordered path AUC `0.2858201`、rank `17/18`、FID-512 `375.8400`、IS-512 `1.4768`；raw endpoint `0.0203134`、rank 同为 `17/18`。聚合结论为 `revise_architecture_or_objective`，自动 50K/300K launch 为 false。记录：`CoFiTok-internal/docs/records/2026-07-20_generation_equal_progress_probe_v3_result.md`。
- v3 固定 `t=500` 时最后两 token 仍占 `94.6%` 能量；训练日志中的 batch energy MSE 较低不能证明逐样本或逐 timestep 平衡。根因是旧 energy loss 先跨混合 timestep batch 聚合，再归一化 token 能量，模型可通过不同样本/噪声级使用不同 token 绕过约束。
- evaluator revision `f1ffbfb298b9abb1a3a4c8b524c976610908d2bd` 已对同一 v3 checkpoint 在 `t=50/250/500/750/950` 各评 256 张。理论路径 target 的 uniform MSE 为 `0.006899/0.002242/0.000275/0.000012/~0`，模型却为 `0.012046/0.038651/0.045466/0.046882/0.047120`；ordered rank 为 `1/14/17/18/18`，learned tail-2 energy 为 `49.3%/88.7%/94.6%/95.9%/96.0%`。记录：`CoFiTok-internal/docs/records/2026-07-20_generation_timestep_energy_diagnostic.md`。
- 下一候选不是固定均匀先验，而是逐样本匹配 exact denoise-path target component energy ratios；它复用同一 prefix-epsilon target 构造，不向受限 `S_k` 注入 timestep/class，也不训练 shuffled-order 判别。字段为 `loss.denoise_path_energy_weight`，v4 使用 `0.5`，旧 `energy_budget_weight` 显式为 `0`。
- 本地与远端 tracked `scale/generative-system` HEAD 现均为 `2a45b27dd197f5c014bded11f0590520e35c3527`。probe 增量 bundle 为 `7,140` bytes、SHA256 `f063706746839297852475478a7bc82deaf7c685cfabd391218f5487073f7e24`，prerequisite 为 `f1ffbfb`；部署 conflict 为 0，远端 `628 passed`，runbook syntax pass，回执位于 `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/probe-2a45b27dd197f5c014bded11f0590520e35c3527/deployment_receipt.json`。
- target-energy v4 fresh 5K 已启动：runbook PID `940283`、monitor PID `940299`、watchdog PID `940377`、trainer PID `940443`。权威 monitor：`/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_probe_v4_monitor.json`；run：`/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_rankcomplete_target_energy_k8_probe5k_v4`；日志：`/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_target_energy_probe_2026-07-20.log`。
- v4 首步已确认 revision/config/参数量正确：`denoise_path_energy=0.016774`、total `1.117636`、grad norm `0.38797`，所有值有限，GPU 利用 `97%`。当前 runbook 结束前禁止移动远端 HEAD 或启动其他 GPU 任务。
- v4 会自动完成 EMA/raw 的 `t=500` 512-image eval、EMA 的 `t=50/250/750/950` 256-image eval、四-prefix 512 DDIM-50 样本和方向性 FID/IS，但永远不会自动启动正式 50K/300K。只有 ordered rank、跨 timestep 角色、endpoint、能量分布、视觉 prefix、zero/shuffle/integrity/provenance 同时通过，才能更新 formal scaling/full config 与 recipe；当前仍不是“大规模生成全面完成”。

## Capacity-path 生成恢复更新（2026-07-20）

本节覆盖上文“v4 正在运行”和远端 HEAD 为 `2a45b27` 的旧快照。

- target-energy v4 已严格完成 5K、EMA/raw 与五 timestep mechanism eval、512 组四-prefix DDIM-50、FID/IS 和 aggregate。EMA 在 `t=500` 的 endpoint MSE 为 `0.0225144`、path AUC `0.2884318`、ordered rank `17/18`；raw 为 endpoint `0.0204233`、rank `16/18`。FID-512 `337.1272`、IS-512 `1.6390`，aggregate 决策为 `revise_architecture_or_objective`，自动 50K/300K 为 false。记录：`CoFiTok-internal/docs/records/2026-07-20_generation_target_energy_probe_v4_result.md`。
- v4 的 learned tail-2 energy 在 `t=50/250/500/750/950` 为 `65.7%/87.2%/89.8%/90.9%/91.4%`，equal-progress target 仅为 `41.2%/33.7%/27.7%/25.5%/25.0%`；对应 ordered rank 为 `2/14/17/17/17`。视觉上 p1 近纯噪声、p2 大块颜色、p4 伪轮廓纹理、p8 高频噪声，v4 不可用且不得晋级。
- 根因是 equal full-resolution progress 与异构 compressed token subspace 不匹配。v5 新增 opt-in `denoise_path_progress_mode="token_capacity"`：以 `r_k=min(C_k,C_image)*H_k*W_k` 估计受限 synthesis rank，幅度增量按 `sqrt(r_k)` 分配，空间 target 使用 token 真实分辨率；旧 `power` 模式保持原语义。当前布局累计进度为 `[0.027547,0.055094,0.110188,0.165282,0.275470,0.385658,0.640127,1.0]`，高噪声目标能量近似 `[0.0034,0.0034,0.0134,0.0134,0.0537,0.0537,0.2864,0.5728]`。
- 本地与远端 tracked HEAD 现均为 `3f98a4e106a1251d5242a41e260bd9fc4686c5f0`。增量 bundle 为 `9,464` bytes、SHA256 `ad9e28570c13a49bb84636ae0c8aaf35e5a1e870aab88b409b5fc93082e4e9e6`，唯一 prerequisite 为 `2a45b27`；远端全量 pytest、runbook `bash -n`、capacity config 和 tracked-clean 检查均通过。receipt：`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/probe-3f98a4e106a1251d5242a41e260bd9fc4686c5f0/deployment_receipt.json`。
- capacity-path v5 已启动：runbook PID `65639`、monitor PID `65655`、watchdog PID `65735`、trainer PID `65801`。权威 monitor：`/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_probe_v5_monitor.json`；run：`/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_rankcomplete_capacity_path_k8_probe5k_v5`；日志：`/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_capacity_path_probe_2026-07-20.log`。
- v5 step 1/50/100/150 均有限。energy-ratio 初始化梯度瞬态从 pre-clip `35.1` 降至 `12.9` 再降至 `1.34`，epsilon loss 已开始下降，watchdog/monitor 为 `running` 且无 issue；未触发止损。v5 完成前禁止移动远端 HEAD 或启动其他 GPU 任务。
- v5 仍只是 non-formal 5K 机制 probe；必须通过 EMA/raw 的 ordered rank `1/18`、五 timestep、endpoint、组件目标匹配、prefix 视觉、zero/shuffle/integrity/provenance 和方向性采样审查后，才可锁定正式 objective。之后仍需 fresh matched 10% 50K、10K promotion、full matched 300K、双方法正式 50K DDIM-250、final gate、EMA export 和 terminal completion audit，当前绝不能称“大规模生成全面完成”。

## Light capacity-path 生成恢复更新（2026-07-20）

本节覆盖上文“v5 正在运行”和远端 HEAD 为 `3f98a4e` 的旧快照。

- capacity-path v5 已严格完成 `5,000/5,000`、EMA/raw 与五 timestep mechanism eval、512 组四-prefix DDIM-50、FID/IS、aggregate 和 monitor。checkpoint SHA256 为 `51138d35be612d94b507bfd20a80ef4f68ff94693c216803e9deeff006fc1644`；最终 validation epsilon MSE `0.02465395`。EMA/raw 在 `t=500` 均为 ordered rank `1/18`，EMA endpoint MSE `0.0216982`、path AUC `0.0872809`；EMA 在 `t=50/250/750/950` 的 rank 为 `2/1/1/1`。zero-token 精确为 0，`t=500` shuffle/ordered endpoint ratio 为 `92.68`。
- v5 只通过机制排序，不是最终 objective：FID-512 `370.5892`、IS-512 `1.3818`，弱于 v2 的 `275.1347/2.2749`；`t=500` 前六 token 仅占约 `0.89%` learned energy，而 capacity target 为约 `14.22%`。固定 prefix 1/2/4 仍是结构化噪声，prefix 8 只有微弱类别轮廓并被高频纹理主导。记录：`CoFiTok-internal/docs/records/2026-07-20_generation_capacity_path_probe_v5_result.md`；本地小型报告和样例归档保持 untracked，不提交 checkpoint 或完整 sample set。
- v6 是最小可归因组合：保留 v2 较好方向质量对应的 `denoise_path_prefix_weight=0.05`、`denoise_path_component_weight=0.10`，仅把错误的 power path 换成已证明可排序的 `denoise_path_progress_mode="token_capacity"`，并令无效的 `denoise_path_energy_weight=0`；数据、layout、骨干、seed、优化器、EMA、5K runtime 与评估协议完全不变。配置：`configs/generation/imagenet256_10pct_rankcomplete_capacity_path_light_k8_probe5k.json`。
- 本地与远端 tracked HEAD 现均为 `ead85dcb304bea8d5b23611add8e5397c65b5e73`。增量 bundle 为 `6,260` bytes、SHA256 `dac7c93714f6d74dfb97bb7d27ae4fed9a7ddc8bfca4d981ba05e22b2f859ee1`，唯一 prerequisite 为 `3f98a4e`；untracked conflict 为 0，本地与远端全量 pytest 通过，远端 tracked runbook `49/49` syntax 通过。receipt：`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/probe-ead85dcb304bea8d5b23611add8e5397c65b5e73/deployment_receipt.json`。
- v6 已启动：runbook PID `303377`、monitor PID `303391`、watchdog PID `303478`、trainer PID `303543`。权威 monitor：`/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_probe_v6_monitor.json`；run：`/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_rankcomplete_capacity_path_light_k8_probe5k_v6`；日志：`/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_capacity_path_light_probe_2026-07-20.log`。step 1 已有限：total `1.01495`、epsilon `0.99981`、grad norm `0.38088`，GPU `100%`，watchdog/monitor 正常。v6 完成前禁止移动远端 HEAD 或启动其他 GPU 任务。
- v6 仍是 non-formal probe，永远不得自动启动 50K/300K。只有它同时保持 EMA/raw ordered rank `1/18`、endpoint 接近 matched dense、方向 FID/IS 显著恢复、prefix 视觉不退化且 integrity/provenance/zero/shuffle 通过，才能原子锁定 formal scaling/full recipe。之后仍必须完成 fresh matched 10% 50K、10K promotion、full matched 300K、双方法正式 50K DDIM-250、final gate、EMA export 和 terminal audit，当前仍不是“大规模生成全面完成”。

## Hellinger prefix-utilization 恢复更新（2026-07-21）

本节覆盖上文“v6 已启动”和远端 HEAD 为 `ead85dc` 的旧快照。

- v6 已严格完成 `5,000/5,000`、EMA/raw 与五 timestep mechanism eval、512 组四-prefix DDIM-50、FID/IS、aggregate 和 monitor。checkpoint SHA256 为 `1032cd8b0702c41f5da64d4df5dae7a628dfec20a1b00fc345150456c62aa53f`；最终 validation epsilon MSE `0.02412383`。EMA/raw 在 `t=500` 都是 ordered rank `1/18`，EMA endpoint `0.0199877`、path AUC `0.0907247`；zero-token 精确为 0，shuffle/ordered endpoint ratio `104.50`。
- v6 仍被拒绝为正式 objective：FID-512 `294.0330`、IS-512 `1.9437`，虽优于 v5，但 FID 比 v2 `275.1347` 差约 `6.87%`；更关键的是 `t=500` 前六 token 只占约 `0.56%` learned energy，而 capacity target 为约 `14.22%`。p1/p2 几乎相同噪声、p4 只有弱低频结构，等效为两 token predictor。记录：`CoFiTok-internal/docs/records/2026-07-21_generation_capacity_path_light_probe_v6_result.md`；小型报告与样例归档保持本地 untracked。
- v7 新增 opt-in `denoise_path_energy_mode="hellinger"`，历史默认仍是 `mse`。它保持 v6 的数据、layout、骨干、seed、优化器、EMA、path 权重、5K runtime 与评估协议完全不变，只启用 squared-Hellinger component-energy distribution loss，权重 `0.1`。配置：`configs/generation/imagenet256_10pct_rankcomplete_capacity_path_hellinger_k8_probe5k.json`。
- 本地与远端 tracked HEAD 现均为 `94a3bb3a11da94ef91f2fe165aa092f5ba5ea129`。增量 bundle 为 `9,229` bytes、SHA256 `05e7629c905e443dd7f0853823cd8949d9daaab0b282a380f38dfc5a70ae68df`，唯一 prerequisite 为 `ead85dc`；本地全量 pytest、隔离 Linux 全量 pytest、部署后 Linux 全量 pytest 和远端 tracked runbook `50/50` syntax 均通过。receipt：`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/probe-94a3bb3a11da94ef91f2fe165aa092f5ba5ea129/deployment_receipt.json`。
- v7 已严格完成 `5,000/5,000`、EMA/raw 与五 timestep mechanism eval、512 组四-prefix DDIM-50、FID/IS、aggregate 和 monitor。checkpoint 为 `1,006,396,450` bytes，SHA256 `0fd5a005adc9e39301f598edd33f1d013c85f5c165369b73107843a857e00996`；最终 validation epsilon MSE `0.02439082`。EMA `t=500` endpoint `0.02072175`、path AUC `0.09709960`、ordered rank `1/18`、zero-token 精确 0、shuffle ratio `101.2445`。EMA 在 `t=50/250/750/950` 全部 rank `1/18`，raw `t=500` 也为 rank `1/18`。
- v7 把 `t=500` 前六 token 的逐样本归一化 component energy 从 v6 的 `0.56%` 提升到 `7.48%`；其他四个 timestep 为 `10.56%/7.92%/7.45%/7.46%`。固定样例在 prefix 4 已出现可重复低频结构，prefix 8 再补高频细节。FID-512 为 `296.3751`，只比 v6 差 `0.80%`；endpoint 比 v6 差 `3.67%`，均未出现明显方向性退化。因此 v7 被选为正式 objective。记录：`CoFiTok-internal/docs/records/2026-07-21_generation_capacity_path_hellinger_probe_v7_result.md`；报告归档：`CoFiTok-internal/artifacts/reports/generation/rank_recovery_probe_2026-07-21_v7/`。
- 正式 scaling/full 配置必须使用 `token_capacity` progress、prefix/component/Hellinger-energy 权重 `0.05/0.1/0.1`；训练 recipe 升级为 `cofitok_generation_training_recipe_v2`。promotion/final gate 升级为 schema v2，并要求 evaluator 的 `component_energy_ratio_per_sample_mean` 中 tokens 1-6 合计至少 `5%`，同时继续要求 rank 1、exact zero-token 和 shuffle mismatch。该 formal-hardening revision 完成测试、提交、部署 attestation 前仍禁止启动 50K；之后必须从零运行 fresh matched 10% 50K、10K promotion、full matched 300K、双方法正式 50K DDIM-250、final gate、EMA export 和 terminal audit，当前仍不是“大规模生成全面完成”。
- formal-hardening 已在提交 `83d5b07f1fd8a169688b77cdf2b06ed7cbb90a24` 落地，Linux 预检暴露并由后续提交 `8e91e92aa3f4902117303c8bd46ffa9ca55b43fe` 修复 derived energy sum 的跨平台浮点等值问题；只有派生求和使用 `1e-12` absolute tolerance，原始指标、协议和 SHA 仍严格比较。最终 target 已 fast-forward 部署，远端 tracked clean。target-specific attestation receipt：`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/generation-upgrade-8e91e92aa3f4902117303c8bd46ffa9ca55b43fe.receipt.json`；aggregate bundle 为 `12,103,891` bytes、SHA256 `bed50fe1e3eb500eac89d79cd72eaf9ec4b8d09192288b3eee2612e116cb7cfe`，唯一 prerequisite `781a01444fddbf0d48a427ba58bdeed50167b5be`。receipt 绑定远端 `641/641` pytest、`50/50` tracked runbook syntax、`0/226` untracked conflicts。下一步允许从零启动同一 revision 的 fresh true-compressed 10% matched 50K pair；在 promotion/final 全链通过前仍不得称“大规模生成全面完成”。
- 首次 formal 50K 启动在六个 runtime benchmark 完成后被 fail-closed selector 拒绝：训练入口会在 PyTorch import 前默认设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`，selector 主进程此前没有设置，导致 runtime-environment SHA 跨进程不一致。修复提交为 `a8f43f4b8695ce1ac92ca9bd774309c62947396e`，新增独立入口回归并通过本地/远端全量 pytest；远端 conflict 为 0、tracked clean，新的 immutable receipt 为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/generation-upgrade-a8f43f4b8695ce1ac92ca9bd774309c62947396e.receipt.json`。旧 `8e91e92` receipt 只证明前一 target，不再授权当前训练。
- fresh true-compressed 10% matched 50K completion supervisor 已于 2026-07-21 启动：supervisor PID `118165`、pair monitor PID `120198`、CoFiTok trainer PID `120279`，权威状态为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_completion_supervisor.status.json`、`generation_complete_pipeline_after_10pct.status.json` 和 `generation_10pct_rankcomplete_v2_pair_monitor.json`。runtime selection schema v2 已冻结为 `micro_batch=64`、`gradient_accumulation=1`、effective batch `64`，最坏方法实测 `2.2261 s/step`、峰值显存比例 `55.10%`，绑定 revision `a8f43f4b...`、runtime SHA `51ef815b...` 与 dataset identity `97cfec24...`。
- 权威 CoFiTok run 为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2`。启动审计已到 step `150/50,000`、samples seen `9,600`、累计 `335.56s`，total/epsilon/prefix/component/Hellinger-energy loss 为 `0.99484/0.98110/0.21164/0.03065/0.000881`，全部有限；首个 monitor 轮询为 `running/cofitok_training`、`issues=[]`，GPU `100%`、约 `69.9 GiB`、63C，watchdog 无 intervention。运行期间禁止移动远端 HEAD 或启动其他 GPU 任务；supervisor 会串行完成 CoFiTok 50K、dense 50K、10K promotion gate，只有通过后才会进入 full 300K 和最终 50K/DDIM-250/release/audit 链。
- 2026-07-21 启动后只读审计确认 full 阶段三个权威路径均不存在：`checkpoints/generation/imagenet256_full_cofitok_k8_300k`、`checkpoints/generation/imagenet256_full_dense_300k`、`CoFiTok-internal/artifacts/reports/generation/imagenet256_full_matched_300k`。因此 promotion 通过后的 full 300K 会从干净目录启动，不存在旧 revision 的 `latest.json`、run manifest、checkpoint 或报告触发 stale exact-resume；不要提前创建或写入这些路径。
- live progress auditor 快照位于 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_rankcomplete_v2_cofitok_progress.json`。它以 `integrity-policy=required`、checkpoint interval `5000`、validation interval `1000` 审计当前 JSONL；step 200 时结果为 `healthy`、`issues=[]`、`warnings=[]`、checkpoint `not_due`、validation logging complete，实测 `2.226465 s/step`、剩余 ETA `110,878s`（约 30.8h）。日志中的 grad norm 是 pre-clip total norm；配置的 clip 上限仍为 `1.0`，不得把 warmup 期间的 pre-clip exceedance 误报为未裁剪更新。
- promotion post-eval 非训练依赖已只读预检：`datasets/imagenet_256/extracted/val` 恰为 `50,000` 个文件、`0` symlink、约 `773 MiB`；`torch_fidelity 0.4.0` 已安装，Inception/VGG 权重已在 `/root/.cache/torch/hub/checkpoints/`；`eval_cache/torch_fidelity` 约 `1.4 GiB`，缓存名绑定 real-set digest 前缀 `19ace4e37bee2fca`，对应完整 tree SHA256 `19ace4e37bee2fcaeac2b7cbd26ed785aa0e6e01015c90b4955027d163f6e44f`。按 20,256 个 post-eval 样本和 32 GiB safety margin 重跑 storage preflight 为 `pass`，当前 headroom 约 `493.2 GB`；因此 10K gate 不依赖临时联网，且数据/缓存/空间均已就绪。
- full 300K 启动条件也已预检：full config validator 为 `pass`，CoFiTok/dense 参数为 `62,836,796 / 62,824,707`、相对差 `0.019242%`，共享数据/骨干/扩散/优化器/runtime 无 mismatch，recipe v2 与 50K/100K/200K/300K protected milestones 有效。`imagenet_256` manifest 为 `405,484,553` bytes、SHA256 `9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`、`1,331,167` 行；物理 train/val 文件数严格为 `1,281,167 / 50,000`，整个数据树 `0` symlink。以 v7 的 `1,006,396,450`-byte checkpoint 代表性估算 16 个 checkpoint、16,384 个 milestone 样本和 64 GiB safety margin，full storage preflight 为 `pass`、headroom 约 `443.7 GB`；正式切换仍必须用届时双 50K 权重重跑权威 preflight。
- final 50K comparison/release 依赖也已预检：锁定的 official-related table 在本地/远端均为 `3,346` bytes、SHA256 `14cd71cb72e7ee121f571503cd75491858840de806a97fba9cda188d034c748a`，D-AR/MAR/ReTok 三行均明确为 50K official pretrained eval-only secondary context；表中六个原始 metrics/NPZ 文件在服务器真实存在，三份 NPZ 各约 `9.2 GiB`。按 100,256 个正式 DDIM-250/prefix 样本与 64 GiB safety margin 的 final storage preflight 为 `pass`、headroom 约 `437.9 GB`。`exports/imagenet256_full_300k` 以及 full report 下的 `exports/`、`comparison/`、`visual_audit/` 均不存在，最终 gate 后会从干净目录导出双 EMA artifact、执行真实前向 preflight 和 smoke；这些 readiness 检查不替代尚未产生的 full 指标、release gate 或 terminal completion audit。
- 分支隔离在 fresh 50K 运行中再次双端核验：本地与远端均为 `scale/generative-system@a8f43f4b8695ce1ac92ca9bd774309c62947396e`，锁定基线仍为 `paper-evidence-locked@88c3aabf5408f19da92ae01e33e23878e018a4db`，且 locked revision 是当前 HEAD 的祖先；远端 tracked dirty count 为 0。`paper/` 是 `CoFiTok-internal` Git 仓库之外的 sibling 工程，生成升级 commit 不能改写论文目录；locked artifacts 也不得被当前 generation run 输出覆盖。
- fresh CoFiTok scaling run manifest 已在运行中只读核验：文件为 `7,013` bytes、SHA256 `48e1a8ee81e37347ce36fe307c528fbae86d5be948b3bf034f84911066489901`，绑定完整 resolved config、v7 Hellinger objective、`64x1` runtime、`62,836,796` 参数、seed `2027`、Git `a8f43f4b...`、dataset identity `97cfec24...`、runtime environment SHA `51ef815b...` 以及 `pyproject.toml`/`uv.lock` 身份。fresh scaling 的 `resume=null`、`metrics_resume_reconciliation=null`、`training_authorization=null` 均正确；首个 step-5000 checkpoint 必须把同一 config/dataset/runtime/Git 身份写入 payload、integrity sidecar 和 `latest.json` 后才算可恢复性实证通过。
- 2026-07-21 08:15 CST，fresh CoFiTok 已完成首个 scheduled validation：step `1,000/50,000`、samples seen `64,000`、total/epsilon loss `0.1157667/0.1007031`，同一权威 metrics 行包含有限的 `validation_epsilon_mse=0.1059973`。正式 progress auditor 为 `healthy`、`issues=[]`、`warnings=[]`，validation events `1/1`、logging complete，实测 `2.226631 s/step`；GPU `100%`、约 `76.2 GiB`、65C。checkpoint interval 为 5,000，因此当前 integrity 状态正确地为 `not_due`；不得把 1K validation 通过误称为 checkpoint 可恢复性已证明或 10% promotion 已通过。
- 首个 5K checkpoint 的一次性只读 waiter 已于 2026-07-21 08:20 CST 启动，PID `124618`、4 小时硬超时；脚本位于远端临时路径 `/tmp/cofitok_step5000_audit_waiter.sh`，PID/log 分别为 `checkpoints/generation/generation_10pct_rankcomplete_v2_step5000_audit.{pid,log}`。它等待 checkpoint、integrity sidecar 和 `latest.json` 同时出现后，仅调用 tracked `audit_generation_training_progress.py --integrity-policy required`，输出 `generation_10pct_rankcomplete_v2_cofitok_step5000_audit.json` 并在 log 中记录四个文件的 SHA256；不加载权重、不使用 GPU、不重启或终止训练。trainer 的 checkpoint 顺序已只读核验为 payload 临时写入 + `fsync` + 原子替换，再原子写 sidecar 和 `latest.json`；最终仍必须读取 audit 内容确认 `healthy/verified`，不能只凭 waiter 退出码宣称通过。
- fresh matched pair 的公平性 preflight 小型报告已同步到本地 `CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/`。`config_pair.json` 为 `11,663` bytes、SHA256 `8fe3c6c8bcf9bc3329cf36f2308dfc49c8c96799dbfcc8dda5843f14272ce24d`，状态 `pass`、mismatch 为空，参数为 `62,836,796 / 62,824,707`、差 `0.0192424%`；`runtime_selection.json` 为 `68,020` bytes、SHA256 `6638f7f63ae5a685f346497bea767feadb637e8ae76c132bc596d9ff5e8aded6`。后者的 selection lock 与实际 CoFiTok run manifest 已交叉核验：`64x1`、effective batch `64`、Git `a8f43f4b...`、dataset identity `97cfec24...`、runtime SHA `51ef815b...` 全部精确相等，两份冻结 config SHA 也与 tracked 文件一致。`config_pair.json` 中的 `16x4` 是被 selector 覆盖前的基准 recipe，实际运行必须以 schema-v2 selection 和 run manifest 的共同 `64x1` 为准。
- 上述本地 evidence 目录还原字节归档了三份服务器控制面证据：`deployment_receipt_a8f43f4.json` 为 `2,305` bytes / SHA256 `de3f460f7b9786fb2b585a779d67a2708619925ca4e60e04c24f09fa0d2b92ca`，`cofitok_run_manifest.json` 为 `7,013` bytes / SHA256 `48e1a8ee81e37347ce36fe307c528fbae86d5be948b3bf034f84911066489901`，`cofitok_progress_step_00001000.json` 为 `2,058` bytes / SHA256 `43162cdc92fe4561b440b370a7fc67e7a4e327182c3d9ad4256e0c98b3c9adbc`。它们与 `config_pair.json`、`runtime_selection.json` 共同构成当前 10% fresh pair 的 pre-checkpoint evidence pack；这些小型报告保持本地 untracked，不移动远端 HEAD，也不把训练权重拉回本地。5K 后应追加 verified progress audit 和 sidecar/latest 摘要，而不是覆盖 1K 快照。
- 该 evidence pack 已增加机器可读索引 `evidence_manifest_precheckpoint.json`，本地/远端均应为 `3,240` bytes、SHA256 `14525fbcae433be44d7018123a029965a2e0fc7ef771982bb1d92acc99087ea9`。索引现对 6 个源文件逐项绑定 filename/bytes/SHA256/role，机器复核为 `6/6` bytes 与 SHA 全匹配；其状态明确为 `in_progress`，`training_complete=false`、`promotion_complete=false`，下一要求固定为 step-5000 audit 的 `healthy` 与 integrity `verified`。该 pre-checkpoint manifest 不是 completion receipt，不得用于宣称 10% pair、promotion 或大规模生成已完成。
- runtime selector 的三组共同候选均为 effective batch `64` 且 eligible，按两方法中较慢者计分：`16x4=2.275394s`、`32x2=2.237220s`、`64x1=2.226135s`，所以 `64x1` 是既定 policy 下的正确共同最优。选中候选的 CoFiTok/dense 实测为 `2.226135/2.017078 s/step`、`28.7494/31.7291 images/s`；CoFiTok 因 factorization 辅助目标有 `10.3643%` wall-time 开销。最终报告可称同数据、骨干、优化器、训练 steps/images 和采样协议，不得把它写成绝对相同 wall-clock compute；必须同时报告双方实际 training hours、throughput 与 peak VRAM。
- 第二次 scheduled validation 的一次性只读 waiter 已启动，PID `125576`、1 小时硬超时；远端脚本为 `/tmp/cofitok_validation2_audit_waiter.sh`，PID/log 为 `checkpoints/generation/generation_10pct_rankcomplete_v2_validation2_audit.{pid,log}`。它仅等待 metrics 中出现 step 2,000，再调用 tracked progress auditor 写入 `generation_10pct_rankcomplete_v2_cofitok_validation2_audit.json` 并记录报告 SHA；不加载模型、不使用 GPU、不干预训练。触发后必须确认 audit 为 `healthy`、validation events `2/2`、logging complete、issues/warnings 为空；该 validation waiter 与 PID `124618` 的 5K checkpoint integrity waiter 职责不同，任一通过都不能替代另一项证据。
- 上述 validation2 waiter 已于 2026-07-21 08:52 CST 正常完成并退出。权威 step-2,000 audit 为 `2,076` bytes、SHA256 `417298f38e44bba8a1318e8e84e487f19af29d1c7a809d294703763ea8e9ed35`，状态 `healthy`、issues/warnings 为空、validation events `2/2`、logging complete；`validation_epsilon_mse=0.03593044`，EMA warmup 精确达到 `0.9999`，实测 `2.227901 s/step`。本地归档名为 `cofitok_progress_step_00002000.json`，1K 快照仍保留；checkpoint 在 5K 前正确为 `not_due`，当前仍未证明可恢复性。
- 5K checkpoint 的真实 exact-resume payload 一次性 CPU waiter 已启动，PID `128227`、3 小时硬超时；远端临时脚本 `/tmp/cofitok_step5000_payload_audit_waiter.py` 已通过目标 Python `py_compile`，PID/log 为 `checkpoints/generation/generation_10pct_rankcomplete_v2_step5000_payload_audit.{pid,log}`。它必须先看到 PID `124618` 产出的 step-5000 integrity audit 为 `healthy/verified`，之后在 `CUDA_VISIBLE_DEVICES=` 下只读加载权重到 CPU，核对 model/EMA/optimizer/scheduler/RNG/sampler/config、Git/runtime/dataset 身份、bf16 scaler、sampler cursor 与 samples seen，并要求加载前后 checkpoint bytes/mtime 不变；输出为 `generation_10pct_rankcomplete_v2_cofitok_step5000_payload_audit.json`。它不调用保存、恢复训练、GPU 或进程控制；只有 integrity audit 与 payload audit 都通过，才可称首个真实 checkpoint 具备受验证的可恢复性。
- 5K 小型证据自动归档 collector 已于 2026-07-21 09:32 CST 启动，timeout PID `254359`、child PID `254361`、3 小时硬超时；PID/log 为 `checkpoints/generation/generation_10pct_rankcomplete_v2_step5000_evidence_collector.{pid,log}`。它只等待上述 integrity/payload 两份审计与 sidecar/`latest.json` 同时出现，然后原样复制到当前 v2 evidence pack，并对源/目标小文件同时执行 SHA256；不复制、不加载、不重新哈希 `.pt` 权重，不使用 GPU，也不控制训练进程。归档后仍需逐字段读取两份审计并在本地 evidence manifest 中验证 bytes/SHA，不能仅凭 collector log 宣称 checkpoint 通过。
- 第四次 scheduled validation 的只读 waiter 已于 2026-07-21 09:34 CST 以修正版启动，timeout PID `254561`、child PID `254563`、2 小时硬超时；PID/log 为 `checkpoints/generation/generation_10pct_rankcomplete_v2_validation4_audit.{pid,log}`，目标报告直接写入当前 v2 evidence pack 的 `cofitok_progress_step_00004000.json`。它按 JSONL 从 step 3,000 的 61 行等待到 step 4,000 对应的 81 行，再调用 tracked progress auditor 验证真实 step 和 validation `4/4`；不加载模型、不使用 GPU、不干预训练。第一次 PID `254500` 因临时 shell quoting 错误在进入等待循环前退出，仅留下错误 log，未生成或修改任何实验报告；不得将该失败 PID 误认为训练故障。
- 2026-07-21 09:50 CST 的 formal-vs-v7 窗口健康对照绑定了两份 config SHA 和各自 through-step-3000 metrics-prefix SHA。formal 与 v7 在 step `1-1000/1001-2000/2001-3000` 的 epsilon 均值分别为 `0.584948/0.047270/0.037920` 与 `0.584126/0.045014/0.037371`，total loss 也近似重合，未发现窗口级训练回退。正式配置相对 v7 的实质差异仅为随机水平翻转、50K 长程 cosine 最低学习率、更多 checkpoint 保留和总步数；单批 validation 波动不能替代 sample-quality gate。报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/cofitok_formal_vs_v7_training_health_step_00003000.json`，SHA256 `42e724bc3d7dff1d3ae66a5ed3e59d31e5470b14ea0a79c61a671e3a7f5086ec`。
- 第四次 scheduled validation waiter 已正常完成：step `4,000` audit 为 `healthy`、issues/warnings 为空、validation `4/4` 且 logging complete，`validation_epsilon_mse=0.03169604`，实测 `2.229785 s/step`。本地归档 `cofitok_progress_step_00004000.json` 为 `2,076` bytes、SHA256 `3f4ef6c769d0a4ce3f5521acb5108b3fcd56562d798cd74461e10a1fea04b1de`；此时 checkpoint 仍正确为 `not_due`。
- 首个原生 step-5,000 checkpoint 已于 2026-07-21 10:44 CST 通过双重实证。progress/integrity audit 为 `healthy`、issues/warnings 为空、validation `5/5`、integrity `verified`；checkpoint 为 `1,006,396,450` bytes、SHA256 `f2d8287252e26d07ce90a9749d7e9dd427be591fe94b5e706fb196f2529e9b6a`。CPU-only payload audit 为 `pass`：model `281` entries、EMA `4`、optimizer `281` states/`2` groups、scheduler/RNG/sampler/config 全部存在，samples seen `320,000`、sampler epoch/position `2/63,744`，Git/data/runtime identity 精确匹配，且 audit 前后 checkpoint bytes/mtime 均未变化。sidecar、`latest.json` 和两份 audit 已按原字节同步回本地 evidence pack；权重本体未拉回本地。
- step-5,000 collector 的四组源/目标 SHA 全部一致；其最后一条临时 `stat` 命令因 shell format 参数 quoting 记录了两条非致命 warning，发生在四组 SHA 已匹配之后，不影响复制、checkpoint、训练或独立 payload audit。显式说明见 `cofitok_step5000_collector_verification.json`。更新后的 evidence manifest 共 `28` 项、`0` mismatch、`11,431` bytes、SHA256 `efd611d4ea1d249c63ef2060dceaf0b17e95e57d13bf4a4285356e6dd4748966`；状态仍为 `in_progress`，下一权威要求是 CoFiTok step-50,000 完成及最终 checkpoint/training report，随后 dense 50K 和正式 10K promotion gate。checkpoint 保存后训练已继续到至少 step `5,050`，pair monitor 为 `running/cofitok_training`、issues 为空，GPU `100%`。
- step-5,000 通过后再次运行 tracked terminal completion auditor，进程按设计以 exit `1` 拒绝提前完成；报告仍为 `in_progress`、`0 failed`、`0 warnings`、`17 missing`，唯一终局 pass 仍是 `controlled_revision_transition`。这证明 5K checkpoint 只解决当前 scaling run 的 integrity/exact-resume 实证，不会错误满足双 50K、promotion、full 300K、formal 50K、EMA release 或 final comparison。re-audit source SHA256 为 `3dff962581ad7c170d3326846d04b8ebcb8909e1d4b94b6c01c74d8af2d0c6f4`，attestation SHA256 为 `cb30dfc1bcf62d498ca86f467efbef704a1c80853ec7dd927efbe62f40274e6d`。evidence manifest 现共 `30` 项、`0` mismatch、`12,222` bytes、SHA256 `4122c00d167a8cc6a8dba0104519a1478cbdf63560ff0a28c9b6a54f885bf49e`；训练已继续到至少 step `5,550`，GPU `100%`、monitor `running/cofitok_training` 且 issues 为空。
- 单方法结束时的 `validate_completed_generation_training` 会核对 training report/config/Git/step、`latest.json` 相等及 checkpoint/sidecar 文件存在；checkpoint bytes/SHA 的重新验证最迟会由 post-eval loader 和 terminal audit fail closed，但原 runbook 在 CoFiTok 结束后会立即启动 dense，因此存在损坏时较晚发现并浪费 dense 算力的观测窗口。运行中不得为修复该窗口移动 revision 或改写 tracked runbook；当前以独立只读 observer 缓解：`cofitok_50k_completion_evidence_waiter.sh` 为 `6,726` bytes、SHA256 `f8388ddac37529ffbf13540bd2ed4cd62a4d950597824280eca737e3c89e7292`，已通过远端 `bash -n` 及全部 `3/3` Python heredoc compile，timeout PID `260148`、child PID `260150`、36 小时硬超时、每 240 秒轮询。它只在 CoFiTok report/step-50,000 checkpoint/sidecar/`latest.json` 齐备后运行 tracked progress auditor 的 `integrity-policy=required`，要求 `healthy`、50/50 validation、SHA verified，再原子复制三份小 JSON 并核对 source/destination SHA；不使用 GPU、不加载/复制权重、不控制训练/supervisor、不改变 pipeline decision。启动记录：`cofitok_50k_completion_evidence_waiter_launch.json` 为 `2,797` bytes、SHA256 `26950a386ec4409925badc5a2f1a0724bfea9c452f67ea83a2c815d88925df59`。evidence manifest 现共 `32` 项、`0` mismatch、`13,032` bytes、SHA256 `7607d396f8f807ddc4abdd0eb777f867d95763ee68c182ed9e8520d96e0d4480`；observer success 也不能替代 dense 50K 或 promotion gate。

## 大规模生成升级最新状态（2026-07-24）

- 上述 2026-07-21 的 live-training 条目现已被终态证据取代。fresh rank-complete true-compressed 10% matched pair 在同一 `scale/generative-system@a8f43f4b8695ce1ac92ca9bd774309c62947396e` 上全部完成：CoFiTok/dense 均为 `50,000/50,000` steps、`3,200,000` images seen、validation `50/50`、terminal progress audit `complete`、issues/warnings 为空，最终 checkpoint integrity 均为 `verified`。CoFiTok checkpoint 为 `1,006,396,450` bytes / SHA256 `4ca727feba29a12045da5aa6bbaf5030c622cf1f9cb4348e208a35da616de239`，dense 为 `1,006,118,998` bytes / SHA256 `6bd395928fa35cb46f3689fea0664556ea50215a092217cb68f642ac6a214227`；pair monitor 终态为 `pass/complete`。
- 双方法正式 scaling 采样均已完成精确 `10,000` 张 EMA、bf16、balanced-modulo、DDIM-100、CFG `1.5`、guidance rescale `0`、eta `0`、`clip_x0=true` 的同协议评估。CoFiTok FID/IS/precision/recall 为 `191.2347 / 3.4265 / 0.9760 / 0.000060`，dense 为 `115.0052 / 9.2240 / 0.6404 / 0.0128`。权威 `promotion_gate.json` 为 `fail/hold`：CoFiTok FID 相对 dense 回退 `66.2835%`（门槛 `<=5%`）、绝对 FID `191.2347`（门槛 `<=100`）、endpoint MSE 回退 `5.1412%`（门槛 `<=5%`）。因此 full ImageNet-256 matched 300K 仍未授权，严禁绕过 gate 启动或把 10% 结果称为大规模生成完成。
- 机制证据在该失败 gate 中仍有效：ordered prefix 在 18 个顺序中排名 `1`，前六个 token 能量占比 `7.8087%`（门槛 `>=5%`），zero-token max abs 为 `0`，shuffle/ordered endpoint ratio 为 `117.6538`。这说明当前失败集中在生成质量与相对 dense 退化，不是有序性、受限 synthesis 或 checkpoint/provenance 契约失效；不得用机制通过掩盖样本质量失败。
- step-50,000 临时 evidence waiter 的终态是 observer false negative：它在脚本 line 66 只接受运行态 `audit.status=healthy`，而权威 auditor 在精确完成时正确返回 `complete`，所以 waiter 在复制剩余小文件前退出。该误报未影响训练、checkpoint 或 formal pipeline decision；独立 completion evidence 来自 training report、`latest.json`、integrity sidecar、pre-promotion audit 以及成功加载 EMA checkpoint 的 1,024-image evaluation。postmortem：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/cofitok_50k_evidence_waiter_postmortem.json`。
- 本地 terminal-training/gate evidence 索引为 `CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/evidence_manifest_postgate_hold.json`，`7,506` bytes、SHA256 `cdc07324d51a8cabae6891ef3411102c004046f9d22367fe15f75527601ca0c9`；其 `18/18` 绑定文件已逐项验证 bytes/SHA 无 mismatch。状态必须保持 `promotion_hold`，不能改成 pass。
- 独立、非 promotion 的 matched sampling-recovery diagnostic 已在原 revision `a8f43f4b8695ce1ac92ca9bd774309c62947396e` 完成，权威状态为 `pass/complete`。raw model 相对 EMA 的 t=500 endpoint MSE 仅改善 CoFiTok `1.4559%`、dense `0.1584%`；CoFiTok 六组 128-sample DDIM-100 CFG/rescale FID 仅在 `301.4799--302.2837` 间波动，dense 为 `265.2326--269.5873`。因此 CFG、guidance rescale 和 EMA lag 均不是当前 CoFiTok 生成质量失败的主因；128-sample 数值仍只用于配置诊断，绝不是 promotion 或正式比较证据。汇总为 `CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2/sampling_recovery_diagnostic/summary.json`。
- 采样诊断最初在 summarizing 阶段因一次性脚本读取旧字段 `counts.generated` 而退出；正式指标实际使用 `counts.generated_image_count`。脚本已改为幂等恢复并只重建汇总，未重跑或改写十二组已完成样本；终态 status 为 `pass/complete`。该观察性脚本错误不影响 checkpoint、采样报告或当前 `promotion_hold`。
- 生成质量恢复现进入 non-formal v8 5K 探针：远端训练 revision 固定为 `scale/generative-system@5a93c7061e4df237234b19b052331fbc3ec10425`；本地 archive HEAD 为纯证据提交 `3d062c6`，不得在 v8 完成前同步到服务器移动训练 revision。完整本地及真实父目录 Linux pytest 均通过，所有 shell runbook 通过 `bash -n`。v8 只把 learned 1x1+3x3 synthesis 换成 deterministic、bias-free、condition-free、zero-preserving fixed RGB channel basis；其 synthesis trainable parameters 为 `0`，总参数 `62,836,011`，最后两个 token 的 RGB projection rank 为 `3`，其余数据、U-Net、loss 和 5K 预算与 v7 相同。
- v8 launcher PID `256821`，run 为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_10pct_rankcomplete_fixed_basis_hellinger_k8_probe5k_v8`，monitor 为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_probe_v8_monitor.json`，runbook log 为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_rank_recovery_probe_v8_runbook.log`。它会自动完成 5K training、EMA/raw t=500 mechanism eval、多 timestep energy scope、512-sample DDIM-50 directional FID/IS 和 v2--v8 聚合；不得从该 probe 自动启动 50K 或 300K。
- 下一步只能基于 v8 探针选择：若 fixed basis 同时保留 ordered rank 1、显著改善 endpoint 与方向性样本质量，则先冻结同版本 CoFiTok/dense fresh matched 50K 与 10K promotion protocol；只有新 gate `pass` 才能启动 full matched 300K。若 v8 不改善，则继续修改 CoFiTok 训练配方/架构。任何 128/512-sample diagnostic、阈值弱化、只重评 CoFiTok、或直接复用当前 failed gate 都不能授权 full 300K。

## 大规模生成升级最新状态（2026-07-25）

- v8 fixed-basis 5K probe 已终态 `pass`。最终 checkpoint 为 `1,006,351,466` bytes、SHA256 `fd69048f442896010333e892f77094f64e223a3e72bdb89ce0aecf5c656035a5`；五个审计 timestep 的 ordered rank 均为 `1`，zero-token max abs 为 `0`。t=500 EMA endpoint MSE 为 `0.01763455`，较 v7 下降 `14.90%`；path AUC 为 `0.08917027`，下降 `8.17%`。512-sample directional FID 为 `260.3077`，较 v7 改善 `12.17%`、较此前最佳 v2 改善 `5.39%`。这只支持选择 formal v3 配方，不是 generation-quality 或 full-300K 授权。终态记录：`CoFiTok-internal/docs/records/2026-07-25_generation_fixed_basis_probe_v8_result.md`。
- formal v3 把 scaling/full CoFiTok 的 synthesis 固定为 `fixed_basis + kernel_size=1 + fixed_one`；`S_k` 无条件、无偏置、无可训练参数且严格满足 `S_k(0)=0`。10% 权威参数为 CoFiTok `62,836,011`、dense `62,824,707`，差 `+0.017993%`。新 run 身份必须保持为 `imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3` 与 `imagenet256_10pct_fixed_basis_dense_50k_v3`，历史 v2 failed gate 和 run 不得覆盖或改名。
- 服务器已受控部署并冻结于 `scale/generative-system@58d83bfce2770eab2565b8c89a5f9a06201a0c86`。目标专属 receipt 为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/generation-upgrade-58d83bfce2770eab2565b8c89a5f9a06201a0c86.receipt.json`，`2,305` bytes、SHA256 `3a85d03023dd66223edb2ee46913b0e48f88e197a430a38650b09708d5b5285a`，状态 `pass`；它绑定 source `781a014`、target `58d83bf`、12,117,313-byte bundle、`647/647` pytest、`51/51` runbook syntax 和 `0` untracked conflict。双 50K 与 promotion 决策终态前严禁移动远端 tracked revision。
- fresh v3 matched 50K completion supervisor 已启动，PID `502309`。权威状态为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_completion_supervisor.status.json` 与 `generation_complete_pipeline_after_10pct.status.json`，日志为 `generation_completion_supervisor.log`。共同 runtime selector 已在双方 `16x4/32x2/64x1` 真实 benchmark 后选择 `64x1`，双方法较慢者 score 为 `2.216154 s/step`、最大显存占比 `55.0936%`、runtime environment SHA256 为 `51ef815bff2dcb9ea3e222cba9f0731dd837d11cf0b42cbf489f91e32075da57`。tracked progress auditor 的启动快照为 `healthy`、issues/warnings 为空、step `200/50,000`、images seen `12,800`、`2.219099 s/step`、ETA `110,511 s`；checkpoint/validation 均尚未到期。队列按 CoFiTok 50K -> dense 50K -> 双方法 10K EMA DDIM-100 -> promotion gate 串行执行。本地 bounded evidence pack：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/`；完整启动记录：`CoFiTok-internal/docs/records/2026-07-25_generation_fixed_basis_v3_matched_50k_launch.md`。
- 1K/2K/3K/4K validation 与 5K checkpoint 的只读 milestone waiter 已升级并启动为 PID `511498`；权威状态为 `/root/autodl-tmp/CoFiTok/checkpoints/generation/generation_10pct_fixed_basis_v3_milestone_waiter.status.json`，日志为同前缀 `.v2.log`，本地源码为 `CoFiTok-internal/artifacts/operations/generation/fixed_basis_v3_milestone_waiter.py`，远端 `/tmp` 副本与本地 SHA256 均为 `10fe73ca0f86835fab52e23263896f8c06111cd2900406e27485ef3415661f6c`。升级版通过 `11/11` targeted tests，会复用已通过的原始 1K 报告，不重写历史证据；旧 observer PID `507430` 已单独 TERM，trainer/watchdog/supervisor 均未中断。它只调用 tracked progress/integrity auditor，不加载模型、不使用 GPU、不控制训练；2K-4K 必须逐次验证 scheduled validation，5K 还必须验证 checkpoint/sidecar/`latest.json`/SHA。
- formal v3 与入选 v8 probe 的 early-trajectory alignment 已按 14 个相同 logged steps 审计到 step 650。`1-200/201-400/401-600` 三个窗口的 epsilon 相对差为 `-0.032%/+0.545%/+1.000%`，total loss 相对差为 `-0.054%/+0.621%/+1.037%`；最大窗口差 `1.037%` 低于诊断边界 `2%`。报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/early_trajectory_vs_v8.json`。这只证明 formal 长训忠实复现已选配方，不预测 50K 样本质量、不替代 promotion gate。
- step-1,000 scheduled validation 已由只读 observer 与 tracked progress auditor 双重确认：状态 `healthy`、issues/warnings 为空、validation `1/1` 且 logging complete、checkpoint 正确为 `not_due`，实测 `2.219961 s/step`。formal training epsilon/total 较 v8 同步数低 `15.10%/14.39%`，但首次 validation MSE 为 `0.04005979`，较 v8 的 `0.03588736` 高 `11.63%`。单次 validation 不足以停止健康长训，也不得用更低训练 loss 掩盖该观察；必须继续读取 2K-5K trend，并由正式样本 gate 裁决。报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step1000_validation_vs_v8.json`。
- step-2,000 scheduled validation 已继续通过 observer 与 tracked progress auditor：状态 `healthy`、issues/warnings 为空、validation `2/2` 且 logging complete、checkpoint 正确为 `not_due`，实测 `2.220870 s/step`。formal validation MSE 为 `0.03190126`，较 step 1,000 下降 `20.37%`，较 v8 同步点 `0.03756297` 低 `15.07%`；formal training epsilon/total 亦低 `5.94%/2.25%`。1K 与 2K 验证后的 GPU 显存点测均为 `77,983 MiB`，未观察到重复 validation 增长；这只支持 allocator plateau 的短期诊断，不是广泛内存安全或生成质量证明。继续保持冻结 revision，等待 3K-5K trend、5K checkpoint integrity 和正式样本 gate。报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step2000_validation_vs_v8.json`。
- step-3,000 scheduled validation 的 operational audit 仍为 `healthy`、issues/warnings 为空、validation `3/3` 且 logging complete、checkpoint 正确为 `not_due`，但质量诊断出现反向波动，必须如实保留：formal validation MSE `0.03765950` 较 2K 上升 `18.05%`、较 v8 同步点 `0.02329669` 高 `61.65%`，但仍较 formal 1K 低 `5.99%`；training epsilon/total 较 v8 同步点高 `29.15%/21.40%`。单个非单调 validation 事件不足以推断 50K 样本质量，且 formal recipe 与 5K probe 的 augmentation/schedule horizon 不同，因此冻结 run 继续不变，但必须加强 4K/5K 与正式 sample gate 审计。第三次验证后显存仍为 `77,983 MiB`，三次点测未见重复增长。报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step3000_validation_vs_v8.json`。
- step-4,000 scheduled validation 继续通过 operational audit：`healthy`、issues/warnings 为空、validation `4/4`、checkpoint `not_due`。formal validation MSE 降至 `0.02938104`，较 3K/2K/1K 分别改善 `21.98%/7.90%/26.66%`，说明 3K 反弹部分属于非单调波动；但其仍较 v8 同步点 `0.01850578` 高 `58.77%`，training epsilon/total 亦高 `40.57%/27.25%`，跨配方差距仍是不利观察。冻结 run 继续到 5K checkpoint integrity，最终样本质量仍只能由 formal gate 裁决。第四次验证后显存仍为 `77,983 MiB`，四次点测完全一致。报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step4000_validation_vs_v8.json`。
- step-5,000 observer 已终态 `pass`，tracked progress audit 为 `healthy`、issues/warnings 为空、validation `5/5` 且 checkpoint integrity `verified`。首个 formal v3 checkpoint 为 `1,006,351,466` bytes / SHA256 `8daedd38f44f719cbd488bc1d528187d1e8799dba6e879eed42bdc3f6bd7ce44`；sidecar、`latest.json` 与 progress report 的 step/bytes/SHA/dataset/runtime/Git 等 10 个核心字段 `0` mismatch，证明首个可恢复信任边界。formal 5K training epsilon/total 较 v8 低 `3.97%/6.31%`，但 validation MSE `0.03096321` 仍较 v8 `0.02271607` 高 `36.31%`，不得用更低训练 loss 或 checkpoint 完整性冒充样本质量。第五次验证及 checkpoint 后显存仍为 `77,983 MiB`。本地不复制 1 GB 权重，只存小型 identity 证据；报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step5000_validation_and_checkpoint_vs_v8.json`。
- 5K checkpoint 通过后已再次运行 terminal completion auditor，按设计 nonzero 返回 `in_progress`、`complete=false`、`1 pass + 17 missing + 0 failed`，唯一 pass 仍是 `controlled_revision_transition`。报告与 1K 快照字节级相同：`3,635` bytes、SHA256 `ab4c258638e300e41b575e46cc665cbaa30e0dd5b99fc1d81a3d919559ffbefc`。这证明可恢复 checkpoint 不会被误判成双 50K、promotion、full 300K、formal 50K、EMA export 或最终比较完成。快照：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/completion_audit_in_progress_after_step5000.json`。
- 1K-5K observer 正常终态退出后，已泛化并启动只读 long-horizon observer，目标为 `10K/25K/50K`，poll `240s`、硬超时 `129,600s`，50K 强制要求 progress `complete`。首次 PID `531838` 在 10K 暴露 observer 自身环境缺少项目 `src` 的 `PYTHONPATH`，导致其调用 tracked auditor 时 `ModuleNotFoundError`；训练、metrics 和 checkpoint 均未受影响。仅该 observer 被 TERM，并以相同 SHA-bound `/tmp` 源码和 `PYTHONPATH=/root/autodl-tmp/CoFiTok/CoFiTok-internal/src` 重启为 PID `555237`，随即补审 10K 为 `pass`。本地 successor 又把 `<project_root>/src` 自动前置到 child `PYTHONPATH` 并保留 inherited entries，新增 absent/existing 两种 regression；targeted `20/20`，完整 `666 collected / 664 passed / 2 skipped`。successor SHA256 为 `9f9d9768f09b9a35977244244408f7df9c6e402b179decc087b24afc6aaa5988`，只留本地，不替换健康运行中的远端 observer，也不移动 frozen revision。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/observer_self_contained_pythonpath_hardening.json`。
- 本地 pytest 入口也已消除同类隐式环境依赖：原 `pyproject.toml` 只声明 `pythonpath=["src"]`，清空父级 `PYTHONPATH` 后完整测试会因 `scripts` 不可导入产生 `51` 个 collection errors；现改为 `pythonpath=[".", "src"]`。干净环境下 observer + progress auditor targeted `31/31`，完整 `666 collected / 664 passed / 2 skipped / 0 failed / 0 collection errors`。该配置只在本地 successor branch，不部署到 frozen 远端。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/pytest_pythonpath_reproducibility_hardening.json`。
- 50K 后继阶段已做只读 transition preflight：冻结 revision 的 path resolver 仅返回 fresh v3 CoFiTok/dense run 与 `imagenet256_10pct_fixed_basis_matched_50k_v3/promotion_gate.json`，不引用历史 v2。completion pipeline、post-eval runbook、path resolver、gate builder 的本地/远端 SHA256 全部匹配；正式 runbook 均先 `cd "$PROJECT"` 并 `export PYTHONPATH=src`。协议保持双方法各 10K EMA DDIM-100、CFG `1.5`、bf16、共享 batch selection、1,024-image t=500 mechanism eval、prefix/visual audit，以及 FID/endpoint 相对回退 `<=5%`、绝对 FID `<=100`、ordered/zero/shuffle/coarse-energy fail-closed gate；gate 未显式通过时 parent pipeline 不得进入 full 300K。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/posteval_transition_preflight_after_step11000.json`。
- CoFiTok step 12K 后 dense transition readiness 已只读复核：权威 dense v3 目录尚不存在，故无 stale metrics/report/checkpoint/resume state；冻结 config validator 重算结果与 launch `config_pair.json` SHA256 完全相同，pair contract `pass`、共享 section/backbone `0` mismatch、dense auxiliary loss 为空、参数差 `0.017993%`。共享 runtime selection 仍为 `64x1`；CoFiTok/dense 真实 benchmark 均 completed 且未写 checkpoint，mean step `2.216154 / 2.020304s`、peak VRAM `56.18 / 55.39GB`，selection 使用较慢 CoFiTok score，不给 dense 不同 runtime。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/dense_transition_readiness_after_step12000.json`。
- 最终 formal evaluation 与 release chain 已静态预检：full post-eval、EMA export、artifact verifier、comparison builder、terminal auditor 的本地/冻结远端 SHA 全部一致，inference/gate/completion targeted `120/120`。full 300K 后必须双方法各 exact 50K EMA DDIM-250、CFG `1.5`、bf16、共享 sampling batch、1,024-image t=500 mechanism eval、prefix/visual audit；final gate 要求相对 FID/endpoint 回退 `<=5%`、绝对 FID `<=20`、precision/recall 各 `>=0.30` 且回退 `<=5%`，以及 ordered/zero/shuffle/coarse-energy。final gate 显式 pass 后才导出 step-300K EMA-only artifacts；每份 artifact 绑定 release gate、source checkpoint/integrity/runtime/training authorization，并必须通过真实 forward preflight、stable `infer_generation.py` smoke 和 RGB 256x256 PNG SHA。terminal auditor 会重开 exact 50K manifests/progress、stable inference API、matched environment、visual/comparison/gate/export/preflight/smoke 全证据后才允许 pipeline 写 `complete/pass`。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/final_release_chain_preflight_after_step12000.json`。
- 首个 checkpoint 后的 step-6,000 validation 出现不利波动：`validation_epsilon_mse=0.03626135`，较 5K/4K 高 `17.11%/23.42%`，但较 1K 低 `9.48%`。tracked progress audit 仍为 `healthy`、issues/warnings 为空、validation `6/6` 且 5K checkpoint integrity 保持 `verified`，因此这是质量诊断，不是日志、训练或可恢复性故障。不得隐藏该数值，也不得仅凭单点停止冻结的 50K 正式轨迹；继续由 10K/25K/50K 和 formal sample gate 裁决。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/cofitok_progress_step_00006000_diagnostic.json`。
- step-7K/8K/9K/10K fixed validation MSE 依次为 `0.03119027 / 0.02415371 / 0.02835677 / 0.03104844`，8K 为前十次最低。后五次均值较前五次下降 `11.15%`，但 10K 较 8K 最低点高 `28.55%`、较 5K 基本持平（`+0.28%`），因此轨迹仍属非单调健康诊断，不能替代生成样本 gate。10K tracked progress audit 为 `healthy`、issues/warnings 为空、validation `10/10` 且 logging complete；第二个 checkpoint 为 `1,006,351,466` bytes / SHA256 `1310dffc22b927946b0fd402c3abdc01623a710fa50b4176374c5168052e33e3`，sidecar、`latest.json` 与 progress report 十个核心字段 `0` mismatch，integrity `verified`。这只证明第二个恢复信任边界；双 50K 与 formal 10K-sample promotion 仍未完成。报告：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step10000_validation_and_checkpoint.json`。
- step-15K 第三个 formal checkpoint 已通过独立 tracked audit：`healthy`、issues/warnings 为空、validation `15/15`、required checkpoints `[5000,10000,15000]` 全部存在，checkpoint/sidecar/`latest.json` integrity `verified`。权重为 `1,006,351,466` bytes / SHA256 `29ad6fc611b93b431e9fdb9c589107edfd6254d044b115f8f4a3234b4e0e42d0`，仍只保留在服务器。11K-15K validation MSE 为 `0.03187243 / 0.03147752 / 0.02831916 / 0.03288554 / 0.03080699`；后五次均值较前五次低 `8.59%`，15K 较 10K 低 `0.78%`，属于有限但非单调的训练健康信号。CoFiTok 当前完成 `15,000/50,000`、`960,000` images seen，dense v3 尚未启动；仍须按冻结队列完成双 50K、formal 10K promotion、full matched 300K、双 formal 50K、final gate、EMA export 和 terminal audit。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step15000_validation_and_checkpoint.json`。
- step-20K 第四个 formal checkpoint 已通过 corrected tracked audit：`healthy`、issues/warnings 为空、validation `20/20`、integrity `verified`，权重为 `1,006,351,466` bytes / SHA256 `856d0b6af13d56de7438b364290fa3eac93084fb3262a93e8c99d4d1d37e57c4`。10% frozen config/run manifest 明确 `keep_last_checkpoints=3`、`protected_checkpoint_steps=[]`，故当前 `10K/15K/20K` 是正确滚动集合，5K 被正常清理；首次 caller 错把 5K 继续列为 required 而得到 `invalid`，复核合同后已用正确集合重跑并覆盖为 `healthy`，这不是训练或 checkpoint 故障。16K-20K validation MSE 为 `0.03343919 / 0.03167019 / 0.03097798 / 0.02683829 / 0.03015572`，后五次均值较前五次低 `9.93%`。CoFiTok 已完成 40%、`1,280,000` images seen；25K 由现有 long-horizon observer 独立审计。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step20000_validation_checkpoint_and_retention.json`。
- 2026-07-25 18:53 CST 用户要求暂停后，已按精确 PID/cmdline 边界向 completion supervisor、completion pipeline、matched-pair runbook、pair monitor、long-horizon observer 和 training watchdog 发送 `SIGTERM`；18:54 CST 复核 CoFiTok 相关进程为 `0`、GPU compute process 为 `0`，dense v3 尚未启动，远端仓库仍 clean/frozen 于 `58d83bfce2770eab2565b8c89a5f9a06201a0c86`。observer 已在暂停前独立通过 25K：audit `healthy`、issues/warnings 为空、validation `25/25`、rolling checkpoints `[15K,20K,25K]`、25K checkpoint `1,006,351,466` bytes / SHA256 `5b056311d7651f3a222ce10b5bdd1a1652b4446f0a88f26185ef89ab09a7a542`、integrity `verified`。停止传播期间 metrics 到 25,350，但未写 signal checkpoint/training report，故权威恢复点保持 25K；后 7 行必须由现有 `--resume auto` reconciliation 归档后精确续训。当前大规模生成状态是 `paused`，不是完成；不得直接启动 dense、promotion 或 full 300K。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/step25000_validation_checkpoint_and_pause.json`。
- step-1,000 后 terminal completion auditor 已用 source `781a014`、10%/full `58d83bf` 精确运行，按设计 nonzero 返回 `in_progress`、`0 failed`、`0 warnings`、`17 missing`；唯一终局 pass 是 `controlled_revision_transition`，证明新版 receipt/revision 被终局审计接受，同时双 50K、promotion、full 300K、formal 50K、EMA export/final gate 等仍不会被误判完成。快照：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/completion_audit_in_progress_after_step1000.json`，`3,635` bytes、SHA256 `ab4c258638e300e41b575e46cc665cbaa30e0dd5b99fc1d81a3d919559ffbefc`。
- formal 10% post-eval 的同参数 early storage preflight 已通过：20,256 samples、256 KiB/sample、额外 16 GiB、32 GiB safety margin 共要求 `56,849,596,416` bytes；实测 free `408,803,782,656`、headroom `351,954,186,240` bytes，clean revision `58d83bf`。证据：`CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/launch_2026-07-25/storage_preflight_early_after_step1000.json`。这只排除当前容量风险；双 50K 完成后 runbook 仍必须重新生成权威 `storage_preflight.json`，此前 terminal `generation_storage_capacity` 保持 missing。
- promotion gate 保持 fail closed，不得弱化或绕过：CoFiTok FID 相对 dense 回退必须 `<=5%`、绝对 FID `<=100`、endpoint MSE 回退 `<=5%`，并同时通过 ordered rank、coarse energy、zero/shuffle、checkpoint integrity、同 revision/环境/采样/评估 provenance。只有新 v3 gate 明确 `pass/promote` 才允许 full ImageNet-256 matched 300K；否则 supervisor 应在 `promotion_gate` 阶段终止并保留失败证据。

## Formal v3 10% promotion 验收终态（2026-07-29）

- formal v3 CoFiTok K8 与 dense identity 均已在
  `scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`
  完成精确 `50,000/50,000` steps、`3,200,000` images seen；matched pair
  contract 通过，参数为 `62,836,011 / 62,824,707`，相差 `+0.017993%`。
- 双方法各 10,000 张 EMA DDIM-100、CFG 1.5、bf16 正式样本与 metrics
  均完成，六份 gate 源报告的 path/bytes/SHA、同 revision、同 real set、同
  evaluator/runtime、sampling manifest/progress/checkpoint provenance 全部验证通过。
- promotion gate 权威终态为 `status=fail / decision=hold`。CoFiTok FID
  `226.4845`，dense FID `117.1660`，相对退化 `+93.30%`，同时超过绝对
  FID `<=100` 门槛；失败项仅为 `fid_within_tolerance` 与
  `absolute_fid_quality`。不得绕过或弱化 gate。
- 机制诊断仍通过：CoFiTok endpoint MSE `0.0164613`，相对 dense
  `0.0163134` 仅退化 `+0.91%`；ordered path 在 18 个顺序中排名第 1，
  coarse-token energy ratio `8.28%`，`S(0)=0` 精确成立，shuffle mismatch
  ratio `124.21x`。这证明 ordered factorization 存在，但不能证明生成系统可用。
- 固定视觉面板显示 CoFiTok 存在严重逐像素高频残留；prefix 1/2 主要为低频绿色
  场，prefix 4 出现规则彩色纹理，prefix 8 才出现弱语义但噪声仍重。前六个
  coarse token 只承担 `8.28%` 能量，最后两个 token 集中 `91.72%`，应优先
  诊断多步误差放大与 tail-token 能量集中。
- completion supervisor 与 pipeline 已按设计在 `promotion_gate` 非重试阶段
  终止，GPU 已空闲。full ImageNet-256 300K、formal 50K、final gate、EMA
  export 均未启动；项目仍未完成“大规模生成系统”目标。
- 本地验收包：
  `CoFiTok-internal/artifacts/reports/generation/imagenet256_10pct_fixed_basis_matched_50k_v3/acceptance_2026-07-29/`。
  详细记录：
  `CoFiTok-internal/docs/records/2026-07-29_generation_10pct_promotion_gate_acceptance.md`。

## Rollout 稳定性修复状态（2026-07-29）

- formal v3 的失败已定位为两个耦合问题：t=500 预测能量有 `91.72%`
  集中在最后两个 token；自由 DDIM rollout 中 CoFiTok/dense predicted-x0
  高频比峰值为 `10.032x`。小的 epsilon 高频差异在低 alpha 区间被放大，
  endpoint-only 评估无法暴露该问题。
- tail 修复保持 `S_k` 无条件、无偏置、线性且 zero-preserving：token
  channels/strides 为
  `[4,4,8,8,8,1,1,1] / [16,16,8,8,4,1,1,1]`，用三个独立 RGB
  full-resolution rank-1 tail bases 分散末端容量，并保留 capacity-blended
  stable-Hellinger energy prior 与低 SNR 高频项。
- 原 detached one-step `clipped_x0` 配方的 matched 5K 已双方法完成，但
  fail-closed multi-seed gate 未通过：n=8 reconstruction ratio
  `1.17100`，n=64 seeds 2029/2039 为 `1.03209/1.05198`；第二个 robust
  seed 擦线失败，CFG 1.0 仍失败，问题被定位为 free-state 多步漂移。
  决策为 `hold_for_stability_correction`，不得据此授权 50K。
- revision `6b77ef7254356d551b2e392aba07df1c96ed067c` 实现 detached two-step
  rollout consistency：`unroll_steps=2`、`batch_fraction=0.125`，每层都有
  bounded `clipped_x0` loss，生成状态和前一层 epsilon 均 detach；CoFiTok
  与 dense 通过 pair contract 共享全部 rollout 字段。
- fresh two-step matched 1K 已双方法精确完成。CoFiTok/dense validation
  ratio 为 `0.99117`；raw n=8 九门全 pass，reconstruction ratio
  `0.99446`、HF peak `0.90377`。n=64 seeds 2029/2039 也全 pass，
  reconstruction ratio 为 `0.97598/0.97530`，CoFiTok amplification 均低于
  dense。EMA 只作 1K 滞后诊断，不参与短探针 scaling 裁决。
- stage-aware 决策
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/scaling_decision1k_rollout_x0_u2_to_5k/scaling_decision.json`
  为 `pass / authorize_fresh_matched_5k`，SHA256
  `d47c2518e3c7fff18e8c4a9d6a2c605e1c875b9100a4641d9b8250285daac53b`；
  它明确不授权 50K 或 300K。
- fresh two-step matched 5K 已在隔离 checkout
  `/tmp/cofitok-generation-stability-u2-5k-2521d87`，训练 revision
  `2521d874a82898a7a2a527d824ea1df285df221d` 精确完成双方法各
  `5,000` steps / `320,000` images，输出
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2`，
  pair summary SHA256
  `ce604c2e9d2a1864bb6ba6fad31cdcee1cc92219256d361333abc00feaded28c`。
  CoFiTok/dense validation ratio `1.003083`，最终 checkpoint SHA256 分别为
  `47cfc77e...acbf4 / e7bf9d53...d4a35`；两边 1,250/2,500/3,750/5,000
  checkpoint 和 sidecar 齐全。正式远端仓库仍保持
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`。
- 上述队列 monitor 已终止为 `pass / complete / issues=[]`；权威报告和
  post-eval waiter 日志已归档到
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/`。
- CoFiTok 首个 `1,250/5,000` milestone 已通过只读验收：checkpoint bytes
  `1,006,321,322`、SHA256
  `8da625d6fb70dab7cab9a607029bd3c9efc9596b75a7acb53576872673836c3d`，
  sidecar 与 `latest.json` 在 filename/bytes/SHA/dataset identity/runtime
  identity/revision 上一致；canonical metrics 含 scheduled validation，
  `validation_epsilon_mse=0.03383226320147514`，rollout scale 已到 `1.0`。
- CoFiTok 第二个 `2,500/5,000` milestone 也通过同一只读完整性验收：
  checkpoint bytes `1,006,321,322`、SHA256
  `ad14dbd6ae47543fbe8d15a1ba86aed928b901bc97d0a15a776bdfefd74b1253`，
  validation MSE 为 `0.0368012972176075`，比 step 1,250 高 `8.78%`，而训练
  loss 仍下降。但每个 milestone 的 deterministic eval iterator 会前进到
  不同图像 batch，仅 noise/timestep stream 重置，因此该 `+8.78%` 不是同批
  学习曲线或泛化回退；有效比较是 matched CoFiTok/dense 的同 milestone
  batch 与 final free-state 评估。
- CoFiTok 第三个 `3,750/5,000` milestone 通过只读完整性验收：checkpoint
  bytes `1,006,321,322`、SHA256
  `6513d12b9131a0276eed465d51b8f6908197d309463ea4d8ef43f7690d0f675e`，
  validation MSE `0.022937312722206116`；1,250/2,500/3,750 三个恢复点均
  保留。该值属于第三个 deterministic 图像 batch，不作同批趋势解释。
- 后继 revision `9f435d4` 为未来训练补齐 validation provenance：每个
  validation row 显式记录 zero-based event index、实际 DataLoader batch
  index、reset noise seed 和 image count；exact-resume 与训练审计测试
  `30/30` 通过，全套本地测试通过且仅 3 个既有 skip。该提交不部署到 active
  `2521d87` checkout，当前 matched 5K 继续保持单 revision。
- 后继 revision `0cecd31` 将稳定推理协议下沉到
  `GenerationSession.generate()`：每个结果直接绑定 API v1、
  `cofitok_ddim_sampling_v1`、实际 DDIM timesteps、CFG/clipping/precision
  与 per-request-seed random-stream 语义，并在返回前运行 protocol contract。
  相关测试 `25/25`、全套本地测试通过且仅 3 个既有 skip；记录：
  `CoFiTok-internal/docs/records/2026-07-30_generation_session_protocol_identity.md`。
  该提交同样不部署到 active `2521d87` checkout。
- 后继 revision `3da40a3` 增加生产 fail-closed 模式：
  `require_release_authorization=True` / CLI
  `--require-release-authorization` 会在模型反序列化前拒绝训练 checkpoint 和
  未发布开发 artifact，仅接受绑定有效 full-stage release authorization 的
  inference artifact；相关测试 `23/23`、全套本地测试通过且仅 3 个既有 skip。
  test-only revision `72b1c54` 以 `torch.load` 哨兵证明两类拒绝均发生在
  反序列化之前。
- 后继 revision `f6b5416` 把 release-only 从可选 API 接入 final export
  pipeline：CoFiTok/dense 的 artifact preflight 与 inference smoke 均强制
  release authorization，completion audit 缺少任一 policy bit 即失败。相关
  测试 `96/96`、全套本地测试通过且仅 3 个既有 skip；远端 `bash -n` 通过，
  runbook SHA256 为
  `dc8aa617a53c3e276e741e7e5701de830fe988b2b7ce00faa5884a1e6b89ce3b`。
- 5K 训练后复评入口
  `CoFiTok-internal/artifacts/runbooks/generation_stability_rollout_x0_u2_posteval5k_2026-07-30.sh`，
  SHA256 `f6358383cabd61c2215dea748014b8d93e189d57c5690e307c4f34078f0a64cd`。
  已由完成后的 pair summary 解锁并按协议停止在 raw n=8：仅
  `predicted_x0_high_frequency` 失败，peak ratio `2.05125`；raw
  reconstruction/endpoint/validation ratio 为 `0.93844/1.00411/1.00308`，
  tail-two/max-token energy 为 `0.56703/0.28473`，其余八门全过。
  qualification SHA256
  `c4603cbcdfad7537e8a53d9cb5493cd0896eb17444a9cf04d7ef1362bf42f6a1`。
  因为存在非 reconstruction 失败，raw n=64 和 scaling decision 均未生成。
- 5K post-eval 有界等待器 tracked source 为
  `CoFiTok-internal/artifacts/runbooks/generation_stability_rollout_x0_u2_posteval5k_waiter_2026-07-30.sh`，
  SHA256 `17dc6a11482aa27e6b0518821708e03668df52dd2536726522adef487beb91c0`，
  已正常观察 monitor pass、绑定 summary SHA 并运行 post-eval 后退出。
- EMA n=8 显示 CoFiTok HF 低于 dense 且 reconstruction 更好；随后完成
  diagnostic-only EMA n=64 seeds 2029/2039，runbook
  `CoFiTok-internal/artifacts/runbooks/generation_stability_rollout_x0_u2_ema_n64_diagnostic5k_2026-07-30.sh`，
  SHA256 `091a2ca9dd4b4422d55eeb25d626fb79e0498ab47a690a7e4906d30f389a98cc`。
  两个种子的 peak HF ratio 为 `0.36133/0.33559`，reconstruction ratio 为
  `0.96585/0.96539`，诊断阈值全过；summary SHA256
  `78e14cd337ea43a24d621f04544a159b933bcc981c1ef82c49f7de7ab2ac7046`。
  其 summary 永久写 `scaling_authorization_allowed=false`，不得替代 raw 失败。
- immutable 5K 轨迹的 raw milestone 只读诊断已完成；runbook
  `CoFiTok-internal/artifacts/runbooks/generation_stability_rollout_x0_u2_raw_milestone_diagnostic5k_2026-07-30.sh`
  SHA256 为
  `e3ba897cb2b2f16bde0fe19bd6b8d47807ffce6318eb0115b9f7857eaf8ccff0`。
  它按同一 n=8/seed 2029 协议评估 1,250/2,500/3,750/5,000 raw
  checkpoint，peak HF ratio 为 `1.27978/0.69352/0.84662/2.05125`，
  reconstruction ratio 为 `1.09488/1.01795/0.84107/0.93844`；HF gate
  仅在最终 5K 首次越界，说明问题局限于 3.75K 到 5K 的 raw 权重末段漂移，
  不是单调 tail-capacity collapse。summary SHA256
  `e741e84b23c090e219b10b36ba0d403d067a9a30d567ef2aa0b80d1f3dbcf9ec`，
  且永久禁止授权 scaling。
- 后继 revision `10f2f6bd9977fb1a63de4b2939ca641107a0ccaa` 增加 matched
  late-stage EMA-teacher output consistency，只在训练 raw 模型后段约束其贴近
  稳定 EMA 路径；teacher 无梯度、eval-mode、只看同一 noisy input，不进入
  `S_k`。weight/start/warmup/batch fraction 被 generation-pair contract
  强制要求 CoFiTok/dense 精确一致。首个 CUDA rehearsal 因
  `persistent=False` 固定 synthesis buffer 与 `functional_call(strict=True)`
  不兼容而在 checkpoint 前失败；修复后显式要求 EMA 与 `state_dict()` keys
  完全一致，仅允许非持久固定常量使用 live module 值。
- active-teacher CUDA benchmark v2 已通过：CoFiTok/dense 为
  `22.1245/24.3071 img/s`、峰值显存
  `15,232,468,992/15,036,313,600` bytes，teacher scale 均为 `1.0`、
  loss 有限正值、checkpoint-free、matched contract 有效；summary SHA256
  `e5a88a1e7d18bef30de56ee43b46e94447c76302a84c0aa09d0928495fbaab49`。
  该结果只授权 fresh matched 1K probe，不授权 5K/50K/300K。
- fresh matched EMA-teacher 1K 已在隔离 checkout
  `/tmp/cofitok-generation-stability-ema-teacher-10f2f6b` 启动；输出
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair1k_rollout_x0_u2_ema_teacher`，
  runbook/trainer/monitor/post-eval waiter PID 分别为
  `853818/853839/854173/854722`。权威实时状态读取
  `.../pair1k_rollout_x0_u2_ema_teacher_monitor.json`；初始为
  `running / cofitok_training / issues=[]`。训练 runbook SHA256
  `4b2d5daa78d819edce64eb530506dc7a9eba323cfe9077d3378ce43b0a9def6c`。
- n=8 post-eval 与 waiter SHA256 分别为
  `dedef29410b6d76c91fc0a626e9f2e1f05e062cf6782d926e8082b8452a434a5` /
  `48eef1da04914e9cf86e19f6e8b9021bf1c2942e85a3fb522edf995908a9cfff`。
  waiter 只在 monitor pass、训练与 runbook 退出、GPU 空闲和 pair/checkpoint
  SHA 全绑定后运行；screening 永久禁止直接授权 5K，仍需后续 raw n64 双种子。
- raw n64 双种子 runbook/waiter SHA256 分别为
  `04ff2aa80762a0e4a25c0608520ef4740c54ce49192d64380321141ee4ddf919` /
  `ceae6c4acdfec5c35d62d09c573f37a8a0cd50f1546206334365e71cdcb765d5`，
  waiter PID `856046`。它在 n=8 fail 时不使用 GPU；只有 n=8 pass 才运行
  seeds 2029/2039，并由统一 builder 形成 `matched_5k` pass/hold 决策，
  不会自动启动 5K。
- EMA-teacher 1K 的 CoFiTok step-250 checkpoint 已通过只读完整性验收：
  bytes `1,006,321,770`、SHA256
  `f78653e9e6441b7b14d6067181b5529fecc5aef33ea417a65c117631c42cb1cf`，
  validation MSE `0.07069774717092514`；sidecar/`latest.json` 的
  filename/bytes/SHA/revision/dataset/runtime identity 一致，teacher scale
  仍为 `0`、rollout scale 已为 `1`。验收未加载或重哈希权重本体。
- 同一运行的 step-500 checkpoint 也已通过只读完整性验收：bytes
  `1,006,321,770`、SHA256
  `09ab41200dcb4a783358c5705d455ba0fa7756db2204e40114badfcc3411fcd0`，
  validation MSE 降至 `0.042676351964473724`，canonical row 为
  `32,000` images；sidecar/`latest.json` 继续完全一致。step 500 仍早于
  teacher 的 step-600 启动边界，因此 teacher scale 为 `0` 符合预期。
- active-teacher 路径已在真实训练循环中验收：step 625/650 的
  scale 分别为 `0.08333334/0.16666667`，teacher loss
  `0.15508846/0.15917403`，均为有限正数；GPU memory 约
  `23,515 MiB`，monitor 仍为 `issues=[]`。step-750 checkpoint bytes
  `1,006,321,770`、SHA256
  `e50371e8bca0843d115826382200c2b651dc7dfb496a6c2cf17686fdd3ed7708`，
  teacher scale `0.5`，validation MSE 继续降至
  `0.029762834310531616`；sidecar/`latest.json`/stat/identity 全一致，
  未加载或重哈希权重本体。
- EMA-teacher 1K 的 CoFiTok 成员已精确完成 `1,000/1,000`、
  `64,000` images、41 个 canonical rows 和 4 次 validation；step
  900/925/1000 teacher scale 均为 `1.0` 且 loss/gradient 有限，最终
  validation MSE `0.026219427585601807`。最终 checkpoint bytes
  `1,006,321,770`、SHA256
  `85c61f83a333f330dde42ab1df1f3eb462788164caf05950327f590ab732695b`，
  sidecar/`latest.json`/stat/report identity 一致；峰值训练显存
  `15,234,796,032` bytes。matched dense 已由原 runbook 自动启动，
  初始 trainer PID `862754`；实时状态仍只读同一 monitor。
- matched dense step-250 checkpoint 已通过只读完整性验收：`16,000`
  images、bytes `1,006,120,150`、SHA256
  `ca4386e700e4549b28f5ca50d8ad10552a7a924b05a173078302f6c181f73e94`、
  validation MSE `0.061262041330337524`；sidecar/`latest.json`/stat 的
  revision/dataset/runtime identity 全一致，teacher scale 在 step 600
  前仍为 `0`。未加载或重哈希权重本体。
- matched dense step-500 checkpoint 同样通过：`32,000` images、bytes
  `1,006,120,150`、SHA256
  `9567e4c5a0be80d7c453161a3cd6daff1f3bbe072b36248fb15c8c9bf2972da1`、
  validation MSE `0.042720943689346313`；sidecar/`latest.json`/stat 与
  revision/dataset/runtime identity 全一致，teacher scale 仍为 `0`。
  同步 step 的 CoFiTok/dense validation MSE 很接近，但不替代外部 raw gate。
- dense active-teacher 路径也已验收：step 625/650 teacher loss
  `0.14063581/0.14415441` 且 gradient 有限；日志 scale
  `0.08349609/0.16699219` 是 dense bf16 output 对 `1/12`、`1/6` 的最近表示，
  调度函数和配置没有步数偏移，teacher MSE 明确为 float32。step-750
  checkpoint bytes `1,006,120,150`、SHA256
  `7036abd8076ea09d2ba3710d729da5fc35b78508a604c3fb74d9f05d3b6e2231`，
  validation MSE `0.02896382473409176`、teacher scale `0.5`；
  sidecar/`latest.json`/stat/identity 全一致，未加载或重哈希权重本体。
- matched dense 已精确完成 `1,000/1,000`、`64,000` images、41 rows 和
  4 次 validation；最终 MSE `0.025760110467672348`，checkpoint bytes
  `1,006,120,150`、SHA256
  `a0494f10d5f36cef707654b9bd5406d464ee1b894f32f302cfa7e4a9a2614aa2`，
  peak VRAM `15,043,471,872` bytes、参数 `62,824,707`。pair monitor 已
  `pass/complete/issues=[]`；pair summary SHA256
  `94e857f0b537a03213e57079ef5a333d6508eaacbcdd067f85e71dbe0077f083`，
  contract valid、参数差 `+0.014924%`、validation ratio
  `1.0178305569964803`。自动 raw n=8 已启动；这仍未授权 5K。
- EMA-teacher matched 1K 的 authoritative raw n=8 已全门槛通过：
  peak HF `0.74367630`、reconstruction `1.04567489`、endpoint
  `1.00346327`、validation `1.01783056`、tail-two `0.56783742`、max token
  `0.28732264`、ordered rank `1`、shuffle ratio `82.2481`、zero `0`。
  qualification SHA256
  `c719b1daef242157cb968bf0d2f49995f64ca6368bb8f8d355e12b96673ee651`，
  screening summary SHA256
  `ccb70d103c9c565591c20c671e7b26c963cbdd830783856c7f842125c22c63ef`。
  n=8 summary 仍禁止直接授权 scaling；raw n64 seed 2029 已自动启动。
- 1K training/monitor/evaluation/qualification 的 29 个小文件已同步到本地
  `artifacts/reports/generation/stability_probe_2026-07-29/`，源/本地
  SHA256 审计为 missing/extra/mismatch 全 0，未复制任何 `.pt`。
- EMA-teacher matched 1K raw n64 双种子已全门槛通过：seed 2029/2039
  peak HF `0.74200056/0.72915726`，reconstruction
  `0.98314020/0.98015742`，endpoint `1.00346327`、validation
  `1.01783056`；qualification SHA256 分别为
  `8467e47d0f406fe954fe3c5d6805b545a2221eb8660efdcd625c81a0e0cb44b6` /
  `9fd020d4fe9ac6e695cc68e954a40ae0d2bbd23b426928b9e37e15340dfa8088`。
  stage-aware decision 为 `authorize_fresh_matched_5k`，SHA256
  `7f2e6e691e26ef24f18d42e0f337229a35f2eec1a774e82f12141bb15dc48c9d`；
  它只授权 fresh 5K，不授权 50K/full。robust 9 个小文件已同步并
  9/9 SHA 对账，GPU 已空闲。
- fresh two-step 5K raw gate 已失败，因此不得准备或启动新的 formal 50K；
  必须先解决 raw high-frequency instability，并重新完成 fresh matched gate。
  详细记录：
  `CoFiTok-internal/docs/records/2026-07-29_generation_rollout_stability_diagnosis_and_probe.md`。
- EMA-teacher 5K 候选已固定在代码 revision
  `59db142fc45d69dc92bb0333be5ac2d0162d9dc4`：shared rollout/teacher schedule
  scalar 统一为 float32，matched 配置均为 `5,000` steps、teacher
  `weight=0.25/start=3000/warmup=1000/fraction=0.0625`，覆盖旧 raw 漂移的
  3.75K 到 5K 窗口。隔离 Linux targeted `39/39` 和全套测试通过；官方远端
  repo 仍锁定 `1ebcc15210e63a776a2ba448481cbd8bb94a4066`。
- decision-bound EMA-teacher 5K CUDA rehearsal 已通过：CoFiTok/dense
  `22.1403/24.3586 img/s`、峰值显存
  `15,232,468,992/15,036,313,600` bytes，teacher scale 均为 `1.0`，
  loss/gradient/contract/checkpoint-free checks 全通过；summary SHA256
  `066a03a0bf9f6a7d40cd98de41468974d5d4e0d629a259c4a550026e6e399305`。
  它只允许 fresh matched 5K 运行，不授权 50K。
- fresh matched 5K 的训练、raw n=8+n64 post-eval 和 monitor-bound waiter
  runbook 已分别固定 SHA256
  `64177647c29227cabe9ababbcb11bf59554371c5391046c59f89122e9bbd8c84`、
  `a0935058edbd1d97c9e2519d97efeb6bed1f589086e6c89ce73ef97b1d07bd08`、
  `913ffc8c90f262ba4db845fab2043e030376395b62d68f381c1391c95ad2e49c`。
  5K pair summary 永久禁止直接授权 scaling；formal 50K 仍必须等待新的 raw
  n=8 与双种子 n=64 决策。
- fresh matched EMA-teacher 5K 已在隔离 checkout 启动，输出为
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher`；
  runbook/trainer/monitor/raw-gate waiter 初始 PID 分别为
  `988582/988641/988609/988625`。实时权威状态读
  `.../pair5k_rollout_x0_u2_ema_teacher_monitor.json`。首个已验收快照为
  `running/cofitok_training/issues=[]`，step 50、3,200 images、total
  `1.31013405`、grad `11.8828`、rollout scale `0.050000000745`；
  teacher 在 step 3,000 前为 0 符合配置。waiter 只会在 pair
  `pass/complete` 后运行 raw n=8 和双种子 n=64，不会自动启动 50K。
- 后续正式 pair monitor 在 revision
  `b7d87158e4fa20ce06988262aff6bfe3ac0ae509` 增加
  `--checkpoint-integrity-policy required`：只读验证稳定 milestone 的
  sidecar 与 `latest.json` filename/bytes/step/declared-SHA 绑定，明确不加载、
  不重哈希 checkpoint payload，并在 grace window 后 fail closed。10% matched
  50K 与 full 300K runbook 已启用；本地全套测试和隔离 Linux 定向测试通过，
  记录见
  `CoFiTok-internal/docs/records/2026-07-31_generation_pair_monitor_checkpoint_metadata_integrity.md`。
  当前 5K 仍固定 `59db142`，未热切换 monitor。
- follow-up revision `fcd2434b3821ed5f2f4501da68a9bb09debafd37`
  修复多 milestone grace 切换：新 checkpoint 刚出现时，`latest.json`
  不再与上一稳定 checkpoint 错配；grace 内记为
  `pending_checkpoint_grace`，超过 grace 后仍 fail closed。新增边界测试、
  本地全套与隔离 Linux 定向测试均通过。active 5K 未热切换，step 400
  仍为 `running/issues=[]`。
- revision `e22e1784ee5040167eeff029a357334990ebcc33` 进一步要求正式
  monitor 将 sidecar 的 `git_revision/git_dirty` 绑定到预期 clean revision。
  后续 revision `63af805b7ec1fe622c7a58aa65ed1d136097142e` 又将
  `run_manifest.json` 的 target steps、checkpoint interval、clean Git revision、
  runtime environment SHA、formal dataset identity 及逐行 float32
  rollout/EMA-teacher schedule 纳入 required monitor；revision
  `164c96e71dd97f0ac85a4f906c5a86cf768b7390` 进一步对缺失/non-formal
  dataset provenance、非对象 JSONL 和非法 scale 类型 fail closed，禁止解析异常
  杀死 monitor。当前 5K 的替换只读 observer PID 为 `38141`，显式期待训练
  checkpoint revision `59db142`；
  报告在
  `.../pair5k_rollout_x0_u2_ema_teacher_provenance_observer_164c96e.json`。
  首次 step 925 为 `running/issues=[]`，38 行 schedule 全匹配且 manifest
  `verified`；旧 PID `37643` 已在替换观察器验收后安全停止。它仍不参与
  authoritative monitor/post-eval waiter，也不加载或哈希 checkpoint payload。
- fresh EMA-teacher 5K 的 CoFiTok step 1,250 protected checkpoint 已通过：
  `80,000` images，文件 `1,006,321,770` bytes，declared SHA256
  `f234c142aaa39e21520bcf8ec2d3458feeef85c26c8431209dfab62d4a9610c9`；
  sidecar/latest 均为 `metadata_verified`，manifest `verified`，
  `health_issues=[]`。该行 total/epsilon 为 `0.06018657/0.03591859`，
  rollout scale `1.0`、teacher scale `0.0`、validation epsilon MSE
  `0.03305597`。独立验收时训练已到 step 1,600；这只证明首个恢复点和
  rollout 稳定性，不授权 formal 50K。
- stability 后继 50K 已完成“代码准备但未授权”：revision
  `c6075159eb04fc855ba4df8bb56e9fb0463fef12` 增加 recipe v4 的
  `stability_scaling/stability_full`、`rgbtail3` matched 50K 配置和会重哈希并
  重建 5K 决策的 validator；revision
  `c8e3def25d179ffe325b329a2d9685302be62f1f` 增加 gated runbook 与 pair
  summary。CoFiTok/dense 参数为 `62,834,083/62,824,707`，差
  `+0.014924%`。增量 bundle `19,223` bytes / SHA256
  `dd2fc45d0eb3835eb1cd219b713abe72879aba37135ef20c1f98d0c7cc1f6d33`
  已在隔离 Linux checkout 通过 targeted `35/35` 与 tracked runbook
  `89/89` syntax；未移动官方 repo、未修改 active 5K、未启动 50K。
  runbook 必须拿到最终 5K decision SHA 才能运行，pair summary 仍写死
  `formal_300k_authorization_allowed=false`。记录：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_scaling_50k_preparation.md`。
- fresh EMA-teacher 5K 的 CoFiTok step 2,500 protected checkpoint 已通过：
  `160,000` images，`1,006,321,770` bytes，SHA256
  `bc9e7942f9fac143064a901a78f33ddf7bd638c8df4d2409481e1305a794d6df`；
  sidecar/latest 与两个 protected checkpoint 均为 `metadata_verified`，
  manifest、formal dataset、runtime environment、revision 和 101 行 float32
  schedule 全部 verified，issues 为空。该步 total/epsilon
  `0.04665259/0.02514767`、validation MSE `0.03686439`、rollout scale `1.0`、
  teacher scale `0.0`；teacher 从 step 3,000 才启动，下一关键恢复点为 3,750。
- revision `2c2c1f5166b73d4f28df93b276901671ac1a7836` 修复 stability
  promotion gate 的 coarse/tail 定义：`rgbtail3` 由 spatial strides 推导
  tokens 1-5 为 coarse、6-8 为 full-resolution tail；缺失或非法 stride
  metadata 时 fail closed，不得退回旧 v3 的 `K-2`。同时新增独立
  `stability_scaling` source profile 和 dormant formal EMA DDIM-100 10K
  post-eval runbook；它会重建 5K 决策与 50K pair summary，但不能启动 full。
  增量 bundle `15,953` bytes / SHA256
  `2c702accc08ade80bbe1be1e0ea9441ddad2d6a2f2f8c06796ecfac0dfccf9a6`
  已在隔离 Linux checkout 通过 targeted `62/62`、runbook `90/90`
  `bash -n`，未部署、未启动任何后继运行。记录：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_true_coarse_energy_gate.md`。
- step 2,500 的 observer/sidecar/latest/run-manifest 小型证据已同步到
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/milestones/step_00002500/`；
  `SOURCE_MANIFEST.json` 的 4/4 bytes/SHA 对账通过，snapshot 时训练到
  step 2,675，latest protected checkpoint 仍为 2,500；未复制权重 payload。
- fresh EMA-teacher 5K 已在 step 3,025 观察到首次非零 teacher 事件：
  step 3,000 正确为 scale/loss `0/0`，step 3,025 的 teacher scale
  `0.02500000037252903` 与 float32 `25/1000` 一致，teacher loss
  `0.0055482672760263085` 有限非零；独立 observer 在 122 行 schedule、
  manifest、revision 和 checkpoint metadata 上均为 verified，issues 为空。
  事件证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/events/teacher_activation_step_00003025/`；
  这不是 checkpoint milestone，也不授权 50K，下一保护点仍为 step 3,750。
- fresh EMA-teacher 5K 的 CoFiTok step 3,750 protected checkpoint 已通过：
  `240,000` images，`1,006,321,770` bytes，SHA256
  `4906500378f77a8ba1af22836e7fd97bf30e89aa6bb908db58fdc120f2219a0c`；
  teacher scale `0.75`，validation epsilon MSE `0.02296529`，比同协议
  step 1,250 低 `30.5261%`、比 step 2,500 低 `37.7033%`。observer 对三个
  checkpoint、sidecar/latest、revision、dataset/runtime、manifest 和 152 行
  schedule 全部 verified，issues 为空。小型证据与 153 行 metrics snapshot
  位于
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/milestones/step_00003750/`；
  未复制权重，仍须 CoFiTok 5K、dense 5K 与 raw n8/n64 全部通过后才可授权
  stability 50K。
- fresh EMA-teacher 5K 的 CoFiTok 成员已完成 `5,000/5,000`、`320,000`
  images，最终 checkpoint `1,006,321,770` bytes，SHA256
  `cb432c75ccbc4eba00dab878e43dd0e45740ebdd9a6b95e0cd014ce97b6d6450`；
  final validation MSE `0.01861709`，比 step 1,250/2,500/3,750 分别低
  `43.6801%/49.4984%/18.9338%`。训练报告为 complete，峰值显存
  `15,238,400,000` bytes，observer 对四个 checkpoint、sidecar/latest、
  revision、dataset/runtime、manifest 和 schedule 均 verified，issues 为空；
  runbook 已自动进入 matched dense。小型证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/milestones/step_00005000_cofitok/`，
  未复制权重；dense 5K 与 raw n8/n64 未完成前仍不得授权 stability 50K。
- fresh EMA-teacher matched 5K 训练对已严格完成：两者均为 `5,000` steps /
  `320,000` images，strict observer 与 authoritative monitor 均为
  `pass/complete/issues=0`。CoFiTok/dense 参数
  `62,834,083/62,824,707`，差 `+0.014924%`；final validation MSE
  `0.01861709/0.01865597`，CoFiTok 低 `0.2084%`，wall time
  `14,222.48/12,996.33 s`，CoFiTok 高 `9.4346%`；endpoint 基本持平，
  不能单独作为质量主张。pair summary 已重验八个 milestone、sidecar、
  revision、images seen、共享稳定化字段和 dense factorization-only
  auxiliary 为零；小型证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/pair_training/`。
  post-eval 已自动启动，但 raw n8/n64 未通过前仍不得授权 stability 50K。
- fresh EMA-teacher 5K raw gate 已完成：n8、n64 seed 2029、n64 seed 2039
  均全 gate `pass`；ordered rank `1`、zero-token `0`、shuffle mismatch
  `119.1765x`、max single-token `28.4717%`、tail-two `56.7326%`、按 stride
  定义的 true coarse tokens 1-5 能量 `14.8341%`。两种 n64 的 CoFiTok/dense
  final reconstruction ratio 为 `0.974483/0.986974`，但 CoFiTok 自身
  amplification 仍为 `1.140490-1.158012`，高于 dense
  `1.112085-1.114014`；应写成“最终误差和能量集中已受控、误差放大被缓解但
  未消除”，不能写成全面解决。decision
  `d5a6fc017f20c7d024abfab1967ba6bc966b624e3a77e9246292ddaaf7dc1da4`
  已由 target `2c2c1f5` 重哈希、重建并通过，授权阶段仅为
  `fresh_matched_50k_preparation`。完整小型证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/raw_gate/`；
  官方 repo 未移动、GPU 已空闲、未启动 stability 50K。
- stability 50K 已完成 preparation-only 隔离预检，训练目标固定为
  `2c2c1f5166b73d4f28df93b276901671ac1a7836`；其后本地提交仅含 docs/
  artifacts，不改变 src/scripts/configs/runbooks/tests。远端独立 worktree
  `/tmp/cofitok-stability-50k-preflight-2c2c1f5` 使用分支
  `scale/generation-stability-50k-preflight`，decision 重哈希重建、recipe-v4
  `stability_scaling` config contract、存储检查均 pass；存储
  free/required/headroom 为
  `230,249,512,960/118,385,312,804/111,864,200,156` bytes，生成测试
  `746 passed + 2 existing skips`，runbook `90/90 bash -n`。4 个 AAAI
  layout 测试仅因独立 `/tmp` 布局没有 sibling `paper/` 而排除。收据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_preflight/`；
  GPU 最终空闲、50K 输出目录不存在，收据只允许 preparation，不允许据此
  启动 50K，更不授权 full 300K。
- matched stability 50K 已于 2026-07-31 10:59:34 CST 从独立 worktree
  `/tmp/cofitok-stability-50k-preflight-2c2c1f5` 实际启动，训练 revision
  固定为 `2c2c1f5166b73d4f28df93b276901671ac1a7836`，分支
  `scale/generation-stability-50k-preflight`；输出根为
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher`。
  runtime selector 在 `16x4/32x2/64x1` 中选定 `64x1`（effective batch 64）；
  首个权威 monitor 为 `running/cofitok_training/issues=[]`，step 100，
  manifest 与 rollout/EMA-teacher schedule contract 均 verified；同步的
  metrics 已到 step 150 / 9,600 images，rollout scale
  `0.014999999664723873`、teacher scale `0`，GPU
  `76,043/97,887 MiB`、99%。runbook/monitor/watchdog/trainer PID 分别为
  `315094/319121/319138/319202`；不得重复启动。小型启动证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_launch/`。
  当前只是 early-running receipt，首个 required-integrity checkpoint 为
  step 5,000；CoFiTok 50K 后才串行进入 dense 50K，之后仍须 formal EMA
  sampling 与机制 gate。full 300K 继续未授权。
- stability 50K post-eval provenance 人工交接点已修复于
  `caab51348d546e98858d1203f2958d9e396e2d18`：旧 gate builder 硬编码
  `scale/generative-system`，会让隔离 stability training/sampling/evaluator
  即使科学指标通过也必然因 branch provenance 失败；现在显式分离并绑定
  training identity `2c2c1f5 / scale/generation-stability-50k-preflight` 与
  evaluation identity `caab513 / scale/generation-stability-50k-posteval`，
  legacy 默认不变。远端 evaluation worktree
  `/tmp/cofitok-stability-50k-posteval-caab513` 已通过
  `756 passed + 2 skipped`、tracked runbook `91/91 bash -n` 且 clean；
  增量 bundle 为 `482,968` bytes / SHA256
  `08783631747bcc4eae186443f5c923a94afaa0c1b4098f675092314858b4075e`。
  绑定 waiter PID `442981` 已启动，初始状态
  `waiting_for_completed_training_pair`；它只在 exact monitor
  `pass/complete`、pair summary 完整且 GPU 空闲后运行 formal EMA 10K
  DDIM-100 post-eval，失败/停滞/identity 变化均 fail closed，不运行 full。
  注意 waiter `pass` 只代表 post-eval runbook 执行成功，不能替代
  `promotion_gate.json` 自身 `pass` 与完整 validator。记录与小型证据：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_50k_posteval_provenance_handoff.md`
  和
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_posteval_handoff/`。
- stability full 路径已完成 preparation-only 代码准备，但仍为 dormant：
  `d5019ea` 新增 full ImageNet-256 matched 300K 配置与交替
  50K/100K/200K/300K milestone runbook；`35208b8` 新增 `stability_full`
  source profile、正式 EMA DDIM-250 50K sampling、checkpoint mechanism eval、
  visual audit、final gate 与 schema-v5 strong comparison。CoFiTok/dense 参数
  `62,834,083/62,824,707`，差 `+0.014924%`。这些 runbook 只可在
  stability 50K promotion gate 精确通过后由新的隔离 training revision
  启动，不得由 waiter 自动运行。
- release-authorized 稳定推理导出已在 `cd88c6c` 准备：full gate SHA、
  clean export revision/branch、EMA-only artifact、CUDA preflight 和
  class/seed/prefix smoke 均为强约束；导出路径为
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/exports/stability_full_300k_ema_teacher`。
  该 runbook 不能训练，且 full final gate 未通过前不可执行。
- stability 专属 13 项 completion audit 已实现于 `aed2014`：
  `scripts/audit_generation_stability_completion.py` 重新哈希并绑定 5K
  qualification、50K monitor/pair/checkpoint/promotion gate、full 300K
  monitor/training pair/四个 protected milestones、物理 50K sample set 与
  ImageNet-256 real tree、正式 runtime/visual、final gate、matched strong
  comparison 和 release-authorized EMA inference。缺失证据只可为
  `incomplete`，存在错误为 `failed`，两者都不能宣称 generation ready。
  `infer_generation.py` 的 smoke report 现在同时记录执行 revision/branch
  与 runtime-environment SHA；completion audit 会拒绝 export/preflight/smoke
  执行身份漂移。
- follow-up `edb2a0dfeb5670974abe253c735d408630f84034` 进一步把 EMA
  artifact 导出动作本身的 clean Git identity 与 CPU runtime-environment
  SHA 写入 export report；release audit 要求 export、GPU preflight 和
  inference smoke 全部绑定指定 export revision/branch，并重算环境 SHA。
- `6af34e8641fc9cf4773e5c746739f3cb47b916a3` 再要求 CoFiTok/dense
  两个 release artifact 共用同一 export runtime environment，且两者的
  GPU preflight/smoke 共用同一推理环境；单方法内部自洽但跨方法漂移也会
  fail closed。
- 旧 `caab513` post-eval waiter 暴露 direct-entry import 缺陷：
  `build_generation_stability_50k_summary.py` 在 runbook 的
  `PYTHONPATH=src` 环境会找不到 `scripts` 包。修复后的隔离 checkout 为
  `/tmp/cofitok-stability-50k-posteval-aed2014`，HEAD
  `aed20142a496c3c16f9b2e8c8aba4a465fbaf4d7`，分支
  `scale/generation-stability-50k-posteval-v2`；Linux 排除 sibling-paper
  layout 的结果为 `777 passed + 3 skipped`，runbook `95/95 bash -n`。
  旧 waiter PID `442981` 已在确认 idle/no-child 后归档并停止；该阶段
  waiter PID `900875` 的状态为
  `waiting_for_completed_training_pair`，显式绑定 training `2c2c1f5` 和
  evaluation `aed2014`，且 `formal_300k_allowed=false`。切换前后 GPU
  进程均只有 trainer PID `319202`，正式 repo 仍固定
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`。切换后快照为 CoFiTok
  step `2,300` / `147,200` images、monitor
  `running/cofitok_training/issues=[]`。记录与证据：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_completion_audit_and_waiter_v2.md`
  和
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_completion_audit_and_waiter_v2/`。
- post-eval waiter v3 修复了 immutable training revision
  `2c2c1f5` 在双训练结束后可能因旧 direct-entry import 失败、无法生成
  `pair_summary.json`，进而让 v2 永久等待的问题。新 waiter 只在 exact
  monitor `pass/complete/issues=[]` 后，使用 clean evaluation checkout
  `/tmp/cofitok-stability-50k-posteval-08b67cc` 中绑定的 builder 重建缺失
  summary，并逐项复核 CoFiTok/dense training、decision validation 和 config
  validation 的 path/bytes/SHA256；已有 summary 也必须通过同样来源校验。
  Linux 结果为 targeted `15 passed`、隔离全量 `783 passed + 3 skipped`
  （786 collected）、runbook `95/95 bash -n`。v2 PID `900875` 已确认
  idle/no-child 后归档并停止；该阶段 v3 PID `25861`，绑定 evaluation
  `08b67cc58026adf3e052a3eeab48b4bc0834cfd4` /
  `scale/generation-stability-50k-posteval-v3`，仍为
  `waiting_for_completed_training_pair` 且 `formal_300k_allowed=false`。
  切换前后 GPU 进程均仅有 trainer PID `319202`，官方 repo 仍固定
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`。切换后训练读数为 step
  `3,000` / `192,000` images、validation epsilon MSE `0.03815802`，
  monitor `running/cofitok_training/issues=[]`。记录与证据：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_50k_posteval_waiter_v3.md`
  和
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_posteval_waiter_v3/`。
- 当前权威 post-eval waiter 已升级为 v4。提交
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289` 让
  `audit_generation_training_progress.py` 通过 `--config` 独立验证 rollout
  consistency 与 EMA-teacher consistency 的 scale/loss schedule，并要求已
  激活 schedule 至少存在非零 loss 行；stability 50K post-eval、full 300K
  training 与 full 50K post-eval 均绑定 CoFiTok/dense 各自 config。Linux
  隔离 rehearsal 为代码 suite `794 passed + 2 CUDA skipped`（排除 4 个
  sibling-paper layout test），真实父目录布局下 4 个 paper test 全通过，
  `95/95` runbook 通过 `bash -n`。对 active CoFiTok run 的只读 live audit
  在 step `6,700` 验证 135 行 metrics、6 次 validation、checkpoint integrity
  `verified`、rollout 135/135 active rows 非零且 expected scale `0.67`、
  EMA-teacher 在 step 30K 前保持 0，issues/warnings 均空。
- waiter v3 PID `25861` 已在确认 idle/no-child 后归档并停止；当前 v4 PID
  `281834`，绑定 evaluation
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289` /
  `scale/generation-stability-50k-posteval-v4`，仍为
  `waiting_for_completed_training_pair` 且 `formal_300k_allowed=false`。
  切换前后 GPU 进程均只有 trainer PID `319202`，正式 repo 仍固定
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`；active training checkout 与
  revision 未移动。记录与证据：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_consistency_schedule_audit_and_waiter_v4.md`
  和
  `CoFiTok-internal/artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_50k_posteval_waiter_v4_schedule_audit/`。
- dormant stability full 已在独立分支 `scale/generation-large-capacity`
  升级为真正 large-capacity matched pair，提交
  `fc6a077576e58357564a0d7e6ac39bdc548308b3`。仅
  `stability_full` 使用 `base_channels=256`；active
  `stability_scaling` 50K 仍保持 128 channels 与约 62.8M 参数。新 full
  参数为 CoFiTok `250,153,763`、dense `250,135,043`，差
  `+0.007484%`。runtime candidates 为
  `1x64,2x32,4x16,8x8,16x4`，显式 baseline `1x64`；storage
  preflight 用现有 50K checkpoint 乘 `4.0`，服务器实测 required
  `154,598,906,496` bytes、free `219,438,235,648`、headroom
  `64,839,329,152`，结果 pass。commit 的增量 bundle SHA256 为
  `f5e6beabdfcdf12a6ca4a52b501341da81e00a0dc35180124b2b743d9dae2bc7`；
  Linux 隔离验证为代码 `803 passed`、真实 sibling-paper 布局
  `4 passed`、runbook `95/95 bash -n`。GPU 仍被 active 50K
  trainer 占用，因此 250M CUDA feasibility 未执行，full 300K 仍必须等待
  source-bound 50K promotion gate `pass`，不得启动。验证时 active
  CoFiTok 为 step `8,450/50,000`、monitor
  `running/issues=[]`，正式 repo 仍固定 `1ebcc15`，waiter v4
  PID `281834` 无 child。详细记录：
  `CoFiTok-internal/docs/records/2026-07-31_generation_large_capacity_full_contract.md`。
- large-capacity 在隔离部署机制落地前曾形成候选 HEAD
  `2314a295a9747ec8be39615d613ba2e8f643568e`；其中核心两阶段实现提交
  `50145a9bf733b0fb7edf4a342883c43ea3fde307` 将 250M CUDA runtime
  qualification 与 300K launch 完全拆开。独立
  `generation_stability_ema_teacher_full_readiness_after_gate.sh` 只在
  source-bound 50K gate 通过、GPU 空闲且两份正式训练目录无状态时运行，
  验证 exact 250M pair、4x storage 和五候选 runtime，写 immutable
  `full_training_readiness.json` 后退出，绝不启动 trainer/monitor。full
  runbook 必须提供 readiness SHA，重放全部源/环境契约并写新鲜
  `storage_capacity_launch.json`，且不再调用 selector。
- completion audit 现在要求外部 readiness SHA，重开六份 source reports，
  可在后续 evaluation revision 和已有 training state 下重放，但 readiness
  内 training revision/branch/run paths/benchmark root 必须精确；launch-time
  storage 独立审计。实现与记录见
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_full_readiness_split.md`
  和
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_full_readiness_validation.md`。
- 该历史候选曾生成并验证 bundle
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-2314a29-from-1ebcc15.bundle`，
  bytes `39,345,914`，SHA256
  `b31a9617b88cdf2d186c23c605c03958ae3040cf59711a7a7165bda368261636`，
  advertised head 为 `2314a29`，prerequisites 为正式 repo 已持有的
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` 与
  `58d83bfce2770eab2565b8c89a5f9a06201a0c86`；已由正式 repo
  `git bundle verify`。该 `2314a29` 候选现已被下文
  `7bab083` 的 CPU-only isolated deployment、完整 receipt 和重放校验取代，
  不再是当前部署入口，也不应重新放回持久化 bundle 目录。旧
  `c1efb12`-only 小 bundle 同样因正式 repo 缺 prerequisite 已删除，
  不得再引用。
- 该候选阶段的 Linux 隔离 checkout
  `/tmp/cofitok-generation-large-capacity-readiness-50145a9/CoFiTok-internal`
  在真实 sibling-paper 布局、禁用 CUDA 下为 `814 tests / 0 failures /
  0 errors / 2 skipped`；全部 `96/96` tracked runbook 通过
  `bash -n`。验证后正式 repo 仍为 `1ebcc15` / `scale/generative-system`
  且 tracked clean，GPU 仍只有 trainer PID `319202`。active monitor
  `running/cofitok_training/issues=[]`，最近读数 CoFiTok
  `10,700/50,000`、dense `0/50,000`；waiter v4 PID `281834`
  存活且无 child。250M CUDA readiness 未运行，
  full 300K 当时仍未授权、未启动；这段只保留为 readiness split 的历史
  验证记录。`7bab083/15f1596` 后续也已被本节末尾的 `847658e`
  版本化部署取代，只作为不可覆盖的历史 evidence 保留。
- large-capacity 已补齐并真实执行 CPU-only isolated deployment。实现提交
  `7bab083b8d303c03d19074af4c1ed3200b1ad445`；当前分支文档 HEAD 为
  `15f1596750e6479fda8d72a3cb4865ac4211c263`，后者只新增真实执行记录，
  已部署代码/receipt 仍精确绑定 `7bab083`。专用 runbook
  `generation_deploy_large_capacity_readiness_checkout.sh` 不 fetch/merge/
  checkout 正式 repo，不运行 readiness/trainer，使用
  `CUDA_VISIBLE_DEVICES=-1` 在项目 checkpoints 下创建隔离 checkout，并将
  bundle、正式 repo、checkout、完整 JUnit 和全部 tracked runbook syntax
  绑定为 deterministic receipt。
- 已部署 checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-7bab083`；
  receipt：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployment_receipt.json`，
  SHA256
  `d07f4149376f8b6f757f607471dd9bac731d9927787e2e3d8526c72116205dda`。
  Linux 实跑为 `819 tests / 0 failures / 0 errors / 2 skipped`，
  `97/97` tracked runbook `bash -n`。receipt 明确
  `readiness_execution_allowed=true`、`readiness_executed=false`、
  `full_training_launch_allowed=false`、
  `formal_generation_completion_claimed=false`。
- 部署前后 GPU 均只有 trainer PID `319202`、约 `84,122 MiB`；
  正式 repo 始终为 `1ebcc15 / scale/generative-system` 且 tracked clean。
  当前 active monitor `running/cofitok_training/issues=[]`，CoFiTok
  `11,700/50,000`、dense `0/50,000`；waiter v4 PID
  `281834` 存活且无 child。服务器上没有
  `full_training_readiness.json`，也没有两份 full 300K run 目录。
  下一步只能在 source-bound 50K gate 通过且 GPU 空闲后，从上述
  `checkout-7bab083` 执行 readiness runbook，并同时提供 gate SHA 与
  deployment receipt SHA；readiness 通过后仍需单独人工授权 full runbook。
  记录：
  `CoFiTok-internal/docs/records/2026-07-31_generation_large_capacity_isolated_deployment.md`
  与
  `CoFiTok-internal/docs/records/2026-07-31_generation_large_capacity_isolated_deployment_execution.md`。
- 上述 `7bab083/15f1596` 固定槽部署现为历史 evidence，旧 checkout
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-7bab083`
  与固定 receipt SHA256
  `d07f4149376f8b6f757f607471dd9bac731d9927787e2e3d8526c72116205dda`
  均原样保留，不得覆盖，也不再作为新的 readiness 入口。
- 当前 large-capacity CPU-only 部署目标为
  `847658eb5bad1a888f6efe80690ab7ca5244329a`，仍在
  `scale/generation-large-capacity`。本次实现新增 immutable
  `full_training_launch_receipt.json`：首次 full 300K launch 前绑定
  readiness/gate/deployment receipt/config/runtime/launch storage/run paths；resume
  必须显式提供 receipt SHA，且只能更新
  `storage_capacity_current.json`，不得重写首次
  `storage_capacity_launch.json`。terminal completion audit 从训练 deployment
  receipt 绑定的 checkout 重放配置，不接受后续同名配置替换训练时源。
- 新增 large-capacity deployment evidence 按 full target revision 版本化：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/847658eb5bad1a888f6efe80690ab7ca5244329a/`。
  当前 checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-847658e`；
  receipt：上述 evidence 目录下 `deployment_receipt.json`，SHA256
  `3af2f2c4c1de7bcf4157d51e427b48797fe2b750d8cb8068b5cef2b24ed8e3dc`。
  receipt replay 已通过；Linux CPU-only 部署实跑为
  `827 tests / 0 failures / 0 errors / 2 skipped`，`97/97` tracked runbook
  `bash -n`。旧固定槽与新版本化 evidence 同时存在。
- 当前持久化 bundle：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-847658e-from-1ebcc15.bundle`，
  bytes `39,371,709`，SHA256
  `d47ed8a3ec354f0f077c4383dd5e428749d0c06fe1118193bac2dbf6b2ebb4f7`；
  advertised head 仅为 `847658e`，prerequisites 仍精确为
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` 与
  `58d83bfce2770eab2565b8c89a5f9a06201a0c86`，正式 repo 已完成
  `git bundle verify`。
- 部署和 receipt 重放后，正式 repo 仍固定在
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066 / scale/generative-system`
  且 tracked clean；GPU 仍只有 active trainer PID `319202`，约
  `84,122 MiB`。最新 monitor 为
  `running/cofitok_training/issues=[]`，CoFiTok `14,250/50,000`、dense
  `0/50,000`。服务器仍无 `pair_summary.json`、`promotion_gate.json`、
  `full_training_readiness.json` 或两份 full 300K run 目录。不得执行 CUDA
  readiness 或 full 300K；只有 source-bound 50K pair 完成、post-eval gate
  `pass` 且 GPU 空闲后，才能从 `checkout-847658e` 单独执行 readiness，之后
  仍需另行人工授权 full launch。
- large-capacity 自动衔接已继续推进到
  `5dd3488ac9b30274f4960195e252cc9fdb161002`（分支仍为
  `scale/generation-large-capacity`）。新增 fail-closed
  `run_generation_stability_full_readiness_waiter.py` 与
  `generation_stability_ema_teacher_full_readiness_waiter.sh`：它只在 50K
  post-eval waiter 成功、`stability_scaling` gate 为 pass、六份 source
  report 哈希及 training/evaluation revision 精确、deployment receipt 重放
  通过且 GPU 空闲时调用 250M readiness runbook。它不引用 full 300K
  runbook、不创建 launch receipt、不启动 `train_generation.py`，所有状态均
  固定 `full_training_launch_allowed=false`。记录：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_full_readiness_waiter.md`。
- 当前版本化部署 checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-5dd3488`；
  receipt：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/5dd3488ac9b30274f4960195e252cc9fdb161002/deployment_receipt.json`，
  SHA256
  `46c72437bf0e44ad7e815ad7b3bb3e29af0f7e1664ab177c031c58e18a0a0758`；
  receipt replay 已通过。Linux exact-commit rehearsal 与正式 CPU-only 部署均为
  `833 tests / 0 failures / 0 errors / 2 skipped`，`98/98` tracked runbook
  `bash -n`。旧 `847658e` 与固定 `7bab083` evidence 均原样保留。
- 当前持久化 bundle：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-5dd3488-from-1ebcc15.bundle`，
  bytes `39,379,637`，SHA256
  `086a4ef34504a746b10b9cef63ffccad97b1dbd03ed9cabe289a750c6315371a`；
  advertised head 仅为 `5dd3488`，prerequisites 仍精确为
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` 与
  `58d83bfce2770eab2565b8c89a5f9a06201a0c86`。
- readiness waiter 已在服务器后台启动，PID `622788`，状态路径：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_300k_ema_teacher/reports/readiness_waiter.json`；
  当前为 `waiting_for_passing_stability_gate`，已绑定上述 deployment receipt、
  training `2c2c1f5`、post-eval `c1efb12`，并观察到 GPU PID `319202`，因此
  不会提前运行 readiness。最新 50K monitor 为
  `running/cofitok_training/issues=[]`，CoFiTok `14,900/50,000`、dense
  `0/50,000`；正式 repo 仍固定 `1ebcc15 / scale/generative-system` 且
  tracked clean。50K post-eval gate 若 fail，waiter 会失败关闭；gate pass
  后只会自动完成 250M CUDA readiness，full 300K 仍需单独人工授权。
- large-capacity 后训练完成链已推进到
  `2f8da43a5643541ef547fee76c334bde0e4c3fcc`。审计时发现并修复了一个
  终局必失败点：`generation_stability_ema_teacher_completion_audit.sh`
  曾硬编码旧 scaling evaluation revision `caab513`，与当前 v4 post-eval
  `c1efb12` 冲突；现在 scaling training/evaluation revision 和 branch 均为
  显式必填输入，终局 audit 不再隐含历史 checkout。
- 新增 source-bound `run_generation_stability_posttraining_supervisor.py` 与
  `generation_stability_ema_teacher_posttraining_supervisor.sh`。它不授权、不
  启动 full training，启动前必须获得人工 full launch 后生成的 immutable
  `full_training_launch_receipt.json` 及 SHA256，并重哈希九份 bound sources。
  只有 exact 300K pair monitor 双方 `300,000/300,000` 且 GPU 空闲后，才会
  串行执行正式 50K DDIM-250 post-eval、strict full scientific gate、两份
  release-authorized EMA inference export/smoke 和 terminal completion audit。
  scientific gate fail、来源漂移或 completion `failed/incomplete` 均不可重试；
  仅 sampling/export 等执行失败有界重试。记录：
  `CoFiTok-internal/docs/records/2026-07-31_generation_stability_posttraining_supervisor.md`。
- 当前最新版本化部署 checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-2f8da43`；
  receipt：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/2f8da43a5643541ef547fee76c334bde0e4c3fcc/deployment_receipt.json`，
  SHA256
  `c76e7b2d069e35cd822210f377404bda3ad78f92176546dad12378aab94779d8`；
  receipt replay 已通过。Linux exact-commit rehearsal 与正式 CPU-only 部署均为
  `838 tests / 0 failures / 0 errors / 2 skipped`，`99/99` tracked runbook
  `bash -n`。旧 `5dd3488/847658e/7bab083` evidence 均保留。
- 当前持久化 bundle：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-2f8da43-from-1ebcc15.bundle`，
  bytes `39,406,053`，SHA256
  `526a65a8a2bf2e05af106388f5fb46ea2ac2bc0521d8148089c351b469526457`；
  advertised head 仅为 `2f8da43`，prerequisites 仍为 `1ebcc152` 与
  `58d83bfc`。该 checkout 只部署，post-training supervisor 尚未启动，因为
  full readiness/full launch receipt/full 300K 均尚未存在，不能越权预启动。
- 当前运行中的 readiness waiter 仍精确绑定 `5dd3488`，PID `622788`，未因
  新部署移动 revision；状态仍为 `waiting_for_passing_stability_gate` 且
  `full_training_launch_allowed=false`。最新 50K monitor 为
  `running/cofitok_training/issues=[]`，CoFiTok `15,450/50,000`、dense
  `0/50,000`，GPU 仍只有 trainer PID `319202`、约 `84,122 MiB`；磁盘 free
  `214,456,291,328` bytes。正式 repo 仍为
  `1ebcc152 / scale/generative-system` 且 tracked clean。
- large-capacity 完成链的当前本地 HEAD 为
  `f89bdc29af3ea66b9e12292b0c438abf75fcfd7d`，分支仍为
  `scale/generation-large-capacity`。提交 `8ec12de` 新增
  `scripts/run_generation_stage_once.py`，将 full post-eval、formal metrics、
  visual audit、final gate、comparison、EMA export、release preflight 和
  smoke inference 改为 source-bound restart-safe stages；提交 `f89bdc2`
  修复 isolated checkout 中 stage worker 的 bootstrap。stage receipt 绑定
  clean Git identity、命令/cwd、输入文件 bytes/SHA、输入树 digest 和完整
  输出集；可在父进程中断后通过独立 worker/result sidecar 恢复，来源或 replay
  漂移以退出码 `86` fail closed。实现记录：
  `CoFiTok-internal/docs/records/2026-07-31_generation_formal_stage_restart_safety.md`。
- `f89bdc2` 已完成 exact-commit Linux CPU-only isolated deployment。checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-f89bdc2`；
  bundle：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-f89bdc2-from-1ebcc15.bundle`，
  bytes `39,425,781`，SHA256
  `f43df99c293a6e20312a76e177dc4bdbeef2bbbb191a553d20a3f471e2730c06`，
  prerequisites 为 `1ebcc15210e63a776a2ba448481cbd8bb94a4066` 与
  `58d83bfce2770eab2565b8c89a5f9a06201a0c86`。deployment receipt：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/f89bdc29af3ea66b9e12292b0c438abf75fcfd7d/deployment_receipt.json`，
  SHA256
  `1869532cd5bcb7f4ea4cc549056579d897549b1bf8e040e430df32db582de293`。
  Linux 验证为 `849 passed / 2 skipped / 0 failures / 0 errors`，全部
  `99/99` tracked runbook 通过 `bash -n`；receipt 明确
  `readiness_execution_allowed=true`、`readiness_executed=false`、
  `full_training_launch_allowed=false`、
  `formal_generation_completion_claimed=false`。早期 `8ec12de` 部署尝试因
  relative `PYTHONPATH` 暴露 bootstrap 问题并在 receipt 前失败，只能视为
  immutable failed attempt，不得作为部署证据。
- 2026-07-31 最新只读检查：50K monitor 仍为
  `running/cofitok_training/issues=[]`，CoFiTok `17,950/50,000`、dense
  `0/50,000`，GPU PID `319202` 使用约 `84,135 MiB`，磁盘 free
  `202,454,462,464` bytes。正式 repo 仍固定
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066 / scale/generative-system`，
  tracked state 未改动；其既有 untracked 论文/实验 artifacts 必须保留。
  现有 readiness waiter PID `622788` 仍精确绑定 `checkout-5dd3488` 和 receipt
  `46c72437bf0e44ad7e815ad7b3bb3e29af0f7e1664ab177c031c58e18a0a0758`，
  状态为 `waiting_for_passing_stability_gate`，无 child，且
  `full_training_launch_allowed=false`。不得用 `f89bdc2` 替换或重启该 waiter；
  source-bound 50K gate 通过且 GPU 空闲前不得运行 CUDA readiness，full 300K
  仍必须等待 readiness 通过后的单独人工授权。post-training supervisor 已部署
  但尚未启动，当前不存在 full launch receipt，也没有 full 300K run。
- large-capacity 当前 HEAD 已推进到
  `b8ba139ae8710baedaa0e2f11d9bfa66f73909dc`。终局重放审计发现并修复了
  一个更窄的 parent-crash 窗口：父 wrapper 已启动独立 worker、但尚未把
  worker PID 写回 receipt 时若中断，旧恢复路径会在等待后继续使用内存中的
  stale `running` sidecar，可能把已成功输出归档并重复昂贵 stage。现在恢复时
  优先等待 sidecar 记录的 worker（worker 已退出时等待 command child），随后
  重新读取 atomic sidecar，再决定复用或归档。新增确定性测试覆盖该窗口；本地
  完整 pytest 通过。提交：`b8ba139 Recover live generation stage workers`。
- `b8ba139` 已完成新的 exact-commit Linux CPU-only isolated deployment。
  checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-b8ba139`；
  bundle：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-b8ba139-from-1ebcc15.bundle`，
  bytes `39,417,553`，SHA256
  `50ccd3b6df532936d35331d72b82034a1435ca5b0d24eebae1116a3b22e169dd`，
  advertised head 仅为 `b8ba139`，prerequisites 仍为 `1ebcc152` 与
  `58d83bfc`。deployment receipt：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/b8ba139ae8710baedaa0e2f11d9bfa66f73909dc/deployment_receipt.json`，
  SHA256
  `47a0909e0f23c6555693c0b580cf11e00ee4469060db0c1ad827b9fc1cd992f2`。
  Linux 验证为 `850 passed / 2 skipped / 0 failures / 0 errors`，全部
  `99/99` tracked runbook 通过 `bash -n`；receipt 仍明确
  `readiness_execution_allowed=true`、`readiness_executed=false`、
  `full_training_launch_allowed=false`、
  `formal_generation_completion_claimed=false`。正式 repo 在部署前后仍固定
  `1ebcc152 / scale/generative-system` 且 tracked clean；现有 `5dd3488`
  readiness waiter 不得替换，当前 50K gate/GPU-idle/readiness/人工 full-launch
  边界均不变。
- stage launch recovery 的最终加固 HEAD 为
  `b4fb415c51c608e68ea5e9484646eb10047f7181`。在 `b8ba139` 的 live-worker
  sidecar 重读之外，本次覆盖更早的 `Popen`-to-first-sidecar 窗口：running
  receipt 中合法的 `child_pid: null` 现在按“尚未持久化 PID”解析，非整数/布尔
  PID 仍 fail closed；缺失 sidecar 获得有界 worker startup grace；worker
  已异常退出但 command child 仍存活时，恢复会等待 child 收敛并再次读取
  sidecar，禁止归档/重试与旧 child 并发写同一路径。新增三类确定性恢复测试，
  完整本地 pytest 再次通过。提交：
  `b4fb415 Harden generation stage launch recovery`。
- `b4fb415` exact-commit Linux CPU-only isolated deployment 已通过。checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-b4fb415`；
  bundle：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-b4fb415-from-1ebcc15.bundle`，
  bytes `39,408,158`，SHA256
  `2370f1114103c11611e3396deaa9f301786fb2bc59e819f14d30d7d5cac8b7ec`，
  advertised head 仅为 `b4fb415`，prerequisites 仍为 `1ebcc152` 与
  `58d83bfc`。deployment receipt：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/b4fb415c51c608e68ea5e9484646eb10047f7181/deployment_receipt.json`，
  SHA256
  `0d1e45aab2d5cbf81e1acc3a20ce2175e383c34c47f3099bae5bfa2debc030a7`。
  Linux 验证为 `852 passed / 2 skipped / 0 failures / 0 errors`，全部
  `99/99` tracked runbook 通过 `bash -n`；receipt 仍为
  `readiness_execution_allowed=true`、`readiness_executed=false`、
  `full_training_launch_allowed=false`、
  `formal_generation_completion_claimed=false`。正式 repo 仍固定
  `1ebcc152 / scale/generative-system` 且 tracked clean；不得把该部署误读为
  readiness 已执行或 full 300K 已获授权。
- 本轮结束前的最新权威运行快照（2026-07-31）：50K monitor 为
  `running/cofitok_training/issues=[]`，CoFiTok `18,900/50,000`、dense
  `0/50,000`，GPU PID `319202`、约 `84,135 MiB`、利用率 `100%`，磁盘 free
  `196,502,274,048` bytes。readiness waiter PID `622788` 仍为
  `waiting_for_passing_stability_gate`，`child_pid=null`，精确绑定
  `5dd3488ac9b30274f4960195e252cc9fdb161002`，且
  `full_training_launch_allowed=false`。继续只读监控；50K source-bound gate
  通过且 GPU 空闲前不得运行 CUDA readiness，之后仍须人工单独授权 full 300K。

## 2026-08-01 readiness revision bridge handoff

- large-capacity 控制面最终 HEAD 为
  `27edb632d8cb9a642040ba6fcd3b1a0d682e6c79`（分支
  `scale/generation-large-capacity`）。新增 source-bound
  `full_training_readiness_bridge.json` 构建/重放链：来源 readiness 继续精确绑定
  `5dd3488ac9b30274f4960195e252cc9fdb161002`，目标训练 checkout 必须绑定本次
  deployed target。bridge 会重放来源 readiness 的七份物理 source、来源/目标两份
  deployment receipt、Git ancestor、目标 CUDA/runtime fingerprint，以及 66 个
  training-critical Git blob；66/66 blob 在 `5dd3488 -> 27edb63` 间完全一致。
- full 300K runbook 不再只按后缀粗略比较。以
  `monitor_report_passes() {` 为边界，训练执行后缀必须逐字节一致，SHA256 为
  `1a4e559c3c8828ceee9907f45e2de2d4513a77f37e4458eb50755ee53ab6382c`；
  授权前导只允许完整 bridge validator、六条 bridge-only binding 和 sample reserve
  `16,384 -> 116,640` 三类受控变换。规范化后的目标前导与来源前导 SHA256 均为
  `3c196801c05153da13501c265da8480714f2ef119b686db8dbc3e153558d2d07`；任何
  额外命令、漏传 source/target receipt、revision/branch 漂移或 marker 变化均
  fail closed。
- full launch receipt 已升级为 schema v2，source 数由 9 增至 10，新增 readiness
  bridge；post-training supervisor 会物理重哈希十份 source，terminal completion
  audit 新增独立 `stability_full_readiness_revision_bridge` 重放检查。aggregate
  completion runway 为 `116,640` samples（训练 allowance `16,384` + 正式 post-eval
  / prefix diagnostics `100,256`），最后只读所需 free 为 `180,880,415,360`
  bytes；2026-08-01 最新 monitor free 为 `187,356,852,224` bytes，仅余约
  `6,476,436,864` bytes，启动或恢复时必须重算，当前数值绝不是 300K 授权。
- 首次 target `35a9b99` 的 Linux 隔离部署在 receipt 写入前发现服务器旧 Git 不支持
  `git ls-tree --format`；部署 helper 自动清理临时 checkout，正式 repo 未变。修复后
  使用稳定默认 `ls-tree` 格式解析，最终部署 target 为 `27edb63`。
- 最终版本化 checkout：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-27edb63`；
  deployment receipt：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/27edb632d8cb9a642040ba6fcd3b1a0d682e6c79/deployment_receipt.json`，
  SHA256
  `db8a530d19a169c4a73a8d2cd5b5742ad366e74cab328b721f42c3eb9c541609`。
  Linux CPU-only isolated validation 为 `858 passed / 2 skipped / 0 failures /
  0 errors`，全部 `100/100` tracked runbook 通过 `bash -n`。receipt 明确
  `readiness_execution_allowed=true`、`readiness_executed=false`、
  `full_training_launch_allowed=false`、`formal_generation_completion_claimed=false`。
- 最终持久化 bundle：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-27edb63-from-1ebcc15.bundle`，
  bytes `39,422,470`，SHA256
  `7ee21caf8ceab3b403e0f7ce3d0276fb44cbb2e5fd85ba7a6d9df794c9a8e559`；
  advertised head 仅为 `27edb632d8cb9a642040ba6fcd3b1a0d682e6c79`，prerequisites
  精确为 `1ebcc15210e63a776a2ba448481cbd8bb94a4066` 与
  `58d83bfce2770eab2565b8c89a5f9a06201a0c86`。正式 repo 部署前后仍固定
  `1ebcc152 / scale/generative-system` 且 tracked clean。
- 当前 live 50K monitor 为 `running/cofitok_training/issues=[]`，CoFiTok
  `21,700/50,000`、dense `0/50,000`；GPU PID `319202` 占用 `84,122 MiB`、
  utilization `100%`。readiness waiter PID `622788` 仍为
  `waiting_for_passing_stability_gate`、`child_pid=null`，绑定来源 readiness
  `5dd3488` 且 `full_training_launch_allowed=false`。不得替换或重启 waiter；source-bound
  50K gate 通过且 GPU 空闲前不得手工运行 CUDA readiness，readiness 通过后 full
  300K 仍必须等待用户单独明确授权。
- 实现记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_full_readiness_revision_bridge.md`。

## 2026-08-01 historical checkpoint retention handoff

- large-capacity 分支当前 HEAD 为
  `caa66eba2e513fc28fe11c9763b1dedbb7f3fc87`。新增严格只读 historical
  checkpoint retention inventory、物理重哈希 validator 和 fail-closed runway
  checker；任何报告都固定 `archive_or_deletion_authorized=false`，且未批准的
  archive candidate 永远不能计入当前磁盘容量。
- 对
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29`
  的持久化审计覆盖 54 个 checkpoint、共 `54,336,513,724` bytes；54/54
  integrity 通过。分类为 33 required（`33,205,581,950` bytes）、21
  indeterminate（`21,130,931,774` bytes）、0 candidate；
  `currently_reclaimable_bytes=0`、`archive_readiness=blocked`。102 个 unresolved
  occurrence 是旧证据只写同名 checkpoint basename、无法在多个 run 间唯一归属，
  不是缺失权重；对应 checkpoint 必须继续保留，不得据此删除。
- 权威持久化报告位于
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/retention_audit_2026-08-01/`。
  inventory SHA256 为
  `62ae7388983e046fc0d0afbb7db2059e1afb5c2674880881e20411cd66f68cd5`，
  runway SHA256 为
  `c8ab71b2d36cdf18530a937d394c19adbfcec02c1d254ca5101e9c69777e290c`。
  该快照 free 为 `186,991,734,784` bytes，相对 aggregate completion requirement
  `180,880,415,360` 仅余 `6,111,319,424` bytes；之后必须动态重算，不能把该
  pass 当成 full 300K 授权。
- 最终隔离 checkout 为
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-caa66eb`；
  bundle 为
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-caa66eb-from-1ebcc15.bundle`，
  bytes `39,463,866`，SHA256
  `d41c257fb4b2e8c1596b8d9f50fd24c99c89c873b0591fcb0410a6f9178d96b5`。
  deployment receipt SHA256 为
  `372615c287e1e467869c7bded2c1471d39f96ebb58204f15fe92ebf093ed8785`；
  Linux 验证为 `865 passed / 2 skipped / 0 failures / 0 errors`，全部
  `101/101` tracked runbook 通过 `bash -n`。receipt 仍明确
  `readiness_executed=false`、`full_training_launch_allowed=false`、
  `formal_generation_completion_claimed=false`。
- 正式 server repo 仍固定 `1ebcc152 / scale/generative-system` 且 tracked clean；
  活跃 50K monitor 最新为 `running/cofitok_training/issues=[]`，CoFiTok
  `23,200/50,000`、dense `0/50,000`，GPU PID `319202` 占用约 `84,122 MiB`。
  readiness waiter 仍为 `waiting_for_passing_stability_gate`、`child_pid=null`、
  `full_training_launch_allowed=false`。不得替换 waiter，不得删除历史 checkpoint，
  不得运行 CUDA readiness 或启动 full 300K，直到 source-bound gate 通过、GPU
  空闲且用户再次明确授权。
- 实现与结果记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_checkpoint_retention_readonly_audit.md`。

## 2026-08-01 deployment Git-pack hardlink dedup handoff

- large-capacity 的最终部署实现 revision 为
  `e971fbdcf0169ca1c17f6d604a0d983a2e102370`；本地证据提交后的当前 HEAD 为
  `1fd2379`。新增/加固的 deployment pack
  去重器只处理 receipt-bound checkout 中字节与 SHA256 完全一致、位于同一文件
  系统的 Git pack/index；apply 必须精确 allow-list，操作前后均运行
  `git fsck --strict`、tracked-clean 检查和原 deployment receipt replay。所有
  checkout 路径与文件均保留，没有删除 bundle、checkpoint、sample 或训练资产。
- 最终隔离 checkout 为
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-e971fbd`；
  bundle bytes/SHA256 为
  `39,478,169 / a9ddd821f532d855a10d92faea892cfca49bf4a75e81d071512cac3826ea3e4f`；
  deployment receipt SHA256 为
  `aa0aa98ac94977dc3c80d6017776d240dcaef985a0b33c84e8d1b55979cddd35`。
  Linux 为 `873 passed / 2 skipped / 0 failures / 0 errors`，全部 `101/101`
  tracked runbook 通过 `bash -n`；独立 receipt replay 通过。receipt 仍明确
  `readiness_executed=false`、`full_training_launch_allowed=false`、
  `formal_generation_completion_claimed=false`。
- 初次 8-checkout hardlink 去重的 expected/observed savings 为
  `1,095,527,744 / 1,095,532,544` bytes；最终 revision 又将
  `checkout-340c2af` 与 `checkout-e971fbd` 的 4 个 common pack/index 路径接入
  同一 inode，expected/observed 增量为 `273,881,936 / 273,883,136` bytes。
  增量 result SHA256 为
  `4178bd63be537c0671d6692b50ad7783796ff10e9a5f5399ccdb853b095de464`，
  已独立 replay 通过。
- post-apply plan 为 `20 already_linked / 0 eligible / 24 indeterminate / 8 required`，
  `currently_saved_bytes=1,369,409,680`、`potential_physical_bytes_saved=0`；
  SHA256 为
  `e3ee61157346cce88fa5eb194f541d7e307c47b61fb310c166ac1d17348e41cc`。
  indeterminate unique packs 未改动；该结果不授权删除任何 deployment evidence。
- 最后只读快照为 CoFiTok `35,000/50,000`、dense `0/50,000`，monitor
  `running/cofitok_training/issues=[]`；GPU PID `319202` 约 `84,122 MiB`。
  readiness waiter 仍绑定 `checkout-5dd3488`，状态
  `waiting_for_passing_stability_gate`，`full_training_launch_allowed=false`。
  当前 `df` free 为 `186,811,084,800` bytes，相对 aggregate completion
  requirement 仅余 `5,930,669,440` bytes；后续必须动态重算。不得替换 waiter、
  不得运行 CUDA readiness 或启动 full 300K，直到 source-bound gate 通过、GPU
  空闲且用户再次明确授权。
- 实现与证据记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_deployment_pack_hardlink_dedup.md`；
  小型证据位于
  `CoFiTok-internal/artifacts/reports/generation/deployment_pack_dedup_2026-08-01/`。

## 2026-08-01 lightweight retention runway handoff

- large-capacity 的当前部署代码 revision 为
  `1604dc45c21ec73d9debf8a81e3d7367cd9fb220`；本地证据提交后的当前 HEAD 为
  `912ec87`。旧 dynamic runway helper 会
  调用 physical inventory replay，从而每次重哈希 `54,336,513,724` bytes
  checkpoint；现已拆成两层：inventory 建立/独立 validator 继续物理重放，日常
  runway 必须显式绑定已审计 inventory 的 expected SHA，拒绝 bytes/SHA/report
  漂移，并写明 `physical_checkpoint_hashes_replayed=false`。
- 一次发现该行为的旧 checker 调用没有生成报告，已只终止本轮 maintenance PID
  `741042/741041`；trainer `319202` 与 readiness waiter `622788` 未受影响。
- exact Linux checkout 为
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-1604dc4`；
  bundle bytes/SHA256 为
  `39,499,901 / 0baaf43696a778cb8fc39202752bed04bfbee0a65556fc77a5458bed62c628b0`；
  deployment receipt SHA256 为
  `38b14b3f20d478f23919cb748f208966dec5669198f7431266728f9f5684431e`。
  Linux 为 `874 passed / 2 skipped / 0 failures / 0 errors`，`101/101`
  tracked runbook 通过 `bash -n`，独立 receipt replay 通过。receipt 仍禁止
  full training。
- 新 checkout 的公共 pack/index 已按精确 allow-list 接入既有共享 inode；expected/
  observed savings 为 `136,940,968 / 136,941,568` bytes，result SHA256
  `e65e2d4749496d26789f991cd1f27a6e8e49d1c1e1754810568f3462a3b726cb`，
  独立 replay 通过。最终 plan 为 `22 already_linked / 0 eligible / 26
  indeterminate / 8 required`，`currently_saved_bytes=1,506,350,648`。
- 最终 lightweight runway 绑定 inventory SHA
  `62ae7388983e046fc0d0afbb7db2059e1afb5c2674880881e20411cd66f68cd5`；
  report SHA256
  `f23de67d99ebe5b4a58f5f66875d163ec857b10b581d7ad14bb1eff39988bcf9`，
  free/required/headroom 为
  `186,542,624,768 / 180,880,415,360 / 5,662,209,408` bytes，status `pass`，
  `currently_reclaimable_bytes=0`。它不授权 archive/delete、CUDA readiness 或
  full 300K。
- 最后权威运行快照：CoFiTok `35,500/50,000`、dense `0/50,000`，monitor
  `running/cofitok_training/issues=[]`；GPU PID `319202` 占用 `84,122 MiB`。
  waiter 仍为 `waiting_for_passing_stability_gate` 且
  `full_training_launch_allowed=false`。下一真实门槛仍是 CoFiTok 完成后串行 dense
  50K、正式 post-eval 与 source-bound gate；不得提前运行 readiness 或 full 300K。
- 实现记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_lightweight_retention_runway.md`。

## 2026-08-01 stability-scaling CoFiTok 35K handoff

- 本地证据 HEAD 为 `1c2ecec`；部署代码仍为 `1604dc4`，active training identity
  未改变，继续固定在
  `2c2c1f5166b73d4f28df93b276901671ac1a7836 / scale/generation-stability-50k-preflight`。
- 使用 clean evaluation checkout
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289 / scale/generation-stability-50k-posteval-v4`
  运行 tracked progress auditor，结果为 `healthy`、`issues=[]`、`warnings=[]`。
  rolling recovery set 精确为 `25K/30K/35K`，35K checkpoint 为
  `1,006,321,770` bytes / SHA256
  `98ce7eac9ef77da24dda42a96d1e17a5a047f60665fb30c10ed286a98c807f10`，
  sidecar/latest/stat/revision/dataset/runtime identity 全一致，integrity `verified`。
- 审计时 canonical metrics 为 716 行、last step `35,750`；rollout schedule
  `716/716` verified，EMA-teacher 在 30K 后有 115 active rows，`115/115`
  loss 有限非零，35,750 expected/observed scale 均为 `0.575`。validation
  `35/35` events 且 provenance complete。不同 event 使用不同 deterministic
  image batch，不得把数值串成同批学习曲线，也不构成样本质量主张。
- post-eval waiter PID `281834` 仍为
  `waiting_for_completed_training_pair`、`child_pid=null`，training/evaluation
  identity 与实际 clean checkout 一致；runbook SHA256 为
  `6cf80146c1696aa0bed9a4a370b41c42aa5be9dc03f753c4c81ce6b54278b064`。
  `pair_summary.json`、`promotion_gate.json`、`full_training_readiness.json`
  当前缺失符合阶段预期。readiness waiter 仍禁止 full training。
- 小型证据：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_2026-08-01/cofitok_step_00035000/`；
  记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_stability_scaling_50k_cofitok_step35000.md`。
  下一真实科学门槛仍是 CoFiTok 50K 后的 dense 50K、正式 free-state post-eval
  和 source-bound gate；35K 健康审计不得替代这些结果。

## 2026-08-01 stability 50K exact-resume handoff

- 用户要求释放 GPU 时，CoFiTok trainer 通过受支持的 `SIGTERM` 路径在当前
  optimizer step 后原子保存并退出；pause checkpoint 为
  `checkpoint_step_00036545.pt`，bytes/SHA256 为
  `1,006,321,642 / 3b02917393ac00d49ece5c61ac357b6e638096919e17c835bdbef6c810544c2e`。
  sidecar、`latest.json` 和 pause `training_report.json` 对账通过，report 明确
  `completed_steps=36545`、`training_complete=false`、`stop_requested=true`、
  `stop_signal=15`。
- 用户允许继续后，唯一 trainer PID `918650` 已从该 checkpoint 以
  `--resume auto` 恢复；训练 identity 仍固定在
  `2c2c1f5166b73d4f28df93b276901671ac1a7836 /
  scale/generation-stability-50k-preflight`。resume manifest 记录 metrics
  reconciliation 为 `unchanged`、`retained_rows=731`、`orphaned_rows=0`；首个
  resumed metric 为 step `36,546`，本地证据快照已推进到 `36,700`。
- 当前队列每个角色只保留一个实例：runbook `918098`、monitor `918502`、
  watchdog `918584`、trainer `918650`、post-eval waiter `919562`、readiness
  waiter `919564`。恢复竞态中多出的 post-eval waiter `919739` 在无 child 时已终止；
  后续不得重复启动 waiter。readiness 继续绑定 `5dd3488`，并明确
  `full_training_launch_allowed=false`；不得把本次 10% resume 解释为 full 300K
  授权。
- evaluation identity 仍为
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289 /
  scale/generation-stability-50k-posteval-v4`；training/post-eval/readiness
  runbook SHA256 分别为
  `dc8e26a54652fb51c8b3070f87e3460c20e0639a0dcf59b86592947966a84a18`、
  `6cf80146c1696aa0bed9a4a370b41c42aa5be9dc03f753c4c81ce6b54278b064`、
  `ed4fb9d6204097cd228b3b52875f7251408832ca7fdb4a4000bff0378e07460a`。
- resume 前精确移除了未验证且原始三棵样本树仍完整的冗余 staging tar，释放
  `3,423,416,320` bytes；matched-50K storage preflight 为 free/required/headroom
  `181,533,130,752 / 118,385,312,804 / 63,147,817,948`，status `pass`。为防 dense
  与 post-eval 阶段触及 aggregate runway，byte-identical archive tar 已用低优先级
  重建，bytes/SHA256 为
  `3,423,416,320 / 7d34b3c24f40c5c426a10c42286feaadcd391c787d238f2b3d61c8ae8557ecf6`，
  并已完成本地断点传输。tracked auditor 对 `3 roots / 20,267 files / 20,256
  PNG` 和六个 sample-set digest 全部 replay `pass`；audit SHA256 为
  `7bd53ffb22e845d1ccbcfaa82b32aeef33b3d4b9f177ada1aa07bef2b27716a3`。
  三棵旧 fixed-basis-v3 failed-gate sample roots 先在远端 tar 保留时精确移除，
  验证训练继续后再移除 staging tar；最终 free 为 `184,964,812,800`，相对归档前
  净增 `3,431,682,048` bytes。receipt SHA256 为
  `e6df789dd277901d31a458e01945ef01b3ea083dfe42a3c24708e7a3a35460bf`，本地 tar、
  repo evidence 和远端 `archive_receipts/` 三处均保留恢复契约。旧报告仍含原绝对
  sample path，直接 replay 前必须恢复 tar 并重跑 auditor。当前 aggregate headroom
  约 `4.08 GB`，仍不足以同时容纳 dense rolling checkpoints 与新的 matched sample
  pair；dense/post-eval 完成前还必须再归档至少一批已失败旧样本，不得删除 checkpoint。
- 本地归档审计实现提交为 `ed37eca`，exact-resume 证据提交为 `b429c86`。记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_stability_scaling_50k_resume_step36545.md`；
  证据：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_2026-08-01/resume_step_00036545/`。
  当前结论仅证明 exact resume 和等待链恢复，不支持样本质量、dense matched 结果、
  promotion pass 或 full 300K claim。
- fixed-basis-v3 样本归档证据提交为 `bac8a5e`；记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_fixed_basis_v3_sample_archive.md`；
  evidence/receipt：
  `CoFiTok-internal/artifacts/reports/generation/fixed_basis_v3_sample_archive_2026-08-01/`。

## 2026-08-01 historical failed-pair sample archive handoff

- 第二批空间收口仅选择两个已终止 `fail / hold` 的历史 matched pair 样本：
  compressed pair 的 promotion gate 失败项为 FID tolerance、absolute FID、endpoint
  tolerance、ordered-prefix path；rank-complete v2 已修复 ordering，但仍失败前三项。
  两份 gate SHA256 分别为
  `2c2d27401834e4d0fd2138ec886595f84ffa96679fad5be9bc189c717fba7eaf` 和
  `a3ac8ec7ee515ff5fe1eb04062f48c0b5ade2e548afd64322210dc969377cfe1`。
- 六棵 sample/prefix-diagnostic root 已归档到
  `C:/Users/zixi-/CoFiTok-archive/generation/historical_failed_compressed_rankcomplete_pairs_2026-08-01/historical_failed_compressed_rankcomplete_pairs_2026-08-01.tar`；
  bytes/SHA256 为
  `6,474,926,080 / 9946c301024e41816e207d6f749cb4385a9af96fb2364736a37fcf8b633438e7`。
  tracked auditor 对 `6 roots / 40,534 files / 40,512 PNG` 和全部 sample-set digest
  replay `pass`；audit SHA256 为
  `9e95cc35a4911440ab3d66a8dc066819853cc6164528b6392059a51e8b9d6219`。
- 六棵源目录先在远端 tar 保留时精确删除；确认训练 `running/issues=[]` 后才删除
  staging tar。checkpoint、报告、parent run dir 均未删除。free 从 staging 存在时的
  `178,488,901,632` 增至最终 `191,443,927,040` bytes。receipt SHA256 为
  `f592789f6cb6bf6e8f63909656dc73c34ff51b34387e977c2760f7f15a0196b9`，并保存在
  repo、离线归档目录和远端 `archive_receipts/`。旧 gate/visual report 保留原绝对
  sample path；直接 replay 前必须验证 tar SHA、恢复六棵 root 并重跑 auditor。
- 固定 retention inventory
  `62ae7388983e046fc0d0afbb7db2059e1afb5c2674880881e20411cd66f68cd5`
  的 lightweight runway 已重跑 `pass`：free/required/headroom 为
  `191,443,562,496 / 180,880,415,360 / 10,563,147,136`，
  `currently_reclaimable_bytes=0`、`physical_checkpoint_hashes_replayed=false`；
  report SHA256 为
  `1f977df3beb395a58d733ee34b2534b15379ede0fd0f2f4cdad0c1ec8a91371f`。
  它只覆盖当前 aggregate completion 模型，仍需随 checkpoint/sample 增长刷新，
  不授权 full 300K。
- 归档证据提交为 `8e035ad`；记录：
  `CoFiTok-internal/docs/records/2026-08-01_generation_historical_failed_pair_sample_archive.md`；
  evidence/receipt/runway：
  `CoFiTok-internal/artifacts/reports/generation/historical_failed_pair_sample_archive_2026-08-01/`。
  归档后训练健康快照已到 step `37,600`，trainer `918650`、monitor `918502`、
  watchdog `918584`、post-eval waiter `919562`、readiness waiter `919564` 均存活；
  readiness 仍必须保持 `full_training_launch_allowed=false`。

## 2026-08-02 CoFiTok stability 40K observer handoff

- 当前 CoFiTok 50K 训练尚未到 40K 时，已部署 bounded read-only milestone
  observer PID `939556`。它每 60 秒只读 metrics/Git identity；达到 40K 且
  checkpoint/sidecar/latest 原子发布后，调用 clean evaluation checkout
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289 / scale/generation-stability-50k-posteval-v4`
  的 tracked progress auditor，并要求 `integrity-policy=required`。它不加载模型、
  不申请 GPU、不发信号、不改变 pipeline decision。
- exact resume 使 40K 时的三点 rolling recovery set 应为
  `35,000 / 36,545 / 40,000`；observer 已显式要求三者，允许非 5K 对齐的 resume
  checkpoint，但 milestone 本身必须 5K 对齐。另要求至少 40 个 validation event、
  provenance complete、rollout/EMA-teacher 对所有 canonical row 的 config-bound
  schedule 验证、active loss 全部有限非零、issues/warnings 为空，以及 checkpoint
  training identity 仍为 `2c2c1f5 / scale/generation-stability-50k-preflight`。
- observer source commit 为 `8bf4bb73ec05543a6e0317ef00a72fed5ac2a3b4`，
  remote source 为 `/tmp/cofitok_stability_40k_milestone_waiter_84c04e9.py`，
  bytes/SHA256 为
  `15,043 / 84c04e91dfb846f1ef3eeabe88e7be1b5d6fd6b02121c0fb2908163f886af99e`。
  exact commit 验证为 targeted `33 passed`、完整 pytest
  `893 collected / 887 passed / 6 skipped`。
- launch 时 observer `waiting`、last step `38,050`、audit attempts `0`；initial
  status SHA256 为
  `e89347a1fc8de93d9987907b1f256e5238654bbc8cf50efb0724ba9774c1867a`。
  launch receipt bytes/SHA256 为
  `3,729 / 452f29aa839f7e5d2458ea13884ee35d8110bcfdf7f9bfb1e01746ee661fd175`，
  repo 与远端 report dir 都保留。authoritative live status/report 分别为
  `.../reports/cofitok_step_00040000/milestone_waiter.json` 和
  `.../reports/cofitok_step_00040000/progress_audit.json`。
- launch evidence commit 为 `a13c510edee5bd36d342f1a2335beccfc929df57`；记录：
  `CoFiTok-internal/docs/records/2026-08-02_generation_stability_50k_cofitok_40k_waiter.md`；
  evidence：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_2026-08-01/cofitok_step_00040000/`。
  40K pass 只证明恢复和训练稳定性；仍不证明样本质量、dense matched 结果、
  promotion pass 或 full 300K readiness。
- 同一只读 source 已预先启动独立 45K observer PID `942412`，避免 40K pass 后
  人工补启动。它写入独立 `.../reports/cofitok_step_00045000/`，要求 rolling set
  `36,545 / 40,000 / 45,000`、45 个 validation event 和相同 config/Git/schedule/
  active-loss/integrity 约束；poll `60s`、bounded timeout `36,000s`。initial last
  step `38,250`，status SHA256
  `03b5b5b6b1a28d40fa9589edec080e5d885be00bea0bf5722defd30f61f13211`；
  launch receipt bytes/SHA256 为
  `2,631 / ca2c6ad700dcb3e108fcf6ccba431041a3f6e42b8e62f4aaa60f7d7ed7679e3f`。
  evidence commit 为 `5029487a9a3c91c5447e3da3f9ee9ab0ff8b7425`。该 observer 同样不能
  授权 post-eval 或 full training。
- 历史 Codex thread heartbeat `cofitok-40k-audit-monitor` 已删除。其 authoritative
  40K observer 因保留的 36,545 stop-requested pause report 落后于 resumed metrics 而
  false-fail；这是 observability incident，不是 checkpoint 或训练失败。不得从该
  milestone audit 推导 full 300K 授权。

## 2026-08-02 CoFiTok 50K completion / dense-transition recovery handoff

- CoFiTok stability member 已精确完成 `50,000/50,000`；watchdog `passed`、
  child exit `0`。最终 checkpoint
  `checkpoint_step_00050000.pt` 的 bytes/SHA256 为
  `1,006,325,418 / ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a`，
  该 SHA 已对远端 payload 独立重算；sidecar SHA256 为
  `4f3f6f3f401f34131f902b016baf41897b35f95d242bb023981feff4a36226a8`。
  report/latest/sidecar 仍绑定 immutable training identity
  `2c2c1f5166b73d4f28df93b276901671ac1a7836 /
  scale/generation-stability-50k-preflight`；rolling set 为 `40K/45K/50K`。
- dense 没有启动。根因不是训练或权重失败，而是 runbook 完成校验器把 branch
  硬编码为旧 `scale/generative-system`，与 runbook 已明确接受的 stability-preflight
  branch 冲突；权威 transition log SHA256 为
  `36b7876a54648f1bf42f535fa8d360014208c8af7055ecd22cd86ed97f17d1eb`。
  pair monitor 随后因无 active runbook/trainer fail-closed，post-eval/readiness waiter
  顺序 fail-closed；`full_training_launch_allowed=false` 保持不变。
- authoritative dense run directory 只有
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/dense_rollout_x0_u2_ema_teacher`；
  不得把非权威 `dense_rgbtail3_rollout_x0_u2_ema_teacher` 别名用于 fresh/resume 判断。
- 40K/45K read-only observer 另有独立 false failure：exact resume 后保留的 36,545
  stop-requested pause report 落后于 canonical resumed metrics，旧 progress auditor
  将其误判为 invalid。修复只在显式
  `--allow-stale-incomplete-training-report` 下接受目标/own-latest 绑定完整的
  stop-requested incomplete report；其他 mismatch 继续 fail-closed。
- completion validator 现要求显式 expected branch，所有 tracked runbook 调用均已
  补齐；新 dense-only recovery runbook 将 clean control checkout 与 immutable
  training checkout 分离，重验 CoFiTok trust boundary、frozen `64x1` runtime、pair
  config 和 storage，只从原 training revision 启动 dense，禁止重复 CoFiTok，也不
  授权 full 300K。验证为 full pytest `903 collected / 897 passed / 6 skipped`、
  runbook contract pass、`git diff --check` pass、远端 `bash -n` pass。
- dense recovery controller 还必须在任何 status/preflight/monitor/trainer 动作前，以
  fd 6 非阻塞持有
  `stability_scaling_50k_ema_teacher/dense_recovery.lock`；并发 controller 以 exit 15
  fail-closed 且不得覆盖 active status。该加固后的完整测试为
  `904 collected / 898 passed / 6 skipped`，不修改 immutable training checkout。
- 上述 lock hardening 已以 `4,410`-byte、SHA256
  `8df4ad23d59fb39ce2663bbbc502d2d2c07345f3ec4b1debe8d6f757c2203963`
  的 prerequisite-bound bundle 将隔离 controller fast-forward 到
  `5f57757a2162507c6166dbfe976246df6ae1af91`；remote `bash -n` 与 5 个 focused
  Linux tests 通过。双-controller `PREFLIGHT_ONLY=true` 实测 primary `prepared`、
  secondary exit `15` 且未覆盖 status；fresh free/required/headroom 为
  `187,194,892,288 / 118,385,312,804 / 68,809,579,484` bytes。heartbeat 的
  authoritative controller 也必须使用 `5f57757`，不得再以 `e5c9ed7` 正式启动。
- `2026-08-02T12:00:14+08:00`，free 为 `187,909,763,072` bytes；原 matched-50K
  plan required `118,385,312,804`，但 recovery 仍必须 launch 前重跑。GPU PID
  `362355` 属 FieldScope（15,412 MiB），未发信号、未杀死、未抢占；dense recovery
  仅 prepared，须等 GPU idle 后再部署/启动。
- recovery-control 初始 commit `00549fe3ec4ea5334c76e7249537224104ee44be` 已通过
  SHA256 `833e7902c251e61ade15c68fc8b55d047c9982d4578a58a9a5a8a6998f734177`
  的 `39,681,857`-byte bundle 部署到新隔离 checkout
  `/tmp/cofitok-stability-dense-recovery-control-00549fe`，branch 为
  `scale/generation-stability-dense-recovery-control-00549fe`；official repo 和
  immutable training checkout 均未移动。`PREFLIGHT_ONLY=true` 已得到
  `status=prepared`，重新验证 checkpoint/runtime/config/storage，fresh
  free/required/headroom 为
  `187,577,434,112 / 118,385,312,804 / 69,192,121,308` bytes，且没有启动任何
  queue/GPU process。正式执行仍必须显式设置 `RECOVERY_EXECUTION_ALLOWED=true`
  并在 GPU idle 后重新跑全部 preflight。
- isolated controller 已再通过 `7,036`-byte、SHA256
  `fa8e41d2531cd89bfbfef23610838a38f3ebb3f3ce11bc38bab11bb76889ef0b`
  的 increment fast-forward 到
  `e5c9ed7bd4590f5dd6dd0d78308c2ff8e868b58a`；第二次 `PREFLIGHT_ONLY=true`
  仍为 `prepared`，free/required/headroom 为
  `187,569,545,216 / 118,385,312,804 / 69,184,232,412` bytes，GPU 仍为 FieldScope
  PID `362355`。heartbeat 的 authoritative controller revision 已同步更新为
  `e5c9ed7`。
- dense recovery 不得自行生成 `pair_summary.json`。monitor 证明 pair complete 后，
  必须由 exact post-eval waiter 使用原 locked `reports/config_validation.json` 创建
  source-bound summary；不得用内容相同但路径不同的
  `dense_recovery_config_validation.json` 替代，否则 provenance 对账会失败。
- 失效 heartbeat `cofitok-40k-audit-monitor` 已删除。记录：
  `CoFiTok-internal/docs/records/2026-08-02_generation_stability_50k_cofitok_completion_dense_transition_recovery.md`；
  evidence：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/cofitok_50k_transition_incident_2026-08-02/`。
- 新 thread heartbeat `cofitok-dense-recovery-launch-monitor` 在等待 GPU handoff 期间每
  5 分钟检查 GPU；busy 时必须静默只读。dense 成功启动并核验后再降回较低频率的
  training-progress monitor。
  GPU busy 时只读且不得触碰任何 process；idle 时必须重验 control/training clean
  identity、CoFiTok 50K checkpoint trust boundary、frozen runtime、dense state、related
  process 和 fresh storage，随后仅以 `RECOVERY_EXECUTION_ALLOWED=true` 启动
  dense-only recovery。启动后须验证唯一 runbook/monitor/watchdog/dense trainer/GPU
  identity，并把 heartbeat 更新为 dense progress monitor；不得重跑 CoFiTok，不得
  启动或授权 full 300K。2026-08-02 12:28 CST 已将上述 authoritative dense path
  显式写入 heartbeat prompt；当时该目录确认为 absent，FieldScope PID `362355`
  仍以 99% GPU utilization 运行，所有 CoFiTok GPU/process 保持未触碰。
- dense launch 验证后，heartbeat 必须先恢复 exact post-eval waiter：checkout
  `/tmp/cofitok-stability-schedule-audit-c1efb12`，identity
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289 /
  scale/generation-stability-50k-posteval-v4`，stability decision SHA
  `d5a6fc017f20c7d024abfab1967ba6bc966b624e3a77e9246292ddaaf7dc1da4`。
  只有其新 status 已变为 waiting/running 后，才能恢复 receipt-bound readiness waiter：
  checkout
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-5dd3488`，
  identity `5dd3488ac9b30274f4960195e252cc9fdb161002 /
  scale/generation-large-capacity`，deployment receipt SHA
  `46c72437bf0e44ad7e815ad7b3bb3e29af0f7e1664ab177c031c58e18a0a0758`。
  readiness 必须持续 `full_training_launch_allowed=false`。

## 2026-08-02 stability dense recovery live handoff

- FieldScope 释放 GPU 后，独立 prelaunch audit 与 recovery runbook 内部 preflight
  均通过；dense-only controller PID `511801` 已从 controller revision
  `5f57757a2162507c6166dbfe976246df6ae1af91` 启动。free/required/headroom 为
  `186,847,199,232 / 118,385,312,804 / 68,461,886,428` bytes。
- live pair monitor/watchdog/GPU trainer leader PID 分别为
  `530064 / 537704 / 541878`；trainer cwd 仍是 immutable training checkout
  `2c2c1f5`，run manifest 绑定 62,824,707 parameters、bf16、effective batch 64、
  50,000 target steps 与 runtime-environment SHA
  `d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`。
  DataLoader children 会继承 trainer argv，但不是额外 GPU trainer；GPU compute PID
  中必须恰有一个 CoFiTok trainer leader，unrelated project process 另行识别且不得触碰。
- first authoritative monitor refresh 为 `running / dense_identity_training / no issues`，
  dense step `100`、images seen `6,400`。controller lock probe exit `1`，证明 fd-6
  lock 仍由 active recovery 持有。不得启动第二 recovery controller。
- exact post-eval waiter PID `717145` 已恢复为
  `waiting_for_completed_training_pair`；receipt-bound readiness waiter PID `748183`
  随后恢复为 `waiting_for_passing_stability_gate`，且
  `full_training_launch_allowed=false`。不得手工生成 `pair_summary.json`。
- heartbeat `cofitok-dense-recovery-launch-monitor` 已从临时 5 分钟 GPU-handoff 检查
  改为每 30 分钟 dense-progress monitor；只在新 5K checkpoint、status transition、
  storage risk 或 failure 等 material change 时通知。禁止杀/重启 trainer、重跑
  CoFiTok 或授权/启动 full 300K。
- launch 记录：
  `CoFiTok-internal/docs/records/2026-08-02_generation_stability_dense_recovery_launch.md`；
  小型 evidence pack：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/dense_recovery_launch_2026-08-02/`。
- 2026-08-02 14:23 CST 起，另一个项目 FieldScope 的 `extract-dataset` GPU
  process（PID `646421`，约 `15.4 GiB`）与 CoFiTok dense trainer 并行；不得
  signal、暂停、杀死或修改它。CoFiTok 约占 `77,970 MiB`，剩余 GPU memory
  约 `3.85 GiB`，近期 50-step interval 从约 `130s` 放慢到约 `278s`。只在该
  process 消失、显存/存储风险增加、CoFiTok stall/failure 或其他 material identity
  change 时通知；消失并恢复吞吐后才把 heartbeat 从 5 分钟降回 30 分钟。
- 2026-08-02 15:28 CST 最近一次权威只读检查中，dense 为 step `2,100`，metrics
  finite/strictly monotonic 且 `samples_seen=step*64`；pair/recovery/watchdog 均
  running，post-eval/readiness 均 waiting，六个 launch identity、controller fd-6
  lock 和 immutable Git identity 均仍有效。当前 checkpoint 尚未到首个 5K 边界。
- 当前 filesystem free bytes 为 `176,743,428,096`。active matched 50K required
  `118,385,312,804`，headroom `58,358,115,292`，因此当前训练 runway 仍安全；
  aggregate completion required `180,880,415,360`，headroom
  `-4,136,987,264`。负值来自 unrelated
  `/root/autodl-tmp/phrasebind_rebuttal_final_checkpoints_20260802.tar`
  （`9,717,831,680` bytes）；不得擅自删除，只在风险继续变化时再通知。
- branch `scale/generation-large-capacity` commit `a22f6da` 新增
  `scripts/audit_generation_stability_live.py`：一次性汇总六份权威状态、exact-byte
  source identity、metadata-only checkpoint integrity、metrics/sample arithmetic、
  Git/process/GPU/lock identity 和 matched-vs-aggregate storage runway。验证为
  `32 passed` focused、`904 passed, 6 skipped` full。该工具当前只在本地提交，
  **尚未部署或运行于 active remote checkout**；不得为部署它移动 immutable training
  identity，也不得由它授权 full 300K。
- 2026-08-02 16:16 CST 的后续权威只读检查中，dense 已到 step `2,550`、
  `163,200` samples；52 行 metrics 全部 finite、strictly monotonic 且
  `samples_seen=step*64`。pair/recovery/watchdog 和两级 waiter 状态、六个 process
  identity、fd-6 lock、四个 clean Git identity 均保持有效；尚未到首个 5K checkpoint。
  FieldScope PID `646421` 仍占约 `15,412 MiB`，CoFiTok PID `541878` 约占
  `77,970 MiB`，剩余约 `3,853 MiB`；所有进程均未触碰。free/active headroom/
  aggregate headroom 为 `176,591,904,768 / 58,206,591,964 / -4,288,510,592`
  bytes，active 50K runway 仍安全，aggregate 负值继续仅作后续容量预警。
- branch `scale/generation-large-capacity` commit
  `f958b21725fa1c2b6b99ba1e451070576fdaae3b` 将 GPU contention provenance 纳入
  matched-compute 证据链：新 pair monitor 持久记录 exact unrelated GPU
  PID/start/argv/cwd/memory、poll coverage 和 clean Git binding；final comparison
  升为 schema v6，并仅在从训练开始前到 pair complete 的连续、完整、exclusive
  GPU observation 下允许把 raw wall-clock/img-s 作直接效率比较。否则质量、数据、
  steps 和参数公平性仍是 direct tier，但耗时/吞吐必须标为 observational-only；
  completion audit 会独立重算并拒绝伪造升级。验证为 `912 passed, 6 skipped`、
  两份改动 post-eval runbook 在 pro6000 原生 `bash -n` 为 `2/2`。该提交只在本地，
  **未部署到当前 immutable 50K 训练，也不授权或启动 full 300K**。

## 2026-08-02 routine inference exact-resume handoff

- branch `scale/generation-large-capacity` commit
  `ccbd058ead5f377e088bed15cc8eb4e915639fd5` 将日常 class/seed/prefix 推理升级为
  可审计 exact resume：首张 PNG 前冻结 source-bound
  `inference_manifest.json`，每张原子发布后更新
  `inference_progress.json`；`--resume` 会重算已有 PNG SHA，只保留身份和 digest
  都匹配的输出，并按原 seed/class/prefix 随机流仅重建 missing/corrupt 项。
- completed resume 是只读复用；checkpoint/request/Git/runtime/report/control-evidence
  drift、缺失 progress、非有限 timing、symlink、跨目录路径或额外/嵌套 PNG 均在改写
  control file 前 fail-closed。`--overwrite` 也不能把不同请求的 PNG 混在同一目录。
  completion audit 现要求 schema-v2 replay evidence，并独立重读 manifest/progress、
  重算物理 PNG SHA；legacy smoke report 不再能作为最终 release evidence。
- 验证为 focused `118 collected / 116 passed / 2 skipped`，完整本地 suite
  `927 collected / 921 passed / 6 skipped in 205.11s`，`py_compile`、CLI help 和
  `git diff --check` 均通过。该提交没有修改 shell runbook，完整 suite 仍覆盖原有
  runbook syntax/contract tests。
- 该能力目前**仅在本地 commit**，尚未部署到任何 active remote checkout；不得为
  部署它移动 immutable stability 50K training identity，也不得据此授权 full 300K。
  详细记录：
  `CoFiTok-internal/docs/records/2026-08-02_generation_inference_exact_resume.md`。
- 先前 FieldScope PID `646421` 于 `2026-08-02T10:00Z` 自行退出，未作任何信号或
  修改；dense 50-step interval 随后从约 `278s` 恢复到约 `127s`，heartbeat 已恢复
  30 分钟间隔。`2026-08-02 23:04 CST` 最近一次权威只读检查中，dense 为 step
  `10,900`、samples `697,600`，metrics finite/strictly monotonic 且
  `samples_seen=step*64`；GPU compute 只剩 CoFiTok trainer PID `541878`，六个 process
  identity、fd-6 lock、四个 clean Git identity、pair/recovery/watchdog 与两级 waiter
  均保持健康，readiness 仍为 `full_training_launch_allowed=false`。
- dense 5K/10K checkpoint 均为 `1,006,120,214` bytes；独立重算 SHA256 分别为
  `f3dbfb9298d3cd91ddcc47b31c9796fafb018083968b95f05d425d86a6ea35df` 与
  `c045bd271c4ab128499c82c7e0e2b088ba5f596a6fcd1d83e5d28eb4c0417a84`，均与
  integrity sidecar/latest、training revision `2c2c1f5` 和 runtime SHA
  `d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e` 一致。
  当时 free bytes 为 `183,921,299,456`；active 50K runway 安全。继续只在新 5K
  checkpoint、状态转换、storage risk 或 failure 时通知，不得杀/重启 trainer、重跑
  CoFiTok 或授权/启动 full 300K。

## 2026-08-03 matched 10K trajectory handoff

- branch `scale/generation-large-capacity` commit
  `a10d3b1588dba3d16b0e2bd2f3fb8557085d5645` 新增 source-bound matched
  training-trajectory diagnostic。它冻结双方精确到 step 10,000 的原始 JSONL
  prefix，验证 clean Git、formal dataset/runtime SHA、generation pair contract、
  参数差、exact log/validation schedule、全数值有限性和
  `samples_seen=step*64`，并只比较成对 fixed-validation epsilon 信号。
- 冻结证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_10k_training_trajectory_2026-08-02/`；
  `trajectory_report.json` SHA256 为
  `2576f53574667ca0e0a8172a3501e9e5baf8f0255a6b6cd5d961797e2a267476`，
  可从四份 source snapshot 逐字节重建为同一 SHA。
- 双方各 `201` 行、step `10,000`、`640,000` images，十个 validation event
  在 step/event index/batch index/`64` images/noise seed `102030` 上完全配对。
  CoFiTok/dense mean validation epsilon MSE 为
  `0.0326398859 / 0.0326587601`，相对差 `-0.057792%`；10K endpoint 相对差
  `+0.034889%`，最大单事件绝对相对差 `1.444119%`，低值事件数 `4/6`。
  正确解释仅为 shared validation trajectory 暂未分叉，不能宣称质量胜出。
- 工具明确设置 `total_loss_comparison_allowed=false`（CoFiTok 有
  factorization-only auxiliary）、`training_wall_clock_comparison_allowed=false`
  （dense 曾受 FieldScope GPU contention），并保持 `quality_claim_allowed=false`、
  `formal_50k_gate_substitute=false`、`promotion_authorization_allowed=false`、
  `full_training_launch_allowed=false`。下一份必要证据仍是 exact matched 50K
  completion 后由既有 waiter 单独执行 formal EMA post-eval。
- 最终验证为 focused `58 passed`，新增工具单测 `8 passed`，完整 suite
  `935 collected / 929 passed / 6 skipped in 195.7s`，报告 exact rebuild、
  `py_compile` 与 `git diff --check` 通过。该提交仍只在本地，没有部署或移动任何
  active remote checkout，也没有修改 runbook 或授权 full 300K。
- `2026-08-03 00:10 CST` 的提交后只读复核中，dense 已到 step `12,450`、
  `796,800` samples；`250` 行 metrics finite/strictly monotonic 且
  `samples_seen=step*64`。pair/recovery/watchdog 继续 running，posteval/readiness
  继续 waiting 且 `full_training_launch_allowed=false`；fd-6 lock、immutable
  revision `2c2c1f5` 和六个 control/process identity 均保持有效。GPU compute
  仍只有 trainer PID `541878`（`77,970 MiB`），free GPU memory `19,271 MiB`，
  filesystem free bytes `183,921,258,496`；尚未到新 15K checkpoint，未作任何
  signal、restart 或远端写入。

## 2026-08-03 matched 12K schedule-transition handoff

- branch `scale/generation-large-capacity` commit
  `9e65ad2d853c1a2065589743f6ea7c3b78b36955` 将 matched trajectory report
  升为 schema 2：从双方 resolved manifest 预声明 rollout/EMA-teacher 的
  inactive/warmup/full-scale 区间，并逐事件核对两侧 observed schedule scale
  相同且与区间一致。这样后续 15K/40K/50K 审计不会把 schedule transition
  混入一个无区分均值。
- 新 evidence pack 冻结双方精确到最后一个完整 scheduled-validation cutoff
  step `12,000` 的原始前缀，而不是使用 live step `12,650`：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_12k_schedule_trajectory_2026-08-03/`。
  `trajectory_report.json` bytes/SHA256 为 `21,443` /
  `64d291ae5730ce80d7b1656728e97dbf930aac68149332375fab8c2dbe8e7d62`，
  exact rebuild 保持同一 SHA。
- 双方各 `241` 行、step `12,000`、`768,000` images、12 个 validation event。
  全窗口 CoFiTok/dense mean epsilon MSE 为
  `0.0325498291 / 0.0327016859`，相对差 `-0.464370%`，低值事件数 `6 / 6`。
  rollout warmup 1K--9K 的 relative delta of means 为 `-0.067741%`；full-scale
  10K--12K 为 `-1.664242%`，但只有 3 个不同 deterministic validation batch，
  12K endpoint `-4.507906%` 只能作为 exploratory observation，不能宣称 durable
  trend、统计显著性或 generation-quality win。EMA teacher 在 30K 前仍未激活。
- report 继续禁止 total-loss 与 wall-clock direct comparison，并显式保持
  schedule-regime significance、sample quality、formal-50K substitution、promotion
  和 `full_training_launch_allowed` 全为 false。下一份 decision-grade evidence
  仍是 exact healthy matched 50K completion 后由既有 waiter 执行 formal EMA
  sample post-eval；不得据 12K shared-epsilon trajectory 授权 full 300K。
- 验证为 schedule builder 单测 `13 passed`，相关 focused `44 passed`，完整 suite
  `940 collected / 934 passed / 6 skipped`，并通过 exact report rebuild、
  `py_compile`、`git diff --check`。该 commit 只在本地，没有部署或移动 active
  remote checkout，也没有修改任何 runbook 或 GPU process。
- `2026-08-03 00:18 CST` 的同步只读审计中，dense 为 step `12,650`、
  `809,600` samples；`254` 行 metrics finite/strictly monotonic 且
  `samples_seen=step*64`。pair/recovery/watchdog running，posteval/readiness waiting，
  readiness 仍 `full_training_launch_allowed=false`。controller/control child/watchdog
  继承同一 `dense_recovery.lock` fd，trainer 自身不持有该文件 fd；两份 Git identity
  clean 且仍为 `2c2c1f5 / 5f57757`。GPU compute 只有 trainer PID `541878`
  （`77,970 MiB`），free GPU memory `19,271 MiB`，filesystem free bytes
  `183,921,254,400`；15K checkpoint 尚未出现，未作 signal、restart、hash-heavy
  audit 或任何远端写入。

## 2026-08-03 checkpoint evaluation completed-replay handoff

- branch `scale/generation-large-capacity` commit
  `9215e5fe0986c7d2f2596d8e2584458ee14164aa` 为
  `scripts/evaluate_generation_checkpoint.py` 增加严格完成态复用。
  evaluator 现在先原子写入 `checkpoint_evaluation_manifest.json`，绑定 checkpoint
  physical path/bytes/SHA/step/format、integrity sidecar identity、evaluator Git、
  num-images/timestep/random-orders/seed/weights/precision 和 canonical output/report
  path；完成报告升为 schema 2 并反向绑定 manifest identity。
- `--resume` 的语义是 fail closed：空目录可首次运行；只有 manifest、尚无报告时
  从头重做该 diagnostic；已有完成报告时，在不加载模型、不使用 GPU 的情况下先重算
  checkpoint/sidecar identity，并核对 request、Git、config/timestep、deterministic
  order set、evaluated count 和 runtime provenance 后原样复用。它**不是** batch-level
  partial resume。report-without-manifest、参数/权重漂移、symlink、unexpected file、
  malformed/tampered report 均拒绝；未传 `--resume` 时也不得覆盖旧证据。
- stability 50K formal post-eval、future full 50K post-eval 与 shared 300K milestone
  evaluator 均已传 `--resume`。locked legacy 10% 结果不迁移、不重跑。实现记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_checkpoint_evaluation_completed_replay.md`。
- 验证为相关 focused `40 passed`；完整 suite `940 passed / 6 skipped in 255.31s`，
  `py_compile`、`git diff --check` 以及三份变更 runbook 经 pro6000 stdin 的 Linux
  `bash -n` 全部通过。提交只包含 9 个 owned 文件；`.codex-bundles/`、7 月 29 日
  acceptance artifacts/record 继续未跟踪且未暂存。
- 该 commit 仍只在本地，没有部署到 active training/posteval/readiness checkout；
  不得为部署它移动 immutable stability identity，也不得据此授权 full 300K。
  `2026-08-03 01:00 CST` 提交后只读复核中，dense live metrics 已到 step `13,650`、
  `873,600` samples；pair monitor step `13,600`（正常刷新滞后），pair/recovery/watchdog
  running，posteval/readiness waiting，`full_training_launch_allowed=false`。15K
  checkpoint 尚未出现；GPU compute 仍只有 trainer PID `541878`（`77,970 MiB`），
  filesystem free bytes `183,921,225,728`。未作 signal、restart 或远端项目写入。

## 2026-08-03 generation metrics completed-replay handoff

- branch `scale/generation-large-capacity` commit
  `48dca0f628c2ce0fe170491bdc6baa9d3d20ded4` 为
  `scripts/evaluate_generation_metrics.py` 增加 formal torch-fidelity 完成态复用。
  metrics report 升为 schema 3；`--resume` 首次可运行，完成后只有在重新枚举/校验
  全部 generated PNG、重算 generated sample-set SHA 与 physical real-set tree SHA、
  并验证 sampling report/manifest/progress 文件身份后才跳过 FID/IS/PRC 核心计算。
- report 额外绑定 evaluator Git/runtime SHA、torch-fidelity version、CPU/CUDA、canonical
  real/generated/sampling/output/report paths、real/generated counts、batch/PRC batch、
  min-samples、seed、cache root/content-addressed cache name 和 PRC enablement。旧 schema、
  source/request/environment drift、symlink、unexpected file、non-finite/out-of-domain metric
  或未传 `--resume` 的覆盖尝试均 fail closed。这是 completed-result replay，不宣称
  torch-fidelity 内部 feature/PRC batch 可断点续算。
- stability 50K formal post-eval、future full 50K post-eval 与 shared full milestone
  evaluator 均已传 `--resume`；locked legacy 10% 不迁移、不重跑。详细记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_metrics_completed_replay.md`。
- 相关 evaluator/runbook tests `37 passed`，gate/comparison/terminal completion-audit
  regression `131 passed`；完整 suite `940 passed / 6 skipped in 254.70s`，`py_compile`、
  `git diff --check` 与三份变更 runbook 的 Linux `bash -n` 均通过。提交只包含 9 个
  owned 文件；用户原有三个 untracked 路径未暂存。
- 该 commit 仍只在本地，没有部署或修改 active immutable checkout。按
  `ml-training-recipes` 的 evidence boundary，metrics recoverability 不等于样本质量，
  仍必须等待真实 EMA 10K/50K samples 与 FID/IS/precision/recall。`2026-08-03 01:15 CST`
  提交后只读复核中，dense live metrics 为 step `14,050`、`899,200` samples；pair
  monitor step `13,950`（正常刷新滞后），pair running/no issues，posteval/readiness
  waiting，`full_training_launch_allowed=false`。15K checkpoint 尚未出现；GPU compute
  仍只有 trainer PID `541878`（`77,970 MiB`），filesystem free bytes
  `183,921,213,440`。未作 signal、restart 或远端项目写入。

## 2026-08-03 formal post-evaluation exclusivity handoff

- branch `scale/generation-large-capacity` commit
  `d2e8daeaada58fbefca79421d28423a187307d08` 在两个当前正式入口
  `generation_stability_ema_teacher_50k_posteval_after_training.sh` 与
  `generation_stability_ema_teacher_full_posteval_50k.sh` 增加 output-root-scoped
  非阻塞 fd-8 `flock`。锁在任何报告写入、sampling batch benchmark、checkpoint eval
  或 10K/50K sampling 之前获取；第二个控制器 fail closed，exit `75`。
- 该修复补上 stage receipt / component-level resume 无法覆盖的顶层并发窗口：receipt
  可以接管 detached worker，却不能阻止两个独立 runbook 同时对相同 immutable sample
  output 和 GPU 启动工作。锁只排斥同一 authoritative output root 的 post-eval，不能
  signal/暂停其他项目进程，也不授权或启动 full 300K。
- 记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_posteval_exclusive_controller.md`。
  focused runbook tests `13 passed`，完整 pytest exit `0`（在既有 `940 passed / 6
  skipped` 上新增 2 个通过测试，即 `942 passed / 6 skipped`）；两个变更脚本以 exact
  working-tree bytes 经 stdin 送入 pro6000，Linux `bash -n` 均通过，`git diff --check`
  通过。
- 变更仍只在本地，未部署/移动 active training revision `2c2c1f5`、posteval
  `c1efb12` 或 readiness `5dd3488`。`2026-08-03 01:31 CST` 的提交后只读复核中 dense 为
  step `14,400`、`921,600` samples；`289` 行 metrics finite/strictly monotonic 且
  `samples_seen=step*64`。15K checkpoint 尚未出现，GPU compute 仍只有 trainer PID
  `541878`（`77,970 MiB`），filesystem free bytes `183,921,205,248`；pair/watchdog
  running，posteval/readiness waiting，`full_training_launch_allowed=false`，未作任何
  signal、restart 或远端写入。

## 2026-08-03 generation output single-writer handoff

- branch `scale/generation-large-capacity` commit
  `83d3054aa5ccc7b9287d44015697313d133d2dd0` 新增跨平台
  `cofitok.output_lock.exclusive_output_lock`，并让 formal
  `scripts/generate_samples.py` 与日常 `scripts/infer_generation.py` 在 checkpoint
  加载、GPU session 初始化、manifest/progress 写入或创建 output directory **之前**
  获取 output-scoped 非阻塞 OS lock。
- Linux 使用 `fcntl.flock(LOCK_EX|LOCK_NB)`，Windows 使用
  `msvcrt.locking(LK_NBLCK)`；锁文件为目标相邻的
  `.<target>.cofitok-output.lock`，保留 schema/role/PID/hostname/target/time 诊断。
  文件故意持久保留，真正所有权跟随 fd 并在 crash 后自动释放；不得删除 held lock
  导致新 inode 双写。symlink target/parent fail closed，Linux 在可用时使用
  `O_NOFOLLOW`。
- 同一路径第二 writer 立即抛 `OutputLockError`，不等待、不 takeover、不加载约 1GB
  checkpoint、不使用 GPU，也不创建目标输出目录。既有 per-index random stream、batch
  invariance、atomic PNG、sampling/inference manifest/progress/report schema 与 exact resume
  语义均未改变；顶层 post-eval `flock` 与 component lock 形成双层保护。
- 记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_output_exclusive_lock.md`。
  Windows 跨进程 contention/reacquire、symlink、两个 CLI lock-before-load focused suite
  `32 passed`；完整 suite `952 collected / 946 passed / 6 skipped`，Linux exact-module
  stdin probe、`py_compile`、`git diff --check` 全通过。提交仅含 7 个 owned 文件，三个
  原有 untracked 路径未暂存。
- 该能力仍只在本地 commit，未部署或移动 active training/posteval/readiness checkout；
  concurrency safety 不是 sample-quality evidence，也不授权 full 300K。`2026-08-03
  01:44 CST` 提交后只读复核中 dense 为 step `14,700`、`940,800` samples；`295`
  行 metrics finite/strictly monotonic 且 `samples_seen=step*64`，15K checkpoint 尚未出现。
  pair running/no issues，posteval/readiness waiting，`full_training_launch_allowed=false`；
  GPU compute 仍只有 trainer PID `541878`（`77,970 MiB`），filesystem free bytes
  `183,921,201,152`，未作 signal、restart 或远端写入。

## 2026-08-03 generation evaluator single-writer handoff

- branch `scale/generation-large-capacity` commit
  `06863742154b7caab242a2fc58dcc31fc7969d99` 将同一 component-level
  `cofitok.output_lock.exclusive_output_lock` 扩展到
  `scripts/evaluate_generation_checkpoint.py` 与
  `scripts/evaluate_generation_metrics.py`。checkpoint diagnostic 在 model load/GPU
  work 前、formal torch-fidelity evaluator 在 image enumeration/tree hash/runtime capture
  与 metrics compute 前获取 output-scoped 非阻塞锁；direct CLI 与未来 orchestration
  因而不能绕过顶层 post-eval controller 的 single-writer 语义。
- contention 测试分别证明 checkpoint model loader、generated/real image enumeration 与
  torch-fidelity compute 均不会被触达，且目标 output directory 不会创建。既有
  manifest/completed-result replay/source provenance/gate/comparison/completion-audit 契约不变。
  记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_evaluator_output_exclusive_lock.md`。
- focused suite `51 passed`；完整 suite `954 collected / 948 passed / 6 skipped`；
  `py_compile` 与 `git diff --check` 通过。该 commit 仍只在本地，没有部署或移动 active
  immutable training/posteval/readiness checkout；evaluator single-writer 不是 sample-quality
  evidence，也不授权 full 300K。

## 2026-08-03 sampling selector matched-output lock and dense 15K handoff

- branch `scale/generation-large-capacity` commit
  `496ab6555c9859a59e42e646eb31b87dcf98e01d` 新增
  `cofitok.output_lock.exclusive_output_locks`：多 target 先 canonicalize、拒绝重复，再按
  normalized path 确定序获取；任一后续 target contention 时 `ExitStack` 自动释放已获取
  的全部锁。`scripts/select_generation_sampling_batch.py` 现在在 checkpoint identity/read、
  frozen sampling state 检查、约 2 GB matched checkpoint loading、GPU preflight 或 benchmark
  output 创建之前，同时持有两侧 formal sampling output directory；selector 与 selector、
  selector 与 direct sampler 的 pre-state race 均 fail closed。
- 记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_sampling_selector_output_exclusive_lock.md`。
  reversed target order、duplicate target 与真实 subprocess second-target contention 测试通过；
  focused suite `17 passed`，完整 suite `957 collected / 951 passed / 6 skipped in 276.9s`，
  `py_compile` 与 `git diff --check` 通过。提交仅含 5 个 owned 文件；用户原有三个 untracked
  路径未暂存。
- pro6000 dense step 15K milestone 已独立验证：
  `checkpoint_step_00015000.pt` 为 `1,006,120,214` bytes，SHA256
  `1ad69fc2cc3dbeaf9612d5f6f1ea62e1791ed97454b2e7d856611c1ad76f6a55`；邻接
  integrity sidecar 与 `latest.json` 精确绑定该 filename/bytes/SHA/step，并绑定 training
  branch `scale/generation-stability-50k-preflight`、revision
  `2c2c1f5166b73d4f28df93b276901671ac1a7836`、dataset SHA
  `97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741` 和 runtime SHA
  `d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`。
- `2026-08-03` 本次提交前只读复核中 dense 已到 step `15,300`、`979,200` samples；
  `307` 行 metrics finite/strictly monotonic 且 `samples_seen=step*64`。pair/recovery/watchdog
  running，posteval/readiness waiting，`full_training_launch_allowed=false`；GPU compute 仍只有
  trainer PID `541878`（`77,970 MiB`），filesystem free bytes `182,915,059,712`。未作
  signal、restart、远端写入或额外 GPU evaluation。15K 是健康/可恢复性证据，不是样本
  质量；必须继续等待 exact healthy 50K 后由既有 waiter 运行 EMA formal sampling 与
  FID/IS/precision/recall，且绝不能从该 milestone 授权 full 300K。

## 2026-08-03 formal EMA rollout-stability gate handoff

- branch `scale/generation-large-capacity` commit
  `876854207aa813ed38647f44c4b14db9d33ddad1` 把真正与 sample quality 直接相关的
  matched EMA free-rollout 诊断接入 formal generation gate，而不是继续增加纯运维锁。
  `build_generation_stability_qualification.py` 现在支持 `--weights ema`，并显式区分
  immutable training revision/branch 与 clean post-evaluation revision/branch；四份
  checkpoint/rollout evaluator 报告必须共享后者，训练报告仍绑定原训练身份。
- stability scaling formal protocol 新增 CoFiTok/dense 各 `64` 张固定 val image、seed
  `2029`、EMA、DDIM-100、CFG `1.5`、clipped-x0、bf16 的 teacher/reconstruction/free
  rollout；dormant stability-full formal protocol 同样使用 EMA、`64` 张、DDIM-250。
  qualification 复用既有 tail-two energy、single-token concentration、ordered rank、
  endpoint/validation regression、free-rollout predicted-x0 high-frequency、reconstruction
  amplification、zero-token 与 shuffle mismatch gates，不替代也不放宽 FID/IS/precision/
  recall。
- generation gate schema 升至 v3：`stability_scaling`/`stability_full` 新 gate 必须包含
  passing `rollout_stability_diagnostic`；其 weights/protocol/checkpoint step、1,024-image
  mechanism count、两侧 checkpoint SHA、pair contract 与 evaluator identity 均重验，且
  qualification 的 authoritative path/bytes/SHA256 作为额外 source 绑定并在 replay 时
  重新哈希。历史 schema-v2 gate 仍可回放，避免反向作废 immutable evidence；新 v3
  stability gate 删除 diagnostic 会 fail closed。
- future full post-eval 的两份 rollout 与 qualification 使用
  `run_generation_stage_once.py` receipts，避免高成本诊断在断点恢复时静默漂移。记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_formal_ema_rollout_stability_gate.md`。
  focused suite `88 passed`；完整 suite `967 collected / 961 passed / 6 skipped in 259.6s`；
  `py_compile`、`git diff --check`、两份 exact working-tree runbook 的 pro6000 Linux
  `bash -n` 全部通过。提交仅包含 16 个 owned 文件；三个既有 untracked 路径未暂存。
- 该 commit 仍只在本地，未部署、替换或重启 active training `2c2c1f5`、posteval
  `c1efb12` 或 readiness `5dd3488`；当前冻结 waiter 将继续按其原始流程独占执行，后续
  可用本提交作 source-bound supplemental/future formal diagnosis。`2026-08-03` 提交后
  只读复核中 dense 已到 step `16,350`、`1,046,400` samples；`328` 行 metrics finite、
  strictly monotonic 且 `samples_seen=step*64`，checkpoints 仍为 5K/10K/15K。pair/recovery/
  watchdog running，posteval/readiness waiting，`full_training_launch_allowed=false`；GPU
  compute 仍只有 trainer PID `541878`（`77,970 MiB`），filesystem free bytes
  `182,915,035,136`。未作 signal、restart 或远端写入，full 300K 仍未授权。

## 2026-08-03 terminal completion-bound inference release handoff

- branch `scale/generation-large-capacity` commit
  `b96cfb6fa5bd0d790dc13e36619c3769bef41cb2` 补上 deployable EMA artifact 的最后一个
  consumer trust boundary：现有 artifact 已绑定 scaling training authorization 与 final
  quality release gate，但 export/preflight/smoke 本身是 terminal completion audit 的输入，
  所以之前的 production loader 无法证明该 terminal audit **已经通过**。
- 两条 terminal completion runbook 现在只在 audit 成功后发布 deterministic schema-1
  `release_receipt.json`。receipt 绑定 completion audit path/bytes/SHA、profile/expectations、
  unique passing inference check，以及 CoFiTok/dense 两份 physical artifact 的 bytes/SHA、
  source checkpoint/environment/Git、training/release authorizations 和 smoke PNG hashes；
  创建时重新验证两份 artifact sidecar/bytes，existing receipt 只有 exact reconstruction
  相同才可复用。
- `GenerationSession`、shared loader、routine inference CLI 与 real-forward preflight 新增
  `--completion-receipt` / `--require-completion-authorization`。consumer load 在
  `torch.load` 前重哈希 receipt 与 bound audit、确认所选 artifact 是 audit 授权的唯一
  method identity，并把 receipt/audit 身份传播到 resumable inference manifest/report。
  completion authorization 自动蕴含原有 final release requirement；exact-resume training
  checkpoint 仍是独立必保资产，receipt 不授权任何训练。
- strong-baseline comparison 同步审计确认已有足够边界：CoFiTok/dense 只在
  `matched_training_direct` 内直接比较，D-AR/MAR/ReTok 继续锁在
  `official_pretrained_contextual`，`cross_tier_numeric_ranking_allowed=false`；real set、
  evaluator、formal sampling、compute、GPU contention 与所有 source reports 已绑定，未重复
  修改该链。`ml-training-recipes` 本次把工作重点从更多训练控制锁转向可实际消费的
  quality-complete inference release。
- focused inference/runbook/completion regression 通过；完整 suite 为
  `966 passed / 6 skipped in 270.40s`，`py_compile`、`git diff --check` 通过，两份变更
  runbook 的 Git-clean-filtered working-tree blob 均通过 pro6000 native Linux `bash -n`。
  三个既有 user-owned untracked 路径仍未暂存。
- 该 commit 仍只在本地，没有部署或移动 active immutable training/posteval/readiness
  checkout。`2026-08-03 03:19 CST` 只读复核中 dense 为 step `16,900`、`1,081,600`
  samples；`339` 行 metrics finite/strictly monotonic 且 `samples_seen=step*64`，最新
  epsilon/rollout/total 为 `0.02598897 / 0.01058363 / 0.02704734`，checkpoints 仍为
  5K/10K/15K。pair/recovery/watchdog running，posteval/readiness waiting，
  `full_training_launch_allowed=false`；GPU compute 仅 trainer PID `541878`
  （`77,970 MiB`），filesystem free bytes `182,914,990,080`。没有 signal、restart、
  远端写入或 full 300K authorization。

## 2026-08-03 stability-scaling distribution-support gate handoff

- branch `scale/generation-large-capacity` commit
  `3d2ec9509adb0d0d7f69882444c052c5803d8dee` 将 formal 50K
  `stability_scaling` 决策从“只以 FID 阻断 distribution quality”升级为同时防止
  precision/recall collapse 的 schema-v4 gate。新 gate 必须包含 passing
  `scaling_precision_recall_quality`：CoFiTok precision 与 recall 各至少 `0.10`，且各自
  不得比 matched dense 低超过 `0.05` absolute。
- authorization validator 不信任 gate 的 status/row：它重新计算两项 absolute floors 与
  两项 matched-retention inequality，拒绝更弱的 declared thresholds，并把 named row 的
  precision/recall 与 summary/thresholds 逐项交叉核验。full gate 的既有 `0.30` floors
  和 `0.05` retention 未放宽；generic scaling 语义也未被重写。历史 schema-v2/v3
  stability evidence 仍可 replay，新要求只作用于 schema-v4 `stability_scaling`。
- formal stability 50K post-eval runbook 现显式传入 `0.10/0.10/0.05/0.05` 四个阈值；
  记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_stability_scaling_distribution_support_gate.md`。
  这些是保守的 non-collapse readiness floors，不是 generation-SOTA 质量目标，也不替代
  FID、EMA rollout diagnostic、机制审计或 full-data 50K formal evaluation。
- `ml-training-recipes` 本次把实现选择锁定在实际 distribution support/sample-quality
  证据，而不是继续叠加纯运维控制。focused suite `85 passed`；完整 suite
  `983 collected / 977 passed / 6 skipped in 257s`；`git diff --check` 通过，变更 runbook
  的 Git clean-filter blob `600586b5e783f3c4ad18272e53067333f1bf0900` 在 pro6000 native
  Linux `bash -n` 通过。提交仅含 8 个 owned 文件，三个 user-owned untracked 路径未暂存。
- 该 commit 仍只在本地；active training `2c2c1f5`、frozen posteval `c1efb12`、readiness
  `5dd3488` 均未部署、替换或重启，因此当前 waiter 会继续产生其历史 schema，之后只能
  由本提交做 supplemental/future replay。最近一次只读复核中 dense 为 step `17,050`、
  `1,091,200` samples；`342` 行 metrics finite/strictly monotonic 且
  `samples_seen=step*64`，checkpoints/sidecars 仍为 5K/10K/15K。pair/recovery/watchdog
  running，posteval/readiness waiting，`full_training_launch_allowed=false`；GPU compute
  仅 trainer PID `541878`（`77,970 MiB`），filesystem free bytes `182,915,018,752`。
  没有 signal、restart、远端写入或 full 300K authorization。

## 2026-08-03 frozen stability distribution-support replay handoff

- branch `scale/generation-large-capacity` commit
  `d6684f7c3347a79ded369a164738581a5404fe09`（short `d6684f7`）为当前冻结
  schema-v2 post-eval 增加独立、CPU-only、source-bound 的 precision/recall 补充审计，
  无需也禁止改写 active `c1efb12` checkout。入口为
  `scripts/build_generation_stability_distribution_support.py`，核心实现位于
  `cofitok.generation.distribution_support`。
- builder 首先重新哈希原 gate 已绑定的全部六份 source reports，再读取 exact
  CoFiTok/dense 10K generation metrics；它验证 frozen schema-v2 与 current schema-v3
  report、50K EMA checkpoint/sidecar、10K completed progress、DDIM-100/CFG/bf16/clipped-x0/
  random-stream formal protocol、K8 与 dense-K1 prefix identity、matched real set、
  torch-fidelity evaluator、clean evaluator/sampler Git、inline runtime hash 与 matched
  runtime。原 gate summary 的 FID/IS/precision/recall 还必须与 exact source bytes 一致；
  输出前再次重新哈希 gate 与全部 bound sources。
- scientific checks 使用与 schema-v4 stability gate 相同且不能放宽的边界：precision、
  recall 各至少 `0.10`，并且各自不得比 matched dense 低超过 `0.05`。report 将
  supplemental result、base-gate result 与二者 conjunction 分成机器可读字段，同时固定
  `scaling_authorization_evaluated=false`、`supplemental_non_authorizing=true`、
  `replaces_generation_gate=false`、`replaces_rollout_stability_qualification=false`、
  `full_training_launch_allowed=false`；即使 supplemental 自身 pass，也不能被解释成扩训
  授权。记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_frozen_stability_distribution_support_replay.md`。
- `ml-training-recipes` 本次将工作焦点继续约束在当前 frozen run 最终能够产出的真实
  distribution-support/sample-quality 证据，而不是追加 future-only 运维控制。dedicated
  tests `17 passed`，gate-related suite `89 passed`，完整 suite
  `1000 collected / 994 passed / 6 skipped in 253.1s`；`py_compile` 与
  `git diff --check` 通过。无 shell runbook 变更，因此没有新增 `bash -n` 对象。提交仅含
  6 个 owned 文件，三个 user-owned untracked 路径仍未暂存。
- 该实现仍只在本地，正式 precision/recall 数值要等 exact healthy 50K 后由 frozen
  waiter 生成，当前不能声称已有 sample-quality pass。提交后 pro6000 只读复核中 dense
  为 step `17,950`、`1,148,800` samples；`360` 行 metrics finite/strictly monotonic 且
  `samples_seen=step*64`，最新 epsilon/rollout/total 为
  `0.02933354 / 0.02805981 / 0.03213952`，checkpoints/sidecars 仍为 5K/10K/15K。
  training/controller/posteval/readiness 四个 checkout 均保持各自 clean immutable identity；
  pair/recovery/watchdog running，posteval/readiness waiting，
  `full_training_launch_allowed=false`。GPU compute 仍只有 trainer PID `541878`
  （`77,970 MiB`），GPU free memory `19,271 MiB`，filesystem free bytes
  `182,915,002,368`。没有 signal、restart、远端写入、部署或 full 300K authorization。

## 2026-08-03 stability 50K matched training-trajectory handoff

- branch `scale/generation-large-capacity` commit
  `275366375da1ecaf1a6fc0f9eb9ffa39a3e012ea`（short `2753663`）把当前实际
  scaled training evidence 固化为 source-bound snapshot，而不再用新增 future-only
  控制替代质量观察。报告目录：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/training_trajectory_snapshot_2026-08-03/`；
  记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_stability_50k_training_trajectory_snapshot.md`。
- snapshot 绑定 completed CoFiTok JSONL 全部 `1,002` 行/step 50K 与 active dense
  JSONL immutable prefix `361` 行/through step 18K 的 path/bytes/SHA256。后续 dense
  继续 append 后再次验证 prefix bytes SHA 均精确通过；两侧有限、step 严格递增，且
  `samples_seen=step*64`。CSV 的 `18` 个同 batch/event scheduled validation 点也由
  bytes/SHA 绑定并通过 JSON/CSV 一致性检查。
- 1K–18K matched validation 的 CoFiTok/dense epsilon-MSE ratio 均值
  `0.996941`、中位数 `0.999905`、范围 `[0.954921, 1.011452]`；最大 CoFiTok
  regression 只有 `+1.1452%`。这证明当前 factorized 与 dense predictor 在共同 18K
  one-step validation 空间紧密匹配，但**不是** sample-quality pass；2026-07-29 失败
  结果已经证明近似 endpoint MSE 可以与严重 FID 回退共存。CoFiTok 30K 后 validation
  均值下降也不能归因于 EMA teacher，因为 event batch、LR 与整体训练进度同时变化。
- 两个 active training 目录实际图像数均为 `0`。exact frozen posteval runbook 位于
  `c1efb12c6640f2d2d62ac7e9982c8804d96e7289`、SHA256
  `6cf80146c1696aa0bed9a4a370b41c42aa5be9dc03f753c4c81ce6b54278b064`；它会生成
  双方法各 10K EMA DDIM-100 样本、64-image prefix diagnostic、visual audit、1,024-image
  checkpoint eval 与 FID/IS/precision/recall，但其 schema-v2 流程**不包含**新版 EMA
  free-rollout qualification。后者必须在 frozen posteval 完成后由 clean checkout 作为
  source-bound supplemental 单独运行，不能伪称当前 waiter 已经提供。
- `ml-training-recipes` 本次直接促使工作回到真实 training/sample evidence：没有新增模型
  控制代码。JSON/CSV consistency、claim boundary、remote prefix rehash 与
  `git diff --check` 均通过；无代码或 runbook 变更，因此未重复跑全套 pytest/`bash -n`。
  提交仅含 4 个 owned 小型 evidence 文件，三个 user-owned untracked 路径仍未暂存。
- 提交后 pro6000 只读复核中 dense 已到 step `18,200`、`1,164,800` samples；`365`
  行 metrics finite/strictly monotonic，checkpoints/sidecars 仍为 5K/10K/15K。pair/recovery/
  watchdog running，posteval/readiness waiting，`full_training_launch_allowed=false`；GPU
  compute 仍只有 trainer PID `541878`（`77,970 MiB`），GPU free `19,271 MiB`，
  filesystem free bytes `182,914,994,176`。没有 signal、restart、GPU evaluation、远端
  写入、部署或 full 300K authorization。

## 2026-08-03 matched-training budget claim-boundary handoff

- branch `scale/generation-large-capacity` commit
  `231cfe09705617cf82f3fa284685b9e3f0c2a98b`（short `231cfe0`）把最终 strong-baseline
  comparison 升级到 schema v7，明确区分“matched optimizer steps/training images”与
  “equal wall-clock/GPU-hours/FLOPs”。JSON 的 `training_budget_policy` 与每个 direct CSV
  row 均固定 `compute_matched_claim_allowed=false`；time、throughput、peak VRAM 只能是
  `measured_outcomes`。若 dataset、resolution、effective batch、steps 或 images 不一致，
  builder 直接 fail closed；completion audit 会重算该 policy 并拒绝 equal-compute overclaim。
- direct quality panel 仍可在 matched data/shared-backbone contract/optimizer schedule/effective
  batch/steps/images/formal evaluator 下比较；cost-efficiency ranking 则仅在 terminal GPU
  contention evidence 证明连续、exclusive coverage 时允许。即使允许比较实测效率，也不把
  实验改写成预先等化的 compute allocation。D-AR/MAR/ReTok 继续仅属
  `official_pretrained_contextual`，`cross_tier_numeric_ranking_allowed=false`。
- 受控 runtime benchmark source
  `artifacts/reports/generation/stability_probe_2026-07-29/runtime_benchmark_rollout_x0_u2_ema_teacher_v2_2026-07-30/benchmark_summary.json`
  的 SHA256 为
  `e5a88a1e7d18bef30de56ee43b46e94447c76302a84c0aa09d0928495fbaab49`；CoFiTok/dense
  为 `22.124531 / 24.307071 img/s`，即 CoFiTok optimizer-step time 约为 dense 的
  `1.09865x`，peak VRAM 为 `15,232,468,992 / 15,036,313,600` bytes。这些是实际成本，
  不是 matched budget。记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_training_budget_claim_boundary.md`。
- focused comparison/completion suite 通过；完整 suite 为
  `1003 collected / 997 passed / 6 skipped in 260.2s`，`git diff --check` 通过。无 runbook
  变更，因此没有新的 `bash -n` 对象；提交仅含 9 个 owned 文件，三个既有 user-owned
  untracked 路径仍未暂存。
- 提交前 pro6000 只读复核中 dense 已到 step `18,350`、`1,174,400` samples、`368`
  行 metrics；pair/recovery running，posteval/readiness waiting，
  `full_training_launch_allowed=false`。GPU compute 仍只有 trainer PID `541878`
  （`77,970 MiB`），filesystem free bytes `182,914,985,984`。当前 common-prefix raw
  elapsed 因 dense 曾受 FieldScope contention 而不支持架构效率排序；没有 signal、restart、
  远端写入、部署或 full 300K authorization。

## 2026-08-03 frozen post-eval supplemental pipeline handoff

- branch `scale/generation-large-capacity` commit
  `e448b7e5ee93da38573f6da2a45dd540ee0613f7`（short `e448b7e`）把当前 frozen
  schema-v2 post-eval 缺失的两项真实质量证据收口成一个独立、可恢复、source-bound 的后续链：
  precision/recall distribution-support replay，以及 matched EMA free-rollout stability
  qualification。入口为
  `artifacts/runbooks/generation_stability_frozen_50k_supplemental_after_posteval.sh`；它不修改
  active training/posteval/readiness checkout。
- runbook 首先精确验证 v4 waiter 的 terminal pass、child exit、training/evaluation identity 和
  `formal_300k_allowed=false`，再验证原始 promotion gate 并拒绝 busy GPU。随后在同一 clean
  supplemental revision 下重跑 CoFiTok/dense 各 1,024-image、timestep-500 EMA checkpoint
  evaluation，并新增双方各 64-image、EMA DDIM-100、seed 2029、CFG 1.5、teacher CFG 1、
  batched CFG、clipped-x0、bf16 rollout。重跑 checkpoint evaluation 是必要条件：后续
  qualification 要求 checkpoint 与 rollout reports 共享 exact evaluation Git identity，不能把
  frozen checkout 的旧 checkpoint report 与新 rollout report 静默混用。
- 两个 checkpoint evaluation、两个 rollout evaluation 和 combined rollout qualification 共 5 个
  high-cost stage 均由 immutable stage receipt 绑定 command/project/input/output tree；CPU-only
  distribution replay 从原 gate 的 exact bound metrics 重建。最终 combined report 会在构建前后
  重哈希 4 份直接输入和所有 nested sources；只有 base gate、distribution support、EMA rollout
  三者都 pass 才报告 supplemental pass，任一科学失败都保留为 hold。
- 该链固定
  `supplemental_non_authorizing=true`、`replaces_generation_gate=false`、
  `replaces_readiness=false`、`scaling_authorization_evaluated=false`、
  `full_training_launch_allowed=false`；不含 trainer，不重跑 CoFiTok，也不能授权或启动 full 300K。
  `ml-training-recipes` 本次使实现继续聚焦 exact EMA samples、matched sampler/checkpoint identity 与
  可复现评估证据，而不是继续增加泛化的运维锁。
- focused module/CLI/runbook/gate suite 为 `38 passed`；完整 suite 为
  `1011 collected / 1005 passed / 6 skipped in 259.75s`。新 module 与两个 CLI 通过
  `py_compile`，`git diff --check` 通过；staged LF runbook blob 通过 pro6000 native Linux
  `bash -n` binary-pipe 检查且未写远端文件。提交仅含 8 个 owned 文件，三个 user-owned
  untracked 路径仍未暂存。
- 提交后 pro6000 只读复核中 dense 已到 step `18,900`、`1,209,600` samples；metrics 有限、
  step 严格递增且 `samples_seen=step*64`，checkpoints/sidecars 仍为 5K/10K/15K，latest 仍绑定
  15K SHA `1ad69fc2cc3dbeaf9612d5f6f1ea62e1791ed97454b2e7d856611c1ad76f6a55`。
  pair/recovery/watchdog running，posteval/readiness waiting，
  `full_training_launch_allowed=false`；GPU compute 仍只有 trainer PID `541878`
  （`77,970 MiB`），filesystem free bytes `182,914,916,352`。没有 signal、restart、GPU
  evaluation、远端写入、部署或 full 300K authorization。

## 2026-08-03 full-launch quality prerequisite handoff

- branch `scale/generation-large-capacity` commit
  `2fddc20c5945005b9cb0b544623777ac620354be`（short `2fddc20`）把 frozen
  supplemental 从可忽略的 side report 提升为 full 300K launch receipt 的必需负向质量前置。
  `full_training_launch_receipt.json` 升级为 schema v3、11 个 exact sources；missing/hold/fail/
  drift 的 supplemental 一律不能生成或重放 launch receipt，但 supplemental pass 本身仍固定
  `supplemental_non_authorizing=true`、`required_for_full_training_launch=true`、
  `full_training_launch_allowed=false`，不能替代 gate/readiness/deployment/storage 或人工授权。
- 新增独立控制面验证器
  `scripts/verify_generation_stability_frozen_supplemental.py`：重哈希 combined report、4 个直接
  reports、raw posteval waiter 及 distribution/EMA-rollout 的全部 nested sources；要求 exact
  physical promotion gate、clean builder Git、frozen thresholds/protocol、7 个 distribution checks、
  9 个 rollout gates 和 4 个 combined checks 全部通过。它不依赖后来新增的
  `cofitok.generation.frozen_supplemental`，因此可进入不改变 trainer package 的专用 control target。
  launch builder、resume validator、posttraining supervisor 与 terminal completion audit 均消费或
  独立重放该 binding；`ml-training-recipes` 本次继续把决策边界绑定在真实 EMA sample/checkpoint
  质量证据上，而不是泛化运维状态。
- readiness ancestry 审计证明当前 HEAD **不能**直接作为 active
  `5dd3488ac9b30274f4960195e252cc9fdb161002` readiness 的 target：50 个后继提交中，最后同时保持
  training-critical manifest 与 training execution suffix 的 revision 为
  `27edb632d8cb9a642040ba6fcd3b1a0d682e6c79`；仅 execution suffix 的最后兼容 revision 为
  `5029487a9a3c91c5447e3da3f9ee9ab0ff8b7425`。后续必须构造专用兼容 control target，或在最终
  target 上获取 fresh readiness；不得把本提交或当前 waiter 伪称为已桥接/已授权。
- 42 项 focused receipt/bridge/runbook/audit/supervisor suite 通过；supervisor nested replay 加固后的
  11 项定向 suite 也通过。最终完整 suite 为
  `1016 collected / 1010 passed / 6 skipped in 269.6s`，相关 scripts 通过 `py_compile`，
  `git diff --check` 通过；staged LF full-training runbook 通过 pro6000 native Linux `bash -n`。
  提交仅含 18 个 owned 文件，三个既有 user-owned untracked 路径仍未暂存。
- 提交前 pro6000 只读复核中 dense 已到 step `20,300`、`1,299,200` samples；metrics finite/
  strictly monotonic 且 `samples_seen=step*64`，pair/recovery/watchdog running、posteval/readiness
  waiting、issues 为空、`full_training_launch_allowed=false`。新的 20K checkpoint 与 sidecar 已落盘，
  `latest.json` 绑定 SHA256
  `86b4f9bd3a76168d94c9c3420346ca1c66dbab9a024dd2d45221edad20162f1f`；GPU compute 仍只有
  trainer PID `541878`（`77,970 MiB`），filesystem free bytes `182,914,940,928`。没有 signal、
  restart、GPU evaluation、远端写入、部署或 full 300K authorization。

## 2026-08-03 readiness-compatible quality-control target handoff

- active readiness source
  `5dd3488ac9b30274f4960195e252cc9fdb161002` 不能直接桥接 main development HEAD；现已在
  branch `scale/generation-stability-full-control-quality-27ed` 准备专用本地 target commit
  `9019dd3f0f504e799c03496ec41a653b61deaa02`（short `9019dd3`），parent 为最后同时保持
  training-critical manifest/训练执行段兼容的
  `27edb632d8cb9a642040ba6fcd3b1a0d682e6c79`。main branch 以 commit
  `f6f4f006ace22540438ee2bc07e72967022db174`（short `f6f4f00`）记录并暴露该 target：
  `docs/records/2026-08-03_generation_readiness_compatible_quality_control_target.md`。
- 对 actual committed target 的 bridge audit 证明 `5dd3488...` 是 ancestor，66 个
  training-critical Git blobs 逐项相同，full-training execution suffix byte-identical，SHA256
  `1a4e559c3c8828ceee9907f45e2de2d4513a77f37e4458eb50755ee53ab6382c`。唯一受控变化位于
  authorization preamble：readiness bridge + frozen supplemental 必需、launch receipt schema v3、
  sample reserve 从 readiness-time `16,384` 升至 completion-time `116,640`。因此 target 可由现有
  fail-closed bridge builder 消费，但尚未生成 bridge/deployment/launch receipt，更未授权训练。
- target 上 modified scripts 通过 `py_compile`，42 个 focused tests 通过；完整 target suite 为
  `865 collected / 861 passed / 4 skipped in 181s`，staged LF runbook 通过 pro6000 `bash -n`，
  `git diff --check` 通过。首次 flat short-path checkout 的 4 个失败仅因 legacy paper tests 从父目录
  推导 `paper/`；未修改测试，改用正确 `parent/CoFiTok-internal` 临时布局后全部通过。临时 junction、
  worktree 和空父目录均已删除，canonical `paper/` 未改动；branch commit 保留，可恢复 checkout。
- `ml-training-recipes` 本次约束实现只搬运质量/checkpoint 控制面，不搬运后来 trainer package 或
  training execution 变化，确保长训与 exact-resume 语义仍绑定 source readiness。该 target 尚未 bundle
  或部署；必须等 dense 50K、formal posteval、source-bound supplemental 全部完成，再由独立人工授权
  决定是否生成 deployment/bridge/launch receipt。milestone/supplemental pass 仍不能单独授权 full 300K。
- 收口时 pro6000 只读状态：dense step `20,700`、`1,324,800` samples、`415` 行 metrics，finite/
  strictly monotonic 且 `samples_seen=step*64`；pair/watchdog running、issues 为空，posteval/readiness
  waiting、`full_training_launch_allowed=false`。GPU compute 仍只有 trainer PID `541878`
  （`77,970 MiB`），filesystem free bytes `182,914,904,064`。没有 signal、restart、GPU evaluation、
  远端写入、bundle/deploy 或 full 300K authorization。

## 2026-08-03 readiness-compatible incremental-bundle rehearsal handoff

- branch `scale/generation-large-capacity` commit
  `3b41d52167b00ec120d4bb68608633496988883a`（short `3b41d52`）记录专用 target 的
  prerequisite-aware bundle rehearsal。receipt：
  `CoFiTok-internal/artifacts/reports/generation/readiness_compatible_quality_control_target_2026-08-03/incremental_bundle_rehearsal_receipt.json`，
  receipt SHA256 为
  `01354831d0be1a2ca64cdb07d6b72202de994454b0235ff3455b0a2253df9460`；同步更新
  `docs/GENERATION_SYSTEM.md` 与原 target record。提交只含 3 个 owned 小型文件，三个既有
  user-owned untracked 路径仍未暂存。
- ephemeral incremental bundle 精确覆盖 readiness source
  `5dd3488ac9b30274f4960195e252cc9fdb161002` 到 target
  `9019dd3f0f504e799c03496ec41a653b61deaa02`：`64,538` bytes、SHA256
  `8beb9d60757e84ded9d426d6b086bef20b1b206a3dc9cb55b79e19ebd0465ad3`、9 commits、
  129 objects。`git bundle list-heads` 只有一个 advertised ref
  `refs/heads/scale/generation-stability-full-control-quality-27ed`，`git bundle verify` 明确要求
  pinned prerequisite `5dd3488...`；local 与 remote bytes/SHA 完全一致。
- pro6000 上只在 `/tmp/cofitok-stability-full-control-quality-9019dd3.bundle` 暂存 bundle，使用
  authoritative readiness checkout 做只读 `git bundle verify`。验证前后 checkout 均为 clean
  `scale/generation-large-capacity@5dd3488...`；没有 fetch、merge、HEAD 移动或文件修改，GPU
  compute 前后均只有 dense trainer PID `541878`（`77,970 MiB`）。remote 与 local 临时 bundle
  均已精确删除，项目内 user-owned `.codex-bundles/` 未触碰。
- `ml-training-recipes` 的 checkpoint/version reproducibility 原则促使本次把“存在兼容 commit”
  提升为“源 revision 可验证消费其增量对象”的实证，但该 rehearsal **不是部署**。没有 bridge、
  deployment receipt、launch receipt 或 full 300K authorization；`full_training_launch_allowed=false`。
  后续仍必须等待 dense exact 50K、frozen formal posteval、source-bound supplemental 与独立人工授权。
- 收口时 pro6000 只读复核中 dense 已到 step `20,950`、`1,340,800` samples、`420` 行 metrics；
  pair/watchdog running、issues 为空，posteval/readiness waiting，
  `full_training_launch_allowed=false`。GPU compute 仍只有 trainer PID `541878`
  （`77,970 MiB`），filesystem free bytes `182,914,904,064`；readiness checkout 仍 clean 固定在
  `5dd3488...`，remote/local 临时 bundle 均不存在。没有 signal、restart、GPU evaluation、fetch、
  merge、部署或 full 300K authorization。

## 2026-08-03 frozen supplemental waiter handoff

- branch `scale/generation-large-capacity` commit
  `c212b9e2b64d1b302b17a9d4e30a296d773d4215`（short `c212b9e`）补齐 frozen formal
  posteval 后新增 sample-quality supplemental 的 source-bound 执行协调器：
  `scripts/run_generation_stability_frozen_supplemental_waiter.py` 与
  `artifacts/runbooks/generation_stability_frozen_50k_supplemental_waiter.sh`；记录：
  `docs/records/2026-08-03_generation_frozen_supplemental_waiter.md`。
- waiter 先验证 exact clean supplemental checkout、training/frozen-evaluation identity 与 posteval
  terminal contract，再验证当前已经排队的 large-capacity readiness waiter identity/freshness；只有
  readiness GPU 阶段进入 terminal（pass 或 failed）且 GPU 完全 idle 后，才启动既有
  `generation_stability_frozen_50k_supplemental_after_posteval.sh`。因此不会与 readiness CUDA
  qualification 抢 GPU；child 自身仍重复 busy-GPU 检查，exit `9`/`75` 作为 transient 回到等待。
- supplemental 产物可能是 scientific `pass` 或 `hold`。waiter 把 `hold` 保留为执行完成但质量不通过，
  不会改写成 pass；scientific pass 则额外由 standalone verifier 重放 direct/nested source bytes。
  status 永久固定 `supplemental_non_authorizing=true`、`full_training_launch_allowed=false`；代码中没有
  trainer、full-300K runbook、launch receipt 或 training authorization 输入。它只解除真实质量诊断
  容易竞态/漏跑的问题，不授权 scale-up。
- `ml-training-recipes` 的 fixed evaluation/checkpoint/version 原则使本次优先补真实 EMA rollout 与
  distribution-support 证据的可靠执行，而不是继续扩张训练控制面。waiter 通过 `py_compile`；focused
  suite `35 passed`；完整本地 suite `1,018 passed / 6 skipped in 260.82s`；staged LF runbook 通过
  pro6000 native `bash -n`。提交只含 6 个 owned 文件，三个既有 user-owned untracked 路径未暂存。
- 本提交尚未部署或启动 waiter。后续必须先建立/attest exact clean supplemental checkout，再在 dense
  50K、frozen posteval 与现有 readiness CUDA 阶段结束后恢复该 waiter；不得修改或替代现有 frozen
  waiters。收口时 pro6000 只读状态：dense step `21,400`、`1,369,600` samples、`429` 行 metrics，
  finite/strict 且 `samples_seen=step*64`；pair/watchdog running、issues 为空，posteval/readiness waiting，
  `full_training_launch_allowed=false`。GPU compute 仍只有 trainer PID `541878`（`77,970 MiB`），
  filesystem free bytes `182,914,904,064`。没有 signal、restart、GPU evaluation、部署或 full 300K
  authorization。

## 2026-08-03 frozen supplemental checkout attestation handoff

- branch `scale/generation-large-capacity` commit
  `e10a06ead21a6335ff49295f84caa19cd0b04049`（short `e10a06e`）收口 frozen
  supplemental 的独立 Linux evaluation checkout 证据。权威 machine-readable receipt：
  `CoFiTok-internal/artifacts/reports/generation/stability_frozen_supplemental_checkout_2026-08-03/checkout_attestation_receipt.json`；
  记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_frozen_supplemental_checkout_attestation.md`。
- source checkout
  `/tmp/cofitok-stability-schedule-audit-c1efb12` 保持 clean
  `scale/generation-stability-50k-posteval-v4@c1efb12c6640f2d2d62ac7e9982c8804d96e7289`；
  新 target checkout
  `/tmp/cofitok-stability-frozen-supplemental-c212b9e/CoFiTok-internal` 为 clean
  `scale/generation-large-capacity@c212b9e2b64d1b302b17a9d4e30a296d773d4215`。target 由
  `git clone --no-hardlinks --no-checkout` 加单一 advertised head 的 prerequisite-aware bundle 建立，
  没有 fetch/merge/移动 source checkout HEAD。
- incremental bundle 为 `650,894` bytes，SHA256
  `4ead1d6ebb4e320a975515fb71faba936cddc07af6ee7ab3d1cc173cc256a915`，含 64 commits / 972
  objects，只广告 `scale/generation-large-capacity@c212b9e...`，唯一 prerequisite 为 `c1efb12...`。
  local/remote `git bundle verify` 均通过，两个临时 bundle 均已删除；user-owned `.codex-bundles/`
  未触碰。
- remote Linux 在 `CUDA_VISIBLE_DEVICES=""` 下通过 waiter `py_compile`、focused `70 passed`
  （0 failures/errors/skips，`34.006s`）与全部 `104/104` tracked runbook syntax；JUnit、log、syntax
  JSON 已同步且 receipt 重新绑定 bytes/SHA。waiter entrypoint SHA256 为
  `f937cafeac19b84bf768c72c8c073149a78526e0aaea117c87fd37e149c6a887`，runbook SHA256 为
  `1d780d06a4391d5f80aa1a1e2dfcd46929dbdf3fb3f0a17d8a73f895ee33d8e6`。
- `ml-training-recipes` 的 exact checkpoint/EMA/version reproducibility 原则促使本次先建立可审计、
  source-bound 的 evaluation checkout，再等待既有 GPU 队列；checkout 已准备但保持 dormant，未启动
  supplemental waiter 或任何 supplemental GPU evaluation。它不是 large-capacity deployment，也没有
  bridge/deployment/launch receipt；`full_training_launch_allowed=false`，不得由 supplemental pass 单独授权
  full 300K。
- 收口时 pro6000 只读状态：dense step `21,700`、`1,388,800` samples、`435` 行 metrics；pair/
  watchdog running、issues 为空，posteval/readiness waiting，`full_training_launch_allowed=false`。GPU
  compute 仍只有 trainer PID `541878`（`77,970 MiB`），filesystem free bytes
  `182,914,899,968`。没有 signal、pause、restart、GPU evaluation、其他项目进程修改或 full 300K
  authorization。

## 2026-08-03 frozen supplemental waiter launch handoff

- branch `scale/generation-large-capacity` commit
  `754ff66442a62c18b840921be5798d5b8dfa7f14`（short `754ff66`）记录 attested frozen
  supplemental waiter 的唯一远端恢复。receipt：
  `CoFiTok-internal/artifacts/reports/generation/stability_frozen_supplemental_waiter_launch_2026-08-03/launch_receipt.json`；
  exact status snapshot 与记录同目录/见
  `CoFiTok-internal/docs/records/2026-08-03_generation_frozen_supplemental_waiter_launch.md`。
- pro6000 上 detached waiter PID `861117`（PPID `1`）从 clean
  `/tmp/cofitok-stability-frozen-supplemental-c212b9e/CoFiTok-internal` 启动，固定
  `scale/generation-large-capacity@c212b9e2b64d1b302b17a9d4e30a296d773d4215`。它持有 fd `3`
  对应
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/frozen_posteval_supplemental/.supplemental_waiter.json.cofitok-output.lock`；
  matching process 精确为 1，child 精确为 0。
- authoritative status：
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/frozen_posteval_supplemental/supplemental_waiter.json`。
  初始写入和完整 300 秒 poll 后均为 `status=waiting`、
  `detail=waiting_for_frozen_postevaluation`、`child_pid=null`、
  `supplemental_non_authorizing=true`、`full_training_launch_allowed=false`。同步的 1,804-byte snapshot
  SHA256 为 `3b5be6e71df61ef49f6c81ec9be540b4cebe0b2f2927447af6e368fa669a87e3`。
- waiter 只有同时看到 frozen posteval complete、既有 readiness GPU stage terminal、GPU idle 才能启动
  `generation_stability_frozen_50k_supplemental_after_posteval.sh`；child 仍重复 GPU-idle 检查并持独立锁。
  因此本次只恢复 CPU coordination，不抢 dense/readiness GPU、不改变 checkpoint/EMA identity。scientific
  hold/failure 不得自动重启；pass 仍只是必要的非授权 quality prerequisite，绝不能单独授权 full 300K。
- `ml-training-recipes` 的 exact checkpoint/EMA/version 原则使本次先完成 clean checkout attestation，再恢复
  ordered waiter；没有改训练 recipe、没有重跑 CoFiTok、没有 signal/pause/restart trainer 或其他项目进程。
- 最终只读复核时 dense step `22,100`、`1,414,400` samples、`443` 行 metrics；pair running /
  `dense_identity_training`、issues 为空，posteval/readiness waiting，supplemental waiter waiting 且无 child，
  `full_training_launch_allowed=false`。GPU compute 仍只有 trainer PID `541878`（`77,970 MiB`），
  generation filesystem free bytes `182,914,883,584`；supplemental status 已再次刷新至
  `2026-08-02T22:54:52.358240+00:00`，仍为唯一持锁 waiter 且无 child。
- 两次通过 Codex `automation_update` 扩展既有 heartbeat 的调用均在桌面端无返回并被有限等待后终止；
  只读复核 `C:/Users/zixi-/.codex/automations/cofitok-dense-recovery-launch-monitor/automation.toml`
  仍是原 dense/posteval/readiness prompt，未被部分改写。不要直接编辑该 TOML，也不要创建重复 automation；
  在工具恢复并成功更新前，每次现有 heartbeat 检查都必须额外读取上述 authoritative
  `frozen_posteval_supplemental/supplemental_waiter.json`、验证 PID/argv/cwd/lock/checkout identity、无提前
  child 以及两个非授权 flag。

## 2026-08-03 EMA inference export exact-recovery handoff

- branch `scale/generation-large-capacity` commit
  `bd23d1f0e9dd4890888df17df7e5ffc492199134`（short `bd23d1f`）补齐正式 EMA-only 推理制品导出的
  component lock、pre-deserialization export manifest、显式 exact resume 与 terminal consumption。
  记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_inference_export_manifest_resume_lock.md`；使用说明：
  `CoFiTok-internal/docs/INFERENCE.md`。
- 每个 artifact 现在必须同时有 `.integrity.json` 和 `.export_manifest.json`。manifest 在
  `torch.load` 前绑定 source checkpoint/sidecar bytes+SHA、step、runtime/Git、training authorization、
  exact target、full release gate 及 exporter runtime/Git；相邻非阻塞 OS lock 在任何 source 哈希或
  反序列化前拒绝并发 writer。完整重放不改 artifact/sidecar/manifest bytes 或 mtime。
- partial artifact XOR sidecar 只有在 existing manifest 与当前请求完全一致且显式 `--resume` 时，才会在
  持锁状态删除这两个精确目标并重建；缺 manifest 的 partial 永不删除，完整但 corrupt/tampered 的 pair
  即使 `--resume` 也只失败不修复，manifest/source/gate/export environment 任一漂移均在反序列化前拒绝。
  正式 stability runbook 把 manifest 同时列为 export stage output 与 preflight/smoke input；legacy full
  runbook 也固定 `--resume` 并要求两个 manifest 存在。
- large-scale 与 stability terminal completion audit 会重新打开并验证 manifest 的物理 descriptor、
  source/gate binding 与 export report；release receipt 进一步绑定 source checkpoint SHA 和 manifest。
  receipt 创建时重放完整 source-aware 验证，routine consumer 重新哈希 manifest 但不要求大 source
  checkpoint 继续在线，因此正式 EMA artifact 在 source checkpoint 归档后仍可携带完整 release provenance。
- `ml-training-recipes` 的 EMA-for-inference、完整 checkpoint/version identity 与 exact-resume 原则使本次
  优先消除“artifact 已原子落盘、sidecar 尚未写入”这一真实 crash window，而不是增加会掩盖漂移的
  overwrite 路径。focused suite `112 passed`；完整本地 suite 为
  `1,035 collected / 1,029 passed / 6 skipped in 263.6s`；相关 Python 通过 `py_compile`，全部
  `104/104` runbook 在 pro6000 隔离 `/tmp` LF 副本通过 `bash -n`，`git diff --check` 通过。
- 提交只含 12 个 owned 文件；`.codex-bundles/`、既有 2026-07-29 acceptance 目录与记录仍未暂存。
  本次没有部署新 revision、没有修改任何 remote training/evaluation checkout、没有 signal/restart GPU
  进程、没有重跑 CoFiTok，也没有授权或启动 full 300K；`full_training_launch_allowed=false` 保持不变。
- 提交后 pro6000 只读复核中 dense 已到 step `23,050`、`1,475,200` samples、`462` 行 metrics；
  pair/watchdog running、issues 为空，posteval/readiness/supplemental 均 waiting，supplemental child 为空，
  `full_training_launch_allowed=false`。GPU compute 仍只有 trainer PID `541878`（`77,970 MiB`），
  filesystem free bytes `182,914,805,760`。语法检查用 `/tmp` 副本已在验证后按精确路径删除，可恢复性
  不适用；没有删除任何项目证据或训练产物。

## 2026-08-03 matched 20K schedule-trajectory handoff

- branch `scale/generation-large-capacity` commit
  `322e36ab11facc96fac9f98982d689b8df5c6814`（short `322e36a`）记录当前 stability dense recovery
  到达 checkpoint-aligned 20K 后的 exact matched
  training trajectory。证据：
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_20k_schedule_trajectory_2026-08-03/`；
  记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_stability_matched_20k_schedule_trajectory.md`。
- 两边都以 immutable `scale/generation-stability-50k-preflight@2c2c1f5166b73d4f28df93b276901671ac1a7836`
  固定到 step `20,000`、`1,280,000` images、`401` 行 metrics 与 `20` 个同 step/event/batch/noise-seed/
  image-count validation events；全数 finite、严格递增并满足 `samples_seen=step*64`。报告 SHA256：
  `7c682e4ca5dba5634b003f6ff1ee06801963454341247b74e31bc35f58ab483a`。
- 1K--20K validation epsilon mean 为 CoFiTok `0.0319879465`、dense `0.0320814812`，ratio-of-means
  delta `-0.291554%`，lower-event count `10/10`；rollout full-scale 的 10K--20K 子段 mean delta
  `-0.481945%`，但 exact 20K endpoint 为 `+0.185878%`。因此结论仅是 closely matched、无可见
  training-space collapse，不能推出 free-rollout/sample quality；EMA-teacher 要到 30K 才开始。
- dense 20K checkpoint 为 `1,006,120,214` bytes，SHA256
  `86b4f9bd3a76168d94c9c3420346ca1c66dbab9a024dd2d45221edad20162f1f`；sidecar 与
  `latest.json` 精确绑定同 step/bytes/SHA/dataset/runtime/Git。controller PID `511801` 的 fd 6
  仍持 `dense_recovery.lock` advisory write flock。
- `tests/test_build_generation_matched_training_trajectory.py` 为 `13 passed`；从 5 个冻结源文件重建
  `trajectory_report.json` 字节相同、SHA 相同。提交只包含本轮 7 个 owned 文件；`.codex-bundles/`、
  既有 2026-07-29 acceptance 目录与记录保持未跟踪且未暂存。
- 本轮只读复核时 live dense 已到 step `23,100`、`1,478,400` samples；pair/watchdog running、issues
  为空，posteval/readiness/supplemental waiter 均 waiting 且无 child，三个路径继续保持
  `full_training_launch_allowed=false`。GPU compute 只有 trainer leader PID `541878`（`77,970 MiB`，
  余量约 `19,271 MiB`），filesystem free bytes 约 `182,914,859,008`。没有修改、signal、pause 或
  restart 任一进程，没有重跑 CoFiTok，也没有授权/启动 full 300K。

## 2026-08-03 generation class-fidelity gate handoff

- branch `scale/generation-large-capacity` 的实现提交为
  `9e4d62271fa31facf998d8cdefb0f5d8cbb4ee32`，Linux test-isolation 修复为
  `2448ce5e6a96beca6762eba1c8eef1b05fc40cfd`，验证记录收口提交为
  `0623d06e16278911cc4ae0feb2bb1be5f2973706`。记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_class_conditional_fidelity_gate.md`。
- 正式 ImageNet-256 class-conditional 采样现在必须以同一 EMA sample set 运行固定
  torchvision ResNet-50 ImageNet-1K V2 evaluator；默认权重
  `/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth`
  必须精确为 `102,540,417` bytes、SHA256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`。requested class
  由零基 PNG index `mod 1000` 重建；报告 top-1/top-5、target probability/NLL、predicted-class
  coverage/entropy，并绑定 classifier、preprocess、sample、checkpoint、Git/runtime identity。
- paired qualification 同时要求两边绝对 floor 与 CoFiTok 相对 dense 的最大 `0.05` top-1/top-5
  regression。scaling 10K/DDIM-100 floor 为 top-1 `0.01`、top-5 `0.05`、coverage `0.25`、
  normalized entropy `0.50`；full 50K/DDIM-250 floor 为 `0.10 / 0.25 / 0.50 / 0.70`。
  generation gate schema v5、stability posteval runbook、comparison schema v8 和两类 completion audit
  都会重新验证 qualification、两份 raw evaluator report 以及表中匹配行，任何 source/delta/row
  漂移 fail closed。
- 该检查只回答 class condition 是否被遵守，不能替代 FID/IS/precision/recall、EMA rollout、机制诊断
  或 visual review；固定
  `standalone_generation_quality_claim_allowed=false`、`full_training_launch_allowed=false`、
  `release_authorization_allowed=false`。`ml-training-recipes` 的 EMA-for-inference 与 exact
  checkpoint/version identity 原则促使 classifier、sample set、checkpoint 与 evaluator runtime 全部进入
  trust boundary。
- 本地 exact follow-up 完整 suite 为 `1041 passed, 6 skipped in 262.46s`。pro6000 隔离 clean
  `scale/generation-large-capacity@2448ce5e6a96beca6762eba1c8eef1b05fc40cfd` 在
  `CUDA_VISIBLE_DEVICES=""`、`PYTHONPATH=.:src` 下为 `1045 passed, 2 skipped in 157.77s`；全部
  `104/104` tracked runbook 通过 `bash -n`，固定 classifier CPU load 和一次 `(1,1000)` forward
  全 finite。base bundle 为 `173,013` bytes / SHA256
  `8129f9a5cbfcdb3c24d76a5bacad50746b406cafb7ff9d338c5f28a018d270dc`，follow-up bundle 为
  `2,080` bytes / SHA256
  `7650abc719abbdf5558ad79b748afa4b0fa251550eb53d1736554dfa56fc6579`；均通过 prerequisite/head
  验证，未移动 active checkout。
- 收口只读状态中 dense 已到 step `26,000`、`1,664,000` samples，最近 metrics finite、严格递增且
  `samples_seen=step*64`；pair/watchdog/recovery running、issues 为空，posteval/readiness/supplemental
  waiter 继续 waiting，supplemental 无 child，`full_training_launch_allowed=false`。GPU compute 仍只有
  trainer PID `541878`（`77,970 MiB`，余量 `19,271 MiB`），generation filesystem free bytes
  `182,812,176,384`。没有修改、signal、pause 或 restart 任一 GPU/其他项目进程，没有重跑 CoFiTok，
  没有部署新 revision，也没有授权/启动 full 300K。

## 2026-08-03 full-training launch current-state boundary handoff

- branch `scale/generation-large-capacity` 实现提交
  `b45c490429c4e7f7ebeb1cc7eeca9ac66eaf2402`，验证记录提交
  `b0c1b6f3637cbd247695f5d127785f1e05f24e1f`。该实现修复 full-training launch receipt 两个已暴露但未完整生效的
  current-state policy：正式 launch/resume 现在会直接重放目标 deployment receipt、验证当前 formal
  repository/clean checkout identity，并把 current runtime requirement 传播到 readiness bridge。记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_full_launch_current_state_boundary.md`。
- readiness bridge 明确分开两条边界：launch/resume 必须从当前目标 checkout 重算 runtime SHA；terminal
  completion audit 只关闭当前 host/repository freshness，仍从所有冻结报告、Git blob manifest、checkout
  identity 与内容 SHA 重建 immutable launch decision。这样历史复核不会错误依赖审计机当时的环境，也不会
  弱化 checkpoint/training/sampling/EMA artifact 的 recorded runtime binding。
- 本地 focused suite 为 `36 passed`，完整 suite 为 `1046 passed, 6 skipped`，`compileall` 与
  `git diff --check` 通过。增量 bundle 为 `10,139` bytes、SHA256
  `8336c4214e30e2a4b4eea2a7836e53bf0a8a944a4088fd969e610795b790851b`；pro6000 clean 隔离 checkout
  `/tmp/cofitok-launch-current-state-b45c490/CoFiTok-internal` 在 CPU-only 环境通过
  `1050 passed, 2 skipped in 151.78s`、`104/104` tracked runbook `bash -n` 和 `4/4` 直接 CLI
  `--help` import，未移动 active checkout。
- `ml-training-recipes` 的 exact runtime/Git/checkpoint identity 原则影响了边界设计：当前 launch 校验
  与历史 immutable replay 必须分别表达，EMA 仍是正式推理路径；本次没有改变模型 recipe、训练配置或
  任何 GPU 作业。
- 最终只读复核中 dense 实时 metrics 到 step `27,400`、`1,753,600` samples、`549` 行，全部 finite、
  严格递增且满足 `samples_seen=step*64`，最近 50-step interval 约 `127.02s`；pair monitor 前一轮为
  step `27,350` / progress `0.547`，pair/watchdog/recovery running、issues 为空。25K checkpoint 仍为
  `1,006,120,214` bytes / SHA256
  `e73c935c488e234e80136b9507d3c84fca6015a760516dfec1bdf557871fb8ea`，与 sidecar/latest 精确绑定。
  GPU compute 仅 trainer PID `541878`（`77,970 MiB`），filesystem free bytes `182,812,217,344`；
  posteval/readiness/supplemental 均 waiting 且无 child，`full_training_launch_allowed=false`。没有修改、
  signal、pause 或 restart 任何进程，没有重跑 CoFiTok，也没有授权或启动 full 300K。

## 2026-08-03 frozen scaling class-fidelity prerequisite Linux handoff

- branch `scale/generation-large-capacity` 实现提交
  `d1ce0330c1b0fc9573570103763c9614d130ee2b`（short `d1ce033`），Linux 证据收口提交
  `81432b04e095f25a21b8c0ffc572a8424c2d9740`（short `81432b0`）。实现把 frozen scaling
  class-fidelity qualification 加为 full-launch receipt schema v4 的第 12 个必要来源。raw/qualification
  schema v2 分开绑定 frozen sampler Git 与 clean evaluator Git；独立 verifier 会重算 classifier/runtime、
  10K EMA DDIM-100 balanced-modulo 协议、指标/paired deltas/十项 threshold、promotion-gate checkpoint 与
  sample-set SHA，并在结束前重哈希 source 关闭 TOCTOU。记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_frozen_scaling_class_fidelity_prerequisite.md`。
- dormant runbook
  `generation_stability_frozen_50k_class_fidelity_after_supplemental.sh` 只允许在 frozen posteval 与 source-bound
  supplemental 都通过、GPU idle、clean evaluator checkout identity 精确匹配后评估现有 10K sample trees；
  它不训练、不重跑 CoFiTok、不启动 full 300K。class fidelity 仅是必要的非授权 quality evidence，固定
  `full_training_launch_allowed=false`；旧 schema-v3 readiness-compatible target 已不再足以满足新 launch
  receipt，后续必须另建并 attest schema-v4 compatible control target，不能静默复用旧 target。
- exact Linux checkout
  `/tmp/cofitok-stability-frozen-class-fidelity-d1ce033/CoFiTok-internal` 保持 clean
  `scale/generation-large-capacity@d1ce033...`。从已验证 base `b45c490...` 构建的两提交 bundle 为
  `37,579` bytes、SHA256
  `acff97f4170b9f803863ff2af15a10db2e29b28d8c870f5cd6ae8b20ab1207fe`，只广告 `d1ce033...`；
  local/remote `git bundle verify` 均通过，没有移动 active training/deployment checkout。确认新 checkout 已独立
  持有目标 commit 后，精确 local TEMP 与 remote `/tmp` bundle 已删除；可由 Git 历史重建，项目证据未删。
- 首次 Linux suite 的 4 个失败只因新隔离根缺少 inner repo 外层 `paper/` sibling；没有改实现或测试。
  从 clean attested `b45c490` source 按权威 SHA 补入两份所需 LaTeX 后，CPU-only
  `CUDA_VISIBLE_DEVICES=""` 复演为 `1061 collected / 1059 passed / 2 skipped / 0 failed`，
  `torch.cuda.is_available()==false`；`compileall`、`105/105` tracked runbook 原生 `bash -n` 与 `4/4`
  关键 CLI `--help` 均通过。固定 classifier 只重哈希未反序列化：`102,540,417` bytes / SHA256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`。机器证据目录：
  `CoFiTok-internal/artifacts/reports/generation/stability_frozen_class_fidelity_linux_rehearsal_2026-08-03/`；
  summary SHA256 为 `8a32392b688780d68bef587ad26a55cdc8345794b22c5dec80e46c8cdf0af7a4`。
- 既有 heartbeat `cofitok-dense-recovery-launch-monitor` 已通过 Codex `automation_update` 成功原位更新，
  保持 30 分钟间隔且未创建重复 automation；`updated_at=1785731889403`。新 prompt 每轮同时审计
  supplemental waiter，并仅在 posteval/supplemental exact pass、既有 GPU stages terminal、GPU idle、
  d1ce evaluator checkout/classifier identity clean 且无既有 process/lock/receipt/qualification 时，才允许
  nohup 启动一次上述 dormant class-fidelity runbook；hold/failure 禁止自动重启。即使 pass 也明确禁止
  heartbeat 创建 schema-v4 bridge/launch receipt 或授权/启动 full 300K。
- 收口只读状态：dense 已到 step `30,050`、`1,923,200` samples、`602` 行；30K checkpoint 为
  `1,006,120,214` bytes / SHA256
  `8d4a4e098febe9c61c687267488f5dcb2339bbad2ac03bc42c3f142ea9d21c45`，sidecar/latest 与 pair
  monitor 的 `metadata_verified` binding 精确一致。EMA-teacher scale 在 exact 30K 为 0，30,050 正常升至
  `0.005`，schedule missing/mismatch 均为空。GPU compute 仍只有 trainer PID `541878`（`77,970 MiB`，
  free `19,271 MiB`），filesystem free bytes `182,812,155,904`；posteval/readiness/supplemental 均
  waiting 且无 child，supplemental non-authorizing、readiness `full_training_launch_allowed=false`。本轮没有
  signal/pause/restart/修改任何进程，没有 GPU evaluation、CoFiTok rerun 或 full 300K authorization/launch。

## 2026-08-03 readiness-compatible schema-v4 control target handoff

- 旧 readiness-compatible target
  `scale/generation-stability-full-control-quality-27ed@9019dd3f0f504e799c03496ec41a653b61deaa02`
  只支持 schema v3，继续保留为历史证据但不可再作为 launch target。新的 local-only 控制目标为
  `scale/generation-stability-full-control-quality-v4-27ed@cac762ee147f645185259f45fdf212e1872838cb`
  （parent `9019dd3...`）。它只移植 `b45c490` 的 launch current-state replay 与 `d1ce033` 的 frozen
  scaling class-fidelity schema-v4 prerequisite；没有移植 class-fidelity sampler/evaluator、新 trainer
  package、训练配置或 300K execution 改动。记录：
  `CoFiTok-internal/docs/records/2026-08-03_generation_readiness_compatible_schema_v4_control_target.md`。
- 对 active readiness source `5dd3488ac9b30274f4960195e252cc9fdb161002` 的 committed Git replay 已通过：
  source 是 target ancestor；`66/66` training-critical blobs byte-identical；
  `monitor_report_passes()` 后 full-training execution suffix byte-identical，SHA256 仍为
  `1a4e559c3c8828ceee9907f45e2de2d4513a77f37e4458eb50755ee53ab6382c`；source preamble 与
  normalized target preamble SHA256 均为
  `3c196801c05153da13501c265da8480714f2ef119b686db8dbc3e153558d2d07`。目标前导只新增
  readiness bridge、frozen supplemental、frozen scaling class fidelity 与 schema-v4 receipt 约束，
  保留 `116,640` sample runway。`generation_gate.py` / `generation_gate_sources.py` 与 frozen evaluator
  `c1efb12...` 的 Git blob 逐字节相同。
- standalone verifier 与 `d1ce033` 审计版本 normalized byte-equivalent，固定 classifier 权重 SHA
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`，重放 raw/qualification、
  runtime、sampling/evaluator Git split、10K EMA DDIM-100 balanced-modulo、metric/check arithmetic、
  promotion-gate checkpoint/sample-set binding 与 TOCTOU rehash；返回值仍固定
  `full_training_launch_allowed=false`。
- 本地 sibling-layout focused suite `50 passed`；完整 suite
  `877 collected / 873 passed / 4 skipped / 0 failed`。pro6000 exact clean isolated checkout 在
  `CUDA_VISIBLE_DEVICES=""` 下为 `877 tests / 875 passed / 2 skipped / 0 failures / 0 errors`
  （JUnit `124.566s`）；`compileall`、`100/100` runbook `bash -n` 和 4 个关键 CLI help 通过。
  prerequisite bundle 为 `77,285` bytes、SHA256
  `1da88d84baeca066daa9a6d9a0886f5613b378145b43cbe9c61505aaba69ebfd`，只广告 `cac762e...`，
  含 `10` commits / `147` objects，local/remote verify 均通过。
- remote bundle 只对 authoritative clean `checkout-5dd3488` 做 prerequisite verify；fetch 只发生在
  `/tmp` isolated clone。source checkout 前后保持 `5dd3488...` / `scale/generation-large-capacity` / tracked
  clean，没有 fetch、merge、checkout mutation。GPU 前后仅 trainer PID `541878`（`77,970 MiB`），
  storage free bytes `182,812,114,944`。remote clone/JUnit/bundle 与 local TEMP bundle 均已删除；机器证据：
  `CoFiTok-internal/artifacts/reports/generation/readiness_compatible_quality_control_target_v4_2026-08-03/`。
- 新 target 仍只在本地 Git 中，未 fetch/merge/deploy 到 authoritative checkout；没有生成 deployment
  receipt、readiness bridge、launch receipt，也没有授权或启动 full 300K。必须继续等 dense exact 50K、
  frozen posteval、source-bound supplemental 与 frozen class fidelity 结束；即使都 pass，也仍需新的显式
  deployment/bridge/receipt 流程和独立用户授权。
- 最终清理后只读复核：dense metrics 已到 step `31,600`、`2,022,400` samples，pair monitor 前一轮为
  step `31,550` / progress `0.631`，pair/dense health issues 均为空；EMA-teacher scale 正常升至 `0.16`，
  当前 total/epsilon/teacher loss 都 finite。GPU 仍只有 trainer PID `541878`（`77,970 MiB`），storage free
  bytes `182,812,106,752`；posteval/readiness/supplemental 均 waiting，readiness/supplemental 继续固定
  `full_training_launch_allowed=false`。authoritative `checkout-5dd3488` 仍是原 HEAD/branch/tracked clean，
  remote rehearsal clone/bundle 已确认不存在。本轮没有 signal、pause、restart 或修改任何训练/其他项目进程。

## 2026-08-03 matched 35K EMA-teacher trajectory and live 40K+ handoff

- 新的 source-bound 35K 训练轨迹证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_35k_schedule_trajectory_2026-08-03/`，
  记录为
  `CoFiTok-internal/docs/records/2026-08-03_generation_stability_matched_35k_schedule_trajectory.md`。
  它冻结每种方法 701 行、2,240,000 images 和 35 个严格配对 fixed-validation events；Git、dataset、
  runtime、pair contract、finite metrics、严格 step 与 `samples_seen=step*64` 均通过。四份 builder source
  已永久纳入 evidence pack，不再依赖本机 TEMP。
- 1K--35K validation epsilon MSE 均值为 CoFiTok `0.0303214049`、dense `0.0303540541`，ratio-of-means
  相对差 `-0.107561%`；35K endpoint 为 `+0.107953%`。首次覆盖的 EMA-teacher 30K--35K warmup 六个
  events 中相对差 `+0.172577%`，lower-event counts `2 / 4`，最大单 event 相对差 `0.550214%`。这是
  closely matched、finite 的训练空间诊断，不是 sample-quality 或 teacher causal claim；report 内 quality、
  promotion、formal-50K substitute 与 full-training authorization 仍全部为 false。
- `trajectory_report.json` SHA256 为
  `e02d63c966dd6450aa40df4a8a94364ed02c4db4da106c3fbbb277bbad4c7329`；CoFiTok/dense 35K metrics
  prefix SHA256 分别为
  `364f08b9646913394c936c0308dcbc5b4c35331bcac3c55c0833c4d68c1cba12` 与
  `fbcc22912b3d3378bbf26535be63bdd99a9a805de0a8ce6aad12a4a8f1a7ded0`。35K dense checkpoint 为
  `1,006,120,214` bytes / SHA256
  `470740ea8b52172070ac7198c480dade4414421369a6e33cb408ba8532cf371d`。focused builder tests
  `13 passed`，由四份永久 source 重建的 report byte-identical。
- `ml-training-recipes` 的 exact runtime/checkpoint/EMA 边界用于本轮证据收口：训练轨迹只证明连续性与
  matched contract；正式推理质量仍必须由 exact 50K 后的 frozen EMA post-eval、supplemental 和
  class-fidelity 链证明，不能从中间 validation epsilon 推断。
- 最终远端只读复核时 dense 已到 step `42,800`、`2,739,200` samples、857 行；全部数值 finite、step
  严格递增且 sample accounting 精确。pair monitor 前一轮为 step `42,750` / progress `0.855`、issues
  为空；pair monitor、watchdog、recovery 均 running。40K checkpoint 实际重哈希为
  `1,006,120,214` bytes / SHA256
  `cb040fb971550bcab42bda32e44d9d962b8044af33e60caac414d56d6062a68f`，与 sidecar/latest 的 step、
  bytes、SHA、Git、dataset、runtime 精确一致。
- controller PID `511801` 的 fd 6 仍对唯一 authoritative `dense_recovery.lock` 持有 advisory write flock。
  argv/cwd/process tree 证明仍为一个 root controller、一个 pair monitor、一个 watchdog、一个 GPU trainer
  leader、一个 posteval waiter、一个 readiness waiter和一个 supplemental waiter；trainer 的 DataLoader
  children 共享 leader argv/PPID，不是额外 trainer leader。五个权威 checkout 的 revision/branch 均精确且
  tracked clean。
- GPU compute 只有 trainer PID `541878`（`77,970 MiB`），GPU free `19,271 MiB`、utilization `99%`；
  filesystem free bytes `182,811,840,512`。posteval/readiness/supplemental 均 waiting 且 child 为空，
  supplemental 继续 `supplemental_non_authorizing=true`，readiness/supplemental 继续
  `full_training_launch_allowed=false`。本轮没有修改、signal、pause、kill 或 restart 任何进程，没有重跑
  CoFiTok、没有启动 class fidelity，也没有创建 bridge/receipt、授权或启动 full 300K。

## 2026-08-03 matched 40K full-scale transition and resume-aware trajectory handoff

- exact 40K EMA-teacher full-scale transition 证据位于
  `CoFiTok-internal/artifacts/reports/generation/stability_scaling_50k_ema_teacher/matched_40k_schedule_trajectory_2026-08-03/`，
  记录为
  `CoFiTok-internal/docs/records/2026-08-03_generation_stability_matched_40k_schedule_trajectory.md`。
  每种方法都绑定 2,560,000 images 和 40 个 matched fixed-validation events；CoFiTok 802 行、dense 801 行，
  差出的唯一一行是 exact resume checkpoint 36,545 后 trainer 预期记录的 first-post-resume step 36,546。
- trajectory builder 升级到 schema 3：只有 manifest 中 resume checkpoint 文件名、schema-1 metrics reconciliation、
  resume step、retained/orphan counts 与 content-addressed orphan evidence 全部自洽时，才允许唯一
  `resume_step+1` off-grid row；任意未绑定 irregular row 仍 fail closed。当前 CoFiTok reconciliation 为
  `unchanged`，`resume_step=36545`、`retained_rows=731`、`orphaned_rows=0`，且 frozen prefix 对 retained
  row count 独立重算一致；dense 没有 resume/off-grid row。
- 1K--40K validation epsilon MSE mean 为 CoFiTok `0.0299178669`、dense `0.0299400262`，ratio-of-means
  delta `-0.074012%`，lower-event counts `15 / 25`；exact 40K endpoint `-0.005674%`。EMA teacher 30K--39K
  warmup delta `+0.202082%`；40K 第一个 full-scale event `-0.005674%`。这仍只是 finite matched trajectory，
  一个 full-scale event 不支持 teacher benefit 或 sample-quality claim。
- physical dense 40K checkpoint 已实际重哈希：`1,006,120,214` bytes / SHA256
  `cb040fb971550bcab42bda32e44d9d962b8044af33e60caac414d56d6062a68f`，与 sidecar/latest 的 step、
  bytes、SHA、Git、dataset、runtime 精确一致。trajectory report SHA256 为
  `3339269636570be75bf53ba811eda018cad8ab01405aaf196838d8c691ee217d`，builder SHA256 为
  `8e2bcfb8b82b6c551472ed826df43c9eb1b4ab61f71873a289d00c2851df7131`；由四份永久 builder source
  重建 byte-identical。
- 35K pack 中三份名为 boundary-time、但实际在 dense 已过 40K 后采集的 live latest/watchdog/pair 附件已删除；
  35K 现在只用 frozen metrics/manifests、35K sidecar 与实际 checkpoint SHA 作证，记录已明确更正。
- builder、metrics-resume 与 pair-contract focused suites 为 `37 passed`，compileall 与 `git diff --check`
  通过。尝试的全量 local suite 在 outer command timeout 前没有失败输出，但未产生 pytest terminal summary，
  因而没有被误报为 pass；终止后无残留 Python/pytest child。
- 最终远端只读复核时 dense 到 step `44,150`、`2,825,600` samples、884 行；pair monitor 前一轮 step
  `44,150` / progress `0.883`，issues/health issues 均为空。metrics 文件 mtime 与 pair/watchdog 更新时间
  均在约两分钟内，trainer leader CPU 约 99.8%，因此没有 stall 证据。watchdog/recovery/pair monitor running，controller
  PID `511801` fd 6 仍持有唯一 authoritative `dense_recovery.lock` advisory write flock。GPU compute 只有
  trainer PID `541878`（`77,970 MiB`），free `19,271 MiB`，utilization `100%`；filesystem free bytes
  `317,870,297,088`。posteval/readiness/supplemental 继续 waiting、child 为空，所有 non-authorizing/full-launch
  false 边界不变。本轮没有修改或 signal/pause/kill/restart 任何远端进程，没有重跑 CoFiTok、没有启动
  class fidelity，也没有授权或启动 full 300K。
