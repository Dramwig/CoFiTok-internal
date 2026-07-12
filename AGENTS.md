# CoFiTok Internal 必读说明

本目录是 CoFiTok 的内部代码、实验脚本和记录工作区。上层项目根目录为：

```text
C:/Users/zixi-/Desktop/PaperField/CoFiTok
```

进入本目录后仍需遵守上层 `AGENTS.md`。研究背景先读：

```text
C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-Denoising Token Diffusion.md
```

## 远端实验位置

正式实验以 `pro6000` 为准：

```bash
ssh pro6000
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
```

远端数据集和权重分别在：

```text
/root/autodl-tmp/CoFiTok/datasets
/root/autodl-tmp/CoFiTok/checkpoints
```

不要把大数据、完整输出、checkpoint、generated samples、feature cache、LoRA / adapter 或完整外部 repo 提交进本仓库。本仓库保存代码、配置、脚本、测试、环境记录、实验记录和小型报告。

跑远端任务前先看 GPU：

```bash
nvidia-smi
```

不要杀其他项目的 GPU 进程，除非用户明确要求。

## 代码与记录布局

建议优先建立这些目录：

```text
src/
configs/
scripts/
tests/
docs/records/
docs/experiment_conditions/
artifacts/reports/
```

`docs/records/` 记录每次实验目的、命令、commit 或文件状态、数据/权重路径、关键结果和失败原因。`docs/experiment_conditions/` 记录环境、数据 manifest、checkpoint 来源和复现条件。

## 默认数据集

第一轮只准备这些数据集，顺序不要反过来：

1. `cifar10`：最小 smoke，目标目录 `/root/autodl-tmp/CoFiTok/datasets/cifar10`。
2. `tiny_imagenet_200`：第一版 MVP 主实验，目标目录 `/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200`。
3. `downsampled_imagenet_64`：P0/P1 跑通后的升级验证，目标目录 `/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64`。

不要在第一轮下载 full ImageNet、ImageNet-256/512、COCO、LAION、FFHQ 或 CelebA-HQ。下载后必须在 `docs/experiment_conditions/` 写数据来源、日期、split、文件数量、大小、sha256/manifest 和预处理规则。

## 研究定位约束

CoFiTok 的核心不是“visual token coarse-to-fine”本身，而是：

> factorizing dense pixel-space diffusion noise prediction into ordered compressed denoising components with restricted synthesis operators.

实现时特别注意：

- `S_k` 只接收当前 token `z^(k)`。
- `S_k` 默认 linear、bias-free、shallow、local。
- 不要给 `S_k` 输入 prompt、time embedding、previous tokens、image features 或 skip connection。
- 必须保留 zero-token、random-token、shuffled-token 和 deep-`S_k` ablation 的测试入口。
- Prefix denoising visualization 和 ordered / reverse / random / simultaneous 消融是 MVP 的核心。

## 本地验证

本地只做轻量验证：

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
uv venv .venv --python 3.10
uv sync
```

如果项目还没有 `pyproject.toml`，先按实际代码骨架补齐，再创建环境。正式 diffusion 训练、批量采样、FID / LPIPS 评估和大规模数据处理都放到 `pro6000`。
