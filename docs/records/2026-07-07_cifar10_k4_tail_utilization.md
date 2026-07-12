# 2026-07-07 CIFAR-10 K4 Tail Utilization

Purpose: compare first attempts at reducing early-token energy collapse after
the CIFAR-10 K4 training loop began to optimize correctly.

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Validation:

```text
pytest: 11 passed
```

Code additions:

- `energy_budget_weight` and `energy_target` in `LossConfig`
- `residual_component_weight` in `LossConfig`
- `energy_budget` and `residual_component` entries in `LossBreakdown`
- report fields:
  - `component_energy_ratio`
  - `head_energy_ratio`
  - `tail_energy_ratio`
  - `active_tail_tokens`

## Runs

Baseline:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_1k_2026-07-07
```

Energy-budget:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_energy_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_energy_1k_2026-07-07
```

Residual-component:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_residual_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_residual_1k_2026-07-07
```

Tail-floor:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_tailfloor_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_tailfloor_1k_2026-07-07
```

Progressive prefix-sampling:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_prefixsample_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_prefixsample_1k_2026-07-07
```

Multiscale restricted synthesis, hard:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_multiscale_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_multiscale_1k_2026-07-07
```

Multiscale restricted synthesis, soft:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_multiscale_soft_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_multiscale_soft_1k_2026-07-07
```

Channel-mask restricted synthesis:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_channelmask_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_channelmask_1k_2026-07-07
```

Soft channel-mask restricted synthesis:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_channelmask_soft_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_channelmask_soft_1k_2026-07-07
```

Soft channel-mask plus light energy budget:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_channelmask_soft_energy_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_channelmask_soft_energy_1k_2026-07-07
```

Channel-mask 3k:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_channelmask_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_channelmask_3k_2026-07-07
```

Local report copies:

```text
artifacts/reports/train_cifar10_k4_energy_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_energy_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_residual_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_residual_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_tailfloor_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_tailfloor_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_prefixsample_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_prefixsample_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_multiscale_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_multiscale_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_multiscale_soft_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_multiscale_soft_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_channelmask_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_channelmask_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_channelmask_soft_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_channelmask_soft_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_channelmask_soft_energy_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_channelmask_soft_energy_1k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k4_channelmask_3k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_channelmask_3k_2026-07-07/prefix_final.png
```

## Comparison

| run | total | epsilon | prefix | prefix MSE | energy ratio | tail ratio | active tail | zero energy |
|---|---:|---:|---:|---|---|---:|---:|---:|
| baseline 1k | 6.5872 | 0.1595 | 25.7108 | [0.7430, 0.3842, 0.3683, 0.3539] | [0.9600, 0.0373, 0.0020, 0.0007] | 0.0400 | 1 | 0.0 |
| energy budget 1k | 6.7119 | 0.1584 | 25.7524 | [0.7926, 0.5662, 0.5479, 0.4587] | [0.9202, 0.0612, 0.0131, 0.0055] | 0.0798 | 2 | 0.0 |
| residual component 1k | 6.5931 | 0.1592 | 25.7072 | [0.7496, 0.3831, 0.3680, 0.3572] | [0.9586, 0.0386, 0.0021, 0.0007] | 0.0414 | 1 | 0.0 |
| tail floor 1k | 7.2623 | 0.1599 | 27.1020 | [0.7511, 0.4123, 0.4024, 0.3707] | [0.9548, 0.0416, 0.0023, 0.0013] | 0.0452 | 1 | 0.0 |
| prefix sampling 1k | 17.8804 | 0.1347 | 46.6671 | [0.7893, 0.3917, 0.3832, 0.3728] | [0.9573, 0.0398, 0.0021, 0.0008] | 0.0427 | 1 | 0.0 |
| multiscale hard 1k | 143.8207 | 0.1781 | 574.5706 | [11.3334, 9.4233, 0.7480, 0.6903] | [0.0342, 0.1282, 0.8306, 0.0071] | 0.9658 | 2 | 0.0 |
| multiscale soft 1k | 71.0740 | 0.1678 | 283.6251 | [9.5100, 0.8347, 0.5764, 0.5663] | [0.1604, 0.8115, 0.0265, 0.0016] | 0.8396 | 2 | 0.0 |
| channel mask 1k | 10.2906 | 0.1684 | 40.4888 | [1.0771, 0.4351, 0.3929, 0.3870] | [0.9273, 0.0666, 0.0050, 0.0012] | 0.0727 | 1 | 0.0 |
| soft channel mask 1k | 8.6166 | 0.1596 | 33.8278 | [0.9260, 0.3944, 0.3762, 0.3674] | [0.9445, 0.0523, 0.0025, 0.0007] | 0.0555 | 1 | 0.0 |
| soft channel mask + energy 1k | 9.1283 | 0.1621 | 35.2683 | [0.9449, 0.4122, 0.3918, 0.3815] | [0.9421, 0.0546, 0.0026, 0.0008] | 0.0579 | 1 | 0.0 |
| channel mask 3k | 1.9978 | 0.1401 | 7.4306 | [0.5352, 0.2384, 0.2250, 0.2212] | [0.9722, 0.0269, 0.0008, 0.0002] | 0.0278 | 1 | 0.0 |

## Interpretation

- The baseline optimizes reliably but collapses almost all energy into token 1.
- Energy-budget loss doubles tail energy ratio from about 4% to about 8%, and
  activates token 3 above the 1% threshold, but worsens prefix MSE.
- A very light residual-component loss preserves baseline quality but does not
  materially improve tail utilization.
- A weak energy-budget plus tail-floor loss barely improves tail ratio and is
  worse than the baseline on prefix loss; this exact combination is not worth
  scaling.
- Progressive prefix-sampling improves epsilon loss but does not solve tail
  collapse. It likely still fails because token 1 has the same spatial
  resolution and synthesis capacity as later tokens.
- Multiscale restricted synthesis proves capacity differences can break token-1
  collapse, but fixed low-pass strides are too blunt on CIFAR-10. Hard
  `[4,2,1,1]` shifts most energy to token 3 and destroys early prefixes. Soft
  `[2,1,1,1]` shifts most energy to token 2 but still damages prefix quality.
- Channel masks are more controlled than spatial low-pass. `[4,8,16,16]`
  raises tail ratio to 7.3% with moderate quality loss, while `[8,12,16,16]`
  raises tail ratio to 5.6% with smaller quality loss. This is the best
  capacity-schedule direction so far.
- Adding a light energy budget to the soft channel mask barely helps tail ratio
  and slightly worsens prefix quality; this specific combination is not worth
  scaling.
- Extending `[4,8,16,16]` from 1k to 3k greatly improves prefix quality, but
  tail ratio falls from 7.3% to 2.8%. The model eventually re-concentrates in
  early tokens even under channel masks.
- `S_k(0)=0` remains exact in all runs.

## Decision

Do not move to Tiny ImageNet as the next main run yet. The next CIFAR experiment
should introduce an actual coarse-to-fine capacity difference in the restricted
synthesis path while preserving the rule that `S_k` sees only the current token.

The next candidate should increase token count before moving to Tiny ImageNet.
K=4 appears too small: after enough optimization, useful information compresses
back into early tokens. Try CIFAR-10 `K=8` with a channel schedule such as
`[4,4,8,8,12,12,16,16]`, and report whether tail groups remain active after
3k steps.
