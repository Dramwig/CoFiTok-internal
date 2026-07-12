# 2026-07-07 CIFAR-10 K4 Short Training

Purpose: move beyond one-batch forward smoke and verify that the first CoFiTok
training loop can optimize on real CIFAR-10 data, save a checkpoint, emit prefix
visualizations, and keep restricted-synthesis diagnostics intact.

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Environment:

```text
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python
```

Code additions:

- `scripts/train_short.py`
- `configs/train_cifar10_k4_smoke_cuda.json`
- `configs/train_cifar10_k4_1k_cuda.json`
- `tests/test_configs.py`
- `OptimizationConfig` in `src/cofitok/configs.py`

Validation:

```text
pytest: 9 passed
```

## Run A: 100-step CIFAR-10 smoke

Command:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_smoke_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_smoke_2026-07-07
```

Artifacts:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_smoke_2026-07-07/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_smoke_2026-07-07/checkpoint_final.pt
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_smoke_2026-07-07/prefix_final.png
```

Local report copies:

```text
artifacts/reports/train_cifar10_k4_smoke_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_smoke_2026-07-07/prefix_final.png
```

Key result:

```text
step 1 total: 311.9259
step 100 total: 76.7500
final epsilon: 0.2827
final prefix: 305.8693
zero-token component energy: 0.0
prefix MSE to clean: [4.9423, 1.8400, 1.3048, 1.2256]
```

## Run B: 1k-step CIFAR-10 smoke

Command:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k4_1k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_1k_2026-07-07
```

Artifacts:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_1k_2026-07-07/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_1k_2026-07-07/checkpoint_final.pt
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k4_1k_2026-07-07/prefix_final.png
```

Local report copies:

```text
artifacts/reports/train_cifar10_k4_1k_2026-07-07/report.json
artifacts/reports/train_cifar10_k4_1k_2026-07-07/prefix_final.png
```

Key result:

```text
step 1 total: 311.9259
step 1000 total: 6.5872
final epsilon: 0.1595
final prefix: 25.7108
zero-token component energy: 0.0
prefix MSE to clean: [0.7430, 0.3842, 0.3683, 0.3539]
component energy: [0.9774, 0.0380, 0.0020, 0.0007]
shuffle component MSE: 0.5123
```

## Interpretation

The short training loop is functional:

- real CIFAR-10 batches load correctly;
- optimization reduces epsilon and prefix losses;
- checkpoint, JSON report, and prefix grid are written;
- `S_k(0)=0` remains exact after training;
- prefix MSE improves monotonically across `m=1..4`.

The main issue is component collapse:

- token 1 carries almost all component energy after 1k steps;
- tail tokens are currently underused;
- visual prefix outputs are still noisy, so this is a plumbing/result-shape
  milestone, not a paper-quality result.

## Decision

Before moving to Tiny ImageNet, run one more CIFAR-10 K4 experiment that directly
targets tail-token utilization. The smallest next change is to add reporting for
component energy ratios and then try a light energy-budget or residual-prefix
variant. Keep `S_k` restricted and condition-free.
