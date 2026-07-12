# 2026-07-07 Code Scaffold v0

Purpose: create the first modular CoFiTok code scaffold before dataset download
and formal experiments.

Local archive:

```text
C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
```

Remote execution target:

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Design:

- `T_k`: currently `TinyTokenPredictor`, an expressive convolutional predictor.
- `S_k`: `RestrictedSynthesis`, condition-free and bias-free.
- Prefix outputs: cumulative epsilon predictions after each token.
- Losses: epsilon MSE, prefix denoising target, monotonic penalty, zero-token regularizer.
- Diagnostics: zero-token energy, random-token energy, shuffled-token delta.

Next verification:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/smoke_forward.py \
  --config configs/smoke_random_cpu.json \
  --output /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_report.json
```

Verification completed on `pro6000` with `pf-vlm`:

```text
pytest: 5 passed
CPU smoke report: /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_report.json
CUDA smoke report: /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_cuda_report.json
```

Both smoke reports were copied back to:

```text
artifacts/reports/smoke/
```

Key diagnostic:

```text
zero-token component energy = 0.0
```
