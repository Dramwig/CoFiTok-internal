# CoFiTok Internal

Internal code scaffold for CoFiTok experiments.

The local directory is the archive. All code execution, experiments, training,
and report generation should run on `pro6000` under:

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Datasets and checkpoints live outside the code tree:

```text
/root/autodl-tmp/CoFiTok/datasets
/root/autodl-tmp/CoFiTok/checkpoints
```

## Architecture

The scaffold is intentionally modular:

- `cofitok.data`: dataset specs, dataset registry, dataloader construction.
- `cofitok.diffusion`: noise schedules and forward noising helpers.
- `cofitok.models`: token predictors, restricted synthesis operators, model wrapper.
- `cofitok.training`: losses, train-step assembly, metrics.
- `cofitok.diagnostics`: non-degeneration checks for `S_k`.
- `cofitok.reporting`: JSON-safe report helpers.

The first runnable target is a server-side random-batch smoke test:

```bash
PYTHONPATH=src python scripts/smoke_forward.py \
  --config configs/smoke_random_cpu.json \
  --output /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_report.json
```
