# MAR Official Eval Probe - 2026-07-10

## Result

Status: `hf_safetensors_smoke_completed_50k_not_run`

MAR remains a secondary related-method baseline. The original official `.pth`
assets were not obtainable on `pro6000`, but the Hugging Face safetensors assets
were later acquired and a one-image official pipeline smoke passed.

## What Passed

- Repo is present at `/root/autodl-tmp/CoFiTok/baselines/repos/mar`.
- Official entrypoint is identified: `main_mar.py --evaluate`.
- Runtime dependencies are available: `torch_fidelity`, `cv2`, `tensorboard`.
- Official ImageNet-256 FID stats exist in the repo: `fid_stats/adm_in256_stats.npz`.
- HF model card file list was reachable and includes `kl16.safetensors`,
  `mar-base.safetensors`, `mar-large.safetensors`, and `mar-huge.safetensors`.

## What Failed

- `util/download.py` wrote 3.7KB Dropbox HTML stubs for both `kl16.ckpt` and
  `checkpoint-last.pth`; `torch.load` rejected them.
- Direct `curl -L --fail` using the same official Dropbox URLs with `dl=1`
  returned HTTP 503 from `pro6000`.
- HF `kl16.safetensors` downloaded successfully to:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/official_imagenet256_eval_only/hf_weights/kl16.safetensors
```

- HF `mar-base.safetensors` download stalled for over 30 minutes. The incomplete
  670MB cache fragment was removed.

## Follow-up: HF Safetensors Smoke

Status: `completed_smoke_only`

Using `HF_HUB_DISABLE_XET=1`, the official Hugging Face model card assets were
downloaded to:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/official_imagenet256_eval_only/hf_repo
```

Verified assets:

```text
mar-base.safetensors 831736456 bytes
kl16.safetensors 265856244 bytes
```

The HF pipeline generated one ImageNet-256 sample on `pro6000`:

```text
CoFiTok-internal/artifacts/reports/baselines/mar/official_hf_imagenet256_smoke_1_2026-07-10/baseline_eval_report.json
```

This validates official HF safetensors loading and generation only. It is not a
50K FID/IS result and must not be merged into the P0 same-budget matrix.

## Protocol Decision

Do not count MAR as completed for P0. Keep it outside the all-dataset same-budget
matrix until one of these is done:

1. Acquire a verified official MAR-B `checkpoint-last.pth` and matching
   `kl16.ckpt`.
2. Run the separate HF-safetensors protocol at a paper-relevant sample count and
   record exact loading, sampling, and metric commands.

Do not mix any future MAR official ImageNet-256 eval-only result into the P0
same-dataset, same-budget generation table.

## Report

```text
CoFiTok-internal/artifacts/reports/baselines/mar/official_imagenet256_probe_2026-07-10/baseline_eval_report.json
```
