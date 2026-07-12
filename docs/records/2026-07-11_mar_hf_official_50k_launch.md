# MAR HF conversion 50K audit - 2026-07-11

## Status

`completed_audit_only`

The 50K Hugging Face safetensors run completed mechanically, but subsequent
tensor-level provenance auditing showed that `mar-base.safetensors` exactly
matches `checkpoint["model"]`, not the official evaluation state
`checkpoint["model_ema"]`. This run is retained only as a conversion audit.

## Adapter Validation

The project-side adapter is:

```text
scripts/baselines/sample_mar_hf_official.py
```

Checks completed before launch:

- strict MAR-B and KL-VAE safetensors loading;
- official `num_iter=256`, DiffLoss steps 100, CFG 2.9 linear, temperature 1.0;
- batch-8 and batch-128 pilots;
- expected batch-1024 VAE decode OOM, followed by chunked decode correction;
- final batch-768/chunk-16 pilot: 768 valid RGB 256x256 PNGs, 2.9412 image/s,
  13.73 GiB PyTorch peak memory, no NaN/OOM.

Local adapter tests:

```text
7 passed
```

## Completed audit run

Runbook:

```text
artifacts/runbooks/mar_hf_official_50k_2026-07-11.sh
```

Outputs:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_hf
```

The chain is resumable and performs:

```text
50K balanced ImageNet sampling -> PNG validation -> arr_0 NPZ packing -> ADM evaluator
```

Final metrics:

```text
PNG/NPZ: 50000 images, uint8 [50000, 256, 256, 3]
FID: 5.5609856852
sFID: 6.2549410871
IS: 52.001953125
Precision: 0.71664
Recall: 0.6248
```

Do not cite these metrics as the official MAR-B result. The corrective LTH14
PTH `model_ema` run is tracked by
`artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.sh`.
