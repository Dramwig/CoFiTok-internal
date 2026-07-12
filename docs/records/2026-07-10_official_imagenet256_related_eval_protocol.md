# Official ImageNet-256 Related Eval Protocol - 2026-07-10

## Decision

D-AR, MAR, and ReTok can only be added under a separate `official_imagenet256_eval_only` protocol. They must remain `protocol_blocked` for the all-dataset same-budget retraining matrix.

This protocol is useful only for a secondary related-method table on ImageNet-256 official checkpoints.

## Added Assets

Configs:

```text
configs/baselines/d_ar/official_imagenet256_eval_only_2026-07-10.json
configs/baselines/mar/official_imagenet256_eval_only_2026-07-10.json
configs/baselines/retok/official_imagenet256_eval_only_2026-07-10.json
```

Runbook:

```text
artifacts/runbooks/official_imagenet256_related_eval_2026-07-10.sh
```

The runbook defaults to `DRY_RUN=1`; it prints commands and does not download weights or generate 50k samples unless explicitly run with `DRY_RUN=0`. `RETOK_SAMPLE_DIR` can override the default ReTok output directory for pilots or official launches.

## Method Notes

| method | official route | current status |
|---|---|---|
| D-AR | Hugging Face tokenizer and D-AR-L checkpoint; ImageNet-256 ADM evaluator | completed 50K eval-only secondary row |
| MAR | `main_mar.py --evaluate` or HF safetensors pipeline on ImageNet-256 | HF safetensors one-image generation smoke passed; 50K metrics not run |
| ReTok | Google Drive checkpoints plus `scripts/sample_c2i_search_cfg.sh` | VQ reconstruction smoke, GPT+VQ two-image generation smoke, 128-sample ADM-evaluator pilot, and 50K eval-only completed |

## Paper Boundary

Do not merge these rows into:

```text
artifacts/reports/p0_horizontal_table_2026-07-10_imagenet256_p0/p0_paper_table.md
```

The current P0 table remains the fair same-budget train/eval comparison. The official ImageNet-256 protocol is a secondary related-method comparison and should be labeled as pretrained/eval-only.

D-AR 50K completion report:

```text
CoFiTok-internal/artifacts/reports/baselines/d_ar/official_imagenet256_50k_2026-07-10/baseline_eval_report.json
CoFiTok-internal/artifacts/reports/baselines/official_related_methods_2026-07-10/official_related_methods_table.md
```

ReTok 50K completion artifacts:

```text
CoFiTok-internal/artifacts/reports/baselines/retok/official_50k_2026-07-10/retok_50k.log
CoFiTok-internal/artifacts/reports/baselines/retok/official_50k_2026-07-10/baseline_eval_report.json
/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/samples_50k/GPT-XL-size-256-size-256-topk-0-topp-1.0-temperature-1.0-cfg-1.5-step-seed-0.txt
```

ReTok 50K metrics: FID `2.2189`, sFID `5.8956`, IS `245.9392`, Precision `0.8159`, Recall `0.5994`.