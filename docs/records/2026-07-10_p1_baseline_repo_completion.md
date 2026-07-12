# P1 Baseline Repo Completion - 2026-07-10

## Decision

The P1 baseline repo pass is complete as a code-asset/protocol pass, not as a full paper result pass.

Status by method:

| method | repo status | adapter status | paper-table status |
|---|---|---|---|
| FlexTok (`ml_flextok`) | cloned and pinned | eval-only reconstruction 256-image all-dataset pass | `completed_eval_only` only |
| TiTok (`titok_1d_tokenizer`) | cloned and pinned | eval-only reconstruction 256-image all-dataset pass | `completed_eval_only` only |
| D-AR (`d_ar`) | cloned and pinned | official ImageNet-256 50K eval-only completed | secondary row only |
| MAR (`mar`) | cloned and pinned | official HF safetensors one-image smoke passed | smoke-only secondary row |
| ReTok (`retok`) | cloned and pinned | official VQ reconstruction smoke, GPT-XL generation smoke, 128-sample ADM-evaluator pilot, and 50K eval-only passed | protocol-progress secondary row |

## Implemented Assets

- Adapter: `scripts/baselines/evaluate_tokenizer_reconstruction.py`
- FlexTok config: `configs/baselines/ml_flextok/reconstruction_eval_256.json`
- TiTok config: `configs/baselines/titok_1d_tokenizer/reconstruction_eval_256.json`
- MAR feasibility config: `configs/baselines/mar/feasibility_2026-07-10.json`
- ReTok feasibility config: `configs/baselines/retok/feasibility_2026-07-10.json`
- Optional all-dataset eval runbook: `artifacts/runbooks/p1_tokenizer_reconstruction_eval_2026-07-10.sh`
- Environment record: `docs/experiment_conditions/p1_tokenizer_adapter_env_2026-07-10.md`

## Smoke Evidence

GPU smoke reports were produced on `pro6000`:

```text
artifacts/reports/baselines/titok_1d_tokenizer/smoke_cifar10_recon256_4img_gpu_2026-07-10/baseline_eval_report.json
artifacts/reports/baselines/ml_flextok/smoke_cifar10_recon256_2img_gpu_2026-07-10/baseline_eval_report.json
```

Both reports record `actual_device = cuda`.

## Eval256 Evidence

The optional all-dataset eval-only pass was run with 256 images per dataset:

```bash
MAX_IMAGES=256 TAG=eval256_2026-07-10 bash artifacts/runbooks/p1_tokenizer_reconstruction_eval_2026-07-10.sh
```

Generated reports:

```text
artifacts/reports/baselines/p1_tokenizer_reconstruction_eval256_2026-07-10/p1_tokenizer_reconstruction_table.md
artifacts/reports/baselines/summary_2026-07-10_p1_eval256/baseline_summary.md
artifacts/reports/paper_comparison_matrix_2026-07-10_p1_eval256_guard_latest/paper_comparison_matrix_audit.md
```

Final guard counts:

```text
completed=72
completed_eval_only=16
protocol_blocked=24
```

## Matrix Handling

`scripts/build_paper_comparison_matrix.py` now distinguishes eval-only tokenizer reconstruction from full completed baselines:

- Full train/eval evidence remains `completed`.
- Eval-only tokenizer reconstruction evidence is `completed_eval_only`.
- D-AR official ImageNet-256 50K is complete only as a secondary pretrained/eval-only related-method row.
- MAR smoke rows stay secondary-only until official 50K metrics or a separate retraining protocol are completed; ReTok official 50K is complete but remains secondary eval-only evidence.

This prevents P1 tokenizer smoke or reconstruction rows from inflating the P0 all-dataset completion count.

## Next Reasonable Step

If stronger P1 related-method evidence is needed in the paper, complete:

- Rebuild reports from the completed ReTok official 50K metrics when refreshing evidence tables.
- MAR HF-safetensors official 50K sampling/eval, if the runtime budget is acceptable.

Report all such rows as secondary related-method evidence unless they are retrained under the same P0 all-dataset budget.
