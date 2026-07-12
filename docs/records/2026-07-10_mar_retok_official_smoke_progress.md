# MAR/ReTok Official Smoke Progress - 2026-07-10

## Decision

MAR and ReTok are no longer pure asset-blocked entries, but only ReTok has a completed official 50K metric row. Treat MAR as secondary official-checkpoint smoke evidence. Treat ReTok as completed secondary official-checkpoint eval-only evidence; it still must not be counted as same-budget all-dataset training evidence.

## MAR

Status: `hf_safetensors_smoke_completed_50k_not_run`

- Repo: `/root/autodl-tmp/CoFiTok/baselines/repos/mar`
- HF assets: `/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/official_imagenet256_eval_only/hf_repo`
- Verified files: `mar-base.safetensors` and `kl16.safetensors`
- Smoke report: `artifacts/reports/baselines/mar/official_hf_imagenet256_smoke_1_2026-07-10/baseline_eval_report.json`

Limitation: one-image generation smoke only. No 50K FID/IS metrics were run.

## ReTok

Status: `completed_eval_only_50k`

- Repo: `/root/autodl-tmp/CoFiTok/baselines/repos/retok`
- VQ checkpoint: `/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/checkpoints/VQ_SB256_e250.pt`
- VQ SHA256: `bde39f28b7dd07aebe1ec09d71d1ced6b65097e93aaf41e0b749ab858414d4b3`
- GPT checkpoint: `/root/autodl-tmp/CoFiTok/checkpoints/baselines/retok/official_imagenet256_eval_only/checkpoints/GPT_XL256_e300_VQ_SB.pt`
- GPT SHA256: `3a8c811772e046bc94ca40c1c3e5db6e01f30bc0f509652144c9f80d40ef279e`
- Smoke report: `artifacts/reports/baselines/retok/official_reconstruction_smoke_4_2026-07-10/baseline_eval_report.json`
- Generation smoke report: `artifacts/reports/baselines/retok/official_generation_smoke_2_2026-07-10/baseline_eval_report.json`
- 128-sample evaluator pilot log: `artifacts/reports/baselines/retok/official_pilot128_eval_2026-07-10/evaluator.log`
- 50K launch log: `artifacts/reports/baselines/retok/official_50k_2026-07-10/retok_50k.log`
- 50K eval report: `artifacts/reports/baselines/retok/official_50k_2026-07-10/baseline_eval_report.json`

Final 50K metrics:

```text
Inception Score: 245.9392
FID: 2.2189
sFID: 5.8956
Precision: 0.8159
Recall: 0.5994
```

## Paper Handling

- D-AR and ReTok official 50K rows are completed secondary related-method metric rows.
- MAR smoke rows can be listed as protocol progress only.
- None of D-AR/MAR/ReTok official eval-only rows belongs in the P0 same-budget all-dataset matrix.