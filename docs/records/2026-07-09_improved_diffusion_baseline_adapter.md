# 2026-07-09 Improved-Diffusion Baseline Adapter

## Scope

Completed the first runnable external P0 baseline path without modifying the external repo.

External repo:

```text
/root/autodl-tmp/CoFiTok/baselines/repos/improved_diffusion
commit: 1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5
```

CoFiTok-side glue:

```text
baselines/adapters/improved_diffusion/
scripts/baselines/probe_improved_diffusion_adapter.py
scripts/baselines/npz_to_png.py
configs/baselines/improved_diffusion/
artifacts/runbooks/improved_diffusion_probe_2026-07-09.sh
artifacts/runbooks/improved_diffusion_formal64_smoke_2026-07-09.sh
artifacts/runbooks/improved_diffusion_formal64_queue_2026-07-09.sh
```

## Validation

Local:

```text
pytest tests/test_npz_to_png.py tests/test_improved_diffusion_adapter.py tests/test_build_paper_comparison_matrix.py tests/test_baseline_registry.py -q
pytest tests/test_configs.py tests/test_data_registry.py -q
python scripts/validate_synthesis_contract.py --config-glob "configs/*.json"
```

Remote:

```text
pytest tests/test_npz_to_png.py tests/test_improved_diffusion_adapter.py tests/test_build_paper_comparison_matrix.py tests/test_baseline_registry.py -q
bash -n artifacts/runbooks/improved_diffusion_formal64_queue_2026-07-09.sh
bash -n artifacts/runbooks/improved_diffusion_formal64_smoke_2026-07-09.sh
```

Smoke reports:

```text
artifacts/reports/baselines/improved_diffusion/adapter_probe_2026-07-09.json
artifacts/reports/baselines/improved_diffusion/smoke_ffhq64_cpu_2026-07-09.json
artifacts/reports/baselines_smoke/improved_diffusion/improved_diffusion_ffhq_64_2steps_ddim10_smoke_gpu_2026-07-09/
```

## Formal Remote Run

The formal 64x64 improved-diffusion queue completed on `pro6000`.

```text
runbook: CoFiTok-internal/artifacts/runbooks/improved_diffusion_formal64_queue_2026-07-09.sh
log: CoFiTok-internal/artifacts/logs/improved_diffusion_formal64_queue_2026-07-09.log
output root: /root/autodl-tmp/CoFiTok/checkpoints/baselines/improved_diffusion
report root: CoFiTok-internal/artifacts/reports/baselines/improved_diffusion
datasets: downsampled_imagenet_64, ffhq_64, afhqv2_64
```

The runbook writes `baseline_train_report.json` and `baseline_eval_report.json`; `build_paper_comparison_matrix.py` now counts those reports for external baseline evidence.

Baseline summary:

```text
artifacts/reports/baselines/summary_2026-07-09/baseline_summary.md

afhqv2_64: lowres Frechet 8.2918, Inception Frechet 335.1814
downsampled_imagenet_64: lowres Frechet 8.8044, Inception Frechet 355.1061
ffhq_64: lowres Frechet 9.4360, Inception Frechet 369.0054
```

Current matrix after internal P0 formal64 completion plus improved-diffusion:

```text
artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64_plus_improved_diffusion/paper_comparison_matrix_audit.md
status_counts: completed=26, missing=37, needs_adapter=48, partial=1
```

## Notes

- `improved_diffusion` is now adapter-ready and has completed formal64 evidence for `downsampled_imagenet_64`, `ffhq_64`, and `afhqv2_64`.
- EDM and D-AR remain cloned but need adapters.
- The short GPU smoke used `BASELINE_REPORT_ROOT=artifacts/reports/baselines_smoke`, so it does not pollute formal matrix evidence.
- The first external baseline is mixed: improved-diffusion is stronger on Inception Frechet for `downsampled_imagenet_64` and `ffhq_64`, while CoFiTok/dense are stronger on the lowres proxy. This supports a scoped mechanism/prefix-control claim, not a broad unconditional-generation-quality claim.
