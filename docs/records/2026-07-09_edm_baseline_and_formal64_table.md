# EDM Baseline and Formal64 Table

Date: 2026-07-09

Scope: complete the second runnable external P0 pixel-diffusion baseline and
build a compact paper-facing table for the current formal 64x64 evidence.

## EDM Adapter Status

External repo:

```text
/root/autodl-tmp/CoFiTok/baselines/repos/edm
commit: 008a4e5316c8e3bfe61a62f874bddba254295afb
```

CoFiTok-side glue:

```text
baselines/adapters/edm/
scripts/baselines/probe_edm_adapter.py
configs/baselines/edm/
artifacts/runbooks/edm_probe_2026-07-09.sh
artifacts/runbooks/edm_formal64_queue_2026-07-09.sh
```

Validation:

```text
pytest tests/test_build_formal64_paper_table.py tests/test_summarize_baseline_reports.py tests/test_build_paper_comparison_matrix.py
```

Result: `5 passed`.

## Formal64 EDM Results

Baseline summary:

```text
artifacts/reports/baselines/summary_2026-07-09_plus_edm/baseline_summary.md
```

| Dataset | EDM lowres Frechet | EDM Inception Frechet | NFE |
|---|---:|---:|---:|
| `downsampled_imagenet_64` | 2.7540 | 248.3076 | 79 |
| `ffhq_64` | 2.5077 | 325.4898 | 79 |
| `afhqv2_64` | 1.9621 | 230.3979 | 79 |

EDM is stronger than CoFiTok, dense epsilon, channel-mask, and
improved-diffusion on current formal64 generated-sample Frechet-style metrics.

## Paper-Facing Table

Generated table:

```text
artifacts/reports/formal64_paper_table_2026-07-09/formal64_paper_table.md
artifacts/reports/formal64_paper_table_2026-07-09/formal64_paper_table.csv
artifacts/reports/formal64_paper_table_2026-07-09/formal64_paper_table.json
```

Current method coverage:

| Method | Status |
|---|---|
| CoFiTok light/full denoise path | completed on all three formal64 datasets |
| Same-backbone dense epsilon | completed on all three formal64 datasets |
| Channel mask | train + quality completed; no generated-sample row |
| improved_diffusion | formal64 train + eval completed |
| EDM | formal64 train + eval completed |
| D-AR | cloned and pinned; no fair formal64 adapter yet |

## Interpretation

- Current evidence supports the scoped claim: ordered restricted denoising-token
  factorization gives prefix-controllable denoising diagnostics not provided by
  monolithic pixel diffusion baselines.
- Current evidence does not support a broad unconditional generation-quality
  advantage. EDM dominates generated-sample Frechet-style metrics on all three
  formal64 datasets.
- The next defensible baseline step is D-AR feasibility/adaptation. If D-AR is
  not protocol-compatible for 64x64 exact-source training, record that explicitly
  and keep it as a nearest-method limitation rather than forcing an unfair run.
