# Formal64 P0 Ablation Refresh

Date: 2026-07-09

Scope: finish the formal64 internal ablation refresh and regenerate the
paper-facing P0 comparison table.

Runbook:

```text
artifacts/runbooks/p0_formal64_ablation_queue_2026-07-09.sh
```

Primary outputs:

```text
artifacts/reports/summary_2026-07-09_p0_formal64_ablation_refresh/
artifacts/reports/formal64_p0_horizontal_table_2026-07-09_ablation_refresh/
artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64_ablation_refresh/
```

Coverage after refresh:

- `downsampled_imagenet_64`, `ffhq_64`, and `afhqv2_64` now have completed
  P0 rows for CoFiTok, dense epsilon, channel mask, no-prefix-loss,
  clean-monotonic, simultaneous-components, deep-`S_k`, improved DDPM, and EDM.
- `tiny_imagenet_200` now has completed internal ablations; external
  improved-DDPM and EDM rows are still missing.
- D-AR remains `protocol_blocked` for formal64 and should not be counted as a
  completed baseline.
- P1 tokenizer/AR baselines remain `needs_adapter`; keep them outside the MVP
  main table unless the paper needs a separate nearest-method section.

Matrix status counts:

```text
completed=43
missing=28
needs_adapter=32
partial=1
protocol_blocked=8
```

Interpretation:

- EDM is the strongest current method on generated-sample Frechet-style metrics.
- CoFiTok should be argued on ordered restricted denoising-token factorization,
  prefix-controllable denoising, and zero/random/shuffle-style diagnostics, not
  on unconditional generation quality.
- The next defensible experiment decision is either to fill the two missing
  Tiny external-baseline rows or to start a separate ImageNet-256 scaling
  protocol; do not mix those into the completed formal64 MVP table.
