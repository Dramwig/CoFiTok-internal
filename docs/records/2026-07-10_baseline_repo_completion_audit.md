# Baseline Repo Completion Audit - 2026-07-10

Last updated: 2026-07-11.

## Decision

The baseline repo pass is complete as a code-asset and protocol-audit pass.
Do not expand it by cloning unrelated P2/SOTA repositories unless the paper
scope changes.

Current statuses:

| method | repo | adapter/protocol | paper status |
|---|---|---|---|
| Same-backbone dense | internal | CoFiTok runner/config | P0 completed |
| EDM | external, cloned/pinned | train/eval adapter | P0 completed |
| Improved DDPM | external, cloned/pinned | train/eval adapter | P0 completed |
| D-AR | external, cloned/pinned | official ImageNet-256 50K eval-only completed | P0 protocol-blocked for same-dataset retraining |
| FlexTok | external, cloned/pinned | eval-only 256-image reconstruction | P1 completed-eval-only |
| TiTok / 1D Tokenizer | external, cloned/pinned | eval-only 256-image reconstruction | P1 completed-eval-only |
| MAR | external, cloned/pinned | non-EMA conversion 50K audited; official PTH EMA 50K running | P1 secondary eval-only |
| ReTok | external, cloned/pinned | official GPT-XL + VQ 50K completed | P1 secondary eval-only |

## Repo Pins

Primary manifest:

```text
CoFiTok-internal/docs/experiment_conditions/baselines_repo_manifest_2026-07-09.json
```

Cloned external repos are isolated on `pro6000`:

```text
/root/autodl-tmp/CoFiTok/baselines/repos
```

Generated `__pycache__/` directories may appear after probes. No external repo
source patch is required for the current tables. Future patches belong under:

```text
CoFiTok-internal/baselines/patches/<alias>/
```

The ADM evaluator also created identical untracked
`classify_image_graph_def.pb` caches in D-AR/ReTok. A hash-verified canonical
copy now lives under `checkpoints/baselines/official_refs/`; remove the repo-local
copies after the active MAR evaluator finishes.

## Evidence State

Main matrix:

```text
CoFiTok-internal/artifacts/reports/paper_comparison_matrix_2026-07-10_p1_eval256_guard_latest/paper_comparison_matrix_audit.md
```

Current counts:

```text
completed=72
completed_eval_only=16
protocol_blocked=24
```

P0 paper table:

```text
CoFiTok-internal/artifacts/reports/p0_horizontal_table_2026-07-10_imagenet256_p0/p0_paper_table.md
```

P1 tokenizer reconstruction table:

```text
CoFiTok-internal/artifacts/reports/baselines/p1_tokenizer_reconstruction_eval256_2026-07-10/p1_tokenizer_reconstruction_table.md
```

D-AR official ImageNet-256 50K eval-only:

```text
CoFiTok-internal/artifacts/reports/baselines/d_ar/official_imagenet256_50k_2026-07-10/baseline_eval_report.json
CoFiTok-internal/artifacts/reports/baselines/official_related_methods_2026-07-10/official_related_methods_table.md
```

MAR provenance and official EMA recovery:

```text
CoFiTok-internal/artifacts/reports/baselines/mar/community_conversion_audit_2026-07-11.json
CoFiTok-internal/docs/records/2026-07-11_mar_official_pth_recovery_protocol.md
```

ReTok official 50K:

```text
CoFiTok-internal/artifacts/reports/baselines/retok/official_50k_2026-07-10/baseline_eval_report.json
```

## Boundary

The current baseline set is sufficient to support the scoped claim:

```text
ordered restricted dense-noise factorization and prefix-controllable denoising
```

It is not sufficient to claim broad unconditional generation SOTA. EDM and/or
dense epsilon remain stronger on the Frechet-style generation metrics in most
datasets.

## License boundary

- EDM is CC-BY-NC-SA-4.0; FlexTok code and weights have non-commercial/separate terms.
- ReTok declares no license in the pinned repository. Use it only for isolated
  research evaluation and citation; do not redistribute its code or weights.
- Third-party repositories remain outside `CoFiTok-internal/` and are not submodules.

## Next Only If Needed

Finish the active MAR PTH EMA 50K and refresh the secondary table. Do not mix
official/eval-only rows into the P0 train/eval generation table.

Current MAR runbook:

```text
CoFiTok-internal/artifacts/runbooks/mar_official_pth_ema_50k_2026-07-11.sh
```

No additional baseline repository should be cloned unless the paper scope or a
reviewer request changes.
