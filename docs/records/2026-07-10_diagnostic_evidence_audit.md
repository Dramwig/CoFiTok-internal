# Diagnostic Evidence Audit - 2026-07-10

## Decision

Zero-token, random-token, and shuffled-token diagnostics are present in the
current artifact set. The paper-facing visual panel currently emphasizes
prefix behavior, zero-token leakage, shuffle mismatch, and deep-`S_k` failure.
An appendix random-token visual panel has also been generated from the
ImageNet-64 K=8 denoise-path checkpoint.

## Coverage

File-level search over `CoFiTok-internal/artifacts/reports/**/*.json` found:

| diagnostic field | files with evidence |
|---|---:|
| `zero_token_ratio` or `zero_token_component_energy_ratio` | 191 |
| `random_token_component_energy_ratio` | 176 |
| `random_token_ratio` | 9 |
| `shuffled_final_ratio` | 12 |

Representative source reports include:

```text
CoFiTok-internal/artifacts/reports/eval_cifar10_k8_denoisepath_p150_3k_ordered_2026-07-08/report.json
CoFiTok-internal/artifacts/reports/eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_ordered_2026-07-08/report.json
CoFiTok-internal/artifacts/reports/summary_2026-07-09_p0_formal64_ablation_refresh/experiment_summary.json
CoFiTok-internal/artifacts/reports/summary_2026-07-10_imagenet_hf_5k_internal_gaps/experiment_summary.json
```

Paper evidence summary:

```text
CoFiTok-internal/artifacts/reports/paper_evidence_report_2026-07-10/paper_evidence_report.md
```

Final visual panel:

```text
CoFiTok-internal/artifacts/figures/final_visual_claim_2026-07-10/cofitok_final_visual_claim_panel.png
```

Appendix random-token panel:

```text
CoFiTok-internal/artifacts/figures/random_token_diagnostic_2026-07-10/random_token_prefix_panel.png
CoFiTok-internal/artifacts/figures/random_token_diagnostic_2026-07-10/random_token_diagnostic_report.json
```

Representative metrics from that appendix report:

```text
checkpoint: /root/autodl-tmp/CoFiTok/checkpoints/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_2026-07-08/checkpoint_final.pt
ordered final MSE: 0.0991
random-token final MSE: 27.3458
zero-token component energy ratio: 0.0
shuffled component relative MSE: 1.9931
```

## Boundary

Do not state that random-token visual examples are in the main paper panel
unless the appendix panel is explicitly included or referenced. The defensible
statement is:

```text
Zero-token, random-token, and shuffled-token degeneracy diagnostics are logged
in the experiment artifacts; the main visual panel shows prefix behavior plus
zero/shuffle/deep-S_k examples, and a separate appendix random-token panel is
available.
```
