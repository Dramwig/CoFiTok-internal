# Matched Min-SNR pilot current state (2026-09-01)

## Scope

This record updates the live status of the source-bound Min-SNR pilot. It does
not create a new authorization or launch a new experiment. The 100K quality
bridge remains the scientific source of truth and retains
`terminal_status=hold`.

## Completed pilot evidence

The exact-gated pilot at
`/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1`
completed both fresh matched training arms at step 50,000 (3,200,000 images
per method) with the sole training change
`loss.min_snr_gamma: 0.0 -> 5.0`. The controller report is bound to revision
`842a34130e82f241330707118a05bf6ed01e263e` and reports
`50K pilot and four-arm 10K evaluation replayed; no continuation authorized`.

All four evaluation arms used EMA, DDIM-100, CFG 1.5, bf16, balanced-modulo
classes, and 10,000 samples per arm. The observed generation metrics were:

| arm | FID | precision | recall |
| --- | ---: | ---: | ---: |
| legacy gamma-0 CoFiTok | 146.3368 | 0.7119 | 0.00830 |
| pilot gamma-5 CoFiTok | 144.3896 | 0.7453 | 0.00458 |
| legacy gamma-0 dense | 189.4888 | 0.8626 | 0.00494 |
| pilot gamma-5 dense | 167.9816 | 0.8337 | 0.00562 |

CoFiTok's relative FID improvement was only 1.33%, below the precommitted 5%
threshold. The dense arm regressed precision by 2.89%, above its 1% bound, and
both arms remained below the absolute class-fidelity thresholds. The shared
candidate result is therefore:

```text
status: completed
scientific_status: screening_only
selection_status: no_shared_min_snr_candidate_at_50k
terminal_status: hold
generation_advantage_proven: false
recommended_next_stage: stop_min_snr_route_and_reassess_training_objective
```

The CoFiTok mechanism replay still passes (ordered rank 1, zero-token maximum
absolute value 0, shuffle-to-ordered endpoint ratio 131.94, coarse-token energy
ratio 0.1425). This supports the scoped factorization claim but does not turn
the Min-SNR treatment into a generation-quality result.

## Control boundary

The pilot result keeps continuation beyond 50K, full 300K, promotion, export,
release, and process signals disabled. GPU compute is currently idle. No new
training or sampling process was started during this audit.
