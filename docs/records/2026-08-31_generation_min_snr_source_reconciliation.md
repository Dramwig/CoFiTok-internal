# Min-SNR Pilot Source Reconciliation (2026-08-31)

## Scope

This record documents a local, CPU-only reconciliation of the matched Min-SNR
pilot implementation with the source revision that executed the pilot. It did
not create a remote output, modify a remote checkout, launch training or
sampling, send process signals, or change any locked quality result.

## Authoritative remote evidence

- Pilot source checkout: `/root/autodl-tmp/CoFiTok/checkouts/min-snr-matched-pilot-842a341/CoFiTok-internal`
- Pilot source revision: `842a34130e82f241330707118a05bf6ed01e263e`
- Pilot source branch: `scale/generation-min-snr-matched-pilot-v1-20260825`
- Pilot implementation SHA256: `81bb9f083170c251736c1bab4bb5f8781d78235ed3befb1ff38db4fd9c67e647`
- Pilot result: `/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/min_snr_pilot_result.json`
- Pilot result SHA256: `dde62640e84a461be3b571b228e7903785e42f118dd7103941e20592bd0059fc`

The result is `completed` with `scientific_status=screening_only`,
`terminal_status=hold`, and `generation_advantage_proven=false`. Its selected
route is `no_shared_min_snr_candidate_at_50k`, with the next-stage id
`stop_min_snr_route_and_reassess_training_objective`. The result boundary keeps
training, sampling, continuation beyond 50K, full training, 300K, promotion,
export, release, and process signals disabled.

## Pilot outcome

The four arms used the same 10K DDIM-100 protocol, balanced-modulo class
schedule, and shared random stream. The precommitted requirement was a shared
improvement for both matched methods, plus class-fidelity floors and the
CoFiTok mechanism checks.

- CoFiTok gamma 0 -> gamma 5: FID `146.3368 -> 144.3896` (1.33% relative improvement), precision `0.7119 -> 0.7453`, recall `0.00830 -> 0.00458`; the relative-FID and absolute class-fidelity gates failed.
- Dense gamma 0 -> gamma 5: FID `189.4888 -> 167.9816` (11.35% relative improvement), precision `0.8626 -> 0.8337`, recall `0.00494 -> 0.00562`; precision and absolute class-fidelity gates failed.
- CoFiTok mechanism checks passed: zero-token `0`, shuffled/ordered endpoint ratio `131.94`, ordered path-AUC rank `1`, coarse-token energy ratio `0.1425`.

These results do not establish a generation advantage and do not authorize a
new GPU stage.

## Local reconciliation

The local implementation now matches the executed pilot semantics:

- `loss.min_snr_gamma=0` preserves the historical raw epsilon reduction.
- Positive gamma uses `min(SNR, gamma) / SNR` with per-sample epsilon MSE.
- Canonical metrics are `epsilon` (weighted), `epsilon_unweighted`, and
  `min_snr_weight_mean`.
- The pilot recipe requires the 100K scheduler horizon and protected 50K/100K
  checkpoints; the runbook stops at 50K. A direct 50K scheduler is rejected by
  the recipe contract.
- The preparation, execution-gate, physical-training-audit, sampling-preflight,
  evaluation-arm, and result validators are source-compatible with the pilot.

CPU verification on the local worktree:

```text
56 passed in 7.25s
compileall: passed
```

The local main worktree remains intentionally dirty; unrelated user changes
were preserved. The remote quality-bridge result remains a non-authorizing
`hold` (`generation_advantage_proven=false`), and no follow-up GPU experiment
is started by this reconciliation.
