# Full-data matched 50K training-exposure snapshot waiter

Date: 2026-08-19

## Evidence gap

The full-data quality-bridge runbook trains each method in 50K and 100K
segments. Each segment atomically writes the same mutable
`training_report.json` path. The paired 50K quality report binds generated
samples and checkpoint evaluations, but it did not retain the two 50K training
reports. Once 100K completes, a reader could no longer reconstruct the exact
matched 50K dataset-normalized exposure from authoritative source reports.

## Implementation

Commit `d8749a6aea927545eb194f290a3b46017d7d99dd` adds two protections:

1. `build_generation_training_exposure_audit.py` can bind a matched milestone
   report. It verifies the two training reports have identical formal dataset
   identity, effective batch, steps, images seen, equivalent epochs, target
   horizon, completion state, clean Git identity, and exact milestone
   checkpoint SHA256 values. It also reopens all four content-addressed
   generation/mechanism reports and requires their Git identities to equal the
   training identity.
2. `wait_for_generation_training_exposure_audit.py` waits for both mutable 50K
   training reports and the exact paired milestone report, copies the source
   reports into a new output directory, builds the audit from the copies, and
   verifies byte/content-addressed replay. Existing outputs are accepted only
   after full replay; partial or changed outputs fail closed.

The waiter cannot launch training, sampling, evaluation, 300K scaling, or a
release. Its tracked runbook forces `CUDA_VISIBLE_DEVICES=-1`, one OMP/MKL
thread, the exact preparation SHA256, and exact code revision/tree/branch.

## Verification

- Local focused suite: `41 passed`.
- Linux focused suite with CUDA hidden: `41 passed`.
- Python compile: pass.
- Linux runbook `bash -n`: pass.
- Bundle: `43,616,182` bytes, SHA256
  `15efd876804e617d988513d33533235f534ce76d9abd0772c6f4792c62e7bb71`.
- Remote checkout: tracked clean at revision `d8749a6...`, tree
  `d4a6adc4655f9954afabfcf56f4a4086ad7815c7`.

## Deployment

The CPU-only waiter is running as PID `400399` from:

```text
/root/autodl-tmp/CoFiTok/checkouts/training-exposure-waiter-d8749a6
```

Its initial status was `waiting_for_dense_training_milestone`. At deployment,
dense identity was healthy at step `24,850/50,000`; the only GPU process was
the existing dense trainer PID `79894`. The formal checkout remained tracked
clean at `scale/generative-system@1ebcc152...`.

The eventual immutable output will be written to:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/reports/
training_exposure_step_00050000
```

The output is evidence for matched training exposure and the non-formal 2,048
sample milestone only. It cannot establish terminal generation quality or
authorize any subsequent stage.

Machine-readable deployment evidence:

```text
artifacts/reports/generation/
training_exposure_snapshot_waiter_2026-08-19/deployment_receipt.json
```
