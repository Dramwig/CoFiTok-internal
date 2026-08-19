# Matched 50K training-exposure snapshot waiter

This evidence records the CPU-only deployment that freezes both mutable
`training_report.json` files when the full-data quality bridge reaches its
matched 50K milestone. The parent runbook subsequently resumes each method to
100K and overwrites those paths, so the snapshot is required to retain an
auditable, checkpoint-bound 50K training exposure comparison.

The waiter is pinned to commit `d8749a6aea927545eb194f290a3b46017d7d99dd`
and tree `d4a6adc4655f9954afabfcf56f4a4086ad7815c7` in an independent remote
checkout. Local and Linux focused suites both passed `41` tests. The Linux
runbook passed `bash -n`.

At deployment, the waiter was PID `400399`, had
`CUDA_VISIBLE_DEVICES=-1`, and reported
`waiting_for_dense_training_milestone`. The only GPU process remained the
active dense trainer PID `79894`; no signal or mutation was applied to that
process or the formal checkout.

When the exact matched milestone report appears, the waiter will:

- copy the CoFiTok and dense 50K training reports before later resume stages
  can replace them;
- bind both reports to the exact milestone checkpoint SHA256 values;
- verify formal dataset identity, effective batch, images seen, equivalent
  epochs, Git revision/branch, and all four milestone evaluator identities;
- write an immutable snapshot manifest and replayable exposure audit;
- exit without launching training, sampling, evaluation, release, or any GPU
  stage.

Machine-readable deployment evidence is in `deployment_receipt.json`.
