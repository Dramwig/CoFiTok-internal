# Stability-full readiness waiter (2026-07-31)

## Gap

The source-bound 50K post-evaluation waiter already turns a completed matched
training pair into formal EMA sampling and a scientific promotion gate. The
next 250M CUDA readiness stage still required a human to notice that the gate
had passed, recover the exact gate SHA256, confirm that the GPU was idle, and
invoke the receipt-bound readiness checkout. A multi-day training/evaluation
chain could therefore leave an expensive GPU idle even though the next
qualified operation was fully specified.

## Implementation

The new chain adds:

- `scripts/run_generation_stability_full_readiness_waiter.py`;
- `artifacts/runbooks/generation_stability_ema_teacher_full_readiness_waiter.sh`;
- `stability_full_300k_ema_teacher/reports/readiness_waiter.json`.

The waiter executes from the exact isolated large-capacity deployment checkout.
Before it can launch CUDA qualification, it requires and replays:

- the successful stability 50K post-evaluation waiter status;
- the passing schema-v2 scaling gate and every bound source report;
- the exact `stability_scaling` source profile;
- exact 50K training and post-evaluation revisions and branches;
- the versioned large-capacity deployment receipt and checkout identity;
- `readiness_execution_allowed=true` and
  `full_training_launch_allowed=false` in that receipt;
- an empty GPU compute-process list.

The readiness runbook repeats the gate, receipt, checkout, training-state, and
GPU-idle checks, so a process appearing between waiter observation and launch
causes a fail-closed exit rather than overlapping workloads.

## Recovery

The waiter writes an atomic status report while waiting, while the child
readiness process runs, and after completion or failure. If an immutable
`full_training_readiness.json` already exists, a restarted waiter does not
rebuild it. It invokes `validate_generation_full_readiness.py` with the current
runtime fingerprint and all bound source paths, then records
`existing_full_readiness_replayed` only after exact replay succeeds.

Partial readiness evidence is never overwritten. If CUDA qualification failed
after creating any evidence path, the readiness runbook refuses a second build;
a human must inspect that failure instead of silently mixing attempts.

## Authorization boundary

This waiter does not reference or invoke the full 300K training runbook, does
not create a launch receipt, and does not start `train_generation.py`. Every
status payload carries `full_training_launch_allowed=false`. Full training
continues to require the separately reviewed readiness SHA256 and an explicit
launch authorization.
