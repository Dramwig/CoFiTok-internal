# Generation training progress auditor

Date: 2026-07-12

Branch: `scale/generative-system`

`scripts/audit_generation_training_progress.py` provides a read-only health
check for multi-day generation runs. It requires only the run directory and
does not instantiate the model or use the GPU.

The report verifies:

- valid JSONL with strictly increasing steps;
- finite total/epsilon loss, gradient norm, LR, and elapsed time;
- resume-aware timing segments when elapsed time resets;
- checkpoint due, grace, available, or overdue state;
- newest checkpoint and `latest.json` filename/step agreement;
- newest checkpoint byte count and SHA256, computed read-only for the pinned
  legacy queue or verified against a mandatory sidecar for upgraded runs;
- training-report completion agreement when a final report exists.

It records recent loss/gradient means, current segment throughput, ETA,
validation event count, checkpoint list, and gradient-clipping statistics. If
an evaluation interval is supplied, it also reports the expected event count
and emits a non-blocking warning when validation values are absent from JSONL.
`grad_norm` from `train_generation.py` is the total norm returned by
`clip_grad_norm_` before clipping; exceedances are counted for observability and
are not by themselves treated as failed clipping.

The first formal snapshot at step 4000 was produced remotely from a temporary
copy of the script and synchronized under
`artifacts/reports/generation/training_progress_2026-07-12/`. This preserved the
active training repository at commit `781a014`.

The pinned `781a014` training loop performs each scheduled validation forward
after emitting that step's JSONL row. Validation therefore runs but is not
persisted in `train_metrics.jsonl`; a live step-6400 audit correctly reports
`event_count=0`, `expected_event_count=6`, and the legacy-observability warning
while retaining `status=healthy`. The upgrade branch emits the log after
validation, so full 300K runs are expected to have complete validation logging.

The integrity policy is deliberately asymmetric during the transition. Current
10% checkpoints predate integrity sidecars, so `legacy_compute` hashes the
newest file without loading or rewriting it and emits a compatibility warning.
After both matched 50K runs finish, the existing migration validates exact-resume
payload fields and binds those same bytes. Full 300K audits pass
`--integrity-policy required`; a missing sidecar, changed byte count, SHA
mismatch, or stale `latest.json` binding makes the audit invalid.
