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

## Step 40,600 snapshot

A second formal read-only audit was run on 2026-07-13 from upgrade commit
`080bc3ac988fa70fb01e654e35c9b1ec8369b91b`, while the active training worktree
remained pinned to `781a014`. The synchronized report is:

```text
artifacts/reports/generation/training_progress_2026-07-12/cofitok_step_040600.json
```

It records `status=healthy`, `issues=[]`, step `40,600/50,000`, and retained
recovery checkpoints at 30K/35K/40K. The 40K checkpoint has
SHA256 `f3af056d3a6945567fcef24c951efbd7e168ec97ec7611a64a4feb33dc16121f`.
Recent means are total loss `0.03161085`, epsilon loss `0.03093035`, and
pre-clip gradient norm `0.03448837`; measured throughput is `2.39359` seconds
per optimizer step with an ETA of about 6.25 hours for the CoFiTok half.

The active 10% legacy configuration intentionally rolls recovery checkpoints
and keeps the latest three. Its live audit must not list already-pruned 5K-25K
states as permanently required. Protected 50K/100K/200K/300K retention applies
to the upgraded full run, where the auditor receives those explicit required
steps and uses `integrity-policy=required`.
