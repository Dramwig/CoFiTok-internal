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
- training-report completion agreement when a final report exists.

It records recent loss/gradient means, current segment throughput, ETA,
validation event count, checkpoint list, and gradient-clipping statistics.
`grad_norm` from `train_generation.py` is the total norm returned by
`clip_grad_norm_` before clipping; exceedances are counted for observability and
are not by themselves treated as failed clipping.

The first formal snapshot at step 4000 was produced remotely from a temporary
copy of the script and synchronized under
`artifacts/reports/generation/training_progress_2026-07-12/`. This preserved the
active training repository at commit `781a014`.
