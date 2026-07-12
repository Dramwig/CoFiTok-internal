# Pinned 10% pair read-only monitor

Date: 2026-07-12

Branch: `scale/generative-system`

The active 10% matched queue must remain on revision
`781a01444fddbf0d48a427ba58bdeed50167b5be`, so the newer bounded completion
supervisor cannot be deployed into that worktree until both 50K runs finish.
`scripts/monitor_generation_10pct_pair.py` provides operational visibility
without changing that scientific revision.

The standard-library monitor reads only metrics, final training reports, and
checkpoint file metadata. Every poll atomically publishes:

- current `cofitok_training`, `dense_identity_training`, transition, or complete
  stage;
- latest step, progress fraction, metric age, and checkpoint names/bytes;
- matched runbook and training process presence;
- filesystem total/used/free bytes and `nvidia-smi` utilization, memory, and
  temperature;
- explicit `running`, `waiting`, `stalled`, `failed`, or `pass` status and issues.

An active stage with no metric update for 1,800 seconds becomes `stalled`. An
incomplete pair without a training/runbook process becomes `failed` after a
600-second transition grace period. Both exact 50K completion reports are
required for `pass`. The monitor exits on terminal state; it does not restart,
signal, or kill any process and does not load or hash large checkpoint payloads.

The first remote one-shot probe reported `running/cofitok_training` at step
12,250 with two 1,008,218,658-byte checkpoints, no issues, approximately 598 GB
free disk, and an active GPU. The continuous copy is launched from `/tmp`, not
the pinned Git worktree.
