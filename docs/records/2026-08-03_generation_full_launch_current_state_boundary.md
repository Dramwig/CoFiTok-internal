# Full-training launch current-state boundary

Date: 2026-08-03
Branch: `scale/generation-large-capacity`
Base revision: `0623d06e16278911cc4ae0feb2bb1be5f2973706`

## Problem

The stability-full launch receipt API accepted
`require_current_runtime_environment` and
`require_current_formal_repository`, but the receipt builder did not fully
propagate those policies. The readiness bridge happened to recompute the
current runtime unconditionally, while the launch builder never directly
replayed the target deployment receipt with the requested formal-repository
freshness policy. This mixed two different questions:

1. Is the machine/repository/checkout exactly valid **now**, before starting or
   resuming an expensive 300K run?
2. Can a later completion audit reproduce the immutable launch decision after
   the repository or host environment has legitimately moved on?

An accepted but ineffective policy flag is unsafe at the launch boundary, and
an unconditional current-host dependency is wrong for historical replay.

## Change

`build_generation_full_launch_receipt.py` now:

- rehashes every launch source before verification;
- directly replays the target deployment receipt and propagates
  `require_current_formal_repository`;
- requires the deployment receipt's clean checkout Git identity to equal the
  requested full-training revision/branch;
- when current Git is required, requires that the launch process is running
  from that exact deployed checkout;
- propagates `require_current_runtime_environment` into the readiness bridge.

`build_generation_full_readiness_bridge.py` now separates immutable replay from
current-state verification. At launch/resume it recomputes the runtime
environment SHA from the current target checkout and requires equality with the
frozen readiness selection. During a historical audit it reuses the recorded
runtime SHA and still rebuilds the bridge from all immutable reports, Git blob
manifests, checkout identities, and hashes without requiring the auditor's
current host to reproduce the launch host.

`audit_generation_stability_completion.py` explicitly selects that historical
mode for both the bridge and launch receipt. This does not weaken training or
artifact runtime identity checks: checkpoints, training reports, formal
sampling reports, and EMA exports remain bound to their recorded runtime
environment SHA values.

## Authorization boundary

This is control-plane hardening only. It does not create a readiness bridge,
deployment receipt, supplemental quality pass, launch receipt, or human launch
authority. It does not authorize or start full 300K training. The active 50K
dense recovery and all waiters remain outside this change.

## Validation

- focused launch/bridge/completion/runbook/entrypoint tests:
  `36 passed in 12.58s`
- `python -m compileall -q scripts src tests`: pass
- complete local pytest: `1046 passed, 6 skipped in 236.77s`
- incremental bundle from prerequisite
  `2448ce5e6a96beca6762eba1c8eef1b05fc40cfd` to implementation revision
  `b45c490429c4e7f7ebeb1cc7eeca9ac66eaf2402`: `10,139` bytes, SHA256
  `8336c4214e30e2a4b4eea2a7836e53bf0a8a944a4088fd969e610795b790851b`;
  local and remote `git bundle verify` passed and advertised only the target
  branch/head
- clean isolated Linux checkout:
  `/tmp/cofitok-launch-current-state-b45c490/CoFiTok-internal`, exact branch
  `scale/generation-large-capacity`, exact revision `b45c490429c4e7f7ebeb1cc7eeca9ac66eaf2402`
- isolated CPU-only Linux pytest with `CUDA_VISIBLE_DEVICES=""` and
  `PYTHONPATH=.:src`: `1050 passed, 2 skipped in 151.78s`
- all tracked runbooks: `104/104` passed `bash -n`
- the launch-receipt validator/builder and readiness-bridge validator/builder
  all completed direct `--help` entrypoint imports: `4/4`
- `git diff --check`: pass

## Live-system boundary check

The rehearsal did not move any active checkout and did not signal, pause,
restart, or replace any process. The final read-only pro6000 snapshot found:

- dense metrics at step `27,400`, `1,753,600` samples, `549` rows, all finite,
  strict step monotonicity, and exact `samples_seen=step*64`; the latest 50-step
  interval was about `127.02s`;
- the pair monitor at its preceding poll remained `running` in
  `dense_identity_training`, with dense step `27,350`, progress `0.547`, and no
  issues; recovery and watchdog were also `running`;
- the latest protected checkpoint remained the exact 25K artifact:
  `1,006,120,214` bytes, SHA256
  `e73c935c488e234e80136b9507d3c84fca6015a760516dfec1bdf557871fb8ea`,
  with matching `latest.json` and integrity sidecar bound to immutable training
  revision `2c2c1f5166b73d4f28df93b276901671ac1a7836`;
- the recovery controller retained fd 6 on `dense_recovery.lock`; its child
  chain inherited the same fd. The training and controller checkouts were clean
  at their required branches/revisions;
- GPU compute contained only trainer leader PID `541878` using `77,970 MiB`;
  the filesystem had `182,812,217,344` free bytes;
- post-eval, readiness, and frozen supplemental waiters remained `waiting`
  with no child; every launch boundary retained
  `full_training_launch_allowed=false`.

This work does not rerun CoFiTok, authorize full training, or launch full 300K.
