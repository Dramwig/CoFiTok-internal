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
- isolated CPU-only Linux pytest and runbook syntax: pending
