# Stability completion audit and post-eval waiter v2

Date: 2026-07-31

## Scope

This change closes two deployment-readiness gaps without modifying the active
50K training checkout or authorizing full ImageNet-256 training:

1. A stability-specific completion audit now binds the 5K qualification,
   matched 50K promotion gate, matched full 300K training, four protected
   milestones, physical formal sample sets, final gate, strong comparison,
   and release-authorized EMA inference artifacts.
2. The stability 50K pair-summary entrypoint now supports direct execution
   with `PYTHONPATH=src`, matching the server runbook environment.

The audit is fail closed. It reports `incomplete` when any required evidence is
absent and `failed` when present evidence is invalid. Neither state can be
reported as a completed generation system.

## Code identities

- Local stability branch commit:
  `aed20142a496c3c16f9b2e8c8aba4a465fbaf4d7`.
- Active 50K training:
  `2c2c1f5166b73d4f28df93b276901671ac1a7836`,
  branch `scale/generation-stability-50k-preflight`.
- New isolated post-evaluation checkout:
  `/tmp/cofitok-stability-50k-posteval-aed2014`,
  branch `scale/generation-stability-50k-posteval-v2`.
- Official server repository remained at
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066`.

## Validation

- Local full suite passed with the existing three skips.
- Linux isolated checkout collected 780 tests after excluding
  `tests/test_aaai27_experiment_structure.py`, whose four tests require the
  sibling `/tmp/paper` tree absent from isolated code worktrees.
- Linux result: 777 passed, 3 skipped.
- Direct CLI checks passed for
  `build_generation_stability_50k_summary.py --help` and
  `audit_generation_stability_completion.py --help`.
- All 95 shell runbooks passed `bash -n`.
- The empty-workspace audit exercise produced 13 missing checks, zero failed
  checks, and status `incomplete`.

## Waiter replacement

The original waiter at PID `442981` was verified idle with no child process,
and its status was archived before termination. The replacement waiter was
launched as PID `900875` from the clean `aed2014` checkout.

The replacement status binds:

- training revision/branch:
  `2c2c1f5` / `scale/generation-stability-50k-preflight`;
- evaluation revision/branch:
  `aed2014` / `scale/generation-stability-50k-posteval-v2`;
- 5K decision SHA256:
  `d5a6fc017f20c7d024abfab1967ba6bc966b624e3a77e9246292ddaaf7dc1da4`;
- `formal_300k_allowed=false`.

GPU process membership was identical before and after the replacement:
training PID `319202` only. A follow-up read observed CoFiTok step 2,300,
147,200 images seen, a running monitor, and no monitor issues.

## Completion boundary

This record proves that the completion-audit machinery and post-evaluation
handoff are ready. It does not prove that the matched 50K gate, full 300K
training, final 50K sampling, final scientific gate, or inference release has
passed. Full 300K remains unauthorized until the 50K promotion gate and its
source-bound validator pass.

Follow-up commit
`edb2a0dfeb5670974abe253c735d408630f84034` also records the execution
Git identity and CPU runtime-environment SHA256 in each EMA artifact export
report. The release audit now requires export, GPU preflight, and inference
smoke to bind the expected clean export revision; preflight and smoke must also
share the same recomputed runtime environment.

Evidence:

`artifacts/reports/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher/completion/stability_completion_audit_and_waiter_v2/`
