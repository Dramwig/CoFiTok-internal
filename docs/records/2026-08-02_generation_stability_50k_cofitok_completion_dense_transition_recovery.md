# Stability 50K CoFiTok completion and dense-transition recovery

Date: 2026-08-02 CST

## Verified outcome

The CoFiTok member of the matched ImageNet-256 10% stability queue completed
all 50,000 optimizer steps. This is a training-completion result, not a matched
pair result and not a sample-quality result.

```text
training revision: 2c2c1f5166b73d4f28df93b276901671ac1a7836
training branch: scale/generation-stability-50k-preflight
completed/target steps: 50000 / 50000
watchdog status: passed
watchdog child exit: 0
checkpoint: checkpoint_step_00050000.pt
checkpoint bytes: 1006325418
checkpoint SHA256: ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a
integrity sidecar SHA256: 4f3f6f3f401f34131f902b016baf41897b35f95d242bb023981feff4a36226a8
retained recovery checkpoints: 40K / 45K / 50K
```

The checkpoint SHA256 above was recomputed over the remote 1.006 GB payload,
not copied from `latest.json`. The report, `latest.json`, sidecar, dataset
identity, runtime-environment identity, revision, branch, dirty flag, byte
count, and step agree.

## Transition failure

The runbook exited immediately after writing the final CoFiTok training report
and before creating the dense run directory. The authoritative transition log
is 19,952 bytes with SHA256
`36b7876a54648f1bf42f535fa8d360014208c8af7055ecd22cd86ed97f17d1eb`.
Its final traceback is:

```text
ValueError: training report Git identity differs
```

The failure was deterministic. The runbook accepted and required
`EXPECTED_TARGET_BRANCH=scale/generation-stability-50k-preflight`, but
`validate_completed_generation_training` hard-coded
`scale/generative-system`. The report's Git identity and the training checkout
were correct; the completion helper's branch expectation was not.

The pair monitor later failed closed because neither a dense trainer nor the
runbook remained. The post-eval and readiness waiters then failed closed in
sequence. `full_training_launch_allowed` remains `false`.

## Milestone-observer false failures

The 40K and 45K observers were read-only and did not affect training. Both
called the live progress auditor after the exact 36,545 resume. The auditor saw
the intentionally retained pause report (`training_complete=false`,
`stop_requested=true`, report checkpoint 36,545) behind canonical resumed
metrics and emitted:

```text
training report completed_steps does not match metrics
```

The progress evidence at each milestone was otherwise present. The fix is
explicit and narrow: only callers opting into
`--allow-stale-incomplete-training-report` may accept a report behind metrics,
and only when it proves a stop-requested incomplete report whose target and
own latest-checkpoint step are internally bound. Complete reports, unrequested
stops, wrong targets, wrong report checkpoints, and all other mismatches remain
invalid.

## Code remediation

The completion validator now requires an explicit expected branch. Every
tracked runbook call supplies either its pinned branch or a branch resolved
once before the command. Regression coverage proves both acceptance of the
stability preflight branch and rejection of mismatched branches.

An incident-specific dense-recovery runbook was added. It separates the clean
recovery-control checkout from the immutable training checkout, independently
validates the completed CoFiTok checkpoint trust boundary, binds the frozen
`64x1` runtime selection, rechecks config fairness and storage, refuses a busy
GPU or duplicate queue, and launches only the dense member from the original
training revision. It never launches full 300K.

The initial remediation commit is
`00549fe3ec4ea5334c76e7249537224104ee44be`. Its bundle from the server's
available prerequisites is `39,681,857` bytes with SHA256
`833e7902c251e61ade15c68fc8b55d047c9982d4578a58a9a5a8a6998f734177`.
Bundle verification passed before fetch, and a new isolated control checkout
was created at
`/tmp/cofitok-stability-dense-recovery-control-00549fe` on branch
`scale/generation-stability-dense-recovery-control-00549fe`. The official
repository HEAD and immutable training checkout were not moved.

A source-binding increment then fast-forwarded only the isolated controller to
`e5c9ed7bd4590f5dd6dd0d78308c2ff8e868b58a`. The incremental bundle is
`7,036` bytes with SHA256
`fa8e41d2531cd89bfbfef23610838a38f3ebb3f3ce11bc38bab11bb76889ef0b`.

`PREFLIGHT_ONLY=true` then completed with `status=prepared`. It independently
revalidated the CoFiTok completion/checkpoint, frozen runtime SHA and `64x1`
selection, exact config hashes, matched config contract, both clean Git
identities, and storage. It did not start a runbook child, monitor, watchdog,
trainer, or GPU process. The fresh storage report records
`187,569,545,216 / 118,385,312,804 / 69,184,232,412` bytes for
free/required/headroom.

The recovery runbook deliberately does not create `pair_summary.json`.
After the monitor proves both training members complete, the exact post-eval
waiter must be the only component that builds that summary from the original
locked `config_validation.json`; this preserves the waiter's source-path and
SHA binding instead of substituting the equivalent recovery-preflight report.

Before any status write, checkpoint validation, monitor launch, or trainer
launch, the recovery controller now takes a non-blocking `flock` on
`stability_scaling_50k_ema_teacher/dense_recovery.lock` and holds file
descriptor 6 for the controller lifetime. A concurrent controller exits 15
without overwriting the active controller's status. This closes the remaining
TOCTOU window between duplicate-process observation and monitor launch while
leaving the immutable training checkout unchanged.

Validation:

```text
targeted regression tests: 50 passed
full pytest before controller-lock hardening: 903 collected / 897 passed / 6 skipped
full pytest after controller-lock hardening: 904 collected / 898 passed / 6 skipped
runbook entrypoint contract: passed
git diff --check: passed
recovery runbook bash -n on pro6000: passed
```

## Current recovery boundary

At `2026-08-02T12:00:14+08:00`, free checkpoint-filesystem space was
`187,909,763,072` bytes. The frozen matched-50K capacity plan requires
`118,385,312,804` free bytes including checkpoint, sample, additional, and
64-GiB safety reserves, so the read-only runway is positive. The recovery
runbook re-runs this check immediately before launch.

GPU PID `362355` was a FieldScope process using 15,412 MiB. It was not killed,
paused, signaled, or otherwise modified. Dense recovery therefore remains
prepared but not launched until the GPU becomes idle.

The authoritative dense state path is
`/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/dense_rollout_x0_u2_ema_teacher`.
At `2026-08-02T12:28:43+08:00` that directory was confirmed absent. The
similarly named `dense_rgbtail3_rollout_x0_u2_ema_teacher` path is not the run
directory selected by the frozen recovery runbook and must not be used to
decide whether a launch is fresh or resumable. The active launch-monitor prompt
was updated to bind this exact path; no remote training or GPU process was
started or modified by that correction.

Evidence:

`artifacts/reports/generation/stability_scaling_50k_ema_teacher/cofitok_50k_transition_incident_2026-08-02/`
