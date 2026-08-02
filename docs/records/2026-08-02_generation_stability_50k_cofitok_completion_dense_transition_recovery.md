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

Validation:

```text
targeted regression tests: 50 passed
full pytest: 903 collected / 897 passed / 6 skipped
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

Evidence:

`artifacts/reports/generation/stability_scaling_50k_ema_teacher/cofitok_50k_transition_incident_2026-08-02/`
