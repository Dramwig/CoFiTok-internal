# Generation capacity qualification pipeline

Date: 2026-09-03 (Asia/Shanghai)

## Scope

This change adds the source-bound control and validation path needed after the
bounded 100K-to-110K exposure continuation. It does not change the running
exposure model, objective, optimizer, sampling protocol, or locked evidence.
It also does not authorize capacity execution, full training, 300K training,
promotion, export, or release.

The implementation covers:

- physical replay and a content-addressed validation receipt for the exposure
  result;
- a non-authorizing scientific exposure/capacity route decision;
- a fresh four-arm base-128/base-256, CoFiTok/dense 10K capacity screen;
- exact stage authorization, execution authorization, live snapshot, storage,
  runtime-selection, launch-receipt, and sibling-lock contracts;
- a separate four-arm 10K-sample confirmation that evaluates only the frozen
  screen checkpoints and cannot launch training;
- fail-closed result builders and physical replay validators for both stages.

## Git identity

- Parent/exposure execution revision:
  `5c23141a24a3385101ed6081c1ed656aad955a99`
- Target revision:
  `3370e68727259fb9144216d5c50ec9262645dc86`
- Target tree:
  `5dc3e72beb969e574aec60e652f761602c2ae31b`
- Target branch: `scale/generation-capacity-source-compatible-v1`
- Target tracked worktree: clean

## Verification

- Focused capacity/exposure tests passed on Windows after the final
  pre-output-root failure handling changes.
- The complete Windows pytest suite reached 100% with exit code 0.
- Pytest collection contains 1,204 tests.
- `python -m compileall -q src scripts tests` passed.
- `git diff --check` and the pre-commit staged diff check passed.
- A Linux isolated checkout at
  `/tmp/cofitok-capacity-rehearsal-3370e68` resolved to the exact target
  revision and tree with a clean tracked status.
- Linux `bash -n` passed for all 110 tracked shell runbooks. The enumeration
  included both `generation_capacity_screen_10k.sh` and
  `generation_capacity_confirmation_10k.sh`.
- The first Linux full-suite invocation omitted the checkout `PYTHONPATH`; one
  subprocess-import test consequently failed with `ModuleNotFoundError`.
  With the formal source environment
  `PYTHONPATH=$checkout:$checkout/src`, that test passed independently and the
  complete CUDA-disabled, single-thread, low-priority Linux pytest suite then
  reached 100% with exit code 0.

## Incremental bundle

- Local/remote filename:
  `cofitok-generation-capacity-3370e68-from-5c23141.bundle`
- Bytes: `132333`
- SHA256:
  `002d92f773cd381a719b15214106f03a2656b5e7c690ae89523c01d4812cfca2`
- Sole advertised ref:
  `3370e68727259fb9144216d5c50ec9262645dc86 refs/heads/scale/generation-capacity-source-compatible-v1`
- Required prerequisite:
  `5c23141a24a3385101ed6081c1ed656aad955a99`
- Remote copy:
  `/tmp/cofitok-generation-capacity-3370e68-from-5c23141.bundle`

The bundle was verified on both hosts. Creating and rehearsing it did not move
the active exposure checkout at `5c23141...` or the formal repository checkout
at `1ebcc152...`.

## Live boundary at verification

The authoritative exposure controller remained healthy in
`training_dense_identity`. CoFiTok was complete at 110,000 steps and 7,040,000
samples seen; dense had reached 109,000 steps and 6,976,000 samples seen. GPU
compute ownership remained solely with dense trainer PID 911658. No process was
signalled, paused, killed, or restarted.

No `exposure_capacity_result.json`, capacity-screen output root, capacity
confirmation output, full-training launch receipt, or 300K authorization was
created by this implementation/rehearsal step. The next scientific action must
be derived from the physically validated exposure result.
