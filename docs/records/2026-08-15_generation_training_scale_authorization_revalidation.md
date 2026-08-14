# Training-scale authorization revalidation

Date: 2026-08-15

## Outcome

The source-bound capacity progression from the matched 250M/10K probe through
50K scaling, 100K completion, fresh full-300K readiness, and full-training
launch was revalidated on Windows and CUDA-hidden Linux. The same 33 test
modules collected 177 tests on both platforms. Both runs passed 177/177 with no
failures, errors, or skips. No implementation change was required.

This proves the fail-closed control and exact-resume contracts exercised by the
tests. It does not prove that the pending quality bridge, capacity probe, or
full-300K training has run; it is not sample-quality evidence; and it does not
create promotion, export, release, or formal-completion authority. The standing
experiment authorization remains an input to the declared supervisors, not a
substitute for their immutable source decision, execution identity, idle-GPU,
storage, checkpoint, and output-root checks.

## Progression boundaries covered

The probe tests cover source-bound preparation, paired configuration and
parameter checks, exact preserved base-128 step-10K references, monitoring,
bounded supervision, intentional step-10K stopping, terminal result replay, and
duplicate or mismatched execution refusal. Probe execution remains limited to
the exact 250M/10K diagnostic; it cannot authorize configured 100K completion or
full 300K.

The scaling tests require the exact probe result and a physically replayed
source decision before permitting matched checkpoint resume to step 50K. They
reject fresh training, wrong revisions or branches, incompatible checkpoints,
drifted reports, duplicate execution, and attempts to infer the next stage from
process exit alone. The scaling supervisor cannot directly launch configured
100K completion or full 300K.

The completion tests require the exact completed 50K scaling result, source
archive, decision, execution revision, and matched checkpoints before permitting
resume from step 50K to step 100K. The terminal result waiter replays the
physical training and decision sources and treats an existing result as valid
only when it is byte-equivalent. Completion output is not a promotion gate and
cannot itself launch full 300K.

The full-readiness tests require a source-compatible selected 100K result, a
new readiness decision, fresh output-root and storage validation, a bounded GPU
runtime benchmark, clean exact Git identities, and an immutable readiness
receipt. The readiness supervisor cannot launch training. The separate training
supervisor may launch only a fresh matched full-300K pair after the exact passed
readiness receipt; it must not resume the capacity-100K checkpoints, claim
formal generation completion, modify an unrelated GPU process, or authorize
release. Its exact-target resume path applies only after that fresh full run has
created its own bound checkpoints.

## Exact test evidence

Module groups and collected counts were:

```text
capacity probe preparation/execution/monitor/training/result       55
capacity scaling decision/execution/resume-to-50K/result           47
capacity completion decision/archive/resume-to-100K/result         54
full-300K readiness and fresh training launch/execution             21
                                                                  ---
                                                                  177
```

The 33 exact modules were selected from:

```text
tests/test_generation_capacity_probe*.py
tests/test_wait_for_generation_capacity_probe_preparation.py
tests/test_generation_capacity_scaling*.py
tests/test_generation_capacity_completion*.py
tests/test_generation_capacity_full_readiness*.py
tests/test_generation_capacity_full_training_execution.py
tests/test_generation_capacity_full_training_launch.py
```

The local Windows run used:

- checkout: `C:\qbfd2`;
- branch: `scale/generation-capacity-control-continuity-v1`;
- revision: `0f8ee080926ba0717ba0703d3df137597ac5718b`;
- tree: `0bfad1b6304fd453cd96f48a50e9c61cb649d7f6`;
- Python: `3.10.20`;
- tracked checkout state: clean;
- result: 177 passed, zero failures/errors/skips;
- JUnit time: `75.646` seconds;
- local JUnit bytes/SHA256: `29,404` /
  `3cab76463d22f0667b14363e66a05ff1402b4eeafb5233895aaf6b9aee1f626f`.

The authoritative Linux revalidation used:

- checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-rehearsal-47e59a0/CoFiTok-internal`;
- branch: `scale/generation-capacity-control-process-relaunch-rehearsal-v1`;
- revision: `47e59a0ce926c3ac1fdaa4860de84d2f27e39f76`;
- tree: `3a99993b66db675f262ed2e04298e1d4d64411cf`;
- Python: `3.10.20`;
- tracked checkout state: clean;
- CUDA policy: `CUDA_VISIBLE_DEVICES=-1`;
- result: 177 passed, zero failures/errors/skips;
- JUnit time: `394.599` seconds.

All files after the Linux revision and through the local test revision were
documentation-only; source and tests were identical between the two runs.

The persistent Linux JUnit report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/training_scale_authorization_revalidation_2026-08-15/pytest_linux_cuda_hidden_47e59a0.xml`;
- bytes: `29,450`;
- SHA256:
  `b62e482eed5cdf77c90f309004b2aed8a9712431c7837da68cf3a051eb49eff1`;
- file mode: `0644`.

## Linux invocation correction

The first Linux wrapper attempted to discover the module list with `rg`, which
was not on the non-login remote shell's `PATH`. The resulting empty argument
array caused pytest to start a broader repository run. After verifying the
exact parent command and its sole training-fixture child, only that newly
started CPU pytest process (PID 616463) and child (PID 633655) were terminated.
Both disappeared, no GPU was visible to the run, and no persistent report from
that interrupted invocation was accepted. The counted run used the explicit
33-file list above. Its pytest exit code was zero; the outer wrapper's later
nonzero status came only from a PowerShell CR character in `exit 0`, so the XML
was independently parsed and rehashed before acceptance.

## Preserved production state

The revalidation used temporary CPU fixtures and did not deserialize production
weights or change checkpoints, samples, metrics, decisions, gates, launch
receipts, or live control-process status. At the pre-record check, the full-data
quality bridge still had no trainer, sampler, evaluator, monitor, or controller
child. Its bounded recovery supervisor remained at attempt zero, waiting for
five consecutive idle-GPU polls while the unrelated FieldScope process occupied
the GPU. The formal remote checkout retained its previously recorded revision,
branch, and porcelain fingerprint.
