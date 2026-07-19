# Selected runtime completion binding

Date: 2026-07-20

## Problem

Formal full ImageNet-256 training benchmarks the matched runtime candidates
`16x4`, `32x2`, and `64x1` before launch. All candidates preserve an effective
batch size of 64, but the training completion validator previously compared the
resolved training report only with the unmodified checked-in `16x4` config.
Consequently, a valid run using a faster selected candidate would finish and
then be rejected as config drift.

The fresh rank-complete 10% pair also used the conservative `16x4` default
without a runtime benchmark, leaving substantial GPU memory and possible
throughput unused on the 96 GB device.

## Resolution

- `validate_completed_generation_training` now accepts the selected
  micro-batch size and gradient-accumulation steps as an all-or-nothing pair.
- The validator applies those two explicit overrides to the checked-in config
  before performing its existing full resolved-config equality check.
- Both formal 50K and 300K runbooks pass the frozen selected runtime into the
  training command and completion validator.
- The fresh 10% pair now uses the same shared runtime selector as full training:
  both CoFiTok and dense must complete a candidate, preserve effective batch 64,
  share environment and dataset provenance, and stay below 90% peak VRAM.
- Existing run state freezes the selection, so resume cannot silently choose a
  different runtime.

This changes runtime efficiency only. CoFiTok and dense still use the same
effective batch, data, optimizer, schedule, augmentation, and training horizon.

## Verification

- Focused completion, runtime-selection, workspace-path, and recipe tests:
  `33 passed`.
- Python compile check for `src/` and `scripts/`: passed.
- `git diff --check`: passed.
- Both modified runbooks copied to remote `/tmp` and passed Linux `bash -n`.

The active 5K rank-recovery probes remain pinned to remote revision
`05bbb4af63a9f1d9b7f11bc4222d50875e382e1d`; this change is not deployed until
the probe and posthoc model/EMA audit finish.
