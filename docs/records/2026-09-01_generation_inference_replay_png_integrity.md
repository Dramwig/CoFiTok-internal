# Inference replay PNG integrity hardening (2026-09-01)

## Scope

This record covers a CPU-only evidence validation improvement for the
reusable inference replay path. It does not authorize training, sampling,
promotion, export, release, or any change to the 100K quality-bridge result.

## Change

`validate_completed_inference_evidence` now derives the expected PNG channel
count and spatial dimensions from the immutable inference manifest request.
Each physically rehashed output must also pass PNG decoding, mode, and
dimension validation through `is_valid_png`. Malformed or unsupported manifest
shapes fail closed before a report can be treated as replayable evidence.

The new regression test updates a completed report and progress file with the
digest of a valid but wrong-sized PNG, then verifies that the completion audit
rejects it. This covers a coherent metadata-and-file tamper that a hash-only
check would otherwise accept.

## Verification

- `tests/test_generation_session.py`: 16 passed.
- `tests/test_generation_inference_artifact.py`,
  `tests/test_generation_sampling.py`, and
  `tests/test_generation_sampling_protocol.py`: 49 passed.
- `compileall` and `git diff --check` passed.

No remote checkout, checkpoint, locked report, or GPU process was modified.
