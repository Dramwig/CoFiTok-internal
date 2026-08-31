# Generation checkpoint SHA256 validation hardening (2026-09-01)

## Scope

This record documents a small control-plane hardening change on the isolated
exposure/capacity preparation branch. It does not authorize or launch training,
sampling, promotion, export, release, or a new GPU experiment. The formal
100K quality-bridge evidence remains unchanged.

## Change

`cofitok.training.checkpointing.verify_training_checkpoint` now requires every
serialized SHA256 value that is present in an integrity sidecar to be exactly
64 lowercase hexadecimal characters. The check covers:

- the checkpoint payload digest;
- optional runtime-environment and dataset-identity digests; and
- authorization-gate and authorization-gate-identity digests when the
  authorization metadata is present.

The validator still preserves the existing optional-field behavior: absent
optional digests are not fabricated, while malformed values fail closed before
the checkpoint can be trusted. A parametrized regression test exercises each
digest field with a non-hex value.

## Verification

Using the isolated worktree's project `.venv` with
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`:

- `tests/test_generation_inference_artifact.py` and
  `tests/test_generation_milestone_report.py` passed sequentially;
- the checkpoint-generation-system regression tests passed;
- `python -m compileall -q src scripts` passed; and
- `git diff --check` passed.

A previous broad duplicate run exposed Windows/OpenBLAS memory pressure in
subprocess-heavy tests; the focused rerun with single-threaded BLAS completed
without failures. No remote checkout, locked artifact, or GPU process was
modified.
