# Generation sampling selector matched-output lock

Date: 2026-08-03

## Problem

Formal post-evaluation selects one shared sampling batch by loading both matched
checkpoints and running measured GPU preflights over several candidate batch
sizes.  The selector already froze its decision as soon as formal sampling
state existed, while the samplers themselves already owned their individual
output directories.  There was still a race before either sample directory had
state: two selectors, or a selector and a direct sampler, could begin expensive
GPU work against the same matched outputs.

That race can duplicate roughly 2 GB of checkpoint loading, compete for GPU
memory, and make the selected runtime batch depend on concurrent work rather
than the locked formal protocol.

## Change

`cofitok.output_lock.exclusive_output_locks` now acquires any set of output
targets in normalized deterministic path order.  Targets must be non-empty and
distinct.  Acquisition is non-blocking; if any later target is already owned,
`ExitStack` releases every target acquired earlier in the same attempt.

`scripts/select_generation_sampling_batch.py` now holds both matched formal
sampling output directories before reading checkpoint identities, checking
sampling state, loading a checkpoint, creating benchmark outputs, or launching
a GPU preflight.  The lock role is
`generation_sampling_batch_selector`.  Existing frozen-selection, preflight
cache, checkpoint-integrity, clean-revision, and selection-policy contracts are
unchanged.

The persistent lock files remain adjacent to the formal sample directories, so
they are not sample-set entries and do not change sample-tree hashes.

## Verification

- Multi-target lock tests prove reversed inputs acquire in deterministic order
  and duplicate targets fail closed.
- A subprocess contention test holds the second matched output while invoking
  the real selector with nonexistent checkpoints.  It fails on output
  ownership before checkpoint access or GPU work, creates no benchmark,
  selection, or sample directory, and releases the first lock acquired during
  the failed attempt.
- Focused output-lock and batch-selection suite: `17 passed`.
- Full repository suite: `957 collected`, `951 passed`, `6 skipped`, exit code
  `0` in `276.9s`.
- `py_compile` and `git diff --check`: passed.

This change protects the reproducibility and availability of the formal 50K
quality evaluation.  It is not itself FID/IS/precision/recall evidence, does
not modify the active stability training, and does not authorize full 300K.
