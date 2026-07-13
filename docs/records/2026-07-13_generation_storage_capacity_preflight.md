# Generation storage-capacity preflight (2026-07-13)

Large-scale generation produces protected recovery checkpoints, milestone PNGs,
two formal 50K sample sets, prefix diagnostics, and evaluator caches. Recording
free disk in a monitor does not prevent a later stage from exhausting storage.

`scripts/check_generation_storage_capacity.py` now writes an atomic structured
report and exits with code `78` when observed free bytes are below the planned
reserve. The plan accounts for conservative checkpoint copies, 256 KiB per
256x256 PNG, evaluator/cache allowance, and an explicit safety margin.

The formal runbooks check capacity before:

- 10% matched 10K post-evaluation: 20,256 planned PNGs, 16 GiB additional,
  and 32 GiB safety margin;
- full matched 300K training: 16 conservative checkpoint slots, 16,384
  milestone PNGs, 16 GiB additional, and 64 GiB safety margin;
- final matched 50K post-evaluation: 100,256 planned PNGs, 16 GiB additional,
  and 64 GiB safety margin.

Checkpoint size for the full-training plan is measured from both completed 10%
reference checkpoints and the larger value is used. Reports include filesystem
usage, every budget component, remaining headroom, hostname, timestamp, and Git
provenance. Exit code `78` is explicitly non-retryable in the bounded supervisor;
capacity recovery requires an operator to inspect or expand storage instead of
relaunching a stage that cannot finish.

The final completion audit requires all three reports, recomputes checkpoint,
sample, total-required, and headroom arithmetic, rejects weaker budgets, verifies
the generation filesystem path, and requires the clean deployed revision.
