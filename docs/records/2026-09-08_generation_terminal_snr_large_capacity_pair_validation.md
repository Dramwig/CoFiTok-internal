# Terminal-SNR large-capacity matched-pair validation receipt (2026-09-08)

## Outcome

The fresh endpoint-0.975 base-256 / approximately 250M-parameter 300K path now
has a dedicated immutable matched-pair validation receipt.  The receipt binds:

- the exact passing large-capacity preparation and user-goal stage
  authorization identities;
- one exact clean execution checkout revision, tree, and branch;
- the canonical CoFiTok and dense config paths, bytes, and SHA256 values;
- the `stability_full` pair/recipe validation, exact parameter counts,
  effective batch 64, and 300K milestone plan;
- fresh initialization with checkpoint resume forbidden.

Implementation:

- `src/cofitok/generation/terminal_snr_large_capacity_execution.py`
- `scripts/terminal_snr_large_capacity_pair_cli.py`
- `scripts/build_generation_terminal_snr_large_capacity_pair_validation.py`
- `scripts/validate_generation_terminal_snr_large_capacity_pair_validation.py`
- `tests/test_generation_terminal_snr_large_capacity_execution.py`

## Fail-closed boundary

This receipt is prelaunch evidence, not an execution or training
authorization.  It keeps GPU execution, training, sampling, evaluation,
checkpoint mutation, resume, full-300K launch, promotion, export, release, and
process signalling false.  Runtime selection, storage capacity, a live
GPU/process/output snapshot, a separate source-bound execution authorization,
and an immutable launch receipt all remain required.

The builder refuses config identity drift from the preparation, noncanonical
config filenames, a changed clean Git identity, a changed stage authorization,
or a pair/recipe replay that differs from the preparation.  The physical CLI
also requires the two configs to be the canonical files in the selected
execution checkout and rehashes every input before and after loading.

## Verification

- focused large-capacity tests pass: `28 passed`;
- broader terminal-SNR / large-capacity tests pass;
- both new CLI `--help` entrypoints, Python compilation, and
  `git diff --check` pass.
- the complete local test suite passes with no failures.

No remote checkout, GPU process, run directory, checkpoint, or locked evidence
was changed while implementing this receipt.
