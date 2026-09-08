# Terminal-SNR capacity branch legacy executor cleanup (2026-09-08)

## Scope

The terminal-SNR large-capacity branch must advance from a physically passing
frozen endpoint-0.975 confirmation to a **freshly initialized** matched
base-256 / 300K stage.  It must not resume the superseded base-256 capacity
probe at step 10K or reinterpret that probe as full-stage authorization.

## Defect

Commits `cbd4e4a` and `d4f2ca7` copied the older 10K-to-50K
capacity-scaling executor, result builder, runbooks, and tests into this branch
without the historical `capacity_probe*` implementation on which they depend.
Consequently, full pytest collection failed on four missing modules:

- `cofitok.generation.capacity_probe`
- `cofitok.generation.capacity_probe_execution`
- `cofitok.generation.capacity_probe_result`
- `cofitok.generation.capacity_probe_training`

Restoring those historical modules would also restore an execution protocol
that conflicts with the selected terminal-SNR route: exact resume from an old
step-10K probe into a 50K partial stop.  That is not the authorized or planned
large-capacity experiment.

## Resolution

The undeployed legacy 10K-to-50K execution surface was removed:

- two GPU runbooks and their supervisor;
- launch, partial-training, and 50K-result builders/verifiers;
- the three corresponding `capacity_scaling_*` implementation modules;
- their tests and milestone-source profile entries.

The source-bound capacity-scaling **decision** remains as immutable,
non-authorizing historical logic.  The active route is
`terminal_snr_large_capacity.py`, whose preparation requires a physically
replayed passing frozen confirmation and explicitly fixes:

- fresh initialization (resume forbidden);
- endpoint fraction `0.975`;
- matched base channels `256` / approximately 250M parameters;
- exact `300,000` training steps and protected 50K/100K/200K/300K milestones;
- per-method 50K EMA DDIM-250 formal evaluation;
- a later, separate source-bound stage/execution authorization and launch
  receipt before any GPU work.

This cleanup does not authorize checkout deployment, GPU use, training,
sampling, evaluation, release, or process signalling.  It does not mutate the
running terminal-SNR screen checkout or any locked evidence.

## Verification

Verification results are recorded in the commit that contains this document.
At minimum, pytest collection, the terminal-SNR focused suites, runbook syntax,
Python compilation, and `git diff --check` must pass before the cleanup is
committed.
