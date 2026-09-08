# Terminal-SNR large-capacity stage authorization boundary (2026-09-08)

## Outcome

The terminal-SNR large-capacity path now has a dedicated user-goal stage
authorization contract.  It binds one physically validated passing
terminal-SNR preparation, one exact clean execution checkout, the two frozen
endpoint-0.975 base-256 configs, and the explicit CoFiTok completion goal:

- fresh matched approximately 250M-parameter CoFiTok/dense training;
- exact 300,000 steps per method at effective batch 64;
- protected 50K/100K/200K/300K milestones;
- per-method 50K EMA DDIM-250 formal evaluation;
- final gate, release-authorized EMA artifacts, locked paper integration, and
  terminal completion audit.

Implementation:

- `src/cofitok/generation/terminal_snr_large_capacity_execution.py`
- `scripts/terminal_snr_large_capacity_stage_cli.py`
- `scripts/build_generation_terminal_snr_large_capacity_stage_authorization.py`
- `scripts/validate_generation_terminal_snr_large_capacity_stage_authorization.py`
- `tests/test_generation_terminal_snr_large_capacity_execution.py`

## Fail-closed boundary

This stage artifact is intentionally not the later execution authorization.
It permits only the exact prelaunch evidence collection required by the active
goal: matched-pair validation, real-forward runtime selection, storage
capacity, and a live GPU/process/output snapshot.  It explicitly keeps all of
the following false:

- training, sampling, and evaluation launch;
- checkpoint mutation or resume;
- full-training / full-300K launch;
- promotion, export, and release;
- process signalling or checkout mutation.

A separate source-bound execution authorization and immutable launch receipt
remain mandatory.  Therefore this implementation cannot start the 300K stage
and cannot reinterpret the current screen or future confirmation as implicit
training authority.

## Verification

- canonical preparation-to-stage integration is covered;
- exact clean Git, output root, config identities, fresh initialization,
  300K horizon, milestones, formal evaluation, and goal binding are checked;
- unexpected nested fields, resume permission, dirty Git, and implicit 300K
  launch permission are rejected;
- focused terminal-SNR large-capacity / recipe suite passes;
- both stage CLI `--help` entrypoints, Python compilation, and
  `git diff --check` pass.
- focused large-capacity / recipe tests: `44 passed`;
- broader terminal-SNR / decision / milestone / runbook tests: `137 passed`;
- full local suite after the legacy-executor cleanup and this implementation:
  `1314 passed, 12 skipped in 307.63s`.

No remote checkout, GPU process, run directory, or locked artifact was changed
while implementing this boundary.
