# Generation-system full CPU regression (2026-09-02)

## Scope

This was a local, CPU-only regression run after the generation-system audit and
path/symlink hardening changes.  It did not connect to `pro6000`, launch a
trainer or sampler, write a checkpoint/output root, send a process signal, or
change any authorization boundary.

## Command and result

From `CoFiTok-internal`:

```text
.venv\\Scripts\\python.exe -m pytest -q
```

The command exited with code `0`.  A separate collection pass reported
`1170` tests across `146` test modules.  The run produced no failure output;
platform-dependent skips remain governed by the existing test markers.

## Interpretation

The current local implementation remains regression-clean across the full
test suite, including training/recovery, sampling and inference provenance,
checkpoint integrity/retention, generation gates, process monitoring, and the
new goal-audit/path-security checks.  This is software/reproducibility
evidence only; it does not change the remote 100K scientific result, whose
terminal status remains `hold` with
`generation_advantage_proven=false`.

After the source-report fail-closed hardening, the same full-suite command was
rerun and again exited with code `0`; the focused goal-audit file reported
`12` passing tests.
