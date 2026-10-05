# Large-Scale Completion Audit Symlink Hardening

Date: 2026-09-02

## Scope

This was a CPU-only provenance hardening change. It did not start training,
sampling, promotion, export, release, or any other GPU work.

The large-scale completion auditor now rejects symlink chains for formal sample
directories, sampling reports/manifests/progress, real-set roots, inference
smoke outputs, checkpoints, inference artifacts, deployment sources, and CLI
project/output roots. Invalid checkpoint and artifact evidence keeps the
lexical path instead of following a symlink while constructing an error report.

The stability completion auditor uses the same fail-closed path policy for
source descriptors, visual panels, discovered PNGs, project/output roots, and
the deployed training checkout.

## Verification

- Focused completion-audit tests passed, including POSIX-only file, parent,
  root, and progress symlink cases.
- Full local `pytest -q` passed at 100% (only pre-existing skip cases).
- `python -m compileall -q src tests scripts/audit_generation_stability_completion.py`
  passed.
- The remote `pro6000` pair monitor remains `pass`/`complete`; both matched
  runs remain at 100,000 steps and 6,400,000 images seen; GPU usage is 0 MiB.
- Terminal completion and strong-baseline comparison remain operational
  `pass` with scientific `hold`; `generation_advantage_proven` remains
  `false`.
