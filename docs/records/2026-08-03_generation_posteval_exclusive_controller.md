# Generation post-evaluation exclusive controller

Date: 2026-08-03

## Problem

The stability 10% and stability-full formal post-evaluation runbooks already
bind training/evaluation revisions, checkpoint integrity, sampling manifests,
sampling progress, metrics reports, and (for the full run) stage receipts.  The
stage receipts can recover a detached worker after its parent is interrupted,
but they do not serialize two independently launched top-level runbooks.  A
duplicate controller could therefore race through runtime batch selection or
start a second checkpoint evaluation / 10K or 50K sampling job against the
same immutable output paths.

That is both a GPU-cost risk and an evidence-integrity risk.  Per-stage resume
contracts are intentionally fail-closed, but they are not a replacement for
single-controller ownership.

## Change

Both current formal stability post-evaluation entrypoints now acquire a
non-blocking file-descriptor lock before the first report write or GPU stage:

- `stability_scaling_50k_ema_teacher/posteval.lock`
- `stability_full_300k_ema_teacher/posteval.lock`

The lock is held by fd 8 for the lifetime of the runbook.  A concurrent
controller fails closed with exit code 75 and cannot enter validation, runtime
selection, checkpoint evaluation, sampling, metrics, visual audit, or gate
construction.

The lock is deliberately scoped to each authoritative output root.  It does
not signal, pause, or inspect unrelated processes, and it does not grant any
authorization to launch the full 300K training queue.

## Verification

Runbook contract tests assert that lock acquisition precedes the first evidence
write and the first GPU stage for both post-evaluation profiles.  The changed
shell entrypoints must also pass Linux `bash -n`; the normal Python test suite
continues to cover stage receipts, sampling resume, completed-result replay,
gate provenance, and the contextual-only treatment of official baselines.

Final verification completed on 2026-08-03:

- focused post-evaluation/runbook tests: `13 passed`;
- full repository pytest: exit code `0` (`942 passed`, `6 skipped`, derived
  from the unchanged prior suite plus the two new collected tests);
- both changed entrypoints: Linux `bash -n` passed from their exact working-tree
  bytes streamed to `pro6000` over stdin;
- `git diff --check`: passed.

No active remote checkout was moved or patched.  The live dense trainer and its
existing post-evaluation/readiness waiters remain on their immutable revisions.
