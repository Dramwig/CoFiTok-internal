# Live audit later-checkpoint boundary

Date: 2026-08-21 (Asia/Shanghai)

## Finding

The active full-data bridge resumed exactly from its verified step-80,000
checkpoint. Once the resumed segment wrote step 85,000, the prior segment's
`training_report.json` was still intentionally stale at step 50,000 and the
content-addressed reconciliation correctly remained bound to step 80,000.

The first live-audit hardening required the reconciliation step to equal the
latest checkpoint step. It therefore rejected the otherwise healthy run after
the later checkpoint appeared.

## Minimal fix

The direct descendant of the active training revision now accepts the stale
segment report only when:

```text
stale_report_step < reconciliation.resume_step <= latest_verified_checkpoint_step
```

The canonical metrics prefix, reconciliation filename and payload, orphan
archive row count and SHA256, stale report target and checkpoint pointer, and
latest checkpoint integrity binding remain mandatory. A positive test covers
an 80K resume followed by an 85K checkpoint; a negative test rejects a
reconciliation newer than the latest verified checkpoint.

This is audit-only hardening. It does not change training, model weights,
sampling, gates, promotion, release, or full-300K authorization, and it was not
deployed into the active execution checkout.

## Validation

Using the independent integration virtual environment with `PYTHONPATH`
explicitly bound to this worktree and CUDA hidden, the auditor,
quality-bridge, and runbook-entrypoint group returned `44 passed`.

A read-only server invocation of this minimal candidate against the live bridge
then returned:

```text
status:                    healthy
issues:                    []
canonical step:            86,800
latest verified checkpoint: 85,000
accepted resume step:      80,000
training report status:    stale_segment_resume_report
```

The live snapshot SHA256 is
`68263a9ca1ab3366ae885007b8892fabe26344cbea5b9e6c5cdf13abcccaa23f`.

## Minimal-branch full regression

The minimal branch's full CPU-only suite collected `1,081` tests. Four legacy
AAAI structure tests derive the paper root from the worktree's physical parent;
because this isolated worktree is at `C:/qblive`, they look for the unrelated
path `C:/paper`. Their source is byte-identical after newline normalization to
the normal-layout test file.

The remaining `1,077` tests ran in this exact minimal worktree to 100% and
exited with code 0. The four layout-sensitive tests were then run from the
normal project layout against the same test source and returned `4 passed`.
CUDA remained hidden throughout. No active remote checkout or process was
modified.
