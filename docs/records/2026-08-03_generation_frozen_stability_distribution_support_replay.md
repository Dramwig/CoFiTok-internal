# Frozen stability-gate distribution-support replay

Date: 2026-08-03

## Problem

The active 50K stability post-evaluation checkout is intentionally frozen at
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289`. Its generation gate is schema v2:
it computes and source-binds 10K FID/IS/precision/recall, but precision and
recall are not blocking checks. Replacing that checkout while training is live
would break the immutable execution identity; ignoring the gap would leave the
current run unable to benefit from the newer schema-v4 non-collapse standard.

## Supplemental qualification

`scripts/build_generation_stability_distribution_support.py` provides a
CPU-only, deterministic replay after the frozen post-evaluation writes
`promotion_gate.json`:

1. Read the original gate bytes and bind their path, size, and SHA256.
2. Use the gate's own `source_reports` table to rehash all six authoritative
   reports and reject path, size, or SHA drift.
3. Reread the exact bound CoFiTok/dense generation-metrics bytes and recheck
   that they are matched 10K EMA DDIM-100 evaluations at final step 50K with
   the formal CFG/clipping/bf16/random-stream protocol.
4. Recheck the shared real set, torch-fidelity implementation, clean evaluator
   and sampler Git identities, and evaluator/sampling runtime identities.
5. Require finite in-domain FID/IS/precision/recall, CoFiTok precision and
   recall each at least `0.10`, and each no more than `0.05` below matched
   dense.
6. Rehash the base gate and all bound sources again before the atomic output
   write. A second writer fails on the output lock; `--resume` accepts only an
   exact deterministic reconstruction from unchanged sources and the same
   clean builder revision.

The report also exposes a separate `decision_boundary`: the original gate
result, this supplemental result, and their conjunction are distinct fields,
while `scaling_authorization_evaluated=false` and
`full_training_launch_allowed=false` remain fixed. This prevents a standalone
supplemental `status=pass` from being consumed as an expansion authorization.

Planned invocation from a clean deployed checkout after the frozen post-eval
finishes is:

```bash
python scripts/build_generation_stability_distribution_support.py \
  --gate /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/promotion_gate.json \
  --output /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/supplemental_distribution_support/qualification_report.json \
  --require-pass
```

## Claim and authorization boundary

The report role is
`generation_stability_distribution_support_qualification`. A pass means only
that the two source-bound 10K EMA sample sets clear the precision/recall
non-collapse floor and matched-retention checks. It does not change a failed or
passing base gate, does not supply the newer rollout-stability diagnostic, and
does not authorize full training. Every report fixes:

```text
supplemental_non_authorizing=true
replaces_generation_gate=false
replaces_rollout_stability_qualification=false
full_training_launch_allowed=false
```

Therefore a supplemental pass beside a base-gate hold is still an overall hold.
A supplemental failure is decision-grade evidence against scaling, not a reason
to kill, restart, or modify the active trainer.

## Active-run boundary

This implementation is local only. It does not deploy into or alter the active
training revision `2c2c1f5166b73d4f28df93b276901671ac1a7836`, frozen post-eval
revision `c1efb12c6640f2d2d62ac7e9982c8804d96e7289`, or readiness revision
`5dd3488ac9b30274f4960195e252cc9fdb161002`. It can run only after the metrics
exist from a separately attested clean checkout; it performs no model load and
uses no GPU.

## Verification

- `17 passed` in the dedicated qualification tests. They cover frozen
  schema-v2 and current schema-v3 metrics, both absolute floors, both matched
  retention failures, source drift, base-gate summary drift, evaluator runtime
  drift, sampling-protocol mismatch, non-EMA/formal protocol rejection,
  failed-base-gate non-upgrade, weaker thresholds, output single-write, and
  exact resume.
- `89 passed` across the dedicated qualification plus generation-gate contract
  and report suites.
- Complete local suite: `1000 collected / 994 passed / 6 skipped` in
  `253.1s` on Windows.
- `py_compile` passed for the new module, CLI, and test module; `git diff
  --check` passed. No shell runbook changed, so this increment requires no new
  Linux `bash -n` attestation.
