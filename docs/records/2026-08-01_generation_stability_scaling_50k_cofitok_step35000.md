# Stability-scaling 50K CoFiTok step-35K audit

Date: 2026-08-01 CST

## Scope

This is a read-only recovery and schedule audit of the active CoFiTok member in
the matched stability-scaling 50K queue. It does not evaluate free-rollout image
quality, compare against the not-yet-trained dense member, authorize post-eval,
or authorize full 300K training.

Training identity remains:

```text
revision: 2c2c1f5166b73d4f28df93b276901671ac1a7836
branch: scale/generation-stability-50k-preflight
config: configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json
config sha256: 4b9bf0d89f639983df6f5c046deb36ad6bd65dac8ec8f9b587721dbcef688f02
```

The audit was run from the clean post-eval checkout at
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289` on
`scale/generation-stability-50k-posteval-v4`, using
`audit_generation_training_progress.py` with required checkpoint integrity.

## Recovery boundary

The tracked audit returned `healthy`, with empty issues and warnings. The
rolling checkpoint set is exactly `25K/30K/35K`; no required step is missing.
The 35K checkpoint identity is:

```text
filename: checkpoint_step_00035000.pt
bytes: 1006321770
sha256: 98ce7eac9ef77da24dda42a96d1e17a5a047f60665fb30c10ed286a98c807f10
integrity: verified
git revision: 2c2c1f5166b73d4f28df93b276901671ac1a7836
```

No checkpoint payload was copied locally. `latest.json`, the sidecar, file stat,
dataset identity, runtime environment identity, step, bytes, and declared
SHA256 all agree.

## Consistency schedules

At audit time the canonical metrics contained 716 rows through step 35,750.
All rows passed rollout-consistency schedule validation. EMA-teacher consistency
had 115 active rows after step 30K; every active row had finite nonzero loss.

```text
rollout scale at 35,750: expected 1.0 / observed 1.0
EMA-teacher scale at 35,750: expected 0.575 / observed 0.575
validation events: 35 expected / 35 present
validation provenance: complete
recent mean epsilon: 0.0276736125
recent mean total: 0.0489834789
recent mean pre-clip grad norm: 0.2871264953
```

The observed maximum logged gradient norm was `14.6480`, with 98 pre-clip
exceedances; this is expected because the field records the total norm before
the configured `1.0` clipping operation. No non-finite metric or schedule
mismatch was found in the independent 30K–35.75K scan.

Validation events use successive deterministic image batches. Their values must
not be interpreted as a same-batch learning curve. They show finite execution
and complete provenance, not a quality improvement claim. Final judgment still
requires the matched dense trajectory and formal free-state sampling.

## Automatic handoff

The post-eval waiter remains alive with:

```text
status: waiting
detail: waiting_for_completed_training_pair
child_pid: null
training revision/branch: 2c2c1f5 / scale/generation-stability-50k-preflight
evaluation revision/branch: c1efb12 / scale/generation-stability-50k-posteval-v4
formal_300k_allowed: false
```

Both source checkouts are tracked-clean. The waiter passes the exact evaluation
revision and branch into the post-eval runbook, whose SHA256 is
`6cf80146c1696aa0bed9a4a370b41c42aa5be9dc03f753c4c81ce6b54278b064`.
At audit time `pair_summary.json`, `promotion_gate.json`, and
`full_training_readiness.json` were correctly absent. The 7-day post-eval
waiter had elapsed about 80,885 seconds, leaving substantial timeout margin.

The latest pair monitor remained `running/cofitok_training/issues=[]`; dense was
still at zero. The readiness waiter remained
`waiting_for_passing_stability_gate`, with no child and
`full_training_launch_allowed=false`.

## Evidence

The authoritative progress report SHA256 is
`3fcfb04a79cf4eeff1310b3e0627ed7066b81e2335ca21db1c1298d1611f39da`.
Small source snapshots and their manifest are stored in
`artifacts/reports/generation/stability_scaling_50k_2026-08-01/cofitok_step_00035000/`.
