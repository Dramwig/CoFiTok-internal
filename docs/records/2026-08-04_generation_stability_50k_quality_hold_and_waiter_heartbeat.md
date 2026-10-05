# Stability 50K quality hold and waiter heartbeat hardening

Date: 2026-08-04

## Outcome

The matched ImageNet-256 10% stability pair completed exactly at 50,000 steps
and 3,200,000 images per method, and the frozen formal EMA post-evaluation
completed successfully. The scaling decision remains a scientific hold:

- CoFiTok FID: `138.29702495267782`
- dense-identity FID: `151.4476773464495`
- CoFiTok relative FID improvement: about `8.68%`
- required absolute scaling FID: `<= 100.0`
- `full_training_launch_allowed=false`

The hold is therefore not a training-integrity, matched-protocol, checkpoint,
or sampling-provenance failure. It is an absolute sample-quality failure, and
the full matched 300K run remains forbidden.

Authoritative remote root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher
```

Training identity:

```text
revision 2c2c1f5166b73d4f28df93b276901671ac1a7836
branch   scale/generation-stability-50k-preflight
```

Frozen evaluation identity:

```text
revision c1efb12c6640f2d2d62ac7e9982c8804d96e7289
branch   scale/generation-stability-50k-posteval-v4
```

## Source-bound 50K training trajectory

The exact remote `train_metrics.jsonl` and `run_manifest.json` files for both
methods were replayed locally through
`scripts/build_generation_matched_training_trajectory.py` at cutoff 50,000.
The schema-v3 report passed all finite-metric, strict-step, sample-accounting,
fixed-validation, pair-contract, Git/runtime/dataset, and CoFiTok resume-boundary
checks.

Key fixed-validation results over 50 paired events:

- CoFiTok mean epsilon MSE: `0.02960815742611885`
- dense mean epsilon MSE: `0.029620671570301058`
- relative delta of means: `-0.042248%`
- lower-MSE events: CoFiTok `18`, dense `32`
- step-50K endpoint relative delta: `+0.081361%`

During the exact EMA-teacher full-scale interval, steps 40K--50K:

- CoFiTok mean epsilon MSE: `0.028556561436165463`
- dense mean epsilon MSE: `0.028533021665432236`
- relative delta of means: `+0.082500%`

The training-space endpoint is essentially matched and does not explain the
large absolute FID gap. Conversely, CoFiTok's better FID is not predicted by a
better final fixed-timestep epsilon MSE. This reinforces the existing boundary:
fixed-validation epsilon is a trajectory diagnostic, not a sample-quality gate.

## Visual diagnosis

The frozen deterministic visual audit was inspected from:

```text
reports/visual_audit/cofitok_fixed_samples.png
reports/visual_audit/dense_fixed_samples.png
reports/visual_audit/cofitok_prefix_paths.png
```

Under the shared sample-index random stream, CoFiTok and dense exhibit nearly
the same coarse layouts, class-color tendencies, and high-frequency/noisy
artifacts. CoFiTok prefix budgets remain visibly ordered, but the budget-8
endpoint is still under-trained. Together with recall near `0.009` for both
methods, this points to a shared capacity/data-coverage/training-budget problem,
not a CoFiTok-only factorization collapse. This is a qualitative diagnosis and
does not replace the formal metrics.

## Waiter heartbeat incident

The frozen post-evaluation waiter eventually passed, but the already launched
supplemental waiter had failed earlier with:

```text
RuntimeError: stability 50K post-evaluation status is stale
```

Root cause: the post-evaluation waiter wrote `running` once and then blocked in
`child.wait()` for the multi-hour sampling/evaluation child. The supplemental
waiter correctly required a status update within 900 seconds, so it interpreted
the healthy but silent post-evaluation child as stale. The readiness and
supplemental waiters used the same blocking pattern for their own long GPU
children.

The implementation now uses
`cofitok.process_monitoring.wait_for_child_with_heartbeat` in all three waiters.
It waits with a bounded timeout and atomically republishes the same source-bound
`running` contract after every timeout. It does not signal, restart, replace, or
otherwise control the child process.

Changed entrypoints:

```text
scripts/run_generation_stability_50k_posteval_waiter.py
scripts/run_generation_stability_full_readiness_waiter.py
scripts/run_generation_stability_frozen_supplemental_waiter.py
```

The change is prospective only. The existing remote waiter was not restarted or
replaced, the frozen reports were not edited, and no new GPU stage or full 300K
authorization was created.

Local verification:

```text
targeted waiter/process-monitoring tests: 28 passed
full repository suite: 1061 passed, 6 skipped
```

## Next quality-recovery boundary

The next GPU action must remain non-authorizing and must wait for an idle GPU and
explicit execution authority. The least expensive useful discriminator is a
source-bound matched inference sweep on the frozen 50K checkpoints, preserving
EMA, DDIM-100, balanced-modulo classes, batch-invariant per-index RNG, and equal
CoFiTok/dense cases while varying only predeclared CFG/guidance-rescale values.
Small-sample rows are configuration diagnostics only; any selected protocol
would require a fresh matched 10K confirmation and cannot rewrite this frozen
gate.

If that bounded sweep cannot plausibly close the absolute FID gap, the next
training experiment should be a separately named, matched larger-capacity/full-
data bridge with its own output root and gate. It must not resume into or
overwrite this 50K evidence, and it must precede any reconsideration of full
matched 300K training.
