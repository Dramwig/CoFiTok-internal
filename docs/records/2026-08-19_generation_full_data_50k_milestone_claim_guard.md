# Full-data 50K milestone claim guard

Date: 2026-08-19

## Outcome

A CPU-only, source-replayed claim guard now waits for the exact matched 50K
quality-bridge milestone. It will publish descriptive FID, Inception Score,
endpoint-MSE, and CoFiTok ordering diagnostics after the active runbook creates
the paired report. It cannot authorize a formal generation-quality claim,
promotion, release, sampling, later training, or full 300K.

The existing active milestone builder was already source-bound and
non-claiming. The added guard closes the remaining interpretation gap by:

1. rehashing all four bound source reports;
2. rebuilding the complete milestone payload from those sources and requiring
   exact payload equality;
3. requiring the exact locked training revision and branch in every source;
4. publishing an explicit machine-readable claim boundary; and
5. requiring the matched 100K, 10,000-sample DDIM-100 uncertainty result and
   class-fidelity qualification before any generation-advantage conclusion.

## Code identity

- branch:
  `analysis/generation-full-data-50k-milestone-claim-guard-v1`
- code revision:
  `ab1f02428d679daf5b751e19492303967f280404`
- code tree:
  `0f000bd0685da14e251e0420b7a0a6188aa01259`
- waiter source SHA256:
  `e111ee6027e5d1930569af0d695d5b588246d7255b585ed7812060e620f3034d`
- exact training revision:
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- exact training branch:
  `scale/generation-stability-quality-bridge-100k`

The incremental bundle requires the exact training revision, advertises only
the code revision above, and has:

- bytes: `128,476`
- SHA256:
  `f50706eeccbc57843153cf3c495bee79ab5b0a419305e835d4349eff1a33d248`

## Deployment

Remote checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-50k-claim-guard-ab1f024/CoFiTok-internal
```

Active evidence root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/milestones/step_00050000_claim_guard_v1
```

Authoritative files:

```text
deployment_receipt.json
waiter_status.json
waiter.pid.json
claim_guard.json  # appears only after the exact paired milestone is valid
```

Deployment receipt SHA256:

```text
f0200a6d4c498dc168eaff42ffc5abe8f348625c6b739e9aff77d2aa77036614
```

The real-path `--once` check returned `waiting` with detail
`matched_50k_milestone_report_missing`, published no claim guard, left no
process, and did not change the GPU process set. The long-running waiter then
started as PID `718498` with `CUDA_VISIBLE_DEVICES=-1`. A later observation
verified the 60-second status heartbeat. A competing exact invocation exited
nonzero because PID `718498` held the lifetime `flock`; the original waiter and
status remained intact.

## Validation

Local and Linux targeted suites each passed 30 tests covering:

- the new claim guard and its waiting/replay behavior;
- the existing source-bound milestone builder;
- the terminal quality-bridge result builder; and
- the exact foreground 50K transition order.

Negative tests prove rejection of:

- a changed bound source report;
- milestone metrics that no longer replay from their sources;
- a source report from another training revision; and
- an existing claim-guard output whose content differs.

The waiter also passed Python compilation, `git diff --check`, tracked-clean
verification, bundle prerequisite verification, and transferred-bundle SHA256
verification. Linux validation hid CUDA and limited OMP/MKL to one thread.

## Claim boundary

The output always records:

- `early_warning_only=true`;
- `formal_generation_quality_claim_allowed=false`;
- `broad_generation_superiority_claim_allowed=false`;
- `promotion_authorization_allowed=false`;
- `release_authorization_allowed=false`;
- `sampling_launch_allowed=false`;
- `training_launch_allowed=false`;
- `full_training_launch_allowed=false`;
- `full_300k_launch_allowed=false`;
- all process-signal permissions as false;
- `cross_tier_numeric_ranking_allowed=false`; and
- `requires_terminal_100k_10000_sample_uncertainty=true`.

This artifact is intentionally descriptive. A favorable 2,048-sample DDIM-50
direction at 50K is not evidence for a formal or broad generation-quality
advantage.

## Live state at deployment

The active bridge remained in dense training. The latest observed row was step
`37,600` (`2,406,400` images), EMA-teacher scale `0.76`. The sole GPU process
remained dense trainer PID `79894`, using `77,970 MiB`; the claim guard uses no
GPU and sends no process signals.
