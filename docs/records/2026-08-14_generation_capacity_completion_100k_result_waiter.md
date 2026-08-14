# Capacity-completion 100K source-replayed result waiter

Date: 2026-08-14 (Asia/Shanghai)

## Scope

This change installs one CPU-only waiter for the matched base-256 capacity
completion stage. It does not train, sample, signal another process, authorize
full 300K training, promote a model, or release an artifact. It waits for the
existing capacity-completion execution status to become exactly
`completed / complete`, physically replays the terminal evidence, and emits a
non-authorizing result plus a fail-closed next-stage recommendation.

The waiter preserves the standing experiment authorization while retaining its
recorded safety boundaries: exact source/revision/output binding, independent
clean checkout, no changes to the formal checkout, no locked-evidence
overwrite, no cross-project process modification, and no conversion of a
non-authorizing protocol into an authorization.

## Exact implementation identity

- Branch: `scale/generation-capacity-completion-100k-result-waiter-v1`
- Revision: `b9f940ddf555837ae5303124af09d80c9f698ce3`
- Tree: `db3760b14b87dec146f4779403de7bd7064ee9bb`
- Prerequisite revision: `892c1d63db872f1135d41adaf0a25ee7388999c7`
- Incremental bundle: `/tmp/cofitok-capacity-completion-result-b9f940d.bundle`
- Bundle bytes: `23,405`
- Bundle SHA256: `5f7ce4eb3dd4747402093c1521a4697d9bb1f9f8e11c83aa138da07b291c955c`
- Isolated remote checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-completion-result-b9f940d/CoFiTok-internal`

The bundle advertised only the exact result-waiter branch and was verified in a
checkout containing the prerequisite revision before the isolated checkout was
created. The isolated checkout was clean and matched the revision, tree, and
branch above.

## Result contract

The result builder binds and physically replays 21 exact sources:

- frozen quality-bridge and capacity-probe preparations;
- capacity-probe, 50K scaling, and 100K completion decisions/results;
- completion launch receipt, execution status, and immutable source-checkpoint
  archive;
- both exact-resume training validations;
- matched step-50K and step-100K 2,048-sample milestones;
- both terminal sampling preflights, 10,000-sample generation reports,
  checkpoint-mechanism evaluations, and class-fidelity reports;
- the matched class-fidelity qualification.

The replay additionally reconstructs the hard-link checkpoint archive without
creating files, recomputes both completion training validations from the actual
reports/configs/checkpoints, verifies physical sample and metric evidence,
re-verifies both milestone source trees, and requires exact terminal checkpoint,
sample-set, real-set, evaluator, runtime, sampling, and class-fidelity bindings.

All recommendation branches remain non-authorizing. A terminal pass recommends
construction of a new source-compatible formal quality gate. If and only if the
remaining failures are absolute quality floors and both matched methods strictly
improve FID from step 50K to step 100K under the same milestone protocol, the
result recommends construction of a separate 250M full-300K readiness decision.
It does not authorize that training. Milestone contradictions, mechanism
failures, matched regressions, class-fidelity failures, non-scale-responsive
quality holds, and unclassified combinations each route to bounded diagnostic or
policy work.

## Validation

- Local targeted result/waiter/runbook tests: `17 passed` after the final
  standing-authorization replay helper.
- Local capacity/quality-bridge regression before that final helper:
  `179 passed`.
- Local formal runbook/CLI entrypoint contract: `2 passed`.
- Linux isolated-checkout capacity/quality-bridge set: `180 passed` under a
  `set -e` validation pipeline.
- Linux formal runbook/CLI entrypoint contract: `2 passed` in `420.7 s`.
- Linux `py_compile`: passed for the result core, builder, verifier, and waiter.
- Linux `bash -n`: passed for all `generation_capacity_*.sh` runbooks.
- Direct `--help` validation: passed for the builder, verifier, and waiter.

## Live deployment

- Waiter PID: `234309`
- Nice level: `19`
- Poll interval: `60 s`
- Status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_result_waiter_status.json`
- Result target:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_result.json`
- PID pointer:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_result_waiter.pid`
- Log:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_result_waiter.log`
- Immutable deployment receipt:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_result_waiter_deployment_receipt.json`
- Receipt bytes: `3,713`
- Receipt SHA256:
  `2d41f9bbd67e89c280b0486d4cd1964ba28f10f39883ff70a22e6c5498812a1e`

The initial live state was `waiting` with detail
`waiting_for_completed_capacity_completion_100k_execution`. The waiter had no
result because the upstream capacity-probe/scaling/completion chain had not yet
run. At the deployment reread, FieldScope PID `910099` was the only GPU process
and used approximately 2,256 MiB; no CoFiTok GPU child existed.

The waiter exports `CUDA_VISIBLE_DEVICES=""`, uses two CPU threads, holds a
non-blocking singleton lock, refuses a dirty or mismatched checkout, validates
the standing-authorization SHA and complete record, and refuses to overwrite a
non-byte-equivalent existing result. A failed/cancelled/error completion status
causes a fail-closed waiter failure rather than a restart or additional launch.

## Preserved formal checkout

The formal checkout remained untouched both before and after deployment:

- HEAD: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
- Porcelain entry count: `87`
- Porcelain SHA256:
  `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`
