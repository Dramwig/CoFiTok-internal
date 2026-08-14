# Capacity-full 300K readiness-decision waiter

Date: 2026-08-14 (Asia/Shanghai)

## Purpose and boundary

This stage prepares the exact handoff from the matched 250M step-100K capacity
result to a possible fresh 250M full-300K readiness qualification. It is a
CPU-only branch router. It neither launches the readiness benchmark nor starts
training.

The decision is buildable only when the source-replayed step-100K result selects
exactly
`build_source_compatible_250m_full_300k_readiness_decision`: the terminal screen
must be a hold containing only frozen absolute-quality failures, both matched
methods must strictly improve FID from step 50K to step 100K under the same
milestone protocol, and no milestone alert, matched regression, mechanism
failure, or class-fidelity failure may be present. Every other result branch
terminates the waiter as `not_selected` without creating a decision.

An emitted decision authorizes only the separate 250M GPU runtime/readiness
benchmark and readiness-artifact build. It keeps training, full-300K launch,
promotion, release, and formal claims false.

## Fresh-training finding

The capacity 100K pair and the dormant stability-full 300K pair have identical
method/backbone contracts but differ in these predeclared horizon-scaled fields:

- rollout-consistency warmup: `10,000 -> 60,000`;
- EMA-teacher start/warmup: `30,000/10,000 -> 180,000/60,000`;
- runtime horizon: `100,000 -> 300,000`;
- evaluation interval: `1,000 -> 2,000`;
- protected checkpoints: `[10,000] -> [50,000, 100,000, 200,000, 300,000]`;
- optimizer warmup/minimum LR/log interval:
  `1,000/1e-5/50 -> 5,000/5e-6/100`;
- config name.

No other config change is accepted. Because exact resume requires recursive
resolved-config equality, the step-100K checkpoints are qualification evidence
only and can never be resume sources for this 300K recipe. A selected full run
must start fresh at step zero, with exact resume allowed only from checkpoints
created by the target 300K configs themselves.

## Exact implementation

- Branch:
  `scale/generation-capacity-full-300k-readiness-decision-waiter-v1`
- Revision: `70b7470a6ca8f43a544cf41dd988802dc5f8c9f1`
- Tree: `bb2b9b0a1e4dc9f73b7eadaaddf6befa14340ac7`
- Bundle prerequisite: `b9f940ddf555837ae5303124af09d80c9f698ce3`
- Incremental bundle:
  `/tmp/cofitok-capacity-full-readiness-decision-70b7470.bundle`
- Bundle bytes: `20,876`
- Bundle SHA256:
  `be0bf3ebb40a55712b33972821d30402ae5a1fb39663b0ce65b6bfbe25847059`
- Isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-full-readiness-decision-70b7470/CoFiTok-internal`

The bundle advertised one exact branch and was verified against the deployed
result-waiter prerequisite. The isolated checkout matched the exact revision,
tree, and branch and had empty tracked status.

## Validation

- Local new-contract tests: `9 passed`.
- Local capacity/quality-bridge regression: `189 passed`.
- Local formal runbook/CLI entrypoint contract: `2 passed`.
- Linux new-contract tests: `9 passed`.
- Linux capacity/quality-bridge regression: `189 passed` in `512.9 s`.
- Linux formal runbook/CLI entrypoint contract: `2 passed` in `448.0 s`.
- Linux Python compile, runbook `bash -n`, and direct CLI help: passed.

The config bridge tests use the actual tracked 100K and 300K pairs. An extra
loss-field change, a non-selected result branch, a changed run path, existing
training state, or any attempt to mark the 100K checkpoint resumable fails
closed.

## Live deployment

- Waiter PID: `244959`
- Nice level: `19`
- Poll interval: `60 s`
- Initial state: `waiting`
- Initial detail: `waiting_for_source_replayed_capacity_completion_result`
- Initial `readiness_execution_allowed`: `false`
- Source result root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1`
- Reserved fresh full root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_capacity_full_300k_v1`
- Status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_readiness_decision_waiter_status.json`
- Decision target:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_readiness_decision.json`
- Deployment receipt:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_full_300k_readiness_decision_waiter_deployment_receipt.json`
- Receipt bytes: `4,319`
- Receipt SHA256:
  `ee0a42c87f44574b97c6c03fb89f8075c8c381fee0cd4a11b21dd6b298ac377b`

At deployment, neither fresh full-training run directory existed. The waiter
exports an empty `CUDA_VISIBLE_DEVICES`, uses two CPU threads and a singleton
lock, does not signal processes, and cannot invoke a training or sampling
entrypoint.

## Preserved formal checkout

- HEAD: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
- Porcelain entry count: `87`
- Porcelain SHA256:
  `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`
