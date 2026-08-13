# Full-data 100K quality bridge follow-up experiment decision

Date: 2026-08-14

## Purpose

The full-data matched 100K quality bridge is queued behind an unrelated
FieldScope GPU workload. Its terminal result does not yet exist. This record
adds a deterministic, source-replaying, non-authorizing decision layer so the
next scientific experiment is selected from the completed 100K evidence rather
than improvised from a single metric.

The decision layer does not launch training, use the GPU, promote a model,
authorize the full 300K queue, or release any artifact. A separate CPU-only
waiter merely waits for the bridge result and builds the decision report.

## Evidence hierarchy

The bridge intentionally has two evaluation tiers:

| tier | samples per method | sampler | role |
|---|---:|---|---|
| 50K and 100K milestones | 2,048 | EMA DDIM-50 | trend and early-warning evidence |
| 100K terminal | 10,000 | EMA DDIM-100 | terminal quality screen |

The 2,048-sample milestones are schema-v2 `training_quality_trend_only`
reports. Their FID direction is used to ask whether both methods strictly
improve from 50K to 100K. Milestone Inception Score direction is recorded as a
sensitivity measure but cannot veto that shared FID signal because the early
warning sample budget is small. The terminal 10K report controls pass/hold and
also supplies precision, recall, mechanism checks, and class fidelity.

## Deterministic decision policy

The report first fully replays the physical terminal bridge result using the
original builder. This rehashes the bound checkpoints, sidecars, numbered PNG
sample trees, sample manifests and progress reports, full validation tree,
metric reports, class-fidelity evidence, and matched training reports. It then
independently verifies both milestone reports and checks that the 100K
milestone checkpoint identities equal the terminal checkpoint identities.

The fail-closed branches are:

| observed terminal evidence | recommended next stage |
|---|---|
| all terminal checks pass and no milestone contradiction | build a new source-compatible formal quality gate |
| only absolute quality checks fail and both methods' FID strictly improves from 50K to 100K | prepare a bounded matched 250M capacity qualification probe |
| only absolute quality checks fail without shared strict FID improvement | audit terminal distribution support, then prepare the smallest matched recipe/objective probe |
| any factorization mechanism check fails | run a matched factorization-mechanism recovery probe |
| any matched FID/precision/recall tolerance fails | run a matched factorization-quality regression probe |
| class fidelity fails | run a class-conditioning fidelity diagnostic |
| a milestone terminal alert contradicts the terminal screen | reconcile DDIM-50/2,048 and DDIM-100/10,000 evidence without retraining |
| any unclassified failure combination | extend and test the policy before executing anything |

The rejected base-128 150K continuation remains rejected: it spends 50% more
training after the main full-data coverage window without isolating capacity or
recipe limitations. Even the capacity branch prepares only a bounded 250M
qualification probe; it does not authorize full training or the 300K queue.

## Exact implementation

The isolated implementation commit is:

```text
branch: scale/generation-quality-bridge-followup-decision-v1
revision: 9b02fa83d20b1459a2706d6d82371caf5c023f54
tree: 474520aa848f3da143379a7b7f72d59a085fbe38
subject: Add non-authorizing quality bridge follow-up decision
```

It changes six files and adds 1,296 lines: the decision module, builder,
independent verifier, locked runbook, branch tests, and entrypoint contract.
The builder is fixed to the exact bridge execution revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a` and branch
`scale/generation-stability-quality-bridge-100k`.

Bundle identity:

```text
bytes: 137665447
sha256: 67839784ac4a2c2ef9e0c5723d55703bae6ba20ffc12fb03e214bf7512953861
advertised head: 9b02fa83d20b1459a2706d6d82371caf5c023f54
```

## Validation

Local Windows validation passed 13 new branch tests, 92 related regression
tests, 15 entrypoint-contract tests, Python compilation, shell static checks,
and an authorization scan.

The final CPU-only Linux rehearsal used the exact clean target checkout at
`/tmp/cofitok-quality-bridge-followup-decision-9b02fa8` with
`CUDA_VISIBLE_DEVICES=""`, `PYTHONPATH=.:src`, and two OMP/MKL threads:

```text
92 related tests passed
2 Linux entrypoint-contract tests passed
1169 tests collected
1167 tests passed, 2 skipped, 0 failed
full log bytes: 1360
full log sha256: 10e31eba00ad80db985ee8a31582eb52f2b1aeb922479b21c06f5aadbe09873c
tracked checkout remained clean
```

## CPU-only deployment

The post-result waiter is active as PID `123342` and currently reports:

```text
status: waiting
detail: waiting_for_quality_bridge_result
poll interval: 60 seconds
result exists: false
decision exists: false
```

It binds:

```text
waiter sha256: 77859b4bdcff20e79449654a1b20a42d6ad21d1c96bec35e189b01b22f01edd2
launcher sha256: d11b43d752fe5e70cddd58ced3d1698b5068efec483eae25f894554818cf9d81
decision runbook sha256: 7d03191f0312ef90e08e25609be93c1ec4ad63badb59dc26bb049c6f971df89b
```

After the terminal bridge result appears, the waiter runs exactly once from the
bound clean checkout, reconstructs and verifies the source evidence, writes
`reports/followup_experiment_decision.json`, verifies its hash and fail-closed
authorization boundary, records the recommended branch, and exits. It neither
loads a model nor executes the recommendation.

## Live operational boundary

At `2026-08-14T05:14:48+08:00`, bridge execution had not launched. The bridge
idle waiter remained PID `927790` with `waiting_for_gpu_idle`, and the bounded
recovery supervisor remained PID `110237` with `observing`. The only GPU
process was unrelated FieldScope PID `910099`, using approximately 2,256 MiB;
it was not signaled or modified. Free space under `/root/autodl-tmp` was
223,185,661,952 bytes versus 103,826,920,100 required by the bridge preflight.

The formal remote checkout remained exactly:

```text
branch: scale/generative-system
revision: 1ebcc15210e63a776a2ba448481cbd8bb94a4066
tree: 659fa94726c4aec0afef49904f82b828bb62872b
tracked porcelain: empty
```

## Authorization boundary

Every generated recommendation records:

```text
recommended_stage_execution_allowed=false
quality_bridge_execution_allowed=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
report_is_promotion_gate=false
release_authorization_allowed=false
new_source_compatible_gate_required=true
```

The machine-readable receipt is:

```text
artifacts/reports/generation/stability_full_data_quality_bridge_followup_decision_2026-08-14/rehearsal_and_deployment_receipt.json
```
