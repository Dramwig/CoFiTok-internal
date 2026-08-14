# Capacity-scaling 50K result and bounded 100K decision waiter

Date: 2026-08-14

## Purpose

The matched 250M capacity-probe chain already has a source-bound execution
supervisor for the exact step-10K to step-50K segment. This change closes the
next CPU-only evidence gap without starting another GPU job: after that segment
finishes, an immutable waiter physically replays every source, emits the
non-authorizing step-50K result, and creates a new source-compatible decision
for at most the configured step-50K to step-100K completion segment.

The waiter cannot use a GPU, launch training, signal another process, authorize
full 300K, promote a model, or authorize release.

## Result policy

The source-replayed step-50K result binds:

- the supported four-arm step-10K capacity result and its decision;
- the exact step-10K to step-50K launch receipt and completed execution status;
- both physically replayed partial-training validations and exact step-50K
  checkpoint/integrity identities;
- the matched 2,048-sample EMA DDIM-50 milestone and its physical sources;
- the 256-image CoFiTok mechanism report and coarse-token energy partition.

Configured completion is supported only when both matched methods strictly
improve FID from 10K to 50K, the step-50K matched quality screen has no alerts,
and CoFiTok retains zero-token exactness, ordered rank 1 over at least six
orders, shuffle mismatch, and qualified coarse-token utilization. The result
itself always records `additional_training_allowed=false` and requires a new
source-compatible decision.

## Bounded completion decision

When and only when the result supports completion, the decision authorizes the
same base-256 CoFiTok/dense pair to resume from their exact step-50K checkpoints
and stop at the already configured step-100K horizon. Fresh training is
forbidden. Terminal evidence is fixed to:

- 10,000 samples per method under one matched random stream;
- EMA, bf16, DDIM-100, CFG 1.5, guidance rescale 0;
- FID, Inception Score, precision, recall, and class fidelity;
- a 256-image CoFiTok mechanism audit at timestep 500;
- a matched 2,048-sample DDIM-50 step-100K trend milestone.

The decision still fixes `full_300k_launch_allowed=false`,
`formal_generation_claim_allowed=false`, and
`release_authorization_allowed=false`. A failed quality or mechanism condition
produces a terminal hold instead of an execution authorization.

## Exact implementation identities

Source-replayed result commit:

```text
c506eb0  Add source-replayed capacity scaling result
```

Waiter deployment target:

```text
revision: 7c3a1b8a06fa4dab8733c808b284fbe86cfeaccb
tree: 02e4ae18d25b1664fa4a98650714a0b4ebf30def
remote branch: scale/generation-capacity-completion-decision-waiter-v1
checkout: /root/autodl-tmp/CoFiTok/checkouts/capacity-completion-waiter-7c3a1b8/CoFiTok-internal
```

Prerequisite-aware bundle:

```text
path: /tmp/cofitok-capacity-completion-7c3a1b8.bundle
bytes: 48,035,245
sha256: 4f99cc8da843b30489dadf7c8391d7cc736782e3238c85407946e76fe8ba5d27
advertised head: 7c3a1b8a06fa4dab8733c808b284fbe86cfeaccb
prerequisites:
  1ebcc15210e63a776a2ba448481cbd8bb94a4066
  58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

The bundle was verified locally and against the formal server repository before
the independent checkout was created.

## Validation

Windows project environment:

```text
capacity-family regression: 110 passed
new result/decision/waiter/runbook tests: 19 passed
Python compile: passed
git diff --check: passed
```

Linux isolated checkout with `CUDA_VISIBLE_DEVICES=""`, two CPU threads, and
the project `PYTHONPATH`:

```text
new result/decision/waiter/runbook tests: 19 passed
seven new Python entrypoints/modules: py_compile passed
new waiter runbook: bash -n passed
commit diff-tree check: passed
tracked status count: 0
```

An attempted broader Linux capacity-family run exceeded the bounded client
timeout before producing a test result; it left no pytest process and no
worktree change. The complete capacity-family set had already passed locally,
and all newly changed Linux tests passed separately.

## Live waiter

The waiter was started once under its own `flock`-protected runbook:

```text
PID: 211714
nice: 19
status: waiting
detail: waiting_for_completed_capacity_scaling_50k_execution
status path: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_decision_waiter_status.json
log path: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_decision_waiter.log
runbook sha256: b1958fb9369f62d5136457e306d41107d064c26fe5cf205f233b4dd4a4841a70
```

Its initial status receipt was 1,184 bytes with SHA256
`6220be807b6aa36dcb66d5665aabad1134b9303fbe4a096ea6455b06c1b5b3a6`;
the status is expected to change as the timestamp advances.

At launch, FieldScope PID `910099` remained the only GPU process at
approximately 2,256 MiB and was not signaled or modified. The upstream
base-128 recovery, capacity-probe, capacity-scaling decision, and step-50K
supervisors remained alive. The formal checkout remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`, with porcelain count `87` and
porcelain SHA256
`a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.
The data filesystem had approximately 202 GB free.

## Automatic next behavior

The waiter will remain CPU-only until the exact step-50K execution status is
`completed/complete`. It then calculates fresh SHA256 identities, physically
rebuilds the result from the raw training, milestone, sampling, and mechanism
sources, requires an existing artifact to be byte-equivalent if present, and
emits the bounded step-100K decision. It exits after the decision; it does not
launch the authorized segment itself.
