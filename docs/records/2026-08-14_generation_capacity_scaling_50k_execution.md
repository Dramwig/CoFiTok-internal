# Source-bound matched 250M capacity scaling to step 50K

Date: 2026-08-14

## Purpose

The bounded base-256 capacity probe stops both matched methods at step 10,000.
When its independently replayed result supports capacity scaling, the next
permitted experiment is an exact continuation of those same trajectories to an
intentional step-50,000 stop. This change implements that execution path and a
CPU-only supervisor without broadening the experiment boundary.

## Scope

The chain permits only:

- the existing base-256 CoFiTok and dense-identity runs;
- exact resume from the source-bound step-10,000 checkpoints;
- recovery resumes within that same 10K-to-50K segment;
- an intentional stop at step 50,000 under the unchanged configured 100K
  schedule;
- EMA, DDIM-50, CFG 1.5 evaluation with 2,048 samples per method;
- the 256-image CoFiTok checkpoint mechanism evaluation used by the existing
  milestone runbook.

It does not authorize fresh training, configured completion to step 100,000,
full 300K training, promotion, formal claims, release, or changes to frozen
evidence.

## Source and launch replay

The immutable launch receipt binds and replays:

- the source-compatible capacity-scaling decision;
- the supported four-arm capacity-probe result;
- the original capacity-probe launch receipt;
- both exact configuration files and Git identities;
- both physical step-10,000 checkpoints and integrity manifests;
- the execution and training checkouts, storage preflight, output root, runtime
  selection, idle GPU observation, and absence of duplicate work.

The step-10,000 checkpoint identities are replayed again after both training
and evaluation arms complete. Per-arm partial-training validation checks the
canonical metrics history, all 50 scheduled validation events, 3.2 million
images seen, cumulative time and peak VRAM, runtime and dataset identities,
source and target checkpoint integrity, and the absence of checkpoints after
step 50,000.

## Recovery supervision

The supervisor waits for the exact decision waiter and requires five
consecutive idle-GPU polls. It observes any already-running matching execution
instead of creating a duplicate. During an owned child attempt it tracks the
training metrics, checkpoints, sampling progress, checkpoint-evaluation
reports, and metric reports. Thirty minutes without source progress is treated
as a stall.

Only the independent process group created by this supervisor can receive
SIGTERM/SIGKILL. Unrelated GPU processes are never signalled. A recoverable
training or evaluation interruption is retried at most four times, and a final
validation accepts an exact intermediate checkpoint resume only inside the
original 10K-to-50K segment.

## Files

- `src/cofitok/generation/capacity_scaling_execution.py`
- `src/cofitok/generation/capacity_scaling_training.py`
- `scripts/build_generation_capacity_scaling_launch_receipt.py`
- `scripts/verify_generation_capacity_scaling_launch_receipt.py`
- `scripts/validate_generation_capacity_scaling_training.py`
- `scripts/verify_generation_capacity_scaling_training.py`
- `scripts/run_generation_capacity_scaling_50k_supervisor.py`
- `artifacts/runbooks/generation_capacity_scaling_250m_50k_execute.sh`
- `artifacts/runbooks/generation_capacity_scaling_250m_50k_supervisor.sh`

## Validation and isolated deployment

- implementation revision: `9c5848e6a3e2f7ba10e9527b4ffc5e30bb4d56dc`
- implementation tree: `979fe91f2eef90a315c0a1426161ad7e818d4778`
- deployment branch: `scale/generation-capacity-scaling-50k-execution-v1`
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-scaling-50k-9c5848e/CoFiTok-internal`
- prerequisite revision: `c18a606953e34725f0bae6f6f568db3ebf7c42c6`
- incremental bundle: 33,881 bytes, SHA256
  `ab6f6fbc0f0827eb54b532174c734b617a16a4e20fdaa24653258ff279102f56`
- local targeted regression: 104 passed
- local compile and diff checks: passed
- Linux runbook `bash -n`: 2/2 passed
- Linux source-bound core regression: 47 passed
- Linux runbook entrypoint contract regression: 2 passed
- active CPU-only supervisor PID at deployment: `204941`, nice level 19
- supervisor state at deployment:
  `waiting_for_source_replayed_capacity_scaling_decision`
- immutable deployment receipt: 6,608 bytes, SHA256
  `545dd04144142cf2afd5884d47f9b094e0cd732e2d2c0e0d7b3a81858e25a429`

The formal remote checkout was not fetched, checked out, or modified. At
deployment validation time the only GPU process belonged to FieldScope, so no
CoFiTok GPU work was launched or duplicated.
