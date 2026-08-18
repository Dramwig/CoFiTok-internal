# Conditioning-ranking four-arm 5K sampling supervisor launch

Date: 2026-08-19

## Scope

This record binds the CPU-only waiter for the separately source-bound four-arm
generated-sample validation:

```text
stage: conditioning_ranking_four_arm_sampling5k_v1
output root:
/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_sampling5k_v1
```

The waiter does not authorize training, checkpoint promotion, release, full
100K/300K execution, or a CoFiTok-specific advantage claim.  It exits without
GPU work for an asymmetric or failed 1K postevaluation and can reserve the
sampling runbook only once after exactly five consecutive idle-GPU polls.

## Exact implementation identity

```text
branch: scale/generation-label-ranking-standing-authorization-v1
revision: f77e311546beba697020debd35e26888096ca258
tree: 02a4ec60e20930beebde75b2bc8c4a5cf6b2c684
subject: Fix ranking sampling classifier path
parent implementation commit: 1a020e4c774847119fdf64ebf93ea33cb2e97d05
```

The final commit corrects the physical classifier location to:

```text
/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth
```

## Incremental deployment bundle

The bundle advertises exactly the target revision and requires the already
present probe revision `64f85fe3a34aebd664c083d43bfb38d1c61aaeb7`.

```text
local: D:/cofitok-label-ranking-sampling-f77e311.bundle
remote: /tmp/cofitok-label-ranking-sampling-f77e311.bundle
bytes: 49,942
SHA256: f021fa2670913fa0b2fcfaa84856429e0c04db58c79f161bd14dd06a9f9ab919
advertised ref: f77e311546beba697020debd35e26888096ca258 HEAD
```

`git bundle verify` passed before deployment.  The formal server checkout was
not fetched, merged, checked out, or modified.

## Linux rehearsal

The exact target was fast-forwarded only inside the isolated rehearsal layout:

```text
/tmp/cofitok-label-ranking-sampling-rehearsal-layout-1a020e4/CoFiTok-internal
```

The sibling `paper/` directory was copied into the rehearsal parent because
four repository tests intentionally resolve paper files relative to the parent
of `CoFiTok-internal`.

Validation used:

```text
CUDA_VISIBLE_DEVICES=-1
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=.:src
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Results on exact revision `f77e311`:

```text
new runbook bash -n: pass
new supervisor tests: 8 passed
full repository suite: 1,153 collected; 1,151 passed; 2 skipped
tracked and full porcelain after tests: empty
```

An earlier full-suite invocation from a clone placed directly under `/tmp`
reported four missing-paper failures because it resolved `/tmp/paper`.  The
correct project-parent layout above removed all four environment-only failures.

## Launched CPU waiter

Dedicated exact checkout:

```text
/tmp/cofitok-label-ranking-sampling-standing-auth-f77e311
```

Control root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/
conditioning_ranking_four_arm_sampling5k_standing_auth_v1/
f77e311546beba697020debd35e26888096ca258
```

Process and initial status:

```text
PID: 245918
role: generation_conditioning_ranking_sampling_validation_supervisor
status: waiting
detail: waiting_for_conditioning_ranking_1k_postevaluation
child_pid: null
idle_gpu_polls: 0
```

Authoritative files:

```text
supervisor_status.json
supervisor.pid.json
supervisor.log
```

The source postevaluation is expected at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
conditioning_ranking_four_arm_probe1k_v1/reports/
conditioning_ranking_posteval_v1/postevaluation.json
```

The standing authorization remains bound by SHA256:

```text
5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
```

At launch, the 5K output root and its lock were both absent.  No preparation,
idle evidence, launch receipt, preflight, or generated sample had been created.

## Active bridge non-interference check

Immediately after waiter launch:

```text
active GPU PID: 79894
GPU memory: 77,970 MiB
GPU utilization: 100%
dense full-data step: 12,050 / 100,000
CoFiTok milestone: 50,000
```

The active quality-bridge trainer continued normally.  The new waiter has
`CUDA_VISIBLE_DEVICES=-1`, is sleeping on the absent 1K postevaluation, and did
not create the 5K output root or signal any process.
