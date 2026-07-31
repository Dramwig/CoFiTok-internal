# Large-capacity stability-full contract (2026-07-31)

## Scope

This change prepares the dormant ImageNet-256 `stability_full` matched pair
for a genuinely larger generation model. It does not modify the active 50K
qualification checkout, move the formal server repository, authorize 300K
training, or claim a completed CUDA feasibility result.

## Capacity tier

- CoFiTok config:
  `configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json`
- Dense config:
  `configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json`
- Shared U-Net base channels: `256`
- CoFiTok parameters: `250,153,763`
- Dense parameters: `250,135,043`
- Relative parameter gap: `+0.00748396%`

The qualified `stability_scaling` pair remains unchanged at 128 base
channels and approximately 62.8M parameters. The recipe contract requires 256
channels only for `stability_full`; legacy `scaling`/`full` and
`stability_scaling` continue to require 128.

## Runtime contract

The training runtime selector now separates the candidate grid from an explicit
baseline candidate. Existing callers retain the `16x4` default and legacy
schema-v2 selection compatibility. The large stability-full runbook binds:

```text
candidates: 1x64,2x32,4x16,8x8,16x4
baseline:   1x64
effective batch: 64
```

New schema-v3 reports store the baseline, use
`estimated_speedup_over_baseline`, and bind the baseline in the frozen
selection lock. Both matched methods must complete the baseline below the VRAM
limit before any faster candidate can be selected.

## Storage contract

The 50K reference checkpoints come from the 128-channel pair, so using their raw
bytes would under-budget the 256-channel run by roughly four times. The full
runbook therefore passes `--checkpoint-size-multiplier 4.0`. Storage report
schema v2 records:

- measured reference checkpoint bytes;
- the multiplier;
- rounded-up planned checkpoint bytes;
- reserve bytes for 16 checkpoint slots;
- sample, additional, safety, required, and headroom bytes.

The stability completion audit requires this real full-training report and
rejects a multiplier below `4.0` or inconsistent arithmetic.

## Authorization boundary

The full runbook still requires a source-bound passing stability 50K promotion
gate before runtime benchmarking or training. At the time of this change the GPU
is occupied by the active 50K CoFiTok-to-dense queue, so CUDA runtime feasibility
for the 250M pair remains pending. No full 300K run is authorized or launched.
