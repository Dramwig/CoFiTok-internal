# Generation training-budget claim boundary (2026-08-03)

## Problem

The CoFiTok and dense-identity direct pair shares the dataset, resolution,
scalable U-Net contract, optimizer schedule, effective batch, optimizer steps,
training images, EMA policy, and formal sampling/evaluator protocol. CoFiTok
still contains method-defining token heads, restricted synthesis operators, and
factorization-only auxiliary losses. Equal steps and images therefore do not by
themselves establish equal wall-clock, GPU-hours, or FLOPs.

The existing comparison already publishes source-derived training time,
images/second, and peak VRAM, and GPU-contention evidence decides whether those
raw measurements support an efficiency ranking. The missing invariant was an
explicit, machine-auditable prohibition on relabeling the direct quality panel
as generic compute-matched evidence.

## Controlled evidence

The selected stability runtime benchmark is stored at:

```text
artifacts/reports/generation/stability_probe_2026-07-29/runtime_benchmark_rollout_x0_u2_ema_teacher_v2_2026-07-30/benchmark_summary.json
SHA256 e5a88a1e7d18bef30de56ee43b46e94447c76302a84c0aa09d0928495fbaab49
```

Under its matched runtime recipe, CoFiTok measured `22.124531` images/s and
dense measured `24.307071` images/s. CoFiTok therefore used about `1.09865x`
the optimizer-step time in this controlled benchmark. Peak allocated VRAM was
`15,232,468,992` versus `15,036,313,600` bytes. These are measured method costs,
not equalized budget constraints.

The active stability dense recovery also experienced an unrelated FieldScope
GPU process. Its raw common-prefix wall-clock can consequently reverse the
controlled benchmark ordering and must remain observational-only. No process
was modified to remove that contention.

## Schema v7 decision

`scripts/build_large_scale_generation_comparison.py` now emits:

- budget basis `matched_steps_and_training_images`;
- exact matched dataset, resolution, effective batch, steps, and images;
- `equal_wall_clock_budget=false`;
- `equal_gpu_hours_budget=false`;
- `equal_training_flops_budget=false`;
- `compute_matched_claim_allowed=false`;
- time, throughput, and peak VRAM as `measured_outcomes`;
- cost-efficiency ranking permission only when the bound GPU-contention report
  proves continuous exclusive coverage.

The completion audit recomputes this policy and rejects an equal-compute
overclaim, a changed matched budget axis, or a row that omits the claim
boundary. This change does not alter any training, checkpoint, post-evaluation,
or full-300K authorization state.
