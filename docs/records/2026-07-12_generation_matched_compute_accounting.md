# Matched generation compute accounting (2026-07-12)

## Direct baseline contract

The direct `dense_identity` baseline shares CoFiTok's ImageNet-256 data,
preprocessing, class conditioning, diffusion schedule, scalable U-Net trunk,
optimizer, effective batch, training steps, EMA policy, checkpoint cadence, and
formal sampling/evaluator protocol. The U-Net trunk runs once per diffusion
evaluation for both methods. CoFiTok adds eight lightweight token heads,
feedback projections, and restricted synthesis operators; dense uses one
three-channel identity head.

The full 300K config-pair preflight is stored at
`artifacts/reports/generation/preflight_2026-07-12/config_pair_full_300k.json`:

```text
CoFiTok parameters: 62,950,800
dense parameters:   62,824,707
relative gap:       +0.200706%
allowed gap:        2.0%
status:             pass
```

Official D-AR, MAR, and ReTok rows remain pretrained contextual evidence with a
different evaluator and unknown/unmatched training compute. They are never
ranked numerically in the direct matched panel.

## Accumulated training cost

Segmented full training now carries `cumulative_elapsed_seconds` and
`cumulative_peak_vram_bytes` in checkpoint `extra_state`. A resumed segment
uses these values when producing its metrics and final training report. This
prevents each 50K/100K/200K/300K segment from presenting only its last segment
as the total cost.

`cofitok.generation_cost.training_cost_summary` is the shared gate/report
definition. It derives and validates:

- micro-batch and gradient accumulation;
- effective batch size;
- expected and observed images seen;
- accumulated training seconds and images/second;
- accumulated peak CUDA memory.

The promotion and final gates require exact images-seen accounting and finite
positive elapsed time; CUDA runs also require positive peak VRAM. The final
matched table exposes effective batch, training images, hours, throughput, and
VRAM alongside parameters, steps, sample count, FID, IS, precision, and recall.

## Verification

Gate tests reject an off-by-one images-seen report. Final-comparison tests bind
the shared cost calculation into JSON, Markdown, and CSV output. Exact-resume
tests continue to require model/EMA/optimizer/scheduler/RNG/sampler trajectory
identity while treating wall-clock accounting as observational metadata.
