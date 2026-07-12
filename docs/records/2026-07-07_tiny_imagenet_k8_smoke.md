# 2026-07-07 Tiny ImageNet K8 Mechanism Smoke

Purpose: test whether the CIFAR-10 K8 tail-stage mechanism transfers at all to
Tiny ImageNet 64x64. This is a small diagnostic smoke, not a main experiment.

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Dataset:

```text
alias: tiny_imagenet_200
path: /root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200
image_size: 64
class_conditional: false
```

Validation:

```text
pytest: 18 passed
python -m py_compile scripts/train_short.py scripts/train_tail_stage.py: passed
```

Code change:

- `scripts/train_short.py` now reports `late_half_energy_ratio` for K8 reports.

## Commands

Base K8 channel-mask smoke:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_smoke_800_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_smoke_800_2026-07-07
```

Tail warmup:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_warm_300_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_smoke_800_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_300_2026-07-07
```

Endpoint guard with strong replay:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_endpoint_500_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_300_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_300_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_500_2026-07-07
```

## Artifacts

Remote:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_smoke_800_2026-07-07/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_smoke_800_2026-07-07/prefix_final.png
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_300_2026-07-07/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_300_2026-07-07/prefix_final.png
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_500_2026-07-07/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_500_2026-07-07/prefix_final.png
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_channelmask_smoke_800_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_channelmask_smoke_800_2026-07-07/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_tailstage_warm_300_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_warm_300_2026-07-07/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_500_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_500_2026-07-07/prefix_final.png
```

## Results

| run | epsilon/tail | endpoint | prefix 5 MSE | prefix 8 MSE | group energy ratio | tail ratio | late-half ratio | zero energy |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Tiny K8 base 800 | 0.2149 | n/a | 0.4651 | 0.4310 | [0.9849, 0.0110, 0.0025, 0.0015] | 0.2007 | 0.0041 | 0.0 |
| Tiny K8 tail warm 300 | 0.1212 | n/a | 0.4362 | 0.3902 | [0.9659, 0.0108, 0.0087, 0.0147] | 0.2135 | 0.0234 | 0.0 |
| Tiny K8 endpoint 500 | 0.1659 | 40.7980 | 0.4643 | 0.3634 | [0.9787, 0.0110, 0.0044, 0.0058] | 0.2029 | 0.0103 | 0.0 |

## Interpretation

- Tail warmup transfers: late-half energy rises from 0.41% to 2.34%.
- Endpoint guard with strong replay preserves 1.03% late-half energy, which is
  higher than the best CIFAR guarded run.
- Final prefix quality improves in the smoke: prefix 8 MSE moves from 0.4310
  for base to 0.3634 after endpoint guard.
- Intermediate prefixes still drift, and the visualization remains very noisy.
  This is not yet evidence of a clean ordered decomposition.

## Decision

The mechanism is worth one longer Tiny ImageNet diagnostic, but still not a main
table experiment. Next run should keep the same model and schedule but increase
only the base and endpoint phases:

```text
base: 2000 steps
tail warm: 500 steps
endpoint guard + strong replay: 1000 steps
```

Required before any paper-facing claim:

- repeat zero / random / shuffled diagnostics in the report;
- inspect prefix rows for monotone visual improvement;
- compare against a no-tail-stage Tiny K8 run at similar total step count.

## Follow-Up: Longer Tiny Diagnostic With No-Tail Control

Purpose: test whether the staged mechanism remains useful when training is
slightly longer, and compare against a no-tail-stage run with a similar total
step count.

Configuration:

```text
staged:
  base: train_tiny_imagenet_k8_channelmask_2k_cuda.json, 2000 steps
  tail warm: train_tiny_imagenet_k8_tailstage_warm_500_cuda.json, 500 steps
  endpoint guard: train_tiny_imagenet_k8_tailstage_endpoint_1k_cuda.json, 1000 steps

control:
  no-tail-stage: train_tiny_imagenet_k8_channelmask_3500_cuda.json, 3500 steps
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_2k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_2k_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_warm_500_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_2k_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_endpoint_1k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_1k_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_3500_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_3500_2026-07-07
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_channelmask_2k_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_channelmask_2k_2026-07-07/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_1k_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_1k_2026-07-07/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_channelmask_3500_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_channelmask_3500_2026-07-07/prefix_final.png
```

| run | epsilon/tail | endpoint | prefix 5 MSE | prefix 8 MSE | group energy ratio | tail ratio | late-half ratio | zero energy |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Tiny K8 base 2k | 0.1427 | n/a | 0.3322 | 0.3007 | [0.9957, 0.0035, 0.0005, 0.0003] | 0.0246 | 0.0009 | 0.0 |
| Tiny K8 tail warm 500 | 0.1368 | n/a | 0.3339 | 0.2590 | [0.9751, 0.0033, 0.0043, 0.0173] | 0.0436 | 0.0216 | 0.0 |
| Tiny K8 endpoint 1k | 0.0663 | 4.9478 | 0.3542 | 0.2488 | [0.9815, 0.0034, 0.0034, 0.0117] | 0.0387 | 0.0151 | 0.0 |
| Tiny K8 no-tail 3500 | 0.1492 | n/a | 0.3021 | 0.2713 | [0.9986, 0.0008, 0.0003, 0.0003] | 0.0220 | 0.0005 | 0.0 |

Additional diagnostics:

| run | original energy | random-token energy | shuffled-token MSE | zero-token energy |
|---|---:|---:|---:|---:|
| Tiny K8 base 2k | 0.1280 | 0.0805 | 0.2546 | 0.0 |
| Tiny K8 tail warm 500 | 0.1304 | 0.0870 | 0.2573 | 0.0 |
| Tiny K8 endpoint 1k | 0.1293 | 0.0820 | 0.2546 | 0.0 |
| Tiny K8 no-tail 3500 | 0.1298 | 0.0793 | 0.2563 | 0.0 |

Interpretation:

- The longer staged run keeps substantially more late-half energy than the
  no-tail control: 1.51% versus 0.05%.
- The staged endpoint run also has better final prefix quality than no-tail:
  prefix 8 MSE 0.2488 versus 0.2713.
- The cost remains non-monotone intermediate prefixes: prefix 5 MSE is 0.3542
  for staged endpoint versus 0.3021 for no-tail.
- Visually, staged endpoint shows stronger final-row object/color recovery,
  while no-tail is smoother but mostly concentrates energy in the first group.
- Zero-token remains exact, and random/shuffled diagnostics do not show an
  obvious `S_k` decoder collapse in this smoke.

## Updated Decision

This is the first result worth promoting from smoke to a controlled diagnostic:
the staged mechanism improves both final prefix quality and late-token
utilization compared with same-scale no-tail training. It is still not a main
paper result because prefix monotonicity is weak.

Next step:

- add a late-prefix smoothness/monotonic guard over prefixes 5-8 only;
- keep strong replay to retain late-token energy;
- rerun Tiny endpoint stage from the same tail-warm checkpoint;
- compare against current endpoint 1k and no-tail 3500.

Success criterion: keep late-half energy above 1.0%, keep prefix 8 MSE near or
below 0.25, and reduce prefix 5-7 drift relative to endpoint-only guard.

## Follow-Up: Late-Prefix Monotonic Guard

Purpose: reduce the prefix 5-7 drift observed in endpoint-only guard while
retaining late-token utilization and final prefix quality.

Code change:

- Added `tail_late_monotonic_weight`, `tail_late_monotonic_start`, and
  `tail_late_monotonic_margin` to `LossConfig`.
- Added a tail-stage-only late monotonic guard over prefixes 5-8 in
  `scripts/train_tail_stage.py`.

Validation:

```text
pytest: 18 passed
python -m py_compile scripts/train_tail_stage.py: passed
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_endpoint_latemono_1k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_latemono_1k_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_2026-07-07
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_latemono_1k_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_latemono_1k_2026-07-07/prefix_final.png
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_2026-07-07/prefix_final.png
```

| run | late mono weight | endpoint | late mono | prefix 5 | prefix 6 | prefix 7 | prefix 8 | late-half ratio | group energy ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| endpoint-only 1k | 0.00 | 4.9478 | n/a | 0.3542 | 0.3970 | 0.3565 | 0.2488 | 0.0151 | [0.9815, 0.0034, 0.0034, 0.0117] |
| endpoint + late mono 0.10 | 0.10 | 7.6115 | 0.0000 | 0.3607 | 0.3885 | 0.3361 | 0.2566 | 0.0133 | [0.9833, 0.0033, 0.0041, 0.0093] |
| endpoint + late mono 0.03 | 0.03 | 5.8345 | 0.3970 | 0.3526 | 0.3852 | 0.3347 | 0.2485 | 0.0140 | [0.9826, 0.0033, 0.0039, 0.0101] |
| no-tail 3500 | n/a | n/a | n/a | 0.3021 | 0.2974 | 0.2928 | 0.2713 | 0.0005 | [0.9986, 0.0008, 0.0003, 0.0003] |

Interpretation:

- Hard late monotonic reduces prefix 6-7 drift but hurts final prefix quality.
- Soft late monotonic is the best balance so far: prefix 8 MSE stays at 0.2485,
  late-half energy remains 1.40%, and prefix 6-7 improve versus endpoint-only.
- Intermediate prefixes are still worse than no-tail 3500, so the sequence is
  not yet cleanly monotone.

## Updated Decision

Use `endpoint + late mono 0.03` as the current Tiny ImageNet diagnostic
candidate. It meets the smoke success criteria:

```text
late-half energy: 1.40% > 1.0%
prefix 8 MSE: 0.2485 <= 0.25
prefix 6-7 drift: improved versus endpoint-only
```

Next controlled step should not add more scalar loss terms. Instead, run a
seed repeat of the same candidate and the no-tail 3500 control. If the
late-half / prefix-8 advantage survives another seed, this becomes the first
paper-facing diagnostic candidate.

## Follow-Up: Seed Repeat

Purpose: repeat the current Tiny diagnostic candidate and the no-tail control
with a second seed.

Configuration:

```text
candidate seed2:
  base: train_tiny_imagenet_k8_channelmask_2k_seed2_cuda.json, seed 103
  tail warm: train_tiny_imagenet_k8_tailstage_warm_500_seed2_cuda.json, seed 107
  endpoint + soft late mono: train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_seed2_cuda.json, seed 109

control seed2:
  no-tail-stage: train_tiny_imagenet_k8_channelmask_3500_seed2_cuda.json, seed 113
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_2k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_2k_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_warm_500_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_2k_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_3500_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_3500_seed2_2026-07-07
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_channelmask_2k_seed2_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_seed2_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_channelmask_3500_seed2_2026-07-07/report.json
```

| run | prefix 5 | prefix 6 | prefix 7 | prefix 8 | late-half ratio | group energy ratio |
|---|---:|---:|---:|---:|---:|---|
| seed1 candidate | 0.3526 | 0.3852 | 0.3347 | 0.2485 | 0.0140 | [0.9826, 0.0033, 0.0039, 0.0101] |
| seed1 no-tail | 0.3021 | 0.2974 | 0.2928 | 0.2713 | 0.0005 | [0.9986, 0.0008, 0.0003, 0.0003] |
| seed2 candidate | 0.4625 | 0.4620 | 0.4590 | 0.3349 | 0.0155 | [0.9779, 0.0066, 0.0076, 0.0079] |
| seed2 no-tail | 0.2839 | 0.2819 | 0.2710 | 0.2347 | 0.0010 | [0.9982, 0.0008, 0.0002, 0.0008] |

Two-seed means:

| method | mean prefix 8 MSE | mean late-half ratio |
|---|---:|---:|
| candidate | 0.2917 | 0.0148 |
| no-tail | 0.2530 | 0.0008 |

Interpretation:

- The candidate reliably activates late tokens: mean late-half ratio is about
  1.48%, roughly 19x the no-tail control.
- The final quality advantage does not survive the seed repeat. Candidate mean
  prefix 8 MSE is 0.2917 versus 0.2530 for no-tail.
- Seed2 is a clear failure case for endpoint quality even though late-token
  utilization remains high.

## Follow-Up: Endpoint Weight Probe On Seed2

Purpose: test whether the seed2 failure is just because endpoint weight 0.03 is
too weak.

Command:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_e006_1k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_e006_1k_seed2_2026-07-07
```

| run | endpoint weight | prefix 8 | late-half ratio | group energy ratio |
|---|---:|---:|---:|---|
| seed2 endpoint soft | 0.03 | 0.3349 | 0.0155 | [0.9779, 0.0066, 0.0076, 0.0079] |
| seed2 endpoint soft e006 | 0.06 | 0.3446 | 0.0143 | [0.9792, 0.0066, 0.0074, 0.0069] |
| seed2 no-tail | n/a | 0.2347 | 0.0010 | [0.9982, 0.0008, 0.0002, 0.0008] |

Interpretation:

- Increasing endpoint weight from 0.03 to 0.06 does not fix seed2. It slightly
  worsens prefix 8 and slightly reduces late-half energy.
- The issue is not simply weak endpoint supervision.

## Updated Decision

Do not promote the current candidate to paper-facing status. The reliable
finding is narrower:

```text
tail-stage training robustly increases late-token utilization,
but current endpoint / late-monotonic guards do not reliably improve final
prefix quality across seeds.
```

Next useful direction is architectural or optimization-level, not more scalar
loss tuning. Two candidates:

- alternate tail updates and endpoint updates with separate optimizer states,
  so endpoint correction cannot erase or destabilize tail specialization;
- initialize tail-stage from a stronger base checkpoint, because seed2 base 2k
  has much worse prefix quality before the tail stage.

## Follow-Up: Alternating Tail / Endpoint Updates

Purpose: test a minimal alternating schedule without writing a new trainer. The
run starts from the seed2 tail-warm checkpoint and chains five short stages with
fresh optimizer state each time:

```text
endpoint 200 -> tail 200 -> endpoint 200 -> tail 200 -> endpoint 200
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_alt_endpoint_200_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_e1_200_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_alt_tail_200_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_e1_200_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_t2_200_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_alt_endpoint_200_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_t2_200_seed2_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_e3_200_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_alt_tail_200_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_e3_200_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_t4_200_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_alt_endpoint_200_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_t4_200_seed2_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_alt_e5_200_seed2_2026-07-07
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_tailstage_alt_e5_200_seed2_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_alt_e5_200_seed2_2026-07-07/prefix_final.png
```

| run | prefix 5 | prefix 6 | prefix 7 | prefix 8 | late-half ratio | group energy ratio |
|---|---:|---:|---:|---:|---:|---|
| seed2 endpoint soft 1k | 0.4625 | 0.4620 | 0.4590 | 0.3349 | 0.0155 | [0.9779, 0.0066, 0.0076, 0.0079] |
| seed2 alternating 5x200 | 0.4382 | 0.4410 | 0.4371 | 0.3282 | 0.0136 | [0.9798, 0.0065, 0.0069, 0.0067] |
| seed2 no-tail 3500 | 0.2839 | 0.2819 | 0.2710 | 0.2347 | 0.0010 | [0.9982, 0.0008, 0.0002, 0.0008] |

Interpretation:

- Alternating gives a small improvement over the monolithic endpoint run on
  seed2: prefix 8 improves from 0.3349 to 0.3282 and prefixes 5-7 improve too.
- It retains high late-half utilization: 1.36% versus 0.10% for no-tail.
- The improvement is far too small to close the quality gap to no-tail.

Updated decision:

Alternating is directionally useful but not sufficient in this minimal form.
The stronger diagnosis remains: late-token utilization can be forced, but it is
not yet coupled to consistently better denoising quality.

The next best experiment is not more alternating chunks. Instead, train a
stronger base checkpoint before tail-stage, because seed2 failure begins with a
weak base 2k checkpoint.

## Follow-Up: Stronger Base Checkpoint

Purpose: test whether the seed2 failure is mainly caused by a weak base
checkpoint. This run trains the seed2 base for 5k steps before applying the same
tail-stage mechanism. A no-tail 6500 control is included for a fairer total-step
comparison.

Configuration:

```text
staged from stronger base:
  base: train_tiny_imagenet_k8_channelmask_5k_seed2_cuda.json, 5000 steps
  tail warm: train_tiny_imagenet_k8_tailstage_warm_500_from5k_seed2_cuda.json, 500 steps
  endpoint + soft late mono: train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_from5k_seed2_cuda.json, 1000 steps

control:
  no-tail-stage: train_tiny_imagenet_k8_channelmask_6500_seed2_cuda.json, 6500 steps
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_5k_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_5k_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_warm_500_from5k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_5k_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_from5k_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_from5k_seed2_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_from5k_seed2_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_warm_500_from5k_seed2_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_from5k_seed2_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_tiny_imagenet_k8_channelmask_6500_seed2_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_channelmask_6500_seed2_2026-07-07
```

Local copies:

```text
artifacts/reports/train_tiny_imagenet_k8_channelmask_5k_seed2_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_warm_500_from5k_seed2_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_tailstage_endpoint_latemono_soft_1k_from5k_seed2_2026-07-07/report.json
artifacts/reports/train_tiny_imagenet_k8_channelmask_6500_seed2_2026-07-07/report.json
```

| run | prefix 5 | prefix 6 | prefix 7 | prefix 8 | late-half ratio | group energy ratio |
|---|---:|---:|---:|---:|---:|---|
| seed2 base 5k | 0.2793 | 0.2777 | 0.2756 | 0.2641 | 0.0003 | [0.9983, 0.0015, 0.0001, 0.0001] |
| seed2 warm from 5k | 0.2792 | 0.2865 | 0.2915 | 0.2383 | 0.0080 | [0.9906, 0.0014, 0.0033, 0.0048] |
| seed2 candidate from 5k | 0.2936 | 0.2895 | 0.2892 | 0.2475 | 0.0048 | [0.9938, 0.0014, 0.0030, 0.0017] |
| seed2 no-tail 3500 | 0.2839 | 0.2819 | 0.2710 | 0.2347 | 0.0010 | [0.9982, 0.0008, 0.0002, 0.0008] |
| seed2 no-tail 6500 | 0.2709 | 0.2687 | 0.2594 | 0.2365 | 0.0007 | [not expanded here] |

Diagnostics:

| run | original energy | random-token energy | shuffled-token MSE | zero-token energy |
|---|---:|---:|---:|---:|
| seed2 candidate from 5k | 0.1307 | 0.0994 | 0.2610 | 0.0 |
| seed2 no-tail 6500 | 0.1321 | 0.0761 | 0.2625 | 0.0 |

Interpretation:

- A stronger base does prevent the severe seed2 collapse seen with the 2k base.
- Tail warm from the stronger base improves final prefix MSE to 0.2383 and
  activates late tokens, but endpoint guard partially gives back that quality.
- Against the fairer no-tail 6500 control, the staged candidate still loses
  final and intermediate prefix quality while retaining more late-half energy.
- The core tradeoff remains: late-token utilization improves, but quality does
  not reliably improve.

## Updated Decision

Do not spend more runs on the current tail-stage objective. The evidence is now
stable enough:

```text
CoFiTok v0 demonstrates controllable late-token utilization under restricted
S_k, but the current training objective does not yet make that utilization
quality-positive.
```

Next substantial move should be a model/objective redesign, not another schedule
tweak. The most plausible directions are:

- make token order explicit in the predictor, e.g. separate per-token timestep
  or residual state, while keeping `S_k` condition-free;
- add a frozen-teacher decomposition target from a trained dense epsilon model;
- introduce a constrained orthogonal/residual basis so late components cannot
  merely repaint early errors.
