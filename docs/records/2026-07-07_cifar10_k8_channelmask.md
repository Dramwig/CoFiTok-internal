# 2026-07-07 CIFAR-10 K8 Channel Mask

Purpose: test whether increasing token count from K=4 to K=8 prevents tail
collapse under a per-token channel capacity schedule.

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Validation:

```text
pytest: 17 passed
```

Code change:

- `_prefix_targets` now caps pooling scale at the image resolution so CIFAR-10
  32x32 supports K=8 prefix targets.
- Added test coverage for K=8 loss computation on 32x32 images.

Command:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_channelmask_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07
```

Config:

```text
dataset: cifar10
K: 8
token_channels: 16
synthesis_active_token_channels: [4, 4, 8, 8, 12, 12, 16, 16]
steps: 3000
```

Artifacts:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07/report.json
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07/checkpoint_final.pt
/root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07/prefix_final.png
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_channelmask_3k_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_channelmask_3k_2026-07-07/prefix_final.png
```

## Results

| run | total | epsilon | prefix | prefix MSE | energy ratio | tail ratio | late-half ratio | active tail | zero energy |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| K4 baseline 1k | 6.5872 | 0.1595 | 25.7108 | [0.7430, 0.3842, 0.3683, 0.3539] | [0.9600, 0.0373, 0.0020, 0.0007] | 0.0400 | 0.0027 | 1 | 0.0 |
| K4 channel-mask 3k | 1.9978 | 0.1401 | 7.4306 | [0.5352, 0.2384, 0.2250, 0.2212] | [0.9722, 0.0269, 0.0008, 0.0002] | 0.0278 | 0.0010 | 1 | 0.0 |
| K8 channel-mask 3k | 1.0490 | 0.1449 | 3.6158 | [0.5103, 0.2502, 0.2144, 0.2086, 0.2092, 0.2073, 0.2039, 0.1960] | [0.9772, 0.0195, 0.0026, 0.0002, 0.0002, 0.0001, 0.0001, 0.0002] | 0.0228 | 0.0005 | 1 | 0.0 |

## Interpretation

- K=8 improves prefix quality and final prefix loss versus K=4.
- Increasing token count alone does not prevent collapse. After 3k steps, token
  1 carries about 97.7% of component energy.
- The late half of K=8 carries almost no energy, about 0.046%.
- `S_k(0)=0` remains exact.

## Decision

Do not move this directly to Tiny ImageNet yet. The next experiment should add
explicit group-level supervision or staged training so late tokens must solve a
distinct objective. Good next candidates:

- train K=8 with prefix dropout that masks early tokens for some batches;
- freeze or detach early components after warmup and train tail components on
  residual epsilon;
- add group residual targets for tokens 1-2, 3-4, 5-6, 7-8.

The data/model pipeline is stable, but the current objective still permits
early-token collapse.

## Follow-Up: Group Residual Supervision

Code change:

- Added `group_residual_weight` and `group_residual_size` to `LossConfig`.
- Added group residual loss over adjacent token groups. For K=8 and group size
  2, the groups are `[1-2], [3-4], [5-6], [7-8]`.

Validation:

```text
pytest: 18 passed
```

Command:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_short.py \
  --config configs/train_cifar10_k8_groupres_3k_cuda.json \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_groupres_3k_2026-07-07
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_groupres_3k_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_groupres_3k_2026-07-07/prefix_final.png
```

| run | total | epsilon | prefix | group residual | group energy ratio | tail ratio | late-half ratio | zero energy |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| K8 channel-mask 3k | 1.0490 | 0.1449 | 3.6158 | n/a | [0.9967, 0.0028, 0.0002, 0.0002] | 0.0228 | 0.0005 | 0.0 |
| K8 group-residual 3k | 1.4583 | 0.1326 | 3.5768 | 1.2193 | [0.9963, 0.0030, 0.0003, 0.0004] | 0.0230 | 0.0007 | 0.0 |

Interpretation:

- Group residual supervision preserves quality and slightly improves epsilon.
- It does not materially activate late groups. The late-half ratio remains below
  0.1%.
- The loss is likely too easy to absorb through earlier groups or too weak
  relative to the full epsilon / prefix losses.

Updated decision:

The next useful experiment is staged tail training rather than another static
regularizer. Use the existing K8 checkpoint as warm start, freeze or detach
early groups, and train tail groups on residual epsilon. This directly tests
whether late components can learn useful residuals when early components are
not allowed to keep absorbing the objective.

## Follow-Up: Staged Tail Training

Code change:

- Added `scripts/train_tail_stage.py`.
- The script loads a warm-start checkpoint, freezes early tokens, and trains
  tail token heads / feedback / synthesis modules on residual epsilon.

Validation:

```text
pytest: 18 passed
python -m py_compile scripts/train_tail_stage.py: passed
```

Command:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_2k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_2k_2026-07-07
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_tailstage_2k_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_2k_2026-07-07/prefix_final.png
```

| run | epsilon/tail | prefix MSE | group energy ratio | tail ratio | late-half ratio | zero energy |
|---|---:|---|---|---:|---:|---:|
| K8 channel-mask 3k | 0.1449 | [0.5103, 0.2502, 0.2144, 0.2086, 0.2092, 0.2073, 0.2039, 0.1960] | [0.9967, 0.0028, 0.0002, 0.0002] | 0.0228 | 0.0005 | 0.0 |
| K8 group-residual 3k | 0.1326 | [0.5047, 0.2715, 0.2245, 0.2153, 0.2161, 0.2108, 0.2074, 0.1896] | [0.9963, 0.0030, 0.0003, 0.0004] | 0.0230 | 0.0007 | 0.0 |
| K8 tail-stage 2k | 0.1514 | [0.5223, 0.2625, 0.2307, 0.2251, 0.2479, 0.2676, 0.2484, 0.2237] | [0.9842, 0.0025, 0.0065, 0.0069] | 0.0355 | 0.0134 | 0.0 |

Interpretation:

- Staged tail training is the first intervention that materially activates late
  groups: late-half energy rises from about 0.046% to 1.34%.
- The cost is visible in prefix quality for prefixes 5-7, which suggests tail
  components can over-correct when trained only on residual epsilon.
- This is a useful proof that late tokens can learn nontrivial residuals, but
  staged training needs a guardrail to preserve prefix quality.

Updated decision:

The next experiment should use staged training with a prefix-preservation term:
keep early groups frozen, train the tail residual, but also penalize degradation
of full-prefix and late-prefix denoising quality. If that stabilizes, this
becomes the first plausible route to Tiny ImageNet.

## Follow-Up: Staged Tail Training With Prefix Guardrail

Code change:

- Updated `scripts/train_tail_stage.py` so the staged tail loss can include the
  regular `prefix_weight` and `monotonic_weight` guardrails from the config.
- Early tokens remain frozen; only tail token heads, tail synthesis modules, and
  feedback into later tokens are trainable.

Validation:

```text
pytest: 18 passed
python -m py_compile scripts/train_tail_stage.py: passed
```

Command:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_guard_2k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_guard_2k_2026-07-07
```

Config:

```text
dataset: cifar10
K: 8
tail_start: 4
steps: 2000
prefix_weight: 0.10
monotonic_weight: 0.05
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_tailstage_guard_2k_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_guard_2k_2026-07-07/prefix_final.png
```

| run | epsilon/tail | prefix guard | monotonic | prefix MSE | group energy ratio | tail ratio | late-half ratio | zero energy |
|---|---:|---:|---:|---|---|---:|---:|---:|
| K8 channel-mask 3k | 0.1449 | n/a | n/a | [0.5103, 0.2502, 0.2144, 0.2086, 0.2092, 0.2073, 0.2039, 0.1960] | [0.9967, 0.0028, 0.0002, 0.0002] | 0.0228 | 0.0005 | 0.0 |
| K8 tail-stage 2k | 0.1514 | n/a | n/a | [0.5223, 0.2625, 0.2307, 0.2251, 0.2479, 0.2676, 0.2484, 0.2237] | [0.9842, 0.0025, 0.0065, 0.0069] | 0.0355 | 0.0134 | 0.0 |
| K8 tail-stage guard 2k | 0.1727 | 7.7539 | 0.0258 | [0.5223, 0.2625, 0.2307, 0.2251, 0.2279, 0.2285, 0.2278, 0.2067] | [0.9954, 0.0025, 0.0014, 0.0007] | 0.0245 | 0.0021 | 0.0 |

Interpretation:

- The guardrail preserves late-prefix quality much better than pure tail-stage
  training. Prefixes 5-8 no longer over-correct as strongly.
- The same guardrail also suppresses tail activation. Late-half energy is
  0.213%, up from the 0.046% warm-start baseline but far below the 1.34% pure
  tail-stage result.
- This confirms the tradeoff: frozen-tail residual training can activate late
  tokens, but prefix preservation quickly pushes energy back toward early
  components.

Updated decision:

Do not promote this to Tiny ImageNet yet. The next CoFiTok step should be a
staged schedule rather than a single static objective:

- warm-start with K8 channel-mask training;
- run a short tail-only residual phase to make late tokens useful;
- anneal in prefix / monotonic guardrails after tail activation appears;
- compare late-half energy, prefix MSE, and visual prefix rows before deciding
  whether this is stable enough for Tiny ImageNet.

## Follow-Up: Two-Stage Tail Warmup Then Weak Guard

Purpose: test whether late-token activation from tail-only residual training can
survive a weaker prefix-preservation phase.

Configuration:

```text
phase A: train_cifar10_k8_tailstage_warm_800_cuda.json
  steps: 800
  prefix_weight: 0.0
  monotonic_weight: 0.0

phase B: train_cifar10_k8_tailstage_anneal_guard_1200_cuda.json
  steps: 1200
  prefix_weight: 0.03
  monotonic_weight: 0.02
  learning_rate: 1e-4
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_warm_800_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_anneal_guard_1200_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_anneal_guard_1200_2026-07-07
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_tailstage_warm_800_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_warm_800_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k8_tailstage_anneal_guard_1200_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_anneal_guard_1200_2026-07-07/prefix_final.png
```

| run | epsilon/tail | prefix guard | monotonic | prefix MSE | group energy ratio | tail ratio | late-half ratio | zero energy |
|---|---:|---:|---:|---|---|---:|---:|---:|
| K8 channel-mask 3k | 0.1449 | n/a | n/a | [0.5103, 0.2502, 0.2144, 0.2086, 0.2092, 0.2073, 0.2039, 0.1960] | [0.9967, 0.0028, 0.0002, 0.0002] | 0.0228 | 0.0005 | 0.0 |
| K8 tail-stage 2k | 0.1514 | n/a | n/a | [0.5223, 0.2625, 0.2307, 0.2251, 0.2479, 0.2676, 0.2484, 0.2237] | [0.9842, 0.0025, 0.0065, 0.0069] | 0.0355 | 0.0134 | 0.0 |
| K8 tail warm 800 | 0.0768 | 0.0000 | 0.0000 | [0.5197, 0.2649, 0.2319, 0.2263, 0.2378, 0.2317, 0.2337, 0.2241] | [0.9878, 0.0025, 0.0044, 0.0053] | 0.0315 | 0.0097 | 0.0 |
| K8 warm + weak guard 1200 | 0.0887 | 10.3365 | 0.0706 | [0.5200, 0.2570, 0.2279, 0.2227, 0.2182, 0.2148, 0.2104, 0.1902] | [0.9954, 0.0024, 0.0013, 0.0009] | 0.0246 | 0.0022 | 0.0 |

Interpretation:

- Tail-only warmup successfully activates late tokens: late-half ratio reaches
  0.97%, close to the longer 2k tail-stage result but with less prefix damage.
- The weak guard phase improves prefix quality substantially. Its final prefix
  row is better than the warmup and comparable to the original K8 baseline.
- The weak guard phase still collapses much of the tail activation: late-half
  ratio falls from 0.97% to 0.22%.
- Static prefix / monotonic guardrails are therefore not enough. They preserve
  prefix quality by re-concentrating energy into early components.

Updated decision:

Still do not move to Tiny ImageNet. The evidence now points to a training
mechanism problem rather than a capacity problem. Next candidates should force
late components to remain causally useful while preserving prefix behavior:

- early-token dropout during tail training, so later tokens sometimes must
  compensate for missing early components;
- stop-gradient residual targets for tail groups, with a replay term that keeps
  late components close to the tail-warm checkpoint;
- alternating batches: tail residual batches for activation, prefix guard
  batches for stability, without letting the guard batch update early-token
  energy allocation.

## Follow-Up: Early-Token Dropout And Tail Replay

Purpose: test two mechanisms that might preserve late-token usefulness after
the guard phase:

- early-token dropout: randomly remove some early-token feedback/component
  contribution during tail-stage training;
- tail replay: use the tail-warm checkpoint as a teacher and penalize late
  components drifting back toward collapse during guarded training.

Code change:

- Added `tail_early_dropout_weight`, `tail_early_dropout_prob`, and
  `tail_early_dropout_start` to `LossConfig`.
- Added a tail-stage-only masked forward path in `scripts/train_tail_stage.py`.
  This does not change the main `CoFiTokTiny` forward path and does not relax
  the restricted `S_k` operator.
- Added `tail_replay_weight` and optional `--replay-checkpoint` support to
  `scripts/train_tail_stage.py`.

Validation:

```text
pytest: 18 passed
python -m py_compile scripts/train_tail_stage.py: passed
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_earlydrop_guard_2k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_channelmask_3k_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_earlydrop_guard_2k_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_replay_guard_1200_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_replay_guard_1200_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_replay_guard_strong_1200_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_replay_guard_strong_1200_2026-07-07
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_tailstage_earlydrop_guard_2k_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_earlydrop_guard_2k_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k8_tailstage_replay_guard_1200_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_replay_guard_1200_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k8_tailstage_replay_guard_strong_1200_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_replay_guard_strong_1200_2026-07-07/prefix_final.png
```

| run | epsilon/tail | early dropout | replay | prefix guard | prefix 8 MSE | group energy ratio | tail ratio | late-half ratio |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| K8 channel-mask 3k | 0.1449 | n/a | n/a | 3.6158 | 0.1960 | [0.9967, 0.0028, 0.0002, 0.0002] | 0.0228 | 0.0005 |
| K8 tail warm 800 | 0.0768 | n/a | n/a | 0.0000 | 0.2241 | [0.9878, 0.0025, 0.0044, 0.0053] | 0.0315 | 0.0097 |
| K8 warm + weak guard 1200 | 0.0887 | n/a | n/a | 10.3365 | 0.1902 | [0.9954, 0.0024, 0.0013, 0.0009] | 0.0246 | 0.0022 |
| K8 early-drop guard 2k | 0.0912 | 0.1021 | n/a | 7.8973 | 0.2023 | [0.9948, 0.0025, 0.0014, 0.0013] | 0.0260 | 0.0027 |
| K8 replay guard 0.5 | 0.0932 | n/a | 0.0022 | 5.3233 | 0.1909 | [0.9952, 0.0027, 0.0013, 0.0008] | 0.0246 | 0.0021 |
| K8 replay guard 20 | 0.1121 | n/a | 0.0005 | 7.3659 | 0.1998 | [0.9929, 0.0022, 0.0022, 0.0026] | 0.0277 | 0.0049 |

Interpretation:

- Conservative early-token dropout is not enough. It slightly increases late
  energy over weak guard but also worsens prefix quality.
- Tail replay with weight 0.5 is too weak; the replay term is numerically tiny
  and does not prevent late-half collapse.
- Strong tail replay is the best balanced mechanism so far. It preserves about
  0.49% late-half energy, versus 0.22% for weak guard, while keeping final
  prefix MSE close to the K8 baseline.
- This is still not a convincing ordered decomposition. Token 1 still carries
  about 97% of component energy.

Updated decision:

Do not move to Tiny ImageNet. The best next experiment should combine the two
useful pieces more explicitly:

- keep tail-warm activation;
- apply strong replay on tail groups;
- replace full prefix guard with late-prefix-only guard, so the objective
  preserves prefix 5-8 quality without rewarding energy migration back into
  token 1.

If late-prefix-only guard plus strong replay keeps late-half energy above 0.5%
with prefix 8 MSE near 0.20, then it becomes worth testing a small Tiny
ImageNet smoke. Otherwise, the K8 objective still has a structural collapse
problem.

## Follow-Up: Strong Replay With Late-Prefix / Endpoint Guard

Purpose: replace full prefix guard with guards that only update tail parameters:

- late-prefix guard: clean-image MSE over prefixes 5-8 only;
- endpoint guard: clean-image MSE over prefix 8 only.

Code change:

- Added `tail_late_prefix_weight`, `tail_late_prefix_start`, and
  `tail_endpoint_weight` to `LossConfig`.
- Added tail-stage-only late-prefix and endpoint guard losses in
  `scripts/train_tail_stage.py`.
- Full `prefix_weight` and `monotonic_weight` are disabled in these runs, so
  the guard no longer supervises prefixes 1-4.

Validation:

```text
pytest: 18 passed
python -m py_compile scripts/train_tail_stage.py: passed
```

Commands:

```bash
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_replay_lateprefix_1200_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_replay_lateprefix_1200_2026-07-07

PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/train_tail_stage.py \
  --config configs/train_cifar10_k8_tailstage_replay_endpoint_1200_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --replay-checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_warm_800_2026-07-07/checkpoint_final.pt \
  --tail-start 4 \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/train_cifar10_k8_tailstage_replay_endpoint_1200_2026-07-07
```

Local copies:

```text
artifacts/reports/train_cifar10_k8_tailstage_replay_lateprefix_1200_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_replay_lateprefix_1200_2026-07-07/prefix_final.png
artifacts/reports/train_cifar10_k8_tailstage_replay_endpoint_1200_2026-07-07/report.json
artifacts/reports/train_cifar10_k8_tailstage_replay_endpoint_1200_2026-07-07/prefix_final.png
```

| run | epsilon/tail | replay | late-prefix | endpoint | prefix 5 MSE | prefix 8 MSE | group energy ratio | tail ratio | late-half ratio |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| K8 channel-mask 3k | 0.1449 | n/a | n/a | n/a | 0.2092 | 0.1960 | [0.9967, 0.0028, 0.0002, 0.0002] | 0.0228 | 0.0005 |
| K8 tail warm 800 | 0.0768 | n/a | n/a | n/a | 0.2378 | 0.2241 | [0.9878, 0.0025, 0.0044, 0.0053] | 0.0315 | 0.0097 |
| K8 replay guard 20 | 0.1121 | 0.0005 | n/a | n/a | 0.2255 | 0.1998 | [0.9929, 0.0022, 0.0022, 0.0026] | 0.0277 | 0.0049 |
| K8 replay late-prefix | 0.0712 | 0.0009 | 2.9308 | n/a | 0.2218 | 0.1998 | [0.9938, 0.0025, 0.0017, 0.0020] | 0.0262 | 0.0037 |
| K8 replay endpoint | 0.1258 | 0.0008 | n/a | 3.4206 | 0.2445 | 0.2075 | [0.9914, 0.0023, 0.0029, 0.0034] | 0.0301 | 0.0063 |

Interpretation:

- Endpoint-only guard retains the most late-half energy in a guarded run:
  0.63%, above the 0.5% threshold set in the previous decision.
- The cost is worse prefix quality. Prefix 5-7 visibly drift more, and prefix 8
  MSE is 0.2075 versus 0.1960 for K8 baseline.
- Late-prefix guard is more stable visually and numerically, but it suppresses
  late-half energy to 0.37%.
- Strong replay with the original weak full guard remains a reasonable balance,
  but endpoint-only guard is the first guarded run that keeps late-half energy
  above 0.5%.

Updated decision:

This is just enough evidence to justify a Tiny ImageNet smoke, but not a main
experiment. The smoke should be small and diagnostic:

- use K8, channel mask, tail warmup, and endpoint guard with strong replay;
- cap steps tightly so it tests transfer of the mechanism, not final quality;
- require the same diagnostics: component energy, zero-token, random-token,
  shuffled-token, and prefix rows.

If Tiny ImageNet shows the same tradeoff, the next design move should be a
proper alternating optimizer or objective that separates tail activation from
endpoint quality, rather than more scalar loss tuning.
