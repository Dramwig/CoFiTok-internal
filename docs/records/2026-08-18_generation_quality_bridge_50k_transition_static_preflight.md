# Full-data quality-bridge 50K transition static preflight

Date: 2026-08-18

## Outcome

The active full-data ImageNet-256 quality bridge is statically ready for its
first 50K checkpoint/evaluation/dense-transition sequence on the recloned
server. The exact training and milestone runbooks pass native shell syntax,
the expected Python environment resolves, ImageNet validation data and the
evaluator cache are present, and all seven milestone/evaluator entry points
import successfully with CUDA hidden.

This was a CPU-only, read-only preflight. It did not load a checkpoint, use the
GPU, signal a process, modify the active checkout, create a training output, or
authorize any later stage.

## Active training boundary

At the audited metrics boundary:

```text
CoFiTok step: 32,650 / 100,000
canonical images: 2,089,600
pair status/stage: running / cofitok_training
pair issues: []
dense step: 0
```

The 52 canonical rows from step 30,100 through 32,650 were re-read and checked:

```text
finite issues: 0
step-interval issues: 0
samples_seen binding issues: 0
EMA-teacher schedule issues: 0
epsilon mean: 0.027535962442365978
epsilon range: [0.016526220366358757, 0.034545853734016418]
maximum grad norm: 3.6201052665710449
last EMA-teacher scale: 0.26499998569488525
```

The observed 2,550-step window averaged `2.82663` seconds per optimizer step.
At this rate, step 40K was estimated near `2026-08-18 09:14 CST`, roughly 5.77
hours after the rate audit.

## Exact execution identity

Training checkout:

```text
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree: 6cef27723196fd363379bca2e7b85b1678ebd777
branch: scale/generation-stability-quality-bridge-100k
tracked state: clean
```

Quality-bridge runbook:

```text
bytes: 27,943
SHA256:
f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73
bash -n: pass
```

Milestone-evaluation runbook:

```text
bytes: 1,889
SHA256:
cf0f53c5c6bcfd7ca861e5f583b2d503213d633275b5751c7ba5964bd3a0cb56
bash -n: pass
```

The exact 50K order encoded by the runbook is:

1. train CoFiTok to exact step 50K and verify the physical checkpoint;
2. run CoFiTok's 2,048-sample DDIM-50 milestone evaluation;
3. train dense identity to exact step 50K and verify its checkpoint;
4. run the matched dense milestone evaluation;
5. build and validate the paired step-50K milestone report.

The milestone protocol uses EMA, bf16, batch 32, DDIM-50, CFG 1.5, and 2,048
samples per method. Prefix budget is 8 for CoFiTok and 1 for dense. CoFiTok's
four random-order evaluations are factorization diagnostics, not additional
dense quality samples.

## Re-cloned server environment

The runbook's conda initialization path exists and activates:

```text
python: /root/autodl-tmp/conda/envs/pf-vlm/bin/python
version: Python 3.10.20
```

The ImageNet-256 validation path contains exactly `50,000` files and occupies
`707,180,897` bytes. The expected Inception feature-cache file is present at
`409,601,577` bytes. Its source-bound expected SHA256 remains:

```text
20103588dca9ce47bfceef6b68b473fdf4be720f149d1b8bdd96341d27c10dcd
```

The cache payload was deliberately not rehashed during active training to
avoid unnecessary I/O; the matched-uncertainty waiter remains responsible for
physical verification before using it.

With `CUDA_VISIBLE_DEVICES=''`, these exact entry points all passed `--help`:

- `preflight_generation_sampling.py`;
- `evaluate_generation_checkpoint.py`;
- `generate_samples.py`;
- `evaluate_generation_metrics.py`;
- `evaluate_generation_class_fidelity.py`;
- `build_generation_milestone_report.py`;
- `validate_generation_milestone_report.py`.

Filesystem free space was `325,207,347,200` bytes.

## Schedule and downstream isolation

The CoFiTok full-warmup waiter remains active as PID `816282`. It is bound to:

```text
target step: 40,000
first retained active step: 39,500
minimum retained active rows: 11
source SHA256:
10c1676d0de5e7aaa94f428a65448ab78da7fdbdc7184a76de7278aaad6d9998
```

Its 12-hour timeout leaves several hours of margin beyond the current 40K ETA.

All downstream branches remain gated:

- matched uncertainty waits for the exact quality-bridge terminal result;
- the follow-up decision waits for `quality_bridge_result.json`;
- capacity preparation waits for that source-bound follow-up decision;
- capacity scaling, completion, and full-300K supervisors have no child;
- both capacity training run directories are absent;
- the capacity lineage v2 observer reports `issues=[]`.

The only GPU compute process was the active base128 CoFiTok trainer PID
`619775`, using `85,284 MiB`. No unrelated GPU process was present.

## Claim boundary

This preflight proves environment and orchestration readiness at the audited
boundary. It does not prove the 40K schedule transition, the 50K checkpoint,
50K sample quality, dense observed runtime, broad generation superiority,
promotion readiness, release readiness, or authority for full-300K training.

Compact evidence artifact:

```text
artifacts/reports/generation/
quality_bridge_50k_transition_static_preflight_2026-08-18.json
bytes: 6,776
SHA256: 9471f615525b209680c6023c5ca5aa74d1b4d2fdebd66f773544cc1e46ae9a00
Git blob OID: 86923d72d948c84ee49ff7a74aef31d5c73a3555
```
