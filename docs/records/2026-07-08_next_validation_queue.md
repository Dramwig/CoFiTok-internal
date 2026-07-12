# CoFiTok Next Validation Queue

Date: 2026-07-08

Status: launched on `pro6000` on 2026-07-08 after SSH recovered.

## Purpose

The MVP evidence already supports the scoped prefix-controllability claim, but
the completion audit still marks generation-quality strength, sample-count
scale, and seed coverage as incomplete. This queue is the next concrete remote
step toward the full objective:

- add seed-103 repeats for the two 20k main datasets and the two main variants;
- add a stronger `multiscale_unet` predictor pilot while keeping restricted
  `S_k` unchanged;
- evaluate larger fixed-timestep reconstruction/prefix quality slices;
- run order diagnostics for the 20k light CoFiTok seed-103 checkpoints;
- add no-prefix and clean-monotonic loss-ablation pilots for the two main
  64x64 datasets;
- generate larger DDIM sample sets with more sampling steps;
- evaluate generated samples against larger validation-reference slices with
  Inception features enabled.

## Added Configs

```text
configs/train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda.json
configs/train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda.json
configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda.json
configs/train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json
configs/train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json
configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda.json
```

The first four configs use `seed = 103`, `steps = 20000`, `K = 8`, and the same
tiny backbone as the seed-139 20k runs. The two multiscale configs use
`predictor_type = multiscale_unet`, `seed = 151`, `steps = 10000`, and keep
`synthesis_mode = restricted`. The four loss-ablation configs use 10k-step
tiny-conv pilots: `nopathprefix` removes the denoise-path prefix loss while
keeping component supervision, and `cleanmono` adds a small clean-prefix
monotonic guard so the current light denoise-path runs serve as the
no-monotonic reference.

All ten queue configs load locally with:

```powershell
$env:PYTHONPATH='src'
python -c "from cofitok.configs import load_config; ..."
```

## Generated Runbook

```text
artifacts/runbooks/next_validation_queue_2026-07-08.sh
artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json
```

Default queue size:

```text
train_count: 10
quality_count: 10
order_eval_count: 24
sampling_count: 6
generated_quality_count: 6
quality_images: 1024
sample_count: 2048
sample_steps: 50
real_count: 8192
```

The bash runbook is idempotent: each step uses a report file as a marker and
skips completed outputs. This allows rerunning after a partial server
interruption.

## Queue Status Snapshots

The runbook now calls:

```text
scripts/inspect_next_validation_queue.py
```

before the first training run and after summary regeneration. It writes:

```text
artifacts/reports/next_validation_queue_status_start_2026-07-08.json
artifacts/reports/next_validation_queue_status_final_2026-07-08.json
```

The status file checks 56 expected report markers: 10 train reports, 10 quality
reports, 24 order-eval reports, 6 sampling reports, and 6 generated-quality
reports. The four loss-ablation pilots intentionally skip sampling/generated
quality to keep the queue focused and tractable. This makes a partially
completed queue easy to resume without manually searching checkpoint
directories.

## Dataset-Condition Gate

The runbook now also calls:

```text
scripts/validate_dataset_conditions.py --conditions-dir docs/experiment_conditions --require-ok
```

before summary and evidence validation. This gate checks that the dataset
condition records remain complete for CIFAR-10, Tiny ImageNet-200, the
ImageNet-1K 64x64 HF fallback, and the failed preferred TFDS
`downsampled_imagenet_64` attempt. It prevents the next queue from being treated
as publication-facing evidence if source, split, manifest, or fallback
provenance records drift.

## Idea-Requirements Matrix Gate

The runbook now generates and validates:

```text
artifacts/reports/idea_requirements_2026-07-08.json
docs/records/2026-07-08_idea_requirements_matrix.md
```

with:

```text
scripts/validate_idea_requirements.py --require-mvp-ok
```

The current local matrix status is `mvp_ok_publication_pending`: method-code
requirements, multi-dataset evidence, prefix denoising, zero/random/shuffle
diagnostics, order/scaling ablations, deep/simultaneous ablations, and
quality/sampling smoke evidence are all `ok`. The only `pending` row is
publication-scale readiness, which is exactly what this remote queue is meant to
complete.

## Publication-Readiness Gate

The runbook now ends with:

```text
scripts/validate_publication_readiness.py --require-ready
```

This is stricter than the MVP evidence gate. It requires two 20k train seeds for
Tiny/ImageNet-64 HF epsilon-only and K8 light CoFiTok, 1024-image quality slices
with Inception and LPIPS, 20k K8 order diagnostics, 2048-image DDIM50 generated
sample sets against 8192 real images, completed `multiscale_unet` pilots, and
the no-prefix / clean-monotonic loss-ablation pilots.

The current local summary intentionally does not pass this gate yet. The queue
above is sized to make it pass after the remote runs finish and the summary is
regenerated.

## Remote Use After SSH Recovers

Status update: SSH recovered on 2026-07-08 and the full default queue was
launched with `scripts/remote_recovery_launch.py --start-queue --timeout 900`.
The active remote log is:

```text
/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/logs/next_validation_queue_2026-07-08.log
```

Preferred local launcher after SSH recovers:

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
python scripts/remote_recovery_launch.py --dry-run --start-queue
python scripts/remote_recovery_launch.py --probe-only
python scripts/remote_recovery_launch.py --start-queue
```

The launcher streams `artifacts/recovery/remote_recovery_2026-07-08.tar.gz`
over SSH, runs the recovery manifest post-extract checks, prints `nvidia-smi`,
and refuses to start the queue if active GPU compute processes are present. It
starts the runbook in `tmux` when available, otherwise `nohup`, with logs under
`artifacts/logs/`.

For ongoing monitoring from the local machine, use:

```powershell
python scripts/remote_queue_status.py --allow-unreachable `
  --output-json artifacts/reports/remote_queue_status_latest_2026-07-08.json
```

Manual equivalent after extracting the recovery bundle on `pro6000`:

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
python scripts/validate_dataset_conditions.py \
  --conditions-dir docs/experiment_conditions \
  --require-ok
python scripts/inspect_next_validation_queue.py \
  --checkpoint-root /root/autodl-tmp/CoFiTok/checkpoints
bash artifacts/runbooks/next_validation_queue_2026-07-08.sh
```

Optional overrides for a shorter first pass:

```bash
REQUIRE_PUBLICATION_READY=0 \
QUALITY_IMAGES=256 SAMPLE_COUNT=256 SAMPLE_STEPS=20 REAL_COUNT=1024 \
  bash artifacts/runbooks/next_validation_queue_2026-07-08.sh
```

The full default queue should be preferred before making stronger
publication-facing generation-quality claims.
