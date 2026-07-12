# CoFiTok Remote I/O Incident

Date: 2026-07-08

Status: open until `pro6000` SSH responds normally and the queued sync/pytest
steps below complete.

## What Completed Before the Incident

- Six ImageNet-64 HF seed-103 quality evaluations completed on `pro6000`:
  epsilon-only, K4/K8/K16 light CoFiTok, simultaneous predictor, and deep
  `S_k`.
- The remote `experiment_summary.json` was regenerated and read successfully
  with counts:

```text
train: 89
order_eval: 38
quality: 35
sampling: 16
generated_quality: 13
```

- Local summary CSV/Markdown files were regenerated from the updated summary
  JSON.
- Local summary consistency validation passed with
  `scripts/validate_summary_consistency.py`.

## Symptoms

After the summary regeneration, `pro6000` developed SSH/SFTP/I/O instability:

- TCP port 29844 remained reachable.
- `ssh pro6000 "echo ok"` timed out during SSH server response.
- Earlier checks showed several stale `D` / `Ds` processes from the interrupted
  summary and failed scp/SFTP transfers.
- Remote document sync and remote pytest could not be completed safely.

## Latest Local Continuation

Timestamp: 2026-07-08T09:13:44+08:00

- Added and validated `scripts/validate_mvp_evidence.py`, which checks dataset
  coverage, 20k Tiny/ImageNet tradeoff rows, K4/K8/K16 seed-repeated order
  ablations, deep-`S_k` and simultaneous ablations, the seed-103 quality slice,
  sampling rows, and generated-quality smoke rows.
- Local syntax check passed for the new validator, its test, and the recovery
  bundle script.
- Local summary consistency validation passed with counts:

```text
train: 89
order_eval: 38
quality: 35
sampling: 16
generated_quality: 13
```

- Local MVP evidence coverage validation passed with six checks:
  dataset coverage, 20k tradeoff, order scaling, deep/simultaneous ablations,
  quality seed slice, and sampling/generated quality.
- Focused local tests passed with:

```powershell
PYTHONPATH=. uvx pytest -q tests\test_validate_mvp_evidence.py `
  tests\test_validate_summary_consistency.py `
  tests\test_remote_recovery_bundle.py
```

  Result: `6 passed`.
- Rebuilt `artifacts/recovery/remote_recovery_2026-07-08.tar.gz` with 25 files.
  The manifest now includes `validate_mvp_evidence.py` in `post_extract_checks`.
- Latest light SSH probe still failed:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 \
  -o ServerAliveInterval=4 -o ServerAliveCountMax=1 pro6000 "echo ok"
Timeout, server connect.bjb1.seetacloud.com not responding.
```

## Latest Queue Preparation

Timestamp: 2026-07-08T09:22:36+08:00

- Added four seed-103 20k configs for the main Tiny/ImageNet-64 HF
  epsilon-only and K8 light CoFiTok comparison.
- Added `scripts/make_next_validation_queue.py` and
  `tests/test_next_validation_queue.py`.
- Generated `artifacts/runbooks/next_validation_queue_2026-07-08.sh` and its
  manifest. The queue contains 4 training runs, 4 1024-image quality
  evaluations, 6 20k order evaluations, 4 2048-sample DDIM50 sampling runs,
  and 4 generated-quality evaluations against 8192 real images.
- Local config loading passed for the four new configs.
- Local Python-side checks passed:

```text
validate_summary_consistency.py: ok
validate_mvp_evidence.py: ok
focused pytest: 9 passed
```

- Rebuilt `artifacts/recovery/remote_recovery_2026-07-08.tar.gz` with 34 files,
  including the next-validation runbook. The recovery manifest queues
  `bash -n artifacts/runbooks/next_validation_queue_2026-07-08.sh` for the
  remote post-extract check.
- Local Windows WSL `bash -n` could not run because `/bin/bash` is absent in the
  local WSL shim; this is not treated as evidence against the remote Linux
  runbook.

## Latest Synthesis Contract Gate

Timestamp: 2026-07-08T09:28:40+08:00

- Added `scripts/validate_synthesis_contract.py` and
  `tests/test_validate_synthesis_contract.py`.
- The gate validates the restricted `S_k` contract: restricted configs must use
  the restricted synthesis mode, token-only synthesis signatures, bias-free
  local linear synthesis, no forbidden activation/normalization/attention
  modules, no suspicious learned constant/prior names, and exact `S_k(0)=0` in
  dynamic mode.
- Local machine lacks `torch` in the bundled Python, so the local run used
  `--static-only`. Result:

```text
checked_count: 101
restricted_ok_count: 99
ablation_count: 2
validation_mode: static
```

- Focused local test for the contract gate passed: `3 passed`.
- The next-validation runbook and recovery post-extract checks now include
  dynamic `validate_synthesis_contract.py` without `--static-only`; that stronger
  check is queued for `pro6000` after SSH recovers.

## Latest Stronger Predictor Preparation

Timestamp: 2026-07-08T09:35:50+08:00

- Added optional `ModelConfig.predictor_type`, defaulting to `tiny_conv`.
- Implemented `MultiScaleTokenPredictor` / `predictor_type = multiscale_unet`,
  a small U-Net-like `T_k` backbone. This only strengthens the next-token
  predictor; it does not change the restricted `S_k` synthesis operator.
- Added two stronger-backbone pilot configs:
  - `configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda.json`
  - `configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda.json`
- Regenerated `artifacts/runbooks/next_validation_queue_2026-07-08.sh`.
  The queue now contains 6 training runs, 6 quality evaluations, 12 order evals,
  6 sampling runs, and 6 generated-quality evaluations.
- Updated `scripts/summarize_experiments.py` to preserve `predictor_type` and
  `predictor_multiscale_levels` in summary rows, so multiscale results remain
  distinguishable from tiny-conv results.
- These pilots were queued while SSH was unavailable; after recovery, the full
  next-validation runbook was launched through the recovery launcher.

## Latest Publication-Readiness Gate

Timestamp: 2026-07-08T09:45:00+08:00

- Added `scripts/validate_publication_readiness.py` and
  `tests/test_validate_publication_readiness.py`.
- The gate is stricter than `validate_mvp_evidence.py`: it checks two 20k
  seeds for the main Tiny/ImageNet-64 HF epsilon-only and K8 light comparisons,
  1024-image quality slices with Inception and LPIPS, 20k K8 order diagnostics,
  2048-image DDIM50 generated sample sets against 8192 real images, and
  completed `multiscale_unet` train/quality/sample/generated-quality pilots.
- Running the gate on the current local summary returns `status: not_ready`.
  This is expected: the missing items match the queued remote validation batch.
- Regenerated `artifacts/runbooks/next_validation_queue_2026-07-08.sh`; by
  default it runs `validate_publication_readiness.py --require-ready` after
  summarization. Short smoke reruns can set `REQUIRE_PUBLICATION_READY=0`.
- Focused local tests passed for the readiness validator, runbook generator, and
  recovery-bundle integration: `8 passed`.
- Earlier light SSH probes timed out with
  `Timeout, server connect.bjb1.seetacloud.com not responding.` Later on
  2026-07-08, SSH recovered and the queue was launched.

## Latest Queue Status Inspector

Timestamp: 2026-07-08T09:55:00+08:00

- Added `scripts/inspect_next_validation_queue.py` and
  `tests/test_inspect_next_validation_queue.py`.
- The inspector checks all expected markers for the queued batch: 6 train
  reports, 6 quality reports, 12 order-eval reports, 6 sampling reports, and 6
  generated-quality reports.
- The runbook now writes status snapshots before and after the queue:
  `artifacts/reports/next_validation_queue_status_start_2026-07-08.json` and
  `artifacts/reports/next_validation_queue_status_final_2026-07-08.json`.
- Local probe against an empty checkpoint root correctly reported
  `status: incomplete` with `missing: 36`.
- Focused local tests passed for the inspector, runbook generator, and recovery
  bundle integration: `8 passed`.

## Latest Loss-Ablation Queue Extension

Timestamp: 2026-07-08T10:08:00+08:00

- Added four 10k loss-ablation configs:
  - `configs/train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json`
  - `configs/train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda.json`
  - `configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json`
  - `configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda.json`
- `nopathprefix` removes the denoise-path prefix loss while keeping component
  supervision; `cleanmono` adds a small clean-prefix monotonic guard. The
  existing light denoise-path runs are the no-monotonic reference.
- Updated `summarize_experiments.py` to classify these rows as
  `no_prefix_loss_ablation` and `clean_monotonic_ablation`.
- Extended the next-validation runbook to 10 training runs, 10 quality
  evaluations, 24 order evals, 6 sampling runs, and 6 generated-quality
  evaluations. The four loss-ablation pilots skip sampling/generated-quality.
- The queue status inspector now expects 56 markers.
- Local focused tests for queue generation, queue inspection, publication
  readiness, summary variant classification, and recovery-bundle integration
  passed: `16 passed`. All 107 configs loaded successfully.

## Latest Dataset-Condition Gate

Timestamp: 2026-07-08T10:22:00+08:00

- Added `scripts/validate_dataset_conditions.py` and
  `tests/test_validate_dataset_conditions.py`.
- The gate validates current dataset provenance records for CIFAR-10, Tiny
  ImageNet-200, the ImageNet-1K 64x64 HF fallback, and the failed preferred
  TFDS `downsampled_imagenet_64` attempt.
- Local validation on real `docs/experiment_conditions/` records passed with
  `check_count: 3`, `ok_count: 3`, `missing_count: 0`.
- The next-validation runbook now runs the dataset-condition gate before
  summary consistency, MVP evidence, publication-readiness, and synthesis
  contract checks.
- Focused local tests covering dataset conditions, queue generation, recovery
  bundle integration, readiness, summary consistency, summary parsing, MVP
  evidence, synthesis contract, and config loading passed: `27 passed`.
- Rebuilt `artifacts/recovery/remote_recovery_2026-07-08.tar.gz` with 63 files,
  including the dataset-condition validator, condition records, and the new
  post-extract dataset-condition check.

## Latest Remote Recovery Launcher

Timestamp: 2026-07-08T10:37:00+08:00

- `pro6000` still times out during SSH server response:

```text
Timeout, server connect.bjb1.seetacloud.com not responding.
```

- Added `scripts/remote_recovery_launch.py` and
  `tests/test_remote_recovery_launch.py`.
- The launcher probes `pro6000`, streams/extracts
  `artifacts/recovery/remote_recovery_2026-07-08.tar.gz` with tar-over-SSH,
  runs manifest post-extract checks, and optionally starts the next-validation
  runbook in `tmux` or `nohup`.
- `--start-queue` now includes a default GPU-idle guard: it prints
  `nvidia-smi` and refuses to start if active GPU compute processes are present.
  Use `--allow-busy-gpu` only after manually confirming the remote job context.
- Local launcher tests passed: `5 passed`.
- Rebuilt `artifacts/recovery/remote_recovery_2026-07-08.tar.gz` with 65 files,
  including the launcher and launcher tests.

## Latest Idea-Requirements Matrix

Timestamp: 2026-07-08T10:45:00+08:00

- Added `scripts/validate_idea_requirements.py` and
  `tests/test_validate_idea_requirements.py`.
- The matrix maps the idea document requirements to current code, summary
  evidence, and the queued publication-scale batch.
- Local run on the current summary produced:

```text
status: mvp_ok_publication_pending
check_count: 11
ok_count: 10
pending_count: 1
missing_count: 0
```

- The generated artifacts are:
  - `artifacts/reports/idea_requirements_2026-07-08.json`
  - `docs/records/2026-07-08_idea_requirements_matrix.md`
- The only pending row is publication-scale readiness: two 20k main seeds,
  1024-image quality slices, and 2048/8192 generated-quality scale. These are
  exactly the queued remote validation targets.
- The next-validation runbook and recovery post-extract checks now run
  `scripts/validate_idea_requirements.py --require-mvp-ok`.
- Rebuilt `artifacts/recovery/remote_recovery_2026-07-08.tar.gz` with 69 files,
  including the idea-requirements validator, tests, JSON matrix, and Markdown
  matrix.

## Queued Recovery Steps

When SSH responds normally again:

1. Sync the local text artifacts to `pro6000`, especially:
   - `scripts/validate_summary_consistency.py`
   - `scripts/validate_mvp_evidence.py`
   - `scripts/validate_publication_readiness.py`
   - `scripts/validate_idea_requirements.py`
   - `scripts/validate_dataset_conditions.py`
   - `scripts/validate_synthesis_contract.py`
   - `scripts/inspect_next_validation_queue.py`
   - `scripts/remote_recovery_launch.py`
   - `scripts/summarize_experiments.py`
   - `src/cofitok/configs.py`
   - `src/cofitok/models/__init__.py`
   - `src/cofitok/models/cofitok.py`
   - `src/cofitok/models/predictors.py`
   - `tests/test_validate_summary_consistency.py`
   - `tests/test_validate_mvp_evidence.py`
   - `tests/test_validate_publication_readiness.py`
   - `tests/test_validate_idea_requirements.py`
   - `tests/test_validate_dataset_conditions.py`
   - `tests/test_validate_synthesis_contract.py`
   - `tests/test_inspect_next_validation_queue.py`
   - `tests/test_remote_recovery_launch.py`
   - `tests/test_configs.py`
   - `tests/test_forward.py`
   - `tests/test_predictors.py`
   - `tests/test_summarize_experiments.py`
   - `docs/records/2026-07-08_quality_seed_slice.md`
   - `docs/records/2026-07-08_next_validation_queue.md`
   - `docs/records/2026-07-08_idea_requirements_matrix.md`
   - `docs/records/2026-07-08_completion_audit.md`
   - `docs/experiment_conditions/datasets_2026-07-07.md`
   - `docs/experiment_conditions/imagenet_1k_64x64_hf_plan_2026-07-08.md`
   - `docs/experiment_conditions/imagenet_1k_64x64_hf_manifest_summary_2026-07-08.json`
   - `docs/experiment_conditions/downsampled_imagenet_64_plan_2026-07-08.md`
   - `docs/experiment_conditions/downsampled_imagenet_64_tfds_inspect_2026-07-08.json`
   - `docs/experiment_conditions/quality_metrics_inception_2026-07-08.md`
   - `docs/reports/cofitok_mvp_report_2026-07-08.md`
   - `docs/ARCHITECTURE.md`
   - `scripts/make_next_validation_queue.py`
   - the four `*_nopathprefix_10k_cuda.json` and `*_cleanmono_10k_cuda.json`
     loss-ablation configs
   - `tests/test_next_validation_queue.py`
   - `artifacts/runbooks/next_validation_queue_2026-07-08.sh`
   - `artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json`
   - `artifacts/reports/idea_requirements_2026-07-08.json`
   - `paper/draft.md`
   - all files under `artifacts/reports/summary_2026-07-08/`
   A recovery tarball can be built locally with:

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
python scripts/make_remote_recovery_bundle.py
```

   Preferred local launch path after SSH responds:

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
python scripts/remote_recovery_launch.py --dry-run --start-queue
python scripts/remote_recovery_launch.py --probe-only
python scripts/remote_recovery_launch.py --start-queue
```

   The launcher refuses to start the queue if `nvidia-smi` reports active GPU
   compute processes. Manual tar-over-SSH equivalent:

```bash
ssh pro6000 "mkdir -p /root/autodl-tmp/CoFiTok && tar -xzf - -C /root/autodl-tmp/CoFiTok" \
  < artifacts/recovery/remote_recovery_2026-07-08.tar.gz
```

2. Run:

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/inspect_next_validation_queue.py \
  --checkpoint-root /root/autodl-tmp/CoFiTok/checkpoints \
  --output artifacts/reports/next_validation_queue_status_recovered_2026-07-08.json
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/validate_dataset_conditions.py \
  --conditions-dir docs/experiment_conditions \
  --require-ok
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/validate_summary_consistency.py \
  --summary-dir artifacts/reports/summary_2026-07-08 \
  --doc docs/reports/cofitok_mvp_report_2026-07-08.md \
  --doc docs/records/2026-07-08_completion_audit.md
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/validate_idea_requirements.py \
  --project-root . \
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \
  --queue-manifest artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json \
  --output-json artifacts/reports/idea_requirements_2026-07-08.json \
  --output-md docs/records/2026-07-08_idea_requirements_matrix.md \
  --require-mvp-ok
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/validate_mvp_evidence.py \
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/validate_publication_readiness.py \
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/validate_synthesis_contract.py \
  --config-glob 'configs/*.json'
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python -m pytest -q
```

3. Update this incident record to closed and update validation counts in the
   report/audit if pytest count changes after the new validator test is synced.

## Recovery and Queue Launch

Status update on 2026-07-08: `pro6000` SSH recovered.

The latest recovery bundle was rebuilt locally with 71 files and 712758 source
payload bytes, then launched with:

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
python scripts/remote_recovery_launch.py --start-queue --timeout 900
```

The launcher extracted the bundle, ran post-extract checks, confirmed the GPU
was idle, and started the next validation runbook through `nohup`:

```text
nohup_pid_started=341508
queue_log=artifacts/logs/next_validation_queue_2026-07-08.log
```

Immediate post-launch status:

- Parent process: `341508`, command `bash artifacts/runbooks/next_validation_queue_2026-07-08.sh`.
- First active training process: `341512`.
- GPU: RTX PRO 6000, about 1629 MiB used, about 52% utilization.
- First active job:
  `train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda`.
- Initial queue status: 0 of 56 expected new markers complete, as expected.

Ongoing monitoring is recorded in
`docs/records/2026-07-08_remote_queue_launch.md`.
