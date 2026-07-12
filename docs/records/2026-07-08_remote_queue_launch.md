# Remote Queue Launch Record

Date: 2026-07-08

## Summary

`pro6000` SSH recovered and the next validation queue was launched through
`scripts/remote_recovery_launch.py --start-queue --timeout 900`.

The launcher streamed the latest recovery bundle, extracted it under
`/root/autodl-tmp/CoFiTok`, ran the post-extract checks, verified the GPU idle
guard, and started the runbook with `nohup`.

## Recovery Bundle

- Bundle: `artifacts/recovery/remote_recovery_2026-07-08.tar.gz`
- Manifest: `artifacts/recovery/remote_recovery_2026-07-08.tar.gz.manifest.json`
- The manifest is the source of truth for file count and source payload bytes.
- Included new publication-readiness artifacts:
  - `scripts/validate_publication_readiness.py`
  - `tests/test_validate_publication_readiness.py`
  - `artifacts/reports/publication_readiness_2026-07-08.json`
  - `docs/records/2026-07-08_publication_readiness_gap.md`
- Included structured remote monitor artifacts:
  - `scripts/remote_queue_status.py`
  - `tests/test_remote_queue_status.py`

## Post-Extract Checks

The remote post-extract checks completed before launch:

- Queue status inspector ran and reported 0 of 56 expected new markers complete,
  as expected before the queue starts.
- Dataset condition gate passed.
- Summary consistency gate passed.
- Idea requirements gate passed with `mvp_ok_publication_pending`.
- MVP evidence gate passed.
- Publication-readiness gate wrote the JSON/Markdown gap report and returned
  `not_ready`, as expected before the queued publication-scale batch finishes.
- Dynamic synthesis contract passed.
- Remote pytest completed successfully.

## Launch State

- Runbook: `artifacts/runbooks/next_validation_queue_2026-07-08.sh`
- Log: `artifacts/logs/next_validation_queue_2026-07-08.log`
- Launcher output: `nohup_pid_started=341508`
- Initial training child process: PID `341512`
- Initial GPU state after launch: about 1629 MiB used, utilization around 52%.
- First active job:
  `train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda`

## First Progress Check

After launch, the first queue item completed the full sequence:

- Training completed at step 20000 and wrote `report.json`,
  `checkpoint_final.pt`, and `prefix_final.png`.
- 1024-image quality evaluation completed with LPIPS Alex and Inception enabled.
- 2048-image DDIM50 sampling completed and wrote `sample_report.json`,
  `samples_prefix_8.png`, and the `samples_prefix_8/` image directory.
- Generated-quality evaluation completed against 8192 real images.

The runbook then advanced automatically to:

```text
train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda
```

At that point the parent runbook process was still alive as PID `341508`, with
the active training child process using the GPU.

## Second Progress Check

The structured remote poller
`scripts/remote_queue_status.py --allow-unreachable` was added so transient SSH
timeouts do not erase queue state. The latest successful poll returned:

- Queue status: `running`
- Completed markers: 11 of 56
- Completed run labels:
  - `tiny_epsilononly_20k_seed2`
  - `tiny_k8_light_20k_seed2`
- Completed by type:
  - train: 2 of 10
  - quality: 2 of 10
  - order eval: 3 of 24
  - sampling: 2 of 6
  - generated quality: 2 of 6
- Current first incomplete label: `imagenet_hf_epsilononly_20k_seed2`
- Active job:
  `train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda`
- GPU snapshot: about 1629 MiB used, about 62% utilization.

## Third Progress Check

A later structured poll showed the runbook had finished the ImageNet-64 HF
epsilon-only seed-2 20k training stage and advanced into its 1024-image quality
evaluation:

- Queue status: `running`
- Completed markers: 12 of 56
- Completed by type:
  - train: 3 of 10
  - quality: 2 of 10
  - order eval: 3 of 24
  - sampling: 2 of 6
  - generated quality: 2 of 6
- Current first incomplete label: `imagenet_hf_epsilononly_20k_seed2`
- Active job:
  `evaluate_quality.py` for
  `quality_imagenet_hf_epsilononly_20k_seed2_1024_t500_lpips_inception_2026-07-08`
- GPU snapshot: about 2235 MiB used by the quality job.

## Fourth Progress Check

The next poll showed the ImageNet-64 HF epsilon-only seed-2 label had completed
its full train/quality/sample/generated-quality sequence:

- Queue status: `running`
- Completed markers: 15 of 56
- Completed run labels:
  - `tiny_epsilononly_20k_seed2`
  - `tiny_k8_light_20k_seed2`
  - `imagenet_hf_epsilononly_20k_seed2`
- Completed by type:
  - train: 3 of 10
  - quality: 3 of 10
  - order eval: 3 of 24
  - sampling: 3 of 6
  - generated quality: 3 of 6
- Current first incomplete label: `imagenet_hf_k8_light_20k_seed2`
- Active job:
  `train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda`

The completed lightweight reports and grid images for the first three labels
were copied back locally under `artifacts/reports/`; checkpoints and full
2048-image sample directories were intentionally not copied. A compact local
progress summary was written to:

```text
artifacts/reports/remote_queue_progress_2026-07-08/completed_seed2_summary.json
artifacts/reports/remote_queue_progress_2026-07-08/completed_seed2_summary.md
```

Current seed-2 interpretation:

- Tiny ImageNet K8 light CoFiTok improves 1024-image prefix clean-MSE AUC
  compared with Tiny epsilon-only (`6.587` vs `7.958`) while final clean MSE and
  LPIPS remain slightly worse.
- Tiny generated-sample Inception Frechet is very close and slightly better for
  K8 light CoFiTok in this seed (`274.75` vs `276.93`), but this is still not a
  robust generation-quality win.
- Restricted `S_k` diagnostics remain clean: zero-token energy ratio is `0.0`,
  shuffled final MSE is badly mismatched, and tail tokens are active.
- The Tiny and ImageNet seed-2 order diagnostics need a metric distinction:
  denoise-path AUC is healthy, with ordered much lower than random/reverse
  (`0.0398/0.1885/0.6895` on Tiny and `0.0365/0.1846/0.6514` on ImageNet HF).
  Clean-MSE AUC is lower for reverse/random because the learned energy is still
  tail-heavy, so this should be treated as an energy-ordering limitation rather
  than a failure of the denoise-path order diagnostic.

## Fifth Progress Check

The following poll showed the fourth main seed-2 label had also completed:

- Queue status: `running`
- Completed markers: 22 of 56
- Completed run labels:
  - `tiny_epsilononly_20k_seed2`
  - `tiny_k8_light_20k_seed2`
  - `imagenet_hf_epsilononly_20k_seed2`
  - `imagenet_hf_k8_light_20k_seed2`
- Completed by type:
  - train: 4 of 10
  - quality: 4 of 10
  - order eval: 6 of 24
  - sampling: 4 of 6
  - generated quality: 4 of 6
- Current first incomplete label: `tiny_k8_light_multiscale_10k`
- Active job:
  `train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda`

The ImageNet HF K8 light seed-2 lightweight reports and grid images were copied
back locally, and the compact progress summary was refreshed with
`scripts/summarize_remote_queue_progress.py`. The updated summary now covers all
four main seed-2 labels.

Do not regenerate the official `artifacts/reports/summary_2026-07-08/` from the
partial local raw-report mirror. The local mirror intentionally contains only
selected newly completed reports, while the official summary still depends on
older remote-only rows. The official summary should be regenerated on `pro6000`
after the runbook finishes, or restored from the remote copy if a partial local
rescan overwrites it. Partial local progress belongs in
`artifacts/reports/remote_queue_progress_2026-07-08/`.

## Sixth Progress Check

The next poll showed the Tiny multiscale stronger-backbone pilot had completed
its full sequence, and the runbook had advanced to the ImageNet HF multiscale
pilot:

- Queue status: `running`
- Completed markers: 29 of 56
- Completed run labels:
  - `tiny_epsilononly_20k_seed2`
  - `tiny_k8_light_20k_seed2`
  - `imagenet_hf_epsilononly_20k_seed2`
  - `imagenet_hf_k8_light_20k_seed2`
  - `tiny_k8_light_multiscale_10k`
- Completed by type:
  - train: 5 of 10
  - quality: 5 of 10
  - order eval: 9 of 24
  - sampling: 5 of 6
  - generated quality: 5 of 6
- Current first incomplete label: `imagenet_hf_k8_light_multiscale_10k`
- Active job:
  `train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda`

The Tiny multiscale lightweight reports and grid images were copied back
locally and added to the compact progress summary. Early interpretation:

- Tiny multiscale 10k improves over Tiny K8 light 20k seed-2 on endpoint
  reconstruction quality (`0.1095` vs `0.1306` final clean MSE), LPIPS
  (`0.5077` vs `0.5843`), and generated Inception Frechet (`174.48` vs
  `274.75`).
- Prefix clean-MSE AUC is similar (`6.651` vs `6.587`), so the main gain is
  endpoint/generation quality rather than a stronger prefix curve.
- Ordered denoise-path AUC remains lower than random/reverse
  (`0.0429/0.2252/0.6555`), and restricted `S_k` zero-token energy ratio remains
  `0.0`.

## Seventh Progress Check

The next poll showed the ImageNet HF multiscale stronger-backbone pilot had
also completed, and the runbook had advanced to the no-prefix loss-ablation
block:

- Queue status: `running`
- Completed markers: 36 of 56
- Completed run labels:
  - `tiny_epsilononly_20k_seed2`
  - `tiny_k8_light_20k_seed2`
  - `imagenet_hf_epsilononly_20k_seed2`
  - `imagenet_hf_k8_light_20k_seed2`
  - `tiny_k8_light_multiscale_10k`
  - `imagenet_hf_k8_light_multiscale_10k`
- Completed by type:
  - train: 6 of 10
  - quality: 6 of 10
  - order eval: 12 of 24
  - sampling: 6 of 6
  - generated quality: 6 of 6
- Current first incomplete label: `tiny_k8_light_nopathprefix_10k`
- Active job:
  `train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda`

The ImageNet HF multiscale lightweight reports and grid images were copied back
locally and added to the compact progress summary. Early interpretation:

- ImageNet HF multiscale 10k improves over ImageNet HF K8 light 20k seed-2 on
  final clean MSE (`0.0971` vs `0.1163`), LPIPS (`0.4814` vs `0.5593`), and
  generated Inception Frechet (`189.05` vs `274.03`).
- Prefix clean-MSE AUC is similar (`6.670` vs `6.608`), matching the Tiny
  pattern: multiscale improves endpoint/generation quality more than prefix
  curve area.
- Ordered denoise-path AUC remains much lower than random/reverse
  (`0.0358/0.1951/0.6176`), and restricted `S_k` zero-token energy ratio remains
  `0.0`.

## Eighth Progress Check

The next poll showed the Tiny no-prefix loss-ablation pilot had completed, and
the runbook had advanced to the Tiny clean-monotonic ablation:

- Queue status: `running`
- Completed markers: 41 of 56
- Completed run labels include `tiny_k8_light_nopathprefix_10k`.
- Completed by type:
  - train: 7 of 10
  - quality: 7 of 10
  - order eval: 15 of 24
  - sampling: 6 of 6
  - generated quality: 6 of 6
- Current first incomplete label: `tiny_k8_light_cleanmono_10k`
- Active job:
  `train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda`

The Tiny no-prefix lightweight reports were copied back locally and added to
the compact progress summary. Early interpretation against the Tiny multiscale
pilot:

- Removing denoise-path prefix supervision worsens final clean MSE (`0.1348` vs
  `0.1095`), LPIPS (`0.5568` vs `0.5077`), and prefix clean-MSE AUC (`7.012` vs
  `6.651`).
- Ordered denoise-path AUC also worsens (`0.0705` vs `0.0429`), while
  random/reverse remain worse than ordered. This supports keeping explicit
  denoise-path prefix supervision.
- Tail energy is more concentrated (`0.9876` vs `0.9766`), consistent with the
  earlier energy-ordering limitation.

## Ninth Progress Check

The next poll showed the Tiny clean-monotonic loss-ablation pilot had completed,
and the runbook had advanced to the ImageNet HF no-prefix ablation:

- Queue status: `running`
- Completed markers: 46 of 56
- Completed run labels include `tiny_k8_light_cleanmono_10k`.
- Completed by type:
  - train: 8 of 10
  - quality: 8 of 10
  - order eval: 18 of 24
  - sampling: 6 of 6
  - generated quality: 6 of 6
- Current first incomplete label: `imagenet_hf_k8_light_nopathprefix_10k`
- Active job:
  `train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda`

The Tiny clean-monotonic lightweight reports were copied back locally and added
to the compact progress summary. Early interpretation against Tiny multiscale
and no-prefix:

- Clean-monotonic improves prefix clean-MSE AUC over no-prefix (`6.736` vs
  `7.012`) and improves ordered denoise-path AUC (`0.0492` vs `0.0705`), but it
  remains behind the multiscale baseline (`6.651`, `0.0429`).
- Clean-monotonic final MSE and LPIPS are close to no-prefix and worse than the
  multiscale baseline (`0.1346`/`0.5687` vs `0.1095`/`0.5077`).
- This suggests clean-monotonic helps order/path behavior but does not replace
  denoise-path prefix supervision or the stronger multiscale predictor.

## Tenth Progress Check

The next poll showed the ImageNet HF no-prefix ablation had completed, and the
runbook had advanced to the final ImageNet HF clean-monotonic ablation:

- Queue status: `running`
- Completed markers: 51 of 56
- Completed run labels include `imagenet_hf_k8_light_nopathprefix_10k`.
- Completed by type:
  - train: 9 of 10
  - quality: 9 of 10
  - order eval: 21 of 24
  - sampling: 6 of 6
  - generated quality: 6 of 6
- Current first incomplete label: `imagenet_hf_k8_light_cleanmono_10k`
- Active job:
  `train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda`

The ImageNet HF no-prefix lightweight reports were copied back locally and
added to the compact progress summary. Early interpretation against ImageNet HF
multiscale:

- Removing denoise-path prefix supervision worsens final clean MSE (`0.1171` vs
  `0.0971`), LPIPS (`0.5377` vs `0.4814`), and prefix clean-MSE AUC (`6.965` vs
  `6.670`).
- Ordered denoise-path AUC worsens (`0.0577` vs `0.0358`), while random/reverse
  remain worse than ordered. This mirrors the Tiny no-prefix ablation and
  strengthens the case for explicit denoise-path prefix supervision.
- Tail energy becomes more concentrated (`0.9868` vs `0.9766`).

## Final Completion Check

The queued experimental markers reached completion on `pro6000`:

- Queue markers: 56 of 56 complete.
- Completed by type:
  - train: 10 of 10
  - quality: 10 of 10
  - order eval: 24 of 24
  - sampling: 6 of 6
  - generated quality: 6 of 6

The first post-queue gate run stopped after summarization because the local
report documents still listed the pre-queue summary counts. The docs were
updated to the final summary counts:

```text
train: 99
order_eval: 62
quality: 45
sampling: 22
generated_quality: 19
```

One stricter publication-readiness check then exposed a real reporting gap:
the two multiscale quality reports were named as 1024-image evaluations, but
the actual `image_count` was 768. Root cause: the runbook used
`QUALITY_BATCHES=32`, while the multiscale configs use `batch_size=24`; this
only covers `32 * 24 = 768` images. The queue generator and runbook were
updated to default `QUALITY_BATCHES=64`, and the two reports were overwritten
on `pro6000` with:

```bash
python scripts/evaluate_quality.py \
  --config configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda_2026-07-08/checkpoint_final.pt \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/quality_tiny_k8_light_multiscale_10k_1024_t500_lpips_inception_2026-07-08 \
  --split val --max-batches 64 --max-images 1024 --timestep 500 \
  --enable-lpips --enable-inception-fid

python scripts/evaluate_quality.py \
  --config configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda_2026-07-08/checkpoint_final.pt \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/quality_imagenet_hf_k8_light_multiscale_10k_1024_t500_lpips_inception_2026-07-08 \
  --split val --max-batches 64 --max-images 1024 --timestep 500 \
  --enable-lpips --enable-inception-fid
```

Both corrected reports now have `image_count=1024` and `batch_count=43`.
After regenerating `artifacts/reports/summary_2026-07-08`, all final gates
passed on `pro6000` and locally:

- `validate_dataset_conditions.py --require-ok`: ok
- `validate_summary_consistency.py`: ok
- `validate_idea_requirements.py --require-mvp-ok`: `ready`
- `validate_mvp_evidence.py`: ok
- `validate_publication_readiness.py --require-ready`: `ready`
- `validate_synthesis_contract.py --config-glob 'configs/*.json'`: ok
- `pytest -q`: pass on remote and local

Final interpretation: the publication-readiness gate is now satisfied for the
current scoped claims. The main remaining caveat is claim scope, not missing
queue evidence: generation quality should still be described as small-model
DDIM/Frechet evidence, not as a full official ImageNet FID win.

## Monitor Commands

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
tail -80 artifacts/logs/next_validation_queue_2026-07-08.log
ps -p 341508 -o pid,ppid,stat,etime,cmd
nvidia-smi
PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python \
  scripts/inspect_next_validation_queue.py \
  --checkpoint-root /root/autodl-tmp/CoFiTok/checkpoints \
  --output artifacts/reports/next_validation_queue_status_manual_2026-07-08.json
```

Local structured poll:

```powershell
cd C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal
python scripts/remote_queue_status.py `
  --allow-unreachable `
  --output-json artifacts/reports/remote_queue_status_latest_2026-07-08.json
```

## Completion Criteria

The queue is complete when `inspect_next_validation_queue.py` reports 56 of 56
expected markers complete and the runbook reaches the final gates:

- `validate_dataset_conditions.py --require-ok`
- `validate_summary_consistency.py`
- `validate_idea_requirements.py --require-mvp-ok`
- `validate_mvp_evidence.py`
- `validate_publication_readiness.py --require-ready`
- `validate_synthesis_contract.py`
- `pytest -q`
