# 2026-08-19 Four-arm 5K conditioning-ranking held-out evaluation

## Purpose

The four-arm 5K training confirmation needs an evaluation that is independent
of the 1K probe evidence and bound to the exact terminal checkpoints. This
stage tests whether the class-conditioning ranking objective improves semantic
alignment for both CoFiTok and the matched dense identity baseline. It cannot
establish a CoFiTok-specific advantage and cannot authorize later training,
sampling, checkpoint promotion, full-scale runs, or release.

## Exact training source

- Training revision: `ed9463eb85ffb97545b5506e264123e46ff2fcfc`
- Training branch: `scale/generation-label-ranking-5k-training-confirmation-v1`
- Training output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_train5k_confirmation_v1`
- Runs: `control_cofitok`, `ranked_cofitok`,
  `control_dense_identity`, and `ranked_dense_identity`
- Required terminal checkpoint: `checkpoint_step_00005000.pt` with its exact
  integrity sidecar

The post-evaluation supervisor must replay the immutable training execution
receipt from the exact clean training checkout before it reserves a launch.
It also requires the completed four-arm training status and waits until every
training process under the exact output root has exited.

## Held-out protocol

- Device: CPU only (`CUDA_VISIBLE_DEVICES=-1`)
- Weights: EMA
- Validation images: 16
- Labels: `192..207`
- Wrong-label offset: `500`
- Timesteps: `100, 500, 900`
- Noise seed base: `304060`
- CPU threads: `2`
- Independent unit: one held-out validation image
- Eligible ranking timesteps: `500, 900`
- Descriptive-only timestep: `100`

The 1K probe used labels starting at `64` and noise seed `204060`, so the 5K
evaluation uses a disjoint held-out label/noise contract. Timestep rows are
aggregated per image before the paired one-sided sign test.

Each method passes only when the ranked arm has positive absolute improvement
against wrong and null conditions, positive improvement over its matched
control arm, significant per-image signs for both comparisons, and no more
than a `1.02x` correct-condition MSE ratio.

If both methods pass, the report may recommend only:

```text
prepare_separately_source_bound_posttraining_5k_sampling_confirmation
```

Asymmetric or failed evidence rejects the shared repair or returns to revising
the training-time semantic-alignment objective. No decision from this stage
authorizes GPU work automatically.

## Implementation

- `scripts/build_generation_conditioning_ranking_training_confirmation_posteval.py`
  validates preparation/config identities, exact training status, physical
  checkpoint and integrity identities, training reports, progress audits,
  schedule audits, CPU sensitivity reports, held-out sample identities, and
  separate clean evaluator Git provenance.
- `scripts/build_generation_conditioning_ranking_probe_posteval.py` now accepts
  an explicit sensitivity request/checkpoint contract and, when available,
  verifies the exact integrity-sidecar identity as well as the checkpoint.
- `artifacts/runbooks/generation_conditioning_ranking_four_arm_train5k_heldout_evaluation_v1.sh`
  is CPU-only, low-priority, source-bound, and refuses to create its output root
  while an exact four-arm trainer is active.
- `scripts/run_generation_conditioning_ranking_training_confirmation_posteval_supervisor.py`
  waits without creating the post-evaluation root, never signals processes,
  waits for and content-addresses the future training execution receipt,
  replays it from the exact training checkout, writes a one-shot launch
  receipt, and forbids automatic relaunch.

## Local validation

- New builder/supervisor plus the earlier 1K post-evaluation suite: `21 passed`
- All `test_generation_conditioning*.py` tests: `84 passed`
- Full repository suite excluding the four sibling-`paper/` layout tests:
  passed with BLAS/OMP/MKL threads limited to one
- New builder and supervisor `--help` entrypoints: passed with
  `PYTHONPATH=.:src`
- Python compilation: passed
- `git diff --check`: passed

The Linux CUDA-hidden rehearsal and deployment of the waiting CPU supervisor
remain pending. The active full-data quality bridge dense trainer is not
modified or contended by this work.
