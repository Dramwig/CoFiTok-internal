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

## Linux regression and waiter deployment

The exact evaluator revision was deployed in the independent clean checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/conditioning-heldout5k-evaluation-90f7206/CoFiTok-internal
revision: 90f7206dd6d65b398a05222821cd6bc4cbf95efd
tree: 987dfcd382f5b7c5dbe52435e42062d0739d73f7
branch: scale/generation-label-ranking-5k-heldout-evaluation-v1
```

The full Linux regression used an isolated real
`CoFiTok/CoFiTok-internal + sibling paper/` layout, exact branch identity,
`CUDA_VISIBLE_DEVICES=-1`, one BLAS/OMP/MKL/NUMEXPR thread, `nice -n 19`, and
idle-class IO priority. Pytest collected `1,181` tests and completed with
`1,179 passed, 2 skipped`; the exact checkout remained clean. An initial
detached-HEAD rehearsal correctly failed two existing inference-export tests
because release provenance requires a named Git branch. Repeating at the exact
required branch passed both targeted tests and the complete suite without a
source change.

A bounded negative supervisor rehearsal captured:

```text
status: waiting
detail: waiting_for_exact_5k_training_status
child_pid: null
```

It then timed out fail-closed. The future four-arm training root, held-out
post-evaluation root, launch receipt, and training-execution replay remained
absent. The GPU process snapshot was unchanged and contained only the active
quality-bridge dense trainer.

The real CPU-only supervisor is now deployed at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/conditioning_ranking_four_arm_train5k_heldout_evaluation_v1/90f7206dd6d65b398a05222821cd6bc4cbf95efd
PID: 305233
runbook SHA256: 975dba32b8933bd0700201261cc72e03d5a37b395ffb83de9ef07062aa6c1659
```

At `2026-08-19T05:37:40+08:00` its authoritative status was `waiting`, detail
`waiting_for_exact_5k_training_status`, and `child_pid=null`. The process had
niceness `19`, idle IO scheduling, CUDA hidden, and all numerical-library
thread limits set to one. It created no training or post-evaluation output
root and added no GPU process.

At the same observation, the active full-data matched quality bridge remained
healthy at revision `cf0e5faa94bf4ab38d947b921935b3b765b5537a`:

```text
status: running
stage: dense_identity_training
issues: []
CoFiTok segment: 50,000 complete
dense tail: 16,900 / 50,000
dense samples_seen: 1,081,600
quality_bridge_result.json: absent
```

The active dense trainer remained the sole GPU compute process. This deployment
does not authorize sampling, training, promotion, full 100K/300K work, or
release; it only waits for the exact source-bound 5K training evidence and may
launch the CPU held-out evaluation once.
