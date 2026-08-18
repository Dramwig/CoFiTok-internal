# Generation checkpoint reproducibility revalidation

Date: 2026-08-15

## Outcome

The exact-resume, checkpoint-integrity, metrics-reconciliation, milestone
retention, runtime-environment, and training-completion boundaries were
revalidated on Windows and CUDA-hidden Linux. Of 76 collected tests, 74 passed
and the same two CUDA-only RNG-remap tests were explicitly skipped in each
environment. There were no assertion failures in the authoritative runs and no
source change was required.

This is CPU contract evidence. It does not replace the physical integrity and
exact-resume verification of future quality-bridge/capacity/full checkpoints,
and it does not claim that CUDA RNG remapping was retested while the GPU was
owned by another project.

## Covered checkpoint boundary

The bounded suite verifies:

- segmented training restores model, EMA, optimizer, scheduler, scaler, Python,
  NumPy, torch CPU RNG, sampler order/consumption, validation iterator position,
  and metrics state to the same result as uninterrupted training;
- resume rejects resolved-config, runtime-environment, Git, dataset, pointer,
  checkpoint, integrity-sidecar, payload-step, and payload-format drift before
  restoring state;
- metrics rows after the resume checkpoint are content-addressed as orphan
  history, canonical JSONL is atomically rewritten, and the retained stream is
  strictly increasing without duplicated scheduled validation events;
- atomic checkpoint publication produces a SHA/bytes/step/format sidecar and a
  matching `latest.json`; legacy backfill preserves weight bytes and refuses an
  unverifiable payload;
- pruning removes only checkpoint/sidecar pairs allowed by retention policy and
  preserves the protected 50K/100K/200K/300K milestones plus current recovery
  states;
- completion validation requires exact steps/images, finite metrics, the bound
  latest checkpoint and integrity metadata, clean Git/runtime provenance, and
  valid pair evidence rather than process exit alone;
- canonical runtime fingerprints capture the required Python/package/torch/
  CUDA/cuDNN/driver/GPU/backend/environment/project-lock fields and reject
  mutation.

## Exact test scope

```text
tests/test_generation_checkpoint_retention.py       8
tests/test_generation_exact_resume.py               24
tests/test_generation_metrics_resume.py             13
tests/test_generation_system.py                     20
tests/test_generation_training_completion.py         8
tests/test_runtime_environment.py                    3
```

The local Windows run completed in 55.3 seconds with `74 passed, 2 skipped`.
The skips were exactly:

- `test_rng_restore_accepts_states_remapped_to_cuda`;
- `test_stateful_sampler_restores_rng_state_loaded_on_cuda`.

Both are intentionally guarded by `torch.cuda.is_available()` and could not be
run without violating the unrelated-project GPU exclusion.

## Authoritative Linux result

- checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-rehearsal-47e59a0/CoFiTok-internal`;
- revision: `47e59a0ce926c3ac1fdaa4860de84d2f27e39f76`;
- tree: `3a99993b66db675f262ed2e04298e1d4d64411cf`;
- tracked checkout state: clean;
- Python: `3.10.20`;
- CUDA policy: `CUDA_VISIBLE_DEVICES=''`;
- result: `76 collected / 74 passed / 2 skipped / 0 failures / 0 errors`;
- JUnit time: `577.603` seconds.

The authoritative persistent JUnit report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/checkpoint_reproducibility_revalidation_2026-08-15/pytest_linux_cuda_hidden_47e59a0_rerun.xml`;
- bytes: `14,245`;
- SHA256:
  `6e20ed05ebfdf1ebeb46bd3e41ec3c89cc85a4ea6b5c10308f389dfd38ed8948`;
- file mode: `0644`.

## Preserved interrupted evidence

The first Linux invocation used a 360-second client timeout. The remote pytest
continued, but when it next flushed terminal output the closed SSH pipe caused
an internal `BrokenPipeError`. That run stopped after 26 testcases and is not
counted as a test pass. Its JUnit is deliberately retained rather than replaced:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/checkpoint_reproducibility_revalidation_2026-08-15/pytest_linux_cuda_hidden_47e59a0.xml`;
- bytes: `10,370`;
- SHA256:
  `4344bc317cd1a2f7ecaae0e04308fe3a4bd25671583f9d07d0f1f3ed8ced73e2`;
- reported result: `27` entries including one pytest-internal error, JUnit time
  `477.300` seconds.

The full rerun used a separate filename and a 15-minute client limit. It
completed normally with exit code zero, proving that the first error came from
the transport boundary rather than a checkpoint assertion.

## Preserved production state

All fixtures used temporary CPU-only runs. No production checkpoint, metrics
stream, latest pointer, sidecar, or training report was modified. The formal
checkout remained untouched, and the quality bridge continued to wait for the
unrelated FieldScope GPU process.
