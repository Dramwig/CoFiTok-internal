# Matched factorization quality-regression diagnostic

Date: 2026-08-20

## Outcome

The exact terminal follow-up route
`run_matched_factorization_quality_regression_probe` now has a source-bound,
standing-authorization-backed execution implementation. The supervisor remains
CPU-only while the full-data matched 100K quality bridge is active. It can launch
the bounded GPU diagnostic only after the terminal quality result selects this
exact route, the terminal-system claim guard is complete, the exact 100K EMA
checkpoints and integrity sidecars are rebound, and five consecutive GPU-idle
polls succeed.

This work does not change or signal the active 100K trainer, does not modify the
formal checkout, and does not create the diagnostic output root while training
or terminal evidence is incomplete.

## Frozen protocol

The diagnostic is matched across CoFiTok K8 and `dense_identity`:

```text
training source: exact full ImageNet-256 matched 100K pair
checkpoint: checkpoint_step_00100000.pt
weights: EMA
seeds: 2029, 2039
images per method per seed: 64
batch size: 8
sampling: DDIM-100
teacher timesteps: 999, 900, 750, 500, 250, 100, 10
CFG: 1.5
teacher guidance: 1.0
guidance rescale: 0.0
precision: bf16
clipped x0: enabled
```

For each seed, the runbook performs a matched checkpoint evaluation, teacher-
forced comparison, free-sampling rollout, reconstruction rollout, high-frequency
diagnostic, component-energy diagnostic, and a replayed stability qualification.
The aggregate report distinguishes no reproduced regression, seed-sensitive
regression, and a regression repeated across both seeds.

## Source and authorization chain

The implementation adds:

```text
src/cofitok/generation/factorization_quality_regression.py
scripts/prepare_generation_factorization_quality_regression.py
scripts/build_generation_factorization_quality_regression_source_binding.py
scripts/build_generation_factorization_quality_regression_execution_authorization.py
scripts/verify_generation_factorization_quality_regression_execution_authorization.py
scripts/build_generation_factorization_quality_regression_report.py
scripts/run_generation_factorization_quality_regression_supervisor.py
artifacts/runbooks/generation_factorization_quality_regression_probe_v1.sh
```

The source binding replays the exposure-aware schema-v2 terminal follow-up
decision and terminal-system guard. It additionally reopens and reproduces the
terminal training-exposure report selected by that decision, then binds the
physical quality result, both training reports, both terminal checkpoint
evaluations, checkpoint bytes/SHA256, and adjacent integrity sidecars. The
supervisor accepts only
`reports/followup_experiment_decision_exposure_aware_v2.json`; the superseded
non-exposure-aware decision path is rejected. Checkpoint payload hashing remains
inside the existing trusted loader; the supervisor never hashes a live
checkpoint while training is active.

The execution authorization must reproduce from the exact preparation, source
binding, clean execution Git identity, and the existing standing authorization.
The runbook replays that authorization before either checkpoint is loaded.

The supervisor fails closed on route drift, source drift, Git drift, an existing
output or lock, malformed GPU process rows, missing standing authorization, or
fewer than five consecutive idle polls. It contains no process-signal path.
Before entering the wait loop it also writes an immutable deployment receipt
binding the exact clean checkout, supervisor source, runbook, regular-file Python
runtime, preparation, standing authorization, waiting sources, control targets,
timing contract, and non-authorizing safety boundary. Status snapshots bind that
receipt by bytes and SHA256.

## Claim boundary

The result is permanently non-authorizing. It may localize whether a terminal
matched quality regression is accompanied by repeatable CoFiTok-specific
teacher-forced, reconstruction, high-frequency, or component-energy instability.
It cannot by itself:

- causally attribute a regression to factorization losses or token layout;
- establish a generation-quality advantage;
- authorize a recipe change or follow-up training;
- authorize checkpoint promotion, full 300K, release, or sample publication.

Any loss/layout intervention requires a new source-compatible decision and a
separately bounded training stage.

## Immutable evidence byte preservation

The existing recovery-supersession v2 contract is byte-immutable at:

```text
configs/generation/diagnostics/quality_bridge_recovery_supersession_20260820_v2.json
bytes: 3,520
SHA256: d42b3bca92087062eb8e2999ac9f1cd3be729bf72b9e3e4cb823d5544c67e379
```

Windows checkout conversion had produced a CRLF working copy while preserving
the Git blob. An exact `-text` attribute now protects this one immutable config;
the working bytes were restored to LF without changing JSON semantics. The
locked SHA256 and its immutable-evidence test both pass.

## Verification before clean-commit rehearsal

```text
new focused tests: 10 passed
relevant regression group: passed
immutable recovery evidence test: passed
Python compile: passed
new runbook bash -n: passed
git diff --check: passed
```

The first dirty-tree full-suite pass was intentionally not treated as final
evidence: three control-plane tests rejected the uncommitted tracked checkout,
and one CLI subprocess imported the separate main-worktree editable install.
Final Windows and Linux full-suite results must be obtained from clean checkouts
using worktree-local/runtime-correct Python environments before deployment.

## GPU visibility boundary hardening

Pre-deployment review found that a supervisor launched with
`CUDA_VISIBLE_DEVICES=-1` would pass that value unchanged to the diagnostic
runbook. The parent process would remain correctly CPU-only, but the eventual
GPU diagnostic would be unable to see the device.

The supervisor now fails closed unless its own environment is exactly
`CUDA_VISIBLE_DEVICES=-1`. Only after the terminal route and source bindings are
valid and five consecutive `nvidia-smi` compute-process polls are idle does it
construct a child environment with exactly `CUDA_VISIBLE_DEVICES=0`. The runbook
independently rejects any other child value before loading a checkpoint. The
deployment receipt and every status snapshot record both visibility values and
the idle-poll transition condition. This change does not broaden the diagnostic
authorization boundary or permit training, promotion, full 300K, or release.
