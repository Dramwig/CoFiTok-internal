# Full-training runtime selection freeze (2026-07-13)

## Finding

The full ImageNet-256 runbook invoked the runtime selector on every launch. Each
individual benchmark cache was provenance-checked, but the selector still
recomputed and rewrote `runtime_selection.json`. After a 50K or later recovery
checkpoint existed, a changed candidate list, benchmark horizon, report, or
measurement could select a different microbatch/accumulation pair. Exact resume
would then correctly reject the config drift, leaving the run unable to resume.

## Contract

`select_generation_training_runtime.py` now receives both formal run
directories and distinguishes two states:

- before training state exists, it may run the real checkpoint-free benchmark
  and atomically publish the selection;
- after either run directory contains any state, it cannot benchmark or rewrite
  the report and may only validate and return the frozen selection.

The selection lock binds the two formal run paths, ordered `16x4/32x2/64x1`
candidates, effective batch 64, 8-step/2-warmup benchmark, 300K training horizon,
90% memory ceiling, clean `scale/generative-system` revision, both config file
SHA256 values, and benchmark root. Reuse additionally verifies the current
canonical runtime-environment SHA, every completed benchmark's resolved config,
Git revision, dataset identity, checkpoint-free role, and a deterministic
recomputation of the selected candidate.

Any existing training state without a selection, any environment/config/branch
or candidate drift, and any internally inconsistent embedded benchmark fail
before a benchmark subprocess or training model load.

## Terminal evidence

`audit_large_scale_generation_completion.py` requires the same lock at final
completion. It reconstructs each candidate benchmark config from the final
CoFiTok or dense training config while changing only microbatch and accumulation,
then verifies the benchmark horizon, effective batch, clean Git state, dataset,
environment, and `checkpoint_written=false` role.

## Verification

Focused tests cover pre-training selection, state-triggered read-only reuse,
missing-selection failure, candidate/contract/environment drift, benchmark
config tampering, and terminal lock rejection.

- implementation commit: `3bd4da72d3e58a9787a88db3b29245d6cc7dd615`;
- complete local suite: `549/549` passed;
- complete isolated Linux suite in the real sibling-`paper/` layout: `549/549`
  passed;
- tracked Linux runbook syntax audit: `40/40` passed;
- formal remote target-added-path scan: `149` paths, `0` conflicts;
- pinned `781a01444fddbf0d48a427ba58bdeed50167b5be` to implementation
  bundle: `424,568` bytes, SHA256
  `9d37c9a7917c3fd93a3f8f1688f9960c4071f7520d599a8b6906ba96691f467f`;
- formal remote HEAD remained pinned at `781a01444fddbf0d48a427ba58bdeed50167b5be`
  with a clean tracked worktree before and after verification.
