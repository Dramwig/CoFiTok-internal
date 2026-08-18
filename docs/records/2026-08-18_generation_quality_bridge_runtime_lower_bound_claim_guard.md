# Quality-bridge runtime lower-bound claim guard

Date: 2026-08-18 (Asia/Shanghai)

## Finding

The deployed runtime claim guard already fails closed for the active full-data
quality bridge because the terminal GPU-contention history contains a
`16377.985275`-second observation gap. That evidence can never become complete
retroactively, so the live guard at revision `6fdeebe...` will publish
`runtime_cost_claims_observational_only` even if all later polls are healthy.

A separate future-safety gap nevertheless existed in the builder. Its direct
runtime decision depended only on exclusive GPU-observation coverage. The
CoFiTok run has a bound recovery adjustment of `560.6116147041321` seconds for
`200` optimizer steps and `12,800` images, and that adjustment is explicitly a
`physical_lower_bound_including_orphaned_recovery_compute`. Complete GPU
coverage cannot turn a recovery-derived lower bound into an exact elapsed-time
measurement. The previous builder could therefore have allowed direct
wall-clock, throughput, and cost-efficiency ranking in a hypothetical complete
coverage case even though one method's adjusted elapsed value remained only a
lower bound.

This is not a current live overclaim: the historical GPU-monitor gap already
forces observational-only treatment. It is a fail-closed regression risk for
future runs and replays.

## Future safeguard

Commit `0d1fa8db971c9b33b16f943f511efaf5e51b0ace` on
`analysis/generation-quality-bridge-runtime-claim-guard-v1` changes the direct
runtime decision to require both:

1. complete, continuous, exclusive terminal GPU-observation coverage; and
2. no method whose elapsed-time role is a recovery-derived physical lower
   bound.

The guard now publishes the two conditions separately:

- `exclusive_gpu_observation_coverage_verified`;
- `recovery_adjusted_elapsed_exact_for_both_methods`.

It also records `physical_lower_bound_methods`, requires a lower-bound label
whenever applicable, and keeps wall-clock, throughput, and cost-efficiency
ranking disabled if either condition fails. Equal wall-clock, GPU-hours, FLOPs,
quality advantage, release, and training authorization remain disallowed.

The source fairness report's nested recovery adjustment is now independently
validated for:

- exact boolean relationships among `required`, `provided`, and `applied`;
- finite seconds/hours and exact conversion;
- event/archive coverage;
- orphaned optimizer-step and image-count consistency with effective batch;
- zero-valued evidence when no adjustment is present;
- exact target-step binding.

## Validation

Local exact-HEAD validation:

```text
commit: 0d1fa8db971c9b33b16f943f511efaf5e51b0ace
tree:   02bed3345634098b598f534721bd3c336eb41058
related tests: 144 passed
Python compilation: pass
git diff --check: pass
tracked state: clean
```

The related suite covers the claim guard/waiter, runtime fairness waiter,
recovery-cost accounting, GPU-contention evidence, final comparison, completion
audit, and full post-evaluation runbook.

Linux CPU-only rehearsal:

```text
path: /tmp/cofitok-runtime-lower-bound-0d1fa8d-G7fIYW/CoFiTok-internal
commit: 0d1fa8db971c9b33b16f943f511efaf5e51b0ace
tree:   02bed3345634098b598f534721bd3c336eb41058
CUDA_VISIBLE_DEVICES: -1
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
related tests: 144 passed
Python compilation: pass
git diff --check: pass
tracked state: clean
```

The incremental rehearsal bundle was `12,171` bytes with SHA256
`8ce1d8b3f0a79acdaa930725bda10ce1e548f7e523d28cb94e2844c365a9bef4`.
It required deployed prerequisite `6fdeebe...`, verified before fetch, and was
removed locally and remotely after the isolated checkout was created.

## Live boundary

This safeguard is deliberately **not deployed** into the active waiter chain.
No running process, output, formal checkout, or training checkout was replaced,
signaled, paused, or restarted.

At `2026-08-18T12:23:15+08:00`:

```text
quality-bridge controller: PID 618821
CoFiTok trainer: PID 619775
CoFiTok progress: 44,000 / 100,000
dense progress: 0 / 100,000
runtime fairness waiter: PID 832803, waiting
deployed runtime claim guard: PID 912744, revision 6fdeebe..., waiting
terminal system guard: PID 965918, waiting
only GPU compute process: PID 619775, 85,286 MiB
unrelated GPU compute: absent
formal checkout: 1ebcc152..., tracked clean
```

The final runtime guard must still wait for both exact 100K training reports and
the terminal pair monitor. Given the immutable historical observation gap and
the recovery lower bound, the scientifically valid runtime conclusion remains
observational-only; no direct speed, throughput, cost-efficiency, equal-compute,
or generation-quality advantage may be claimed from this run.

