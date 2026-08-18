# Quality-bridge runtime claim guard waiter

Date: 2026-08-18 (Asia/Shanghai)

## Finding

The deployed full-data quality-bridge runtime/compute fairness waiter correctly
binds the shared resolved runtime, matched optimizer steps/images, parameter
counts, terminal training reports, and physical recovery compute. Its standalone
schema-v1 claim boundary nevertheless contains
`observed_pair_runtime_parity_established=true`. That wording can be read more
broadly than the evidence: it proves configuration/steps/images parity, not
that wall-clock or throughput are directly comparable.

The automatic final comparison and completion audit were already safe. Both
reopen the terminal pair monitor through
`validate_gpu_contention_evidence`; their wall-clock, throughput, and
cost-efficiency policies are derived from continuous GPU-observation coverage,
not from the standalone fairness report. A repository-wide consumer trace found
no programmatic path from `runtime_compute_fairness/final_report.json` into the
paper/comparison builders. The remaining risk was manual citation of the raw
standalone field.

At deployment time the live pair monitor retained a maximum observation gap of
`16377.985275` seconds, so direct runtime comparison was already disallowed as
`incomplete_gpu_observation_coverage`. Historical maximum-gap evidence cannot
be repaired by later polls.

## Guard

Commit `6fdeebe7c0ca6b2f7fbe593b3690f14d36f98e46` adds:

- `scripts/build_generation_runtime_compute_claim_guard.py`;
- `scripts/run_generation_quality_bridge_runtime_claim_guard_waiter.py`;
- `tests/test_generation_runtime_compute_claim_guard.py`.

The builder cryptographically binds the terminal runtime/compute fairness
report and terminal pair monitor. It recomputes recovery-adjusted elapsed time,
throughput, parameter parity, runtime parity, and descriptive differences, then
replays `validate_gpu_contention_evidence` before publishing claim policy.

The guard always separates these statements:

- same resolved runtime configuration, optimizer steps, and training images;
- recovery-adjusted physical cost accounting;
- eligibility for direct wall-clock/throughput/cost ranking.

The first two can be verified even when GPU observation coverage is incomplete.
The third is allowed only with complete, continuous, exclusive terminal GPU
coverage. Otherwise elapsed time and throughput are labeled observational
physical lower bounds. Equal wall-clock, GPU-hours, FLOPs, a preregistered speed
advantage, quality advantage, release, and broad superiority remain disallowed.

## Validation

Local validation:

- focused guard/waiter suite: `8 passed`;
- related runtime-fairness, GPU-contention, comparison, completion-audit, and
  full-posteval suite: `131 passed`;
- Python compilation and `git diff --check`: pass;
- full repository collection: `1114` tests;
- the full run reached 100% with only four failures, all existing
  `test_aaai27_experiment_structure.py` path-layout failures caused by this
  independent worktree resolving the external paper tree as `C:/paper/...`.
  All tests in the changed runtime/claim paths passed.

The exact Linux checkout under
`/tmp/cofitok-runtime-claim-guard-rehearsal-6fdeebe/CoFiTok-internal` passed the
same `131` related tests with CUDA hidden and one OMP/MKL thread. The focused
suite passed `8/8`, Python compilation and both CLI help paths passed with
`PYTHONPATH=.:src`, `git diff --check` passed, and tracked state remained clean.

## New-server bundle handling

The first `14,407`-byte bundle required local prerequisite `7b19e96...`, which
the newly cloned formal server repository did not contain. `git bundle verify`
failed closed before any fetch or checkout mutation. The formal checkout stayed
at `1ebcc152...`.

The deployment bundle was rebuilt from prerequisites actually present in the
new server repository:

```text
bytes: 40,214,926
sha256: c0132b144a95df42b3820dbd08ae37d553909994eb8d200dfa69a789c117dd0e
advertised head: 6fdeebe7c0ca6b2f7fbe593b3690f14d36f98e46
prerequisites:
  1ebcc15210e63a776a2ba448481cbd8bb94a4066
  58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

The bundle verified against the formal repository without moving its HEAD.

## Deployment

Persistent control checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-runtime-claim-guard-6fdeebe/CoFiTok-internal

revision: 6fdeebe7c0ca6b2f7fbe593b3690f14d36f98e46
tree: bc41a4df45b83cef3c81cc63aad1668c49b85a75
branch: analysis/generation-quality-bridge-runtime-claim-guard-v1
tracked state: clean
git fsck: pass
```

The CPU-only waiter launched as PID `912744`. Its output root is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/
reports/runtime_compute_claim_guard_v1
```

Immutable deployment receipt:

```text
bytes: 3,186
sha256: 5909fda4073088aa6b8b52a6eaf5c1ca2c8481a1d9e11b837f1a10cf30a53098
```

Initial authoritative state:

```text
status: waiting
phase: source
detail: waiting_for_runtime_fairness_source
```

The waiter binds source runtime-fairness PID `832803`, its immutable deployment
receipt SHA256 `f1e60458...`, the exact training revision `cf0e5faa...`, and the
terminal pair monitor. It neither replaces nor signals the source waiter,
monitor, trainer, or unrelated processes.

At deployment, trainer PID `619775` remained the only GPU compute process at
`85,286 MiB`; CoFiTok advanced to step `38,900` and dense had not started. The
formal checkout remained `1ebcc152...` on `scale/generative-system` with zero
tracked changes.

Machine-readable deployment evidence:

```text
artifacts/reports/generation/
quality_bridge_runtime_claim_guard_waiter_deployment_2026-08-18.json
```

The final guard does not yet exist. It will be written only after both exact
100K training reports, the source fairness audit, and the terminal pair monitor
are stable and mutually consistent.
