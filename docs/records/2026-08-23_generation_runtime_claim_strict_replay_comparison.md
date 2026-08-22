# Runtime claim strict-replay comparison (2026-08-23)

## Purpose

The post-restart canonical runtime-claim waiter uses control revision
`6fdeebe7c0ca6b2f7fbe593b3690f14d36f98e46`. Its builder predates the rule
that any recovery-adjusted elapsed value derived from orphaned metrics remains a
physical lower bound and therefore disables wall-clock, throughput, and
cost-efficiency ranking even when GPU-observation coverage is complete.

The independently replayed strict guard is pinned to:

- revision `0d1fa8db971c9b33b16f943f511efaf5e51b0ace`;
- tree `02bed3345634098b598f534721bd3c336eb41058`;
- branch `analysis/generation-quality-bridge-runtime-claim-guard-rebind-v2`.

The active pair already has incomplete historical GPU-observation coverage, so
both builders are expected to produce an observational-only runtime decision.
That coincidence is not enough to trust the old builder: the strict replay must
independently validate the complete recovery-adjustment contract and both
physical lower-bound roles.

## Implementation

`scripts/build_generation_runtime_claim_guard_comparison.py` binds and reopens:

1. the canonical waiter status and guard;
2. the independently replayed strict waiter status and guard;
3. the non-authorizing strict-replay wrapper deployment receipt.

It requires the strict guard to prove that both `cofitok` and `dense_identity`
use `physical_lower_bound_including_orphaned_recovery_compute`, and permanently
disables direct wall-clock, throughput, cost-efficiency, equal-budget, quality,
promotion, release, and full-300K claims. It compares the canonical result only
for semantic observational-runtime equivalence; byte equality is intentionally
not required because the strict schema adds validation and policy fields.

If the canonical result enables any direct ranking or differs in bound sources,
matched contract, GPU-contention evidence, or physical cost evidence, the report
selects `canonical_runtime_claim_rejected_strict_guard_controls` and keeps all
runtime ranking fail-closed.

`scripts/wait_generation_runtime_claim_guard_comparison.py` is a CPU-only,
CUDA-hidden, signal-free waiter. It writes to an independent output root and
does not replace the canonical guard or authorize another experiment.

## Verification

The targeted suite passed from an isolated D: worktree:

```text
tests/test_generation_runtime_compute_claim_guard.py
tests/test_generation_runtime_claim_guard_comparison.py

19 passed
```

Coverage includes:

- safe semantic equivalence of observational-only canonical output;
- rejection of canonical direct runtime ranking;
- both-method physical lower-bound binding;
- wrapper signal prohibition and CPU-only identity;
- physical input SHA mismatch rejection;
- completed waiter output and PID cleanup.

## Claim boundary

This comparison is permanently non-authorizing. It does not establish sample
quality, factorization advantage, class fidelity, broad generation superiority,
promotion readiness, release readiness, or permission for full 300K training.
`generation_advantage_proven` remains false until matched terminal sampling,
mechanism, class-fidelity, and terminal-system evidence passes its own guards.
