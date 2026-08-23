# Terminal runtime strict conjunct: fail-closed comparator support (2026-08-23)

## Incident

The recovered strict runtime comparator can correctly return either of two
operationally passing, non-authorizing decisions:

- `canonical_observational_runtime_claim_semantically_verified`; or
- `canonical_runtime_claim_rejected_strict_guard_controls`.

The original terminal runtime conjunct accepted only the first decision.  The
recovered v3 strict replay produced the second, more conservative result, so an
unchanged conjunct waiter would wait normally and then fail only after terminal
completion became available.  This is a downstream evidence-wiring defect; it
does not affect the terminal GPU sampling chain or any quality metric.

## Fix

`scripts/build_generation_terminal_runtime_strict_conjunct.py` now accepts both
safe comparator decisions while preserving their different meanings.  When the
canonical claim is rejected, the conjunct requires:

- `canonical_runtime_claim_trusted=false`;
- `strict_recovery_binding_required=true`;
- `strict_recovery_binding_verified=true`; and
- direct wall-clock, throughput, cost-efficiency, equal-budget, quality,
  promotion, release, and full-300K claims to remain disabled.

The output explicitly records whether the canonical observational claim is
trusted or the strict guard controls.  A rejected canonical claim cannot be
silently promoted to an accepted claim.

## Deployment boundary

The corrected waiter must use a new versioned output directory.  Existing
conjunct v1/v2 processes and artifacts remain immutable and receive no process
signals.  The corrected path is CPU-only, CUDA-hidden, low priority, and
permanently non-authorizing.  It can conjoin a completed terminal audit with the
runtime boundary, but it cannot launch training or sampling, authorize 300K,
promote or release a model, or modify the upstream scientific decision.

