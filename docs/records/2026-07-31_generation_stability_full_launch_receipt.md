# Stability-full immutable launch receipt (2026-07-31)

## Problem

The 250M readiness/launch split already made CUDA qualification an immutable
input to the full 300K runbook. The launch stage still had one provenance gap:
`storage_capacity_launch.json` was rewritten every time the resumable runbook
was invoked. The terminal audit could therefore prove only the most recent
resume-time storage state, not the exact first-launch authorization that existed
before any trainer or monitor started.

The later completion audit also replayed training-time readiness from config
paths under its current evaluation checkout. That contradicted the intended
later-revision replay model because the immutable readiness source identities
point to the deployed training checkout.

## Implementation

The full 300K chain now adds:

- `scripts/build_generation_full_launch_receipt.py`;
- `scripts/validate_generation_full_launch_receipt.py`;
- `reports/full_training_launch_receipt.json`.

On the first launch, before the monitor or trainer starts, the receipt binds:

- the immutable full readiness report and SHA256;
- the source-bound scaling promotion gate;
- the isolated large-capacity deployment receipt;
- both exact 250M training configs;
- readiness config validation and storage reports;
- the frozen runtime selection and benchmark root;
- the first-launch `storage_capacity_launch.json`;
- both full 300K run paths and the training Git identity.

Receipt construction requires both run directories to contain no state. If the
builder fails, the runbook removes the still-unbound launch-storage report. It
does not remove any training state or any completed receipt.

## Versioned deployment evidence

Large-capacity deployment evidence is now target-versioned at
`deployment/large_capacity/deployments/<full-target-sha>/`. Each successor
checkout receives its own deployment receipt, JUnit XML, and runbook syntax
report, so deploying the launch-receipt hardening revision cannot overwrite the
historical `7bab083` validation. Readiness, launch, and completion audit all
resolve this path from the expected full training revision. The old fixed
`deployment/large_capacity/deployment_receipt.json` key remains only for
historical compatibility and is not used by the active chain.
## Resume contract

Once the receipt exists, every runbook invocation must provide
`EXPECTED_FULL_LAUNCH_RECEIPT_SHA256`. The validator rehashes and reopens every
bound source and requires the deployed training checkout to remain exact and
tracked-clean. It permits the formal repository to advance after deployment,
because that repository is not the training checkout and is not a training
state source.

Resume never overwrites `storage_capacity_launch.json`. A fresh capacity check
is written to `storage_capacity_current.json`; it remains an operational guard,
while the original launch-time report stays immutable evidence.

## Completion audit

The stability completion audit now requires
`--expected-full-launch-receipt-sha256` and adds the fail-closed check
`stability_full_launch_receipt`. It resolves the training source root from the
deployment receipt's checkout path, not from the later evaluation/audit
checkout. It therefore supports later evaluation code while still reopening
the exact training-time configs and receipt-bound reports.

The completion runbook exposes the matching required environment variable:
`EXPECTED_FULL_LAUNCH_RECEIPT_SHA256`.

## Boundary

This change strengthens reproducible full-training authorization and recovery.
It does not execute CUDA readiness, authorize or start full 300K training, run
formal sampling, or claim large-scale generation completion. Those stages
remain blocked on the active source-bound 50K pair, its formal post-evaluation,
and a passing promotion gate.
