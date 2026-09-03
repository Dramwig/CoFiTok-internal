# Source-bound capacity scaling preparation decision

Date: 2026-09-03

## Purpose

The current qualification path is the fresh four-arm base-128/base-256 screen
followed by a frozen-checkpoint 10,000-sample confirmation. The historical
capacity-scaling decision consumed obsolete `capacity_probe*` reports and
directly authorized training. It was incompatible with the current evidence
chain and could not be deployed.

This revision consumes the canonical `capacity_confirmation_result.json` and
physically replays its preparation, launch receipt, and all four confirmation
arm validations. Each arm validation recursively replays the screen validation
and the bound sampling, distribution, class-fidelity, checkpoint, and rollout
evidence. The decision also freezes the exact base-256 step-10K checkpoint,
sidecar, config, training-report, dataset, runtime, parameter-count, and Git
identities selected for any later continuation.

## Decision boundary

A passing confirmation selects only preparation of a new matched base-256
continuation from step 10,000 to an intentional step-50,000 stop. Continuation
uses new output directories under `capacity_scaling_50000`, so the screen and
confirmation evidence remain immutable.

The decision is not an execution authorization. It keeps remote mutation, GPU
use, training, sampling, evaluation, configured step-100K completion, full
300K, promotion, export, release, and process signals disabled. Launching the
50K segment will require a separate source-bound preparation, a user-created
stage authorization, an execution authorization, a storage preflight, an idle
live snapshot, and an immutable launch receipt.

If any predeclared confirmation check fails, the decision routes to `hold` and
does not permit even preparation of the scaling stage.

## Additional integrity fix

The capacity partial-training validator now includes the exact optimizer step
in its checkpoint summary. Confirmation preparation already required this
field, so omitting it would have caused a real screen-to-confirmation handoff to
fail despite unit fixtures containing the value. Confirmation arm validations
also carry the content-addressed frozen config and training-report identities,
which lets downstream preparation bind the unchanged model and objective.

## Files

- `src/cofitok/generation/capacity_scaling_decision.py`
- `scripts/build_generation_capacity_scaling_decision.py`
- `scripts/verify_generation_capacity_scaling_decision.py`
- `scripts/wait_for_generation_capacity_scaling_decision.py`
- `artifacts/runbooks/generation_capacity_scaling_decision_after_confirmation.sh`
- `src/cofitok/generation/capacity_qualification_training.py`
- `src/cofitok/generation/capacity_confirmation_arm.py`
- `tests/test_generation_capacity_scaling_decision.py`
- `tests/test_generation_capacity_scaling_waiter.py`

The waiter is CPU-only and produces one immutable, replayable decision after
the exact confirmation result exists. It contains no training or GPU entrypoint.
