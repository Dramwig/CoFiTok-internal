# Bounded exposure/capacity gate contract (2026-08-31)

## Why a new gate is required

The older `9f6b2f2` exposure-qualification preparation consumes the old v2
follow-up decision and a historical capacity-probe result. It also describes a
fresh 300K exposure stage. Those sources are not the current authoritative
post-100K decision, and the current terminal result is still a scientific
`hold`. Reusing that chain would silently widen both the source and the
execution scope.

The 8/31 preparation therefore remains the only source for the two current
candidates. The new module
`src/cofitok/generation/exposure_capacity_gate.py` and its CLI scripts build a
single-arm, source-bound candidate gate from that preparation.

## Contract

The gate binds the preparation identity, the exact 100K bridge checkout
(`cf0e5faa...`, tree `6cef2772...`, branch
`scale/generation-stability-quality-bridge-100k`), all preparation source
identities, a dataset/runtime hash, an idle-GPU snapshot, no conflicting
processes, a free output root, an unlocked execution lock, and a storage
reserve. The arm is mutually exclusive:

* `capacity_qualification`: fresh matched 256-channel initialization with a
  10K qualification and a 1K screen before any 10K confirmation.
* `exposure_continuation`: exact 100K checkpoint resume for a fixed first target
  of 110K steps, changing exposure only.

Both arms keep the objective, conditioning, token layout, and formal quality
claim fixed. The gate is `status=prepared`, `execution_ready=false`, and
`terminal_status=hold`. Its authorization boundary sets every training,
sampling, evaluation, GPU, 300K, promotion, export, release, and process-signal
permission to `false`; a separate stage authorization is required later.
Automatic 300K escalation and replacement of the terminal hold are forbidden.

The validator physically rehashes each preparation source before accepting the
gate. It is safe to run on a new server for preflight, but it cannot launch a
process or mutate the authoritative checkout. No gate artifact was executed in
this record.

## Verification

Focused CPU tests cover both arms, the fixed target, permission tampering,
wrong source checkout, GPU/process contention, and storage failure. Execution
requires the project test environment; no GPU experiment is part of this
record.
