# Inference artifact final release authorization

Date: 2026-07-13

Branch: `scale/generative-system`

## Problem

The completion pipeline validated the full generation gate immediately before
EMA export, but the exporter did not consume that gate. The ordering was correct
while the runbook was intact, yet the standalone artifact could prove only that
its 300K training had been authorized by the earlier scaling gate. It could not
prove that its final FID, precision, recall, endpoint, ordering, synthesis, and
provenance checks had passed before deployment.

## Contract

Artifact payload and integrity schemas are version 4. Export from a checkpoint
that carries formal full-training authorization now requires
`--release-gate <final_generation_gate.json>`. The exporter validates the shared
full gate contract, then captures:

- canonical full-gate identity SHA256;
- exact gate file SHA256 and byte count;
- absolute gate path, full stage, readiness decision, and validated thresholds.

This release authorization is embedded in the artifact payload, integrity
sidecar, and export report. The stable loader compares payload with sidecar and
propagates it through `GenerationSession`, real-forward preflight, and inference
reports. Existing artifact reuse requires the exact same release authorization.

The terminal completion audit recomputes the semantic binding against the actual
full gate, requires artifact-file/export/preflight/smoke equality, and requires
CoFiTok and dense artifacts to use the same gate file binding. A failed, weakened,
changed, or post-hoc gate invalidates both the final-gate check and deployable
artifact evidence.

## Compatibility

Tiny and legacy non-formal checkpoints cannot claim a release gate and continue
to emit consistent `null` training/release authorization. Formal sources must
bind both authorizations: scaling authorizes long training; full readiness
authorizes deployment. No formal artifact was produced before this upgrade.

## Verification

Tests use a real passing full-gate file and tiny authorized checkpoint to cover
EMA export, exact artifact reuse, stable-session and preflight propagation,
missing release gate, changed gate reuse, payload/sidecar release drift, final
gate mutation, artifact/preflight mutation, and different CoFiTok/dense gate-file
bindings. The authoritative export runbook must pass the same final gate to both
methods.
