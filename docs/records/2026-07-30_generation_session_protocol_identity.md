# Generation session protocol identity

Date: 2026-07-30

## Motivation

Formal sampling reports already bound `cofitok_ddim_sampling_v1` and inference
API version 1, but direct callers of `GenerationSession.generate()` received
checkpoint and request metadata without the protocol identity. Stable inference
should remain provenance-complete when the reusable Python API is used without
an outer CLI.

## Implementation

Revision `0cecd31` makes every `GenerationResult` carry:

- `cofitok.generation.GenerationSession` API version 1;
- `cofitok_ddim_sampling_v1`;
- sampler and training-timestep identities;
- requested and actual DDIM timesteps;
- prefix budget, CFG mode and scales, eta, clipping, and precision;
- explicit per-request-seed random-stream semantics.

The session validates this object with `sampling_protocol_contract()` before
returning a result. The lightweight inference CLI propagates the API and
protocol identity into its atomic report.

Revision `3da40a3` adds an explicit production trust policy. Callers may set
`require_release_authorization=True`, or pass
`--require-release-authorization` to the lightweight inference CLI. In that
mode the loader rejects training checkpoints and unreleased development
artifacts before model deserialization; only an inference artifact carrying a
valid full-stage release authorization is accepted. Result and CLI metadata
record whether this production policy was required.

## Verification

The focused session, formal-protocol, and inference-artifact tests passed
`25/25`. A new test proves exact CPU equality between generating two explicit
seeds in one batch and generating the same seeds as two single-image requests.
The release-only follow-up passed `23/23` focused tests. The full local suite
passed after each production-code change with three existing
environment-related skips. Revision `72b1c54` strengthens the two rejection
tests with a `torch.load` sentinel, proving that both a training checkpoint and
an unreleased inference artifact fail policy validation before deserialization.

Revision `f6b5416` consumes the policy in the final export pipeline rather than
leaving it as an optional library capability. Both exported-artifact preflights
and both inference smoke invocations pass the release-only flag. Their reports
record the enforced policy, and the large-scale completion audit rejects a
missing or false policy field from either layer. The focused artifact,
preflight, completion-audit, and runbook tests passed `96/96`; the full local
suite passed with three existing skips. The uploaded runbook passed remote
`bash -n` and matched local SHA256
`dc8aa617a53c3e276e741e7e5701de830fe988b2b7ce00faa5884a1e6b89ce3b`.

This revision is not deployed into the active matched two-step 5K checkout at
`2521d874a82898a7a2a527d824ea1df285df221d`. The active pair remains
single-revision evidence; future formal training and inference will consume the
post-training upgrade revision only after the stability gate allows preparation.
