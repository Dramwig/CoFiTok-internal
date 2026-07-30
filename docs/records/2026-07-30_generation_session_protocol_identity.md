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

## Verification

The focused session, formal-protocol, and inference-artifact tests passed
`25/25`. A new test proves exact CPU equality between generating two explicit
seeds in one batch and generating the same seeds as two single-image requests.
The full local suite passed with three existing environment-related skips.

This revision is not deployed into the active matched two-step 5K checkout at
`2521d874a82898a7a2a527d824ea1df285df221d`. The active pair remains
single-revision evidence; future formal training and inference will consume the
post-training upgrade revision only after the stability gate allows preparation.
