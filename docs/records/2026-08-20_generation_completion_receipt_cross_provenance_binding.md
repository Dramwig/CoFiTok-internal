# Generation completion receipt cross-provenance binding

Date: 2026-08-20

## Purpose

Close the remaining final-release trust-boundary gap after the exact
required-check contract was added. A completion audit could previously carry
the correct ordered check names while leaving its top-level expectations,
supporting check evidence, evaluated checkpoints, and exported inference
artifacts only loosely related at receipt construction time.

The individual auditors and artifact verifier already validated their own
sources. This change makes the release receipt independently reconstruct and
cross-bind those results before it authorizes an inference artifact.

## Implementation

Implementation commit:

```text
5814bf51d113746ffca0ad1fac9f282791e3dd4a
```

`cofitok.generation.release` now:

- requires the exact expectation field schema emitted by each supported
  completion auditor;
- validates revision, tree, SHA256, and branch values without permissive type
  coercion;
- binds the legacy large-scale revision transition and full-training result to
  its expectations;
- binds the stability decision, scaling gate, full readiness, launch receipt,
  full/final training and evaluation Git identities, and export Git identity;
- binds the capacity-full training/post-evaluation deployment receipts, exact
  training launch receipt, training/evaluation Git trees, and final gate;
- validates the complete training and release authorization schemas;
- requires CoFiTok and `dense_identity` to use shared runtime, Git,
  authorization, and export provenance while retaining distinct artifact bytes
  and distinct source checkpoints;
- requires the exact checkpoints evaluated by formal generation to be the
  checkpoints exported into the release artifacts;
- rejects any forged or drifted provenance before physical inference artifact
  verification during receipt construction.

Receipt consumption continues to reconstruct the payload from the bound audit,
so the same cross-provenance checks run again before an artifact can be used as
completion-authorized inference evidence.

## Auditor-schema verification

The release expectation contracts were compared with the current producers:

- `large_scale_generation_v1`: 3/3 exact fields;
- `stability_generation_system_v1`: 16/16 exact fields;
- `capacity_full_generation_system_v1`: 12/12 exact fields.

The capacity-full path was verified against its real authorization identity:

```text
stage: capacity_full_experimental
decision: authorize_fresh_matched_300k_training
source: capacity_full_300k_training_launch_receipt.json
```

It is intentionally not treated as a scaling promotion gate.

## Verification

Focused completion/release regression set passed, including inference artifact,
stability completion, capacity-full post-evaluation, legacy completion, capacity
training launch, and full launch receipt tests.

The final clean-HEAD local CPU suite used Python 3.10.20 and collected 1,757
tests:

```text
1746 passed, 11 skipped in 1363.91s (0:22:43)
```

Negative tests cover expectation type/field drift, source and execution Git
drift, readiness/launch/deployment SHA drift, authorization schema drift,
cross-method provenance drift, reused checkpoints, mismatched evaluated versus
exported checkpoints, and malformed artifact digests. Sentinel monkeypatches
confirm these cases fail before artifact verification.

## Operational boundary

This work was performed only in the clean local worktree
`C:/qbfinalreleaseaudit` on branch
`analysis/generation-final-release-provenance-audit-v1`. It did not deploy to or
move any remote checkout, use the remote GPU, signal any process, modify the
active matched 100K queue, or authorize any later training, sampling,
promotion, or release stage.

This is reproducibility and release-integrity hardening. It is not evidence of
CoFiTok generation-quality superiority over the matched dense baseline.
