# Inference artifact authorization provenance

Date: 2026-07-13

Branch: `scale/generative-system`

## Problem

Formal full-training checkpoints bind the exact scaling promotion gate, but the
previous EMA-only inference artifact retained only the source checkpoint hash,
runtime environment, and Git identity. Export verified the source authorization
once, yet a deployed artifact separated from the training directory could not
state which promotion gate authorized its weights.

## Contract

Inference artifact payload and integrity schemas introduced this binding in
version 3. A formal export
copies the complete source `training_authorization` into both the artifact
payload and its adjacent integrity sidecar. The export report, stable generation
loader, `GenerationSession` metadata, real-forward preflight, and inference CLI
report propagate the same mapping.

The authorization contains the scaling stage and decision, canonical gate
identity, exact gate-file SHA256 and byte count, absolute source path, and the
thresholds validated before full training. Existing-artifact reuse compares
these fields with the still-verified source checkpoint sidecar. Loading compares
artifact payload with sidecar before weights become available to the caller.

The terminal completion audit independently requires the artifact file,
export report, preflight, and smoke inference to equal each method's full 300K
training report authorization. CoFiTok and dense therefore cannot be deployed
from a different or post-hoc promotion decision.

## Compatibility

Non-formal tiny and legacy checkpoints may have no training authorization; their
schema-v3 artifacts record `null` consistently. Formal ImageNet-256 300K source
checkpoints already fail export when the authorization is missing, so no legacy
exception is introduced for the large-scale completion path. No formal
inference artifact existed before this schema upgrade.

## Verification

Tests cover a real tiny training checkpoint with authorization, EMA export,
artifact reuse, exact sample equivalence, stable session metadata, real-forward
preflight propagation, artifact-sidecar authorization drift, source-sidecar
reuse drift, and terminal completion failures for artifact or preflight drift.

Artifact schema v4 subsequently adds the independent final quality release
authorization; see
`docs/records/2026-07-13_inference_artifact_final_release_authorization.md`.
