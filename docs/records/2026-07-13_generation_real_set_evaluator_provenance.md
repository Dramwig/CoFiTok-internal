# Formal real-set and evaluator provenance (2026-07-13)

## Finding

Formal torch-fidelity reports previously identified the ImageNet validation set
only by directory path and image count. The real-feature cache used a fixed name.
Changed image bytes, path changes, or a stale cache could therefore survive the
matched-protocol checks and undermine absolute FID reproducibility.

The evaluator also recorded package and Torch versions but did not carry the
same canonical runtime-environment fingerprint used by training and sampling.

## Contract

`evaluate_generation_metrics.py` now computes
`cofitok_image_tree_sha256_v1`, hashing each real image's root-relative path,
declared size, and complete file bytes with explicit framing. Symlinks and files
outside the declared root are rejected. Metrics report schema v2 stores the full
real-set identity and evaluator environment.

The effective torch-fidelity real cache name is derived from the first 16 hex
characters of the full tree digest. A changed validation set therefore receives
a different cache namespace before FID/precision/recall are computed.

## Fail-closed chain

Promotion and final reports add `matched_real_set_provenance` and
`matched_evaluator_runtime_environment`. The final comparison moves to schema v3
and carries both SHA values in every matched row. The terminal completion audit
revalidates the reports, named gate evidence, and comparison rows; mismatched
real bytes or evaluator environments fail formal generation, the final gate,
and the final comparison independently.

Tests cover digest order invariance, byte mutation, out-of-root rejection,
content-addressed cache construction, a real tiny metrics-report path, and
CoFiTok/dense provenance drift at every downstream boundary.
