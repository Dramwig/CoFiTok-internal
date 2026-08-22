# 2026-08-22 terminal generation-metrics cache trust boundary

## Scope

This change closes a fail-closed evidence gap in the ImageNet-256 matched 100K
quality-bridge terminal completion audit. The terminal FID/IS/precision/recall
reports use the shared torch-fidelity real-set cache. Before this change, the
reports bound the content-addressed cache name and real image-tree SHA256, but
the completion audit did not physically bind the cache payload bytes, feature
extractor weights, or installed torch-fidelity source tree.

The work is CPU-only and permanently non-authorizing. It cannot launch or alter
training, sampling, promotion, release, export, 300K scaling, or any upstream
decision.

## Added evidence contract

`cofitok.generation_metrics_integrity` and
`scripts/audit_generation_metrics_trust_boundary.py` now build and replay a
canonical receipt that verifies:

- the exact 50,000-image ImageNet-256 validation tree and its
  `cofitok_image_tree_sha256_v1` digest;
- the exact four torch-fidelity real-cache files, with physical bytes and
  SHA256 identities;
- cache tensor shape, dtype, device, and finiteness;
- exact cached Inception-feature mean equality and covariance diagonal/probe
  consistency with the cached FID statistics;
- physical Inception and VGG weight identities, including filename hash-prefix
  checks;
- the installed `torch-fidelity==0.4.0` package and dist-info file trees;
- the archived first cache-creation report, ordered creation log events, and
  Inception dependency-repair report;
- exact binding of both terminal schema-v3 generation metric reports to the
  attested real set, cache root, cache name, implementation, and enabled
  precision/recall protocol.

The receipt explicitly states that it does not independently regenerate all
50,000 real-set feature tensors. It physically rehashes and semantically checks
the existing cache and replays the recorded creation evidence.

## Completion-audit integration

The terminal completion builder now requires the canonical receipt at:

```text
reports/metrics_trust_boundary_v1/metrics_trust_receipt.json
```

It verifies the receipt and both terminal generation reports before and after
the checkpoint/source replay. The terminal completion waiter binds the exact
receipt SHA256 in its static context, deployment receipt, readiness checks, and
metadata drift snapshot. A missing, failed, changed, or non-canonical receipt
therefore prevents terminal completion from passing.

## Validation

Local isolated worktree validation:

```text
python -m py_compile: pass
targeted pytest: 29 passed
ruff: pass
full pytest: 1,145 passed, 6 skipped, 12 unrelated failures
```

The full-suite failures were outside the changed metric-integrity and terminal
completion files: four paper-layout tests require sibling `paper/` assets not
present beside this isolated D: worktree, and eight comparison-waiter tests
intentionally reject a tracked-dirty development checkout. The targeted suite
for all changed code passed; the exact committed checkout must be rerun remotely
with CUDA hidden before deployment.

The live GPU chain was not modified while developing or testing this change.
`generation_advantage_proven` remains `false` until the matched terminal
quality evidence and downstream claim guards complete.
