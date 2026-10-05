# Preparation source-provenance audit (2026-09-01)

## Scope

This is a local, CPU-only audit of the non-authorizing exposure/capacity
preparation. It does not create a checkout, build an execution authorization,
modify the remote server, or launch compute.

## Findings

The preparation report remains valid under the current main-tree validator:

```text
preparation SHA256
b34d0e0a057bed34511c88f83671c80039b86ea45cf548636daff02f91dcb074
status pass; source_count 10
```

Its recorded builder is the dirty main worktree at:

```text
revision 354f9dc52aec295dff439d4198efdff51beacdb8
tree     e829077cf0289a9414964d8c2e53fb8c1ac907a1
tracked_dirty true
```

The relevant preparation builder and module are untracked in that worktree.
Their current SHA256 values are:

```text
scripts/build_generation_exposure_capacity_preparation.py
ab7908d153a201561706a9c26992a66bb8d9743d4381278463e941516d41b5f9
src/cofitok/generation/exposure_capacity.py
a5b44ac68a23b7a81752c72ca4b7b710063c569f96a8a2e5c8ec04bfb77a9c0f
```

The existing D: isolated builder is tracked-clean, but it is a different
source state:

```text
revision eb5821cca5ddf00743954a7b3c2e3828f29f7084
tree     95309daebc2dd121a2558b1f1c1e2736ae53461a
src/cofitok/generation/exposure_capacity.py
fa9c12de4cbca1b18c62f76d89eea4f5d8e1e892eed0344f33d54aa62a9f160a
```

Running the D: validator against the current preparation fails deterministically
with `preparation objective reassessment evidence must be a JSON object`. The
current main module and the D: module therefore do not implement the same
preparation schema. This is a source mismatch, not evidence that either
candidate arm executed.

## Decision

The preparation stays non-authorizing and `execution_ready=false`. The dirty
builder provenance is retained rather than silently replaced. No clean
source-compatible rebuild is claimed until an explicitly allowed isolated
source snapshot can include the exact builder and module. This preserves the
fail-closed boundary for any future exposure/capacity gate.
