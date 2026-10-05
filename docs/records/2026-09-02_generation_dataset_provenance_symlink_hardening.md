# Dataset Provenance Symlink Hardening (2026-09-02)

## Scope

This was a local CPU-only provenance hardening change. It did not modify the
remote checkout, datasets, checkpoints, generated samples, locks, or any GPU
process.

## Change

Formal dataset provenance now rejects symlinks in the dataset root, any parent
component, and the manifest path before hashing the manifest or reporting the
dataset identity. The previous root `.resolve()` could silently canonicalize a
symlinked parent and make a different physical dataset look like the bound
source.

## Verification

- Added POSIX-only regression coverage for symlinked dataset parents and
  symlinked manifests, plus the existing regular provenance tests.
- Focused dataset/checkpoint provenance tests passed; Windows symlink cases
  remain skipped under the existing platform limitation.
- `python -m compileall -q src tests` passed.
- No remote training, sampling, promotion, export, release, or process signal
  was performed. The terminal scientific status remains `hold` and
  `generation_advantage_proven=false`.
