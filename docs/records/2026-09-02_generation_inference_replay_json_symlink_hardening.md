# Inference Replay JSON Symlink Hardening (2026-09-02)

## Scope

This was a local CPU-only provenance hardening change. It did not modify the
remote checkout, checkpoints, generated samples, locks, or any GPU process.

## Change

The shared `cofitok.inference_replay.read_json_object` boundary now rejects a
symlink in the JSON file itself or any parent directory before opening it.
This helper is used by inference manifests, progress, export manifests, and
several formal report readers, so a path alias cannot silently change the
source whose bytes are later treated as evidence.

## Verification

- Added POSIX-only regression coverage for symlinked JSON files and symlinked
  parent directories, plus a regular-file behavior check.
- Focused inference/session/artifact tests passed; the Windows symlink cases
  are skipped under the existing platform limitation.
- `python -m compileall -q src tests` passed.
- The remote matched 100K pair remains complete at 100,000 steps and
  6,400,000 images per method; the GPU remains idle and no new experiment was
  launched.

The terminal scientific status remains `hold` and
`generation_advantage_proven=false`.
