# 2026-08-03 EMA inference export manifest, lock, and exact recovery

## Scope

This change closes a stable-inference gap in the large-scale generation path.
The EMA exporter already wrote an artifact atomically and then wrote its
integrity sidecar, but it had no component output lock and no durable request
identity before deserializing a full training checkpoint. A crash between the
two writes left an ambiguous partial target that could neither be safely reused
nor safely repaired.

This is deployment/reproducibility work only. It does not change CoFiTok's
token-only restricted synthesis operator, the dense matched baseline, any
training configuration, locked paper evidence, or the authorization state of
full 300K training.

## Contract

Every exported artifact now has three bound files:

1. `<artifact>.pt`: EMA-applied model/config payload;
2. `<artifact>.pt.integrity.json`: artifact bytes/SHA and embedded source,
   training-authorization, and final-release provenance;
3. `<artifact>.pt.export_manifest.json`: the pre-deserialization export
   request identity.

The export manifest binds:

- source checkpoint absolute path, bytes, SHA256, format, and step;
- source integrity-sidecar absolute path, bytes, and SHA256;
- source runtime-environment SHA, Git identity, and projected training
  authorization;
- exact artifact and integrity-sidecar target paths;
- final full-gate authorization, including the gate and source-report binding;
- exporter clean Git and complete CPU runtime-environment identity;
- EMA-only artifact type and format.

The exporter acquires a persistent adjacent non-blocking OS lock before hashing
or deserializing the source. A concurrent exporter fails closed. The lock file
may remain after release, but exclusivity belongs to the file descriptor, so a
crash does not create a stale logical lock.

## Recovery state machine

- No artifact, sidecar, or manifest: atomically write the manifest and export.
  `--resume` is allowed here so formal runbooks use one command for first launch
  and recovery.
- Complete artifact + sidecar + exact manifest: verify all bytes and provenance,
  then reuse without modifying artifact, sidecar, or manifest.
- Artifact XOR sidecar + exact manifest: refuse without explicit `--resume`;
  with `--resume`, remove only those exact manifest-bound partial targets and
  rebuild.
- Manifest only: require `--resume`, then rebuild and remove only regular
  `<artifact>.tmp-*` leftovers while holding the lock.
- Partial outputs without a manifest: refuse and preserve them.
- Complete but invalid/tampered outputs: refuse even with `--resume`; never
  auto-delete or auto-repair possible tampering.
- Any source, release gate, exporter Git/runtime, or target-path drift relative
  to the manifest: refuse before checkpoint deserialization.

The formal stability runbook declares each export manifest as a stage output and
as an input to preflight and smoke stages. The legacy full-generation runbook
also uses explicit `--resume` and requires both manifests after export.

## Terminal consumption

Both large-scale and stability completion audits independently reopen the
manifest, revalidate its source checkpoint/sidecar and release gate, and compare
its physical descriptor and payload against the export report. Their passing
inference evidence now includes the manifest identity and source-checkpoint SHA.

The terminal release receipt copies those bindings. Receipt creation performs
the strict source-aware verification again. Routine inference receipt
verification rehashes the unchanged manifest and checks its payload against the
receipt without reopening the large source checkpoint, preserving deployment
portability after source-checkpoint archival.

## Verification

- Focused local suite:
  `python -m pytest -q tests/test_generation_inference_artifact.py tests/test_generation_stability_inference_export_runbook.py tests/test_large_scale_generation_completion_audit.py`
  — `112 passed`.
- Covered fault cases include completed byte/mtime replay, artifact-only crash,
  missing manifest, manifest drift before deserialization, stale temporary
  cleanup, complete tampering, concurrent lock contention, completion-audit
  descriptor/payload drift, release-time manifest deletion, and source
  checkpoint archival after receipt creation.
- Full local pytest: `1,035 collected`, `1,029 passed`, `6 skipped`, no
  failures (`263.6 s`).
- `python -m py_compile` passed for the exporter, artifact/release libraries,
  and completion auditor.
- All `104/104` tracked shell runbooks passed `bash -n` on `pro6000` after
  normalizing the Windows transfer copy to the LF bytes a Linux Git checkout
  uses. The isolated syntax-only copy was
  `/tmp/cofitok-runbooks-syntax-20260803-export-manifest`; no training checkout
  or GPU process was modified.
