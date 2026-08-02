# Exact routine-inference resume (2026-08-02)

## Problem

`GenerationSession` already made one request deterministic and the formal 10K/
50K sampler had exact progress recovery. The smaller class/seed/prefix inference
CLI did not: an interruption left atomically valid PNGs plus a failed report,
and recovery required `--overwrite`, regenerating every requested image. That
was operationally fragile for the intended release-authorized EMA artifact and
made routine inference weaker than the scientific sampler.

## Source-bound request and progress

`scripts/infer_generation.py` now writes two atomic control files inside the
output root before and during generation:

- `inference_manifest.json` freezes the verified checkpoint and integrity
  identity, weight type and release policy, clean Git provenance, canonical
  runtime environment, complete DDIM/CFG request, report path, and every
  expected seed/class/prefix output path;
- `inference_progress.json` binds that manifest by path/bytes/SHA256 and records
  attempt count, cumulative elapsed time, status, exact completed output rows,
  and each physical PNG digest.

`--resume` requires the newly resolved manifest to equal the stored manifest
recursively. It rehashes every progress-bound PNG, leaves valid outputs byte- and
mtime-identical, and regenerates only missing or digest-mismatched identities
using their original independent random streams. A completed exact resume is
read-only. `--resume` and `--overwrite` are mutually exclusive, and request,
checkpoint, Git, runtime, report, manifest, or completed-progress drift fails
closed.

The output directory is also a closed set: every PNG must be one of the
manifest's direct children. A stale or nested PNG from another request is
rejected before manifest/progress/report mutation, including under
`--overwrite`. The terminal replay validator rehashes every physical PNG itself
and rejects non-finite timing or missing progress evidence.

The inference report advances to schema v2 and binds the immutable manifest and
terminal progress identities. Running and failed attempts also publish bound
reports, so a later process can distinguish a legitimate interruption from an
unrelated output directory.

## Release boundary

The terminal large-scale completion audit now independently reopens and hashes
the two smoke-inference control files, checks their schemas and source binding,
and reconciles manifest expectations, progress rows, final report rows, and the
already decoded/rehash-verified PNG set. The release gate rejects a legacy smoke
report without verified resume evidence even if its PNGs happen to exist.

This is local CPU/control-plane hardening only. It does not deploy into or move
the immutable stability 50K training checkout, signal any GPU process, alter a
training/sampling quality threshold, or authorize full 300K.

## Verification

- `uv run python -m py_compile src/cofitok/inference_replay.py
  scripts/infer_generation.py scripts/audit_large_scale_generation_completion.py`
  passed.
- Focused inference/completion/runbook-contract regression collected 118 tests:
  `116 passed, 2 skipped`.
- Complete local regression collected 927 tests:
  `921 passed, 6 skipped in 205.11s`.
- `git diff --check` passed. No shell runbook changed in this work, so no new
  remote checkout or native runbook execution was needed; the focused and full
  suites still exercised the existing runbook contract and syntax tests.
