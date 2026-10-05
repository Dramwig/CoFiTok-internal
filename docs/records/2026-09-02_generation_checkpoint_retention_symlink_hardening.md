# Checkpoint-retention symlink hardening (2026-09-02)

## Scope

This is a CPU-only provenance hardening change. It does not launch training,
sampling, promotion, export, release, or any other GPU work, and it does not
modify the frozen 100K bridge evidence.

## Change

`src/cofitok/checkpoint_retention.py` now applies the project
`reject_symlink_chain` policy at every internal JSON and checkpoint-retention
boundary: inventory/reference roots, discovered sources, run directories,
latest pointers, training reports, checkpoint payloads, integrity sidecars,
retention reports, and runway filesystem paths. The run-state serializer no
longer canonicalizes a validated path with `.resolve()`, so a later symlink
cannot be silently normalized into the recorded identity.

The retention tests cover symlinked inventory roots, checkpoint payloads,
reference roots, latest pointers, integrity sidecars, and bound retention
reports on POSIX systems. These tests are skipped on Windows because creating
real symlinks requires privileges; the non-symlink regression cases still run.

## Verification

```text
.venv/Scripts/python.exe -m pytest -q
  passed at 100% (pre-existing platform skips only)

.venv/Scripts/python.exe -m compileall -q src/cofitok/checkpoint_retention.py tests/test_generation_checkpoint_retention.py
  passed

Focused completion, exposure/capacity, and checkpoint-retention tests
  passed
```

The authoritative remote bridge remains at 100K for both matched methods;
`terminal_status=hold` and `generation_advantage_proven=false` remain
unchanged. No new stage authorization was found, and the remote GPU remained
idle during this work.
