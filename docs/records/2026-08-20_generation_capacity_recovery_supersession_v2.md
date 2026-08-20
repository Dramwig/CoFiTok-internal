# Capacity recovery supersession contract v2

Date: 2026-08-20

## Outcome

The capacity lineage observer false failure has a source-bound fix. The old
schema-v1 contract treated `run_manifest.json` as byte-immutable even though
the production trainer legitimately rewrites that file after every exact
resume. The 50K resume therefore changed the manifest SHA256 and made the v2
observer report a recovery failure while training and all downstream waiting
supervisors remained healthy.

Commit `f9ed9b5b1958887ef490a9251f5e64eadf26ef6b` adds schema-v2 recovery
supersession contracts while preserving schema-v1 replay compatibility. The
checked-in live v2 contract is:

```text
configs/generation/diagnostics/quality_bridge_recovery_supersession_20260820_v2.json
bytes: 3,520
SHA256: d42b3bca92087062eb8e2999ac9f1cd3be729bf72b9e3e4cb823d5544c67e379
```

The historical v1 contract remains unchanged.

## Validation policy

Schema v2 labels only the current `run_manifest.json` as
`mutable_semantic`. It still requires the path to be the direct
`run_manifest.json` child of the exact expected run directory. The observer
reopens the current manifest and strictly checks:

- training revision, branch, and tracked-clean status;
- output root and run directory containment;
- dataset identity and recomputed runtime-environment identity;
- resolved micro-batch times gradient accumulation equals effective batch;
- absence of an unauthorized revision transition;
- an in-run `checkpoint_step_XXXXXXXX.pt` resume path;
- a current resume step no earlier than the original recovery;
- live pair-monitor progress no earlier than the current resume;
- exact `unchanged` or content-addressed `reconciled` metrics semantics.

For `unchanged`, orphan rows must be zero and orphan path/SHA must be null. For
`reconciled`, the report and orphan archive must be direct children of the run
directory, filenames must encode resume step and digest prefix, the archive
SHA256 must match, and the report must equal the manifest payload after
removing only its self-reference.

The original 20K reconciliation and its orphan archive remain immutable and
content-addressed. The physically audited 25K checkpoint also remains bound by
its original SHA256. The observer exposes original and current resume steps
separately.

## Verification

Clean-commit validation completed with no failures:

```text
Windows focused: 31 passed
Windows full: 1,669 passed, 11 skipped, 0 failed
Linux focused: 31 passed
Linux full: 1,678 passed, 2 skipped, 0 failed
tracked runbooks: 141/141 passed bash -n
Python compile: passed
git diff --check: passed
```

Two fail-closed environment checks were retained rather than hidden. Before
the implementation commit, three tests correctly rejected a dirty control
checkout. On the server, six inference-waiter tests correctly rejected the
symlinked launcher `.../bin/python`; they and the full suite passed using its
resolved regular-file target `.../bin/python3.10`. No production environment
path was modified.

## Live-source rehearsal

The incremental bundle advertised only `f9ed9b5` and required exact
prerequisite `a963ff4`:

```text
bytes: 10,909
SHA256: 49d548bc33c844756d49079ce18443473a29987ae6bc129cb59610827b26da2d
```

It was verified against an existing isolated server repository containing the
prerequisite. The newly cloned rehearsal checkout was:

```text
/tmp/cofitok-recovery-supersession-f9ed9b5.Rq2Ymk/CoFiTok-internal
```

CUDA was hidden and CPU/IO priority was lowered. A real-source one-shot report
was written only under `/tmp`:

```text
bytes: 25,842
SHA256: 266ecefb6285c7a2f03bac6b419894b92dbc5a697ece1af9f9a457f13e22afa8
status: running
issues: []
recovery stage health: running
original resume: 20,000
current resume: 50,000
current reconciliation: unchanged
trusted checkpoint: 25,000
live CoFiTok step: 57,350
watchdog target: 100,000
```

This proves that the new contract repairs the false observer verdict against
the current live sources without weakening the immutable recovery anchors.

## Boundary

This work is read-only control-plane repair. It does not launch or authorize
training, sampling, evaluation, capacity scaling, full 300K, promotion, or
release. It does not establish a generation-quality advantage. Active trainer
PID `543758`, failed historical observer PID `765120`, and the formal checkout
were not signaled or modified.

Machine-readable rehearsal evidence is stored at:

```text
artifacts/reports/generation/capacity_recovery_supersession_v2_2026-08-20/rehearsal_report.json
```
