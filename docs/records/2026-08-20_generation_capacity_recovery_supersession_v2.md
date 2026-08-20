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

The repaired observer was subsequently deployed as v3 from evidence commit
`3338f6c9474cae800df92fa6465039ccc578a424`. V3 is now the operationally
authoritative capacity-lineage observer. The still-running v2 process and its
false-failure report were not signaled or rewritten; a read-only copy of that
report remains frozen as historical evidence.

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

## Operational authority transition

The persistent v3 observer is bound to the exact evidence checkout:

```text
PID: 812832
parent PID: 1
revision: 3338f6c9474cae800df92fa6465039ccc578a424
tree: 1a5498312264c923ca4a4db2f58d5d82b5cf0388
branch: fix/generation-recovery-supersession-v2
executable: /root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10
start ticks: 1684805252
cmdline SHA256: 69ac93e3483c97099b8508ee44aff742ac8884836691345469d1a0992f11798a
nice: 19
I/O class: idle
CUDA_VISIBLE_DEVICES: -1
```

File descriptor 9 resolves to the exact v3 lock. The descriptor and lock path
share device/inode `2304:8595729046`; the process has one thread and does not
appear in the GPU compute-process list. The PID file is 7 bytes with SHA256
`fca5be84896cd576b5ee7daec94025796b959c64b62518b930b2acaeb78c755f`.

Four live reports were copied byte-for-byte into the remote authority-transition
directory and made mode `0444`. They are also mirrored beside the local receipt:

| poll | observed at UTC | bytes | SHA256 | live step |
|---|---:|---:|---|---:|
| 01 | 2026-08-20 04:02:56 | 26,002 | `62e68acda9d85af4c278073f30b5f1aa1323dff5ac5ae26b1878ff907c5d8ea2` | 58,000 |
| 02 | 2026-08-20 04:03:57 | 26,003 | `9090ac41dd366875ba8f1ca02c1d539946401395ecaf53dede435c37acf26f69` | 58,000 |
| 03 | 2026-08-20 04:05:57 | 26,004 | `8bbffcc995859f39032a455043e43ace0b230338b7c878ca4a5b131db41fbaae` | 58,100 |
| 04 | 2026-08-20 04:06:57 | 26,003 | `4903d4bd52b45a30e3b5204bf4ad3dbb8ec72bb1886427713c5103decb767009` | 58,100 |

Every snapshot is `running`, has an empty top-level issue list, and reports the
recovery stage as healthy and running with no stage issues. All four preserve:

```text
original resume step: 20,000
current resume step: 50,000
current reconciliation: unchanged
trusted physical checkpoint step: 25,000
watchdog target: 100,000
```

The authority-transition receipt is immutable on the server and mirrored in
the versioned evidence directory:

```text
remote: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/authority_transition_2026-08-20/capacity_generation_pipeline_lineage_observer_v3_authority_receipt.json
local: artifacts/reports/generation/capacity_recovery_supersession_v2_2026-08-20/authority_transition_receipt.json
bytes: 12,748
SHA256: a3133c0b1f23b761e171606fdf4d9d7fc592853aab6e9e29c42a7d48c0984f99
remote mode: 0444
```

The receipt declares v3 operationally authoritative and v2 non-authoritative
for the current exact-resume lineage. The preserved v2 false-failure snapshot
remains 20,928 bytes, mode `0444`, SHA256
`ee905d9254739f470543122bd9197417176d5c2d4e68816a1544bdd841f17ed7`.
V2 PID `765120` remains alive and was not signaled.

At the final frozen poll, trainer PID `543758` was still the sole GPU compute
process, using 89,398 MiB. Dense remained complete at 50K while CoFiTok had
advanced to 58.1K toward the matched 100K target. The formal checkout remained
tracked-clean at `scale/generative-system@1ebcc152`, with the established
default-porcelain binding of 89 rows and SHA256
`18e5981f22a2ac255c7f60343429daa6ceea86412b1d7bd09bc474f2faf74004`.

## Boundary

This work is read-only control-plane repair. It does not launch or authorize
training, sampling, evaluation, capacity scaling, full 300K, promotion, or
release. It does not establish a generation-quality advantage. Active trainer
PID `543758`, failed historical observer PID `765120`, and the formal checkout
were not signaled or modified.

Machine-readable rehearsal evidence is stored at:

```text
artifacts/reports/generation/capacity_recovery_supersession_v2_2026-08-20/rehearsal_report.json
artifacts/reports/generation/capacity_recovery_supersession_v2_2026-08-20/authority_transition_receipt.json
```
