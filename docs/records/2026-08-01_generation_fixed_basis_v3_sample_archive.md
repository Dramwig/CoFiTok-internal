# Fixed-basis-v3 failed-gate sample archive

Date: 2026-08-01 CST

## Outcome

Three completed sample roots from the failed fixed-basis-v3 10K gate were
archived off the server and then removed from the checkpoint filesystem. The
operation did not touch any checkpoint, current stability-scaling run, dataset,
evaluation checkout, or active process.

The offline archive is:

```text
path: C:/Users/zixi-/CoFiTok-archive/generation/imagenet256_10pct_fixed_basis_v3_samples_2026-08-01/imagenet256_10pct_fixed_basis_v3_samples_2026-08-01_r2.tar
bytes: 3423416320
sha256: 7d34b3c24f40c5c426a10c42286feaadcd391c787d238f2b3d61c8ae8557ecf6
```

The preferred `D:` archive hub is not mounted in this desktop environment, so
the verified offline copy is under the user's `C:` profile and outside the
CoFiTok repository. A 1.3 GB interrupted transfer and a small per-file partial
copy remain explicitly quarantined under the same archive parent and are not
part of the verified archive.

## Verification

The tracked `audit_generation_sample_archive.py` auditor at implementation
commit `ed37eca` replayed the complete tar without extraction. It rejected
absolute paths, traversal, links, and extra roots, then checked the immutable
sampling manifest/progress/report chain and recomputed every sample-set digest.

```text
audit status: pass
audit sha256: 7bd53ffb22e845d1ccbcfaa82b32aeef33b3d4b9f177ada1aa07bef2b27716a3
sampling roots: 3
regular files: 20267
PNG samples: 20256
logical file bytes: 3387380474
```

The six verified sample sets are:

```text
CoFiTok formal prefix 8: 10000 / 1e88b4a727c939280dbf2689e3ba2ffcb81d9ee490d3900c55265f2a710659c1
CoFiTok diagnostic prefix 1: 64 / 7672c5d41df35a1f0ec0f0669b9e038cb5933e47e5acf67c7357f5dc9428c2da
CoFiTok diagnostic prefix 2: 64 / 7abfdd2248a6b3ea06a6b8eb40e4396cc5b8cacef25bd86e74ae7098513d9029
CoFiTok diagnostic prefix 4: 64 / d22eba34d511875c15457214cb124dff19c1fdf733d2bc308094d8a43b2e9d4f
CoFiTok diagnostic prefix 8: 64 / 536799ba6f31db24585197a0f98401ee4c089481cef19f468a262fb0855e0dac
dense formal prefix 1: 10000 / d3e615ab019f9b1fd21ebc4acafbd473404193db6c0cdfb56cec737ebd43cdd4
```

The locally audited archive and the server staging tar had the same full-file
SHA256. Immediately before removal the source roots were real directories,
contained no symlinks, had file counts `10004 / 259 / 10004`, and no process
other than the read-only pre-deletion check referenced their paths.

## Removal and recovery

The source roots consumed 3,431,690,240 allocated bytes. They were removed
first while the verified staging tar remained on the server. Only after source
absence, continued training, and free-space recovery were confirmed was the
remote staging tar removed.

```text
free before staging: 181533130752
free with staging: 178109706240
free after source removal, staging retained: 181541396480
free after staging removal: 184964812800
observed net gain: 3431682048
```

Recovery must fail closed: verify the local tar SHA, require all three original
roots to be absent, copy the tar back, extract only under
`/root/autodl-tmp/CoFiTok/checkpoints/generation`, and rerun the tracked archive
auditor. Existing gate/visual reports still retain the original absolute sample
paths, so direct replay requires restoration first.

## Remaining runway risk

This archive raises free space relative to the aggregate completion requirement
of 180,880,415,360 bytes from roughly 0.65 GB to roughly 4.08 GB. That is not
enough to absorb both the future dense rolling checkpoints and a fresh matched
post-eval sample pair. At least one additional old failed sample generation must
be archived before those stages complete. No checkpoint candidate is counted as
reclaimable, and this operation does not weaken checkpoint retention.

Machine-readable audit and retirement receipt are stored under
`artifacts/reports/generation/fixed_basis_v3_sample_archive_2026-08-01/`.
