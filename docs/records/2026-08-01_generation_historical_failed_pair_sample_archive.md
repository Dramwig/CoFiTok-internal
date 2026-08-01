# Historical failed-pair sample archive

Date: 2026-08-01 CST

## Outcome

Six completed sample roots from two superseded, failed 10K promotion gates were
archived off the server, replay-audited, and removed from the generation
filesystem. Checkpoints, training/evaluation reports, gate reports, parent run
directories, the active stability-scaling queue, and datasets were retained.

The offline archive is:

```text
path: C:/Users/zixi-/CoFiTok-archive/generation/historical_failed_compressed_rankcomplete_pairs_2026-08-01/historical_failed_compressed_rankcomplete_pairs_2026-08-01.tar
bytes: 6474926080
sha256: 9946c301024e41816e207d6f749cb4385a9af96fb2364736a37fcf8b633438e7
```

## Selection boundary

Both archived pairs were already terminal `fail / hold` results:

- The compressed matched pair failed FID tolerance, absolute FID quality,
  endpoint tolerance, and ordered-prefix-path gates. Its locked promotion-gate
  report SHA256 is
  `2c2d27401834e4d0fd2138ec886595f84ffa96679fad5be9bc189c717fba7eaf`.
- The rank-complete v2 pair repaired ordered-prefix ranking but still failed FID
  tolerance, absolute FID quality, and endpoint tolerance. Its locked
  promotion-gate report SHA256 is
  `a3ac8ec7ee515ff5fe1eb04062f48c0b5ade2e548afd64322210dc969377cfe1`.

Only each pair's 10,000-sample formal roots and CoFiTok 4-prefix x 64-image
diagnostic roots were selected. This archive does not weaken checkpoint
retention and is not a generation-quality promotion.

## Verification

The tracked `audit_generation_sample_archive.py` auditor replayed the complete
tar without extraction. It rejected absolute paths, traversal, links, and
unexpected roots, validated each immutable sampling manifest/progress/report
chain, and recomputed all sample-set digests.

```text
audit status: pass
audit bytes: 14756
audit sha256: 9e95cc35a4911440ab3d66a8dc066819853cc6164528b6392059a51e8b9d6219
sampling roots: 6
regular files: 40534
PNG samples: 40512
logical file bytes: 6402280151
```

The local archive and server staging tar matched in bytes and full-file SHA256.
Immediately before removal, all six source roots were real directories, no
symlink was found, and no active process referenced the archived paths.

## Removal and recovery

The source roots declared `6,479,757,312` allocated bytes. They were removed
first while the verified staging tar remained available on the server. After
source absence and active training health were confirmed, the exact staging tar
was removed.

```text
free with staging: 178488901632
free after source removal, staging retained: 184968998912
free after staging removal: 191443927040
source-removal free-byte delta: 6480097280
staging-removal free-byte delta: 6474928128
```

The small differences between declared allocation and observed deltas are
consistent with concurrent writes from the active trainer. Post-removal health
was `running`, monitor `issues=[]`, finite metric row at step 37,450, trainer PID
`918650`, and 85,294 MiB GPU allocation.

Recovery is fail-closed: verify the offline tar SHA256, require all six original
roots to be absent, copy the tar to the server, extract only under
`/root/autodl-tmp/CoFiTok/checkpoints/generation`, and rerun the tracked archive
auditor. Gate and visual reports retain the original absolute sample paths, so
direct replay requires restoration first.

Machine-readable audit and retirement receipt are stored under
`artifacts/reports/generation/historical_failed_pair_sample_archive_2026-08-01/`.

## Post-archive aggregate runway

The lightweight checker at deployed revision
`1604dc45c21ec73d9debf8a81e3d7367cd9fb220` was rerun against the pinned,
previously physically replayed retention inventory:

```text
inventory sha256: 62ae7388983e046fc0d0afbb7db2059e1afb5c2674880881e20411cd66f68cd5
status: pass
filesystem free bytes: 191443562496
required free bytes: 180880415360
current headroom bytes: 10563147136
currently reclaimable bytes: 0
physical checkpoint hashes replayed: false
report sha256: 1f977df3beb395a58d733ee34b2534b15379ede0fd0f2f4cdad0c1ec8a91371f
```

The report is bound to the inventory SHA and does not count potential archive
candidates as current capacity. Its headroom covers the existing aggregate
completion model for the active matched 50K queue and post-evaluation, subject
to refresh as the queue writes new checkpoints and samples.

## Authorization boundary

This operation creates storage runway only. It does not authorize full 300K
training, change the pinned checkpoint-retention inventory, or claim formal
generation completion. CUDA readiness and all promotion/final gates remain
mandatory.
