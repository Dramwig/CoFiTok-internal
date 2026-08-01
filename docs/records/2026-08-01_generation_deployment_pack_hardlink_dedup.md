# Generation deployment Git-pack hardlink deduplication

Date: 2026-08-01 CST

## Scope and authorization boundary

This maintenance pass reduces physical duplication among receipt-bound
large-capacity deployment checkouts. It operates only on identical Git pack and
index files and preserves every checkout path, tracked file, bundle, deployment
receipt, checkpoint, sample, and training process.

It does not execute CUDA readiness, create a full-training launch receipt, start
full ImageNet-256 training, or claim generation completion. Every deployment
receipt in this pass keeps `readiness_executed=false`,
`full_training_launch_allowed=false`, and
`formal_generation_completion_claimed=false`.

## Implementation

The first safe dedup implementation was committed as `58ae56c`. Commit
`340c2af` made result replay independent of later hardlink-count growth. Final
commit `e971fbdcf0169ca1c17f6d604a0d983a2e102370` reports existing physical
savings from the current inode graph, including a previously linked checkout
that later becomes active.

The plan builder is process-aware and fail closed:

- the canonical checkout and process-referenced checkouts are required;
- a missing receipt, tracked dirt, cross-device target, or unmatched pack is
  indeterminate;
- apply requires the explicit allow-list to equal the complete eligible set;
- plan bytes and SHA256, checkout state, pack bytes/SHA/device/inode, and
  receipt identity are rechecked before replacement;
- each replacement is an atomic hardlink swap;
- `git fsck --strict`, tracked-clean state, and the original deployment receipt
  are verified before and after apply;
- result replay ignores mutable `st_nlink` but rechecks stable file identity,
  shared inode, Git state, and the original receipt.

## Exact Linux deployment

Target checkout:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-e971fbd
```

Bundle:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/cofitok-generation-large-capacity-e971fbd-from-1ebcc15.bundle
bytes: 39478169
sha256: a9ddd821f532d855a10d92faea892cfca49bf4a75e81d071512cac3826ea3e4f
advertised head: e971fbdcf0169ca1c17f6d604a0d983a2e102370
prerequisites: 1ebcc15210e63a776a2ba448481cbd8bb94a4066, 58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

Deployment receipt:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/deployments/e971fbdcf0169ca1c17f6d604a0d983a2e102370/deployment_receipt.json
sha256: aa0aa98ac94977dc3c80d6017776d240dcaef985a0b33c84e8d1b55979cddd35
```

Linux validation was `873 passed / 2 skipped / 0 failures / 0 errors`.
All `101/101` tracked runbooks passed `bash -n`. Independent receipt replay
passed. The formal repository remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
`scale/generative-system` with clean tracked state.

## Applied maintenance

The canonical common pack remained in `checkout-58ae56c`. The initial exact plan
linked the common pack/index in eight inactive, receipt-bound historical
checkouts. Its result recorded:

```text
expected physical bytes saved: 1095527744
observed free-byte delta:       1095532544
result sha256:                  91a174163cf18be6dd78a12cd2766b5d730c8fe20316dfceb1c8d761efd5a734
```

The source-bound incremental plan at final revision `e971fbd` found exactly four
eligible files: the common pack/index in `checkout-340c2af` and
`checkout-e971fbd`. No other file was eligible. Apply retained all four paths,
then passed `git fsck` and original receipt replay for both checkouts.

```text
incremental plan sha256:        7364f006bc9c62e4e176322def5a3020838dc7aac566c14f64e4928ceaa5cfdc
incremental result sha256:      4178bd63be537c0671d6692b50ad7783796ff10e9a5f5399ccdb853b095de464
expected physical bytes saved: 273881936
observed free-byte delta:       273883136
```

Independent result replay passed. A fresh post-apply plan at the same exact
revision reported:

```text
checkout_count:                 13
already_linked files:           20
eligible files:                 0
indeterminate files:            24
required files:                 8
currently_saved_bytes:          1369409680
potential_physical_bytes_saved: 0
post-plan sha256:               e3ee61157346cce88fa5eb194f541d7e307c47b61fb310c166ac1d17348e41cc
```

`currently_saved_bytes` is the exact logical size of noncanonical pack/index
paths sharing the canonical inode. The sum of observed filesystem deltas is
`1,369,415,680` bytes; the 6,000-byte difference is filesystem accounting
granularity. Unique packs remain indeterminate and were not modified.

## Live training boundary

During final deployment and maintenance, the active stability matched 50K run
continued under trainer PID `319202`; the readiness waiter remained bound to
`checkout-5dd3488` and reported `full_training_launch_allowed=false`. No process
was stopped, restarted, or moved to another checkout. The authoritative live
state remains the server pair monitor and readiness waiter, not the snapshots in
this record. The final read-only snapshot for this maintenance pass was
`running/cofitok_training/issues=[]`, CoFiTok `35,000/50,000`, dense
`0/50,000`, and waiter `waiting_for_passing_stability_gate`. Filesystem free
space after maintenance was `186,811,084,800` bytes, only `5,930,669,440`
bytes above the aggregate completion requirement of `180,880,415,360`; every
later readiness or launch decision must recompute this value.

Small source-bound evidence is stored in
`artifacts/reports/generation/deployment_pack_dedup_2026-08-01/`.
