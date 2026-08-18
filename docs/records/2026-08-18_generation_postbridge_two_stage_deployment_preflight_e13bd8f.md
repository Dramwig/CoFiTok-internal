# Post-bridge two-stage deployment preflight for `e13bd8f`

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

The default read-only two-stage deployment preflight was rerun for the current
post-bridge integration target
`e13bd8fdedd0054cb7e292c2e04355751233b7f7`. The complete bootstrap and
integration bundle chain passed in an isolated clone, including
`git fsck --no-dangling`, and resolved the exact target tree
`a7bc8d0756dbcd9d00ff2b0dff41c79e2b9d0a7f`.

The preflight correctly remained blocked with `apply_ready=false`. The target
adds 1,397 paths relative to the formal revision and 92 of those paths already
exist as untracked files in the formal checkout. Content analysis proved that
all 92 are byte-identical regular files relative to the target Git blobs. There
are no divergent files, directories, symlinks, missing paths, or ignored
conflicts, and no automatic conflict resolution was performed.

This was a safety-only preflight. It did not fetch into or fast-forward the
formal repository, did not archive or alter untracked files, did not signal or
modify the active full-data matched 100K execution, and did not launch training,
sampling, evaluation, promotion, release, or full-300K work.

## Exact target

```text
worktree: C:/qbintegration
branch:   integration/generation-postbridge-hardening-v1
revision: e13bd8fdedd0054cb7e292c2e04355751233b7f7
tree:     a7bc8d0756dbcd9d00ff2b0dff41c79e2b9d0a7f
```

The operational code is revision
`9a90c4c629b53fafc87707105817fe6cebebf713`. The target revision adds only its
two validation-evidence files after that operational revision. The deployment
entrypoint remains 60,030 bytes with SHA256
`284efe1ecd2b21346e78a87714161ae729d6e4a402a0397191891e93921b7484`.

## Bundle and report identities

| source | bytes | SHA256 |
|---|---:|---|
| bootstrap bundle | 40,134,667 | `21a3fc0dc38a563db334823e4b6e17659702915d2f327999e100d88382260c82` |
| integration bundle | 8,689,100 | `46e8604650385a315d7d4a3e0b5e8b1f5345f81fab206dbbcef943eeba3fbafc` |
| manifest | 4,044 | `5d5839697222d779a6b73c8877177c6613c44edd8210c34d30a7f5910474a5df` |
| remote preflight report | 258,914 | `8c03a41a16e559a6aff44ecaa46e2f67fb5eb501a04e8450405f7c61d78de4e8` |

The bootstrap advertises exactly `cf0e5faa...` and requires exactly the formal
prerequisites `1ebcc152...` and `58d83bfc...`. The integration bundle advertises
exactly `e13bd8f...` and requires exactly `cf0e5faa...`. Direct integration
verification against the pinned formal repository is unavailable until the
bootstrap object is supplied, so verification was intentionally performed only
inside an independent no-local, no-hardlink clone.

## Conflict result

```text
target-added paths:             1,397
untracked conflicts:               92
byte-identical regular files:      92
divergent files:                    0
directories:                        0
symlinks:                           0
missing local paths:                0
ignored local paths:                0
automatic resolution performed: false
```

A future apply may use the existing explicit identical-conflict archive path,
but only after the bound 100K trainer and controller are terminal. That future
operation must rebuild a fresh manifest and rerun the complete preflight against
the then-current formal checkout; this report is not reusable as an apply token.

## Formal repository preservation

The formal repository was byte-for-byte unchanged across the preflight:

```text
path:                /root/autodl-tmp/CoFiTok/CoFiTok-internal
branch:              scale/generative-system
revision:            1ebcc15210e63a776a2ba448481cbd8bb94a4066
tree:                659fa94726c4aec0afef49904f82b828bb62872b
tracked status rows: 0 -> 0
full status rows:    89 -> 89
full-status SHA256:  637e549971bf34c161c9629b899d298daa6b50506fbefecfa62e0dad69c41b05
refs SHA256:         8d1fc5fde7d228bcdfcf2dbfbee571eeacaa115733ed6ec1b5f820fa89432225
object-store SHA256: 1fb487517e3e47945f01e4f6b264834b85e8ad9fe7a6e9e8c2cdc1e08c51ee4b
fetch performed:     false
fast-forward:        false
```

## Active 100K snapshot

At `2026-08-18T16:19:21.678813+08:00`:

```text
pair:                    running / cofitok_training
issues:                  []
CoFiTok:                 49,000 / 100,000
images seen:             3,136,000
latest bound checkpoint: step 45,000 / metadata_verified
50K checkpoint:          absent
50K integrity waiter:    waiting / checkpoint_missing / PID 822372
dense run:               absent
quality result:          absent
GPU trainer PID:         619775
unrelated GPU processes: []
free bytes:              321,598,697,472
```

The 50K checkpoint integrity waiter is already active and bound to the exact
training revision, tree, dataset identity, runtime identity, and effective
batch. Its `checkpoint_missing` state is expected before publication of the
50K checkpoint; it is not a scheduling gap.

## Validation binding

The operational revision passed 31 focused tests and the complete 1,646-test
repository suite: 1,637 passed, 9 skipped, with zero failures or errors. The
JUnit artifact is 248,572 bytes with SHA256
`85c8b23b8669cd46ab3ca25628f35958899ce2a641d68a6b0bb54e8665a0f2c0`.
The exact validation receipt is:

```text
artifacts/reports/generation/
postbridge_waiter_heartbeat_integration_2026-08-18/validation_receipt.json
```

## Cleanup and boundary

After this durable record and its machine receipt passed JSON and diff
validation, the exact remote transfer/preflight directory
`/tmp/cofitok-postbridge-preflight-e13bd8f` was resolved, verified as a
non-symlink child of `/tmp`, removed, and verified absent. It contained exactly
five temporary files: the two bundles, deployer copy, manifest, and preflight
report. No project checkout, checkpoint, sample, active report root, or training
artifact was removed. Bundle/manifest sources remain local or reproducible, and
the report identity plus all decision-relevant fields are retained in the
machine receipt. The local temporary directory remains outside all project and
checkpoint roots and was not removed.

Machine-readable receipt:

```text
artifacts/reports/generation/
postbridge_two_stage_deployment_preflight_e13bd8f_2026-08-18/preflight_receipt.json
```
