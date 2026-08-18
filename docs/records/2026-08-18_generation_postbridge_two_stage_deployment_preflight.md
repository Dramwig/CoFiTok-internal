# Full-data 100K post-bridge two-stage deployment preflight

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

The fail-closed two-stage post-bridge deployment path was exercised against the
new `pro6000` formal repository in its default read-only `preflight` mode.  The
bundle chain itself passed, including isolated bootstrap/integration fetches and
`git fsck --no-dangling`, but the preflight correctly returned exit `76` with
`status=blocked` and `apply_ready=false` after discovering 92 target-added paths
that already exist as untracked files in the formal checkout.

This is a safety finding rather than a bundle-chain failure.  All 92 conflicts
are ordinary byte-identical files relative to the target Git blobs.  There are
zero divergent files, directories, symlinks, missing local paths, or ignored
conflicts.  No automatic resolution was performed.

The preflight did not fetch into, fast-forward, or otherwise modify the formal
repository.  It did not quarantine, move, delete, or overwrite any untracked
file.  It did not signal, pause, restart, or replace the active full-data 100K
trainer, controller, monitor, guard, or waiter, and it did not launch training,
sampling, evaluation, promotion, release, or full-300K work.

## Implementation identity

The tested implementation is:

```text
worktree: C:/qbintegration
branch:   integration/generation-postbridge-hardening-v1
revision: 6cd02bb9a3adcd76056bbcbfd7dbbd5b7fb308eb
tree:     112fb8bb665e60d53c5b92193ae8fc9a12e6fc0c
status:   tracked clean
```

The four implementation commits are:

```text
fb72266  Add fail-closed postbridge bundle deployment
0792b00  Report blocked postbridge preflights
eaf4bac  Classify postbridge deployment conflicts
6cd02bb  Quarantine identical postbridge conflicts
```

The deployed preflight entrypoint was `60,030` bytes with SHA256
`284efe1ecd2b21346e78a87714161ae729d6e4a402a0397191891e93921b7484`.

## Two-stage bundle chain

The formal repository is pinned at:

```text
path:     /root/autodl-tmp/CoFiTok/CoFiTok-internal
branch:   scale/generative-system
revision: 1ebcc15210e63a776a2ba448481cbd8bb94a4066
tree:     659fa94726c4aec0afef49904f82b828bb62872b
```

The integration bundle cannot be verified directly from that checkout because
its prerequisite is `cf0e5faa...`, which is not present there.  The preflight
therefore verifies the exact two-stage chain in an independent clone:

1. bootstrap `cf0e5faa...` from the two formal prerequisites;
2. integration target `6cd02bb...` from bootstrap `cf0e5faa...`;
3. verify target tree `112fb8bb...` and run `git fsck --no-dangling`.

The source identities are:

| source | bytes | SHA256 |
|---|---:|---|
| bootstrap bundle | 40,134,667 | `21a3fc0dc38a563db334823e4b6e17659702915d2f327999e100d88382260c82` |
| integration bundle | 8,664,485 | `5c87b1ca2ced8347fae41ef1e18e56fcb35190ac8cb33b236546ed8c31d58c01` |
| manifest | 4,044 | `53e6799c7026df001f3732e7485037fa2fda7f685960e157b0a6276d5d533713` |
| preflight report | 258,432 | `56cd970870e700618ba92ceb00a4e0ce1e77f2648a1583095c9f2d9b885e68aa` |

The bootstrap advertises exactly `cf0e5faa...` and requires exactly
`1ebcc152...` plus `58d83bfc...`.  The integration bundle advertises exactly
`6cd02bb...` and requires exactly `cf0e5faa...`.

## Conflict result

The target adds 1,391 paths relative to the formal revision.  The formal
checkout has 92 colliding untracked paths:

```text
conflicts:                    92
byte-identical regular files: 92
divergent files:               0
directories:                   0
symlinks:                      0
missing local paths:           0
ignored local paths:           0
```

The tool recomputed local file bytes/SHA256 and the target Git blob bytes/SHA256
for every conflict.  Because every conflict is byte-identical, a future explicit
`apply` may use the opt-in `--identical-conflict-archive` flow to move those
files to an external recoverable archive, fast-forward, and then verify that the
new tracked blobs reproduce the original bytes.  This path was not run here.

`apply` remains prohibited until the bound 100K execution and pair monitor are
terminal and no matching trainer/controller process remains.  It additionally
requires explicit `--mode apply`, the exact manifest SHA256, the exact target
revision, and—while these conflicts exist—an external archive path.  A future
apply must rerun the complete preflight; this report is not reusable as an
authorization token.

## Formal repository preservation

The formal repository before and after preflight was identical:

```text
HEAD:                 1ebcc15210e63a776a2ba448481cbd8bb94a4066
branch:               scale/generative-system
tracked status rows:  0
full status rows:     89
full-status SHA256:   637e549971bf34c161c9629b899d298daa6b50506fbefecfa62e0dad69c41b05
refs SHA256:          8d1fc5fde7d228bcdfcf2dbfbee571eeacaa115733ed6ec1b5f820fa89432225
object-store SHA256:  1fb487517e3e47945f01e4f6b264834b85e8ad9fe7a6e9e8c2cdc1e08c51ee4b
fetch performed:      false
fast-forward:         false
```

## Active 100K snapshot

At `2026-08-18T15:20:46+08:00`, after the preflight and local validation:

```text
pair status/stage: running / cofitok_training
pair issues:       []
CoFiTok:           47,750 / 100,000
images seen:       3,056,000
dense metrics:     absent
quality result:    absent
50K checkpoint:    not yet published
50K waiter:        waiting / checkpoint_missing
GPU trainer PID:   619775
unrelated GPU:     none observed by the bound monitor
free bytes:        321,599,909,888
```

The formal repository was still at `1ebcc152...`, tracked-clean, with the same
89-row full-status digest.  The active training checkout remained bound to
`scale/generation-stability-quality-bridge-100k@cf0e5faa...`.

## Validation

The final local validation on the exact clean target included:

```text
targeted new-tool tests: 14 passed in 89.15s
full suite:              1,632 passed, 9 skipped, 0 failed, 0 errors
full-suite wall time:    772.04s
Python compileall:       pass
both CLI --help paths:   pass
git diff --check:        pass
```

The full-suite JUnit report contains 1,641 collected tests and is `247,924`
bytes with SHA256
`f839c93d85946ad6258415b78ef09a57f8d5c355537695e4d6673201e8101eb0`.

## Cleanup and authority boundary

After the durable record and machine receipt were written and validated, the
exact remote temporary directory used only for bundle transfer and preflight
was removed and verified absent:

```text
remote: /tmp/cofitok-postbridge-preflight-fb72266
```

The corresponding local temporary directory remains present:

```text
C:/Users/17194/AppData/Local/Temp/cofitok-postbridge-preflight-fb72266
```

Two exact-path `Remove-Item -LiteralPath ... -Recurse` attempts were rejected by
the host execution policy before process creation.  No alternate shell or
deletion bypass was used.  The retained directory contains only the already
hashed local bundle and manifest transfer copies; it is outside every project,
checkpoint, sample, and active execution root.

No project checkout, checkpoint, sample, report root, active process, or GPU
allocation was deleted or modified.  The preflight remains non-authorizing:
training, sampling, evaluation, promotion, release, full 300K, process signals,
and automatic conflict resolution are all disallowed.

Machine-readable receipt:

```text
artifacts/reports/generation/
postbridge_two_stage_deployment_preflight_2026-08-18/preflight_receipt.json
```
