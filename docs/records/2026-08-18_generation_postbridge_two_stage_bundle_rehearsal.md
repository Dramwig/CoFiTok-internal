# Generation post-bridge two-stage bundle rehearsal

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

The validated post-bridge integration can be transferred to the newly recloned
`pro6000` server without changing the active trainer or the formal checkout.
The rehearsal passed through a two-stage bundle chain in an isolated `/tmp`
clone, followed by exact cleanup of all rehearsal artifacts.

This rehearsal performed no deployment. It did not fetch into the formal
repository, move a branch, modify a checkout, signal a process, start GPU work,
sample images, evaluate checkpoints, or authorize full 300K.

## New-server prerequisite gap

The formal remote repository was still:

```text
path:          /root/autodl-tmp/CoFiTok/CoFiTok-internal
branch:        scale/generative-system
HEAD:          1ebcc15210e63a776a2ba448481cbd8bb94a4066
tracked rows:  0
```

Because the server was recloned, it did not contain the active training
revision `cf0e5faa94bf4ab38d947b921935b3b765b5537a`. Therefore, the direct
integration delta correctly failed its prerequisite precheck. Silently fetching
the missing object into the formal repository during active training was not
acceptable.

The repository did contain both prerequisites needed for a bounded bootstrap:

```text
1ebcc15210e63a776a2ba448481cbd8bb94a4066
58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

## Bundle chain

### Stage 1: reclone bootstrap

```text
advertised head: cf0e5faa94bf4ab38d947b921935b3b765b5537a
commits:         232 after formal HEAD
bytes:           40,132,480
SHA256:          f815fb52478e49cc3c1b1815e8abc46b5d3cc12a6c24fb74a9f1b0c6eeb780b1
local verify:    pass
formal verify:   pass (read-only)
isolated fetch:  pass
```

### Stage 2: post-bridge integration

```text
prerequisite:    cf0e5faa94bf4ab38d947b921935b3b765b5537a
advertised head: dc02bb8a94068bd33157b446cf4f19fa16f7a5f3
commits:         164 after training base
bytes:           8,638,145
SHA256:          9735184a62e10fddbcc7b06f9b0c5f6f6ffcb6ee5d12316882564fb3fca1b789
local verify:    pass
isolated verify: pass after bootstrap
isolated fetch:  pass
```

The isolated clone then passed `git fsck --no-dangling`. The formal repository
remained at the same HEAD, branch, and zero tracked-change count before and
after rehearsal.

## Live 50K-boundary continuity

At `2026-08-18T13:44:55+08:00`:

```text
pair:                 running / cofitok_training
issues:               []
CoFiTok:              45,700 / 100,000
dense:                0 / 100,000
latest checkpoint:    step 45,000
30K EMA transition:   pass
31K first validation: pass
40K full warmup:      pass
50K checkpoint wait: waiting / checkpoint_missing
controller identity:  observing, zero loss polls, zero mismatches
```

All terminal, capacity-probe, capacity-scaling, capacity-completion, and
full-300K supervisors remained in their source-bound `waiting` states with no
child launch.

## Cleanup

The two local bundles, the two remote `/tmp` bundles, and the isolated remote
clone were removed after verification. No checkpoint, report, project checkout,
or dataset path was deleted.

Machine-readable receipt:

```text
artifacts/reports/generation/postbridge_bundle_rehearsal_2026-08-18/
rehearsal_receipt.json
bytes:  4,376
SHA256: 8131f1a30bcfe9600327ad2d651d29fff655f977dc3798b2e400f4304258b211
```
