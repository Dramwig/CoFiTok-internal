# 2026-08-21 quality-bridge 90K/95K/100K checkpoint waiters

## Purpose

The active matched 100K quality bridge must preserve independently auditable
checkpoint evidence after the 2026-08-21 server restart. The training monitor
intentionally performs metadata-only checkpoint inspection during live
training, so it is not sufficient evidence that a physical checkpoint payload
still matches its sidecar.

Six CPU-only waiters were therefore deployed for CoFiTok and dense at steps
90,000, 95,000, and 100,000. Each waiter remains idle until its exact target is
the active `latest.json` checkpoint. It then computes the physical payload
SHA256 and verifies the sidecar, latest pointer, Git/data/runtime identity, and
canonical metric continuity.

## Locked identities

Training remains pinned to:

```text
checkout: /tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree: 6cef27723196fd363379bca2e7b85b1678ebd777
branch: scale/generation-stability-quality-bridge-100k
dataset identity: 6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659
runtime identity: d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
effective batch: 64
```

The already proven physical-integrity implementation is reused unchanged:

```text
checkout: /root/autodl-tmp/CoFiTok/checkouts/quality-bridge-checkpoint-waiter-172388f
revision: 172388fc4d873bb1001313f979442516ea7b5069
tree: d7e54a8ea9ff5b6c6a3eea342ec787605d0ebc50
source bytes: 14,973
source SHA256: eafc4e1f7b33f8b890883ac4878944a1d57f41aca02dc0483b9e82d1259fca8b
```

This implementation already produced passing physical 50K audits on the real
approximately 1 GB CoFiTok and dense checkpoints. Those reports bind payload
SHA256 values `d7100a6e...343b4` and `db3fb9a5...33c1`, respectively.

## Deployed waiters

Deployment began at `2026-08-21T09:55:10.403830+00:00`:

| method | step | PID | initial state |
|---|---:|---:|---|
| CoFiTok | 90,000 | 71602 | waiting / checkpoint_missing |
| CoFiTok | 95,000 | 71604 | waiting / checkpoint_missing |
| CoFiTok | 100,000 | 71606 | waiting / checkpoint_missing |
| dense | 90,000 | 71608 | waiting / checkpoint_missing |
| dense | 95,000 | 71610 | waiting / checkpoint_missing |
| dense | 100,000 | 71612 | waiting / checkpoint_missing |

Target reports are written under:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/
reports/checkpoint_audits/
```

The machine-readable deployment receipt records every PID, process start tick,
command-line SHA256, initial status identity, and target report path:

```text
artifacts/reports/generation/
quality_bridge_90k_95k_100k_checkpoint_waiters_deployment_2026-08-21.json
```

An immutable copy was transferred into the authoritative remote output tree
after first proving that the destination did not exist:

```text
path:
  /root/autodl-tmp/CoFiTok/checkpoints/generation/
  stability_full_data_100k_base128_quality_bridge_v1/reports/
  checkpoint_audits/deployment_receipt_90k_95k_100k_2026-08-21.json
bytes: 6,432
SHA256: 47611e7f21d6f04e24148ec4a8ee55ba7d538cecd80793f28f20aa413de06a70
```

## Resource and authorization boundary

All six waiters were verified after launch with:

```text
parent PID: 1
CUDA_VISIBLE_DEVICES: empty
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
nice: 10
I/O class: idle
GPU compute processes added: 0
```

At the deployment snapshot CoFiTok was at step 88,050. The only GPU compute
process was the existing quality-bridge trainer PID 3773 using 89,398 MiB.

The waiters cannot launch or signal training, cannot use GPU work, cannot
authorize promotion or release, and explicitly leave
`full_300k_launch_allowed=false`. They do not alter the pinned training
revision, runbook, checkpoints, or locked paper evidence.
