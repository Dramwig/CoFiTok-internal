# 2026-08-18 quality-bridge 40K EMA full-warmup waiter

## Purpose

The 30K audit proves that EMA-teacher consistency activates correctly. It does
not by itself prove that the complete `10,000`-step warmup reaches full scale
without schedule drift. A second CPU-only waiter therefore verifies the last
eleven logged warmup rows from step `39,500` through exact step `40,000`.

The waiter reuses the already locked and Linux-rehearsed implementation from
the 30K transition audit. It only reads the canonical metrics, checked-in
config, resolved run manifest, and clean training checkout identity. It cannot
load a checkpoint, use GPU, signal a process, promote a model, or authorize a
larger training stage.

## Exact contract

```text
training revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
training tree: 6cef27723196fd363379bca2e7b85b1678ebd777
training branch: scale/generation-stability-quality-bridge-100k
effective batch: 64
target training steps: 100,000
EMA-teacher start: 30,000
EMA-teacher warmup: 10,000
EMA-teacher weight: 0.25
selected window: 39,500 through 40,000
selected rows: 11 at interval 50
expected first scale: 0.95
expected target scale: 1.00
```

Although the compact output highlights the final eleven rows, the waiter also
validates every canonical row through step `40,000` against the exact schedule,
finite loss fields, strictly increasing metrics, and
`samples_seen == step * 64`.

## Preflight

Before launch:

- CoFiTok had reached step `30,450`; pair monitor and watchdog were running
  with empty issues.
- GPU compute contained only trainer PID `619775` using `85,284 MiB`.
- The target report, status, and status lock were absent.
- No transition waiter process remained from the completed 30.25K audit.
- The audit checkout was detached, tracked-clean
  `2a80f6e55f71c9c25c405564bc4393bf06cb3154`, tree
  `60221d9e3ac5faf5f27fae555d2320048612f2a8`.
- Waiter source bytes/SHA256 were `20,796` /
  `10c1676d0de5e7aaa94f428a65448ab78da7fdbdc7184a76de7278aaad6d9998`.
- Generation filesystem free space was `305,813,778,432` bytes.

## Launch incident and correction

The first process, PID `816166`, exited at import time because the isolated
checkout was launched without its required `PYTHONPATH`. The failure occurred
before status or lock creation and before any audit read beyond Python import.
It created no GPU process and did not signal or modify training. Its traceback
is preserved in the waiter log as:

```text
ModuleNotFoundError: No module named 'cofitok'
log bytes: 289
log SHA256: 74eedd6d5c682b03f9b2c2a32ba9d8689863928189f7d5e8ac2d8602c216c86e
```

The corrected launch explicitly bound the exact audit checkout and its `src/`
directory through `PYTHONPATH`. The wrapper's immediate final `cat` check saw a
PowerShell-to-bash CRLF suffix and exited nonzero after the child was already
detached; a direct process and status read then verified the child normally.
This wrapper issue did not affect the waiter or training.

## Active deployment

```text
PID: 816282
started at: 2026-08-18 01:46:26 CST
status: waiting / transition_target_not_reached
poll interval: 60 seconds
timeout: 43,200 seconds
nice: 10
ionice: idle
CUDA_VISIBLE_DEVICES: empty
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
child processes: none
```

Authoritative outputs:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/schedule_audits/cofitok_ema_teacher_full_warmup_00030000_00040000_waiter_status.json
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/schedule_audits/cofitok_ema_teacher_full_warmup_00030000_00040000.json
```

The second-poll status snapshot was `992` bytes with SHA256
`b86a618a5e4e6b5eebb52cd4342da483796b823b5d446556649a7b2032d894e4`.
At that snapshot, training had advanced to step `30,500`, pair monitor and
watchdog were still healthy, trainer PID was unchanged, and the waiter had no
GPU process or child process.

Compact deployment evidence:

```text
artifacts/reports/generation/quality_bridge_40k_ema_full_warmup_waiter_deployment_2026-08-18.json
canonical LF bytes: 5,258
canonical LF SHA256: 7d2f2f9a44f7c37876ec46680e949dc0a90abfa33f0bffb3c59876ac9901ffce
Git blob OID: 279700f51659e03a21ca833236e3f1a05da15241
```

## Interpretation boundary

A passing 40K report will prove exact schedule execution and training
continuity through full EMA-teacher scale. It will not establish EMA-teacher
causal benefit, free-generation quality, broad CoFiTok superiority, promotion
readiness, full-training authorization, or release readiness. Those still
depend on the matched 50K/100K sampling and uncertainty evidence.
