# 2026-08-17 quality-bridge 30K EMA-teacher transition waiter

## Purpose

The active full-data matched 100K quality bridge enables EMA-teacher
consistency at optimizer step `30,000` with a `10,000`-step linear warmup and
weight `0.25`. A checkpoint audit alone can verify payload integrity at 30K but
cannot prove that the first active training rows actually follow the configured
schedule.

`scripts/wait_generation_consistency_schedule_transition.py` adds a reusable,
CPU-only, read-only transition waiter. It never loads a checkpoint, uses no GPU,
and cannot signal either training or unrelated processes.

## Exact 30K contract

The planned CoFiTok transition audit is source-bound to:

```text
training revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
training tree: 6cef27723196fd363379bca2e7b85b1678ebd777
training branch: scale/generation-stability-quality-bridge-100k
effective batch: 64
target training steps: 100,000
EMA-teacher start: 30,000
EMA-teacher warmup: 10,000
EMA-teacher weight: 0.25
metrics log interval: 50
first active row: 30,050, expected scale 0.005
audit target row: 30,250, expected scale 0.025
minimum active rows: 5
```

The waiter verifies:

1. exact training checkout revision, branch, tree, and tracked-clean state;
2. checked-in config and resolved `run_manifest.json` schedule equality;
3. strictly increasing canonical metrics and `samples_seen == step * 64`;
4. exact step-30K disabled boundary with zero scale and zero loss;
5. five exact active rows from 30,050 through 30,250;
6. exact linear scales, strictly increasing warmup, positive finite teacher loss,
   and finite total/epsilon/gradient values;
7. content identities for the waiter, config, run manifest, metrics snapshot, and
   final report.

The report is diagnostic and non-authorizing. It cannot change the quality
bridge, launch another training stage, promote a model, or support a broad
generation-quality claim.

## Local validation

Development uses an isolated clean worktree:

```text
C:/qbschedule
branch: analysis/generation-quality-bridge-ema-transition-v1
base: cf0e5faa94bf4ab38d947b921935b3b765b5537a
```

Current targeted validation:

```text
10 tests passed
68 transition + training-progress + loss + trajectory tests passed
ruff format/check passed
Python compile passed
git diff --check passed
```

The tests cover the exact disabled-to-active boundary, scale drift, premature
loss, zero active loss, missing target row, samples-seen drift, canonical tail
readiness, resolved run-manifest binding, checked-in config binding, and the
non-authorizing waiting status.

Locked implementation identity:

```text
revision: 2a80f6e55f71c9c25c405564bc4393bf06cb3154
tree: 60221d9e3ac5faf5f27fae555d2320048612f2a8
incremental bundle bytes: 8,947
incremental bundle SHA256: b20a12e88ef9bd462fbf4aedd9ad9a6eb81203ec44f06e08d5ee21210ebb4ee8
bundle prerequisite: cf0e5faa94bf4ab38d947b921935b3b765b5537a
waiter source SHA256: 10c1676d0de5e7aaa94f428a65448ab78da7fdbdc7184a76de7278aaad6d9998
```

The bundle was verified against the active training repository, fetched only
into an isolated detached Linux checkout, and rehearsed with CUDA hidden plus
one OMP/MKL thread. The same 68-test group and Python compilation passed;
tracked status remained empty and the active training checkout stayed at
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`.

## Execution boundary

The script must be committed, bundled, Linux-rehearsed with CUDA hidden, and
launched from a separate clean checkout before the 30,250 row is reached. Its
output belongs under the active quality-bridge `reports/` tree. It must not
modify the active training checkout or compete for GPU resources.

## Deployment

The exact waiter was deployed from:

```text
/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-ema-transition-2a80f6e
```

Runtime identity:

```text
PID: 798909
CUDA_VISIBLE_DEVICES: empty
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
nice: 10
ionice: idle class
poll interval: 30 seconds
timeout: 43,200 seconds
```

Authoritative outputs:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/schedule_audits/cofitok_ema_teacher_transition_00030000_00030250_waiter_status.json
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/schedule_audits/cofitok_ema_teacher_transition_00030000_00030250.json
```

Initial status was `waiting / transition_target_not_reached`. The process was
alive, the active training checkout remained tracked-clean, and no GPU work was
started by this deployment.

## Verified result

The 30K checkpoint and the disabled-to-active EMA-teacher boundary both passed
their source-bound audits on `2026-08-18` CST.

The physical checkpoint audit verified:

```text
checkpoint: checkpoint_step_00030000.pt
bytes: 1,010,937,514
SHA256: 1b6492d5095c3da8cc0853bd7efcdac5ed36cd91aced8a19000ee95aebb5ac5e
audit bytes: 5,221
audit SHA256: 3bdc9bed421b1fe4080034a3862bcab851559d26192577517e1eb757d504d7a9
status: pass
physical payload SHA256 verified: true
```

At exact step `30,000`, both EMA-teacher scale and loss were zero. The five
required active rows were:

| step | scale | teacher loss | samples seen |
|---:|---:|---:|---:|
| 30,050 | 0.005 | 0.0005355001 | 1,923,200 |
| 30,100 | 0.010 | 0.0014735833 | 1,926,400 |
| 30,150 | 0.015 | 0.0029341625 | 1,929,600 |
| 30,200 | 0.020 | 0.0005869395 | 1,932,800 |
| 30,250 | 0.025 | 0.0013662397 | 1,936,000 |

The scales are strictly increasing, every active teacher loss is positive and
finite, all audited total/epsilon/gradient values are finite, and every row
satisfies `samples_seen == step * 64`. The authoritative transition report is:

```text
bytes: 13,304
SHA256: 8ac14c70faef01930d99273f658e5dd6e659e10d5f85d560fc13cebb2e9faa35
status: pass
```

The checked-in config retains its default `16x4` runtime while the resolved run
manifest records the launch-selected `64x1` runtime. Both have effective batch
`64`; the schedule, target steps, Git revision, branch, and clean-state binding
match exactly.

Training continued beyond the audited boundary. At the continuity snapshot it
had reached step `30,350`, scale `0.035`, and `1,942,400` samples. The runbook,
pair monitor, watchdog, and trainer remained alive; the pair monitor and
watchdog both reported `running` with empty issues. GPU compute contained only
trainer PID `619775` using `85,284 MiB`, and the active training checkout stayed
tracked-clean at `cf0e5faa94bf4ab38d947b921935b3b765b5537a`.

The compact local evidence receipt is:

```text
artifacts/reports/generation/quality_bridge_30k_ema_transition_evidence_2026-08-18.json
canonical LF bytes: 7,411
canonical LF SHA256: e120c4b4ffedd04a33fed682b6b5abbb141716c7070ebc988f295a69a9f0db36
Git blob OID: 3d0fa3bcf767625bf9a37fa3f23d0e2e2cc649c1
```

This proves checkpoint integrity, schedule execution, sample accounting, and
post-transition continuity. It does not prove EMA-teacher causal benefit,
sample quality, broad generation superiority, promotion readiness, or release
readiness. The full-data matched quality bridge must continue unchanged.
