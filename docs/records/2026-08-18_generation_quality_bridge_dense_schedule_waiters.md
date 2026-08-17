# 2026-08-18 quality-bridge dense schedule waiters

## Purpose

The active full-data 100K runbook trains CoFiTok to 50K, evaluates its first
milestone, and then starts dense. The runbook verifies terminal training and
milestone outputs, but it does not independently preserve source-bound evidence
for dense's first post-EMA-teacher validation event or the fully warmed
EMA-teacher schedule.

Two CPU-only, read-only waiters were deployed before dense training starts so
the dense side will receive the same schedule evidence already collected for
CoFiTok.

## Locked identities

Training identity:

```text
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree: 6cef27723196fd363379bca2e7b85b1678ebd777
branch: scale/generation-stability-quality-bridge-100k
tracked dirty: false
```

Waiter implementation:

```text
checkout: /root/autodl-tmp/CoFiTok/checkouts/quality-bridge-ema-transition-2a80f6e
source bytes: 20,796
source SHA256: 10c1676d0de5e7aaa94f428a65448ab78da7fdbdc7184a76de7278aaad6d9998
```

Dense config:

```text
path: /tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal/configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json
bytes: 2,459
SHA256: f8f1c1967468bef4db1e9a00e8f80c3da5af22e417d8ab4864afcb4fb68f884c
```

The exact schedule is start step `30,000`, warmup `10,000`, weight `0.25`,
log interval `50`, effective batch `64`, and target training length `100,000`.

## First post-teacher validation waiter

```text
PID: 821572
target: 31,000
first active step: 30,050
minimum active rows: 20
poll interval: 60 seconds
timeout: 604,800 seconds
initial status: waiting / metrics_missing
```

Target report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/schedule_audits/dense_ema_teacher_first_post_validation_00030000_00031000.json
```

Initial status identity:

```text
bytes: 975
SHA256: 3251b6bfbabcfc7ad35f29e08fa9a58294124620a6358f4a60bf166ea81d459a
```

At target, the waiter will verify all canonical rows through 31K, exact schedule
scales, positive finite active losses, sample accounting, Git/config/run-manifest
identity, and the 31K validation row. Interpretation of the validation value
remains a separate descriptive audit.

## Full-warmup waiter

```text
PID: 821573
target: 40,000
selected window: 39,500 through 40,000
minimum active rows: 11
poll interval: 60 seconds
timeout: 604,800 seconds
initial status: waiting / metrics_missing
```

Target report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/schedule_audits/dense_ema_teacher_full_warmup_00030000_00040000.json
```

Initial status identity:

```text
bytes: 965
SHA256: f816e0b5fa092988b3d354d25822514b3b852a8e8cc044828c7f795ebf6c0136
```

## Resource and safety boundary

Both waiters run with:

```text
CUDA_VISIBLE_DEVICES: empty
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
nice: 10
ionice: idle class
parent PID: 1
```

Immediately after deployment, GPU compute still contained only CoFiTok trainer
PID `619775` using `85,284 MiB`; CoFiTok continued to step `31,150`. The waiters
cannot launch training, use GPU, signal a process, authorize promotion, release
a model, or authorize 300K training. A waiter failure is diagnostic evidence
only and must not trigger a restart.

Compact deployment evidence:

```text
artifacts/reports/generation/quality_bridge_dense_schedule_waiters_deployment_2026-08-18.json
bytes: 4,720
SHA256: 95dca0e461c0ec186277a9ad696ef5e945f326b520cb6d05696292a1d24abe61
Git blob OID: dbda89299dc1ef593a38abaaa33e853c9454eb8f
```
