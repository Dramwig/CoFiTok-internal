# Stability sampling-confirmation distribution-support hardening

Date: 2026-08-05

## Outcome

The dormant matched 10K sampling confirmation now fails closed on distribution
support as well as FID. The exact code candidate is:

```text
development branch: scale/generation-stability-sampling-recovery-v1
execution branch identity: scale/generation-large-capacity
revision: 11d8f954030915f1d8848594683ac70518ce39bc
tree: 95a5e656b1e0a8b772ce5d1fe1badc487ef98d16
parent: cdec65745868d39b16646ba1b52bd182592b82a0
```

This revision supersedes
`aed68853914dc0ecf4b3f15d5d090ddee740d1df` and all earlier sampling
candidates. Those revisions remain historical evidence and must not be used as
execution targets.

## Why the former contract was insufficient

The frozen 10K metrics physically report:

| method | FID | precision | recall |
|---|---:|---:|---:|
| CoFiTok | 138.2970249527 | 0.7505000234 | 0.00867999997 |
| dense identity | 151.4476773464 | 0.7832999825 | 0.00977999996 |

The frozen schema-v2 gate predates the stability distribution-support check,
so its sole failed gate is `absolute_fid_quality`. The former confirmation
builder also decided `quality_confirmed` from four FID checks only. A candidate
could therefore reduce FID while retaining recall near `0.009` and be reported
as confirmed. That would mistake severe mode-coverage collapse for recovered
sample quality.

## Corrected contract

The plan now binds the existing stability-scaling schema-v4 constants from
`cofitok.generation_gate` rather than introducing a second threshold policy:

```text
min_precision = 0.10
min_recall = 0.10
max_precision_regression versus matched dense = 0.05
max_recall_regression versus matched dense = 0.05
```

The preflight and confirmation report schemas are both bumped to version 2.
The report can pass only when every existing FID check and all four new checks
are true:

- CoFiTok precision is at least `0.10`;
- CoFiTok recall is at least `0.10`;
- CoFiTok precision is no more than `0.05` below matched dense;
- CoFiTok recall is no more than `0.05` below matched dense.

The builder exposes separate `fid_quality_confirmed` and
`distribution_support_confirmed` booleans, while `quality_confirmed` is their
conjunction. It rejects missing, extra, non-finite, or weakened distribution
thresholds in both the physical plan and a loaded preflight. A passing report
remains non-authorizing and is not a promotion gate.

## Local verification

At exact candidate `11d8f954030915f1d8848594683ac70518ce39bc`:

- Python compile, both builder `--help` calls, JSON parse/plan validation, and
  `git diff --check`: pass;
- sampling/approval/confirmation tests plus runbook entrypoints: `65 passed`;
- the seven directly selected distribution-support tests: `7 passed`;
- full repository excluding the four parent-layout paper tests: `1,138`
  collected, `1,132 passed`, `6 skipped`, `0 failed`;
- the exact candidate paper-test blob was replayed against the real sibling
  `paper/` layout: `4 passed`.

One monolithic full-suite invocation exceeded the 900-second tool window
without returning a result, so it was not counted as evidence. The same 139
test files were then partitioned deterministically into four disjoint groups;
all four groups exited zero and their union produced the counts above.

## Isolated Linux rehearsal

The prerequisite-aware bundle used for the accepted rehearsal was:

```text
bytes: 40207095
sha256: d526b4e332a4bdd7db11fa42200e54615fa3166ab5c38b53fbcc8170fa59771e
advertised refs: one, HEAD=11d8f954030915f1d8848594683ac70518ce39bc
prerequisites: 1ebcc15210e63a776a2ba448481cbd8bb94a4066,
               58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

In the isolated pro6000 checkout, the exact commit/tree was attached to the
actual execution branch `scale/generation-large-capacity`, with
`CUDA_VISIBLE_DEVICES` empty. The 65 targeted tests, Python compile, builder
CLI help, direct plan reconstruction, four random-stream independence checks,
exact threshold check, and both runbooks' native `bash -n` all passed. Full
porcelain remained empty after testing.

An earlier CPU-only rehearsal used the development branch name. Its code and
tests passed, but it was explicitly rejected as execution-identity evidence;
the accepted rehearsal above matches the runbook's branch contract.

The formal checkout remained exactly
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
`scale/generative-system`. Its tracked state stayed clean and its complete
porcelain SHA256 was unchanged before and after rehearsal. The remote bundle,
remote checkout, and local bundle were removed afterward. The machine-readable
receipt is:

```text
artifacts/reports/generation/sampling_confirmation_distribution_support_rehearsal_2026-08-05.json
```

## Live and authorization boundary

FieldScope PID `433140` still owns unrelated GPU compute, with about
`15,412 MiB` in use. It was not modified or signaled. At the final snapshot,
available storage was `297,856,917,504` bytes and the sampling-recovery,
sampling-confirmation, and authoritative 100K quality-bridge output roots were
all absent.

No approval sentinel was created. No sampling, confirmation, 100K bridge,
training, or full-300K process was launched. A future 1K recovery diagnostic
still requires a clean checkout at exact revision `11d8f954...` on execution
branch `scale/generation-large-capacity`, an idle GPU, the exact recovery-only
user approval, its content-bound sentinel, and the explicit boolean opt-in.
Recovery approval cannot authorize the independent 10K confirmation; neither
stage can authorize training or full 300K.
