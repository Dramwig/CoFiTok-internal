# Terminal distribution-support diagnostic preparation

Date: 2026-08-20

## Outcome

Commit `4119ce4817795c7465ab9eacf448a44405c752d8` implements a CPU-only,
permanently non-authorizing terminal distribution-support diagnostic for the
full-data 100K quality bridge. The diagnostic does not select a training
recipe. It runs only if the canonical v2 follow-up decision selects the exact
route:

```text
diagnose_terminal_distribution_support_then_recipe_probe
```

An unselected route exits successfully without starting a child. A selected
route must bind the exact terminal `quality_bridge_result.json`; source drift
fails closed. The waiter never authorizes or launches training, sampling,
evaluation, a recipe probe, promotion, release, or full 300K execution.

## Diagnostic scope

The implementation replays the canonical v2 follow-up decision and terminal
quality result, then reopens both matched 10,000-sample DDIM-100 cohorts. It
revalidates sampling manifests, progress, checkpoint identity, sample-set
identity, evaluator identity, and the common real set before reading images.

For CoFiTok, dense identity, and a deterministic class-balanced real cohort,
the report measures:

- exact encoded and decoded-pixel duplicates;
- dHash uniqueness and nearest-neighbor diversity;
- descriptive RGB contrast and saturation;
- horizontal and vertical neighbor differences;
- Laplacian response statistics.

These statistics are descriptive. They have no post-hoc quality threshold and
cannot establish a causal training or recipe claim.

## Exact implementation identity

```text
branch: analysis/generation-terminal-distribution-support-v1
revision: 4119ce4817795c7465ab9eacf448a44405c752d8
tree: db57efdd7c6c7cb438d2c780fd31c2cd80259fb9
tracked dirty: false
```

The prerequisite-bound implementation bundle was verified locally and on
`pro6000`:

```text
bytes: 43,709,405
sha256: d53fe614fcd486cb113908821b5f51b0e924a49f67935f63b6db429dfbf70467
advertised revision: 4119ce4817795c7465ab9eacf448a44405c752d8
prerequisites:
  1ebcc15210e63a776a2ba448481cbd8bb94a4066
  58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

## Isolated Linux rehearsal

The rehearsal used:

```text
/tmp/cofitok-terminal-support-rehearsal-4119ce4-20260820a/CoFiTok-internal
```

The checkout had a copied sibling `paper/` directory, CUDA was hidden, and
CPU/IO priority was lowered. Verification completed as follows:

- focused diagnostic tests: 18 passed;
- full suite: 1,268 collected, 1,266 passed, 2 skipped, 0 failed;
- all tracked shell runbooks: 119/119 passed `bash -n`;
- four modified/new Python entrypoints and modules passed `py_compile`;
- bounded wait, unselected route, selected CPU route, and selected-source drift
  synthetic waiter rehearsals all passed.

The first full-suite attempt intentionally exposed the repository's branch
provenance guard: a detached checkout produced two inference-export failures
before their target assertions. Attaching the same revision and tree to the
exact diagnostic branch made both tests pass; the subsequent full suite was
green. This is expected fail-closed behavior, not a hidden test exclusion.

Structured evidence is stored at:

```text
artifacts/reports/generation/terminal_distribution_support_preparation_2026-08-20/rehearsal_report.json
```

## Conditional waiter deployment

After the rehearsal evidence commit, the final target was packaged and
verified locally and remotely:

```text
revision: 37cb1fe474197b8ec8c2f94c6cc97cfd11fc29f0
tree: 5a214941de008cf2782d568d749097134d9364bc
bundle bytes: 43,713,791
bundle SHA256: 9885c50235dbb9b9334b030116d26cdd53aad26750e9bc79b52228f2e678a66f
```

The final bundle advertised only that HEAD and retained the two exact
prerequisites recorded above. It was fetched into a new isolated checkout,
without fetching into or moving the formal repository:

```text
/root/autodl-tmp/CoFiTok/checkouts/terminal-distribution-support-37cb1fe/CoFiTok-internal
```

The final checkout was clean, exact, and passed the focused 18-test suite. The
conditional waiter was then launched with CUDA hidden, `nice=10`, idle IO
priority, and one-thread BLAS/OpenMP limits. Its deployed identity is:

```text
PID: 720592
start ticks: 1684120835
parent PID: 1
cmdline SHA256: 17bc23ce0b01534e438ed3f9713fb96f523c655c4a13cd27656e1d26dafe58be
status: waiting
detail: waiting_for_quality_bridge_followup_decision
decision exists: false
diagnostic output exists: false
```

File descriptor 9 resolves to the exact waiter lock, and an independent
nonblocking lock attempt returned exit code 1. After two polls, the status
still bound the expected revision, tree, branch, runbook, diagnostic script,
and waiter SHA256. The deployment receipt is:

```text
artifacts/reports/generation/terminal_distribution_support_waiter_deployment_2026-08-20/deployment_receipt.json
```

## Concurrent training safety

The formal checkout remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`, with tracked state clean and the
same 89-row full porcelain identity SHA256
`18e5981f22a2ac255c7f60343429daa6ceea86412b1d7bd09bc474f2faf74004`.

The only GPU process remained quality-bridge trainer PID `543758`, using
89,398 MiB. Its metric step advanced from 54,700 to 55,150 during rehearsal
and to 55,450 after waiter deployment. No GPU process was started, stopped,
signaled, or modified.

## Authorization boundary

This preparation does not prove a generation-quality advantage and does not
authorize any follow-up experiment. A completed terminal 100K result and the
exact canonical v2 route selection are required before this CPU diagnostic can
run. Even after it runs, recipe, training, 300K, promotion, and release remain
forbidden unless separately established by their own exact evidence chain.
