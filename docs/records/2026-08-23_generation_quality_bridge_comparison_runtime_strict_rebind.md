# Quality-bridge comparison runtime-strict rebind

Date: 2026-08-23

## Problem

The terminal runtime strict replay v3 correctly binds the current authoritative
`pair_monitor.json`, but the existing terminal-system/comparison chain points at
the older canonical runtime guard. A corrected terminal-system guard was therefore
deployed under a versioned report directory.

The first attempt to point the strong-baseline comparison waiter at that corrected
guard failed closed before writing a status or deployment receipt because the
waiter accepted only the original `quality_bridge_comparison_v1` and
`terminal_system_claim_guard_v1` directory names. It did not create comparison
JSON, Markdown, or CSV output and did not touch the original waiter.

## Change

`wait_generation_quality_bridge_comparison.py` now accepts two explicit path-contract
arguments:

- `--expected-output-dir-name`
- `--expected-terminal-dir-name`

Both default to the original v1 names. A non-default value must remain a single,
lower-case report-directory component with the corresponding role prefix. The
waiter still requires every output/status/receipt/lock path to be an exact child of
that declared directory and every terminal source to be an exact child of the
declared terminal guard directory.

This preserves the original fail-closed path contract while permitting immutable,
versioned downstream rebinds.

## Authorization boundary

The change does not authorize training, sampling, checkpoint loading, 300K work,
promotion, export, release, or process signals. The comparison waiter remains
CPU-only and permanently non-authorizing. Existing canonical and failed artifacts
remain immutable.

## Verified deployment

- Code revision: `ac0388869ee5dd0f3a36bbf9f79550eed99a6f9c`
- Tree: `c001efd3c44c1c0c6b8f660a0b9c8f613fc217a4`
- Branch: `analysis/generation-quality-bridge-comparison-runtime-strict-v2-20260823`
- Remote checkout: `/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-comparison-runtime-strict-ac03888/CoFiTok-internal`
- Waiter source SHA256: `a022ddb285703995d2cc7013adb7eebac05b39386a42d36c6233135af1f188c2`
- Builder source SHA256: `3f3e7f7d99ba25ad41e93db3c357ca50d1ce41164aa5afc65b657d4595ad4a61`
- Incremental bundle: `D:/cofitok-bundles/quality-bridge-comparison-runtime-strict-v2-ac03888.bundle`
- Bundle bytes/SHA256: `2925` / `b6a7e5ecb97a7928101736376be2ec4d24a3ec48e5c281f0703d79a4eb32e37b`
- Remote Linux targeted tests: `41 passed`
- Corrected terminal guard: `reports/terminal_system_claim_guard_v2_runtime_strict`
- Corrected comparison directory: `reports/quality_bridge_comparison_v3_runtime_strict`
- Initial PID/start ticks: `122409` / `1714622728`
- Initial deployment receipt bytes/SHA256: `3420` / `f7e2824ee4bd4d228a4d7e087e0ee174b2d8c75d2123610602ac3459e7e96a53`

The live waiter was re-read after deployment: PPID `1`, CUDA hidden,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, nice `10`, ionice `idle`, clean
checkout, and no GPU process identity. Its initial status is `waiting` /
`waiting_for_exact_terminal_system_sources`; no comparison JSON, Markdown, or
CSV exists while the corrected terminal guard is still waiting.

The pre-fix formal attempt used
`reports/quality_bridge_comparison_v2_runtime_strict` and failed before status or
deployment receipt creation with `comparison waiter canonical path contract differs`.
Only its lock owner record and log are retained as immutable fail-closed evidence.
