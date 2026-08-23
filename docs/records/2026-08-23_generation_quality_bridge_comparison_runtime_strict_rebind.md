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
