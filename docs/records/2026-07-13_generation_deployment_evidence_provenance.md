# Generation deployment evidence provenance

Date: 2026-07-13

Branch: `scale/generative-system`

## Problem

The controlled transition executed a bounded conflict check, the full pytest
suite, and selected shell syntax checks, but deployment receipt schema v1 stored
only literal pass values. It did not bind the conflict scanner's structured
output or a test artifact, and the shell check covered seven critical runbooks
rather than proving coverage of every target-tracked shell entrypoint. The
bundle path also pointed into `/tmp`, so a reboot could remove the transition
artifact before the terminal completion audit.

## Contract

Deployment receipt schema v2 is generated only from four durable sources:

1. the prerequisite-aware Git bundle, atomically archived below
   `checkpoints/generation/deployment/`;
2. `generation_upgrade_conflict_scan.json`, atomically written by the exact
   target conflict checker before fast-forward;
3. `generation_upgrade_pytest.xml`, emitted by the complete remote pytest run;
4. `generation_upgrade_runbook_syntax.json`, which obtains the authoritative
   shell list with `git ls-files` and runs `bash -n` on every listed path.

The receipt records path, bytes, and SHA256 for every source. It additionally
records the bundle head and exact pinned prerequisite, conflict count and target-added count, pytest test and
failure counts, and the exact runbook count. An idempotent retry on an already
deployed target requires the original pre-merge conflict report instead of
fabricating a zero-path scan after the transition.

`audit_large_scale_generation_completion.py` reads the four sources from fixed
authoritative paths, recomputes bytes/SHA256, asks Git to list the archived
bundle head, reparses JUnit, and reruns the conflict/runbook semantic validators.
Receipt-only edits and source-file tampering therefore fail the named
`controlled_revision_transition` check.

Before fetch, the remote helper also parses the uploaded bundle header and
requires exactly one prerequisite equal to the pinned training revision. The
receipt writer and terminal audit independently parse the archived header, so
an extra prerequisite or a bundle cut from another baseline is rejected even
when its advertised target head is correct.

## Verification

Targeted deployment, receipt, completion-audit, runbook-entrypoint, and real-Git
tests pass `79/79`. They cover atomic conflict reports for both pass and reject
outcomes, exclusion of unrelated untracked shell history, partial runbook lists,
JUnit failures, source SHA drift, conflict-content tampering, and an archived
bundle whose bytes change while its header remains parseable. Full local and
isolated Linux verification are recorded with the implementation commit. The
complete local suite passes `541/541`; a real local CLI run also proved that a
missing Bash runtime is captured as a structured 40-path failure report instead
of crashing the evidence writer. Linux is the authoritative shell-syntax
environment for deployment.

The implementation commit is
`91095b907862b5fe72b7dff2196e2b76b046f3cd`. Its prerequisite-aware bundle
from pinned `781a01444fddbf0d48a427ba58bdeed50167b5be` is `425,496` bytes with
SHA256 `3471000b2fc5954d6bc54da3d27fad5552416c873886211d2be945ac4795f7c7`;
it advertises only the implementation commit and declares the pinned revision
as its prerequisite.

The server verified and fetched the bundle objects without moving the formal
worktree. An initial `/tmp/CoFiTok-internal` rehearsal correctly exposed four
paper-layout test failures because those tests require the real sibling
`paper/` directory. Repeating in an isolated `CoFiTok-internal + sibling paper`
layout passed the complete Linux suite `541/541`. The real syntax auditor found
`40` target-tracked shell runbooks, checked all `40`, and reported zero
failures. The bounded scanner then checked the formal remote's large untracked
tree against the implementation target: `148` target-added paths, zero
conflicts, and `0.054` seconds elapsed. The formal worktree remained at pinned
`781a01444fddbf0d48a427ba58bdeed50167b5be` with clean tracked status.
