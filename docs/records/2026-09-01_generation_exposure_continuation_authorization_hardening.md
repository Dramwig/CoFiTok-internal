# Exposure continuation authorization hardening (2026-09-01)

## Scope

This record covers control-plane work for a possible, bounded 100K to 110K
exposure continuation. It does not authorize that continuation, capacity
qualification, 300K training, promotion, export, or release. The immutable
100K quality-bridge evidence remains the scientific source of truth.

## Current remote evidence

The authoritative pro6000 checkout remains
`cf0e5faa94bf4ab38d947b921935b3b765b5537a` with tree
`6cef27723196fd363379bca2e7b85b1678ebd777` on
`scale/generation-stability-quality-bridge-100k`. The standing authorization
SHA256 remains
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`.

At the final 2026-09-01 check, the pair monitor was `pass/complete`, both
100K runs were complete, and the GPU reported `0 MiB` used with no matching
training or sampling process. The terminal completion audit was operationally
`pass` but retained `terminal_status=hold` and
`generation_advantage_proven=false`.

The non-authorizing matched 1000-sample sampling-recovery result at
`/tmp/cofitok-exposure-capacity-gate-20260831/evidence/sampling_recovery_result.json`
has 16 observations and records
`selection_status=no_shared_sampling_recovery_candidate` with
`scientific_status=screening_only`. It cannot support a generation-quality
claim or a larger training stage.

The exposure and capacity gates remain `prepared` with
`execution_ready=false`. Their authorization boundaries keep GPU execution,
training, sampling, promotion, export, and release disabled.

## Control-plane changes

Isolated branch `analysis/generation-exposure-continuation-v1` contains commit
`c12d2cb` (`Bind exposure continuation to exact stage authorization`). The
continuation controller and validators now:

- require a user-created, content-addressed exact-stage authorization sentinel;
- bind that sentinel, the source checkout, target checkout, configs, source
  checkpoint payloads, sidecars, and `latest.json` pointers by SHA256;
- reject a candidate gate unless it remains `execution_ready=false` and
  preserves the terminal hold;
- acquire the sibling execution flock before creating the candidate output
  root, preventing a competing controller from racing the absence check; and
- keep all automatic 300K, promotion, export, release, and process-signal
  permissions false.

Only source code, tests, and the dormant runbook were committed. The local
`.source_*` snapshots and `.tmp_resume_test` directory are generated fixtures
and were intentionally excluded from the commit.

## Verification

- Project `.venv` focused authorization, continuation-lock, and exact-resume
  tests passed.
- The complete CPU test suite passed when excluding the pre-existing
  `test_aaai27_experiment_structure.py` environment-path test; that test uses
  a hard-coded `D:\\paper\\...` path unrelated to this change.
- `compileall` passed for `src`, `scripts`, and `tests`.
- The continuation runbook passed `bash -n` using Git for Windows bash.
- With `EXPOSURE_CONTINUATION_EXECUTION_ENABLED=false`, the runbook exited
  with its intentional disabled status before reading execution credentials.

No remote checkout, locked artifact, or GPU process was modified by this
work. A future execution requires a separately created exact-stage
authorization and a new source-compatible gate; this record is not that
authorization.
