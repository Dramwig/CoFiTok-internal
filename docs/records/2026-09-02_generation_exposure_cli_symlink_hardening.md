# Exposure/capacity CLI symlink hardening (2026-09-02)

## Scope

This was a local CPU-only engineering change. It did not modify the remote
checkout, training checkpoints, generated samples, locks, or any GPU process.

## Change

The four exposure/capacity preparation and execution-gate CLIs now pass input,
checkpoint, sidecar, `latest.json`, lock, storage, candidate-output, and report
paths through `cofitok.path_security.reject_symlink_chain`. Direct
`Path.resolve()` use for bound files was removed so a symlink cannot be
silently reinterpreted as the content-addressed source. The builders also
reject a symlinked script path before deriving their project root.

Regression coverage was added for source identity readers on POSIX systems;
the test is skipped on Windows because creating a real symlink requires the
platform privilege that is unavailable in the local environment.

## Verification

The complete local test suite finished with exit code `0` after the final
change, and `python -m compileall -q src tests` also passed. The focused
exposure/capacity and provenance suites passed with only the existing Windows
platform skips.

## Remote boundary snapshot

At the read-only snapshot on 2026-09-02 11:15 CST, the standing-authorization
SHA256 matched the expected value, both matched runs were complete at
`100000 / 6400000`, the authoritative pair monitor was `pass`, and the RTX
PRO 6000 reported `0 MiB` with no compute applications. The authoritative
comparison and terminal completion audit remained operationally `pass` but
scientific `hold`; `generation_advantage_proven=false` remains unchanged.
The factorization, conditioning, exposure, and distribution supervisors were
CPU-only waiters with no child process. No follow-up GPU route was selected.
