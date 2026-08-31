# Exposure/capacity gate replay (2026-09-01)

## Scope

This record documents a read-only Linux replay of the two source-bound
candidate gates prepared after the 100K quality hold. It did not create an
execution authorization, launch training or sampling, modify the formal
100K checkout, or change any locked evidence.

## Replay evidence

The replay used the exact builder bundle at
`/tmp/cofitok-exposure-capacity-gate-20260831/builder/exposure-capacity-gate-builder-d6c4f5d.bundle`
and detached revision `d6c4f5daa1fb8fcb3cb8cf2d6e106abffdb7866f` in a temporary
Linux checkout. Both candidate reports were rehashed and validated against the
same preparation (`source_count=10`):

| candidate | validator status | execution ready | stage authorization |
| --- | --- | --- | --- |
| `capacity_qualification` | `pass` | `false` | not authorized |
| `exposure_continuation` | `pass` | `false` | not authorized |

The replay preserved the exact 100K bridge revision/tree/branch, checkpoint
sidecar and latest bindings, and the all-false non-authorizing boundary. The
capacity candidate remains a fresh 256-channel 10K qualification; the exposure
candidate remains an exact 100K-to-110K resume. Neither is a quality claim or an
automatic 300K route.

## Runtime boundary

Immediately after replay, the target GPU remained at `0 MiB` with no compute
application. The candidate output roots and stage-authorization sentinel were
absent, and the temporary checkout was removed after verification. No remote
training or sampling process was started.

The Windows validator cannot directly consume the Linux preparation copy
because its serialized source descriptors intentionally contain POSIX paths;
the authoritative replay therefore runs on Linux where those paths exist.

## Local verification

On the isolated branch, the exposure-authorization and continuation suites
reported `7 passed, 1 skipped`; `compileall` and `git diff --check` passed.

The next GPU action still requires a separately created exact-stage
authorization. Until then, `generation_advantage_proven=false` and the
terminal scientific status remains `hold`.
