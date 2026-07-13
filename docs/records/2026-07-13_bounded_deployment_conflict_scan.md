# Bounded deployment conflict scan

Date: 2026-07-13

Branch: `scale/generative-system`

## Problem

The controlled deployment helper originally enumerated every untracked path in
the pinned remote repository and intersected that list with every target tracked
path. The repository intentionally contains many historical untracked reports
and artifacts. A real read-only rehearsal did not finish within 30 seconds even
though the target revision changed only a small set of code paths. The guard was
correct but its cost scaled with unrelated experiment history.

## Contract

`scripts/check_generation_deployment_conflicts.py` derives target-added files
with a NUL-delimited, rename-disabled Git diff from the exact current and target
commits. Modified files are already tracked by the pinned revision and cannot be
untracked conflicts, so they need no scan. Added paths are queried in bounded
chunks of 256 pathspecs using `git ls-files --others --exclude-standard -z`.

The checker separately walks each added path's parents and rejects an untracked
file or symbolic link that would block creation of a target directory. Existing
directories are allowed, and a tracked parent that the target revision replaces
is not misclassified as untracked. Output is structured JSON; any conflict exits
with the deployment guard code `76` before fast-forward.

The remote helper extracts the checker from the exact fetched target commit into
the same temporary prevalidation tree as the pair validator. It runs after pair
validation and process-exit checks but before `git merge --ff-only`. Formal HEAD
therefore remains pinned on checker error.

## Verification

Integration tests create real Git repositories and prove that hundreds of
unrelated untracked artifact files do not affect a target-added source path,
while both an exact untracked file and an untracked parent-file blocker are
reported and rejected. Transition tests require the target checker to execute
before merge. A server rehearsal must also verify the current large untracked
tree completes quickly and reports zero conflicts before deployment is allowed.

The implementation was committed as
`b2b8e8feb67b16126e4e82cd0e88e45576039e38`. Local verification passed all
`530/530` tests. A pinned incremental bundle from
`781a01444fddbf0d48a427ba58bdeed50167b5be` contained one advertised target
head, was `414,475` bytes, and had SHA256
`f49fe4e7216e94ad1ce1b2cba17ba19425e1fbe4778a6d51f17082546369fd1f`.

The bundle was verified and fetched into the formal remote object database
without moving its working-tree HEAD. The checker was extracted from the exact
target commit and run against the real remote repository. It examined `145`
target-added paths in `0.047` seconds and returned `status=pass` with zero
conflicts. The affected isolated Linux tests passed `29/29`, and all `40/40`
shell runbooks passed `bash -n`. After rehearsal and cleanup, the formal remote
HEAD remained pinned at `781a01444fddbf0d48a427ba58bdeed50167b5be` with a
clean tracked status.
