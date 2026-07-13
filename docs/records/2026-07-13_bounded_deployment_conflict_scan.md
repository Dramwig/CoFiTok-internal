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
