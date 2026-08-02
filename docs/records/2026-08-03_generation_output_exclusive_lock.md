# Generation output exclusive lock

Date: 2026-08-03

## Problem

Sampling manifests, progress reports, atomic PNG writes, and exact resume make a
single generation invocation recoverable, but they do not make two concurrent
writers safe.  Before this change, both `scripts/generate_samples.py` and
`scripts/infer_generation.py` loaded the checkpoint and initialized the
generation session before discovering that another invocation was writing the
same output directory.  Two matching `--resume` invocations could also pass the
same manifest checks and race while updating progress and numbered PNG files.

The formal post-evaluation runbooks have a controller-level `flock`, but stable
inference needs the invariant at the component boundary as well: direct CLI
use, retry controllers, and future callers must not depend on one particular
runbook for single-writer ownership.

## Implementation

`cofitok.output_lock.exclusive_output_lock` provides an output-scoped,
non-blocking operating-system lock:

- Linux uses `fcntl.flock(LOCK_EX | LOCK_NB)`; Windows uses
  `msvcrt.locking(LK_NBLCK)`.
- The lock file is adjacent to the target, named
  `.<target>.cofitok-output.lock`, so it is not part of the generated sample
  tree or immutable inference evidence.
- The file intentionally persists.  The actual lock is tied to the open file
  descriptor and is released automatically after a crash; deleting a held lock
  file would permit a second inode and break exclusivity.
- While held, the file records schema, role, PID, hostname, canonical target,
  and acquisition time for diagnostics.  Symlink targets/parents are rejected,
  and Linux opens the lock with `O_NOFOLLOW` when available.
- Contention raises `OutputLockError` immediately.  It does not wait, kill,
  signal, or take over the current writer.

Both generation CLIs now acquire this lock before checkpoint loading, runtime
capture, manifest/progress writes, or output-directory creation.  Existing
sampling protocols, random streams, report schemas, atomic writes, and exact
resume behavior are unchanged.

## Verification

- Cross-process Windows test proves a contender fails while a holder is live,
  owner metadata persists, and a new process can acquire after release.
- Direct sampling and inference tests prove contention is detected before
  `GenerationSession.from_checkpoint` and before the target output directory is
  created.
- Symlink targets fail closed.
- An exact copy of the module streamed to `pro6000` passes Linux contention and
  post-release reacquisition using `fcntl.flock`.
- Focused output-lock/sampling/inference suite: `32 passed`.
- Full repository suite: `952 collected`, `946 passed`, `6 skipped`, exit code
  `0`.
- `py_compile` and `git diff --check`: passed.

This change improves concurrent execution safety; it is not sample-quality
evidence and does not authorize stability promotion or full 300K training.
