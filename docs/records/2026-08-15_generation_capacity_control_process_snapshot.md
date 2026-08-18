# Capacity control-process relaunch snapshot

Date: 2026-08-15

## Outcome

The active quality-bridge-to-capacity control chain now has one immutable,
source-bound process snapshot covering all 16 blocking/auxiliary stages plus the
lineage observer. It records enough inert data to audit a future relaunch recipe
without requiring the original PIDs to remain alive.

This artifact does not relaunch a process and does not authorize training,
sampling, evaluation, promotion, export, or release. A future recovery executor
must separately prove that all original processes are absent, reverify the
static sources, and write its own execution receipt before starting anything.

## Captured process contract

For every one of the 17 processes, the manifest binds:

- stage index, name, role, blocking flag, status path and status snapshot;
- PID and Linux start ticks, parent/process-group/session IDs, nice value and
  umask;
- complete argv, absolute CWD and resolved Python executable target;
- stdin/stdout/stderr targets and observed Linux descriptor flags;
- the safe environment whitelist only: `CUDA_VISIBLE_DEVICES`, `PATH`,
  `PYTHONPATH`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` when present;
- entrypoint argument index plus physical source bytes/SHA256;
- exact checkout path, revision, tree, branch and clean tracked state.

The snapshot also binds the v3 static-continuity manifest, so the four live
`/tmp` checkouts and five non-Git runtime files used by those commands are not
assumed to survive a host restart.

All 17 processes currently use `/dev/null` for stdin. Only one observed stdout
descriptor had Linux `O_APPEND`; the others were long-lived descriptors opened
without it. The future recipe therefore preserves the observed flags as
historical evidence but requires both stdout and stderr to be reopened in
append mode at the same absolute targets. This prevents a recovery launch from
truncating the existing logs.

## Implementation identity

- branch: `scale/generation-capacity-control-process-snapshot-v1`;
- revision: `4eb041d6d385df905a120fba1e5d0fcab575d170`;
- tree: `5ac2d3257f82a490181c20df76897d741887dd84`;
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-snapshot-4eb041d/CoFiTok-internal`.

The incremental deployment bundle is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_process_relaunch_v1/deployment/cofitok-capacity-control-process-snapshot-4eb041d-from-2108255.bundle`;
- bytes: `14,515`;
- SHA256:
  `46e686e43fd43803699137c7627ad680e8869c593764b3f1888232fd7dd3c601`;
- prerequisite: `2108255d00b0c822dd93636339fd912a8150484a`;
- advertised head: `4eb041d6d385df905a120fba1e5d0fcab575d170`.

## Persistent evidence

The output root is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_process_relaunch_v1
```

Evidence identities are:

| artifact | bytes | SHA256 |
|---|---:|---|
| process relaunch manifest | 106,174 | `455ddc919802c697afa4a80d43e70d79fc3d72057c8a194460772cb15f1c08c3` |
| capture-time live verification | 1,336 | `1553fc27fb017b660955116d88ef52a80954159a4eb339f9698d3f4d3acdc6b9` |
| independent final live verification | 1,336 | `c20e0efaf52b7b6d42d8142ad00fcd5f99c9239e04d090f02a655a5761b49651` |
| independent source-only verification | 1,336 | `eee67dc8ee14e7058abf2d4994d33897a293390b5792c234db0ca2e4cf079ac5` |

Both live verifiers rechecked all 17 source and runtime identities. The
source-only verifier rechecked all 17 entrypoint and checkout identities without
reading `/proc`, which is the relevant mode after a host restart.

## Validation

- process-snapshot unit/negative suite: `6 passed` locally;
- continuity, lineage, and runbook-syntax compatibility: `24 passed` locally;
- complete runbook CLI contract: `2 passed` locally;
- exact Linux process-snapshot, continuity, and lineage suite: `21 passed`
  with CUDA hidden;
- Python compilation, `bash -n`, `git diff --check`, and clean tracked Linux
  checkout: passed.

One local compatibility invocation accidentally used Bash-style `\` line
continuations in PowerShell, causing pytest to scan the drive root. The exact
project-local pytest PID was identified and terminated; the same bounded test
set was then rerun with a PowerShell argument array and passed. No remote or
experiment process was involved.

## Live-state preservation

After capture, all 16 stage processes and the lineage observer remained alive,
the observer reported no issues, and the first blocker remained
`quality_bridge_100k_recovery / waiting_for_gpu_idle`. The only GPU process was
the unrelated FieldScope PID `910099` at approximately `2,256 MiB`; it was not
signaled or shared.

The formal checkout remained exactly:

- revision: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`;
- branch: `scale/generative-system`;
- porcelain count: `87`;
- porcelain SHA256:
  `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.

No controller, trainer, sampler, evaluator, or recovery process was launched by
this work.
