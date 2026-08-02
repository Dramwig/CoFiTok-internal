# Frozen supplemental waiter launch

Date: 2026-08-03

## Outcome

Exactly one source-bound frozen supplemental waiter was restored on `pro6000`
after its clean Linux evaluation checkout and entrypoints were independently
attested. The waiter is CPU coordination only while it is waiting. It did not
start the supplemental evaluation, reserve GPU memory, signal a process, or
change any checkpoint, report, post-evaluation, or readiness identity.

The detached process is PID `861117`, parent PID `1`, started at
`2026-08-02T22:44:48+00:00`, with cwd:

```text
/tmp/cofitok-stability-frozen-supplemental-c212b9e/CoFiTok-internal
```

It is fixed to clean
`scale/generation-large-capacity@c212b9e2b64d1b302b17a9d4e30a296d773d4215`.
The prerequisite checkout attestation is commit
`e10a06ead21a6335ff49295f84caa19cd0b04049`.

## Singleton and ordering evidence

The process holds fd `3` on:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/frozen_posteval_supplemental/.supplemental_waiter.json.cofitok-output.lock
```

Only one matching waiter process exists. Its first status and a later full poll
both reported:

```text
status=waiting
detail=waiting_for_frozen_postevaluation
child_pid=null
supplemental_non_authorizing=true
full_training_launch_allowed=false
```

The later snapshot was written at `2026-08-02T22:49:52.176773+00:00`, proving
the detached process survived a full 300-second poll rather than merely
starting. Its exact 1,804-byte status SHA256 was
`3b5be6e71df61ef49f6c81ec9be540b4cebe0b2f2927447af6e368fa669a87e3`.
The synchronized bytes and launch receipt are in:

```text
artifacts/reports/generation/stability_frozen_supplemental_waiter_launch_2026-08-03/
```

The waiter cannot start its child unless all three conditions hold in the same
poll: frozen formal post-evaluation is complete, the already queued readiness
GPU stage is terminal, and `nvidia-smi` has no compute process. The child
runbook repeats the GPU-idle check and has its own lock, so the coordinator
does not weaken the existing queue ordering.

## Launch-time system state

The later read-only observation found dense step `21,950`, `1,404,800` samples,
and `440` metric rows. The pair monitor remained `running` in
`dense_identity_training` with no issues. Post-evaluation PID `717145` and
readiness PID `748183` both remained `waiting`; readiness retained
`full_training_launch_allowed=false`.

GPU compute remained only the pre-existing dense trainer PID `541878` using
`77,970 MiB`. No other project process was modified. Generation filesystem free
space was `182,914,887,680` bytes.

## Decision boundary

This launch prevents the later sample-quality diagnostics from being forgotten
or racing the readiness CUDA stage. It is not the supplemental scientific
result itself, does not replace post-evaluation or readiness, and is not a
large-capacity deployment or full-training authorization. A later supplemental
`hold` or failure remains decision-grade evidence against scale-up; a pass is
still only a necessary, non-authorizing prerequisite. Full 300K remains
unlaunched and disallowed.
