# Frozen 50K post-evaluation supplemental pipeline

Date: 2026-08-03

## Gap

The active stability pair and its post-evaluation waiter are immutable:

```text
training   2c2c1f5166b73d4f28df93b276901671ac1a7836
post-eval  c1efb12c6640f2d2d62ac7e9982c8804d96e7289
```

The frozen post-evaluation produces matched 10K EMA DDIM-100 metrics and
1,024-image timestep-500 mechanism diagnostics, but predates the current EMA
free-rollout qualification and blocking distribution-support interpretation.
Changing its checkout while training is active would invalidate the execution
identity. Merely running the later rollout evaluator would also be insufficient:
the stability qualification requires its checkpoint and rollout reports to
share one exact clean evaluation revision.

## Execution chain

`artifacts/runbooks/generation_stability_frozen_50k_supplemental_after_posteval.sh`
is an independent follow-up that can run only after the existing post-evaluation
waiter reports its exact successful terminal state. It:

1. Verifies the waiter schema, terminal status, child exit, training identity,
   frozen evaluation identity, and `formal_300k_allowed=false`; exact resume is
   accepted only from unchanged waiter bytes and the same clean supplemental
   revision.
2. Revalidates the original promotion gate and refuses to start while any GPU
   compute process exists.
3. Reruns matched 1,024-image, timestep-500 checkpoint evaluations with EMA in
   the supplemental checkout.
4. Runs matched 64-image EMA DDIM-100 free rollouts with seed `2029`, CFG `1.5`,
   guidance rescale `0`, teacher CFG `1`, batched CFG, clipped `x0`, and bf16.
5. Builds the existing matched stability qualification from those four reports
   and the immutable training reports.
6. Replays the precision/recall distribution-support qualification from the
   original gate and its bound 10K metrics.
7. Emits one source-bound combined report. `status=pass` requires a passing base
   gate, passing distribution support, and passing EMA rollout qualification;
   any scientific failure becomes `hold` without erasing the diagnostic.

The two checkpoint evaluations, two rollout evaluations, and qualification
builder are each wrapped by `run_generation_stage_once.py`. Their receipts bind
the command, clean project identity, inputs, and output tree, permitting exact
reuse after interruption while refusing source or command drift.

## Trust and authorization boundary

The combined builder rehashes its four direct reports before and after the
build. It also rehashes the raw post-evaluation status and every nested source
bound by the distribution-support and rollout-qualification reports. The
post-evaluation verification must have been built by the same exact clean
supplemental revision used for the final report.

This chain is deliberately non-authorizing:

```text
supplemental_non_authorizing=true
replaces_generation_gate=false
replaces_readiness=false
scaling_authorization_evaluated=false
full_training_launch_allowed=false
```

It contains no training entry point, never reruns CoFiTok, and cannot launch
full 300K. A pass means that the frozen 50K evidence has also cleared the newer
distribution-support and EMA rollout diagnostics; it is not a scaling launch
receipt.

Downstream consumption is fail-closed without changing that claim boundary.
The original schema-v3 full-training launch receipt required this exact report
as an eleventh source. The current schema-v4 receipt still replays all direct
and nested supplemental bindings and additionally requires frozen formal-sample
class fidelity as a twelfth source. Missing, held, failed, drifted, or
non-reproducible supplemental evidence blocks receipt creation. A pass is only
a necessary quality prerequisite: it does not replace class fidelity,
readiness, the readiness revision bridge, fresh storage, deployment identity,
or separate human launch authority.


## Active-run boundary

This implementation is local only. It does not deploy to, signal, pause,
restart, or otherwise modify the active dense trainer, recovery controller,
post-evaluation waiter, readiness waiter, or another project's process. The
runbook remains dormant until the existing dense 50K member and frozen formal
post-evaluation have completed and a separately attested clean supplemental
checkout is available.

## Verification

- Focused module, CLI, runbook-entrypoint, distribution-support,
  stability-qualification, and frozen-posteval tests pass (`38 passed`).
- The new module and both CLIs pass `py_compile`.
- The new runbook is included in the source-to-source runbook/CLI option
  contract test and has a dedicated diagnostic-only/resume contract test.
- Complete local suite: `1011 collected / 1005 passed / 6 skipped` in
  `259.75s` on Windows.
- The staged LF runbook blob passed remote Linux `bash -n` on `pro6000` through
  a binary pipe; the check wrote no remote file.
