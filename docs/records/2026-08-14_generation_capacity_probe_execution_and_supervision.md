# Matched 250M / 10K capacity-probe execution and supervision

Date: 2026-08-14

## Purpose

The full-data base-128 quality bridge remains queued behind an unrelated GPU
workload. If its source-replayed follow-up decision selects the capacity branch,
this change provides the exact next execution stage: a matched base-256 probe
that trains CoFiTok and dense identity only to step 10,000 and evaluates four
matched arms. It also provides a non-preemptive supervisor so the stage can
start automatically after the source-bound preparation exists and the GPU is
confirmed idle.

This is a bounded diagnostic, not full training. It cannot continue either new
run to the configured 100K horizon, cannot authorize or launch 300K, is not a
promotion gate, and cannot release an artifact.

## Experimental contract

The only training intervention is `model.base_channels: 128 -> 256`. Both new
arms preserve the full-data ImageNet-256 recipe, effective batch 64, 100K
configured schedule, and fresh initialization, but intentionally stop at exact
step 10K. The matched parameter counts are:

| method | base-256 parameters |
|---|---:|
| CoFiTok K8 | 250,153,763 |
| dense identity | 250,135,043 |

The four evaluation arms are the preserved base-128 CoFiTok and dense step-10K
checkpoints plus the two newly trained base-256 step-10K checkpoints. Every arm
uses EMA, bf16, DDIM-50, CFG 1.5, the same balanced-modulo random stream, and
2,048 generated images. CoFiTok arms additionally receive the 256-image
mechanism audit with four random orders. Precision and recall are intentionally
omitted because this is a capacity causal diagnostic rather than a formal
generation gate.

Capacity is considered supported only when base-256 strictly improves FID for
both matched methods and both CoFiTok arms preserve zero-token, ordering, and
shuffle-mismatch invariants. Even that result only recommends constructing a
new source-compatible decision; it records `full_300k_launch_allowed=false`.

## Trust and recovery boundary

Execution requires all of the following:

- the exact preparation report selected by the source-replayed quality-bridge
  follow-up decision;
- the exact standing user instruction and its preserved safety boundaries;
- a clean isolated execution checkout, exact revision/tree/branch, exclusive
  controller lock, and no existing capacity trainer;
- five consecutive idle-GPU observations before launch and a second idle check
  inside the controller;
- passing matched-config, runtime-selection, and launch-time storage evidence;
- an immutable execution authorization and launch receipt before training;
- exact step-10K checkpoint, integrity-sidecar, latest-pointer, metrics,
  validation-event, dataset, runtime, and compute-accounting validation;
- physical rehashing of checkpoints, sidecars, sampling manifests/progress,
  numbered PNG trees, sample-set digests, and the real validation tree before
  constructing the four-arm result.

The supervisor never sends signals to unrelated processes. A GPU race returns
to idle waiting without consuming a retry. Only interrupted training or arm
evaluation stages receive bounded recovery attempts. Authorization, config,
runtime, storage, launch-receipt, partial-training validation, and result replay
fail closed without automatic retry.

Exact resume now persists the completed scheduled-validation count in every
checkpoint and aligns the validation iterator from that value. The legacy
fallback accounts for both interval validations and a non-divisible final-step
validation, preventing validation-batch drift after resume.

## Validation before deployment

Local project-environment validation completed with:

- 94 focused capacity-probe, exact-resume, runtime-selection, storage, and
  runbook-entrypoint tests passing;
- Python compilation for every new module and modified training entrypoint;
- `git diff --check` with no whitespace errors.

The final Linux checkout then passed `bash -n` for both execution runbooks and
the complete repository suite: 1,233 tests were collected, 1,231 passed, two
were skipped, and none failed. The authoritative CPU-only log is
`/root/autodl-tmp/CoFiTok/checkouts/.tmp-capacity-execution-3a7dc9d/final_rehearsal_v2.log`
(`1,440` bytes, SHA256
`05d72dc2395ee62fe3fa0679b9c93b6fc2df8f203c1e5fac9d6da404b6d08c2c`),
with exit record SHA256
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

The first full-suite attempt exposed four missing-file failures because the
isolated checkout's parent initially lacked the two sibling `paper/` files
read by `test_aaai27_experiment_structure.py`. No implementation test failed.
Only that already-failed CPU rehearsal was terminated. The second rehearsal
used byte-identical copies of the formal files: AAAI `main.tex` SHA256
`d63ae7b2509166ba222f6fa63b3c1667794ab80b3a2424eae67116bd0107646a`
and venue-neutral `main.tex` SHA256
`d73e829739db55dee86047357c98c439bf25941e385a033a7beb058cf2b1d27b`.

## Server deployment

The immutable execution revision is
`3a7dc9db6950055829db00c8ffdd6e906501fbb4`, tree
`2b187e1bc6a36342ef2d803b26b4ae6f92b1a9f1`, on branch
`scale/generation-capacity-probe-v1`. Its clean isolated checkout is
`/root/autodl-tmp/CoFiTok/checkouts/capacity-execution-3a7dc9d`.

The incremental execution bundle is
`/tmp/cofitok-capacity-execution-3a7dc9d.bundle` (`59,902` bytes, SHA256
`e843a0f2cc1b122a1a14682125e7b574f7b347fe70d9ff2c83a99d42d0c4e79e`).
It advertises only the execution head and requires preparation revision
`22a5994f0475c29cca38a80188ed6ce00c6e6d15`; remote verification from that
exact prerequisite checkout passed.

The immutable deployment receipt is
`reports/execution_supervisor_deployment_receipt.json` (`6,497` bytes, SHA256
`92db48265df63ad3cb540858b496c6d578178d9142ce37af036e1a55879fb044`).
The launch receipt is `reports/execution_supervisor_launch_receipt.json`
(`4,401` bytes, SHA256
`9c04632adf19fa74120be5430220e8dd46635610924d68393106f4e6e0a6ae6d`),
under output root
`/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1`.

Exactly one non-preemptive CPU supervisor was launched as PID `153330`. Its
initial status is `waiting_for_source_bound_capacity_preparation`; the capacity
training controller has not started. At launch the sole GPU process remained
the unrelated FieldScope job PID `910099`, and no CoFiTok process used or
signalled the GPU. The formal remote checkout remained unchanged at revision
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` with its pre-existing 87 porcelain
entries and porcelain SHA256
`a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.

The supervisor can execute only the exact 250M/10K diagnostic after the
source-bound preparation exists and five consecutive GPU-idle observations.
It cannot complete either configured run to 100K, launch 300K, promote, or
release an artifact.

## Post-deployment wait-chain recovery

A process-level audit after deployment found that the quality-bridge idle
waiter, its bounded recovery supervisor, and the follow-up decision waiter had
all exited near 08:29 CST even though their last JSON statuses still said
`waiting` or `observing`. Their preserved logs show the same root cause:
`OSError: [Errno 28] No space left on device` while creating an adjacent atomic
status temporary under `/tmp`. No quality-bridge controller had launched, no
training checkpoint existed, and no GPU work had started.

Seven old, clean, process-unreferenced verification checkouts were moved from
`/tmp` to the recoverable data-disk archive
`/root/autodl-tmp/CoFiTok/checkouts/tmp-archive-20260814`. Nothing was deleted.
The two moves freed `3,303,632,896` bytes and are bound by
`move_receipt.json` (SHA256
`abde8de0849773bc83465dd5db33eafb1c9623ca7ebf9d48b5058c0c8a0a35dd`)
and `move_receipt_part2.json` (SHA256
`9b04672d1f4159ce5d2c4e8acb24637ed280684233f94c2ecd61af85f670c717`),
leaving approximately 3.4 GiB free on the system overlay.

After revalidating exact code, Git, authorization, output, process, storage,
and GPU identities, the wait chain was restored without launching training:

- quality-bridge bounded recovery supervisor PID `155032`, waiting for GPU
  idle behind FieldScope;
- quality-bridge follow-up decision waiter PID `155125`, waiting for the
  quality-bridge result;
- capacity-preparation waiter PID `132393`, still waiting for that decision;
- capacity execution supervisor PID `153330`, still waiting for source-bound
  preparation.

All four statuses refreshed successfully across a complete poll after the
recovery. The immutable recovery receipt is
`stability_full_data_100k_base128_quality_bridge_v1/reports/enospc_wait_chain_recovery_receipt.json`
(`6,803` bytes, SHA256
`a9533eb7375fce2d4323c42eef1f68b10ba6b818384945c236266fcd4513ac37`).
The sole GPU process remained the unrelated FieldScope PID `910099`; no signal
was sent to it.
