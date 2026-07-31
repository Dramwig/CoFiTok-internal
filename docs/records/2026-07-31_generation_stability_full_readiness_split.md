# Stability-full readiness and launch split (2026-07-31)

## Decision

The 250M CUDA runtime qualification is now separated from the full ImageNet-256
300K training launch. A single invocation may no longer benchmark five matched
runtime candidates and immediately begin a multi-week training queue.

## Readiness stage

`artifacts/runbooks/generation_stability_ema_teacher_full_readiness_after_gate.sh`
is a one-shot, no-training runbook. It requires:

- the passing `stability_scaling` gate and its expected SHA256;
- the exact clean target revision and branch;
- absent CoFiTok and dense full-training state;
- the exact 256-channel pair (`250,153,763` versus `250,135,043` parameters);
- a schema-v2 storage reserve using checkpoint multiplier `4.0`;
- an idle GPU and all candidates `1x64,2x32,4x16,8x8,16x4`, with `1x64`
  as the fail-closed baseline.

It writes source-bound config, storage, and runtime reports, then writes
`reports/full_training_readiness.json` and exits. Existing evidence or benchmark
directories are never overwritten. The readiness report is deterministic and
records absolute source paths, bytes, SHA256 values, clean training Git identity,
the selected runtime, and `training_state_absent_at_build=true`. It explicitly
does not claim formal generation completion.

## Launch stage

`generation_stability_ema_teacher_full_matched_300k_after_gate.sh` now requires
`EXPECTED_READINESS_SHA256`. It does not call the runtime selector. Before model
load it:

1. verifies the scaling gate and readiness SHA256;
2. replays the exact config, gate-source, storage, runtime-candidate, baseline,
   dataset, Git, and current CUDA-environment contracts;
3. reads the selected microbatch and accumulation from the verified artifact;
4. requires an idle GPU;
5. writes a fresh `storage_capacity_launch.json` using the same 4x reserve.

Only after those checks does the existing monitor, watchdog, exact-resume, and
50K/100K/200K/300K alternating milestone loop begin.

## Completion evidence

The stability completion audit requires the externally supplied readiness
SHA256 and replays the immutable report and all six bound source files. Replay
may occur on a later evaluation revision and after training state exists, but
the readiness report must still bind the exact full-training revision, branch,
run paths, benchmark root, and source bytes. The audit separately validates the
fresh launch-time storage report.

This change does not authorize or start full 300K training. The active 50K pair,
its checkout, and the formal remote repository remain untouched until the
source-bound promotion gate passes.
