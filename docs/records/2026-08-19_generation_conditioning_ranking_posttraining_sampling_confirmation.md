# Posttraining four-arm 5K sampling confirmation

Date: 2026-08-19

## Purpose

Close the evidence gap after the fresh four-arm 5K conditioning-ranking
training and its held-out CPU evaluation.  A shared held-out pass is not yet
generated-sample evidence, so this stage samples all four terminal step-5000
checkpoints on a new random stream and evaluates the control/ranked difference
for both CoFiTok and dense identity.

This stage is diagnostic and permanently non-authorizing.  It cannot launch
training, promote a checkpoint, replace the active full-data quality bridge,
authorize 100K/300K scaling, authorize release, or support a CoFiTok-specific
advantage claim.

## Frozen protocol

- Stage: `conditioning_ranking_four_arm_posttraining_sampling5k_confirmation_v1`
- Output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_posttraining_sampling5k_confirmation_v1`
- Arms: control/ranked CoFiTok and control/ranked dense identity
- Checkpoint: exact terminal `checkpoint_step_00005000.pt` plus integrity
  sidecar and training report for every arm
- Weights: EMA
- Samples: 5,000 per arm
- Sampler: DDIM-50, CFG 1.5, guidance rescale 0, eta 0
- Seed: `506020`
- Global indices: `[5000, 9999]`
- Per-sample seeds: `[511020, 516019]`
- Class schedule: balanced modulo, exactly five samples per ImageNet class
- Excluded prior stream: seed `406020`, global indices `[0, 4999]`
- Global-index overlap: zero
- Per-sample-seed overlap: zero

## Source and launch controls

Preparation must byte-replay the exact held-out shared-pass report and bind its
training status, original training execution receipt, standing authorization,
four physical checkpoints, integrity manifests, training reports, classifier,
runbook, clean Git revision, and output root.

The supervisor:

1. accepts only the exact shared-pass held-out decision;
2. rejects symlink chains for every control source, including the original
   training execution receipt;
3. requires five ordered consecutive idle-GPU observations;
4. runs one shared four-arm sampling preflight and batch selection;
5. reserves a one-shot launch receipt before starting the runbook;
6. never signals unrelated processes and never automatically relaunches;
7. leaves all follow-up training, scaling, promotion, and release decisions
   outside this stage.

The runbook independently replays preparation and receipt construction, checks
the exact step-5000 filename and step for each arm, samples all four arms,
builds matched generation metrics, evaluates paired class fidelity on CPU, and
builds the final report twice to prove replay stability.

## Decision routing

- Both methods pass: retain the shared conditioning-ranking recipe only for a
  separately authorized future scaling decision.
- Exactly one method passes: reject the shared repair because of posttraining
  method asymmetry.
- Neither method passes: revise the training-time semantic-alignment objective.

No route grants automatic GPU work after this stage.

## Local verification

- Modified/new Python entrypoints: `py_compile` passed.
- New posttraining tests: `12 passed`.
- Complete conditioning-ranking regression group: `84 passed`.
- Full repository regression excluding the four parent-layout paper tests:
  `1183 passed, 6 skipped` using a D-drive pytest base temporary directory.
- Parent-layout paper tests from the canonical local project root: `4 passed`.
- Runbook syntax: Git Bash `bash -n` passed.
- No GPU sampling or training was launched during implementation or testing.
