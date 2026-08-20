# Training-exposure qualification preparation

Date: 2026-08-20

## Outcome

Commit `9f6b2f2e8b871d627a4ad93944cd0ff832a971ed` prepares a conditional,
permanently non-authorizing matched training-exposure qualification. It is not
an execution receipt and did not launch training, sampling, or evaluation.

The preparation is selected only when all of the following source-bound facts
hold:

- the canonical quality-bridge v2 decision selects the capacity probe with an
  exposure fallback;
- the capacity result reports `capacity_supported=false` and no shared strict
  FID improvement from base128 to base256;
- CoFiTok mechanism invariants remain valid;
- the capacity result and every direct source file reopen with their recorded
  bytes and SHA256 identities.

Any other capacity result fails closed.

## Matched intervention

The candidate changes training exposure only. Both arms use fresh
initialization, full `imagenet_256`, base channels 128, effective batch 64,
seed 2027, the same optimizer hyperparameters, and the same absolute rollout
and EMA-teacher schedules as the 100K quality bridge. Source and target configs
may differ only in:

- `name`;
- `runtime.steps`;
- `runtime.protected_checkpoint_steps`.

The target is 300,000 steps and 19,200,000 images per method. Protected and
trend-evaluation milestones are 50K, 100K, 150K, 200K, and 300K. The terminal
protocol is matched 10,000-sample DDIM-100 evaluation with FID, IS, precision,
recall, class fidelity, and CoFiTok mechanism diagnostics.

At 1,281,167 full-data training images, 300K is 14.986336675858807 equivalent
epochs. The historical 10% 50K reference is 24.96859419012024 equivalent
epochs, requiring approximately step 499,828 for equal exposure. Consequently,
the preparation records `residual_underexposure_hypothesis_after_300k=true` and
forbids interpreting 300K as exposure saturation.

## Verification

The exact implementation identity is:

```text
branch: analysis/generation-training-exposure-qualification-v1
revision: 9f6b2f2e8b871d627a4ad93944cd0ff832a971ed
tree: 15e59dcfd89b7c346d88b4050e9ffded7a84a7b9
```

Local verification:

- focused exposure/recipe/runbook tests: 25 passed;
- Python compileall: pass;
- full suite: 1,241 passed, 6 skipped, and 4 sibling-paper-layout failures.
  Those four existing tests resolve `paper/` through the nested production
  layout and cannot resolve it from the standalone `C:/qbexposurequalification`
  worktree.

An incremental prerequisite-bound rehearsal bundle was verified locally and on
the server:

```text
bytes: 22178
sha256: 2fec1685e8a92a4d7e5181c80129ccf64f47f775491ab1b1f177696ba536d8b6
prerequisite: 85e3ece1196fd318cd6823439824e19fca4275a3
advertised revision: 9f6b2f2e8b871d627a4ad93944cd0ff832a971ed
```

The Linux checkout at
`/tmp/cofitok-exposure-rehearsal-9f6b2f2/CoFiTok-internal` was clean and exact.
With `CUDA_VISIBLE_DEVICES=-1`, one-thread CPU limits, and low process priority:

- the new runbook passed `bash -n`;
- focused tests passed 25/25;
- the complete suite passed with 1,249 passed and 2 skipped in 159.10 seconds.

The formal checkout remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` before and after rehearsal. The
active quality-bridge GPU process remained PID 543758; no GPU process was
started, signaled, or modified by this work.

Structured evidence is in
`artifacts/reports/generation/training_exposure_qualification_preparation_2026-08-20/rehearsal_report.json`.

## Authorization boundary

This work only prepares a possible future stage. It does not authorize 300K,
does not authorize GPU use, cannot produce a formal generation claim, and
cannot authorize release. A completed source-compatible 100K terminal decision,
a completed negative capacity result, and a new exact execution decision are
all required before any exposure-qualification run can start.
