# Conditioning-ranking 5K training-confirmation execution control

Date: 2026-08-19

## Outcome

The previously preparation-only 5K confirmation stage now has a source-bound,
one-shot execution control path. This change does not deploy or start GPU work.
It closes the control-plane gap that existed after the matched four-arm 5K
sampling validator selected:

```text
prepare_separately_bound_matched_5k_training_recipe_confirmation
```

## Exact source route

The supervisor consumes only the completed report at the exact role, stage,
output root, and Git identity of the shared four-arm 5K sampling validation.
It independently checks both methods' exact gate sets and accepts GPU work only
when CoFiTok and dense identity both pass. Asymmetric or jointly failed results
complete without launching training; malformed results fail closed.

The accepted source remains diagnostic and non-authorizing. Training authority
comes from the active standing experiment authorization plus a newly generated
execution receipt that binds the physical source report, preparation, standing
authorization, four configs, runbook, revision, branch, output root, and five
ordered idle-GPU observations.

## One-shot execution boundary

The exact stage is:

```text
stage: conditioning_ranking_four_arm_train5k_confirmation_v1
output root:
/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_train5k_confirmation_v1
```

It trains four fresh runs sequentially:

1. control CoFiTok K8;
2. ranked CoFiTok K8;
3. control dense identity;
4. ranked dense identity.

Every run is fixed to ImageNet-256 10%, 5,000 optimizer steps, effective batch
64, seed 2027, bf16, and protected checkpoints at 1,250, 2,500, and 5,000.
No run may resume the 1K probe or any partial 5K state.

The supervisor requires five consecutive idle-GPU polls, reserves a launch by
an immutable launch receipt, and may invoke the runbook once maximum. It never
signals another process and cannot relaunch after a child failure. The runbook
recomputes the preparation, replays the execution receipt, checks the GPU again,
and refuses a pre-existing output root or lock before starting training.

The execution receipt and terminal training status explicitly keep all of the
following false:

- sampling authority;
- checkpoint promotion authority;
- follow-up training authority;
- full 100K or 300K authority;
- release authority;
- CoFiTok-specific or broad generation-quality claims.

The required next evidence after exact training completion is a separately
source-bound matched held-out 5K evaluation.

## Implemented files

```text
src/cofitok/generation/conditioning_ranking_training_confirmation.py
scripts/build_generation_conditioning_ranking_training_confirmation_execution_receipt.py
scripts/verify_generation_conditioning_ranking_training_confirmation_execution_receipt.py
scripts/run_generation_conditioning_ranking_training_confirmation_supervisor.py
artifacts/runbooks/generation_conditioning_ranking_four_arm_train5k_confirmation_v1.sh
tests/test_generation_conditioning_ranking_training_confirmation_execution.py
tests/test_generation_conditioning_ranking_training_confirmation_supervisor.py
```

Local validation before the code commit:

```text
new/changed confirmation tests: 17 passed
full conditioning/ranking control-chain selection: 91 passed
repository regression excluding the four sibling-paper path tests: 1,167 passed
new runbook bash -n: passed
Python compilation: passed
git diff --check: passed
```

The four excluded tests read `paper/` through the original repository's sibling
layout; this isolated `C:/c5k` worktree has no such sibling and therefore cannot
resolve those files. An initial unrestricted run also exhausted the nearly full
local `C:` temporary drive. Re-running with a task-specific `D:` temporary
directory and single-threaded BLAS completed all 1,167 code/repository tests in
scope without a failure. The exact task-owned temporary directories were then
removed.

The execution path also replays the complete physical 5K sampling-validation
source graph before it can prepare or authorize training: 1K postevaluation,
four integrity-checked checkpoints, four sampling reports and sample sets, four
generation-metrics reports, and two paired class-fidelity reports are rebuilt
and compared with the claimed terminal report. Surface-level JSON agreement is
therefore insufficient to trigger the confirmation stage.

Remote deployment and GPU launch remain intentionally absent while the active
full-data 100K quality bridge owns the GPU. A clean Linux rehearsal of the
eventual exact code revision is required before installing any waiting
supervisor.
