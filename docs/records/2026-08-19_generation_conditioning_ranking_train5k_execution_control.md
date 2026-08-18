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

At code-commit time, remote deployment and GPU launch were intentionally absent
while the active full-data 100K quality bridge owned the GPU. The clean Linux
rehearsal and subsequent CPU-only waiting deployment are recorded below; GPU
training remains unlaunched.

## Exact Linux rehearsal

The complete execution-control revision was rehearsed on `pro6000` in an
isolated checkout with CUDA hidden and without touching the formal checkout:

```text
revision: ed9463eb85ffb97545b5506e264123e46ff2fcfc
tree: 423ab336401bf79eb21ccce822880171bbf246b2
prerequisite: f77e311546beba697020debd35e26888096ca258
bundle bytes: 35,679
bundle SHA256: a9990f744a557e7442fc6c37e4a876226c6965b8c613be287d9fd8ae716cd056
isolated checkout:
/tmp/cofitok-conditioning-confirm5k-execution-ed9463e.HlABkJ/CoFiTok-internal
CUDA_VISIBLE_DEVICES: -1
CPU/IO priority: nice 15, ionice idle class
related tests: 91 passed
runbook bash -n: passed
post-test tracked/untracked status bytes: 0
```

The rehearsal checkout was re-read before deployment and still resolved to the
exact revision/tree with an empty porcelain status. It performed no GPU work.

## Persistent waiting deployment

At `2026-08-19T04:20:21+08:00`, the exact execution revision was installed as
a persistent isolated checkout:

```text
checkout:
/root/autodl-tmp/CoFiTok/checkouts/conditioning-confirm5k-execution-ed9463e/CoFiTok-internal
revision: ed9463eb85ffb97545b5506e264123e46ff2fcfc
tree: 423ab336401bf79eb21ccce822880171bbf246b2
branch: scale/generation-label-ranking-5k-training-confirmation-v1
full porcelain count: 0
supervisor PID: 280369
```

The first installation attempt cloned from the formal repository and stopped
at `git bundle verify` because that object database did not contain prerequisite
`f77e311546beba697020debd35e26888096ca258`. It stopped before fetch, checkout,
control-directory creation, or supervisor launch. That incomplete clone was
moved, not deleted, to:

```text
/root/autodl-tmp/CoFiTok/checkouts/conditioning-confirm5k-execution-ed9463e.failed-prerequisite-20260819T0418
```

The successful installation cloned the existing clean, running source checkout
`/tmp/cofitok-label-ranking-sampling-standing-auth-f77e311`, verified the same
prerequisite, verified the incremental bundle, fetched only the advertised
execution branch, and switched to the exact execution revision. The formal
checkout remained unchanged at
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066` with zero
tracked changes.

The persistent supervisor is CPU-only while waiting:

```text
nice: 15
ionice: idle class
CUDA_VISIBLE_DEVICES: -1
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
child PID: null
status: waiting
detail: waiting_for_shared_pass_5k_sampling_validation
idle GPU polls: 0
```

The immutable source report was still absent, and the 5K training output root
and its lock were both absent. Therefore the supervisor created only its
control-plane PID/status/log files and did not prepare, reserve, or launch GPU
work. The first verified status snapshot was `3,742` bytes with SHA256
`802aa404081cbce92cb4d87045d48db4dfcb390915aa8a11a4024fc750ff932b`;
the PID manifest was `384` bytes with SHA256
`24eae1c3c2773964590561aa26142529fa9ba38512366f704882d5677748d01c`.
The status file is expected to be atomically refreshed while polling, so this
identity is a timestamped deployment snapshot rather than a permanent expected
SHA.

At the `2026-08-19T04:21:07+08:00` verification, the only GPU process remained
PID `79894`, the active matched quality-bridge dense trainer, which had reached
step `15,100/50,000`. The new supervisor had no child process and did not alter
GPU occupancy. Its execution revision remains `ed9463e` even when the local
evidence branch advances with this deployment record.
