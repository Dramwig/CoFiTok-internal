# Full-data 100K quality bridge exact resume after server restart

Date: 2026-08-21 (Asia/Shanghai)

## Scope and boundary

The existing full-data matched 100K quality bridge was resumed after the
server-side controller, monitor, and trainer processes disappeared during a
server restart. This is a continuation of the already-authorized quality
bridge only. It does not authorize full 300K training, overwrite locked paper
evidence, modify the formal remote checkout, or permit signals to unrelated
projects.

Authoritative output root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
```

## Pre-resume live audit

At `2026-08-21 11:12:34 CST`:

- hostname: `autodl-container-scvpc4sj5x-0e706d96`;
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition;
- GPU memory/utilization: `0 MiB / 0%`;
- no CoFiTok, FieldScope, or other GPU compute process was present;
- `/root/autodl-tmp` had approximately `289 GiB` free;
- the formal checkout remained untouched at
  `scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`.

The immutable execution checkout remained tracked-clean at:

```text
branch:   scale/generation-stability-quality-bridge-100k
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
```

The latest atomic CoFiTok recovery point was step 80,000:

```text
checkpoint: checkpoint_step_00080000.pt
bytes:      1,010,937,514
SHA256:     ff95c24fc1d4d6723f4792167b0e699abe0bccc30f83a714d84950298378d1e8
sidecar:    checkpoint_step_00080000.pt.integrity.json
sidecar SHA256: 54244a208d08173209860acd31aa04faf3b82210c29a5d88c9300b0863625b61
```

The full 1.01 GB checkpoint payload was rehashed before launch and matched the
integrity sidecar and `latest.json`. Dense remained at its verified step-50,000
checkpoint.

The interrupted metrics file had reached step 83,450, but no checkpoint newer
than step 80,000 existed. Those rows were therefore not accepted as canonical
training progress.

## Bound authorization and launch identity

The resumed runbook revalidated the existing immutable inputs:

```text
standing authorization SHA256:
  5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df

execution approval SHA256:
  e9da52a4e7ff1b4700b70aadaa8703ee40fb1a9862e847dcfb6e75933d295a4b

preparation SHA256:
  7398d9a6f096ea9c178295c9016bb56fd38e28dff30f26662ae4225aded208ea

launch receipt SHA256:
  4a9fd8577c24a92583d9538846dcab35c2d73e6066d164050f49b976985a3a21

runbook SHA256:
  f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73
```

The launch receipt continues to state:

```text
quality_bridge_execution_authorized: true
full_training_launch_allowed:         false
full_300k_launch_allowed:              false
report_is_promotion_gate:              false
```

## Exact metrics reconciliation

The trainer resumed with `--resume auto` from step 80,000 and invoked the
canonical metrics reconciliation path before creating train/eval iterators.
The resulting report is:

```text
cofitok_rgbtail3_rollout_x0_u2_ema_teacher/
  metrics_resume_reconciliation_00080000_ea0f91f8fde6.json
```

It records:

```text
status:        reconciled
resume_step:   80000
retained_rows: 1603
orphaned_rows: 69
orphan SHA256: ea0f91f8fde6a2c04fbf3578d9184709503c5833645cb9ee7f3ecb4cd26e89c1
```

The 69 post-checkpoint rows were preserved as the content-addressed artifact:

```text
train_metrics_orphaned_at_resume_00080000_ea0f91f8fde6.jsonl
```

The first new canonical row is exactly step `80001` with
`samples_seen=5,120,064`, proving that the resumed trajectory starts from the
checkpoint rather than silently continuing from the stale step-83,450 log.

## Relaunch state

The exact original runbook was relaunched at `2026-08-21 11:36:06 CST`.

```text
controller PID: 2737
pair monitor PID: 3422
watchdog PID: 3707
trainer PID: 3773
```

Relaunch log:

```text
reports/relaunch_after_server_restart_2026-08-21T1133.log
```

The runbook recognized the paired step-50,000 milestone as already complete
and did not repeat it. It launched only the remaining CoFiTok 80K-to-100K
segment with effective batch 64. At the first fresh monitor snapshot
(`2026-08-21T03:41:20.859504+00:00`), the state was:

```text
pair:              running / cofitok_training
issues:            []
monitor step:      80050
canonical step:    80100
dense step:        50000
GPU memory:        81321 MiB
GPU utilization:   99%
```

After CoFiTok reaches exactly 100K, the same locked runbook will perform the
100K CoFiTok milestone evaluation, resume dense from 50K to 100K, build the
paired milestone, and execute the matched terminal evaluation chain.

## Downstream waiter recovery after restart

The server restart also terminated the CPU-only evidence waiters that had
been deployed around the quality bridge. Their pre-restart status files ended
at approximately `2026-08-21 07:59--08:00 CST`; those stale files were not
treated as live supervision.

At `2026-08-21 12:18:31 CST`, the downstream chain was recovered only after
revalidating all pinned checkouts, the active standing-authorization hash,
the new controller process identity, and the unchanged launch boundary. No
old PID or deployment identity was reused. A restart-specific receipt was
written at:

```text
reports/downstream_waiter_recovery_2026-08-21_server_restart_v1/
  recovery_receipt.json
```

Recovered process identities:

```text
controller identity guard:                7717
runtime compute fairness waiter:          7736
runtime claim guard:                      7748
requested-class visual audit waiter:      7767
terminal training-exposure waiter:        7777
exposure-aware follow-up waiter:          7791
terminal distribution-support waiter:     7804
preceding matched-uncertainty waiter:      7817
terminal matched-uncertainty waiter:       7878
statistical claim-qualification waiter:   7937
statistical claim-language guard:         7960
terminal system claim guard:              7981
factorization-regression supervisor:      7992
```

Every recovered supervisor was launched with:

```text
CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

The factorization-regression supervisor remains CPU-only while waiting. Its
status explicitly forbids training, promotion, release, full-300K launch, and
unrelated-process signalling. A later diagnostic child may receive GPU access
only after the terminal decision route and five consecutive idle-GPU polls;
no child or execution authorization exists at recovery time.

Post-recovery verification at approximately `2026-08-21 12:19 CST` showed:

```text
all 13 recovered PIDs: alive
controller identity guard: observing exact bound controller
all other downstream stages: waiting for exact upstream evidence
GPU compute process: trainer PID 3773 only
GPU memory/utilization: 81321 MiB / 100%
unrelated GPU processes: none
pair monitor: running / cofitok_training, issues=[]
canonical CoFiTok step: 80900
dense step: 50000
/root/autodl-tmp free space: approximately 289 GiB
quality_bridge_result.json: not yet present
```

The recovery does not restore the obsolete capacity/full-300K pipeline and
does not change the quality bridge's non-promotion boundary:

```text
full_training_launch_allowed: false
full_300k_launch_allowed:      false
promotion_or_release_allowed:  false
```

At the observed post-startup rate, the remaining CoFiTok 80K-to-100K segment
was progressing normally. This is an operational recovery result only; it is
not evidence of generation-quality superiority. That conclusion remains
blocked on the paired exact-100K training, terminal matched 10K DDIM-100
evaluation, runtime/support/statistical audits, and the eventual
`quality_bridge_result.json`.

## Follow-up live audit

At `2026-08-21 12:24 CST`, a fresh remote audit found:

```text
CoFiTok canonical metric step: 80950
CoFiTok samples seen:          5,180,800
dense canonical metric step:   50000
quality_bridge_result.json:    absent
GPU compute PID:               3773 only
GPU memory/utilization:        81308 MiB / 100%
free space:                    approximately 289 GiB
pair-monitor issues:           []
```

The pair monitor continues to bind the active training revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a` and reports no unrelated GPU
compute. The current process wall-clock comparison remains explicitly
disallowed because monitor coverage contains a historical observation gap;
the eventual runtime fairness report must be used instead.

Before leaving the running bridge untouched, the exact execution checkout ran
the CPU-only regression group:

```text
tests/test_generation_quality_bridge.py
tests/test_generation_runbook_entrypoints.py
tests/test_generation_checkpoint_evaluation.py
tests/test_generation_gate_contract.py
result: 53 passed, CUDA_VISIBLE_DEVICES=-1
```

No terminal sampling/evaluation artifact has been created yet. The quality
claim remains unqualified until the runbook produces the exact 100K paired
checkpoints, terminal matched 10K DDIM-100 reports, support/uncertainty
audits, and the verified `quality_bridge_result.json`.

At `2026-08-21 12:31 CST`, the canonical CoFiTok metric had advanced to
step `81100` (`5,190,400` images seen). The last eleven metric intervals imply
approximately `0.354` steps/s (`2.824` s/step), or roughly `14.8` hours for
the remaining 18,900 CoFiTok steps at the observed rate. This is only an
operational estimate; it is not a quality or convergence claim. GPU memory
remained stable near `89,411 MiB` for the short observation window, with the
trainer as the sole current compute application.

At `2026-08-21 12:52 CST`, the canonical metric reached step `81600`
(`5,222,400` images seen); dense remained at step `50000`. The pair monitor
still reported `running / cofitok_training` with `issues=[]`, and the GPU
contained only trainer PID `3773`.

While the GPU was occupied, the server `pf-vlm` environment ran two CPU-only
test groups for the stable-inference and audit pillars:

```text
session / sampling / preflight / checkpoint / completion tests: 196 passed
gate / provenance / metrics / resume / pair-contract tests:    109 passed
total:                                                          305 passed
```

These are implementation evidence only; no quality claim or release status
was inferred from them.

## Live audit artifact and post-run hardening candidate

The read-only training-progress audit snapshot is mirrored locally under:

```text
artifacts/reports/generation/stability_full_data_100k_base128_quality_bridge_v1/
  live_training_audit_2026-08-21_1232/
```

The dense audit was `healthy`. The initial CoFiTok audit was `invalid` only
because the previous segment's report had `completed_steps=50000` while the
canonical resumed metrics had reached `81200`; checkpoint integrity, schedule,
validation provenance, and exact-resume reconciliation checks all passed. The
resume-aware rerun preserved the same diagnostic because the old report has
`stop_requested=false`, which is not the legacy pause-report contract.

An isolated post-run hardening candidate was implemented and tested in commit
`056073337efe7bf089d28bb7a7d0553e18c33d6a` (worktree `C:/qblive`). It adds a
separate opt-in acceptance route requiring a verified newest checkpoint and a
content-addressed reconciliation artifact. It was not deployed into the
active training checkout, so the running bridge remains exactly pinned to
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`.

## Post-restart canonical-metrics stability audit

At approximately `2026-08-21 13:05 CST`, a fresh CPU-only read of the
canonical JSONL showed CoFiTok at step `81,850` (`5,238,400` images seen).
The audit checked every canonical row, rather than only the latest line:

```text
rows:                         1,641
strictly increasing steps:   yes
duplicate/non-increasing:    0 / 0
non-finite numeric values:   0
samples_seen mismatches:     0
validation events:           81
validation event indices:    continuous 0..80
recent validation steps:     77K, 78K, 79K, 80K, 81K
recent throughput:            2.8262 seconds/optimizer-step
```

Windowed means remained stable across the latest exact-resume boundary:

| window | epsilon mean | total-loss mean | grad-norm mean |
|---|---:|---:|---:|
| 70K--80K | 0.0267208 | 0.0475859 | 0.2891745 |
| 80K--81.85K | 0.0268046 | 0.0471896 | 0.1501876 |

The short 80K+ window therefore shows no numerical instability or resume
discontinuity. This is operational training-health evidence only. It does not
establish image quality, distribution support, class fidelity, or superiority
over the matched dense predictor; those conclusions still require both exact
100K reports and the terminal matched sampling/evaluation chain.

## Existing 50K full-data trend milestone recheck

The already-completed quality-bridge milestone at step `50,000` was re-read
from the remote source and validated with the exact pinned checkout. The
report is bound to SHA256
`14992df2cc5aad1576dc10d147a8f1caa4f196f066339cbce7c7a55fc85bac42` and the
validator returned `status=verified` with no warnings. It remains explicitly a
training-quality trend diagnostic, not a formal generation gate.

| method | FID | IS | endpoint clean MSE | path-AUC rank |
|---|---:|---:|---:|---:|
| CoFiTok K8 | 221.0507 | 4.4842 | 0.0156311 | 1/6 |
| dense identity | 192.1684 | 5.1053 | 0.0155271 | 1/1 |

The milestone therefore shows a relative FID disadvantage for CoFiTok at this
early full-data checkpoint, while the ordered prefix mechanism diagnostics
remain valid (`zero_token_max_abs=0`, shuffled/ordered endpoint ratio
`130.35`, ordered path rank `1/6`). This mixed result is important: mechanism
evidence and sample-quality evidence must remain separate, and no claim of
generation-quality superiority is justified before the exact paired 100K
terminal protocol completes.

## Follow-up regression-probe guard recheck

The prepared factorization-regression route was re-read after the server
restart. Its exact execution checkout is
`scale/generation-factorization-quality-regression-v1@8749138f4b8e144ccbcae0f897e3237dc5aacada`
with tree `994c0837c7c3e6f3a2f389cb0a6a351f06d89757`; tracked files are clean,
the runbook passes `bash -n`, and the preparation SHA is bound in the live
supervisor receipt. The protocol is observational only: two seeds, 64 images
per method, EMA checkpoint step 100K, DDIM-100 rollout stability and checkpoint
mechanism evaluation.

The live supervisor still reports:

```text
followup_experiment_launch_allowed: false
training_launch_allowed:             false
full_300k_launch_allowed:            false
checkpoint_promotion_allowed:        false
child_pid:                            null
idle_gpu_polls:                       0
```

The runbook itself fail-closes on a missing source binding, execution
authorization, terminal-system guard, or five consecutive idle-GPU polls. It
cannot train a model or promote a checkpoint. This recheck is preparation
evidence only; no follow-up GPU diagnostic was launched while the quality bridge
trainer remains active.

The exact isolated checkout then reran its two directly relevant CPU-only test
files with CUDA hidden:

```text
tests/test_generation_factorization_quality_regression.py
tests/test_generation_factorization_quality_regression_supervisor.py
result: 15 passed

The exposure-aware decision waiter was also rechecked. Its live scope remains
source observation and CPU-only decision construction; it explicitly forbids
GPU use, sampling/evaluation launch, training, promotion, release, and full
300K launch. Its exact checkout is
`analysis/generation-quality-bridge-exposure-routing-v1@85e3ece1196fd318cd6823439824e19fca4275a3`
with tree `1bf21fa0c44cec90011287b78f6f2ec14ed5eb17`, and the related CPU-only
test files returned:

```text
tests/test_generation_quality_bridge_exposure_followup_waiter.py
tests/test_generation_quality_bridge_followup_decision.py
result: 18 passed
```

The waiter remains `waiting_for_quality_bridge_result`; no follow-up decision
or execution authorization has been created before the terminal result exists.

## Matched-baseline provenance version check

The active pinned execution checkout currently defines
`COMPARISON_REPORT_SCHEMA_VERSION = 8`. The older July provenance note that
mentions schema v7 is retained as a historical record and is not treated as the
current terminal comparison contract. The current v8 implementation and its
completion-audit tests continue to enforce:

```text
primary_direct_tier:              matched_training_direct
external_context_tier:            official_pretrained_contextual
cross_tier_numeric_ranking_allowed: false
```

Thus D-AR, MAR, and ReTok remain contextual/eval-only rows, while only the
matched CoFiTok K=8 versus dense identity pair can receive a direct quality
comparison. This version check changes no locked run or artifact.

The exact quality-bridge checkout reran the comparison builder and terminal
completion-audit test files with CUDA hidden; both completed successfully
(`98 passed`).

## Removal of accidental substring-monitor contamination

A previous read-only shell inspection had been malformed by an unquoted
extended-regex expression. It left a blocked `grep` subprocess whose command
line contained the quality-bridge runbook name, so the substring-based pair
monitor listed it as an additional runbook candidate. Exact inspection showed:

```text
PID 17253: bash -c ... grep -nE stop-after-steps|...|runbook ...
PID 17254: grep -nE stop-after-steps
GPU allocation: none
project/controller role: none
```

Only these two exact accidental PIDs were sent `SIGTERM`. The authoritative
controller PID `2737`, pair monitor PID `3422`, training watchdog PID `3707`,
and trainer PID `3773` remained alive; the trainer stayed the sole GPU compute
process and canonical metrics advanced to step `82,100`. No project training,
evaluation, waiter, or unrelated-process PID was signalled.

The next authoritative pair-monitor poll at
`2026-08-21T05:16:26.416567+00:00` no longer listed PID `17253` and retained
`status=running`, `stage=cofitok_training`, `issues=[]`, and no current
unrelated GPU process. The monitor's broad substring list still includes the
runtime-fairness waiter because that waiter's argv names the active runbook;
the independently running controller-identity guard resolves this expected
candidate contamination by exact PID, start ticks, executable, cwd, and
cmdline SHA binding to controller PID `2737`.
```

## Live invariant recheck during resumed segment

At `2026-08-21 13:35:27 CST`, the source-bound controller remained active on
the same execution revision. The only GPU compute application was trainer PID
`3773`; pair-monitor status was `running / cofitok_training` with
`issues=[]`. No unrelated GPU process was present and approximately 289 GiB
remained free on `/root/autodl-tmp`.

An independent CPU-only read of both canonical metrics/checkpoint paths found:

```text
CoFiTok canonical last step:       82,500
CoFiTok samples_seen:              5,280,000
CoFiTok metric rows:               1,654
CoFiTok validation events:         82
CoFiTok latest protected ckpt:     step 80,000
CoFiTok checkpoint SHA256:         ff95c24fc1d4d6723f4792167b0e699abe0bccc30f83a714d84950298378d1e8

dense canonical last step:         50,000
dense samples_seen:                3,200,000
dense metric rows:                 1,001
dense validation events:           50
dense latest protected ckpt:       step 50,000
dense checkpoint SHA256:           db3fb9a5283fbf6b68eb30279618476768b81fac746bfbbc2b09abdfdcda33c1
```

For both methods, the independent audit returned no issues: steps were
strictly increasing, all numeric metrics were finite, `samples_seen` matched
`step * 64`, validation events were present, and the latest checkpoint bytes
and SHA256 matched the adjacent integrity sidecar and `latest.json` binding.
The terminal 100K milestone, runtime fairness report, quality bridge result,
and downstream claim guards remain absent or waiting; no terminal quality
claim or larger-training authorization was inferred from this operational
check.

## Downstream lineage and stale-preparation audit

At `2026-08-21 13:39:19 CST`, the server was checked for downstream control
plane contamination before the quality bridge reaches its next checkpoint.
Several older conditioning/capacity preparation directories remain under
`checkpoints/generation`, but the PIDs recorded in their historical
`supervisor_status.json` files were not alive. They were treated as archived
state, not as active supervisors, and no process was signalled or removed.

The live post-restart processes are the source-bound quality-bridge controller,
pair monitor, watchdog, identity guard, runtime-fairness/claim waiters,
terminal visual/support/uncertainty waiters, statistical claim guards, and the
factorization supervisor. The latter has `child_pid=null` and remains waiting
for the exact terminal decision. The active GPU query still reports only
trainer PID `3773`.

The live scope files continue to enforce:

```text
gpu_execution_allowed:             false   (all downstream waiters)
training_launch_allowed:          false
sampling/evaluation launch:        false until the exact upstream source exists
followup_experiment_launch_allowed: false
full_300k_launch_allowed:         false
promotion_or_release_allowed:     false
```

This confirms that completion of the current 100K bridge will be consumed only
by its bound terminal evidence chain; historical capacity/conditioning
preparations cannot silently start a competing pipeline.

## Current post-restart continuation check

At approximately `2026-08-21 13:51 CST`, a fresh read confirmed that the
controller and trainer were still bound to the immutable execution identity
`scale/generation-stability-quality-bridge-100k@cf0e5faa94bf4ab38d947b921935b3b765b5537a`
with tree `6cef27723196fd363379bca2e7b85b1678ebd777` and a clean tracked tree.
The pair monitor reported `running / cofitok_training` with `issues=[]`; the
only current GPU compute process was trainer PID `3773` (about 89.4 GiB and
100% utilization), and no unrelated GPU process was present.

Canonical metrics had advanced to:

```text
CoFiTok: 82,850 steps / 5,302,400 images seen; latest protected checkpoint: 80,000
dense:   50,000 steps / 3,200,000 images seen; latest protected checkpoint: 50,000
```

The dense run remains queued for the controller's automatic post-CoFiTok
continuation. `quality_bridge_result.json`, terminal sampling reports, and
claim guards are still absent; no terminal quality decision, promotion, release,
or 300K launch authorization exists.

The local CPU-only quality-bridge regression suite completed successfully
(`tests/test_generation_quality_bridge.py`). This verifies the non-authorizing
boundaries and replay checks only; it is not generation-quality evidence.

## Afternoon continuation verification

At approximately `2026-08-21 14:29 CST`, two short independent polls confirmed
that the resumed canonical trajectory continued advancing from step `83,550`
through step `83,650` without a process restart. The latest observation was:

```text
CoFiTok: 83,650 steps / 5,353,600 images seen / 1,677 canonical rows
dense:   50,000 steps / 3,200,000 images seen / 1,001 canonical rows
pair monitor: running / cofitok_training / issues=[]
GPU compute: trainer PID 3773 only, approximately 89.4 GiB
quality_bridge_result.json: absent
followup_experiment_decision_exposure_aware_v2.json: absent
```

The immutable execution checkout still resolved to revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`, tree
`6cef27723196fd363379bca2e7b85b1678ebd777`, branch
`scale/generation-stability-quality-bridge-100k`. The formal remote checkout
remained tracked-clean at
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`.
Available space under `/root/autodl-tmp` was `310,112,661,504` bytes.

The post-run segment-resume audit hardening was separately integrated in the
local clean worktree `C:/qbintegration`. The implementation commit remains
`3d85b9d31f641476cd18966ec42c00c48333bf82`; its validation record was committed
as `3278e94729f4a071f94866b0c641222dbbaae0e2` (tree
`7ca51cef93d2a3a6304663ad3ce943bb06fd2b6e`). The full CPU-only suite reached
100% and exited with code 0, and the targeted resume/audit suite was
`34 passed`. Neither commit was deployed to the active remote execution.

This continuation check changes no scientific or authorization result. The
terminal quality and follow-up decisions do not exist yet, and full 300K,
promotion, checkpoint release, and broad generation-superiority claims remain
disallowed.
