# Full-data 100K base-128 quality bridge preparation

Date: 2026-08-05

## Purpose

The frozen stability 50K matched pair completed correctly, and its formal EMA
post-evaluation is source-valid. The resulting scaling gate is a scientific
`hold`, not an operational failure. This record defines a non-authorizing
intermediate experiment that separates data coverage and additional optimization
from the later 250M-parameter capacity increase.

No GPU job was launched while preparing this bridge. The active FieldScope
process on `pro6000` was left untouched.

## Frozen source evidence

Authoritative root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher
```

Frozen promotion gate:

```text
reports/promotion_gate.json
bytes: 31870
sha256: 2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90
status: fail
decision: hold
only failed row: absolute_fid_quality
```

The formal metrics were:

| method | FID | precision | recall |
|---|---:|---:|---:|
| CoFiTok | 138.29702495267782 | 0.7505000233650208 | 0.008679999969899654 |
| dense identity | 151.4476773464495 | 0.78329998254776 | 0.00977999996393919 |

CoFiTok remained better than dense in FID, and the endpoint, ordered-prefix,
compressed-token-energy, zero-token, and shuffled-token checks passed. The
shared recall collapse is therefore stronger evidence for data-distribution
coverage or training-scale limits than for a CoFiTok-only factorization
collapse.

The source pair used `imagenet_256_10pct` with 128,161 train images, effective
batch 64, and 50,000 optimizer steps. Each method therefore saw 3,200,000
training images, or 24.968594 equivalent passes over the 10% split. The final LR
had already reached `1e-5`.

The full `imagenet_256` record contains 1,281,167 train images and 50,000 val
images. Its manifest is 405,484,553 bytes with SHA256
`9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`.

## Selected bridge

The bridge uses:

- full `imagenet_256`;
- the same 128-channel scalable U-Net backbone and approximately 62.8M matched
  parameter counts as the completed stability pair;
- the same fixed-basis K8 CoFiTok factorization, dense identity baseline,
  factorization losses, rollout consistency, and EMA-teacher schedule;
- effective batch 64, bf16, EMA, required integrity sidecars, and exact resume;
- 100,000 optimizer steps, with protected matched checkpoints at 50K and 100K.

Budget selection is evidence-driven:

| step | images seen | full-data equivalent epochs | role |
|---:|---:|---:|---|
| 50K | 3,200,000 | 2.497722 | intermediate matched trend point |
| 100K | 6,400,000 | 4.995445 | selected terminal bridge budget |
| 150K | 9,600,000 | 7.493167 | rejected; 50% more cost mainly after the main unique-data coverage window |

The 100K horizon brackets the approximately four-epoch data-repetition
inflection while retaining a 50K slope point. A 50K-only run would be cheaper
but would not distinguish first-pass full-data coverage from continued
optimization; 150K has a weaker information-per-GPU-hour tradeoff.

The stability-loss windows remain at the qualified absolute schedule:
rollout-consistency warmup ends at 10K, EMA-teacher consistency starts at 30K
and reaches full scale at 40K. The bridge changes neither model capacity nor the
qualified mechanism objective. The cosine LR horizon necessarily extends to
100K; therefore the 50K milestone is a trend diagnostic, not an exact
single-variable replay of the old 50K terminal schedule.

## Evaluation contract

At both 50K and 100K, the future execution runbook must produce matched EMA
DDIM-50 / CFG 1.5 / guidance-rescale 0 samples (2,048 per method) plus a
256-image checkpoint mechanism audit. These are non-claim trend diagnostics.

At exact 100K, the terminal decision evidence must use at least 10,000 samples
per method under one fixed random stream, EMA, bf16, DDIM-100, CFG 1.5, and the
same full validation directory and evaluator. It must compute FID, Inception
Score, precision, and recall; `--skip-prc` is forbidden. A later result builder
must bind the physical training, checkpoint, sampling, metric, and class-fidelity
sources before any scientific decision is consumed.

## Added artifacts

Configurations:

```text
configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json
configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json
```

Preparation and replay validation:

```text
src/cofitok/generation/quality_bridge.py
scripts/build_generation_quality_bridge_preparation.py
scripts/validate_generation_quality_bridge_preparation.py
scripts/validate_generation_quality_bridge_execution_approval.py
scripts/build_generation_quality_bridge_launch_receipt.py
scripts/validate_generation_quality_bridge_launch_receipt.py
scripts/build_generation_quality_bridge_result.py
scripts/verify_generation_quality_bridge_result.py
artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_prepare.sh
artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh
```

The deterministic preparation report validates the frozen gate sources, exact
matched configurations, formal dataset identities, unchanged model parameter
counts, and preserved model/loss recipe. Its authorization boundary is fixed to:

```text
quality_bridge_launch_allowed=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
report_is_promotion_gate=false
new_gate_required=true
explicit_execution_approval_required=true
```

The preparation runbook only writes small validation reports and a storage
preflight. It contains no training entrypoint and cannot launch the bridge or
the full 300K queue.

## Dormant execution and terminal evidence contract

The execution runbook is now implemented but remains inert without all of:

- a clean exact execution revision and branch;
- immutable preparation and user-created execution-approval SHA256 values;
- `QUALITY_BRIDGE_EXECUTION_ALLOWED=true`;
- an idle GPU and passing launch-time storage preflight;
- an exclusive execution lock and exactly bound pair monitor;
- either absent initial training state or a byte-reproducible launch receipt
  whose SHA is supplied for resume.

The runbook selects a matched effective-batch-64 runtime, freezes it in a launch
receipt before training, alternates CoFiTok and dense at 50K/100K, and uses
required checkpoint integrity plus progress audits. An active matching trainer,
an unreceipted run directory, or an unbound/duplicate monitor is rejected. The
runbook contains no full-300K entrypoint.

The terminal result builder independently replays and physically binds:

- both training reports, progress audits, protected 50K/100K checkpoints and
  current integrity sidecars;
- both matched milestone reports and their raw source reports;
- terminal sampling preflights and checkpoint mechanism evaluations;
- each physical numbered PNG tree, sampling report, immutable manifest,
  completed progress receipt, and sample-set SHA256;
- the physical full validation tree and its content digest;
- FID/IS/precision/recall reports and exact class-fidelity reports.

The quality screen passes only when absolute FID, matched FID, absolute and
matched precision/recall, endpoint tolerance, ordered-prefix rank, coarse-token
utilization, zero-token, shuffle mismatch, and class fidelity all pass. It is
still non-authorizing, rejects a training-pair report carrying a full-launch
authorization, and always records:

```text
full_training_launch_allowed=false
full_300k_launch_allowed=false
release_authorization_allowed=false
new_gate_required=true
```

## Read-only operational recheck

At `2026-08-05T06:40:40+08:00`, no quality-bridge or 300K run had been launched.
The frozen promotion gate still had SHA256
`2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90`,
`decision=hold`, and only `absolute_fid_quality` failed. The post-evaluation
waiter remained `pass/formal_ema_postevaluation_completed`, while the older
supplemental waiter was terminal `failed` on a stale-posteval check, the frozen
class-fidelity qualification was absent, and readiness remained failed with
`full_training_launch_allowed=false`. These states do not authorize or launch
the quality bridge.

An unrelated FieldScope extractor was active as PID `433140`, started
`2026-08-04 19:07:03 +08:00`, from
`/root/autodl-tmp/FieldScope/FieldScope-internal`, using approximately
`15,412 MiB` GPU memory. It was left untouched. The bridge must not execute
while that or any other unrelated GPU compute process remains.

## Validation completed

The bridge-owned implementation was isolated into the dedicated clean code
commit:

```text
branch: scale/generation-stability-quality-bridge-100k
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree: 6cef27723196fd363379bca2e7b85b1678ebd777
subject: Prepare non-authorizing full-data 100K quality bridge
```

The commit changes exactly 24 bridge-owned files. Mixed sampling-recovery,
trajectory, waiter, acceptance, and other user worktree changes were not added
to the commit and were not cleaned or overwritten.

A prerequisite-aware bundle was then built directly from this exact commit. It
advertised one head, required both formal history anchors, and verified locally
and against the formal remote repository:

```text
bundle bytes: 40133208
bundle sha256: 6b867f3789bef059442ee689ca468d81f9535570be17d56a364250b10c759e14
advertised revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
prerequisite 1: 1ebcc15210e63a776a2ba448481cbd8bb94a4066
prerequisite 2: 58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

The final CPU-only Linux rehearsal used a named isolated branch at the exact
target commit under `/tmp/cofitok-quality-bridge-cf0e5fa-rehearsal`, with an
independent `cp -a` copy of the real sibling `paper/` directory. It did not use
an overlay or detached synthetic source. Environment controls were
`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
`PYTHONPATH=.:src`, Python 3.12.3, and pytest 9.0.3. Results:

```text
37 targeted tests passed
1077 tests collected
1075 tests passed, 2 skipped, 0 failed
8/8 quality-bridge Python entrypoints passed py_compile
107/107 tracked shell runbooks passed bash -n
git diff-tree --check passed
isolated tracked worktree remained clean
linux_rehearsal_status=pass
```

All 24 changed files were independently rehashed from the exact Linux checkout.
The formal remote checkout remained tracked-clean at
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066` before and
after the rehearsal; it was not fetched, checked out, merged, or deployed. No
quality-bridge process existed afterward. FieldScope PID `433140` remained the
only GPU compute process, with the same argv/cwd/start identity and approximately
15,412 MiB allocation; it was not signaled or modified. Final filesystem free
space was `303,760,764,928` bytes.

The isolated remote checkout, remote bundle, local bundle, synthetic local
rehearsal worktree/branch, paper junction, and empty `C:\qb` temporary root were
removed after verification. Unrelated worktrees, bundles, and dirty evidence
were left untouched.

The source-bound machine-readable rehearsal receipt is:

```text
artifacts/reports/generation/stability_full_data_quality_bridge_100k_linux_rehearsal_2026-08-05/rehearsal_summary.json
```

The receipt is `12,728` bytes with SHA256
`eb6f79a8daa45447af7c6aaa64d849c74911bbd62731bc57fdb7dab105ec2cb2`.
The schema-v2 receipt supersedes the earlier dirty-overlay draft that targeted
`e02ba00` and bound only 20 files. The current receipt fixes the execution target
to `cf0e5fa`, binds the complete 24-file source closure, records both bundle
prerequisites, and retains every non-authorizing safety flag as false.

## Remaining work before execution

1. Obtain explicit user approval for this exact `cf0e5fa` execution target.
2. Wait for all unrelated GPU compute to disappear, then independently recheck
   GPU idleness, storage, the frozen gate SHA, and the formal repository state.
3. Deploy only to a dedicated clean checkout and run the prepare/approval/launch
   receipt chain without moving or mutating the formal checkout.
4. Execute the matched 100K bridge only from the immutable launch receipt; do
   not resume unreceipted state or duplicate a trainer/monitor.
5. Never interpret a bridge pass as a full-300K authorization; a new
   source-compatible schema-v4+ gate and explicit user decision remain required.
