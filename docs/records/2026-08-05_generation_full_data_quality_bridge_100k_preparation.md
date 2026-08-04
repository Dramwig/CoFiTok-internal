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

Local validation on the project `.venv` completed with:

- Python compilation of all eight quality-bridge library/CLI entrypoints;
- the complete repository suite: `1,101 passed`, `6 skipped`, no failures;
- all quality-bridge, milestone, matched-pair, training-audit, sampling,
  generation-metrics, class-fidelity, and direct-runbook-entrypoint tests;
- `git diff --check` and an explicit trailing-whitespace scan of the new
  quality-bridge files;
- Linux `bash -n` for all `109` generation runbooks, including both untracked
  quality-bridge runbooks.

A second CPU-only Linux rehearsal ran in an isolated
`/tmp/cofitok-quality-bridge-rehearsal.*` checkout on `pro6000`. It reconstructed
the target `e02ba00ea4793c6cd74f3aa667fe5ea9332f9cfe` from a prerequisite-bound
incremental Git bundle, overlaid the current quality-bridge worktree files,
normalized the transport-only CRLF endings to the LF form produced by a Git
checkout, and ran with `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, and
`MKL_NUM_THREADS=2`. Results:

```text
109 targeted tests passed
109/109 generation runbooks passed bash -n
quality-bridge entrypoint py_compile passed
git diff --check passed
linux_rehearsal_status=pass
```

The formal remote checkout remained at `1ebcc15210e63a776a2ba448481cbd8bb94a4066`;
no fetch, checkout, deployment, training, or GPU evaluation was performed there.
The rehearsal checkout, bundle, overlay, and all local/remote temporary archives
were removed after verification. FieldScope PID `433140` remained the only GPU
compute process and was not signaled or modified.

The source-bound machine-readable rehearsal receipt is:

```text
artifacts/reports/generation/stability_full_data_quality_bridge_100k_linux_rehearsal_2026-08-05/rehearsal_summary.json
```

Its 20 bound source files were independently rehashed after writing the receipt.

## Remaining work before execution

1. Finish repository-wide validation and review the complete dirty worktree
   without overwriting unrelated user evidence.
2. Commit the bridge as a dedicated clean target revision and rehearse it in an
   isolated Linux checkout.
3. Deploy only to a dedicated clean checkout after explicit user approval.
4. Never interpret a bridge pass as a full-300K authorization; a new
   source-compatible schema-v4+ gate and explicit user decision remain required.
