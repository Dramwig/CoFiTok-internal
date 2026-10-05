# Live generation-system gap and conditioning-pipeline audit

Date: 2026-08-19

## Outcome

The large-model generation objective remains incomplete, but its next two
bounded experiments are correctly source-bound and ready to proceed without a
new per-stage authorization. The active full-data 100K matched quality bridge
must remain the only GPU workload until it reaches its terminal result.

The machine-readable snapshot is:

```text
artifacts/reports/generation/generative_system_gap_audit_2026-08-19/live_gap_audit.json
```

This snapshot updates the stale 2026-08-05 audit, which predated deployment of
the quality bridge.

## Active matched bridge

The authoritative root is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
```

At the audit snapshot, `pair_monitor.json` reported:

```text
status: running
stage: dense_identity_training
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
branch: scale/generation-stability-quality-bridge-100k
issues: []
```

Progress and checkpoint identities were:

| method | latest monitored step | latest checkpoint step | checkpoint SHA256 |
|---|---:|---:|---|
| CoFiTok | 50,000 | 50,000 | `d7100a6e8f67adb2b1630d36c17d0f2bc94fb93ed99d3e270a0867979c3343b4` |
| dense identity | 12,650 | 10,000 | `7ee56aff9a966668490c83bd5552c710cbc3c683594e1f2a7b0809e0012c26cc` |

The dense JSONL tail had already reached step `12,700` and `812,800` images;
the difference is the monitor's expected polling lag. PID `79894` was the only
GPU compute process, using about `77,970 MiB` at `100%` utilization. No unrelated
GPU process or health issue was present. Available project storage was about
`319.2 GB`.

The execution record still correctly declares this run as quality-bridge only:

```text
full_training_launch_allowed=false
full_300k_launch_allowed=false
report_is_promotion_gate=false
```

Neither `quality_bridge_result.json` nor
`followup_experiment_decision.json` exists yet, so no terminal quality or
follow-up decision can be claimed.

## Automatic class-conditioning recovery chain

The 1K four-arm training supervisor is PID `210203`, bound to revision
`64f85fe3a34aebd664c083d43bfb38d1c61aaeb7`. It is waiting for the exact
quality-bridge follow-up decision and has no child process. Its immutable
preparation is `8,937` bytes with SHA256
`5872be0fe5b910fd20955c3ac4041cd64bc107447e5d47289f296e98fb769bba`.
It compares matched control and class-ranking variants for both CoFiTok and
dense identity for exactly 1,000 steps. It cannot authorize sampling, promotion,
full training, or release.

The 5K four-arm sampling supervisor is PID `245918`, bound to revision
`f77e311546beba697020debd35e26888096ca258`. It is waiting for the 1K held-out
post-evaluation and also has no child process. Its currently absent
`preparation.json` is intentional: the supervisor creates it atomically only
after both methods pass the exact 1K post-evaluation. It then requires five
consecutive idle-GPU observations and may launch its runbook at most once.

## Shared class-conditioning implementation audit

The base model and sampler use an explicit null class, training-time class
dropout, conditional/null batched CFG halves, and a separate
`force_unconditional` path for sequential CFG. Together with the existing real
ImageNet calibration of `78.37%` Top-1, no obvious label-index, null-class, or
CFG routing defect was found.

The proposed ranking loss is shared by CoFiTok and dense. It retains gradients
only through the correct-label prediction, detaches wrong-label and null-label
references, disables class dropout for the three-way comparison, and requires
both methods to pass a held-out paired decision before sampling is selected.

The exact 5K supervisor checkout was tested with CUDA disabled, one CPU thread,
and low CPU/IO priority. The nine conditioning/ranking/sampling test files
collected `62` tests and all `62` passed. Both the 1K and 5K checkouts remained
Git-clean after the test.

This evidence does not prove the repair will work. It establishes that the
current four-arm probe is a valid bounded test of a training-time semantic
alignment repair rather than a workaround for an obvious sampling bug.

## Remaining completion gaps

1. Sample quality still lacks a completed full-data 100K terminal result and a
   passing absolute FID/precision/recall/class-fidelity decision. Existing
   frozen recall and class fidelity remain contradictory evidence against a
   usable-system claim.
2. Training scale is now in progress on full ImageNet-256, but neither method
   has reached the exact 100K terminal checkpoint; no 250M matched 300K result
   exists.
3. The direct matched baseline protocol is correct, but the terminal full-scale
   comparison with measured cost, throughput, VRAM, checkpoint, sample-set,
   evaluator, and runtime identities does not exist.
4. Stable inference code is implemented but no release-authorized EMA artifact,
   passing real-checkpoint smoke, completion receipt, or release receipt exists.
5. Current 100K checkpoints use required integrity metadata, but final-system
   reproducibility still needs terminal physical rehash, protected milestones,
   exact-resume terminal checkpoints, completion replay, and distinct inference
   artifacts.

## Required next order

1. Protect the active bridge and do not start any competing GPU workload.
2. Let the existing 1K supervisor consume only the source-bound terminal
   follow-up decision.
3. If both methods pass the 1K held-out decision, let the existing 5K supervisor
   perform the single matched sampling validation after five idle polls.
4. Use the paired 5K FID/IS and requested-class fidelity evidence to decide
   whether semantic ranking is viable before any longer repair training.
5. Keep full-300K and release authorization false until a new source-bound
   scientific gate explicitly permits them.

## Continuation audit at 15:55 CST

The active bridge had advanced beyond the original snapshot. Dense identity's
source log reached step `31,450` with `2,012,800` images seen, while the pair
monitor still reported `running`, zero pair issues, zero dense health issues,
and no unrelated GPU process. The exact dense EMA-teacher transition from
30K through the first post-start validation at 31K passed its existing
source-bound waiter:

- 20 exact active rows from step `30,050` through `31,000`;
- scale `0.005` through `0.1` with finite positive consistency losses;
- exact `samples_seen == step * 64` binding;
- fixed validation at step `31,000`, seed `102030`, 64 images, MSE
  `0.025684325024485588`;
- report SHA256
  `fa1acd17f0f8aaa381683b7d4fece11f26b56afd199ba9190387ec500ee1ff53`.

The follow-up chain is now longer than the two-stage chain described in the
original snapshot. A source-bound lineage observer at revision
`a58f2e28abdd671550d7c70b0303184bbdb029e4` reports five ordered stages:

1. 1K four-arm conditioning probe, supervisor PID `210203`;
2. 5K four-arm sampling validation, supervisor PID `245918`;
3. fresh 5K four-arm training confirmation, supervisor PID `280369`;
4. CPU-only 5K held-out evaluation, supervisor PID `305233`;
5. post-training 5K sampling confirmation, supervisor PID `333312`.

At `2026-08-19T07:55:52.759100+00:00`, every stage was waiting on its exact
upstream source, every `child_pid` was null, and the observer reported zero
issues. The conditioning entry route accepts only
`run_class_conditioning_fidelity_diagnostic`. The separate capacity waiter,
PID `11787`, accepts only
`prepare_matched_250m_capacity_qualification_probe` and may build a preparation
but cannot launch training. Later conditioning stages require shared paired
passes, five idle-GPU observations where GPU work is needed, a clean exact
checkout, and a one-launch receipt. Therefore the conditioning and capacity
branches remain mutually exclusive and fail closed.

The standing authorization was independently snapshotted at 865 bytes with
SHA256
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`.
It preserves the locked-evidence, clean-checkout, exact-revision/stage/output,
and unrelated-process boundaries. It does not by itself authorize promotion,
full 300K training, or release.

Small local evidence snapshots are stored beside the machine-readable gap
audit:

```text
artifacts/reports/generation/generative_system_gap_audit_2026-08-19/dense_ema_teacher_first_post_validation_00030000_00031000.json
artifacts/reports/generation/generative_system_gap_audit_2026-08-19/conditioning_pipeline_lineage_after_dense31k.json
artifacts/reports/generation/generative_system_gap_audit_2026-08-19/standing_authorization.json
```

The lineage snapshot is 24,454 bytes with SHA256
`26db21e94dcafb8380c44c7215a0ebb4722562b20e6314f6d96d3be096ccae1d`.
The local affected test selection passed `292/292` before the supervision
hardening below. An intermediate full project suite passed `1,104` tests with
`6` expected skips under the project Python 3.10 uv environment and CUDA
disabled.

The shared prospective waiter helper originally propagated a heartbeat write
exception while leaving its already launched child running. Under ENOSPC or a
transient status-write failure, that could terminate the supervising waiter and
orphan a GPU child. `cofitok.process_monitoring.wait_for_child_with_heartbeat`
now reports the first consecutive heartbeat error, keeps waiting for the exact
child, and retries the heartbeat on every poll. It never converts a status-file
failure into child success or authorization. Targeted helper and caller tests
passed `29/29`, including a persistent heartbeat failure followed by the
child's nonzero exit code.

The same helper now covers the prospective stability post-training supervisor's
three potentially multi-hour children: formal 50K post-evaluation, inference
artifact export, and terminal completion audit. Their source-bound `running`
status is refreshed at the configured poll interval without changing retry or
scientific-gate behavior. Helper/supervisor tests passed `11/11`, runbook/CLI
contracts passed `2/2`, and the final full repository suite passed `1,105`
tests with `6` expected skips.

This prospective local hardening was not deployed into or used to alter the
active bridge. The broad usable-generation claim remains unsupported until the
bridge and its source-selected follow-up evidence complete.

## Capacity-lineage continuation audit at 16:31 CST

The earlier continuation text understated the capacity branch. It is not only a
preparation waiter: the complete source-bound capacity pipeline is already
deployed. The recovery-aware lineage observer v2 reported `14` blocking stages,
all process-bound and all free of issues:

| index | stage | PID | state |
|---:|---|---:|---|
| 0 | quality-bridge exact-resume recovery | `630988` | running |
| 1 | quality-bridge follow-up decision | `11777` | waiting |
| 2 | 250M capacity-probe preparation | `11787` | waiting |
| 3 | matched 250M 10K execution | `11853` | waiting, `child_pid=null` |
| 4 | capacity-scaling decision | `11945` | waiting |
| 5 | matched 250M 50K scaling | `12020` | waiting, `child_pid=null` |
| 6 | 100K-completion decision | `12114` | waiting |
| 7 | matched 250M 100K completion | `12127` | waiting, `child_pid=null` |
| 8 | 100K-completion result replay | `12221` | waiting |
| 9 | full-300K readiness decision | `12233` | waiting |
| 10 | full-300K readiness | `12247` | waiting, `child_pid=null` |
| 11 | full-300K matched training | `12259` | waiting, `child_pid=null` |
| 12 | full-300K formal post-evaluation | `12345` | waiting, `child_pid=null` |
| 13 | completion audit, export, and finalization | `12434` | waiting, `child_pid=null` |

The branch accepts only the exact follow-up recommendation
`prepare_matched_250m_capacity_qualification_probe`. Each execution stage is
guarded by its source replay, exact Git identities, the standing-authorization
SHA, idle-GPU observations, and a one-child launch boundary. No capacity child
has been launched.

Observer v2 PID `765120` is the current healthy recovery-aware source. At
`2026-08-19T08:30:57.601975+00:00` it was at revision
`1ff6bb3db932ef9e43ddfafc067db779dab797d9`, reported `running`, and had no
issues. Its exact report at that observation was `24,690` bytes with SHA256
`d653ad9b2995b613e02539deca8926b4fe992bf9a54e806a2501562f104b779b`.
The immutable v2 deployment receipt is `12,867` bytes with SHA256
`925e62a420bb9b6f5d84e161cc31f62ae7def0f69ec8582794ccdd4066f9de9a`.

The authority distinction is important. V2 does not formally supersede the old
observer process or report. Its deployment receipt deliberately preserves old
PID `12692` without signaling or modification, and it contains no
observer-level supersession contract. The old observer currently reports
`failed` with `formal_checkout_snapshot_changed`. What v2 formally supersedes
is only the obsolete quality-bridge recovery failure, through contract SHA256
`28318ff89c009be65899117bbb9d45f533bc116f63b6bd335e21463818d5f609`.
Accordingly, v2 is the operationally authoritative live source; the old report
is retained as historical failure evidence, not treated as a competing current
health decision.

The frozen local authority snapshot is:

```text
artifacts/reports/generation/generative_system_gap_audit_2026-08-19/capacity_pipeline_lineage_authority_after_dense32k.json
```

It is `7,498` bytes with SHA256
`e3b81d59a3e20c37c724b796e5ac802ba9188efe141e01b1123ac955928dd0a8`.
This snapshot is non-authorizing: it proves deployment and fail-closed waiting,
not generation quality, capacity benefit, full-300K readiness, promotion, or
release.

## Matched 40K trajectory freeze at 21:59 CST

Dense identity reached exact step `40,000` while the same trainer PID `79894`
remained the only GPU compute process. The training log continued to step
`40,050` after the diagnostic freeze, so no training process was paused or
signaled. Available `/root/autodl-tmp` storage was `314,068,975,616` bytes.

The source-bound dense EMA-teacher transition audit passed at full strength:

- report bytes/SHA256: `19,988` /
  `a72ff3a8cac2b7be8f9c35d852836818a9144104a665e576083c441521496d5a`;
- exact warmup-tail rows: `11`, from step `39,500` through `40,000`;
- scale range: `0.95` through `1.0`, with finite positive consistency losses;
- exact exposure binding: `samples_seen == step * 64`;
- step-40K fixed validation: seed `102030`, `64` images, epsilon MSE
  `0.030421655625104904`.

The matched trajectory waiter then atomically froze both metric prefixes and
performed two byte-identical replays with the clean schema-3 builder. The main
identities are:

| artifact | bytes | SHA256 |
|---|---:|---|
| trajectory report | 45,691 | `78a747d59988c5e02953f411aa60495c98c25f3f20b8899c79b8a528d8ff5186` |
| waiter receipt | 3,421 | `790b21413bf0574b15816dfd17924eb96bacc9c41af18a999ec0f5184675f9b6` |
| CoFiTok metrics prefix | 756,538 | `1cab16d3d08d7e24749f9776a6de1d76354d3b83e49139d708ab332c4cc88e2e` |
| dense metrics prefix | 703,263 | `15e537cdc1c301b127a88b82ce88d13c26db883195a37ac1ba4a260c3ea30218` |

Both methods had exact `2,560,000`-image exposure and `40` paired validation
events under the valid matched-generation contract. CoFiTok won `19` events
and dense identity won `21`. Their mean epsilon MSE values were
`0.029991481546312572` and `0.029969786899164318`, respectively, making the
CoFiTok ratio-of-means delta `+0.072388%`. At the single 40K endpoint CoFiTok
was lower by `0.117986%` (`0.030385762453079224` versus
`0.030421655625104904`). This is optimization-equivalence evidence, not a
sample-quality advantage.

The frozen prefix also preserves the real CoFiTok exact-resume boundary:
checkpoint step `20,000`, first post-resume step `20,001`, `401` retained
rows, and `4` orphaned rows archived with SHA256
`6b3c4bca8ffa99e814ee77559d140f3a0db6cf7b78a8d78e58ef1b716e2f1cec`.
Dense identity remained a fresh uninterrupted trajectory. Both sides have
finite metrics, strict step/exposure checks, and complete fixed-validation
provenance.

The copied local evidence is under:

```text
artifacts/reports/generation/stability_full_data_100k_base128_quality_bridge_v1/matched_40k_trajectory_2026-08-19/
```

Every local copy was rehashed and matched its remote identity. The report
explicitly keeps `quality_claim_allowed=false`,
`promotion_authorization_allowed=false`, and
`full_training_launch_allowed=false`. Required next evidence remains the exact
matched 50K milestone and its EMA sample-quality comparison; this 40K report
does not change the broad-generation hold.
