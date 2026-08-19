# Conditional full-data conditioning-ranking 100K preparation

Date: 2026-08-20

## Purpose

Prepare a fail-closed, CPU-only contract for a possible fresh full-data
ImageNet-256 matched 100K conditioning-ranking bridge. This route is eligible
only if the active quality bridge eventually selects class-conditioning
recovery and the independent four-arm posttraining 5K sampling confirmation
passes for both CoFiTok and `dense_identity`.

This work does not launch training or sampling. It does not repair or resume
the active quality-bridge checkpoints, authorize 300K training, promote a
checkpoint, authorize release, or support a CoFiTok-specific generation claim.

## Current upstream state

The preparation was deliberately completed before its source gates became
available so the future decision can fail closed without an implementation
delay. At the live observation used for this record:

- the active full-data quality bridge was still running;
- CoFiTok was stopped at its protected 50K milestone;
- dense was at step 47,950 with 3,068,800 images seen and finite metrics;
- the terminal `quality_bridge_result.json` and follow-up experiment decision
  were absent;
- the posttraining sampling-confirmation supervisor was `waiting` with detail
  `waiting_for_exact_5k_heldout_evaluation` and `child_pid=null`;
- the posttraining sampling-confirmation output root was absent;
- the proposed ranked full-data 100K output root was absent.

The mutable posttraining supervisor snapshot had SHA256
`ecd739f1c340fbec825209b4e6e116a479ae85f63926f3a8aa3351bd39231d47`.
That digest binds only this observation and is not terminal evidence.

Therefore the ranked full-data bridge is not currently execution-eligible.
No placeholder report, synthetic pass, or handoff assertion can satisfy the
missing source gates.

## Frozen candidate recipe

The two ranked configs add the same class-conditioning ranking schedule to the
existing full-data quality-bridge recipes:

```text
weight:             0.05
start step:         10,000
warmup steps:       20,000
batch fraction:     0.0625
margin:             0.01
wrong-label offset: 500
minimum timestep:   500
```

All non-ranking fields remain byte-equivalent after config normalization to
their method-specific base recipes. CoFiTok and dense share data, diffusion,
runtime, optimization, U-Net fields, epsilon loss, rollout consistency,
EMA-teacher consistency, and the complete ranking schedule.

The actual instantiated parameter counts are:

| method | base | ranked |
|---|---:|---:|
| CoFiTok K8 | 62,834,083 | 62,834,083 |
| dense identity | 62,824,707 | 62,824,707 |

The direct matched relative gap is `+0.0149240648%`, below the enforced 2%
limit. The ranking objective therefore does not create a method-specific
capacity increase.

Config identities:

| config | SHA256 |
|---|---|
| ranked CoFiTok | `81d65bf8f4a81806e563eb581be305091cbb7ac11d4a9cc2fc13ab94b21c5bca` |
| ranked dense | `b5cdafedbba720ab52a7f0981b35f0e9ff0952f9c404a946f980db1a3a0c3625` |

Both runs must start from fresh random initialization and execute serially on
one GPU for 100,000 optimizer steps at effective batch 64. Existing
quality-bridge checkpoints are evidence sources only and cannot be training
inputs. Protected checkpoints are required at 50K and 100K, followed by a
matched terminal sampling evaluation.

## Method boundary

The ranking loss acts on the shared predictor training objective for both
methods. It does not add class labels, timestep embeddings, noisy images,
previous tokens, or any other condition to `S_k`. CoFiTok synthesis remains
the same restricted, token-only, bias-free fixed-basis operator, and
`dense_identity` remains the matched direct predictor.

## Source and execution contract

The preparation builder requires and byte-binds all of the following:

1. the exact completed posttraining 5K sampling confirmation with both methods
   passing generated-sample class-alignment recovery;
2. the exact terminal quality-bridge follow-up decision selecting the
   class-conditioning recovery route because class fidelity failed;
3. the active standing experiment authorization record;
4. all four base/ranked config identities and actual parameter counts;
5. a clean exact Git revision, branch, and output root.

The report remains non-authorizing. A separately source-bound execution
receipt and five consecutive idle-GPU polls are still required before a future
runbook can launch. The standing authorization supplies the user authority;
the separate receipt supplies exact revision, stage, config, source, and
output binding without asking for per-stage approval again.

## Implementation

- `src/cofitok/generation/conditioning_ranking_full_data_bridge.py`
- `scripts/prepare_generation_conditioning_ranking_full_data_bridge.py`
- `scripts/run_generation_conditioning_ranking_full_data_bridge_preparation_supervisor.py`
- two ranked full-data 100K configs under `configs/generation/`
- `tests/test_generation_conditioning_ranking_full_data_bridge.py`
- `tests/test_generation_conditioning_ranking_full_data_bridge_supervisor.py`

The builder rejects non-ranking recipe drift, asymmetric posttraining results,
wrong quality-bridge routing, parameter changes between base and ranked arms,
a direct parameter gap above 2%, malformed evidence identities, altered claim
boundaries, and any report that claims GPU authorization.

## Source-gated preparation supervisor

Commit `d93f509` adds the missing CPU-only link from the active terminal
decision chain to this preparation builder. The supervisor is pinned to an
exact fully clean revision and branch, validates and byte-binds the standing
authorization, and waits for the source-replayed quality-bridge follow-up
decision.

The follow-up validator requires the exact decision-builder Git identity, all
12 terminal gate rows, consistent failed-check ordering and status, both
50K/100K milestone identities, the non-authorizing decision boundary, and one
of the frozen follow-up route IDs. A valid route other than
class-conditioning recovery terminates as `not_selected` without creating a
preparation. A malformed route fails closed.

If class-conditioning recovery is selected, the supervisor waits for the
physical posttraining 5K confirmation and calls
`replay_posttraining_sampling_confirmation`. An asymmetric or double-failed
generated-sample result terminates as `not_selected`; only the exact CoFiTok
and dense double-pass is eligible. The eligible path rebuilds the preparation
from the four tracked configs and actual CPU-instantiated parameter counts,
then atomically writes it once. A restarted supervisor must rebuild the same
payload and replay the existing preparation exactly.

The PID, mutable status, persistent preparation, and any failure status are
required to live outside both the clean Git checkout and the future ranked
training output root. The supervisor never queries `nvidia-smi`, launches an
experiment child, trains, samples, signals a process, creates the future
training root, promotes a checkpoint, or authorizes 300K/release.

At the post-implementation live check, dense was still healthy at step
`48,750/50,000` with `3,120,000` images seen. Its protected 50K checkpoint,
paired milestone report, and terminal bridge result were still absent, so the
new supervisor remained preparation-only and no downstream eligibility was
claimed.

## Verification

- Python compile: passed for the new module, CLI, and tests.
- Focused new/config/pair/training suite: 24 passed.
- Expanded conditioning-ranking, pair-contract, exact-resume, conditioning,
  and loss regression selection: 160 passed, 1,057 deselected.
- Supervisor plus full-data preparation focused suite: 17 passed.
- Expanded conditioning-ranking, pair-contract, exact-resume, and
  class-conditioning regression selection after the supervisor change:
  144 passed, 1,083 deselected.
- Real config model instantiation: counts reproduced exactly as listed above.
- No GPU process was launched, signaled, paused, or modified by this work.

Development branch:
`scale/generation-conditioning-ranking-full-data-100k-v1`

Base revision:
`0604fff6296db334b939fbe2dc2b7cf2c5936f4c`
