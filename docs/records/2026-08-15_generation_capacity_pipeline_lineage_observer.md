# Capacity generation pipeline lineage observer

Date: 2026-08-15

## Outcome

A single read-only observer now covers the complete active dependency chain
from the full-data matched 100K quality bridge through capacity probing,
capacity scaling/completion, the fresh matched 300K pair, formal 50K
post-evaluation, and finalization.

The observer identifies the earliest blocking stage while independently
checking every already-deployed downstream waiter for process liveness and
status freshness. This closes the operational gap where reading only the last
300K supervisor made the actual upstream blocker difficult to locate.

It cannot launch training, sampling, evaluation, promotion, export, or release.
It cannot allocate CUDA memory or signal any process. Its only write is one
atomically replaced JSON status report plus the runbook lock and log.

## Exact identity

- branch: `scale/generation-capacity-pipeline-observer-v1`
- deployed revision: `0bbd1cc43b9bb713ab2da8d4da6404f304f88ffe`
- deployed tree: `460b001ba7459de3abc9de3f634fc81e9249a8c4`
- prerequisite postprocessing revision:
  `5e04dbe488ee166d2756bc99c71ee5fffbe0bd83`
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-pipeline-observer-0bbd1cc/CoFiTok-internal`
- incremental bundle:
  `/tmp/cofitok-capacity-pipeline-observer-0bbd1cc.bundle`
- bundle bytes: `14,197`
- bundle SHA256:
  `4ba05a67d2c448f432683972fb6bd3605c522c08dd4871fa4b5a0d55dfed9d0e`
- lineage-plan SHA256:
  `f6e194873f8c526b79aa2380ede271a4319d985e8935aab2dc21ae2067e28c04`

The bundle advertises one exact branch and requires the deployed
postprocessing revision. The formal checkout was not fetched, merged, or
moved.

## Observed lineage

The plan contains 14 blocking stages in order:

1. quality-bridge matched 100K recovery;
2. quality-bridge follow-up decision;
3. capacity-probe preparation;
4. matched 250M 10K capacity probe;
5. capacity-scaling decision;
6. matched 250M 50K capacity scaling;
7. capacity-completion decision;
8. matched 250M 100K capacity completion;
9. source-replayed 100K completion result;
10. capacity-full 300K readiness decision;
11. capacity-full CUDA/storage readiness;
12. fresh matched 300K training;
13. formal matched 50K post-evaluation and final gate;
14. inference export, completion audit, and release receipt.

Two nonblocking auxiliary waiters are also observed: preservation of the
quality-bridge step-10K references and the requested terminal class visual
audit.

Each stage binds an exact expected role and absolute status path. Active states
must contain a live PID and a timezone-aware update no older than 240 seconds.
Swapped roles, malformed JSON/status values, missing files, dead processes, or
stale active reports fail the observer. A hold remains a hold and cannot be
reported as completion. `pass` requires every blocking stage to report a
recognized success state.

## Validation

Local:

- targeted lineage tests: `8 passed`;
- observer, runbook syntax, and capacity-postprocessing compatibility:
  `21 passed`;
- complete runbook entrypoint contract: `2 passed`;
- Python compile, `bash -n`, and `git diff --check`: passed.

Exact Linux isolated revision with CUDA hidden:

- source-scoped lineage, syntax, and entrypoint suite: `13 passed`;
- Python compile and `bash -n`: passed;
- a real one-shot snapshot returned `waiting` with no issues and verified all
  16 status processes alive.

## Deployment

- PID: `319645`
- output:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_generation_pipeline_lineage_observer.json`
- deployment receipt:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_generation_pipeline_lineage_observer_deployment_receipt.json`
- deployment receipt bytes: `8,581`
- deployment receipt SHA256:
  `f3546a98b0adf498e6e7464199e987761073739258e94d8bafe82bc3d8938463`

The first live report correctly identifies
`quality_bridge_100k_recovery / waiting_for_gpu_idle` as the current blocker.
The sole GPU process is unrelated FieldScope PID `910099`; the observer records
it but does not signal, share, or otherwise modify it. The capacity-full output
root and training launch receipt remained absent.

The formal checkout snapshot remained:

- revision: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
- branch: `scale/generative-system`
- porcelain count: `87`
- porcelain SHA256:
  `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`
