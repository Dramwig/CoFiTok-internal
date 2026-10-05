# Downstream visual-audit reconciliation (2026-08-31)

## Scope

This is a CPU-only state reconciliation. It does not launch training,
sampling, checkpoint replay, promotion, release, or inference export. No
remote files or existing reports are modified.

## Live evidence

The authoritative quality-bridge root remains:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1
```

The pair monitor is `pass` with `issues=[]`; both 100K training processes are
absent and the RTX PRO 6000 reports `0 MiB` used with no compute application.
The standing authorization SHA256 is still:

```text
5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
```

The terminal quality result remains `status=hold`. The matched terminal rows
are:

| method | FID | precision | recall |
| --- | ---: | ---: | ---: |
| CoFiTok K8 | 115.2622 | 0.7556 | 0.00832 |
| dense identity | 123.0210 | 0.6653 | 0.01000 |

Matched FID and precision favor CoFiTok, but the absolute FID ceiling,
recall floor, and class-fidelity qualification fail. The scientific claim is
therefore still `generation_advantage_proven=false`.

## Downstream status correction

The current canonical waiter statuses are fail-closed, not operational pass:

- `terminal_system_claim_guard_v1/waiter_status.json`: `failed`, detail
  `upstream_terminal_evidence_failed`.
- `quality_bridge_comparison_v1/waiter_status.json`: `failed`, detail
  `terminal_system_claim_guard_waiter_failed`.
- `terminal_completion_audit_v1/waiter_status.json`: `failed`, detail
  `upstream_terminal_evidence_failed`.
- `requested_class_visual_audit_waiter_status.json`: `failed`, with
  `child_exit_code=1` and detail `terminal_visual_audit_child_failed`.

The visual-audit failure is explained by the quality checks: absolute FID,
recall floor, and `class_fidelity=hold` are failed checks. A later schema-6
source-rebind report completed its physical/source validation, but it records
the same scientific hold and does not replace the canonical terminal claim
guard or the missing class-fidelity qualification.

The existing downstream waiters remain untouched. No failed status is
overwritten, no duplicate waiter is started, and no GPU route is inferred
from the schema-6 diagnostic.

## Boundary and next action

The 1000-sample sampling-recovery diagnostic remains technically passing but
has `no_shared_sampling_recovery_candidate`; the Min-SNR gamma-5 pilot has
`no_shared_min_snr_candidate_at_50k`. Factorization and conditioning
supervisors remain waiting without diagnostic children. Training, sampling,
full-300K, promotion, release, and process-signal permissions remain false.

The next permitted work is static diagnosis and preparation of a new
source-compatible training-objective gate. This reconciliation does not
authorize that gate or any GPU experiment.
