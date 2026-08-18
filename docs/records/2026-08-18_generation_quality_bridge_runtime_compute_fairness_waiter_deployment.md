# Full-data quality-bridge runtime/compute fairness waiter deployment

Date: 2026-08-18

## Outcome

A dedicated CPU-only terminal waiter is now deployed for the active matched
ImageNet-256 full-data 100K quality bridge. It waits for both exact 100K
training reports and then publishes the observed matched runtime and physical
compute comparison, including recovery work that is absent from the canonical
resumed metrics timeline.

The waiter is active and healthy:

```text
PID: 832803
status: waiting
detail: waiting_for_both_exact_100k_training_reports
error: null
GPU process: none
```

This deployment does not modify the active trainer, runbook, pair monitor,
checkpoints, or canonical metrics. It cannot authorize promotion, release, or
full-300K training.

## Implementation identity

```text
commit: 85bf1d6ae5b44f401153d0d1e390fc3973ff4bed
tree: 7900c5617c28c18e328c93aba4c547a3889f1aa7
development branch: fix/quality-bridge-ipc-recovery-v2
deployment branch: analysis/generation-quality-bridge-runtime-compute-fairness-v1
source: scripts/wait_generation_quality_bridge_runtime_compute_fairness.py
source bytes: 34,408
source SHA256:
0caba8d7af93c83c8862e18cc1e77e0a7d497d4d8289173942fa5f2118efdf57
```

The waiter requires both exact 100K training reports, verifies actual
micro-batch `64`, gradient accumulation `1`, effective batch `64`, and exact
`6,400,000` canonical training images per method, then reports elapsed time,
corrected throughput, peak VRAM, and the identity of any method-specific
recovery adjustment.

CoFiTok's final physical lower-bound elapsed time must include:

```text
560.6116147041321 seconds
200 optimizer steps
12,800 training images
```

If dense later produces orphaned metrics, the waiter builds and physically
verifies a dense-specific adjustment. It may use no adjustment only after the
completed dense report and run directory prove that no orphan archive exists.

## Validation

The exact implementation passed:

- Windows targeted suite: `30 passed`;
- Linux CUDA-hidden targeted suite: `30 passed`;
- Python compile validation: pass;
- `git diff --check`: pass.

Deployment bundle:

```text
path: /tmp/cofitok_runtime_compute_fairness_85bf1d6.bundle
bytes: 13,939
SHA256:
53c8dbd21f18c6988ec87ee6f2ff83ee2b26f7026ea3202644ccd5cf21d808c4
advertised head: 85bf1d6ae5b44f401153d0d1e390fc3973ff4bed
```

Persistent remote checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-runtime-compute-fairness-85bf1d6/CoFiTok-internal
```

The checkout is at the exact commit/tree above and has no tracked changes.

## Import-only preflight incident

The first `--once` preflight supplied only `<project>/src` in `PYTHONPATH`.
The waiter imports helpers from the project-root `scripts` package before
argument parsing, so this attempt exited with:

```text
ModuleNotFoundError: No module named 'scripts'
```

Because the failure occurred during module import, it did not parse output
arguments, start a waiter, use the GPU, create output files, or signal any
training or unrelated process. The corrected invocation used:

```text
PYTHONPATH=<project>:<project>/src
```

The corrected one-shot preflight passed, after which the long-lived waiter was
started once.

## Deployment receipt

Authoritative receipt:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_base128_quality_bridge_v1/reports/
runtime_compute_fairness/deployment_receipt.json
```

Receipt identity:

```text
bytes: 4,403
SHA256:
f1e60458fac0e5c6145f8a872d5bcc9f382b4afc844b7fc11739942a074fc418
status: pass
```

The receipt binds the exact auditor checkout and source, immutable launch
receipt, config validation, runtime selection, active runbook, both configs,
the expected `64x1` runtime, all output targets, and the non-authorizing scope.

## Live boundary

At deployment, the pair remained:

```text
CoFiTok: 32,300 / 100,000, 2,067,200 canonical images
dense: 0 / 100,000
pair: running / cofitok_training
issues: []
```

A later refresh at `2026-08-18T03:18:25+08:00` observed CoFiTok at step
`32,450` and `2,076,800` canonical images. The only GPU compute process was
the active trainer PID `619775` using `85,284 MiB`; the waiter was absent from
the GPU process list and no unrelated GPU process was present.

The waiter status continued to refresh every 60 seconds with `error=null`.
Neither terminal training report exists yet, so the final runtime/compute
comparison remains pending.

## Claim boundary

This evidence establishes that the terminal fairness auditor was tested,
source-bound, deployed in a clean isolated checkout, and is actively waiting.
It does not establish a training-speed advantage, memory advantage, sample
quality advantage, promotion readiness, release readiness, or authority for
full-300K training.

Compact evidence artifact:

```text
artifacts/reports/generation/
quality_bridge_runtime_compute_fairness_waiter_deployment_2026-08-18.json
bytes: 6,260
SHA256: 871aa3273453eb0e43e6e4576825561ee0d5e2c7d714015674d4ec16e6b6697e
Git blob OID: f6e944f8a4bced749a360c5aff3e3cd5d1b9ed4f
```
