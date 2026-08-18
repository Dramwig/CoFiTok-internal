# Sample-quality gate revalidation

Date: 2026-08-15

## Outcome

The full-data quality-bridge result, follow-up decision, class-fidelity,
visual-audit, requested-class waiter, and formal generation-gate boundaries were
revalidated on Windows and CUDA-hidden Linux. The cross-platform subset passed
`103/103` locally; Linux additionally passed all three server-only waiter tests,
for `106/106` with no failures, errors, or skips. No source change was required.

This proves how incomplete, weak, mismatched, or tampered quality evidence is
rejected. It is not evidence that the pending bridge generated good samples,
does not replace the physical 10K/50K sample sets, and creates no training,
capacity, full-300K, promotion, export, or release authorization.

## Quality evidence covered

The quality-bridge and gate tests require more than a single FID number:

- exact matched CoFiTok/dense training revisions, configs, full ImageNet-256
  dataset identity, steps/images, checkpoints, integrity sidecars, runtime
  environment, and pair contract;
- source-bound 50K/100K milestones and exact terminal 10K-sample EMA DDIM-100
  manifests, progress, PNG counts, checkpoint/sample-set SHA256, shared real-set
  tree/evaluator/cache identity, FID, Inception Score, precision, and recall;
- matched endpoint error, requested prefix-order rank, coarse-token energy,
  tail utilization, exact zero-token behavior, shuffled-token mismatch, and
  checkpoint-mechanism provenance;
- fixed-classifier calibration on real validation data plus generated class
  top-1/top-5, mean target probability, target NLL, predicted-class coverage,
  normalized entropy, sample count, checkpoint SHA, and sample-set SHA;
- deterministic endpoint and prefix panels with source identities, requested
  classes/indices, uniqueness checks, and an explicitly non-quantitative role.

Each gate row must bind the same checkpoint and sample set as its source metric
or diagnostic. Missing/failed thresholds produce `hold`; the builder cannot
silently weaken thresholds or relabel a partial report as success.

## Follow-up authorization boundary

The follow-up-decision tests verify that bridge output is non-authorizing by
itself. A source-bound result can select only the declared next diagnostic or
hold path. It cannot directly authorize 50K/100K continuation, full training,
300K, promotion, export, or release. Later capacity stages still require their
own immutable decision, execution revision, output root, storage/runtime checks,
and standing authorization.

The Linux-only requested-class waiter uses `fcntl` locking, waits for the exact
terminal quality result and sampling reports, and launches only the bounded
non-quantitative panel builder. Its three tests passed on Linux. Windows cannot
import that server-only entrypoint because `fcntl` is unavailable; this platform
collection boundary was kept explicit rather than hidden by changing the code.

## Exact test evidence

Cross-platform modules and counts:

```text
tests/test_generation_class_fidelity.py                    18
tests/test_generation_class_fidelity_calibration.py         4
tests/test_generation_gate_report.py                       48
tests/test_generation_quality_bridge.py                    15
tests/test_generation_quality_bridge_followup_decision.py  13
tests/test_generation_requested_class_visual_audit.py       3
tests/test_generation_visual_audit.py                       2
```

Local Windows result: `103 passed` in 48.6 seconds.

Linux additionally ran:

```text
tests/test_generation_requested_class_visual_audit_waiter.py  3
```

The exact Linux environment was:

- checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-rehearsal-47e59a0/CoFiTok-internal`;
- revision: `47e59a0ce926c3ac1fdaa4860de84d2f27e39f76`;
- tree: `3a99993b66db675f262ed2e04298e1d4d64411cf`;
- tracked checkout state: clean;
- Python: `3.10.20`;
- CUDA policy: `CUDA_VISIBLE_DEVICES=''`;
- result: `106 passed`, zero failures/errors/skips;
- JUnit time: `167.987` seconds.

The persistent JUnit report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/sample_quality_gate_revalidation_2026-08-15/pytest_linux_cuda_hidden_47e59a0.xml`;
- bytes: `18,923`;
- SHA256:
  `fd9c20ed002b4b597d18b7625df59018ab32bdaab5c57d61f5910f964c0adba8`;
- file mode: `0644`.

## Preserved production state

The revalidation used temporary CPU-only reports and tiny fixtures. It did not
read model weights or modify production checkpoints, samples, metrics, gates,
decisions, or visual panels. The formal checkout remained untouched. At the
pre-record check, the quality bridge still waited for the unrelated FieldScope
GPU process to exit.
