# Strong-baseline comparison and completion-audit revalidation

Date: 2026-08-15

## Outcome

The schema-v8 strong-baseline comparison, full-generation completion audit,
capacity-full post-evaluation/finalization supervisors, and stability terminal
audit were revalidated on Windows and CUDA-hidden Linux. All `119` bounded
tests passed in both environments; no source change was required.

This result proves the executable comparison and provenance failure boundaries.
It is not a generation-quality result, does not synthesize missing 300K/50K
evidence, and creates no training, evaluation, promotion, export, or release
authorization.

## Fair-comparison boundary covered

The suite verifies that the terminal report has exactly two comparison tiers:

- CoFiTok K8 and `dense_identity` are `matched_training_direct`, with matched
  dataset, resolution, shared backbone contract, optimizer schedule, effective
  batch, optimizer steps, training images, formal evaluator, and sampling
  protocol;
- D-AR, MAR, and ReTok are `official_pretrained_contextual`, retain their
  source aliases/protocols/eval-only role, and are never relabeled as matched
  training evidence.

Schema v8 permanently requires
`cross_tier_numeric_ranking_allowed=false`. It binds the tracked official table
by SHA256 and rejects changed aliases, metrics, sample counts, roles, protocols,
or source identities. It also keeps training wall-clock/throughput as raw
observations unless the bound pair monitor proves continuous uncontended GPU
coverage; equal wall-clock, GPU-hours, FLOPs, or generic compute-matched claims
remain false.

For stability/capacity full profiles, the matched direct rows must replay both
training reports, both formal 50K metrics reports, the final gate, terminal pair
monitor, class-fidelity qualification when required, real-set/evaluator/runtime
identity, exact sampling protocol, checkpoints and sample-set hashes. Displayed
metrics are recalculated from those bound sources.

## Terminal completion boundary covered

The completion tests exercise success and negative cases for:

- exact 300K matched training, images seen, protected 50K/100K/200K/300K
  checkpoints, integrity sidecars, `latest.json`, runtime environment, dataset,
  config/pair contract, and operational monitor evidence;
- formal paired 50K sample counts, immutable manifests/progress/reports,
  checkpoint and sample-set SHA256, shared batch selection, per-index streams,
  real-set tree/cache, evaluator identity, FID/IS/precision/recall, and visual
  endpoint/prefix audit;
- non-weakenable final-gate thresholds, class fidelity, mechanism diagnostics,
  and matched CoFiTok/dense provenance;
- physical EMA artifacts, sidecars, export manifests, preflights, smoke PNG
  resume chains, final release authorization, completion audit, and release
  receipt ordering;
- capacity-full `pass` versus scientific `hold`: a held final gate cannot enter
  export/finalization, while a passing gate still requires a complete audit
  before release.

Tampering or omission in any of those sources is expected to fail the named
check rather than producing a partially trusted comparison.

## Exact test evidence

The four modules and collected counts were:

```text
tests/test_large_scale_generation_comparison.py              16
tests/test_large_scale_generation_completion_audit.py        82
tests/test_generation_capacity_full_posteval.py               10
tests/test_generation_stability_completion_audit.py           11
```

Local Windows result: `119 passed` in 20.8 seconds.

The Linux run used:

- checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-rehearsal-47e59a0/CoFiTok-internal`;
- revision: `47e59a0ce926c3ac1fdaa4860de84d2f27e39f76`;
- tree: `3a99993b66db675f262ed2e04298e1d4d64411cf`;
- tracked checkout state: clean;
- Python: `3.10.20`;
- CUDA policy: `CUDA_VISIBLE_DEVICES=''`;
- result: `119 passed`, zero failures/errors/skips;
- JUnit time: `44.897` seconds.

The persistent JUnit report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/strong_baseline_completion_revalidation_2026-08-15/pytest_linux_cuda_hidden_47e59a0.xml`;
- bytes: `21,864`;
- SHA256:
  `e29e24988119d3f18c9b193252f0ece44d038105a312159e6e7dc997526a2101`;
- file mode: `0644`.

## Preserved production state

The revalidation used only CPU fixtures. No production checkpoint, sample, gate,
comparison, or release artifact was written. The formal checkout remained
untouched. At the pre-record live check, the quality bridge was still waiting
for the unrelated FieldScope GPU process to exit.
