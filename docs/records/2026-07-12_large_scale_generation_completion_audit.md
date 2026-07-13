# Large-scale generation completion audit (2026-07-12)

## Purpose

A successful shell process or a single passing gate is not sufficient evidence
that the large-scale generation objective is complete. The final system now has
an independent, fail-closed completion audit:

```text
scripts/audit_large_scale_generation_completion.py
```

It is read-only with respect to experiment assets and writes one atomic JSON
report. Missing evidence yields `in_progress`; contradictory or invalid evidence
yields `failed`; only a fully verified evidence chain yields `complete`.

## Required evidence

The audit requires all of the following:

- matched 10% ImageNet-256 CoFiTok/dense 50K training at pinned revision
  `781a01444fddbf0d48a427ba58bdeed50167b5be`;
- a controlled fast-forward transition to the exact upgrade revision, bound to
  the deployment bundle and validated 10% pair by SHA256 receipt;
- passing storage-capacity preflights for 10K post-evaluation, full 300K
  training, and formal 50K post-evaluation, all bound to the clean deployed
  revision and the generation filesystem with non-weakened reserves;
- a clean-revision full-training operational-monitor pass with finite monotonic
  metrics, exact 300K completion for both methods, and stat evidence for all
  protected milestone checkpoints;
- a passing 10K scaling gate authorizing full ImageNet-256;
- matched full ImageNet-256 CoFiTok/dense 300K training at the deployed upgrade
  revision;
- matching canonical runtime-environment fingerprints in both full training
  reports and latest checkpoint pointers, with required Python, package,
  PyTorch/CUDA/cuDNN, GPU, and project-lock provenance;
- latest full checkpoint pointers bound to the same clean deployed Git revision,
  branch, and tracked state as their training reports;
- a revision-bound shared runtime selection whose microbatch and accumulation
  are present in both 300K training reports and preserve effective batch 64;
- complete training audits with validation events and protected 50K, 100K,
  200K, and 300K checkpoints;
- all four paired 2,048-sample DDIM-50 milestone reports;
- paired formal 50K DDIM-250 EMA generation with completed atomic sampling
  progress, checkpoint/sample SHA256, integrity sidecars, and positive sampling
  elapsed time;
- a shared, revision/checkpoint-bound formal sampling batch selection that both
  50K sample reports actually use with batch-invariant random streams;
- deterministic fixed-index CoFiTok/dense endpoint panels and a separate
  `1/2/4/8` prefix-path panel bound to the formal checkpoint/sample-set hashes;
- verified, smaller EMA-only artifacts for both methods, each bound to the source
  300K checkpoint and proven by real-forward preflight plus short DDIM PNG smoke;
- a passing final gate with decision `large_scale_generation_ready`;
- the ready two-tier comparison containing the matched pair and three official
  contextual methods, with direct compute fields recomputed from training and
  sampling reports and official rows bound to the locked source table SHA256.

Milestone quality alerts are preserved as warnings because those 2,048-sample
runs are trend diagnostics. They cannot override a failed final 50K gate.

The final-gate check is structural and provenance-bound. It requires the
absolute/relative FID, metric-range, precision/recall, endpoint, ordered-prefix,
restricted-synthesis, shuffle, and training-checkpoint-integrity gates. Its
thresholds may be stricter but not weaker than FID 20.0, precision/recall 0.30,
and 0.05 relative regression limits. Reported FID/precision/recall must exactly
match the formal 50K generation reports bound by checkpoint and sample-set SHA.

## Pipeline integration

`generation_complete_pipeline_after_10pct.sh` runs the audit after the final
gate and before publishing pipeline status `pass`. A missing or invalid artifact
therefore prevents a false completion state even when preceding commands happen
to exit successfully.

For a non-terminal status snapshot, use `--allow-incomplete`. The strict mode
used by the completion pipeline exits nonzero unless `complete=true`.
