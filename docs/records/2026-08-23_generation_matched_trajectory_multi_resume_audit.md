# 2026-08-23 matched training trajectory multi-resume audit

## Scope

This change is CPU-only, diagnostic, and permanently non-authorizing. It does not
start training or sampling, signal any process, modify the active quality-bridge
checkout, authorize promotion/release, or permit full 300K training.

The active matched ImageNet-256 quality bridge had accumulated canonical metrics
across more than one exact resume:

- CoFiTok reconciliation reports at steps 20,000 and 80,000;
- a planned CoFiTok protected-milestone continuation at step 50,000;
- a dense reconciliation report at step 50,000.

The previous trajectory builder only consumed the latest reconciliation stored in
`run_manifest.json`. After the 80K restart replaced the manifest's earlier 20K
binding, a source-bound 78K prefix correctly failed closed because steps 20,001 and
50,001 could no longer both be explained.

## Implementation

Branch:

```text
analysis/generation-matched-trajectory-multi-resume-v1-20260823
```

Code revision and tree:

```text
48e2a6f00f544ba48412dd4e45426e095c06c411
f4e12d2ea84eae2f0d340dd9086e2798c6f7a929
```

The trajectory builder now:

- accepts ordered, repeatable physical reconciliation reports for each method;
- verifies every report's metrics path and physically hashes each orphan archive;
- requires the explicit history to end at the current manifest resume;
- validates strict increasing reconciliation steps and retained-row counts at every
  resume boundary;
- permits an observed first-post-resume row without a reconciliation report only at
  a source-bound protected checkpoint step from the matched resolved configs;
- continues to reject every other irregular logging step;
- records per-boundary evidence and binds all reconciliation/orphan sources in the
  report;
- keeps quality, promotion, release, full-training, and 300K claims disabled.

Builder source identity:

```text
bytes: 41019
sha256: 1360a2e9f0f9f19d920c4df839433e967b198e4651a942b9047d897530dad911
```

## Validation

Local Windows CPU-only selected tests:

```text
40 passed
```

Linux CPU-only selected tests in the exact committed checkout:

```text
40 passed
CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Remote isolated checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/matched-trajectory-multi-resume-48e2a6f/CoFiTok-internal
```

Incremental bundle:

```text
D:/cofitok-bundles/matched-trajectory-multi-resume-48e2a6f-v2.bundle
bytes: 7468
sha256: 0bf42c37b4f15a87b50b03f0738b9bac73b1d4de26a70e6a563841b08a0308d8
prerequisite: 40ee55202545ea01e48fb6642d1b5a51b5264663
advertised head: 48e2a6f00f544ba48412dd4e45426e095c06c411
```

## Real 78K replay

Authoritative report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/matched_training_trajectory_step_00078000_multi_resume_v1/trajectory_report.json
bytes: 81504
sha256: 2997d14cf9b78829dd81f0bb70521c6258d78ef01b58365b70a2aec91911cdb5
```

Small local evidence copy:

```text
artifacts/reports/generation/matched_training_trajectory_78k_multi_resume_2026-08-23.json
```

The exact command was run twice against frozen 78K prefix snapshots and produced the
same report SHA256 both times.

Prefix snapshots:

```text
CoFiTok sha256: b13d1e0b4a1413771caa247a2fc0327b7bd3b10e2694a31ceaf612ef4934aebb
dense sha256:   63eec1ebba3d708dfaf3eedfc9295a1acaf70b15bb1850bc50c66d2f5dd1c21b
```

Resume evidence replayed by the report:

- CoFiTok 20K: physical reconciliation plus orphan archive;
- CoFiTok 50K: observed step 50,001 bound to matched protected checkpoint 50K;
- CoFiTok 80K: physical reconciliation plus orphan archive, retained-row count 1,603;
- dense 50K: physical reconciliation plus orphan archive.

The 78 paired fixed-validation events remain extremely close:

```text
CoFiTok mean epsilon MSE: 0.028925960071575947
dense mean epsilon MSE:   0.02891648121369191
relative delta of means:  +0.0003278012222161441
CoFiTok lower events:     37
dense lower events:       41
endpoint relative delta:  +0.0013549201849135234
```

This supports continued matched optimization through 78K and does not show a
CoFiTok-specific training divergence. It does not establish sample-quality or broad
generation advantage. `generation_advantage_proven` remains false until matched
terminal DDIM-100 quality/class-fidelity evidence and all claim guards pass.

## Failed-attempt cleanup

Two partial 78K snapshot directories created by the obsolete single-resume builder
were verified to contain no report, only reproducible prefix copies, and were removed
before publishing the successful multi-resume report. No checkpoint, training log,
sample, or user-authored artifact was removed.
