# Matched 10K class-fidelity diagnostic

This directory records a CPU-only, source-bound diagnostic over the already
completed matched 10K sampling-confirmation images. It does not resample,
train, alter a checkpoint, replace the frozen promotion gate, or authorize a
larger launch.

The exact evaluator checkout was revision
`7b0149632afd16e1e9122241451654ec0d9b60a7` on branch
`scale/generation-sample-diversity-diagnostic-v1`. Both raw reports were
independently replayed with `--resume`, which revalidated the complete PNG
window, immutable sampling manifest, sampling progress, sample-set SHA256,
checkpoint identity, classifier identity, evaluator runtime, and report
contents.

Key result: generated predictions cover many ImageNet classifier classes, but
agreement with the requested class is close to the 1000-class chance
reference. CoFiTok reaches top-1/top-5 `0.0018/0.0091`; dense identity reaches
`0.0011/0.0068`. Together with the separate exact-duplicate/dHash diagnostic,
this rules out literal sample-copy collapse but does not establish usable class
conditioning. Low image quality can itself depress classifier agreement, so
the diagnostic does not prove a single causal mechanism.

Files:

- `summary.json`: machine-readable matched interpretation and claim boundary.
- `metrics.csv`: compact metric table.
- `cofitok_class_fidelity_report.json`: exact remote raw report.
- `dense_identity_class_fidelity_report.json`: exact remote raw report.

