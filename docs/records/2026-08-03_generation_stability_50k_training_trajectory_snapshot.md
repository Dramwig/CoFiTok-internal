# Stability 50K matched training-trajectory snapshot

Date: 2026-08-03

## Purpose

The active stability experiment has a completed 50K CoFiTok member and a
still-running dense member. The most direct available evidence before formal
sampling is the matched scheduled-validation trajectory. This record keeps
that evidence useful without presenting it as generation quality.

The source-bound audit is in
`artifacts/reports/generation/stability_scaling_50k_ema_teacher/training_trajectory_snapshot_2026-08-03/`.
It binds the complete 1,002-row CoFiTok JSONL and the immutable dense JSONL
prefix through step 18,000 by path, byte count, and SHA256. It also binds the
CoFiTok terminal training report, `latest.json`, integrity sidecar, checkpoint
bytes/SHA, dataset identity, runtime identity, and immutable training Git
identity.

## Finding

The 18 common validation events have mean CoFiTok/dense MSE ratio `0.996941`,
median `0.999905`, and range `[0.954921, 1.011452]`. Thus no one-step validation
regression larger than `1.15%` is present through the current common horizon.
Both metric streams are finite, monotonic in step, and exact in images seen.

This narrows the remaining uncertainty rather than resolving it: the current
factorization is not visibly losing endpoint-prediction capacity at matched
scale, but free-rollout sample quality is still completely unmeasured for the
50K recipe. Training directories contain zero images. The formal frozen
post-evaluation remains responsible for matched EMA DDIM-100 samples, prefix
visuals, checkpoint evaluation, and FID/IS/precision/recall. Its exact frozen
runbook is SHA256
`6cf80146c1696aa0bed9a4a370b41c42aa5be9dc03f753c4c81ce6b54278b064`
at `c1efb12c6640f2d2d62ac7e9982c8804d96e7289`.

The frozen schema-v2 path does not include the newer EMA free-rollout
qualification. That remains a separate supplemental requirement after frozen
post-evaluation and cannot be inferred from the one-step trajectory here.

No process was signaled, restarted, or modified. No GPU task or remote write
was performed. The evidence cannot authorize full 300K.
