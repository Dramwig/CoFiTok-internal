# Stability 50K training-trajectory snapshot

This is a read-only, source-bound scientific snapshot of the active stability
pair. It records the completed CoFiTok trajectory through step 50,000 and the
immutable prefix of the still-running dense trajectory through step 18,000.

## Result

Across the 18 matched scheduled-validation events from 1K through 18K, the
CoFiTok/dense epsilon-MSE ratio has mean `0.996941`, median `0.999905`, and
range `[0.954921, 1.011452]`. The largest observed CoFiTok regression is only
`+1.1452%`; the largest improvement is `-4.5079%`. Both metric streams are
finite and strictly increasing in step, with `samples_seen = step * 64`.

This is evidence that the factorized and dense predictors remain closely
matched in one-step validation space over the common 18K prefix. It is not
evidence that the CoFiTok free rollout is usable. The failed 2026-07-29 formal
v3 run already demonstrated that near-matched endpoint MSE can coexist with a
large FID regression, so this audit deliberately does not infer sample quality
from the training curve.

The completed CoFiTok run contains no training-period PNG/JPEG/WebP outputs.
The existing frozen post-evaluation runbook is therefore still the first
authoritative source of actual current-recipe images: it will generate matched
10K EMA DDIM-100 sample sets, a 64-image prefix diagnostic, a visual audit,
1,024-image checkpoint evaluations, and FID/IS/precision/recall after dense
reaches an exact healthy 50K. The exact frozen runbook is SHA256
`6cf80146c1696aa0bed9a4a370b41c42aa5be9dc03f753c4c81ce6b54278b064` at
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289`.

That frozen schema-v2 runbook does **not** build the newer EMA free-rollout
stability qualification. The latter remains a separate source-bound
supplemental check from a clean checkout after frozen post-evaluation; it must
not be represented as evidence the current waiter already produces.

## Interpretation boundary

- The lower validation means after the 30K EMA-teacher boundary do not identify
  a causal teacher benefit: validation batches differ by event and LR/overall
  training progress change at the same time.
- Auxiliary total loss is not compared between CoFiTok and dense because the
  factorized model has additional path/energy/high-frequency objectives.
- No GPU work, sampling, process signal, checkpoint load, or remote write was
  performed for this snapshot.
- `full_training_launch_allowed=false` remains fixed. Only the formal matched
  samples plus the separate EMA rollout qualification can decide whether this
  stability recipe solved the earlier high-frequency rollout failure.
