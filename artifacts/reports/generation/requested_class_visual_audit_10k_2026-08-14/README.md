# Requested-class visual audit (2026-08-14)

This evidence pack records a deterministic qualitative audit of the existing
matched 10K confirmation samples. The first 16 contiguous global indices were
chosen before visual inspection; no sample was selected by appearance or
metric.

Each panel has three rows:

1. the lexicographically first real ImageNet validation image for the requested
   class;
2. the CoFiTok K8 sample;
3. the dense-identity sample.

The two generated rows use the same global-index random stream. Panel 00 covers
indices `10000..10007` (tench through cock); panel 01 covers `10008..10015`
(hen through robin). The source report binds every real/generated PNG, both
sampling reports, checkpoints, sample-set digests, class mapping, classifier
calibration, execution Git identity, and panel PNG.

This is a qualitative, non-authorizing diagnostic. It does not replace
class-fidelity, FID/IS/precision/recall, the frozen gate, or a formal terminal
evaluation.
