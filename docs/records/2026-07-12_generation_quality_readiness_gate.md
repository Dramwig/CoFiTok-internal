# Generation quality readiness gate

Date: 2026-07-12

Branch: `scale/generative-system`

Process completion and relative parity with `dense_identity` do not by
themselves prove a usable generator. The full ImageNet-256 gate therefore
combines an absolute quality ceiling, distribution-support floors, matched
regression limits, and the original CoFiTok mechanism contract.

The formal gate requires:

- exactly 50,000 EMA samples per method under one DDIM-250/CFG/evaluator protocol;
- matched clean sampler and evaluator code provenance bound to the full training
  revision and `scale/generative-system` branch;
- finite metrics with FID >= 0, IS mean > 0, IS standard deviation >= 0, and
  precision/recall in `[0, 1]`;
- CoFiTok FID <= 20.0 and no more than 5% above matched dense FID;
- CoFiTok precision and recall each >= 0.30 and no more than 0.05 below matched
  dense;
- endpoint clean MSE no more than 5% above dense;
- ordered-prefix rank 1, exact zero-token synthesis, and shuffle mismatch;
- verified final checkpoint integrity and checkpoint/sample-set provenance.

The 0.30 precision/recall values are conservative non-collapse readiness floors,
not claims of state-of-the-art quality. The final completion audit requires all
named checks, rejects weaker thresholds, and cross-checks FID/precision/recall
against the formal generation reports. A hand-edited or stale passing gate
cannot establish readiness. The named sampling/evaluator code provenance gates
are mandatory; omitting either is a completion-audit failure.
