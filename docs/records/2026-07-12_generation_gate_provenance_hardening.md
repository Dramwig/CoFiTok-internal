# Generation gate provenance hardening (2026-07-12)

## Finding

The post-training remote deployer originally looked for `git.commit` in a
training report. `scripts/train_generation.py` records the authoritative field
as `git.revision`, so the completed 10% pair would have been rejected even when
both reports were valid. This was a transition-control bug, not a training bug;
the active queue and its checkpoint bytes were not modified.

## Correction

The remote deployer and the locked completion pipeline now validate
`git.revision` against the pinned 10% training revision
`781a01444fddbf0d48a427ba58bdeed50167b5be`. The completion pipeline therefore
also rejects a direct launch with reports from another revision.

The scaling and full generation gates additionally require:

- CoFiTok and dense reports from the same 40-character revision on
  `scale/generative-system`;
- completed `torch_fidelity_directory_metrics` reports using the same real
  image directory and real-image count;
- exactly the requested 10,000 or 50,000 generated images for each method;
- zero-based sample streams, exact requested sample counts, balanced modulo
  class scheduling, EMA weights, matching image shapes, and batch/resume
  invariant per-sample random streams;
- checkpoint and sample-set SHA256 bindings already required by the gate.

These checks keep the direct dense comparison genuinely matched and prevent a
larger, mismatched, or differently conditioned sample set from satisfying the
readiness decision.

## Verification

The focused transition and gate tests pass, including negative cases for a
mismatched training revision, a different real set, an off-by-one sample count,
and an unbalanced class schedule. The complete local test suite also passes.
The patched shell runbooks are syntax-checked on `pro6000` from `/tmp`; they are
not deployed into the active training worktree before both 10% runs complete.
