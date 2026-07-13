# DDIM closed-form correctness

Date: 2026-07-13

Branch: `scale/generative-system`

The formal generation path already tested fixed protocol fields, CFG batching,
per-index random streams, exact resume, PNG integrity, and report provenance.
Those checks did not independently prove the numerical DDIM recurrence.

`tests/test_generation_sampling.py` now includes an oracle epsilon predictor
defined from a fixed known `x0` and the active `DiffusionSchedule`. Starting from
random Gaussian noise, the test traverses a strided cosine schedule and requires
the final sample to reconstruct the known image. It runs once with deterministic
DDIM (`eta=0`) and once with stochastic DDIM (`eta=0.5`), so both the base update
and variance branch are covered.

Verification on 2026-07-13:

```text
targeted generation sampling tests: 12 passed
full local suite: 477 passed
```

This is a sampler-correctness invariant only. It does not substitute for the
10K promotion samples, full 300K matched training, or final 50K distribution
metrics required by the readiness gate.
