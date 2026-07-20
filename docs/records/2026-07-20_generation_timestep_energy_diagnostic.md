# Generation timestep-energy diagnostic

Date: 2026-07-20

Role: post-hoc mechanism diagnosis for the rejected equal-progress v3 probe.
No new model was trained and this report cannot authorize scaling.

## Provenance

- Training revision: `a37b2e410aa558e831688009b13012a6a7da616b`
- Evaluator revision: `f1ffbfb298b9abb1a3a4c8b524c976610908d2bd`
- Checkpoint SHA256:
  `b03e3bee66dfb368330f056df3afc2bac451ed2a52a8c90ddae292825d1ef83c`
- Dataset: `imagenet_256_10pct` validation split
- Weights: EMA
- Protocol: 256 images per timestep, ordered/reverse plus 16 fixed random
  orders, bf16
- Timesteps: `50/250/500/750/950`

The evaluator reports both the learned component-energy distribution and the
theoretical component energies implied by the exact denoise-path targets used
by training.

## Result

| timestep | ordered rank / 18 | ordered path AUC | target uniform MSE | learned uniform MSE | learned tail-2 energy |
|---:|---:|---:|---:|---:|---:|
| 50 | 1 | 0.009512 | 0.006899 | 0.012046 | 49.3% |
| 250 | 14 | 0.056711 | 0.002242 | 0.038651 | 88.7% |
| 500 | 17 | 0.285558 | 0.000275 | 0.045466 | 94.6% |
| 750 | 18 | 1.616672 | 0.000012 | 0.046882 | 95.9% |
| 950 | 18 | 45.838748 | 0.000000 | 0.047120 | 96.0% |

The theoretical denoise-path target becomes almost exactly uniform across the
eight component energies as noise increases. The learned model does the
opposite: it moves nearly all work into tokens 7 and 8. The learned-versus-
target energy-ratio L1 distance rises from `0.5611` at timestep 50 to `1.4206`
at timestep 950.

Ordering follows the same failure curve. The learned order is rank 1 only at
the low-noise timestep 50, then falls to ranks 14, 17, 18, and 18. At all four
higher-noise points, the same fixed random order `random_12` has the best path
AUC. This is systematic timestep-dependent token-role collapse, not random
ranking noise.

## Decision

A static batch-aggregated uniform energy target is insufficient. A fixed
per-sample uniform target would close most of the gap, but it is still an
unnecessary approximation at low noise, where the exact path target is mildly
tail-heavy.

The next probe will therefore add a denoise-path energy-ratio term that, for
each sample independently:

1. constructs the same theoretical prefix-epsilon targets used by the path
   prefix/component losses;
2. differences adjacent prefixes into target components;
3. normalizes predicted and target component energies across tokens;
4. minimizes their ratio MSE.

This target is timestep-aware without feeding timestep or condition into the
restricted synthesis operator. It closes the mixed-timestep batch loophole and
does not train against shuffled permutations, preserving order diagnostics as
an independent test.

The five machine-readable reports are mirrored under the untracked local
archive:

```text
artifacts/reports/generation/rank_recovery_probe_2026-07-20_v3/
  equal_progress/timestep_diagnostic/
```
