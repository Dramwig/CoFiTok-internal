# Matched frozen-checkpoint class-conditioning sensitivity diagnostic

Date: 2026-08-18

## Purpose and boundary

This diagnostic tests whether the frozen matched ImageNet-256 10% CoFiTok and
`dense_identity` checkpoints internally respond to the requested class label.
It complements generated-sample class-fidelity evaluation by holding the clean
image, forward noise, and timestep fixed while changing only the class label.

The work is permanently diagnostic-only. It did not generate samples, use the
GPU, alter a trainer/controller/monitor, authorize a later stage, or replace a
formal FID, support, or class-fidelity gate.

## Reproducible implementation

The diagnostic was implemented in an isolated branch based on the active
quality-bridge training revision:

```text
branch: analysis/generation-conditioning-sensitivity-diagnostic-v1
base: cf0e5faa94bf4ab38d947b921935b3b765b5537a
evaluator commit: a5c647885ee455cbd63e093db36747625fc7e0f1
comparison commit: 2aef72c0edf6e464b391392ed213a1f95be2f254
comparison tree: a2833fbc8674e0aa83cd82007d2df4cebd9ce11a
remote isolated checkout: /tmp/cofitok-conditioning-sensitivity-a5c6478
```

Implemented entry points:

```text
scripts/evaluate_generation_conditioning_sensitivity.py
scripts/build_generation_conditioning_sensitivity_comparison.py
```

The evaluator verifies the adjacent checkpoint integrity sidecar before
deserialization, reconstructs exact seeded initialization, binds the label map
and every input image by bytes/SHA256, writes an immutable manifest, and
atomically publishes a source-bound report. The matched builder reopens and
validates both source manifests/reports, requires identical images, labels,
noise seeds, timesteps, evaluator Git, and weight selection, then builds a
paired comparison.

Both entry points are CPU-only in this execution. Commands used:

```text
CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
nice -n 19
ionice -c 3
```

## Source checkpoints

Both checkpoints are the frozen 50K EMA endpoints from the same matched
stability pair:

| method | parameters | checkpoint SHA256 |
|---|---:|---|
| CoFiTok K8 | 62,834,083 | `ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a` |
| dense identity | 62,824,707 | `325da25f9fd228ab224abd977da7f9e1d000ba36e3e8136b55edae9c9f47c716` |

The parameter gap remains `+0.014924%` for CoFiTok.

## Protocol

- Dataset: `imagenet_256_10pct` validation split.
- Images: the first deterministic validation image for labels 0 through 7.
- Correct label: the image's canonical ImageNet label.
- Wrong label: `(correct + 500) mod 1000`.
- Null label: classifier-free null index 1000.
- Timesteps: `100`, `500`, and `900`.
- Noise seeds: `102030 + sample_index`; the same image/noise is reused across
  correct, wrong, and null conditions and across both methods.
- Weights: EMA.
- Total paired rows per method: `8 images x 3 timesteps = 24`.

The earlier one-image CoFiTok `/tmp` diagnostic was first replayed with the new
tool. It reproduced every reported conditioning and parameter-update value,
including the `4.3114%` EMA class-embedding relative update, before the matched
eight-image run was executed.

## Internal response results

Mean epsilon-output difference relative to the correct-label epsilon RMS:

| method | timestep | correct vs wrong | correct vs null | correct better than wrong | correct better than null |
|---|---:|---:|---:|---:|---:|
| CoFiTok K8 | 100 | 1.4011% | 1.1172% | 4/8 | 0/8 |
| CoFiTok K8 | 500 | 0.6577% | 0.4687% | 5/8 | 4/8 |
| CoFiTok K8 | 900 | 0.3117% | 0.2353% | 4/8 | 3/8 |
| dense identity | 100 | 1.4095% | 1.0705% | 5/8 | 2/8 |
| dense identity | 500 | 0.5837% | 0.4265% | 5/8 | 1/8 |
| dense identity | 900 | 0.2944% | 0.2374% | 3/8 | 2/8 |

Across all 24 paired rows:

| method | mean correct-vs-wrong delta | mean correct-vs-null delta | mean correct MSE advantage vs wrong | mean correct MSE advantage vs null |
|---|---:|---:|---:|---:|
| CoFiTok K8 | 0.7902% | 0.6071% | +0.0310% | -0.1940% |
| dense identity | 0.7625% | 0.5781% | -0.0734% | -0.2562% |

Both methods select the correct label over the wrong label on exactly `13/24`
rows. The exact one-sided sign-test p-value is `0.4194098711` for each method.
Correct-over-null counts are `7/24` for CoFiTok and `5/24` for dense, with
one-sided p-values `0.9886720777` and `0.9992280602`. Thus neither model shows a
consistent correct-label advantage in this internal diagnostic.

The matched CoFiTok-minus-dense mean differences are small:

```text
correct-vs-wrong relative delta:       +0.02763 percentage points
correct-vs-null relative delta:        +0.02896 percentage points
correct-vs-wrong MSE improvement:      +0.10441 percentage points
correct-vs-null MSE improvement:       +0.06215 percentage points
```

These small differences do not establish a method-specific conditioning
advantage.

## Exact-initialization parameter audit

EMA relative update from exact `seed=2027` initialization:

| parameter | CoFiTok K8 | dense identity |
|---|---:|---:|
| class embedding | 3.4907% | 3.1391% |
| first conditioning projection | 49.9216% | 54.4585% |
| first conditioned-block convolution | 88.3580% | 88.5311% |
| input projection | 14.2084% | 14.7429% |

The shared trunk and conditioning projection learned substantially, while the
class embedding moved only about 3% from initialization. Combined with the
near-random generated-sample Top-1/Top-5 results, this supports a shared failure
to use class identity rather than a CoFiTok-specific restricted-synthesis
collapse.

## Evidence identities

Remote authoritative directory:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/conditioning_sensitivity_a5c6478
```

Local evidence copy:

```text
artifacts/reports/generation/stability_scaling_50k_ema_teacher/conditioning_sensitivity_2026-08-18/
```

| artifact | SHA256 |
|---|---|
| CoFiTok manifest | `c97a9e12c1a65fb20b615ac48cf9f6687d43b792873613476f19c07390d9bbfd` |
| CoFiTok report | `2aa880fbf00b49a0d62ec624292c39b30296307794a6b69599aa7507bfce504f` |
| dense manifest | `b8f4857744470e6ae665aa73c626b98356e0064244809e6f50390be709ffd18b` |
| dense report | `d26bed83596bbb09e9951c97402fc9882a076f770c511b24f061bd8a2a4dec27` |
| matched comparison | `85509ae19cdf4ec7605ff04380a89bd0f8e51ae23d6383cff816501230fdb68b` |

Deployment bundles:

```text
a5c6478 bundle: 10,430 bytes
SHA256: 3aa0390a6d7115f71fe22af71887e6cbdab4ceb492f35e21798d2c3e5567e4cd

2aef72c follow-up bundle: 6,878 bytes
SHA256: e05df7e0708cc18589f8387fd89be9347f231145982bb00ec31352b75d52add9
```

## Validation

- Local syntax compilation: pass.
- Linux targeted evaluator/checkpoint/class-fidelity suite: `35 passed`.
- Linux `compileall`: pass.
- Linux full code suite excluding the known outer-sibling paper-layout test:
  `1,078 collected`, `1,076 passed`, `2 skipped`, `0 failed`.
- The unexcluded run had only the four known failures in
  `tests/test_aaai27_experiment_structure.py`, because an isolated `/tmp`
  checkout does not contain `/tmp/paper/venues/aaai27/main.tex` and
  `/tmp/paper/latex/main.tex`; no implementation test failed.

The formal remote checkout was not moved. No trainer, controller, monitor,
waiter, or GPU process was signaled, paused, restarted, or replaced.

## Interpretation and next decision

This result strengthens the current diagnosis:

1. CoFiTok's ordered restricted synthesis remains mechanistically valid.
2. The current generation-quality failure is not explained by a uniquely weak
   CoFiTok class path; the matched dense model exhibits the same internal class
   insensitivity.
3. Better FID for frozen CoFiTok does not imply requested-class alignment.
4. Any future recipe repair should target the shared predictor conditioning
   path and must be applied identically to CoFiTok and dense where it is a
   shared stabilization/conditioning mechanism. Class or time information must
   still never enter `S_k`.

The diagnostic sample count is intentionally small and the 24 rows are
correlated within eight images. It therefore supports root-cause diagnosis but
does not replace the active full-data matched 50K/100K sample-quality and class-
fidelity chain. The active quality bridge remains authoritative.
