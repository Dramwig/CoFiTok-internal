# Exploratory class-fidelity diagnostics on frozen generation samples

Date: 2026-08-18

## Purpose and scope

This record diagnoses whether the existing ImageNet-256 samples follow their
requested class labels. It does not create new samples, load a training
checkpoint, use the GPU, modify a trainer, or replace any formal milestone or
terminal quality gate.

All three evaluations used the fixed torchvision ImageNet-1K ResNet-50 V2
classifier through `scripts/evaluate_generation_class_fidelity.py` with
`--cpu`, `batch-size=8`, `num-workers=2`, and low-priority CPU/IO scheduling.
The evaluator was executed from:

```text
checkout: /tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal
branch: scale/generation-stability-quality-bridge-100k
revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
tracked dirty: false
```

Classifier identity:

```text
name: torchvision_resnet50_imagenet1k_v2
path: /root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/resnet50-11ad3fa6.pth
bytes: 102540417
sha256: 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
```

Each source sampling report binds `class_schedule=balanced_modulo`, so the
requested class for zero-based sample index `i` is `i mod 1000`. The evaluator
revalidated every sampling report, manifest, image count, checkpoint SHA256,
and sample-set SHA256 before scoring.

## Label-order and real-set calibration

Near-chance generated-sample accuracy would be uninterpretable if the dataset's
training label IDs differed from the torchvision classifier's canonical
ImageNet IDs. That alternative explanation was checked directly.

The full-data export contains both:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_256/metadata/label_to_wnid.json
/root/autodl-tmp/CoFiTok/datasets/imagenet_256/metadata/imagenet_label_to_wnid_train_order.json
```

After unwrapping their `label_to_wnid` objects, both mappings contain 1,000
entries and are exactly equal at every index. They begin with
`n01440764, n01443537, n01484850, n01491361, n01494475` and end with
`n13044778, n13052670, n13054560, n13133613, n15075141`, matching the canonical
ImageNet training order. `PreparedImageDataset` independently obtains the same
indices by lexicographically sorting the manifest WNIDs.

The same classifier was then evaluated on 2,048 real validation images selected
as two images per WNID plus a third image for labels 0--47. This exactly matches
the generated 2,048-sample requested-label count range of two to three images
per class. The calibration rehashed the 405,484,553-byte dataset manifest and
verified its authoritative SHA256
`9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`.

Calibration report:

```text
/tmp/imagenet256-real-resnet50-class-calibration-2048-cpu-20260818.json
bytes: 2151
SHA256: e86fd47a4cf033b986849c9e082e0dd118d55f78e7c71835704485cc825899f8
sample-path digest SHA256: f0acefa33259e859d498b80b4ab6606ed6f9008bd59bd98956a1aa57b6268eb4
```

Real-set calibration metrics:

| sample count | Top-1 | Top-5 | mean target probability | predicted-class fraction | normalized predicted entropy |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 78.3691% | 94.8730% | 34.0489% | 95.4% | 0.9812 |

This calibration rules out label-order mismatch and classifier/preprocessing
failure as explanations for the generated-sample results below.

## Source identities

### Full-data CoFiTok 50K trend sample

```text
dataset: imagenet_256
checkpoint step: 50000
checkpoint SHA256: d7100a6e8f67adb2b1630d36c17d0f2bc94fb93ed99d3e270a0867979c3343b4
sampling: EMA, bf16, DDIM-50, CFG 1.5, guidance rescale 0.0
sample count: 2048
sample-set SHA256: 2d1efc1c15d057ee6c916f9685145d324b9c4deb08f28a27eaea31401f5dfe61
generation FID: 221.05073384081678
generation IS: 4.484192228313476
```

Exploratory report:

```text
/tmp/cofitok-full-data-50k-class-fidelity-2048-cpu-20260818/class_fidelity_report.json
bytes: 12722
SHA256: 3e2fec1f09b00488e50c3bbd276806f396fd703f4d73a19607c76f6c1ea2356b
```

### Frozen 10% matched 50K pair

Both sources use `imagenet_256_10pct`, EMA, bf16, DDIM-100, CFG 1.5,
guidance-rescale 0.0, the same balanced-modulo class schedule, and 10,000
samples.

| method | checkpoint SHA256 | sample-set SHA256 | FID | IS |
|---|---|---|---:|---:|
| CoFiTok K8 | `ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a` | `de17e26507c57f6ab473aa20fe5749e19dcd22941391dd7709b3d700297731da` | 138.29702495267782 | 7.924543688266955 |
| dense identity | `325da25f9fd228ab224abd977da7f9e1d000ba36e3e8136b55edae9c9f47c716` | `97c21492f4146ea1df889b639687e836dab54e46d5a4aff3c6ec4a59f2394e83` | 151.4476773464495 | 7.053977445069071 |

Exploratory reports:

```text
CoFiTok:
/tmp/cofitok-stability-10pct-50k-class-fidelity-10000-cpu-20260818/class_fidelity_report.json
bytes: 13822
SHA256: 1e3d57331f5cde7ba50b97b97bf1ebce518ea08a79f323cf22b5b8b75b72841d

dense identity:
/tmp/dense-stability-10pct-50k-class-fidelity-10000-cpu-20260818/class_fidelity_report.json
bytes: 13717
SHA256: b6a52c6d9383a377c5f7a599323e972751afc2f90f65809b325632b8d8e9d97f
```

Each report was rerun with `--resume`; the evaluator accepted and reused the
completed report only after exact provenance and metric-contract validation.

## Results

Chance accuracy is 0.1% for Top-1 and 0.5% for Top-5 over 1,000 classes.

| source | Top-1 | Top-5 | mean target probability | predicted-class fraction | normalized predicted entropy |
|---|---:|---:|---:|---:|---:|
| full-data CoFiTok 50K, 2,048 DDIM-50 | 3/2048 = 0.1465% | 14/2048 = 0.6836% | 0.1055% | 21.3% | 0.5390 |
| 10% CoFiTok 50K, 10K DDIM-100 | 12/10000 = 0.1200% | 59/10000 = 0.5900% | 0.1291% | 64.5% | 0.7385 |
| 10% dense 50K, 10K DDIM-100 | 17/10000 = 0.1700% | 69/10000 = 0.6900% | 0.1233% | 60.7% | 0.7103 |

The full-data CoFiTok result is not significantly above random accuracy:

```text
Top-1 exact one-sided binomial p-value versus p=0.001: 0.3363
Top-5 exact one-sided binomial p-value versus p=0.005: 0.1531
```

For the frozen 10% pair, the direct CoFiTok-minus-dense differences are:

```text
Top-1: -0.0005 absolute (-0.05 percentage points)
Top-5: -0.0010 absolute (-0.10 percentage points)
predicted-class fraction: +0.038
normalized predicted entropy: +0.0282
```

An unpaired two-proportion sanity check does not resolve either accuracy
difference (`p=0.353` for Top-1 and `p=0.375` for Top-5). This is not a formal
paired test because the saved aggregate reports do not retain per-sample
correctness indicators.

The scaling class-fidelity contract requires, for each method:

```text
Top-1 >= 1%
Top-5 >= 5%
predicted-class fraction >= 25%
normalized predicted entropy >= 0.50
```

Both frozen 10% methods pass the broad prediction-support rows but miss the
requested-class Top-1 and Top-5 rows by a large margin. The full-data CoFiTok
50K trend sample additionally falls below the 25% prediction-support row, but
its sample count and DDIM step count differ from the formal 10K protocol, so
that cross-run support comparison is diagnostic only.

## Interpretation

The frozen 10% pair produces a reasonably broad distribution of classifier
predictions, yet those predictions are nearly unrelated to the requested
labels. The class-conditioning failure is therefore shared by CoFiTok and its
matched dense baseline; it is not evidence of a CoFiTok-specific factorization
collapse. The 78.37%/94.87% real-set calibration confirms this is not an
ImageNet label-order artifact. CoFiTok's better frozen FID and IS do not imply
better requested-class alignment.

The current full-data CoFiTok 50K result confirms that its poor FID and visible
high-frequency residuals are accompanied by absent class semantics under the
current DDIM-50 trend protocol. It cannot support an absolute-quality,
class-conditional, release, or broad generation-superiority claim.

The active quality bridge remains authoritative. Its next decisive evidence is
the matched dense 50K evaluation, followed by the paired 50K report and the
formal matched 100K DDIM-100 10K terminal class-fidelity gate. These exploratory
reports do not alter, authorize, or replace that execution chain.
