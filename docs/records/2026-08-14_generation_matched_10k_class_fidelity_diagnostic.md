# 2026-08-14 matched 10K class-fidelity diagnostic

## Outcome

The existing matched 10K confirmation sample sets do not show usable alignment
to their requested ImageNet classes. CoFiTok reaches top-1/top-5
`0.18%/0.91%`; dense identity reaches `0.11%/0.68%`. The corresponding
1000-class independent-label references are `0.10%/0.50%`.

This is not literal sample-copy collapse. The separately replayed diversity
diagnostic found `10,000/10,000` pixel-unique images for both methods, while
the classifier predictions in this diagnostic cover `628/1000` classes for
CoFiTok and `635/1000` for dense. Instead, the current images are broadly
distributed across predicted labels but are almost unrelated to the requested
label. Poor/off-manifold image quality can itself cause this behavior, so the
evidence does not identify one causal mechanism.

A source-bound real-validation calibration then tested the same pinned
classifier, preprocessing, and exact dataset/timm/torchvision class order on a
deterministic balanced subset of 1,000 ImageNet validation images (one per
class). It passed with top-1/top-5 `77.7%/94.1%`. This rules out a broken
classifier, preprocessing path, or class-index mapping as the explanation for
the generated samples' near-chance requested-label alignment. The remaining
interpretation is model-side: the 50K models are undertrained, insufficiently
capable, or produce images too far off the ImageNet manifold for their
requested class to be recognized.

A deterministic visual audit of the first 16 global indices subsequently
confirmed the off-manifold appearance directly. Both generated rows contain
pronounced high-frequency colored speckle, unstable texture, and strong color
distortion; no fixed column is a clear requested fish or bird match. CoFiTok
and dense also share the same qualitative failure family under their matched
random stream, so the panels do not isolate factorization as the cause. The
source-bound record is
`docs/records/2026-08-14_generation_requested_class_visual_audit.md`.

## Scope and authority

This was a CPU-only diagnostic over already generated PNGs. It did not sample,
train, load a diffusion checkpoint, alter a checkpoint, change the formal
protocol, replace the frozen promotion gate, or authorize any larger launch.
The source confirmation uses the independent global-index window
`10000..19999` and guidance rescale `1.0`; therefore it is intentionally not a
formal scaling class-fidelity qualification.

The real-validation calibration was also CPU-only and non-authorizing. It did
not evaluate generated images and cannot replace a generation-quality metric,
the frozen promotion gate, or the terminal evaluation of a larger training
run.

The first attempt used the older `cf0e5fa` training execution checkout and was
rejected before inference because that revision restricted its formal metrics
provenance path to `start_index=0`. No report was written. The successful run
used the already deployed and tested diagnostic checkout at exact revision
`7b0149632afd16e1e9122241451654ec0d9b60a7`, whose provenance validator accepts
a complete declared nonzero global-index window while formal stage contracts
continue to require zero-based sampling.

## Protocol and identities

- Evaluator branch: `scale/generation-sample-diversity-diagnostic-v1`.
- Evaluator revision: `7b0149632afd16e1e9122241451654ec0d9b60a7`.
- Evaluator tree: `edac801b0f47924b18c6b5d5d7016f6ce2110481`.
- Classifier: torchvision ResNet-50 ImageNet-1K V2.
- Classifier bytes: `102,540,417`.
- Classifier SHA256:
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
- Device: CPU (`CUDA_VISIBLE_DEVICES=''`), `nice=19`.
- Matched resources: batch `64`, four data workers and four declared CPU
  compute threads per method.
- Samples: 10,000 per method, 10 per requested class, balanced modulo,
  global indices `10000..19999`.
- Sampling: EMA, DDIM-100, CFG `1.5`, guidance rescale `1.0`, bf16 generation.
- CoFiTok evaluator runtime: `2,180.0694 s`.
- Dense evaluator runtime: `2,202.2042 s`.

Both completed reports were invoked again with identical parameters and
`--resume`. The original evaluator revalidated and reused both reports,
including all PNGs, sample-set digests, immutable manifests, progress files,
checkpoint identities, classifier identity, evaluator runtime identity, and
completed report contract.

The calibration used exact revision
`b9a34542b905e5b2d6a4cb241dae2f46cac785c7` (tree
`66595fa27f3ab977b9a3c4ffe2d3be5ebf754022`) on branch
`scale/generation-classifier-calibration-v1`. It selected the lexicographically
first image in each locked WNID directory, producing sample-set SHA256
`459245ae24230f7d8b409355988d90101a5ed7d8e50d7be707235114c6f7b441`.
The report was replayed with identical parameters and `--resume`; the source
identities, selection bytes, Git/runtime identity, metrics population, checks,
and completed report contract all revalidated.

## Metrics

| Cohort | Top-1 | Top-5 | Mean target probability | Target NLL | Predicted classes | Normalized entropy |
|---|---:|---:|---:|---:|---:|---:|
| independent 1000-class reference | 0.10% | 0.50% | 0.10% | - | - | - |
| CoFiTok | 0.18% | 0.91% | 0.1263% | 7.4307 | 628/1000 | 0.7332 |
| dense identity | 0.11% | 0.68% | 0.1236% | 7.4837 | 635/1000 | 0.7092 |
| real-validation classifier calibration | 77.70% | 94.10% | 33.4238% | 1.5508 | 828/1000 | 0.9635 |

CoFiTok is slightly better than dense on requested-label alignment and FID,
but both are far below the eventual scaling class-fidelity floors of top-1
`1%` and top-5 `5%`. Those floors are cited only as a diagnostic reference:
this nonzero-start, rescale-1.0 sample set is not eligible to satisfy them.

## Source evidence

- Confirmation report: `13,003` bytes, SHA256
  `bf25ae5efa0c4e478568d9ee27f9cdc09aec02a7cac9313993dd2420f75e4032`.
- Diversity report: `1,116,048` bytes, SHA256
  `e1ad08e6abb56d9c844836746230c7a95224408b2719166a4b36f6989dcb9245`.
- CoFiTok class-fidelity report: `13,692` bytes, SHA256
  `0e66acffa007e570e4b83c7f3c59b81e4954d82d6351276c521d9e6ba6d7a68a`.
- Dense class-fidelity report: `13,730` bytes, SHA256
  `ff9e1ad079f8378defe686a5f92ca712379ea866df818b1020ced88cfc0a6d14`.
- Real-validation classifier calibration: `5,861` bytes, SHA256
  `a4c68b9b1c4dffda89f622887455f46d554fdb0c306d467fecb51d8f5abe6132`.

## Consequence for the generation program

The right next experiment remains the already queued matched full-data base128
100K quality bridge. Its terminal evaluation uses a new zero-based sample set,
the frozen guidance rescale `0.0`, FID/IS/precision/recall, and the same fixed
class-fidelity classifier. A further CFG-only sweep is not justified. If the
100K bridge still has near-chance requested-class alignment, the locked
follow-up decision should favor the prepared approximately 250M-capacity 10K
probe rather than protocol tuning.

At the end of this diagnostic, the bridge supervisor remained healthy and was
waiting only for the unrelated FieldScope CUDA process to exit; no CoFiTok GPU
controller or trainer was active.
