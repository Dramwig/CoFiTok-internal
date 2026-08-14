# 2026-08-14 matched 10K sample-diversity diagnostic

## Decision context

The non-authorizing matched 10K sampling confirmation held quality:

- CoFiTok FID `138.9917`, precision `0.74390`, recall `0.00756`.
- Dense identity FID `150.5660`, precision `0.77900`, recall `0.00752`.
- Shared CFG `1.5`, guidance rescale `1.0` did not confirm the 1K sampling-recovery candidate.

While the sole GPU remained occupied by an unrelated FieldScope process, a CPU-only diagnostic was used to distinguish literal/near-copy collapse from broader support or quality failure. The diagnostic neither used nor requested a GPU and did not modify any checkpoint or sample.

## Immutable execution

- Branch: `scale/generation-sample-diversity-diagnostic-v1`
- Revision: `7b0149632afd16e1e9122241451654ec0d9b60a7`
- Tree: `edac801b0f47924b18c6b5d5d7016f6ce2110481`
- Clean checkout: `/root/autodl-tmp/CoFiTok/checkouts/sample-diversity-execution-7b01496/CoFiTok-internal`
- Bundle: `/tmp/cofitok-sample-diversity-7b01496-from-1eb.bundle`, `40,375,809` bytes, SHA-256 `c550a8a5fb164d8a90abbaec7a40cd98f50e7de091bc58e84d0be9366464615e`
- Remote output: `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_confirmation_10k_v1/diagnostics/sample_diversity_v1`
- PID: `164326`
- Runtime: `449.305` seconds, CPU-only, Python `3.10.20`, NumPy `2.2.6`, Pillow `12.2.0`.

The execution verified the exact confirmation report and both methods' sampling/metrics report identities, recomputed both generated sample-set SHA-256 values, enforced exact contiguous indices `[10,000, 20,000)`, decoded every image as `256x256`, and rejected symlinks.

## Results

| Cohort | Exact decoded duplicates | Global dHash `<=4` | Global dHash `<=8` | Global nearest median | Within-class nearest median |
|---|---:|---:|---:|---:|---:|
| CoFiTok | `0/10,000` | `0/10,000` | `6/10,000` | `16` | `26` |
| Dense identity | `0/10,000` | `0/10,000` | `3/10,000` | `16` | `26` |
| Balanced real reference | `1/10,000` globally | `6/10,000` | `83/10,000` | `15` | `25` |

The one exact real-reference duplicate occurs across classes; the within-class exact duplicate-pair count is zero. Each generated cohort has zero encoded duplicates, zero decoded-pixel duplicates, and zero within-class dHash pairs at Hamming distance `<=8` across `45,000` pairs.

Result identity:

- `report.json`: `1,116,048` bytes, SHA-256 `e1ad08e6abb56d9c844836746230c7a95224408b2719166a4b36f6989dcb9245`.
- `diagnostic.log`: `1,902` bytes, SHA-256 `4fdeb314b41be402e377920be652f09cde9c5b0d89bf2188af2ff241ebb7cd59`.

## Interpretation

Literal duplicate collapse is ruled out for the observed generated sets. The dHash heuristic also provides no evidence that either generated cohort contains an excess of low-level near copies relative to the matched real reference. Therefore the approximately `0.008` recall should not be treated as a file/sample duplication bug.

This result does not prove semantic diversity. Noisy or semantically off-manifold outputs can be far apart under dHash while still having poor Inception recall and high FID. The evidence is most consistent with insufficient learned semantic support or insufficient image quality, leaving full-data training duration/data scale as the next bounded test.

## Verification

- Local targeted set: `67` relevant tests passed.
- Linux exact-revision collection: `1,237` tests.
- Linux full run: `1,234` passed, `2` skipped, and one subprocess import failed solely because the initial command omitted `PYTHONPATH=.:src`; log SHA-256 `47960fd8888dcefecc178f95eb408ed608bcf718dfdf9d760745bb00c824e348`.
- Exact failed test rerun with `PYTHONPATH=.:src`: passed; log SHA-256 `423b1d0e014eb1eab96f4420f7b344c2615be505dd574b756ac884826ca74f2d`.
- Execution checkout remained clean.
- Formal remote checkout remained at `1ebcc15210e63a776a2ba448481cbd8bb94a4066`; its tracked porcelain digest remained `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.

## Boundary and next step

The diagnostic is non-authorizing and cannot launch training, full 300K, promotion, or release. The already-restored quality-bridge supervisor remains the next experiment and will start only after five consecutive idle-GPU polls. If the matched 50K-to-100K trend improves for both methods but absolute quality remains poor, the locked follow-up decision may advance to the already-prepared bounded base-256/10K capacity probe. No further CFG-only sweep is warranted from current evidence.
