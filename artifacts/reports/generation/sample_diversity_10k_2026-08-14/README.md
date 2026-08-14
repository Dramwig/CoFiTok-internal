# CoFiTok 10K sampling-diversity diagnostic

## Technical summary

The matched 10K confirmation's approximately `0.008` recall is **not explained by literal sample duplication or by an excess of low-level dHash near-duplicate candidates**.

- CoFiTok: `10,000/10,000` unique encoded files and `10,000/10,000` unique decoded RGB images.
- Dense identity: `10,000/10,000` unique encoded files and `10,000/10,000` unique decoded RGB images.
- Global nearest-neighbor dHash distance `<=8`: CoFiTok `6/10,000`, dense `3/10,000`, balanced real reference `83/10,000`.
- Median global nearest-neighbor dHash distance: CoFiTok `16`, dense `16`, balanced real reference `15` bits.
- Within-class nearest-neighbor median over 10 samples per class: CoFiTok `26`, dense `26`, balanced real reference `25` bits.

These measurements rule out exact-copy collapse in the observed 10K sets. They also provide no low-level perceptual-hash evidence for near-copy collapse: both generated cohorts have fewer dHash-close neighbors than the matched real reference. This does **not** show that the generated distributions have good semantic coverage. High dHash separation can coexist with noisy, low-quality, or semantically misplaced images. Combined with FID `138.99/150.57` and recall `0.00756/0.00752`, the remaining diagnosis is limited semantic support or image quality, with undertraining/data scale still the leading actionable hypothesis.

## Scope and method

The diagnostic re-read the completed, non-authorizing matched 10K confirmation:

- CoFiTok and dense identity, each `10,000` images.
- Balanced-modulo ImageNet-1K schedule, 10 samples per class.
- EMA, DDIM-100, CFG `1.5`, guidance rescale `1.0`, seed `0`, global indices `[10,000, 20,000)`.
- A balanced real reference selected deterministically from ImageNet-256 validation: 10 of 50 images per each of 1,000 lexicographically ordered class directories, ranked by a fixed salted SHA-256 of the root-relative path.

Every generated file name, image count, image size, sampling-report identity, metrics-report identity, and declared sample-set SHA-256 was checked before analysis. Exact duplicates use SHA-256 of encoded bytes and of decoded RGB mode/dimensions/pixels. Near-duplicate candidates use a 64-bit grayscale `9x8` horizontal difference hash after Lanczos resize. Global nearest neighbors are exact Hamming searches within each 10K cohort; within-class statistics cover all `45,000` pairs per cohort.

## Limitations

dHash is a low-frequency appearance heuristic, not a semantic embedding, formal generation metric, or promotion gate. It can miss semantic mode collapse and can give large distances to noisy images. The real reference is a deterministic balanced subset rather than a repeated random sample. The diagnostic establishes no causal explanation for low recall and does not replace FID, precision, recall, the frozen promotion gate, or the full-data quality bridge.

## Next action

Keep the already-armed matched full-data 100K quality bridge as the next authoritative experiment. If both methods improve from 50K to 100K but absolute quality remains below threshold, follow the locked decision policy into the bounded base-256 capacity probe. If semantic support remains poor without shared improvement, add class-conditional Inception-feature coverage and predicted-class fidelity diagnostics before changing the model recipe. Additional CFG-only sweeps are not justified by the failed 10K confirmation.

## Evidence identity

- Execution revision: `7b0149632afd16e1e9122241451654ec0d9b60a7`
- Execution tree: `edac801b0f47924b18c6b5d5d7016f6ce2110481`
- Report: `report.json`, `1,116,048` bytes, SHA-256 `e1ad08e6abb56d9c844836746230c7a95224408b2719166a4b36f6989dcb9245`
- Diagnostic log: `diagnostic.log`, `1,902` bytes, SHA-256 `4fdeb314b41be402e377920be652f09cde9c5b0d89bf2188af2ff241ebb7cd59`
- Linux full regression: all tests except one environment-only subprocess import passed; the sole failure was reproduced as passing after restoring the project-required `PYTHONPATH=.:src` (`failed_test_rerun.log` SHA-256 `423b1d0e014eb1eab96f4420f7b344c2615be505dd574b756ac884826ca74f2d`).

The report is explicitly non-authorizing: no training, 300K launch, promotion, or release permission is carried by this result.
